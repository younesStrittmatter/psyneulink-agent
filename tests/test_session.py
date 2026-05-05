"""Unit tests for ``core.session.Session`` and ``render_system_prompt``.

We don't construct a real Anthropic client or MCP session — those paths
are exercised in ``test_loop`` and integration tests. Here we only
verify the dataclass-level behaviour: attaching, detaching, prompt
rendering, snapshotting, plus the long-lived MCP plumbing
(``lifespan`` + ``call_tool``) that the upcoming web UI relies on.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from psyneulink_agent.core.backends import AnthropicSdkBackend, ClaudeCliBackend
from psyneulink_agent.core.resources import (
    DataResource,
    ModelFileResource,
    PdfResource,
)
from psyneulink_agent.core.session import Session
from psyneulink_agent.core.system_prompt import SYSTEM_PROMPT, render_system_prompt

# ---------------------------------------------------------------------------
# render_system_prompt
# ---------------------------------------------------------------------------


def test_render_system_prompt_with_no_resources_returns_base_prompt() -> None:
    assert render_system_prompt(None) == SYSTEM_PROMPT
    assert render_system_prompt([]) == SYSTEM_PROMPT


def test_render_system_prompt_appends_attached_resource_summary(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")

    rendered = render_system_prompt([PdfResource(pdf), DataResource(csv)])

    assert rendered.startswith(SYSTEM_PROMPT)
    assert "Attached resources for this session:" in rendered
    assert "- [pdf] paper.pdf" in rendered
    assert "- [data] data.csv" in rendered


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_attach_and_detach(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    pdf = PdfResource(pdf_path)

    session = Session()
    assert session.resources == []
    session.attach(pdf)
    assert session.resources == [pdf]
    session.detach(pdf)
    assert session.resources == []


def test_session_system_prompt_includes_resource_summary(tmp_path: Path) -> None:
    py_path = tmp_path / "model.py"
    py_path.write_text("# pretend model")
    session = Session()
    assert session.system_prompt() == SYSTEM_PROMPT  # bare session = base prompt
    session.attach(ModelFileResource(py_path))
    out = session.system_prompt()
    assert "model.py" in out
    assert out.startswith(SYSTEM_PROMPT)


def test_session_snapshot_is_json_serialisable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    session = Session(model="claude-test-model")
    session.attach(PdfResource(pdf_path))
    session.history.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})

    snap = session.snapshot()
    assert snap["model"] == "claude-test-model"
    assert snap["resources"] == [{"kind": "pdf", "label": "paper.pdf"}]
    assert snap["history"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    # Round-trip through JSON to confirm it's serialisable.
    json.dumps(snap)


def test_session_snapshot_history_is_a_copy(tmp_path: Path) -> None:
    """Mutating the returned history must not affect the live session."""
    session = Session()
    session.history.append({"role": "user", "content": []})
    snap = session.snapshot()
    snap["history"].append({"role": "user", "content": [{"type": "text", "text": "x"}]})
    assert len(session.history) == 1


# ---------------------------------------------------------------------------
# lifespan() + call_tool() — long-lived MCP connection
#
# We mock the two collaborators that ``Session`` reaches into for MCP
# (``mcp_session`` and ``call_mcp_tool``) so no real subprocess is
# spawned. The fixtures track entry counts so tests can distinguish
# "MCP opened once for whole lifespan" from "MCP opened per call".
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mcp_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``mcp_session`` with a fake yielding a stub ClientSession.

    Returns a dict so tests can both assert on the number of times the
    ctx manager was entered AND identify *which* fake client object was
    handed out (to check that ``call_mcp_tool`` / ``run_turn`` were
    invoked against that exact object — i.e. the active connection,
    not a freshly-opened one).
    """
    enters: list[Path | None] = []
    fake_client = MagicMock(name="FakeMCPClient")

    @asynccontextmanager
    async def _fake(project: Path | None) -> AsyncIterator[Any]:
        enters.append(project)
        yield fake_client

    monkeypatch.setattr("psyneulink_agent.core.session.mcp_session", _fake)
    return {"enters": enters, "client": fake_client}


@pytest.fixture
def fake_call_mcp_tool(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    fake = AsyncMock(return_value="(stub result)")
    monkeypatch.setattr("psyneulink_agent.core.session.call_mcp_tool", fake)
    return fake


def _drive(coro_iter: Any) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for ev in coro_iter:
            out.append(ev)
        return out

    return asyncio.run(_collect())


def _make_fake_anthropic_end_turn() -> Any:
    """Build a fake AsyncAnthropic that always returns a one-block end_turn message.

    Mirrors the pattern in ``tests/test_loop.py`` but kept self-contained
    so this file doesn't grow a cross-test import. The returned client
    also exposes ``messages.calls`` so tests can introspect what
    ``run_turn`` sent.
    """

    class _Block:
        type = "text"
        text = "ok"

        def model_dump(self, mode: str = "json", exclude_none: bool = True) -> dict[str, Any]:
            return {"type": "text", "text": "ok"}

    class _Message:
        def __init__(self) -> None:
            self.content = [_Block()]
            self.stop_reason = "end_turn"

    class _Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> _Message:
            self.calls.append(kwargs)
            return _Message()

    class _Client:
        def __init__(self) -> None:
            self.messages = _Messages()

    return _Client()


def test_lifespan_opens_mcp_once_across_multiple_turns(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """Two ``send_user_message`` calls inside one lifespan share one MCP session."""
    session = Session()
    client = _make_fake_anthropic_end_turn()

    # ``list_anthropic_tools`` is called against the active MCP client;
    # mock it to return an empty tool list so ``run_turn`` is happy.
    fake_mcp_session["client"].list_tools = AsyncMock(
        return_value=MagicMock(tools=[])
    )

    async def _go() -> None:
        async with session.lifespan():
            async for _ in session.send_user_message("hi", anthropic_client=client):
                pass
            async for _ in session.send_user_message("again", anthropic_client=client):
                pass

    asyncio.run(_go())
    assert len(fake_mcp_session["enters"]) == 1


def test_per_turn_fallback_opens_mcp_per_turn(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """Without ``lifespan``, two turns spawn two MCP sessions (legacy path)."""
    session = Session()
    client = _make_fake_anthropic_end_turn()
    fake_mcp_session["client"].list_tools = AsyncMock(
        return_value=MagicMock(tools=[])
    )

    async def _go() -> None:
        async for _ in session.send_user_message("a", anthropic_client=client):
            pass
        async for _ in session.send_user_message("b", anthropic_client=client):
            pass

    asyncio.run(_go())
    assert len(fake_mcp_session["enters"]) == 2


def test_lifespan_is_not_reentrant(fake_mcp_session: dict[str, Any]) -> None:
    """Nesting ``lifespan()`` is a programming error; raise loudly."""
    session = Session()

    async def _go() -> None:
        async with session.lifespan():
            with pytest.raises(RuntimeError, match="re-entrant"):
                async with session.lifespan():
                    pass

    asyncio.run(_go())


def test_lifespan_clears_mcp_attr_on_exit(fake_mcp_session: dict[str, Any]) -> None:
    """After ``lifespan()`` exits, ``Session`` reverts to per-call MCP spawns."""
    session = Session()

    async def _go() -> None:
        assert session._mcp is None
        async with session.lifespan():
            assert session._mcp is fake_mcp_session["client"]
        assert session._mcp is None

    asyncio.run(_go())


def test_call_tool_outside_lifespan_opens_per_call_session(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """Without ``lifespan``, ``call_tool`` opens + closes its own MCP session."""
    session = Session()
    result = asyncio.run(session.call_tool("foo", {"x": 1}))

    assert result == "(stub result)"
    assert len(fake_mcp_session["enters"]) == 1
    fake_call_mcp_tool.assert_awaited_once_with(
        fake_mcp_session["client"], "foo", {"x": 1}
    )


def test_call_tool_inside_lifespan_uses_active_session(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """Two ``call_tool`` invocations inside one lifespan share one MCP session."""
    session = Session()

    async def _go() -> None:
        async with session.lifespan():
            await session.call_tool("foo", {"a": 1})
            await session.call_tool("bar", {"b": 2})

    asyncio.run(_go())

    assert len(fake_mcp_session["enters"]) == 1
    assert fake_call_mcp_tool.await_count == 2
    assert all(
        call.args[0] is fake_mcp_session["client"]
        for call in fake_call_mcp_tool.await_args_list
    )


def test_call_tool_passes_args(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """Args dict is forwarded verbatim (not wrapped, mutated, or copied through JSON)."""
    session = Session()
    payload = {"handle": "tm_42", "options": {"verbose": True, "limit": 10}}
    asyncio.run(session.call_tool("inspect_mechanism", payload))

    fake_call_mcp_tool.assert_awaited_once_with(
        fake_mcp_session["client"], "inspect_mechanism", payload
    )


def test_call_tool_with_none_args_defaults_to_empty_dict(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """``args=None`` (or omitted) is normalised to ``{}`` before dispatch."""
    session = Session()
    asyncio.run(session.call_tool("ping"))

    fake_call_mcp_tool.assert_awaited_once_with(
        fake_mcp_session["client"], "ping", {}
    )


def test_send_user_message_inside_lifespan_uses_active_session(
    fake_mcp_session: dict[str, Any], fake_call_mcp_tool: AsyncMock
) -> None:
    """``run_turn`` receives the lifespan's MCP client, not a freshly-spawned one."""
    captured: dict[str, Any] = {}

    async def _fake_run_turn(**kwargs: Any) -> Any:
        captured.update(kwargs)
        if False:
            yield {}  # make this a (degenerate) async generator

    # Patch the loop's run_turn at the binding ``Session`` imports it
    # under, so we don't have to hand-fake an Anthropic client + tool list.
    import psyneulink_agent.core.session as session_mod

    original_run_turn = session_mod.run_turn
    session_mod.run_turn = _fake_run_turn  # type: ignore[assignment]
    fake_mcp_session["client"].list_tools = AsyncMock(
        return_value=MagicMock(tools=[])
    )

    try:
        session = Session()
        client = _make_fake_anthropic_end_turn()

        async def _go() -> None:
            async with session.lifespan():
                async for _ in session.send_user_message("hello", anthropic_client=client):
                    pass

        asyncio.run(_go())
    finally:
        session_mod.run_turn = original_run_turn  # type: ignore[assignment]

    assert len(fake_mcp_session["enters"]) == 1
    assert captured["mcp"] is fake_mcp_session["client"]


# ---------------------------------------------------------------------------
# Default-backend selection
#
# Each test below explicitly clears ``PSYNEULINK_LLM_BACKEND`` (the
# global conftest fixture pins it to ``sdk`` so older tests still see
# SDK semantics). The selection helper itself is what we want to
# exercise here.
# ---------------------------------------------------------------------------


def test_default_backend_picks_sdk_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSYNEULINK_LLM_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    session = Session()
    assert session.llm_backend.kind == "sdk"
    assert isinstance(session.llm_backend, AnthropicSdkBackend)


def test_default_backend_picks_cli_without_api_key_with_claude_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PSYNEULINK_LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "psyneulink_agent.core.session.shutil.which",
        lambda name: "/usr/bin/claude" if name == "claude" else None,
    )
    session = Session()
    assert session.llm_backend.kind == "cli"
    assert isinstance(session.llm_backend, ClaudeCliBackend)


def test_default_backend_falls_back_to_sdk_when_nothing_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PSYNEULINK_LLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("psyneulink_agent.core.session.shutil.which", lambda name: None)
    session = Session()
    assert session.llm_backend.kind == "sdk"


def test_default_backend_env_override_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSYNEULINK_LLM_BACKEND", "sdk")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "psyneulink_agent.core.session.shutil.which",
        lambda name: "/usr/bin/claude" if name == "claude" else None,
    )
    session = Session()
    assert session.llm_backend.kind == "sdk"


def test_default_backend_env_override_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSYNEULINK_LLM_BACKEND", "cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # would normally win
    session = Session()
    assert session.llm_backend.kind == "cli"


# ---------------------------------------------------------------------------
# Lifespan transport selection
# ---------------------------------------------------------------------------


def test_lifespan_with_sdk_backend_uses_stdio_mcp(
    fake_mcp_session: dict[str, Any],
) -> None:
    """SDK backend → stdio ``mcp_session``; sse plumbing is never touched."""
    session = Session(llm_backend=AnthropicSdkBackend())

    async def _go() -> None:
        async with session.lifespan():
            assert session._mcp is fake_mcp_session["client"]

    asyncio.run(_go())
    assert len(fake_mcp_session["enters"]) == 1


def test_lifespan_with_cli_backend_spawns_sse_mcp_and_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI backend → ``_spawn_mcp_sse`` + ``sse_mcp_session`` + URL rebound."""
    backend = ClaudeCliBackend(mcp_url="http://placeholder/sse")
    session = Session(llm_backend=backend)

    spawn_calls: list[tuple[Any, int]] = []
    fake_proc = MagicMock(name="FakeMcpProc")
    fake_proc.terminate = MagicMock()
    fake_proc.wait = AsyncMock(return_value=0)

    async def _fake_spawn(cmd: Any, port: int, *, timeout: float = 30.0) -> Any:
        spawn_calls.append((cmd, port))
        return fake_proc

    sse_enters: list[str] = []
    fake_client = MagicMock(name="FakeMcpClient")

    @asynccontextmanager
    async def _fake_sse(url: str) -> AsyncIterator[Any]:
        sse_enters.append(url)
        yield fake_client

    # Avoid touching real psyneulink-mcp resolution.
    monkeypatch.setattr(
        "psyneulink_agent.core.session.resolve_server_command",
        lambda mcp_project=None: ["/fake/psyneulink-mcp"],
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session._spawn_mcp_sse", _fake_spawn
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session.sse_mcp_session", _fake_sse
    )

    async def _go() -> None:
        async with session.lifespan():
            assert session._mcp is fake_client
            assert backend.mcp_url == sse_enters[0]
            assert backend.mcp_url.startswith("http://127.0.0.1:")
            assert backend.mcp_url.endswith("/sse")

    asyncio.run(_go())

    assert len(spawn_calls) == 1
    assert spawn_calls[0][0] == ["/fake/psyneulink-mcp"]
    assert isinstance(spawn_calls[0][1], int)
    assert len(sse_enters) == 1


def test_lifespan_cli_backend_terminates_subprocess_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ClaudeCliBackend(mcp_url="http://placeholder/sse")
    session = Session(llm_backend=backend)

    fake_proc = MagicMock(name="FakeMcpProc")
    fake_proc.terminate = MagicMock()
    fake_proc.wait = AsyncMock(return_value=0)

    async def _fake_spawn(cmd: Any, port: int, *, timeout: float = 30.0) -> Any:
        return fake_proc

    fake_client = MagicMock(name="FakeMcpClient")

    @asynccontextmanager
    async def _fake_sse(url: str) -> AsyncIterator[Any]:
        yield fake_client

    monkeypatch.setattr(
        "psyneulink_agent.core.session.resolve_server_command",
        lambda mcp_project=None: ["/fake/psyneulink-mcp"],
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session._spawn_mcp_sse", _fake_spawn
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session.sse_mcp_session", _fake_sse
    )

    async def _go() -> None:
        async with session.lifespan():
            pass

    asyncio.run(_go())

    fake_proc.terminate.assert_called_once()
    fake_proc.wait.assert_awaited()
    assert session._mcp is None
    assert session._mcp_subprocess is None
    assert session._mcp_url is None


def test_lifespan_cli_backend_cleans_up_temp_mcp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ClaudeCliBackend(mcp_url="http://placeholder/sse")
    # Force config creation up front so we can capture the path.
    cfg_path = backend._build_mcp_config()
    assert cfg_path.exists()

    session = Session(llm_backend=backend)

    fake_proc = MagicMock()
    fake_proc.terminate = MagicMock()
    fake_proc.wait = AsyncMock(return_value=0)

    async def _fake_spawn(cmd: Any, port: int, *, timeout: float = 30.0) -> Any:
        return fake_proc

    fake_client = MagicMock()

    @asynccontextmanager
    async def _fake_sse(url: str) -> AsyncIterator[Any]:
        yield fake_client

    monkeypatch.setattr(
        "psyneulink_agent.core.session.resolve_server_command",
        lambda mcp_project=None: ["/fake/psyneulink-mcp"],
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session._spawn_mcp_sse", _fake_spawn
    )
    monkeypatch.setattr(
        "psyneulink_agent.core.session.sse_mcp_session", _fake_sse
    )

    async def _go() -> None:
        async with session.lifespan():
            pass

    asyncio.run(_go())

    assert not cfg_path.exists()


def test_send_user_message_outside_lifespan_with_cli_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``lifespan``, the CLI backend has no SSE MCP to talk to → loud error."""
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    session = Session(llm_backend=backend)

    async def _go() -> None:
        with pytest.raises(RuntimeError, match="lifespan"):
            async for _ in session.send_user_message("hi"):
                pass

    asyncio.run(_go())
