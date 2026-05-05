"""Unit tests for ``core.backends.anthropic_sdk.AnthropicSdkBackend``.

The backend is a thin wrapper around ``loop.run_turn`` so the only
interesting things to verify are (a) the ``kind`` attribute and (b)
that calling ``backend.run_turn(...)`` actually delegates with the
expected kwargs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from psyneulink_agent.core.backends import AnthropicSdkBackend


def _drive(coro_iter: Any) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for ev in coro_iter:
            out.append(ev)
        return out

    return asyncio.run(_collect())


def test_anthropic_sdk_backend_kind_is_sdk() -> None:
    assert AnthropicSdkBackend.kind == "sdk"
    assert AnthropicSdkBackend(model="x")._client is None


def test_anthropic_sdk_backend_run_turn_delegates_to_run_turn(monkeypatch) -> None:
    """``backend.run_turn`` calls into the loop's ``run_turn`` with our kwargs."""
    captured: dict[str, Any] = {}

    async def _fake_run_turn(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        captured.update(kwargs)
        yield {"type": "text_chunk", "delta": "hi"}
        yield {"type": "turn_complete", "stop_reason": "end_turn"}

    # The SDK backend looks up ``run_turn`` via the session module
    # (re-exported there) so monkeypatching here flows through the
    # backend's lazy import.
    monkeypatch.setattr("psyneulink_agent.core.session.run_turn", _fake_run_turn)

    fake_client = object()
    backend = AnthropicSdkBackend(model="claude-test", anthropic_client=fake_client)
    history: list[dict[str, Any]] = []
    user_content = [{"type": "text", "text": "hello"}]
    mcp = object()
    tools = [{"name": "x", "description": "", "input_schema": {}}]

    events = _drive(
        backend.run_turn(
            history=history,
            system_prompt="sys",
            user_content=user_content,
            mcp=mcp,
            tools=tools,
        )
    )

    assert events == [
        {"type": "text_chunk", "delta": "hi"},
        {"type": "turn_complete", "stop_reason": "end_turn"},
    ]
    assert captured["anthropic_client"] is fake_client
    assert captured["model"] == "claude-test"
    assert captured["system_prompt"] == "sys"
    assert captured["history"] is history
    assert captured["user_content"] is user_content
    assert captured["mcp"] is mcp
    assert captured["tools"] is tools


def test_anthropic_sdk_backend_lazy_constructs_client(monkeypatch) -> None:
    """If no client is injected, a real ``AsyncAnthropic`` is constructed lazily."""
    captured: dict[str, Any] = {}
    constructed: list[object] = []

    class _FakeAsyncAnthropic:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(self)

    async def _fake_run_turn(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        captured.update(kwargs)
        if False:
            yield {}

    monkeypatch.setattr("psyneulink_agent.core.session.run_turn", _fake_run_turn)
    # Patch the import target so the backend's local
    # ``from anthropic import AsyncAnthropic`` returns our fake.
    import sys
    import types

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = _FakeAsyncAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    backend = AnthropicSdkBackend(model="m")
    _drive(
        backend.run_turn(
            history=[],
            system_prompt="s",
            user_content=[{"type": "text", "text": "hi"}],
            mcp=object(),
            tools=[],
        )
    )

    assert len(constructed) == 1
    assert captured["anthropic_client"] is constructed[0]
