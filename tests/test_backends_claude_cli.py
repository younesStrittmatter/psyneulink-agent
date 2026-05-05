"""Unit tests for ``core.backends.claude_cli.ClaudeCliBackend``.

Everything is mocked — we never spawn a real ``claude`` subprocess
in CI. The fake ``Process`` swaps in for ``asyncio.create_subprocess_exec``
and yields a canned sequence of stream-json bytes that the backend's
parser walks through. That's enough to verify argv shape, event
translation, history bookkeeping, and error surfacing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from psyneulink_agent.core.backends import ClaudeCliBackend


def _drive(coro_iter: Any) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for ev in coro_iter:
            out.append(ev)
        return out

    return asyncio.run(_collect())


# ---------------------------------------------------------------------------
# Fake subprocess plumbing
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-iterable wrapper around a list of ``bytes`` lines.

    Mirrors the bits of ``asyncio.StreamReader`` that ``ClaudeCliBackend``
    actually uses (async iteration line-by-line + ``read()``).
    """

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self._idx = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> bytes:
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line

    async def read(self, n: int = -1) -> bytes:
        return b""


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(
        self,
        *,
        stdout_lines: list[bytes],
        stderr_lines: list[bytes] | None = None,
        returncode: int = 0,
    ):
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines or [])
        self.stdin = _FakeStdin()
        self._returncode = returncode

    @property
    def returncode(self) -> int:
        return self._returncode

    async def wait(self) -> int:
        return self._returncode


def _install_fake_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout_lines: list[bytes],
    stderr_lines: list[bytes] | None = None,
    returncode: int = 0,
) -> dict[str, Any]:
    """Patch ``asyncio.create_subprocess_exec`` to return a canned ``_FakeProc``.

    Returns a dict the test can inspect afterwards:
    * ``"argv_history"`` — list of argv tuples for each spawn
    * ``"procs"`` — list of fake ``_FakeProc`` instances
    """
    capture: dict[str, Any] = {"argv_history": [], "procs": []}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        capture["argv_history"].append(args)
        proc = _FakeProc(
            stdout_lines=list(stdout_lines),
            stderr_lines=list(stderr_lines or []),
            returncode=returncode,
        )
        capture["procs"].append(proc)
        return proc

    monkeypatch.setattr(
        "psyneulink_agent.core.backends.claude_cli.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    return capture


# ---------------------------------------------------------------------------
# Canned stream-json fragments (verified shape — see claude_cli.py docstring)
# ---------------------------------------------------------------------------


def _line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


def _success_result(stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 100,
        "result": "ok",
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_claude_cli_backend_kind_is_cli() -> None:
    assert ClaudeCliBackend.kind == "cli"


def test_build_mcp_config_writes_sse_pointer(tmp_path) -> None:
    backend = ClaudeCliBackend(mcp_url="http://127.0.0.1:54321/sse")
    path = backend._build_mcp_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mcpServers"]["psyneulink"]["type"] == "sse"
        assert data["mcpServers"]["psyneulink"]["url"] == "http://127.0.0.1:54321/sse"
    finally:
        backend.cleanup()


def test_cleanup_removes_temp_config_file() -> None:
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    path = backend._build_mcp_config()
    assert path.exists()
    backend.cleanup()
    assert not path.exists()
    # Idempotent.
    backend.cleanup()


def test_run_turn_spawns_claude_with_expected_argv(monkeypatch) -> None:
    capture = _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[_line(_success_result())],
    )
    backend = ClaudeCliBackend(
        mcp_url="http://127.0.0.1:1234/sse",
        claude_path="/fake/claude",
        session_id="00000000-0000-0000-0000-000000000001",
        model="claude-test",
    )
    cfg_data: dict[str, Any] = {}
    try:
        _drive(
            backend.run_turn(
                history=[],
                system_prompt="SYS",
                user_content=[{"type": "text", "text": "hello"}],
                mcp=object(),
                tools=[],
            )
        )
        # Inspect the temp config file before ``cleanup`` deletes it.
        argv_now = capture["argv_history"][0]
        cfg_idx_now = argv_now.index("--mcp-config")
        with open(argv_now[cfg_idx_now + 1], encoding="utf-8") as fh:
            cfg_data = json.loads(fh.read())
    finally:
        backend.cleanup()

    assert len(capture["argv_history"]) == 1
    argv = capture["argv_history"][0]
    assert argv[0] == "/fake/claude"
    assert "--print" in argv
    out_idx = argv.index("--output-format")
    assert argv[out_idx + 1] == "stream-json"
    in_idx = argv.index("--input-format")
    assert argv[in_idx + 1] == "stream-json"
    assert "--include-partial-messages" in argv
    assert "--verbose" in argv
    assert "--strict-mcp-config" in argv
    tools_idx = argv.index("--tools")
    assert argv[tools_idx + 1] == ""
    pm_idx = argv.index("--permission-mode")
    assert argv[pm_idx + 1] == "bypassPermissions"
    assert "--mcp-config" in argv
    assert cfg_data["mcpServers"]["psyneulink"]["url"] == "http://127.0.0.1:1234/sse"
    sp_idx = argv.index("--append-system-prompt")
    assert argv[sp_idx + 1] == "SYS"
    sid_idx = argv.index("--session-id")
    assert argv[sid_idx + 1] == "00000000-0000-0000-0000-000000000001"
    m_idx = argv.index("--model")
    assert argv[m_idx + 1] == "claude-test"

    # Prompt is delivered via stdin as a stream-json user message line.
    proc = capture["procs"][0]
    assert len(proc.stdin.written) == 1
    payload = proc.stdin.written[0]
    assert payload.endswith(b"\n")
    msg = json.loads(payload)
    assert msg == {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        },
    }
    assert proc.stdin.closed


def test_run_turn_translates_text_chunks(monkeypatch) -> None:
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "hi "},
                    },
                }
            ),
            _line(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "there"},
                    },
                }
            ),
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    history: list[dict[str, Any]] = []
    try:
        events = _drive(
            backend.run_turn(
                history=history,
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    text_chunks = [e for e in events if e["type"] == "text_chunk"]
    assert [e["delta"] for e in text_chunks] == ["hi ", "there"]
    assert events[-1] == {"type": "turn_complete", "stop_reason": "end_turn"}


def test_run_turn_translates_tool_use_and_result(monkeypatch) -> None:
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_abc",
                                "name": "create_x",
                                "input": {"a": 1},
                            }
                        ],
                    },
                }
            ),
            _line(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_abc",
                                "content": "tool ran ok",
                                "is_error": False,
                            }
                        ],
                    },
                }
            ),
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "build x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    use = next(e for e in events if e["type"] == "tool_use")
    assert use == {
        "type": "tool_use",
        "id": "toolu_abc",
        "name": "create_x",
        "input": {"a": 1},
    }
    result = next(e for e in events if e["type"] == "tool_result")
    assert result == {
        "type": "tool_result",
        "id": "toolu_abc",
        "content": "tool ran ok",
        "is_error": False,
    }


def test_run_turn_translates_tool_result_with_list_content(monkeypatch) -> None:
    """``tool_result.content`` may arrive as a list of text blocks; flatten them."""
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_x",
                                "content": [
                                    {"type": "text", "text": "first"},
                                    {"type": "text", "text": "second"},
                                ],
                                "is_error": False,
                            }
                        ],
                    },
                }
            ),
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    result = next(e for e in events if e["type"] == "tool_result")
    assert result["content"] == "first\nsecond"


def test_run_turn_appends_assistant_to_history(monkeypatch) -> None:
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "hello world"},
                    },
                }
            ),
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    history: list[dict[str, Any]] = []
    user_content = [{"type": "text", "text": "ping"}]
    try:
        _drive(
            backend.run_turn(
                history=history,
                system_prompt="s",
                user_content=user_content,
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": user_content}
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == [{"type": "text", "text": "hello world"}]


def test_run_turn_text_only_in_assistant_block_falls_back(monkeypatch) -> None:
    """When --include-partial-messages doesn't deliver, text from the full
    ``assistant`` event is still emitted as a ``text_chunk``."""
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi from assistant"}],
                    },
                }
            ),
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    chunks = [e for e in events if e["type"] == "text_chunk"]
    assert chunks == [{"type": "text_chunk", "delta": "hi from assistant"}]


def test_run_turn_surfaces_nonzero_exit_as_tool_result_error(monkeypatch) -> None:
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[],
        stderr_lines=[b"boom\n"],
        returncode=1,
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    errors = [e for e in events if e["type"] == "tool_result"]
    assert len(errors) == 1
    assert errors[0]["is_error"] is True
    assert "exit" in errors[0]["content"]
    assert events[-1] == {"type": "turn_complete", "stop_reason": "error"}


def test_run_turn_pipes_full_content_blocks_via_stream_json(monkeypatch) -> None:
    """PDF document blocks (and any non-text block) ride through verbatim
    in the stream-json user message — no warning, no flattening."""
    capture = _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[_line(_success_result())],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    pdf_block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQ=",  # tiny "%PDF-1.4" stub, base64
        },
        "title": "paper.pdf",
    }
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[
                    pdf_block,
                    {"type": "text", "text": "summarise this paper"},
                ],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    # No warning event — non-text blocks are first-class now.
    assert [e for e in events if e["type"] == "warning"] == []

    # The PDF block landed on stdin verbatim.
    proc = capture["procs"][0]
    assert len(proc.stdin.written) == 1
    msg = json.loads(proc.stdin.written[0])
    sent_content = msg["message"]["content"]
    assert sent_content[0] == pdf_block
    assert sent_content[1] == {"type": "text", "text": "summarise this paper"}


def test_run_turn_session_id_then_resume_for_multi_turn(monkeypatch) -> None:
    """Turn 1 creates the on-disk session with ``--session-id``; turn ≥ 2
    must re-attach with ``--resume`` (claude refuses ID reuse with
    ``--session-id``)."""
    capture = _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[_line(_success_result())],
    )
    backend = ClaudeCliBackend(
        mcp_url="http://x/sse",
        claude_path="/fake/claude",
        session_id="11111111-2222-3333-4444-555555555555",
    )
    try:
        for prompt in ["first", "second", "third"]:
            _drive(
                backend.run_turn(
                    history=[],
                    system_prompt="s",
                    user_content=[{"type": "text", "text": prompt}],
                    mcp=object(),
                    tools=[],
                )
            )
    finally:
        backend.cleanup()

    assert len(capture["argv_history"]) == 3

    argv_t1 = capture["argv_history"][0]
    assert "--session-id" in argv_t1
    assert "--resume" not in argv_t1
    assert argv_t1[argv_t1.index("--session-id") + 1] == (
        "11111111-2222-3333-4444-555555555555"
    )

    for argv in capture["argv_history"][1:]:
        assert "--resume" in argv
        assert "--session-id" not in argv
        assert argv[argv.index("--resume") + 1] == (
            "11111111-2222-3333-4444-555555555555"
        )


def test_failed_first_turn_keeps_session_id_for_retry(monkeypatch) -> None:
    """If turn 1 fails (e.g. auth error), the next turn should still use
    --session-id to *create* the session rather than --resume a session
    that was never written to disk."""
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[],
        stderr_lines=[b"Invalid API key\n"],
        returncode=1,
    )
    backend = ClaudeCliBackend(
        mcp_url="http://x/sse",
        claude_path="/fake/claude",
        session_id="22222222-3333-4444-5555-666666666666",
    )
    try:
        _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    assert backend._session_created is False


def test_run_turn_skips_unknown_event_shapes_silently(monkeypatch) -> None:
    """Unknown / future event shapes must not crash the parser."""
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line({"type": "system", "subtype": "init", "session_id": "x"}),
            _line({"type": "system", "subtype": "status", "status": "requesting"}),
            _line({"type": "totally_new_thing", "blah": 1}),
            b"not even json\n",
            _line(_success_result()),
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    assert events[-1] == {"type": "turn_complete", "stop_reason": "end_turn"}


def test_run_turn_cancel_event_kills_subprocess_and_yields_turn_cancelled(
    monkeypatch,
) -> None:
    """``cancel_current_turn()`` flips the event; the CLI backend must
    ``terminate()`` the subprocess so the streaming stdout loop unblocks
    and the run yields a single ``turn_cancelled`` event before
    ``turn_complete`` ever fires."""

    cancel_event = asyncio.Event()
    teardown_signal = asyncio.Event()
    terminate_calls: list[int] = []

    class _BlockingStream:
        """Async-iterable that hangs forever — until ``teardown_signal`` fires.

        Mirrors what real subprocess stdout looks like before the model
        decides it's done: nothing comes out, the consumer await-blocks.
        ``ClaudeCliBackend``'s cancel watcher should call ``terminate()``
        on the proc when the event flips, which (in the real world)
        flushes stdout and EOFs the pipe — we model that by setting
        ``teardown_signal`` from the fake ``terminate()``.
        """

        def __aiter__(self) -> _BlockingStream:
            return self

        async def __anext__(self) -> bytes:
            await teardown_signal.wait()
            raise StopAsyncIteration

        async def read(self, n: int = -1) -> bytes:
            return b""

    class _CancellableProc:
        def __init__(self) -> None:
            self.stdout = _BlockingStream()
            self.stderr = _BlockingStream()
            self.stdin = _FakeStdin()
            self._returncode = -15  # SIGTERM-ish

        @property
        def returncode(self) -> int:
            return self._returncode

        def terminate(self) -> None:
            terminate_calls.append(1)
            teardown_signal.set()

        def kill(self) -> None:
            teardown_signal.set()

        async def wait(self) -> int:
            await teardown_signal.wait()
            return self._returncode

    proc = _CancellableProc()

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _CancellableProc:
        return proc

    monkeypatch.setattr(
        "psyneulink_agent.core.backends.claude_cli.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    backend = ClaudeCliBackend(mcp_url="http://x/sse")

    async def _drive_with_cancel() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        agen = backend.run_turn(
            history=[],
            system_prompt="s",
            user_content=[{"type": "text", "text": "long task"}],
            mcp=object(),
            tools=[],
            cancel_event=cancel_event,
        )

        async def _consume() -> None:
            async for ev in agen:
                events.append(ev)

        consumer = asyncio.create_task(_consume())
        # Let the generator spawn the subprocess and install its
        # cancel watcher before we flip the flag. A few yields is
        # plenty — the cancel-watcher task has to be scheduled.
        for _ in range(5):
            await asyncio.sleep(0)
        cancel_event.set()
        await consumer
        return events

    try:
        events = asyncio.run(asyncio.wait_for(_drive_with_cancel(), timeout=5.0))
    finally:
        backend.cleanup()

    assert terminate_calls == [1]
    assert any(e["type"] == "turn_cancelled" for e in events)
    assert all(e["type"] != "turn_complete" for e in events)


def test_run_turn_treats_result_is_error_as_error_stop_reason(monkeypatch) -> None:
    _install_fake_subprocess(
        monkeypatch,
        stdout_lines=[
            _line(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": True,
                    "result": "Failed to authenticate",
                    "stop_reason": "stop_sequence",
                }
            )
        ],
    )
    backend = ClaudeCliBackend(mcp_url="http://x/sse")
    try:
        events = _drive(
            backend.run_turn(
                history=[],
                system_prompt="s",
                user_content=[{"type": "text", "text": "x"}],
                mcp=object(),
                tools=[],
            )
        )
    finally:
        backend.cleanup()

    assert events[-1] == {"type": "turn_complete", "stop_reason": "error"}
