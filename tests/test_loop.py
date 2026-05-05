"""Unit tests for ``core.loop.run_turn``.

We mock the Anthropic SDK and the MCP session entirely; nothing here
ever calls a real network endpoint or spawns a subprocess. This is the
load-bearing test for the SDK path — it verifies the tool-use round
trip semantics and history bookkeeping that ``Session`` depends on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from psyneulink_agent.core.loop import run_turn

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _TextBlock:
    text: str
    type: str = "text"

    def model_dump(self, mode: str = "json", exclude_none: bool = True) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"

    def model_dump(self, mode: str = "json", exclude_none: bool = True) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class _Message:
    content: list[Any]
    stop_reason: str


class _FakeMessages:
    """Stand-in for ``anthropic_client.messages``."""

    def __init__(self, scripted: list[_Message]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Message:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("ran out of scripted Anthropic responses")
        return self._scripted.pop(0)


@dataclass
class _FakeAnthropic:
    messages: _FakeMessages


class _FakeMCP:
    """Stand-in for the MCP ``ClientSession`` passed into the loop.

    Only ``call_tool`` is actually used by ``run_turn`` (via
    ``call_mcp_tool``). We return a single text block per tool call.
    """

    def __init__(self, response_text: str = "tool ran") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))

        @dataclass
        class _R:
            content: list[Any]

        @dataclass
        class _T:
            text: str

        return _R(content=[_T(text=self.response_text)])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _drive(coro_iter: Any) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for ev in coro_iter:
            out.append(ev)
        return out

    return asyncio.run(_collect())


def test_run_turn_simple_text_response() -> None:
    history: list[dict[str, Any]] = []
    anthropic = _FakeAnthropic(
        messages=_FakeMessages(
            scripted=[_Message(content=[_TextBlock("hello there")], stop_reason="end_turn")]
        )
    )

    events = _drive(
        run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "hi"}],
            mcp=_FakeMCP(),
            tools=[],
        )
    )

    types = [e["type"] for e in events]
    assert types == ["text_chunk", "turn_complete"]
    assert events[0]["delta"] == "hello there"
    assert events[1]["stop_reason"] == "end_turn"

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == [{"type": "text", "text": "hi"}]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == [{"type": "text", "text": "hello there"}]


def test_run_turn_with_one_tool_call_round_trip() -> None:
    history: list[dict[str, Any]] = []
    anthropic = _FakeAnthropic(
        messages=_FakeMessages(
            scripted=[
                _Message(
                    content=[_ToolUseBlock(id="tu_1", name="create_x", input={"a": 1})],
                    stop_reason="tool_use",
                ),
                _Message(
                    content=[_TextBlock("done")],
                    stop_reason="end_turn",
                ),
            ]
        )
    )
    mcp = _FakeMCP(response_text="tool result body")

    events = _drive(
        run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "build x"}],
            mcp=mcp,
            tools=[{"name": "create_x", "description": "", "input_schema": {}}],
        )
    )

    types = [e["type"] for e in events]
    assert types == ["tool_use", "tool_result", "text_chunk", "turn_complete"]
    assert events[0] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "create_x",
        "input": {"a": 1},
    }
    assert events[1]["content"] == "tool result body"
    assert events[1]["is_error"] is False
    assert events[2]["delta"] == "done"
    assert events[3]["stop_reason"] == "end_turn"

    assert mcp.calls == [("create_x", {"a": 1})]

    # history layout: [user, assistant(tool_use), user(tool_result), assistant(text)]
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == [
        {"type": "tool_use", "id": "tu_1", "name": "create_x", "input": {"a": 1}}
    ]
    assert history[2]["role"] == "user"
    assert history[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "tool result body",
        }
    ]
    assert history[3]["role"] == "assistant"


def test_run_turn_tool_iteration_cap_emits_synthetic_complete() -> None:
    """If the model insists on tool_use forever, we must bail with a clean event."""
    history: list[dict[str, Any]] = []

    class _InfiniteToolUse(_FakeMessages):
        async def create(self, **kwargs: Any) -> _Message:
            return _Message(
                content=[_ToolUseBlock(id=f"tu_{len(self.calls)}", name="x")],
                stop_reason="tool_use",
            )

    anthropic = _FakeAnthropic(messages=_InfiniteToolUse(scripted=[]))
    mcp = _FakeMCP()

    events = _drive(
        run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "loop forever"}],
            mcp=mcp,
            tools=[{"name": "x", "description": "", "input_schema": {}}],
            max_tool_iterations=3,
        )
    )

    last = events[-1]
    assert last == {"type": "turn_complete", "stop_reason": "tool_iteration_cap"}


def test_run_turn_cancel_event_set_before_first_iteration_emits_turn_cancelled() -> None:
    """If the cancel flag is already set when ``run_turn`` starts, we
    bail before hitting the network: no ``messages.create`` call, one
    ``turn_cancelled`` event, done."""
    history: list[dict[str, Any]] = []
    fake_messages = _FakeMessages(scripted=[])
    anthropic = _FakeAnthropic(messages=fake_messages)

    cancel_event = asyncio.Event()
    cancel_event.set()

    events = _drive(
        run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "irrelevant"}],
            mcp=_FakeMCP(),
            tools=[],
            cancel_event=cancel_event,
        )
    )

    assert events == [{"type": "turn_cancelled"}]
    assert fake_messages.calls == []


def test_run_turn_cancel_during_messages_create_emits_turn_cancelled() -> None:
    """Flipping the cancel flag while ``messages.create`` is awaited
    must cause that in-flight request to be cancelled and the loop to
    exit with a single ``turn_cancelled`` event — proving the
    interruptible await wraps the SDK call, not just the iteration
    boundary."""
    history: list[dict[str, Any]] = []
    cancel_event = asyncio.Event()
    create_started = asyncio.Event()

    class _SlowMessages:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> _Message:
            self.calls.append(kwargs)
            create_started.set()
            await asyncio.sleep(60)
            raise AssertionError("messages.create should have been cancelled")

    @dataclass
    class _SlowAnthropic:
        messages: _SlowMessages

    anthropic = _SlowAnthropic(messages=_SlowMessages())

    async def _drive_with_cancel() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        agen = run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "long task"}],
            mcp=_FakeMCP(),
            tools=[],
            cancel_event=cancel_event,
        )

        async def _trip_cancel() -> None:
            await create_started.wait()
            cancel_event.set()

        canceller = asyncio.create_task(_trip_cancel())
        async for ev in agen:
            events.append(ev)
        await canceller
        return events

    events = asyncio.run(asyncio.wait_for(_drive_with_cancel(), timeout=5.0))
    assert events == [{"type": "turn_cancelled"}]


def test_run_turn_records_tool_error_without_crashing() -> None:
    history: list[dict[str, Any]] = []

    class _BoomMCP:
        async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            raise RuntimeError("backend down")

    anthropic = _FakeAnthropic(
        messages=_FakeMessages(
            scripted=[
                _Message(
                    content=[_ToolUseBlock(id="tu_1", name="x")],
                    stop_reason="tool_use",
                ),
                _Message(content=[_TextBlock("ok")], stop_reason="end_turn"),
            ]
        )
    )

    events = _drive(
        run_turn(
            anthropic_client=anthropic,
            model="claude-test",
            system_prompt="sys",
            history=history,
            user_content=[{"type": "text", "text": "go"}],
            mcp=_BoomMCP(),
            tools=[{"name": "x", "description": "", "input_schema": {}}],
        )
    )

    tool_result_evt = next(e for e in events if e["type"] == "tool_result")
    assert tool_result_evt["is_error"] is True
    assert "RuntimeError" in tool_result_evt["content"]

    # The injected tool_result block in history must carry is_error=True
    # so the model can react appropriately on its retry.
    assert history[2]["content"][0]["is_error"] is True
