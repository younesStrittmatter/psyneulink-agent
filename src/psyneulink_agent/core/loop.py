"""One-turn Anthropic SDK loop with MCP-backed tool calls.

``run_turn`` is the canonical "send a user turn, get an assistant turn
back, possibly with tool calls in the middle" cycle. It's an async
iterator so a UI / REPL can render events as they happen instead of
blocking until the whole turn finishes.

Design constraints:

* The ``anthropic_client`` is **injectable** — production code passes
  a real ``anthropic.AsyncAnthropic()``; tests pass a mock that yields
  prebuilt ``Message`` shapes. Nothing inside this module imports the
  real Anthropic client at module load time.
* History is mutated **in place**. The caller (``Session``) owns the
  list and re-uses it across turns; we just append assistant + tool
  result messages to it.
* Streaming is OPTIONAL for the MVP. The current implementation calls
  the non-streaming ``messages.create`` and emits one ``text_chunk``
  event per text block in the assistant response. The async-iterator
  signature is preserved so the future UI can switch to true SSE
  streaming without changing callers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .mcp_bridge import call_mcp_tool


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an anthropic SDK content block (pydantic model) → plain dict.

    Used to round-trip the assistant message back into ``messages``
    history on subsequent turns.
    """
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    if isinstance(block, dict):
        return dict(block)
    raise TypeError(f"unsupported content block: {type(block).__name__}")


async def run_turn(
    *,
    anthropic_client: Any,
    model: str,
    system_prompt: str,
    history: list[dict[str, Any]],
    user_content: list[dict[str, Any]],
    mcp: Any,
    tools: list[dict[str, Any]],
    max_tool_iterations: int = 16,
    max_tokens: int = 4096,
) -> AsyncIterator[dict[str, Any]]:
    """One full user-turn → assistant-turn cycle, possibly with tool calls.

    Yields events shaped like:

    * ``{"type": "text_chunk", "delta": "..."}`` — assistant text
    * ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``
    * ``{"type": "tool_result", "id": "...", "name": "...", "content": "..."}``
    * ``{"type": "turn_complete", "stop_reason": "..."}``

    On completion, ``history`` has the user turn and every assistant +
    tool-result message appended in order, so the next turn just calls
    ``run_turn`` again with another ``user_content``.
    """
    history.append({"role": "user", "content": user_content})

    for _ in range(max_tool_iterations):
        response = await anthropic_client.messages.create(
            model=model,
            system=system_prompt,
            messages=history,
            tools=tools,
            max_tokens=max_tokens,
        )

        assistant_blocks = [_block_to_dict(b) for b in response.content]
        history.append({"role": "assistant", "content": assistant_blocks})

        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text = getattr(block, "text", "")
                if text:
                    yield {"type": "text_chunk", "delta": text}

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason != "tool_use":
            yield {"type": "turn_complete", "stop_reason": stop_reason}
            return

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_id = getattr(block, "id", "")
            tool_name = getattr(block, "name", "")
            tool_input = getattr(block, "input", {}) or {}
            yield {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }
            try:
                result_str = await call_mcp_tool(mcp, tool_name, tool_input)
                is_error = False
            except Exception as exc:  # noqa: BLE001 — surface to LLM, don't crash
                result_str = f"tool error: {type(exc).__name__}: {exc}"
                is_error = True
            yield {
                "type": "tool_result",
                "id": tool_id,
                "name": tool_name,
                "content": result_str,
                "is_error": is_error,
            }
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                    **({"is_error": True} if is_error else {}),
                }
            )

        history.append({"role": "user", "content": tool_results})

    yield {"type": "turn_complete", "stop_reason": "tool_iteration_cap"}
