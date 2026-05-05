"""ABC for LLM backends.

The contract is intentionally narrow: one async method, ``run_turn``,
that drives a single user-turn → assistant-turn cycle and yields events
in the same protocol the existing :func:`psyneulink_agent.core.loop.run_turn`
emits (``text_chunk`` / ``tool_use`` / ``tool_result`` / ``turn_complete``,
plus an optional ``warning`` for backends that can't honor every
content-block kind).

History is mutated in place — backends append the user turn and any
assistant / tool-result turns produced during this cycle. Front-ends
re-pass the same ``history`` list across turns and never have to
reach into backend internals.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMBackend(ABC):
    """Strategy for driving one chat turn against an LLM with MCP tools available."""

    #: ``"sdk"`` or ``"cli"``. :class:`Session` reads this to pick the
    #: MCP transport (stdio for sdk, sse for cli) at lifespan time.
    kind: str

    @abstractmethod
    def run_turn(
        self,
        *,
        history: list[dict[str, Any]],
        system_prompt: str,
        user_content: list[dict[str, Any]],
        mcp: Any,
        tools: list[dict[str, Any]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive one user-turn → assistant-turn cycle.

        Yields events shaped like
        ``{"type": "text_chunk", "delta": "..."}`` /
        ``{"type": "tool_use", ...}`` / ``{"type": "tool_result", ...}`` /
        ``{"type": "turn_complete", "stop_reason": "..."}``.

        ``mcp`` and ``tools`` are passed for backends that route tool
        calls through the agent itself (the SDK backend). Backends that
        delegate tool wiring to the LLM client (the CLI backend, where
        ``claude`` reads the MCP config directly) may ignore them.

        ``cancel_event`` is the cooperative cancellation signal:
        :meth:`Session.cancel_current_turn` flips it, and a backend
        that supports cancellation must observe it (between LLM
        round-trips for the SDK, by killing the spawned ``claude``
        subprocess for the CLI), then yield a single
        ``{"type": "turn_cancelled"}`` event before returning. Backends
        may legitimately ignore the parameter — older / simpler
        implementations will just run to completion — but every front-
        end-facing backend in this repo honours it.
        """
