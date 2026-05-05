"""``Session`` — opaque modeling-conversation handle for any front-end.

A ``Session`` owns the state that has to persist across user turns:

* Conversation ``history`` (mutated in place by ``run_turn``).
* Attached ``Resource`` instances (PDFs, data files, model files).
* Model name to dispatch against.
* The MCP project path to spawn psyneulink-mcp from.

Front-ends (the ``--chat-sdk`` REPL, the upcoming web UI, the future
``--run`` headless mode) all consume the same public API:

* ``attach`` / ``detach`` / ``resources``
* ``send_user_message(text)`` — async iterator yielding events
* ``lifespan()`` — async context manager that holds one MCP session open
  for the duration of the front-end (web UI, long REPL); inside it,
  ``send_user_message`` and ``call_tool`` reuse the same MCP connection
  so handles + journal state survive across calls
* ``call_tool(name, args)`` — invoke an MCP tool directly without going
  through the LLM; mainly for front-ends that need to poll the MCP for
  side data (e.g. the web UI rendering a graph between turns)
* ``snapshot()`` — JSON-serialisable summary for autosave / debugging

The Anthropic client is constructed lazily on the first
``send_user_message`` call so that constructing a ``Session`` does NOT
require ``ANTHROPIC_API_KEY`` to be set (handy for tests + for the
slash-command REPL where ``/help`` should work even without an API key).

Follow-up (out of scope for the refactor that introduced ``lifespan``):
the ``--chat-sdk`` REPL currently re-spawns the MCP per turn via the
fallback path. A future change should wrap the REPL in
``async with session.lifespan(): ...`` so its tool calls share handles
across turns the same way the web UI will.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loop import run_turn
from .mcp_bridge import call_mcp_tool, list_anthropic_tools, mcp_session
from .resources import Resource
from .system_prompt import render_system_prompt

DEFAULT_MODEL = os.environ.get("PSYNEULINK_AGENT_MODEL", "claude-sonnet-4-5-20250929")


@dataclass
class Session:
    """One modeling conversation. One front-end instance owns one of these."""

    mcp_project: Path | None = None
    model: str = DEFAULT_MODEL
    resources: list[Resource] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    # Active long-lived MCP ``ClientSession`` while inside ``lifespan()``.
    # ``None`` outside, in which case ``send_user_message`` and
    # ``call_tool`` fall back to per-call ``mcp_session`` spawns. Private
    # on purpose — front-ends interact with it via ``lifespan()`` /
    # ``call_tool()`` and never touch this attribute directly.
    _mcp: Any | None = field(default=None, init=False, repr=False)

    def attach(self, resource: Resource) -> None:
        self.resources.append(resource)

    def detach(self, resource: Resource) -> None:
        self.resources.remove(resource)

    def system_prompt(self) -> str:
        return render_system_prompt(self.resources)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[Session]:
        """Open a long-lived MCP connection for the duration of this session.

        Use this when the front-end needs to call MCP tools outside the
        LLM loop (e.g. the web UI polling ``render_composition_graph``
        between turns) AND share state — handles registered by tool calls
        in turn N must still resolve in turn N+1, in a direct
        :meth:`call_tool` invocation, or in a follow-up
        :meth:`send_user_message`.

        While inside the context manager :meth:`send_user_message` and
        :meth:`call_tool` reuse this same MCP session instead of spawning
        a fresh one per call. On exit, the connection is closed and the
        session reverts to per-call MCP spawns (the existing behaviour
        used by ``--chat-sdk`` and ``--run``).

        Not re-entrant: nesting ``lifespan()`` calls on the same Session
        raises :class:`RuntimeError`. Front-ends should hold the context
        manager open for the whole front-end lifetime (one per browser
        tab, one per ``--chat-sdk`` REPL, etc.).
        """
        if self._mcp is not None:
            raise RuntimeError(
                "Session.lifespan() is not re-entrant; only one active span at a time."
            )
        async with mcp_session(self.mcp_project) as mcp:
            self._mcp = mcp
            try:
                yield self
            finally:
                self._mcp = None

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> str:
        """Invoke an MCP tool by name and return its tool_result text.

        Works both inside and outside :meth:`lifespan`:

        * **Inside** ``lifespan()``: uses the active MCP session, so
          handles registered by previous turns / calls are still
          resolvable. This is the path the web UI takes when it calls
          ``render_composition_graph(handle)`` outside the LLM loop.
        * **Outside** ``lifespan()``: opens a per-call MCP session for
          this single tool call. Handles registered here do NOT persist
          across calls — useful for one-shot probes from tests or
          scripts, but for any UI-style polling pattern (graph render,
          revision check) you almost certainly want to be inside a
          ``lifespan()`` block.
        """
        args = args or {}
        if self._mcp is not None:
            return await call_mcp_tool(self._mcp, name, args)
        async with mcp_session(self.mcp_project) as mcp:
            return await call_mcp_tool(mcp, name, args)

    async def send_user_message(
        self,
        text: str,
        *,
        anthropic_client: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send ``text`` (plus pending resource attachments on first turn) and
        yield events from the model + tool calls.

        ``anthropic_client`` is injectable for tests; in production it
        defaults to ``anthropic.AsyncAnthropic()`` (which reads
        ``ANTHROPIC_API_KEY`` from the environment).

        If a :meth:`lifespan` is currently active, the long-lived MCP
        session is reused so MCP-side state (handle registry, journal,
        composition revisions) survives across turns and across direct
        :meth:`call_tool` invocations. Otherwise the legacy per-turn
        path is used: a fresh MCP server is spawned for this turn and
        torn down when it finishes. The fallback keeps ``--chat-sdk``
        and ``--run`` (which never enter ``lifespan()``) working
        unchanged.
        """
        if anthropic_client is None:
            from anthropic import AsyncAnthropic

            anthropic_client = AsyncAnthropic()

        is_first_turn = len(self.history) == 0
        content_blocks: list[dict[str, Any]] = []
        if is_first_turn:
            for res in self.resources:
                content_blocks.extend(res.as_anthropic_blocks())
        content_blocks.append({"type": "text", "text": text})

        if self._mcp is not None:
            tools = await list_anthropic_tools(self._mcp)
            async for event in run_turn(
                anthropic_client=anthropic_client,
                model=self.model,
                system_prompt=self.system_prompt(),
                history=self.history,
                user_content=content_blocks,
                mcp=self._mcp,
                tools=tools,
            ):
                yield event
            return

        async with mcp_session(self.mcp_project) as mcp:
            tools = await list_anthropic_tools(mcp)
            async for event in run_turn(
                anthropic_client=anthropic_client,
                model=self.model,
                system_prompt=self.system_prompt(),
                history=self.history,
                user_content=content_blocks,
                mcp=mcp,
                tools=tools,
            ):
                yield event

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable summary of the session.

        Useful for autosave (future) and for surfacing the session
        state to the future UI's resource dock without exposing the
        live ``Resource`` objects.
        """
        return {
            "model": self.model,
            "history": list(self.history),
            "resources": [
                {"kind": r.kind(), "label": r.label()} for r in self.resources
            ],
        }
