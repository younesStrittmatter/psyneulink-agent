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
* ``snapshot()`` — JSON-serialisable summary for autosave / debugging

The Anthropic client is constructed lazily on the first
``send_user_message`` call so that constructing a ``Session`` does NOT
require ``ANTHROPIC_API_KEY`` to be set (handy for tests + for the
slash-command REPL where ``/help`` should work even without an API key).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loop import run_turn
from .mcp_bridge import list_anthropic_tools, mcp_session
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

    def attach(self, resource: Resource) -> None:
        self.resources.append(resource)

    def detach(self, resource: Resource) -> None:
        self.resources.remove(resource)

    def system_prompt(self) -> str:
        return render_system_prompt(self.resources)

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
