"""Anthropic SDK backend — wraps :func:`psyneulink_agent.core.loop.run_turn`.

This is the original code path lifted into a strategy class; behavior
is identical to today's ``Session.send_user_message`` body. The MCP
transport in this mode is stdio (handled by ``Session.lifespan()``);
``run_turn`` calls ``mcp.call_tool`` directly through ``call_mcp_tool``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from .base import LLMBackend

DEFAULT_MODEL = os.environ.get("PSYNEULINK_AGENT_MODEL", "claude-sonnet-4-5-20250929")


class AnthropicSdkBackend(LLMBackend):
    """Drives one chat turn via Anthropic's Messages API."""

    kind = "sdk"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        anthropic_client: Any | None = None,
    ):
        self.model = model
        # Stored as ``_client`` so that ``Session.send_user_message`` can
        # rebind a test-injected client onto the live backend instance
        # without going through the constructor.
        self._client = anthropic_client

    async def run_turn(
        self,
        *,
        history: list[dict[str, Any]],
        system_prompt: str,
        user_content: list[dict[str, Any]],
        mcp: Any,
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        # Lazy import via the session module so tests that monkeypatch
        # ``psyneulink_agent.core.session.run_turn`` (the original
        # binding before the backend split) still see their override
        # flow through this backend. The session module re-exports
        # ``loop.run_turn`` for exactly this reason.
        from .. import session as _session

        client = self._client
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        async for event in _session.run_turn(
            anthropic_client=client,
            model=self.model,
            system_prompt=system_prompt,
            history=history,
            user_content=user_content,
            mcp=mcp,
            tools=tools,
        ):
            yield event
