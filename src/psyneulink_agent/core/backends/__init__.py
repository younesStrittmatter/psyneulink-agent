"""LLM backends — strategies for driving one chat turn against an LLM.

Two interchangeable backends sit behind the same :class:`LLMBackend`
ABC:

* :class:`AnthropicSdkBackend` — calls Anthropic's Messages API via the
  Python SDK. Requires ``ANTHROPIC_API_KEY``. This is the path
  ``--chat-sdk`` and ``--run`` have always taken.
* :class:`ClaudeCliBackend` — spawns the ``claude`` CLI with
  ``--print --output-format stream-json --mcp-config <…>`` per turn,
  parses its stream-json output back into our backend-agnostic event
  shapes. No ``ANTHROPIC_API_KEY`` required — the user's Claude Max
  subscription pays the bill via the CLI's own auth.

:class:`Session` picks one at construction time (auto-detected from
the environment, overridable via ``PSYNEULINK_LLM_BACKEND={sdk,cli}``)
and uses ``backend.kind`` to decide the MCP transport: stdio for SDK,
SSE for CLI (so the CLI subprocess can share the same long-lived MCP
the front-end uses for out-of-loop ``call_tool`` invocations).
"""

from __future__ import annotations

from .anthropic_sdk import AnthropicSdkBackend
from .base import LLMBackend
from .claude_cli import ClaudeCliBackend

__all__ = ["LLMBackend", "AnthropicSdkBackend", "ClaudeCliBackend"]
