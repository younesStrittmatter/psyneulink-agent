"""Agent core — Anthropic-SDK-driven modeling loop, decoupled from any front-end.

This package is what every front-end (the ``--chat-sdk`` REPL today,
the upcoming web UI, the future ``--run`` headless mode) imports to
drive a modeling conversation against psyneulink-mcp. The legacy
``--chat`` (claude CLI subprocess) path still lives in
``psyneulink_agent.chat`` and shares only the system prompt with this
package.
"""

from __future__ import annotations

from .loop import run_turn
from .resources import DataResource, ModelFileResource, PdfResource, Resource
from .session import Session
from .system_prompt import SYSTEM_PROMPT, render_system_prompt

__all__ = [
    "Session",
    "Resource",
    "PdfResource",
    "DataResource",
    "ModelFileResource",
    "run_turn",
    "SYSTEM_PROMPT",
    "render_system_prompt",
]
