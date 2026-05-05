"""Interactive chat: spawn ``claude`` CLI with the MCP server attached.

LEGACY PATH — kept intact as the Claude Max fallback.

This module wires up ``--chat`` by spawning the ``claude`` CLI with our
MCP attached via ``--mcp-config``. It survives because users without an
``ANTHROPIC_API_KEY`` (e.g. Claude Max subscribers using the CLI)
should still be able to drive the agent against psyneulink-mcp without
buying API credit.

The newer ``--chat-sdk`` REPL (``psyneulink_agent.repl``) talks to
Anthropic via the Python SDK directly and is the foundation for the
upcoming web UI and ``--run`` headless mode. The two front-ends share
the modeling system prompt via ``psyneulink_agent.core.system_prompt``
so they can never drift.

Sandboxed environments (Cursor's shell tool) often can't actually run
this — ``claude`` opens a TTY. Run from a real terminal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import resolve_server_command
from .core.system_prompt import SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "chat"]


def _build_mcp_config(mcp_project: Path | None) -> dict[str, dict[str, dict[str, object]]]:
    """Produce a ``--mcp-config`` JSON document referencing our MCP server."""
    cmd = resolve_server_command(mcp_project)
    return {
        "mcpServers": {
            "psyneulink": {
                "command": cmd[0],
                "args": cmd[1:],
            }
        }
    }


def chat(
    mcp_project: Path | None = None,
    *,
    extra_claude_args: list[str] | None = None,
) -> int:
    """Drop the user into an interactive Claude session backed by the MCP.

    ``extra_claude_args`` is appended verbatim, so callers (or future
    CLI flags) can pass ``["--print", "build me ..."]`` for a one-shot
    smoke test, or ``["--model", "opus"]`` to override the model, etc.
    """
    if shutil.which("claude") is None:
        print(
            "error: `claude` CLI not found on PATH. Install it from "
            "https://docs.claude.com/en/docs/claude-code or set up an "
            "alternative LLM client.",
            file=sys.stderr,
        )
        return 2

    config = _build_mcp_config(mcp_project)

    # Use a tempfile rather than passing config as a literal string:
    # the JSON contains absolute paths and the command can be long
    # enough to upset some shells.
    fd, path_str = tempfile.mkstemp(prefix="psyneulink-agent-mcp-", suffix=".json")
    os.close(fd)
    cfg_path = Path(path_str)
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    argv = [
        "claude",
        "--mcp-config",
        str(cfg_path),
        "--append-system-prompt",
        SYSTEM_PROMPT,
    ]
    if extra_claude_args:
        argv.extend(extra_claude_args)

    try:
        return subprocess.run(argv).returncode
    finally:
        cfg_path.unlink(missing_ok=True)
