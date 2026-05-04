"""Server command resolution for psyneulink-agent.

Centralises all decisions about *how* to launch psyneulink-mcp so that
``client.py`` and tests can import a single helper instead of duplicating
path-resolution logic.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV_MCP_PROJECT = "PSYNEULINK_MCP_PROJECT"

# src/psyneulink_agent/config.py  →  ../../..  = repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_server_command(mcp_project: Path | None = None) -> list[str]:
    """Return the argv required to spawn the MCP server.

    Resolution order:

    1. Explicit *mcp_project* argument (absolute, or relative to CWD).
    2. ``$PSYNEULINK_MCP_PROJECT`` environment variable.
    3. ``../psyneulink-mcp`` resolved against this repo's parent.
    4. Plain ``psyneulink-mcp`` found on ``$PATH`` (no ``uv run`` wrapper).

    Raises ``FileNotFoundError`` when none of the four options yields a
    usable command.
    """
    project: Path | None = mcp_project

    if project is None:
        env_val = os.environ.get(ENV_MCP_PROJECT)
        if env_val:
            project = Path(env_val)

    if project is None:
        candidate = _REPO_ROOT.parent / "psyneulink-mcp"
        if candidate.exists():
            project = candidate

    if project is not None:
        return ["uv", "run", "--project", str(project), "psyneulink-mcp"]

    if shutil.which("psyneulink-mcp"):
        return ["psyneulink-mcp"]

    raise FileNotFoundError(
        "psyneulink-mcp not found. "
        f"Set ${ENV_MCP_PROJECT} to the MCP repo path, "
        "or install psyneulink-mcp into the active environment."
    )
