"""Integration test: actually spawns psyneulink-mcp via ``uv run``.

Skipped by default (``-m 'not integration'`` in addopts).
Run explicitly with::

    uv run pytest -q -m integration

Requirements:
- ``uv`` must be on PATH.
- The sibling ``../psyneulink-mcp`` repo must exist (or set
  ``$PSYNEULINK_MCP_PROJECT`` to an alternate path).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from psyneulink_agent.client import connect_and_list
from psyneulink_agent.config import _REPO_ROOT, ENV_MCP_PROJECT

_MCP_SIBLING = _REPO_ROOT.parent / "psyneulink-mcp"

# Tools that Phase 3 guarantees to be present
_EXPECTED_TOOLS = {"get_community_brainlike_views", "get_my_brainlike_view"}


@pytest.mark.integration
def test_list_tools_via_mcp() -> None:
    """Spawn psyneulink-mcp, list tools, assert the Phase-3 tools are present."""
    if not shutil.which("uv"):
        pytest.skip("uv not found on PATH — cannot spawn MCP server")

    mcp_path: Path | None = None
    env_val = os.environ.get(ENV_MCP_PROJECT)
    if env_val:
        mcp_path = Path(env_val)
    elif _MCP_SIBLING.exists():
        mcp_path = _MCP_SIBLING
    else:
        pytest.skip(
            f"psyneulink-mcp not found at {_MCP_SIBLING} "
            f"and ${ENV_MCP_PROJECT} is not set"
        )

    tools = asyncio.run(connect_and_list(mcp_path))
    names = {t.name for t in tools}

    for expected in _EXPECTED_TOOLS:
        assert expected in names, f"{expected!r} missing from tool list: {sorted(names)}"
