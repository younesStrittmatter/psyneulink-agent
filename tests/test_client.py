"""Unit tests for ``config.resolve_server_command``.

These tests cover command resolution logic only — no live MCP server needed.
Live-server coverage lives in ``test_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from psyneulink_agent.config import (
    _REPO_ROOT,
    ENV_MCP_PROJECT,
    resolve_server_command,
)


def test_explicit_path_is_used(tmp_path: Path) -> None:
    """An explicit mcp_project argument takes priority over everything else."""
    project = tmp_path / "fake-mcp"
    project.mkdir()
    cmd = resolve_server_command(project)
    assert cmd == ["uv", "run", "--project", str(project), "psyneulink-mcp"]


def test_env_var_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``$PSYNEULINK_MCP_PROJECT`` is used when no explicit path is given."""
    project = tmp_path / "env-mcp"
    project.mkdir()
    monkeypatch.setenv(ENV_MCP_PROJECT, str(project))
    # Patch _REPO_ROOT so the sibling auto-detect doesn't fire first
    with patch("psyneulink_agent.config._REPO_ROOT", tmp_path / "agent"):
        cmd = resolve_server_command()
    assert cmd == ["uv", "run", "--project", str(project), "psyneulink-mcp"]


def test_sibling_path_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to ``../psyneulink-mcp`` when env var is absent and path exists."""
    monkeypatch.delenv(ENV_MCP_PROJECT, raising=False)
    sibling = _REPO_ROOT.parent / "psyneulink-mcp"
    if not sibling.exists():
        pytest.skip(f"sibling psyneulink-mcp not present at {sibling}")
    cmd = resolve_server_command()
    assert cmd == ["uv", "run", "--project", str(sibling), "psyneulink-mcp"]


def test_path_fallback_to_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Returns ``["psyneulink-mcp"]`` when no project path exists but binary is on PATH."""
    monkeypatch.delenv(ENV_MCP_PROJECT, raising=False)
    fake_root = tmp_path / "fake-agent"
    fake_root.mkdir()
    with (
        patch("psyneulink_agent.config._REPO_ROOT", fake_root),
        patch("shutil.which", return_value="/usr/local/bin/psyneulink-mcp"),
    ):
        cmd = resolve_server_command()
    assert cmd == ["psyneulink-mcp"]
