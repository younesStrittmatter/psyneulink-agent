"""Unit tests for the ``--chat`` plumbing.

We don't actually exec ``claude`` from these tests — that would require
an interactive TTY and a real LLM round trip. We assert that the MCP
config we hand to ``claude`` is well-formed and that the resolver chain
flows through ``resolve_server_command``.
"""

from __future__ import annotations

import json
from pathlib import Path

from psyneulink_agent import chat as chat_module


def test_build_mcp_config_is_well_formed(tmp_path: Path) -> None:
    fake_project = tmp_path / "psyneulink-mcp"
    fake_project.mkdir()
    config = chat_module._build_mcp_config(fake_project)
    assert "mcpServers" in config
    server = config["mcpServers"]["psyneulink"]
    assert server["command"]
    # `args` is the rest of the resolved argv, possibly empty for a
    # PATH-installed binary; either way it must be a list.
    assert isinstance(server["args"], list)
    # The whole thing must round-trip through JSON since that's how we
    # hand it to the `claude` CLI.
    json.dumps(config)


def test_system_prompt_mentions_handles_and_run_composition() -> None:
    """Cheap regression: the modeling instructions agents need most are present."""
    text = chat_module.SYSTEM_PROMPT
    for needle in (
        "handle",
        "create_transfer_mechanism",
        "create_composition",
        "add_linear_pathway",
        "run_composition",
    ):
        assert needle in text, f"system prompt missing {needle!r}"
