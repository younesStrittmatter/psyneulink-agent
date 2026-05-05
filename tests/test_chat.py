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


def test_system_prompt_pins_model_means_composition() -> None:
    """A "model" must be defined as a Composition — pin this without over-fitting wording.

    The agent's contract with the user is that the top-level artifact is
    always a ``pnl.Composition``. We keep this assertion loose: it must
    say (case-insensitively) that "model" means a "Composition", and the
    two words must appear close enough together to plausibly be the same
    sentence. We don't pin the exact phrasing so the prompt can keep
    being edited for tone without breaking the test.
    """
    text = chat_module.SYSTEM_PROMPT
    lower = text.lower()
    assert "composition" in lower
    model_idx = lower.find('"model"')
    assert model_idx != -1, 'system prompt must scope what "model" means'
    window = lower[model_idx : model_idx + 400]
    assert "composition" in window, (
        '"model" must be explicitly tied to "Composition" in the system prompt'
    )
    assert "nested" in lower or "subcomposition" in lower, (
        "system prompt must acknowledge that Compositions can contain Compositions"
    )
