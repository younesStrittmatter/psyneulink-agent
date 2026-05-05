"""Unit tests for ``core.session.Session`` and ``render_system_prompt``.

We don't construct a real Anthropic client or MCP session — those paths
are exercised in ``test_loop`` and integration tests. Here we only
verify the dataclass-level behaviour: attaching, detaching, prompt
rendering, snapshotting.
"""

from __future__ import annotations

import json
from pathlib import Path

from psyneulink_agent.core.resources import (
    DataResource,
    ModelFileResource,
    PdfResource,
)
from psyneulink_agent.core.session import Session
from psyneulink_agent.core.system_prompt import SYSTEM_PROMPT, render_system_prompt

# ---------------------------------------------------------------------------
# render_system_prompt
# ---------------------------------------------------------------------------


def test_render_system_prompt_with_no_resources_returns_base_prompt() -> None:
    assert render_system_prompt(None) == SYSTEM_PROMPT
    assert render_system_prompt([]) == SYSTEM_PROMPT


def test_render_system_prompt_appends_attached_resource_summary(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")

    rendered = render_system_prompt([PdfResource(pdf), DataResource(csv)])

    assert rendered.startswith(SYSTEM_PROMPT)
    assert "Attached resources for this session:" in rendered
    assert "- [pdf] paper.pdf" in rendered
    assert "- [data] data.csv" in rendered


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_attach_and_detach(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    pdf = PdfResource(pdf_path)

    session = Session()
    assert session.resources == []
    session.attach(pdf)
    assert session.resources == [pdf]
    session.detach(pdf)
    assert session.resources == []


def test_session_system_prompt_includes_resource_summary(tmp_path: Path) -> None:
    py_path = tmp_path / "model.py"
    py_path.write_text("# pretend model")
    session = Session()
    assert session.system_prompt() == SYSTEM_PROMPT  # bare session = base prompt
    session.attach(ModelFileResource(py_path))
    out = session.system_prompt()
    assert "model.py" in out
    assert out.startswith(SYSTEM_PROMPT)


def test_session_snapshot_is_json_serialisable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    session = Session(model="claude-test-model")
    session.attach(PdfResource(pdf_path))
    session.history.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})

    snap = session.snapshot()
    assert snap["model"] == "claude-test-model"
    assert snap["resources"] == [{"kind": "pdf", "label": "paper.pdf"}]
    assert snap["history"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    # Round-trip through JSON to confirm it's serialisable.
    json.dumps(snap)


def test_session_snapshot_history_is_a_copy(tmp_path: Path) -> None:
    """Mutating the returned history must not affect the live session."""
    session = Session()
    session.history.append({"role": "user", "content": []})
    snap = session.snapshot()
    snap["history"].append({"role": "user", "content": [{"type": "text", "text": "x"}]})
    assert len(session.history) == 1
