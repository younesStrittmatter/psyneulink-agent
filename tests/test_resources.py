"""Unit tests for the ``core.resources`` Resource hierarchy.

We don't need a live Anthropic client or MCP server here — Resources
just translate file paths into Anthropic-shaped content blocks.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from psyneulink_agent.core.resources import (
    DataResource,
    ModelFileResource,
    PdfResource,
)

# ---------------------------------------------------------------------------
# PdfResource
# ---------------------------------------------------------------------------


def test_pdf_resource_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfResource(tmp_path / "nope.pdf")


def test_pdf_resource_rejects_non_pdf_suffix(tmp_path: Path) -> None:
    not_pdf = tmp_path / "paper.txt"
    not_pdf.write_text("hello")
    with pytest.raises(ValueError, match="not a .pdf"):
        PdfResource(not_pdf)


def test_pdf_resource_emits_base64_document_block(tmp_path: Path) -> None:
    payload = b"%PDF-1.4 fake pdf bytes"
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(payload)

    res = PdfResource(pdf_path)
    blocks = res.as_anthropic_blocks()

    assert len(blocks) == 1
    block = blocks[0]
    assert block["type"] == "document"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["source"]["data"] == base64.b64encode(payload).decode("ascii")
    assert block["title"] == "paper.pdf"

    assert res.kind() == "pdf"
    assert res.label() == "paper.pdf"
    assert res.summary_line() == "- [pdf] paper.pdf"


# ---------------------------------------------------------------------------
# DataResource
# ---------------------------------------------------------------------------


def test_data_resource_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DataResource(tmp_path / "nope.csv")


def test_data_resource_emits_text_block_pointing_at_psyche_tool(tmp_path: Path) -> None:
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text("subject_id,trial_global,step\n1,1,1\n")

    res = DataResource(csv_path)
    blocks = res.as_anthropic_blocks()

    assert len(blocks) == 1
    block = blocks[0]
    assert block["type"] == "text"
    assert "load_psyche_data" in block["text"]
    assert str(csv_path.resolve()) in block["text"]

    assert res.kind() == "data"
    assert res.label() == "trials.csv"
    assert res.summary_line() == "- [data] trials.csv"


# ---------------------------------------------------------------------------
# ModelFileResource
# ---------------------------------------------------------------------------


def test_model_file_resource_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelFileResource(tmp_path / "nope.py")


def test_model_file_resource_rejects_non_py_suffix(tmp_path: Path) -> None:
    not_py = tmp_path / "thing.txt"
    not_py.write_text("hello")
    with pytest.raises(ValueError, match="not a .py"):
        ModelFileResource(not_py)


def test_model_file_resource_emits_text_block_with_head_and_load_tool(
    tmp_path: Path,
) -> None:
    lines = [f"line {i}" for i in range(50)]
    py_path = tmp_path / "model.py"
    py_path.write_text("\n".join(lines), encoding="utf-8")

    res = ModelFileResource(py_path)
    blocks = res.as_anthropic_blocks()

    assert len(blocks) == 1
    block = blocks[0]
    assert block["type"] == "text"
    assert "load_python_script" in block["text"]
    assert str(py_path.resolve()) in block["text"]
    # Head preview present for the first lines, but not for line 30+ (cap is 30 lines).
    assert "line 0" in block["text"]
    assert "line 29" in block["text"]
    assert "line 35" not in block["text"]

    assert res.kind() == "model"
    assert res.label() == "model.py"
    assert res.summary_line() == "- [model] model.py"
