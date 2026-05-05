"""Unit tests for the ``--chat-sdk`` REPL.

We test ``_handle_slash`` and ``_render_event`` in isolation. The
async stdin loop in ``repl()`` is not exercised here — that's a TTY
integration concern.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from psyneulink_agent import repl as repl_module


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _handle_slash
# ---------------------------------------------------------------------------


def test_help_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash("/help", session))
    assert cont is True
    out = capsys.readouterr().out
    assert "/load-pdf" in out
    assert "/exit" in out


def test_exit_command_returns_false() -> None:
    session = repl_module.Session()
    assert _run(repl_module._handle_slash("/exit", session)) is False
    assert _run(repl_module._handle_slash("/quit", session)) is False


def test_unknown_command_keeps_repl_alive(capsys: pytest.CaptureFixture[str]) -> None:
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash("/wat", session))
    assert cont is True
    assert "unknown command" in capsys.readouterr().out


def test_load_pdf_attaches_resource(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    session = repl_module.Session()

    cont = _run(repl_module._handle_slash(f"/load-pdf {pdf}", session))
    assert cont is True
    assert len(session.resources) == 1
    assert session.resources[0].kind() == "pdf"
    assert "attached" in capsys.readouterr().out


def test_load_pdf_with_missing_arg_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash("/load-pdf", session))
    assert cont is True
    assert "usage" in capsys.readouterr().out


def test_load_pdf_with_bad_path_keeps_repl_alive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash("/load-pdf /no/such/file.pdf", session))
    assert cont is True
    assert "failed" in capsys.readouterr().out
    assert session.resources == []


def test_load_data_attaches_resource(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "trials.csv"
    csv.write_text("a,b\n1,2\n")
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash(f"/load-data {csv}", session))
    assert cont is True
    assert len(session.resources) == 1
    assert session.resources[0].kind() == "data"
    assert "attached" in capsys.readouterr().out


def test_load_model_attaches_resource(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    py = tmp_path / "m.py"
    py.write_text("x = 1\n")
    session = repl_module.Session()
    cont = _run(repl_module._handle_slash(f"/load-model {py}", session))
    assert cont is True
    assert len(session.resources) == 1
    assert session.resources[0].kind() == "model"
    out = capsys.readouterr().out
    assert "attached" in out
    assert "load_python_script" in out


def test_resources_command_lists_or_says_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = repl_module.Session()
    _run(repl_module._handle_slash("/resources", session))
    assert "no resources" in capsys.readouterr().out

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    session.attach(repl_module.PdfResource(pdf))
    _run(repl_module._handle_slash("/resources", session))
    assert "paper.pdf" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _render_event
# ---------------------------------------------------------------------------


def test_render_event_handles_every_event_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repl_module._render_event({"type": "text_chunk", "delta": "hello "})
    repl_module._render_event(
        {"type": "tool_use", "name": "create_x", "input": {"a": 1}, "id": "tu_1"}
    )
    repl_module._render_event(
        {
            "type": "tool_result",
            "name": "create_x",
            "id": "tu_1",
            "content": "ok",
            "is_error": False,
        }
    )
    repl_module._render_event(
        {
            "type": "tool_result",
            "name": "create_x",
            "id": "tu_1",
            "content": "boom",
            "is_error": True,
        }
    )
    repl_module._render_event({"type": "turn_complete", "stop_reason": "end_turn"})
    repl_module._render_event({"type": "wat_unknown"})  # must not crash

    captured = capsys.readouterr()
    assert "hello " in captured.out
    assert "[tool]" in captured.err
    assert "tool ok" in captured.err
    assert "tool err" in captured.err
