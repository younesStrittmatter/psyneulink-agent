"""Unit tests for ``psyneulink_agent.runner`` and the ``--run`` CLI flag.

We mock :class:`psyneulink_agent.runner.Session` entirely — neither
Anthropic nor the MCP server is touched. The runner is a thin
orchestration layer over ``Session.send_user_message``, so canned async
events from a fake Session is the right level to test it at.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from psyneulink_agent import runner as runner_mod
from psyneulink_agent.main import _build_parser
from psyneulink_agent.runner import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_MAX_TURNS,
    RunSpec,
    SpecError,
    load_spec,
)

# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def _write_spec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_spec_minimal(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 1\ngoal: hello\n",
    )
    spec = load_spec(spec_file)
    assert spec.goal == "hello"
    assert spec.resources == []
    assert spec.artifacts == {}
    assert spec.max_turns == DEFAULT_MAX_TURNS
    assert spec.max_tool_iterations == DEFAULT_MAX_TOOL_ITERATIONS
    assert spec.model is None
    assert spec.spec_path == spec_file.resolve()
    # Default report path sits next to the spec.
    assert spec.report_path == spec_file.with_suffix(".yaml.report.json").resolve()


def test_load_spec_with_resources_paths_resolved_relative_to_spec(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    sub = tmp_path / "specs"
    sub.mkdir()
    spec_file = _write_spec(
        sub / "spec.yaml",
        "version: 1\n"
        "goal: anything\n"
        "resources:\n"
        "  - kind: pdf\n"
        "    path: ../paper.pdf\n",
    )
    spec = load_spec(spec_file)
    assert len(spec.resources) == 1
    assert spec.resources[0].kind() == "pdf"
    assert spec.resources[0].path == pdf_path.resolve()


def test_load_spec_rejects_missing_goal(tmp_path: Path) -> None:
    spec_file = _write_spec(tmp_path / "spec.yaml", "version: 1\n")
    with pytest.raises(SpecError, match="goal"):
        load_spec(spec_file)


def test_load_spec_rejects_unknown_version(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 99\ngoal: hi\n",
    )
    with pytest.raises(SpecError, match="version"):
        load_spec(spec_file)


def test_load_spec_rejects_unknown_kind(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 1\n"
        "goal: hi\n"
        "resources:\n"
        "  - kind: weird\n"
        "    path: /tmp/nope\n",
    )
    with pytest.raises(SpecError, match="kind must be"):
        load_spec(spec_file)


def test_load_spec_rejects_missing_resource_file(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 1\n"
        "goal: hi\n"
        "resources:\n"
        "  - kind: pdf\n"
        "    path: ./does_not_exist.pdf\n",
    )
    with pytest.raises(SpecError):
        load_spec(spec_file)


def test_load_spec_artifacts_paths_resolved(tmp_path: Path) -> None:
    sub = tmp_path / "specs"
    sub.mkdir()
    spec_file = _write_spec(
        sub / "spec.yaml",
        "version: 1\n"
        "goal: hi\n"
        "artifacts:\n"
        "  python_script: ../out/model.py\n",
    )
    spec = load_spec(spec_file)
    assert "python_script" in spec.artifacts
    assert spec.artifacts["python_script"] == (tmp_path / "out" / "model.py").resolve()


def test_load_spec_default_report_path(tmp_path: Path) -> None:
    spec_file = _write_spec(tmp_path / "smoke.yaml", "version: 1\ngoal: x\n")
    spec = load_spec(spec_file)
    assert spec.report_path is not None
    assert spec.report_path.name == "smoke.yaml.report.json"
    assert spec.report_path.parent == tmp_path.resolve()


def test_load_spec_explicit_report_path_relative(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 1\ngoal: x\nreport: ./reports/run.json\n",
    )
    spec = load_spec(spec_file)
    assert spec.report_path == (tmp_path / "reports" / "run.json").resolve()


def test_load_spec_limits_and_model(tmp_path: Path) -> None:
    spec_file = _write_spec(
        tmp_path / "spec.yaml",
        "version: 1\n"
        "goal: x\n"
        "limits:\n"
        "  max_turns: 3\n"
        "  max_tool_iterations: 5\n"
        "model: claude-test-model\n",
    )
    spec = load_spec(spec_file)
    assert spec.max_turns == 3
    assert spec.max_tool_iterations == 5
    assert spec.model == "claude-test-model"


def test_load_spec_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "absent.yaml")


def test_load_spec_non_mapping_top_level(tmp_path: Path) -> None:
    spec_file = _write_spec(tmp_path / "spec.yaml", "- version: 1\n  goal: x\n")
    with pytest.raises(SpecError, match="mapping"):
        load_spec(spec_file)


# ---------------------------------------------------------------------------
# run_spec — driven by a fake Session
# ---------------------------------------------------------------------------


_DEFAULT_TURN_EVENTS = [{"type": "turn_complete", "stop_reason": "end_turn"}]


class _FakeSession:
    """Stand-in for :class:`psyneulink_agent.core.Session`.

    Records every ``send_user_message`` call. Yields a canned event
    stream per call so the runner's record-keeping can be inspected.
    ``model`` is left mutable so the runner's ``spec.model`` override
    path can be exercised.
    """

    def __init__(
        self,
        *,
        mcp_project: Path | None = None,
        events_per_turn: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.mcp_project = mcp_project
        self.model = "fake-default-model"
        self.resources: list[Any] = []
        self.history: list[dict[str, Any]] = []
        self.messages: list[str] = []
        self._events_per_turn = list(events_per_turn) if events_per_turn else []

    def attach(self, resource: Any) -> None:
        self.resources.append(resource)

    async def send_user_message(
        self,
        text: str,
        *,
        anthropic_client: Any | None = None,
    ):
        self.messages.append(text)
        events = self._events_per_turn.pop(0) if self._events_per_turn else list(
            _DEFAULT_TURN_EVENTS
        )
        for ev in events:
            yield ev


def _install_fake_session(monkeypatch: pytest.MonkeyPatch, fake_factory: Any) -> list[_FakeSession]:
    """Patch ``runner.Session`` so each call returns the next fake instance."""
    created: list[_FakeSession] = []

    def _factory(*args: Any, **kwargs: Any) -> _FakeSession:
        sess = fake_factory(*args, **kwargs)
        created.append(sess)
        return sess

    monkeypatch.setattr(runner_mod, "Session", _factory)
    return created


def test_run_spec_sends_goal_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec(
        goal="build me a model",
        report_path=tmp_path / "out.report.json",
        spec_path=tmp_path / "spec.yaml",
    )

    created = _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [
                    {"type": "text_chunk", "delta": "hi"},
                    {"type": "turn_complete", "stop_reason": "end_turn"},
                ]
            ],
            **kw,
        ),
    )

    report = runner_mod.run_spec(spec)

    assert len(created) == 1
    sess = created[0]
    assert sess.messages == ["build me a model"]
    assert report["turns_sent"] == 1
    assert report["goal"] == "build me a model"
    assert report["model"] == "fake-default-model"
    assert report["artifacts"] == {}
    assert report["ok"] is True
    assert report["tool_calls"] == []
    assert (tmp_path / "out.report.json").exists()
    on_disk = json.loads((tmp_path / "out.report.json").read_text())
    assert on_disk["goal"] == "build me a model"


def test_run_spec_applies_model_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec(
        goal="x",
        model="claude-spec-model",
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    created = _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [{"type": "turn_complete", "stop_reason": "end_turn"}]
            ],
            **kw,
        ),
    )
    report = runner_mod.run_spec(spec)
    assert created[0].model == "claude-spec-model"
    assert report["model"] == "claude-spec-model"


def test_run_spec_sends_artifact_nudges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model_out.py"
    artifact.write_text("# pretend the LLM wrote this\n")  # exists before run

    spec = RunSpec(
        goal="construct it",
        artifacts={"python_script": artifact},
        report_path=tmp_path / "report.json",
        spec_path=tmp_path / "spec.yaml",
    )
    created = _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
            ],
            **kw,
        ),
    )

    report = runner_mod.run_spec(spec)

    sess = created[0]
    assert len(sess.messages) == 2
    assert sess.messages[0] == "construct it"
    nudge = sess.messages[1]
    assert str(artifact) in nudge
    assert "export_python_script" in nudge
    assert report["turns_sent"] == 2
    assert report["artifacts"]["python_script"]["status"] == "exists"
    assert report["ok"] is True


def test_run_spec_respects_max_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.py"  # NOT created
    spec = RunSpec(
        goal="g",
        artifacts={"python_script": artifact},
        max_turns=1,
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    created = _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
            ],
            **kw,
        ),
    )

    report = runner_mod.run_spec(spec)

    assert len(created[0].messages) == 1
    assert created[0].messages[0] == "g"
    assert report["turns_sent"] == 1
    assert report["artifacts"]["python_script"]["status"] == "missing"
    assert report["ok"] is False


def test_run_spec_returns_ok_false_when_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "absent.py"
    spec = RunSpec(
        goal="g",
        artifacts={"python_script": artifact},
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
            ],
            **kw,
        ),
    )
    report = runner_mod.run_spec(spec)
    assert report["ok"] is False
    assert report["artifacts"]["python_script"]["status"] == "missing"


def test_run_spec_records_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec(
        goal="g",
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "create_transfer_mechanism",
                        "input": {"args": {"name": "input"}},
                    },
                    {
                        "type": "tool_result",
                        "id": "tu_1",
                        "name": "create_transfer_mechanism",
                        "content": "h_abc",
                        "is_error": False,
                    },
                    {"type": "text_chunk", "delta": "made it"},
                    {"type": "turn_complete", "stop_reason": "end_turn"},
                ],
            ],
            **kw,
        ),
    )

    report = runner_mod.run_spec(spec)

    assert report["tool_calls"] == [
        {"name": "create_transfer_mechanism", "input": {"args": {"name": "input"}}}
    ]
    # Tool-use + tool-result + text + turn_complete = 4
    assert report["n_events"] == 4


def test_run_spec_invokes_on_event_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec(
        goal="g",
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [
                    {"type": "text_chunk", "delta": "hi"},
                    {"type": "turn_complete", "stop_reason": "end_turn"},
                ]
            ],
            **kw,
        ),
    )

    seen: list[dict[str, Any]] = []
    runner_mod.run_spec(spec, on_event=seen.append)
    assert [e["type"] for e in seen] == ["text_chunk", "turn_complete"]


def test_run_spec_creates_report_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "deep" / "nested" / "report.json"
    spec = RunSpec(
        goal="g",
        report_path=nested,
        spec_path=tmp_path / "s.yaml",
    )
    _install_fake_session(
        monkeypatch,
        lambda **kw: _FakeSession(
            events_per_turn=[
                [{"type": "turn_complete", "stop_reason": "end_turn"}],
            ],
            **kw,
        ),
    )
    runner_mod.run_spec(spec)
    assert nested.exists()


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------


def test_main_run_flag_is_parsed() -> None:
    ns = _build_parser().parse_args(["--run", "/tmp/spec.yaml"])
    assert ns.run == "/tmp/spec.yaml"


def test_main_run_default_is_none() -> None:
    ns = _build_parser().parse_args([])
    assert ns.run is None


def _run_main_with_argv(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> SystemExit:
    """Invoke ``main()`` with a synthesised ``sys.argv`` and return the SystemExit."""
    from psyneulink_agent import main as main_mod

    monkeypatch.setattr(main_mod.sys, "argv", ["psyneulink-agent", *argv])
    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()
    return exc_info.value


def test_main_run_and_chat_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_exc = _run_main_with_argv(
        monkeypatch, ["--run", "/tmp/spec.yaml", "--chat"]
    )
    assert exit_exc.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_run_and_chat_sdk_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_exc = _run_main_with_argv(
        monkeypatch, ["--run", "/tmp/spec.yaml", "--chat-sdk"]
    )
    assert exit_exc.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_run_and_call_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_exc = _run_main_with_argv(
        monkeypatch, ["--run", "/tmp/spec.yaml", "--call", "list_handles"]
    )
    assert exit_exc.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_run_dispatches_to_runner_and_exits_with_report_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end CLI smoke test with a fully mocked runner."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("version: 1\ngoal: hello\n", encoding="utf-8")

    sentinel_report = {"turns_sent": 1, "artifacts": {}, "ok": True}
    sentinel_spec = RunSpec(
        goal="hello",
        report_path=tmp_path / "spec.yaml.report.json",
        spec_path=spec_file,
    )

    captured: dict[str, Any] = {}

    def _fake_load(path: str) -> RunSpec:
        captured["spec_arg"] = path
        return sentinel_spec

    def _fake_run(spec: RunSpec, *, mcp_project: Path | None = None) -> dict:
        captured["ran"] = True
        captured["mcp_project"] = mcp_project
        return sentinel_report

    monkeypatch.setattr(runner_mod, "load_spec", _fake_load)
    monkeypatch.setattr(runner_mod, "run_spec", _fake_run)

    exit_exc = _run_main_with_argv(monkeypatch, ["--run", str(spec_file)])
    assert exit_exc.code == 0
    assert captured["spec_arg"] == str(spec_file)
    assert captured["ran"] is True
    assert "run complete" in capsys.readouterr().err


def test_main_run_exits_nonzero_when_artifact_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("version: 1\ngoal: hi\n", encoding="utf-8")

    monkeypatch.setattr(
        runner_mod,
        "load_spec",
        lambda path: RunSpec(
            goal="hi",
            artifacts={"python_script": tmp_path / "absent.py"},
            report_path=tmp_path / "r.json",
            spec_path=spec_file,
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "run_spec",
        lambda spec, **kw: {
            "turns_sent": 2,
            "artifacts": {
                "python_script": {"path": str(tmp_path / "absent.py"), "status": "missing"}
            },
            "ok": False,
        },
    )

    exit_exc = _run_main_with_argv(monkeypatch, ["--run", str(spec_file)])
    assert exit_exc.code == 1
    err = capsys.readouterr().err
    assert "missing" in err


# Sanity: calling the module-level helper drains an async generator without
# reaching live Anthropic / MCP code. This guards against accidental
# regressions where someone "simplifies" the runner into using
# ``anthropic`` directly.
def test_drive_session_yields_via_session_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = RunSpec(
        goal="g",
        report_path=tmp_path / "r.json",
        spec_path=tmp_path / "s.yaml",
    )
    sess = _FakeSession(
        events_per_turn=[
            [{"type": "turn_complete", "stop_reason": "end_turn"}],
        ]
    )
    report = asyncio.run(runner_mod._drive_session(sess, spec))
    assert report["turns_sent"] == 1
    assert sess.messages == ["g"]
