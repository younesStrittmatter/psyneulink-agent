"""Headless / batch runner for psyneulink-agent.

Reads a YAML spec describing a modeling goal + resources + desired
artifacts, drives the agent core to completion, and writes a run
report. Designed for cron jobs, CI, parameter sweeps, and any
non-interactive use of the agent.

Spec schema (version 1)::

    # Schema version (currently only "1" supported).
    version: 1

    # The natural-language instruction for the agent. Multi-line OK.
    goal: |
      Build a 2-layer feed-forward TransferMechanism network ...

    # Optional: resources attached at session start.
    resources:
      - kind: pdf
        path: /abs/path/to/paper.pdf
      - kind: data
        path: /abs/path/to/behavioural.csv
      - kind: model
        path: /abs/path/to/prior_model.py

    # Optional: artifacts the runner WILL ENSURE EXIST after the goal is
    # pursued. If the model didn't already produce them mid-conversation,
    # the runner sends a final follow-up nudge to call the relevant tool.
    artifacts:
      python_script: /abs/path/to/output_model.py    # uses export_python_script
      # mdf: /abs/path/to/output.yaml                # would use dump_mdf_model

    # Optional: safety / quota knobs. Defaults shown.
    limits:
      max_turns: 20            # hard cap on user turns the runner sends
      max_tool_iterations: 16  # passed through to run_turn

    # Optional: model override (otherwise uses Session default).
    model: claude-sonnet-4-5-20250929

    # Optional: where to write the run report (JSON). Defaults to
    # <spec_path>.report.json next to the spec.
    report: /abs/path/to/run.report.json

All paths in the spec are absolute or relative to the spec file's
directory; the runner resolves them once at load time.

The runner is a thin orchestration layer over
:class:`psyneulink_agent.core.Session` — it does not implement any
modeling logic itself; that all lives in the MCP tools the LLM is
choosing to call. Artifact nudges are sent as plain user messages; we
trust the LLM to call the right tool. After the run we report whether
each artifact appeared on disk (the report's ``ok`` flag), but we do
NOT try to re-call MCP tools ourselves to enforce existence — that is a
v2 feature.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .core import (
    DataResource,
    ModelFileResource,
    PdfResource,
    Resource,
    Session,
)

SUPPORTED_SPEC_VERSIONS = {1}
DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_TOOL_ITERATIONS = 16


class SpecError(ValueError):
    """Raised when the YAML spec is malformed or refers to missing files."""


@dataclass
class RunSpec:
    """Resolved, validated form of a YAML run spec."""

    goal: str
    resources: list[Resource] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    model: str | None = None
    report_path: Path | None = None
    spec_path: Path | None = None


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def load_spec(path: str | Path) -> RunSpec:
    """Parse a YAML spec from disk. Resolves all paths relative to spec dir."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise SpecError(f"spec not found: {spec_path}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SpecError(f"spec must be a YAML mapping, got {type(raw).__name__}")
    return _build_spec(raw, spec_path)


def _build_spec(raw: dict, spec_path: Path) -> RunSpec:
    version = raw.get("version", 1)
    if version not in SUPPORTED_SPEC_VERSIONS:
        raise SpecError(f"unsupported spec version: {version}")

    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise SpecError("spec.goal is required and must be a non-empty string")

    spec_dir = spec_path.parent

    def _resolve_path(p: str) -> Path:
        candidate = Path(p).expanduser()
        if not candidate.is_absolute():
            candidate = (spec_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    resources: list[Resource] = []
    for i, item in enumerate(raw.get("resources") or []):
        if not isinstance(item, dict):
            raise SpecError(f"spec.resources[{i}] must be a mapping")
        kind = item.get("kind")
        path_str = item.get("path")
        if not kind or not path_str:
            raise SpecError(f"spec.resources[{i}] needs kind + path")
        resolved = _resolve_path(path_str)
        try:
            if kind == "pdf":
                resources.append(PdfResource(resolved))
            elif kind == "data":
                resources.append(DataResource(resolved))
            elif kind == "model":
                resources.append(ModelFileResource(resolved))
            else:
                raise SpecError(
                    f"spec.resources[{i}].kind must be pdf|data|model, got {kind!r}"
                )
        except (FileNotFoundError, ValueError) as exc:
            raise SpecError(f"spec.resources[{i}] failed: {exc}") from exc

    artifacts: dict[str, Path] = {}
    for key, raw_path in (raw.get("artifacts") or {}).items():
        if not isinstance(raw_path, str):
            raise SpecError(f"spec.artifacts.{key} must be a string path")
        artifacts[key] = _resolve_path(raw_path)

    limits = raw.get("limits") or {}
    max_turns = int(limits.get("max_turns", DEFAULT_MAX_TURNS))
    max_tool_iterations = int(limits.get("max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS))

    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise SpecError("spec.model must be a string")

    report = raw.get("report")
    if report is not None:
        report_path = _resolve_path(report)
    else:
        report_path = spec_path.with_suffix(spec_path.suffix + ".report.json")

    return RunSpec(
        goal=goal.strip(),
        resources=resources,
        artifacts=artifacts,
        max_turns=max_turns,
        max_tool_iterations=max_tool_iterations,
        model=model,
        report_path=report_path,
        spec_path=spec_path,
    )


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


async def _drive_session(
    session: Session,
    spec: RunSpec,
    *,
    anthropic_client: Any | None = None,
    on_event: Any | None = None,
) -> dict[str, Any]:
    """Send the goal, then any artifact-nudges, watching for completion.

    Returns a structured run report dict (also written to disk by
    :func:`run_spec`).
    """
    events: list[dict[str, Any]] = []
    turns_sent = 0

    def _record(event: dict[str, Any]) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    async def _send(text: str) -> None:
        nonlocal turns_sent
        turns_sent += 1
        async for event in session.send_user_message(text, anthropic_client=anthropic_client):
            _record(event)

    await _send(spec.goal)

    artifact_status: dict[str, str] = {}

    if "python_script" in spec.artifacts:
        target = spec.artifacts["python_script"]
        if turns_sent < spec.max_turns:
            await _send(
                f"Now save the current model as a Python file at exactly "
                f"this path: {target} . Use the export_python_script tool. "
                f"After it returns, briefly confirm the file exists."
            )
        artifact_status["python_script"] = "exists" if target.exists() else "missing"

    if "mdf" in spec.artifacts:
        target = spec.artifacts["mdf"]
        if turns_sent < spec.max_turns:
            await _send(
                f"Now also dump the model as MDF YAML at exactly this path: "
                f"{target} . Use the dump_mdf_model tool if it's available. "
                f"If the tool isn't registered, say so explicitly."
            )
        artifact_status["mdf"] = "exists" if target.exists() else "missing"

    return {
        "spec_path": str(spec.spec_path) if spec.spec_path else None,
        "goal": spec.goal,
        "model": session.model,
        "turns_sent": turns_sent,
        "n_events": len(events),
        "tool_calls": [
            {"name": e["name"], "input": e.get("input")}
            for e in events
            if e.get("type") == "tool_use"
        ],
        "artifacts": {
            k: {"path": str(v), "status": artifact_status.get(k, "not_requested")}
            for k, v in spec.artifacts.items()
        },
        "ok": all(s == "exists" for s in artifact_status.values()) if artifact_status else True,
    }


def run_spec(
    spec: RunSpec,
    *,
    mcp_project: Path | None = None,
    anthropic_client: Any | None = None,
    on_event: Any | None = None,
) -> dict[str, Any]:
    """Synchronous entry point. Returns the run report (also writes to disk)."""
    session = Session(mcp_project=mcp_project)
    if spec.model:
        session.model = spec.model
    for r in spec.resources:
        session.attach(r)

    report = asyncio.run(
        _drive_session(
            session,
            spec,
            anthropic_client=anthropic_client,
            on_event=on_event,
        )
    )

    if spec.report_path is not None:
        spec.report_path.parent.mkdir(parents=True, exist_ok=True)
        spec.report_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    return report
