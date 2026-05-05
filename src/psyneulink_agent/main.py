"""Console entry point for ``psyneulink-agent``.

Modes:

* ``--run <SPEC.yaml>`` (headless) — load a YAML run spec, drive the
  agent core to completion, write a JSON report, exit 0/1 based on
  whether the requested artifacts were produced. Designed for cron
  jobs, CI, parameter sweeps. See :mod:`psyneulink_agent.runner`.
* ``--chat`` (interactive) — spawn ``claude`` with the MCP attached and
  the modeling system prompt. The Claude Max fallback for users without
  an Anthropic API key.
* ``--chat-sdk`` (interactive) — drive the modeling loop directly via
  the Anthropic Python SDK. The new default, foundation for the web UI.
  Accepts ``--pdf``, ``--data``, ``--model`` to pre-attach resources
  before the first turn.
* ``--list-tools`` — print every MCP-exposed tool, one per line. Sanity
  check that the server is reachable.
* ``--call TOOL --arg KEY=VALUE`` — invoke one tool directly, no LLM.
  Useful for debugging tool wiring without burning a chat round.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .chat import chat as run_chat
from .client import connect_and_call, connect_and_list

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup as _ExcGroup
else:  # pragma: no cover — exercised only on 3.10
    try:
        from exceptiongroup import BaseExceptionGroup as _ExcGroup
    except ImportError:
        _ExcGroup = type("_NeverMatches", (), {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(exc: BaseException) -> list[BaseException]:
    """Flatten a (possibly nested) ``BaseExceptionGroup`` into leaf exceptions.

    ``stdio_client`` wraps subprocess + stream errors in an anyio TaskGroup,
    so a crashed server surfaces as an ``ExceptionGroup``.  Flattening lets
    the CLI print one friendly line per real cause.
    """
    if isinstance(exc, _ExcGroup):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_flatten(sub))
        return out
    return [exc]


def _parse_tool_args(raw: list[str]) -> dict[str, Any]:
    """Convert ``["x=1", "y=hello"]`` → ``{"x": 1, "y": "hello"}``.

    Values are JSON-decoded where possible, otherwise passed as strings.
    """
    result: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--arg must be KEY=VALUE, got: {item!r}")
        key, _, val_str = item.partition("=")
        try:
            result[key] = json.loads(val_str)
        except json.JSONDecodeError:
            result[key] = val_str
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psyneulink-agent",
        description="Modeling agent for psyneulink-ai — connect to and inspect the MCP server.",
    )
    parser.add_argument(
        "--run",
        metavar="SPEC",
        default=None,
        help=(
            "Run the agent headlessly against a YAML spec describing the "
            "modeling goal, resources, and desired artifacts. Writes a "
            "JSON run report and exits 0/1 based on artifact existence."
        ),
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help=(
            "Drop into an interactive Claude session with psyneulink-mcp "
            "attached. Requires the `claude` CLI on PATH (Claude Max fallback)."
        ),
    )
    parser.add_argument(
        "--chat-sdk",
        action="store_true",
        default=False,
        dest="chat_sdk",
        help=(
            "Drop into an interactive REPL driven by the Anthropic SDK. "
            "Requires $ANTHROPIC_API_KEY. Supports /load-pdf, /load-data, "
            "/load-model, /save-model, /resources, /tools, /help, /exit."
        ),
    )
    parser.add_argument(
        "--pdf",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Pre-attach a PDF resource to the SDK chat session "
            "(may be repeated). Ignored outside --chat-sdk."
        ),
    )
    parser.add_argument(
        "--data",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Pre-attach a data file resource to the SDK chat session "
            "(may be repeated). Ignored outside --chat-sdk."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Pre-attach a saved .py model file to the SDK chat session "
            "(may be repeated). Ignored outside --chat-sdk."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        default=False,
        help="List all available tools (default action when nothing else is given).",
    )
    parser.add_argument(
        "--call",
        metavar="TOOL",
        help="Call a tool by name (no LLM; for debugging tool wiring).",
    )
    parser.add_argument(
        "--arg",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Argument for --call (may be repeated). Values are JSON-decoded if parseable.",
    )
    parser.add_argument(
        "--mcp-project",
        metavar="PATH",
        help="Override the MCP project path ($PSYNEULINK_MCP_PROJECT or ../psyneulink-mcp).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON instead of pretty text.",
    )
    return parser


# ---------------------------------------------------------------------------
# Async runners
# ---------------------------------------------------------------------------


async def _run_list(mcp_project: Path | None, as_json: bool) -> int:
    try:
        tools = await connect_and_list(mcp_project)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BaseException as exc:  # noqa: BLE001 — top-level: report cleanly, exit non-zero
        for sub in _flatten(exc):
            print(f"MCP session failed: {type(sub).__name__}: {sub}", file=sys.stderr)
        return 1

    if as_json:
        data = [{"name": t.name, "description": t.description} for t in tools]
        print(json.dumps(data, indent=2))
    else:
        for tool in tools:
            first_line = (tool.description or "").splitlines()[0] if tool.description else ""
            print(f"{tool.name}\t{first_line}")
        print(f"\n{len(tools)} tool(s) available.", file=sys.stderr)
    return 0


async def _run_call(
    tool_name: str,
    arguments: dict[str, Any],
    mcp_project: Path | None,
    as_json: bool,
) -> int:
    try:
        result = await connect_and_call(tool_name, arguments, mcp_project)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BaseException as exc:  # noqa: BLE001
        for sub in _flatten(exc):
            print(f"MCP call failed: {type(sub).__name__}: {sub}", file=sys.stderr)
        return 1

    if as_json:
        payload = result.model_dump() if hasattr(result, "model_dump") else str(result)
        print(json.dumps(payload, indent=2))
    else:
        if hasattr(result, "content"):
            for item in result.content:
                print(item.text if hasattr(item, "text") else item)
        else:
            print(result)
    return 0


def _build_initial_resources(
    pdfs: list[str], datas: list[str], models: list[str]
) -> list[Any]:
    """Construct ``Resource`` instances from CLI flag values, preserving order."""
    from .core import DataResource, ModelFileResource, PdfResource

    out: list[Any] = []
    for p in pdfs:
        out.append(PdfResource(p))
    for p in datas:
        out.append(DataResource(p))
    for p in models:
        out.append(ModelFileResource(p))
    return out


async def _run_chat_sdk(
    mcp_project: Path | None,
    pdfs: list[str],
    datas: list[str],
    models: list[str],
) -> int:
    from .repl import repl

    try:
        initial_resources = _build_initial_resources(pdfs, datas, models)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return await repl(mcp_project, initial_resources) or 0


def _run_headless(spec_path: str, mcp_project: Path | None) -> int:
    """Load + run a YAML spec; print a one-line summary; return 0/1."""
    from .runner import SpecError, load_spec, run_spec

    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_spec(spec, mcp_project=mcp_project)
    except BaseException as exc:  # noqa: BLE001 — top-level: report cleanly, exit non-zero
        for sub in _flatten(exc):
            print(f"run failed: {type(sub).__name__}: {sub}", file=sys.stderr)
        return 1

    artifacts = report.get("artifacts", {})
    if artifacts:
        artifact_summary = ", ".join(
            f"{k}={v.get('status')}" for k, v in artifacts.items()
        )
    else:
        artifact_summary = "no artifacts requested"
    print(
        f"run complete: turns_sent={report.get('turns_sent')} "
        f"artifacts=[{artifact_summary}] "
        f"report={spec.report_path}",
        file=sys.stderr,
    )
    return 0 if report.get("ok") else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ns = _build_parser().parse_args()
    mcp_project = Path(ns.mcp_project) if ns.mcp_project else None

    resource_flags_used = bool(ns.pdf or ns.data or ns.model)

    # --run is exclusive with the interactive / one-shot-call modes. Resource
    # pre-attach flags only feed --chat-sdk, so warn (don't error) if combined
    # with --run; the spec's `resources:` field is the headless equivalent.
    if ns.run is not None:
        conflicting = [
            name for name, val in (
                ("--chat", ns.chat),
                ("--chat-sdk", ns.chat_sdk),
                ("--call", ns.call),
            )
            if val
        ]
        if conflicting:
            print(
                f"error: --run is mutually exclusive with {', '.join(conflicting)}",
                file=sys.stderr,
            )
            sys.exit(2)
        if resource_flags_used:
            print(
                "warning: --pdf/--data/--model are ignored with --run "
                "(declare resources in the spec's `resources:` block).",
                file=sys.stderr,
            )
        sys.exit(_run_headless(ns.run, mcp_project))

    if ns.chat_sdk:
        sys.exit(asyncio.run(_run_chat_sdk(mcp_project, ns.pdf, ns.data, ns.model)))

    if ns.chat:
        if resource_flags_used:
            print(
                "warning: --pdf/--data/--model are ignored with --chat "
                "(use --chat-sdk to attach resources).",
                file=sys.stderr,
            )
        # Auto-detect notice: surface the SDK alternative without forcing a
        # switch — the user explicitly chose --chat.
        if (
            os.environ.get("ANTHROPIC_API_KEY")
            and os.environ.get("PSYNEULINK_AGENT_USE_CLI") != "1"
        ):
            print(
                "note: $ANTHROPIC_API_KEY is set; --chat-sdk is also available.",
                file=sys.stderr,
            )
        sys.exit(run_chat(mcp_project))

    if ns.call:
        try:
            arguments = _parse_tool_args(ns.arg)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(asyncio.run(_run_call(ns.call, arguments, mcp_project, ns.json)))
    else:
        sys.exit(asyncio.run(_run_list(mcp_project, ns.json)))


if __name__ == "__main__":
    main()
