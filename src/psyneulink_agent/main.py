"""Console entry point for ``psyneulink-agent``.

Connects to psyneulink-mcp over stdio, lists tools, and optionally calls one.
No LLM yet — this is the wiring proof (Phase 4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

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
        "--list-tools",
        action="store_true",
        default=False,
        help="List all available tools (default action when nothing else is given).",
    )
    parser.add_argument(
        "--call",
        metavar="TOOL",
        help="Call a tool by name.",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ns = _build_parser().parse_args()
    mcp_project = Path(ns.mcp_project) if ns.mcp_project else None

    if ns.call:
        try:
            arguments = _parse_tool_args(ns.arg)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(asyncio.run(_run_call(ns.call, arguments, mcp_project, ns.json)))
    else:
        # --list-tools or bare invocation
        sys.exit(asyncio.run(_run_list(mcp_project, ns.json)))


if __name__ == "__main__":
    main()
