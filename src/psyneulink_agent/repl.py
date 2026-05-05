"""Interactive REPL for ``--chat-sdk``.

Reads stdin lines, parses slash commands (``/load-pdf``, ``/load-data``,
``/load-model``, ``/save-model``, ``/resources``, ``/tools``, ``/help``,
``/exit``), and otherwise forwards the line as a user message to a
:class:`~psyneulink_agent.core.session.Session`. Session events are
rendered to stdout / stderr as they arrive.

The REPL is the simplest front-end on top of the agent core. The
upcoming web UI consumes the same ``Session`` API; the slash-command
behaviour here is the reference for what the UI's resource dock and
buttons need to do.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable
from pathlib import Path

from .core import (
    DataResource,
    ModelFileResource,
    PdfResource,
    Resource,
    Session,
)

SLASH_HELP = """\
Slash commands:
  /load-pdf <path>       attach a PDF as reading material
  /load-data <path>      attach a data file (call load_psyche_data tool to actually load it)
  /load-model <path>     attach a .py model file (call load_python_script to re-materialise)
  /save-model [path]     ask the agent to call export_python_script
  /resources             list attached resources
  /tools                 list available MCP tools
  /help                  show this message
  /exit                  quit
"""


def _render_event(event: dict) -> None:
    et = event.get("type")
    if et == "text_chunk":
        print(event.get("delta", ""), end="", flush=True)
    elif et == "tool_use":
        print(
            f"\n[tool] {event.get('name')} {event.get('input')!r}",
            file=sys.stderr,
        )
    elif et == "tool_result":
        marker = "tool err" if event.get("is_error") else "tool ok"
        print(f"[{marker}] {event.get('name')}", file=sys.stderr)
    elif et == "turn_complete":
        # Newline so the next prompt starts cleanly.
        print()


async def _stream_user_message(session: Session, text: str) -> None:
    async for event in session.send_user_message(text):
        _render_event(event)


async def _handle_slash(line: str, session: Session) -> bool:
    """Returns ``False`` if the user asked to exit, else ``True``."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return False

    if cmd == "/help":
        print(SLASH_HELP)
        return True

    if cmd == "/load-pdf":
        if not arg:
            print("usage: /load-pdf <path>")
            return True
        try:
            session.attach(PdfResource(arg))
            print(f"attached: {arg}")
        except Exception as exc:  # noqa: BLE001 — surface the error to the user, keep the REPL alive
            print(f"failed: {exc}")
        return True

    if cmd == "/load-data":
        if not arg:
            print("usage: /load-data <path>")
            return True
        try:
            session.attach(DataResource(arg))
            print(f"attached: {arg}")
        except Exception as exc:  # noqa: BLE001
            print(f"failed: {exc}")
        return True

    if cmd == "/load-model":
        if not arg:
            print("usage: /load-model <path>")
            return True
        try:
            session.attach(ModelFileResource(arg))
            print(f"attached: {arg}")
            print(
                "Tip: ask the agent to call load_python_script on it to materialise."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"failed: {exc}")
        return True

    if cmd == "/save-model":
        # Translate to a user message that nudges the LLM to call the tool.
        nudge = "Please save the current model as a Python file"
        if arg:
            nudge += f" at {arg}"
        nudge += " using the export_python_script tool."
        await _stream_user_message(session, nudge)
        return True

    if cmd == "/resources":
        if not session.resources:
            print("(no resources attached)")
        else:
            for r in session.resources:
                print(r.summary_line())
        return True

    if cmd == "/tools":
        from .core.mcp_bridge import list_anthropic_tools, mcp_session

        async with mcp_session(session.mcp_project) as mcp:
            tools = await list_anthropic_tools(mcp)
            for t in tools:
                desc = t.get("description") or ""
                first_line = desc.splitlines()[0] if desc else ""
                print(f"  {t['name']}\t{first_line}")
        return True

    print(f"unknown command: {cmd!r} (try /help)")
    return True


async def repl(
    mcp_project: Path | None = None,
    initial_resources: Iterable[Resource] | None = None,
) -> int:
    """Run the SDK-mode REPL until the user types ``/exit`` or hits EOF."""
    session = Session(mcp_project=mcp_project)
    for r in initial_resources or ():
        session.attach(r)

    print("psyneulink-agent (SDK mode). Type /help for commands, /exit to quit.")
    if session.resources:
        print("Attached resources:")
        for r in session.resources:
            print("  " + r.summary_line())
    print()

    while True:
        try:
            line = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line.strip():
            continue
        if line.startswith("/"):
            should_continue = await _handle_slash(line, session)
            if not should_continue:
                return 0
            continue

        try:
            await _stream_user_message(session, line)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")
            continue
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive across LLM/tool failures
            print(f"\nerror: {type(exc).__name__}: {exc}", file=sys.stderr)
