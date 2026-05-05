"""Interactive chat: spawn ``claude`` CLI with the MCP server attached.

MVP rationale: the user already has Claude Max plan and the ``claude``
CLI on ``PATH``. ``claude`` natively understands MCP via ``--mcp-config``,
which means we don't yet need to roll our own LLM tool-calling loop —
``claude`` does all of that for us, and we get streaming UI, history,
permissions UI, etc. for free.

When this MVP grows up (custom tool routing, prompt caching, modeling-
strategy logic that needs to live in *our* loop rather than Anthropic's
generic one), this module is the seam to replace: swap the subprocess
for an Anthropic SDK loop that owns the conversation while still
delegating tool execution to the same MCP. The system prompt, MCP
config, and the user-facing CLI flag (``--chat``) stay unchanged.

Sandboxed environments (Cursor's shell tool) often can't actually run
this — ``claude`` opens a TTY. Run from a real terminal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import resolve_server_command

SYSTEM_PROMPT = """\
You are PsyNeuLink Agent, a modeling assistant for cognitive and \
neural network models built with the PsyNeuLink (PNL) framework.

You have an MCP server attached named `psyneulink`. Every PsyNeuLink \
class and function the user is likely to need is exposed as an MCP \
tool. Use the tools — do not write Python or pretend to run code.

How modeling works in this MCP:

1. Each `create_*` tool constructs one PNL object (a Mechanism, a \
Function, a Composition, a Projection, etc.) and returns a HANDLE \
shaped like `{"handle": "h_abc123def456", "type": "...", "name": \
"...", "repr": "..."}`. The `handle` string is the live object's ID \
in this session — use it everywhere a previously-created object is \
expected.

2. Construct a model in this order, every time:
   a. Create the component Mechanisms (`create_transfer_mechanism`, \
`create_processing_mechanism`, `create_lca_mechanism`, …).
   b. Create a Composition (`create_composition`).
   c. Wire mechanisms inside it. For a feed-forward chain use \
`add_linear_pathway(composition=<h>, nodes=[<h_in>, <h_hidden>, \
<h_out>])`. For arbitrary topologies use `add_node` + `add_projection` \
(pass `matrix=` for a custom weight matrix; otherwise PNL chooses).
   d. Run it with `run_composition(composition=<h>, inputs={<h_input>: \
[[trial1_values], [trial2_values], ...]})`. PNL inputs are nested \
lists — the outer dimension is trials, the inner dimension is the \
node's input shape.

3. Handles are valid as values **inside** any tool's arguments too. \
For example, to use a custom transfer function:
   `create_transfer_mechanism(args={"function": "h_<linear_handle>"})`.

4. If you lose track of what exists, call `list_handles`. \
`describe_handle` gives type / name / repr for one handle.

5. Mechanism arguments live inside an `args` dict:
   `create_transfer_mechanism(args={"name": "input", "default_variable": \
[[0.0, 0.0]]})`. Look at each tool's description for the JSON Schema \
of its `args`.

After building a model, give the user a one-paragraph summary of what \
was built and (if you ran it) the run output. Be concise — they want \
models, not essays. If a tool returns an error, fix it and continue \
rather than dumping the traceback at the user.

You may also use `report_tool_issue` when an MCP tool's description, \
schema, or behavior is genuinely wrong (not for ordinary modeling \
errors). The corpus has a feedback loop that consumes those reports \
into the next regen.
"""


def _build_mcp_config(mcp_project: Path | None) -> dict[str, dict[str, dict[str, object]]]:
    """Produce a ``--mcp-config`` JSON document referencing our MCP server."""
    cmd = resolve_server_command(mcp_project)
    return {
        "mcpServers": {
            "psyneulink": {
                "command": cmd[0],
                "args": cmd[1:],
            }
        }
    }


def chat(
    mcp_project: Path | None = None,
    *,
    extra_claude_args: list[str] | None = None,
) -> int:
    """Drop the user into an interactive Claude session backed by the MCP.

    ``extra_claude_args`` is appended verbatim, so callers (or future
    CLI flags) can pass ``["--print", "build me ..."]`` for a one-shot
    smoke test, or ``["--model", "opus"]`` to override the model, etc.
    """
    if shutil.which("claude") is None:
        print(
            "error: `claude` CLI not found on PATH. Install it from "
            "https://docs.claude.com/en/docs/claude-code or set up an "
            "alternative LLM client.",
            file=sys.stderr,
        )
        return 2

    config = _build_mcp_config(mcp_project)

    # Use a tempfile rather than passing config as a literal string:
    # the JSON contains absolute paths and the command can be long
    # enough to upset some shells.
    fd, path_str = tempfile.mkstemp(prefix="psyneulink-agent-mcp-", suffix=".json")
    os.close(fd)
    cfg_path = Path(path_str)
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    argv = [
        "claude",
        "--mcp-config",
        str(cfg_path),
        "--append-system-prompt",
        SYSTEM_PROMPT,
    ]
    if extra_claude_args:
        argv.extend(extra_claude_args)

    try:
        return subprocess.run(argv).returncode
    finally:
        cfg_path.unlink(missing_ok=True)
