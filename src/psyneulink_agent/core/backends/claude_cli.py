"""Claude CLI backend — drives one turn by spawning ``claude --print``.

This backend is the path for users on the Claude Max subscription who
don't have an Anthropic API key. Each call to :meth:`run_turn` spawns
a one-shot ``claude --print --output-format stream-json`` subprocess,
hands it our user message on stdin, and translates the stream-json
events the CLI emits back into the backend-agnostic event protocol the
agent loop expects.

MCP wiring
----------

``claude`` manages MCP itself. We hand it a tiny ``--mcp-config`` JSON
that points at the long-lived SSE MCP server :class:`Session.lifespan`
already started. The same MCP serves both ``claude``'s per-turn
subprocess and the front-end's out-of-loop ``call_tool`` invocations
(graph render, etc.) so handle / journal / revision state stays
coherent across both.

Multi-turn coherence
--------------------

``claude`` persists conversation state on disk under ``--session-id
<uuid>``. We generate one UUID per :class:`ClaudeCliBackend` instance
and pass it on every spawn; ``claude`` resumes the existing on-disk
session automatically.

Stream-JSON schema (verified against ``claude 2.1.128``)
-------------------------------------------------------

A typical run produces, in order:

* ``{"type":"system","subtype":"init","cwd":...,"session_id":...,
  "tools":[...],...}`` — initial handshake; we ignore it.
* ``{"type":"system","subtype":"status","status":"requesting",...}`` —
  status pings; ignored.
* ``{"type":"system","subtype":"api_retry",...}`` — transient retry
  notices; ignored.
* ``{"type":"stream_event","event":{"type":"content_block_delta",
  "delta":{"type":"text_delta","text":"..."}, ...}}`` — partial text
  deltas (only when ``--include-partial-messages`` is on). We
  translate to ``text_chunk`` and accumulate for ``history``.
* ``{"type":"assistant","message":{"id":...,"role":"assistant",
  "content":[{"type":"text","text":"..."} | {"type":"tool_use",
  "id":"toolu_...","name":"...","input":{...}}],...}}`` — full
  assistant message (also delivered as the final message after all
  partial deltas in stream mode). Tool-use blocks are surfaced;
  text blocks are only re-emitted as ``text_chunk`` if no partial
  deltas arrived (defensive; usually a no-op when partials are on).
* ``{"type":"user","message":{"role":"user","content":[{"type":
  "tool_result","tool_use_id":"toolu_...","content":"...",
  "is_error":bool}]}}`` — tool results coming back from MCP, surfaced
  as our ``tool_result`` event.
* ``{"type":"result","subtype":"success","is_error":bool,
  "result":"...","stop_reason":"...","duration_ms":...}`` — terminal
  marker. Our ``stop_reason`` ends up either ``"end_turn"`` (success)
  or ``"error"`` (e.g. auth failure: result.is_error=true,
  api_error_status=401).

Schema can drift between ``claude`` versions; the parser uses
``.get(...)`` everywhere and silently skips unknown shapes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .base import LLMBackend


class ClaudeCliBackend(LLMBackend):
    """Drives one chat turn by spawning ``claude --print --output-format stream-json``.

    Parameters
    ----------
    mcp_url:
        SSE MCP endpoint claude should attach to (e.g.
        ``http://127.0.0.1:54321/sse``). For the typical case where
        :class:`Session.lifespan` starts the MCP, the URL isn't known
        at construction time; pass any placeholder and let ``Session``
        rebind ``self.mcp_url`` once the SSE server is listening.
    claude_path:
        Override path to the ``claude`` binary (defaults to
        ``shutil.which("claude")`` or the literal string ``"claude"``).
    model:
        Optional model override forwarded to ``claude --model``.
    session_id:
        UUID for ``claude``'s on-disk session store. Defaults to a
        fresh UUID generated once per backend instance, so subsequent
        :meth:`run_turn` calls share conversation state.
    extra_args:
        Verbatim extra argv appended to every spawn. Useful for tests
        and one-off knobs (e.g. ``["--debug", "api"]``).
    """

    kind = "cli"

    def __init__(
        self,
        *,
        mcp_url: str,
        claude_path: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.mcp_url = mcp_url
        self.claude_path = claude_path or shutil.which("claude") or "claude"
        self.model = model
        self.session_id = session_id or str(uuid.uuid4())
        self.extra_args = list(extra_args or [])
        self._mcp_config_path: Path | None = None
        # Multi-turn coherence: claude CLI rejects ``--session-id <UUID>``
        # on a UUID it has already seen ("Session ID is already in use").
        # The contract is: ``--session-id`` *creates* a new on-disk session;
        # ``--resume <UUID>`` re-attaches to an existing one. We track
        # whether we've created the session yet and switch flags on turn
        # ≥ 2.
        self._session_created = False

    def _build_mcp_config(self) -> Path:
        """Write a one-shot ``--mcp-config`` file pointing claude at our SSE MCP."""
        if self._mcp_config_path is not None and self._mcp_config_path.exists():
            return self._mcp_config_path
        config = {
            "mcpServers": {
                "psyneulink": {
                    "type": "sse",
                    "url": self.mcp_url,
                }
            }
        }
        fd, path = tempfile.mkstemp(prefix="psyneulink-claude-mcp-", suffix=".json")
        os.close(fd)
        Path(path).write_text(json.dumps(config), encoding="utf-8")
        self._mcp_config_path = Path(path)
        return self._mcp_config_path

    def cleanup(self) -> None:
        """Remove the temp MCP config file. Idempotent."""
        if self._mcp_config_path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._mcp_config_path.unlink()
            self._mcp_config_path = None

    async def run_turn(
        self,
        *,
        history: list[dict[str, Any]],
        system_prompt: str,
        user_content: list[dict[str, Any]],
        mcp: Any,
        tools: list[dict[str, Any]],
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        config_path = self._build_mcp_config()

        if cancel_event is not None and cancel_event.is_set():
            yield {"type": "turn_cancelled"}
            return

        argv = [
            self.claude_path,
            "--print",
            # Stream-json BOTH ways: input keeps Anthropic content blocks
            # intact (PDF documents, images, etc.) instead of forcing
            # everything through a flattened text prompt; output gives us
            # the structured event stream we already parse.
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            # --verbose is required by claude when --print +
            # --output-format=stream-json (per claude --help).
            "--verbose",
            "--mcp-config",
            str(config_path),
            # Don't load any other MCP servers the user happens to have
            # configured globally (Gmail, Slack, etc.) — the agent
            # should see the psyneulink MCP and nothing else.
            "--strict-mcp-config",
            # Disable Claude Code's built-in tools (Read, Bash, Edit,
            # Write, Grep, …). Without this the agent is happy to
            # `Read` arbitrary filesystem paths it sees mentioned in
            # the conversation, which is both the wrong abstraction
            # for a modelling agent and a real footgun (it can also
            # Bash / Edit / Write). MCP tools come via --mcp-config and
            # are unaffected.
            "--tools",
            "",
            # Skip per-tool permission prompts. Safe in this config
            # because: (a) builtins are disabled above (no Bash / Edit
            # / Write to bypass for) and (b) --strict-mcp-config means
            # the only MCP tools available are from the psyneulink
            # server we explicitly point at, all of which are
            # in-process modelling primitives we trust. Without this,
            # claude returns a "permissions not granted" tool_result
            # for every MCP call and the agent dead-locks waiting for
            # an interactive approval that no one is around to give.
            "--permission-mode",
            "bypassPermissions",
            "--append-system-prompt",
            system_prompt,
        ]
        # Turn 1: create the on-disk session with --session-id.
        # Turn ≥ 2: re-attach with --resume so claude carries forward
        # its own conversation memory (and we don't have to re-send
        # `history` ourselves).
        if self._session_created:
            argv.extend(["--resume", self.session_id])
        else:
            argv.extend(["--session-id", self.session_id])
        if self.model:
            argv.extend(["--model", self.model])
        argv.extend(self.extra_args)

        # Build a stream-json user message carrying ALL content blocks
        # verbatim (PDF document blocks included). Anthropic's content
        # block schema is preserved end-to-end, so PDFs ride as native
        # document blocks instead of being flattened to text + a
        # follow-up `Read` call.
        message_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": user_content,
                },
            },
            ensure_ascii=False,
        )

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None

        proc.stdin.write(message_line.encode("utf-8") + b"\n")
        await proc.stdin.drain()
        proc.stdin.close()

        # Drain stderr concurrently to prevent pipe back-pressure
        # deadlocks on chatty error paths (auth retries, etc.).
        async def _drain(stream: Any) -> bytes:
            chunks: list[bytes] = []
            if stream is None:
                return b""
            async for line in stream:
                chunks.append(line)
            return b"".join(chunks)

        stderr_task = asyncio.create_task(_drain(proc.stderr))

        # Cancel watcher: when the front-end calls
        # ``Session.cancel_current_turn()``, ``cancel_event`` flips.
        # We translate that into a SIGTERM (then SIGKILL grace) on the
        # ``claude`` subprocess so the streaming stdout loop unblocks
        # promptly. The ``cancelled`` flag tells the post-loop code to
        # emit ``turn_cancelled`` instead of ``turn_complete``.
        cancel_state = {"cancelled": False}
        cancel_watcher_task: asyncio.Task[None] | None = None
        if cancel_event is not None:

            async def _cancel_watcher() -> None:
                await cancel_event.wait()
                cancel_state["cancelled"] = True
                try:
                    proc.terminate()
                except ProcessLookupError:
                    return
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()

            cancel_watcher_task = asyncio.create_task(_cancel_watcher())

        assistant_chunks: list[str] = []
        seen_tool_uses: dict[str, dict[str, Any]] = {}
        stop_reason: str | None = None

        try:
            async for raw_line in proc.stdout:
                text_line = raw_line.strip()
                if not text_line:
                    continue
                try:
                    event = json.loads(text_line)
                except json.JSONDecodeError:
                    continue

                translated, side_effects = _translate_claude_event(
                    event, assistant_chunks, seen_tool_uses
                )
                for ev in translated:
                    yield ev
                if side_effects.get("stop_reason"):
                    stop_reason = side_effects["stop_reason"]
        finally:
            if cancel_watcher_task is not None:
                cancel_watcher_task.cancel()
                with contextlib.suppress(BaseException):
                    await cancel_watcher_task

        rc = await proc.wait()
        stderr_bytes = await stderr_task
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # On a clean exit the on-disk session for ``self.session_id`` is
        # now persisted; subsequent turns must re-attach with --resume,
        # not --session-id (claude rejects ID reuse). On a failed first
        # turn we leave the flag false so the caller can retry the
        # initial create rather than fail with "session not found".
        if rc == 0 and stop_reason != "error":
            self._session_created = True

        if cancel_state["cancelled"]:
            # Don't conflate user cancellation with a real CLI failure,
            # even though ``proc.terminate()`` produces a non-zero rc.
            yield {"type": "turn_cancelled"}
            return

        if rc != 0 and stop_reason in (None, "end_turn"):
            yield {
                "type": "tool_result",
                "id": None,
                "content": (
                    f"claude CLI exited with code {rc}.\n"
                    f"stderr (tail):\n{stderr[-2000:]}"
                ),
                "is_error": True,
            }
            stop_reason = "error"

        full_text = "".join(assistant_chunks)
        if full_text:
            history.append({"role": "user", "content": user_content})
            history.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                }
            )

        yield {"type": "turn_complete", "stop_reason": stop_reason or "end_turn"}


def _translate_claude_event(
    event: dict[str, Any],
    assistant_chunks: list[str],
    seen_tool_uses: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translate one claude stream-json event into our backend protocol.

    Returns ``(events, side_effects)``. ``events`` is the list of
    backend-protocol events to yield; ``side_effects`` carries
    out-of-band signals (currently just ``stop_reason``). Mutates
    ``assistant_chunks`` (text accumulator for history) and
    ``seen_tool_uses`` (id → block, kept for debugging / future
    correlation).
    """
    translated: list[dict[str, Any]] = []
    side_effects: dict[str, Any] = {}

    etype = event.get("type")

    if etype == "stream_event":
        inner = event.get("event") or {}
        if inner.get("type") == "content_block_delta":
            delta = inner.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    translated.append({"type": "text_chunk", "delta": text})
                    assistant_chunks.append(text)
        return translated, side_effects

    if etype == "assistant":
        message = event.get("message") or {}
        content = message.get("content") or []
        text_in_message: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                tu_id = block.get("id", "") or ""
                tu_name = block.get("name", "") or ""
                tu_input = block.get("input") or {}
                seen_tool_uses[tu_id] = block
                translated.append(
                    {
                        "type": "tool_use",
                        "id": tu_id,
                        "name": tu_name,
                        "input": tu_input,
                    }
                )
            elif btype == "text":
                text = block.get("text", "")
                if text:
                    text_in_message.append(text)

        # Backstop: only emit a text_chunk from the full message when
        # partial deltas didn't already deliver it (e.g. callers running
        # without --include-partial-messages, or schema variants that
        # skip per-delta events). Avoids duplicating text into history.
        if text_in_message and not assistant_chunks:
            for text in text_in_message:
                translated.append({"type": "text_chunk", "delta": text})
                assistant_chunks.append(text)
        return translated, side_effects

    if etype == "user":
        message = event.get("message") or {}
        content = message.get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tu_id = block.get("tool_use_id", "") or ""
            raw_content = block.get("content", "")
            if isinstance(raw_content, list):
                parts: list[str] = []
                for c in raw_content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    else:
                        parts.append(str(c))
                content_str = "\n".join(parts)
            else:
                content_str = str(raw_content)
            is_error = bool(block.get("is_error", False))
            translated.append(
                {
                    "type": "tool_result",
                    "id": tu_id,
                    "content": content_str,
                    "is_error": is_error,
                }
            )
        return translated, side_effects

    if etype == "result":
        is_error = bool(event.get("is_error", False))
        if is_error:
            side_effects["stop_reason"] = "error"
        else:
            side_effects["stop_reason"] = event.get("stop_reason") or "end_turn"
        return translated, side_effects

    return translated, side_effects
