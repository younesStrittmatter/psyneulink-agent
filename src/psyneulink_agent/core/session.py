"""``Session`` — opaque modeling-conversation handle for any front-end.

A ``Session`` owns the state that has to persist across user turns:

* Conversation ``history`` (mutated in place by the active backend's
  ``run_turn``).
* Attached ``Resource`` instances (PDFs, data files, model files).
* Model name to dispatch against.
* The MCP project path to spawn psyneulink-mcp from.
* A pluggable ``llm_backend`` strategy — either
  :class:`AnthropicSdkBackend` (calls Anthropic's Messages API
  directly; needs ``ANTHROPIC_API_KEY``) or :class:`ClaudeCliBackend`
  (spawns the ``claude`` CLI per turn; uses the user's Claude Max
  subscription instead of an API key).

Front-ends (the ``--chat-sdk`` REPL, the upcoming web UI, the future
``--run`` headless mode) all consume the same public API:

* ``attach`` / ``detach`` / ``resources``
* ``send_user_message(text)`` — async iterator yielding events
* ``cancel_current_turn()`` — interrupt the in-flight turn (if any).
  Both backends honour it: the SDK loop bails between LLM round-trips
  and yields ``{"type": "turn_cancelled"}``; the CLI backend kills its
  ``claude`` subprocess (SIGTERM, then SIGKILL after a short grace)
  so its streaming stdout loop unblocks. Returns ``True`` if a turn
  was interrupted, ``False`` if no turn was in flight.
* ``lifespan()`` — async context manager that holds one MCP session
  open for the duration of the front-end (web UI, long REPL); inside
  it, ``send_user_message`` and ``call_tool`` reuse the same MCP
  connection so handles + journal state survive across calls
* ``call_tool(name, args)`` — invoke an MCP tool directly without
  going through the LLM; mainly for front-ends that need to poll the
  MCP for side data (e.g. the web UI rendering a graph between turns)
* ``snapshot()`` — JSON-serialisable summary for autosave / debugging

Lifespan transport (sdk vs cli)
-------------------------------

The MCP transport :meth:`lifespan` opens depends on the backend's
``kind``:

* ``"sdk"`` (Anthropic SDK path): stdio MCP — today's behaviour.
  ``mcp_session`` spawns ``psyneulink-mcp`` as a child stdio process.
* ``"cli"`` (claude CLI path): SSE MCP. ``lifespan`` first launches
  ``psyneulink-mcp --transport sse --port <free>`` as a subprocess,
  waits for the ``serving sse on …`` readiness line, then connects via
  ``sse_mcp_session(url)``. The same URL is bound onto the backend
  (``backend.mcp_url``) so each per-turn ``claude`` subprocess attaches
  to the same long-lived MCP via its ``--mcp-config``. This is what
  keeps handles + journal state coherent across (a) multiple chat
  turns and (b) out-of-loop ``call_tool`` invocations from the UI.

Default backend selection
-------------------------

The default factory :func:`_default_backend` picks:

1. ``PSYNEULINK_LLM_BACKEND={sdk,cli}`` (explicit override) — wins.
2. ``ANTHROPIC_API_KEY`` set → ``AnthropicSdkBackend``.
3. ``claude`` on ``$PATH`` → ``ClaudeCliBackend``.
4. Last resort: ``AnthropicSdkBackend`` (will fail loudly on first
   turn if no key is set, which is a clear error message).

Front-ends can override by passing ``llm_backend=`` to ``Session(...)``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import resolve_server_command
from .backends import AnthropicSdkBackend, ClaudeCliBackend, LLMBackend
from .loop import run_turn
from .mcp_bridge import call_mcp_tool, list_anthropic_tools, mcp_session, sse_mcp_session
from .resources import Resource
from .system_prompt import render_system_prompt

DEFAULT_MODEL = os.environ.get("PSYNEULINK_AGENT_MODEL", "claude-sonnet-4-5-20250929")

# Sentinel for "no per-turn model override was applied" so we can tell that
# apart from "the override happened to be None" in the swap-and-restore path.
_UNSET: Any = object()

# Stderr substring psyneulink-mcp prints once it's listening on SSE.
# Substring match (rather than full-line) keeps us forward-compatible
# with version-stamp prefixes the MCP may add later.
_SSE_READY_SUBSTRING = "serving sse on"


def _default_backend() -> LLMBackend:
    """Pick a sensible default backend for ``Session()``.

    Order:

    1. ``PSYNEULINK_LLM_BACKEND={sdk,cli}`` — explicit override wins.
    2. ``ANTHROPIC_API_KEY`` set → SDK.
    3. ``claude`` on ``$PATH`` → CLI.
    4. SDK (fails loudly at first turn if no key).
    """
    explicit = os.environ.get("PSYNEULINK_LLM_BACKEND", "").strip().lower()
    if explicit == "sdk":
        return AnthropicSdkBackend()
    if explicit == "cli":
        # The real URL is set by ``lifespan()`` once the SSE server is
        # listening; placeholder is fine here.
        return ClaudeCliBackend(mcp_url="http://127.0.0.1:0/sse")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicSdkBackend()
    if shutil.which("claude"):
        return ClaudeCliBackend(mcp_url="http://127.0.0.1:0/sse")
    return AnthropicSdkBackend()


def _pick_free_port() -> int:
    """Bind a TCP socket to port 0 to find an unused port, then close it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


async def _wait_for_port_open(
    host: str,
    port: int,
    *,
    timeout: float = 10.0,
    interval: float = 0.05,
) -> None:
    """Poll until ``host:port`` accepts a TCP connection (or timeout).

    psyneulink-mcp prints its readiness line *before* uvicorn calls
    ``socket.listen()`` — so the stderr signal alone races the actual
    bind. We close the gap by also dialling the port until it answers.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_exc: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.5
            )
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        except (OSError, asyncio.TimeoutError) as exc:
            last_exc = exc
            await asyncio.sleep(interval)
    raise RuntimeError(
        f"psyneulink-mcp printed its readiness line but {host}:{port} "
        f"did not start accepting TCP connections within {timeout}s "
        f"(last error: {last_exc!r})"
    )


async def _spawn_mcp_sse(
    cmd: list[str],
    port: int,
    *,
    timeout: float = 30.0,
) -> asyncio.subprocess.Process:
    """Spawn psyneulink-mcp in SSE mode and wait for its readiness line.

    Wires ``PSYNEULINK_MCP_TRANSPORT/HOST/PORT`` via env so the agent
    doesn't have to know whether ``cmd`` already had positional args.
    The MCP prints ``psyneulink-mcp: serving sse on
    http://127.0.0.1:<port>/sse`` to stderr once uvicorn boots, but
    that line fires *before* the listening socket is actually bound,
    so after seeing it we also poll the port until it's accepting
    connections.

    Raises :class:`RuntimeError` if the readiness line doesn't appear
    within ``timeout`` seconds, if the process exits early, or if the
    port doesn't open within an additional 10s after the line.
    """
    env = {
        **os.environ,
        "PSYNEULINK_MCP_TRANSPORT": "sse",
        "PSYNEULINK_MCP_HOST": "127.0.0.1",
        "PSYNEULINK_MCP_PORT": str(port),
    }
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stderr is not None
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            raise RuntimeError(
                "psyneulink-mcp --transport sse failed to become ready in time"
            )
        try:
            line = await asyncio.wait_for(proc.stderr.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            continue
        if not line:
            rc = proc.returncode
            tail = b""
            with contextlib.suppress(asyncio.TimeoutError):
                tail = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
            raise RuntimeError(
                f"psyneulink-mcp exited rc={rc} before becoming ready. "
                f"stderr: {tail.decode('utf-8', errors='replace')}"
            )
        text = line.decode("utf-8", errors="replace")
        if _SSE_READY_SUBSTRING in text and str(port) in text:
            try:
                await _wait_for_port_open("127.0.0.1", port)
            except RuntimeError:
                proc.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                raise
            return proc


@dataclass
class Session:
    """One modeling conversation. One front-end instance owns one of these."""

    mcp_project: Path | None = None
    model: str = DEFAULT_MODEL
    resources: list[Resource] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    llm_backend: LLMBackend = field(default_factory=_default_backend)

    # Active long-lived MCP ``ClientSession`` while inside ``lifespan()``.
    # ``None`` outside, in which case ``send_user_message`` and
    # ``call_tool`` fall back to per-call ``mcp_session`` spawns. Private
    # on purpose — front-ends interact with it via ``lifespan()`` /
    # ``call_tool()`` and never touch this attribute directly.
    _mcp: Any | None = field(default=None, init=False, repr=False)
    # When the active backend is the CLI backend, ``lifespan`` also owns
    # the SSE psyneulink-mcp subprocess and the URL it's listening on.
    _mcp_subprocess: Any | None = field(default=None, init=False, repr=False)
    _mcp_url: str | None = field(default=None, init=False, repr=False)
    # Per-turn cancellation flag. Created at the top of every
    # ``send_user_message`` call and cleared in its ``finally`` so
    # ``cancel_current_turn()`` is a no-op when nothing is running.
    # Front-ends never touch this directly — they call
    # ``cancel_current_turn()``.
    _cancel_event: asyncio.Event | None = field(default=None, init=False, repr=False)

    def attach(self, resource: Resource) -> None:
        self.resources.append(resource)

    def detach(self, resource: Resource) -> None:
        self.resources.remove(resource)

    def system_prompt(self) -> str:
        return render_system_prompt(self.resources)

    def cancel_current_turn(self) -> bool:
        """Signal cancellation of the in-flight ``send_user_message`` turn.

        Sets the per-turn cancellation event that backends watch. The
        SDK backend exits cleanly between LLM round-trips and yields a
        single ``{"type": "turn_cancelled"}`` event; the CLI backend
        kills its ``claude`` subprocess (SIGTERM, then SIGKILL after a
        short grace) so its streaming stdout loop unblocks and the
        same ``turn_cancelled`` event fires before ``run_turn``
        returns.

        Safe to call from any thread/task — the underlying
        :class:`asyncio.Event` is thread-safe for ``set()``. Idempotent
        within a single turn.

        Returns:
            ``True`` if a turn was in flight and the cancel signal was
            delivered; ``False`` if no turn was active (the front-end
            should treat this as "nothing to cancel" rather than an
            error). Front-ends typically wire this to a Stop button —
            see ``psyneulink-ui``'s ``POST /api/sessions/{sid}/cancel``.
        """
        ev = self._cancel_event
        if ev is None:
            return False
        ev.set()
        return True

    def reset_history(self, *, clear_resources: bool = False) -> None:
        """Flush the conversation history to reclaim context window space.

        For the CLI backend this also generates a fresh ``session_id``
        so the next ``send_user_message`` starts a brand-new ``claude``
        on-disk session instead of resuming the old one (which would
        still carry the full history). For the SDK backend, clearing
        the in-memory list is sufficient.

        After this call ``send_user_message`` behaves as if it were the
        first turn: any still-attached resources will be included in
        the first message again. Pass ``clear_resources=True`` to also
        detach all resources (e.g. a large PDF that caused the context
        overflow in the first place).
        """
        self.history.clear()
        if clear_resources:
            self.resources.clear()
        backend = self.llm_backend
        if backend.kind == "cli":
            import uuid as _uuid
            backend.session_id = str(_uuid.uuid4())  # type: ignore[attr-defined]
            backend._session_created = False  # type: ignore[attr-defined]

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[Session]:
        """Open a long-lived MCP connection for the duration of this session.

        Use this when the front-end needs to call MCP tools outside the
        LLM loop (e.g. the web UI polling ``render_composition_graph``
        between turns) AND share state — handles registered by tool
        calls in turn N must still resolve in turn N+1, in a direct
        :meth:`call_tool` invocation, or in a follow-up
        :meth:`send_user_message`.

        While inside the context manager :meth:`send_user_message` and
        :meth:`call_tool` reuse this same MCP session instead of
        spawning a fresh one per call. On exit, the connection is
        closed and the session reverts to per-call MCP spawns (the
        existing behaviour used by ``--chat-sdk`` and ``--run``).

        Transport depends on ``self.llm_backend.kind``:

        * ``"sdk"`` → stdio MCP (today's behaviour).
        * ``"cli"`` → SSE MCP. We spawn ``psyneulink-mcp --transport
          sse --port <free>`` as a subprocess, wait for its readiness
          line on stderr, expose the URL on the backend (so each
          per-turn ``claude`` subprocess attaches to it via
          ``--mcp-config``), then open ``sse_mcp_session(url)`` for
          the agent's own MCP usage.

        Not re-entrant: nesting ``lifespan()`` calls on the same
        Session raises :class:`RuntimeError`.
        """
        if self._mcp is not None:
            raise RuntimeError(
                "Session.lifespan() is not re-entrant; only one active span at a time."
            )

        if self.llm_backend.kind == "cli":
            port = _pick_free_port()
            cmd = resolve_server_command(self.mcp_project)
            proc = await _spawn_mcp_sse(cmd, port)
            self._mcp_subprocess = proc
            self._mcp_url = f"http://127.0.0.1:{port}/sse"
            try:
                # Rebind the backend's MCP URL so each per-turn claude
                # subprocess attaches to the live SSE server we just
                # started. Default-constructed backends start with a
                # placeholder URL; this is when it becomes real.
                self.llm_backend.mcp_url = self._mcp_url  # type: ignore[attr-defined]
                async with sse_mcp_session(self._mcp_url) as mcp:
                    self._mcp = mcp
                    try:
                        yield self
                    finally:
                        self._mcp = None
            finally:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                except ProcessLookupError:
                    pass
                self._mcp_subprocess = None
                self._mcp_url = None
                cleanup = getattr(self.llm_backend, "cleanup", None)
                if callable(cleanup):
                    cleanup()
            return

        async with mcp_session(self.mcp_project) as mcp:
            self._mcp = mcp
            try:
                yield self
            finally:
                self._mcp = None

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> str:
        """Invoke an MCP tool by name and return its tool_result text.

        Works both inside and outside :meth:`lifespan`:

        * **Inside** ``lifespan()``: uses the active MCP session, so
          handles registered by previous turns / calls are still
          resolvable. This is the path the web UI takes when it calls
          ``render_composition_graph(handle)`` outside the LLM loop.
        * **Outside** ``lifespan()``: opens a per-call MCP session for
          this single tool call. Handles registered here do NOT
          persist across calls — useful for one-shot probes from
          tests or scripts, but for any UI-style polling pattern
          (graph render, revision check) you almost certainly want to
          be inside a ``lifespan()`` block.
        """
        args = args or {}
        if self._mcp is not None:
            return await call_mcp_tool(self._mcp, name, args)
        async with mcp_session(self.mcp_project) as mcp:
            return await call_mcp_tool(mcp, name, args)

    async def send_user_message(
        self,
        text: str,
        *,
        anthropic_client: Any | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send ``text`` (plus pending resource attachments on first turn)
        and yield events from the model + tool calls.

        ``anthropic_client`` is a back-compat hook for callers that
        explicitly want SDK semantics for this turn. If passed:

        * If the active backend is :class:`AnthropicSdkBackend`, the
          client is rebound onto it.
        * Otherwise, a one-off :class:`AnthropicSdkBackend` is used
          for this call only — the session's ``llm_backend`` is not
          mutated. This keeps tests that pre-date the backend split
          working without modification (they pass a fake client and
          expect SDK behaviour).

        ``model`` is a per-turn override. When set, the active backend's
        ``model`` attribute is temporarily swapped for the duration of
        this call so the orchestrator can forward its own model choice
        (e.g. opus 4.7 with 1M context) to the child without permanently
        mutating the child's session.

        If a :meth:`lifespan` is currently active, the long-lived MCP
        session is reused so MCP-side state (handle registry, journal,
        composition revisions) survives across turns and across direct
        :meth:`call_tool` invocations. Outside ``lifespan``, the SDK
        backend falls back to a per-turn ``mcp_session`` spawn (legacy
        ``--chat-sdk`` / ``--run`` behaviour); the CLI backend
        requires ``lifespan`` (its SSE MCP is started there) and
        raises :class:`RuntimeError` otherwise.
        """
        backend: LLMBackend = self.llm_backend
        if anthropic_client is not None:
            if isinstance(backend, AnthropicSdkBackend):
                backend._client = anthropic_client
            else:
                backend = AnthropicSdkBackend(
                    model=self.model, anthropic_client=anthropic_client
                )

        # Per-turn model override: swap backend.model for one call.
        saved_model: Any = _UNSET
        if model is not None:
            current = getattr(backend, "model", None)
            if current != model:
                saved_model = current
                backend.model = model  # type: ignore[attr-defined]

        is_first_turn = len(self.history) == 0
        content_blocks: list[dict[str, Any]] = []
        if is_first_turn:
            for res in self.resources:
                content_blocks.extend(res.as_anthropic_blocks())
        content_blocks.append({"type": "text", "text": text})

        cancel_event = asyncio.Event()
        self._cancel_event = cancel_event
        try:
            if self._mcp is not None:
                tools = await list_anthropic_tools(self._mcp)
                async for event in backend.run_turn(
                    history=self.history,
                    system_prompt=self.system_prompt(),
                    user_content=content_blocks,
                    mcp=self._mcp,
                    tools=tools,
                    cancel_event=cancel_event,
                ):
                    yield event
                return

            if backend.kind == "cli":
                raise RuntimeError(
                    "ClaudeCliBackend requires an active Session.lifespan() — "
                    "the SSE MCP server is started there. Wrap your front-end "
                    "in `async with session.lifespan(): ...` before sending "
                    "user messages."
                )

            async with mcp_session(self.mcp_project) as mcp:
                tools = await list_anthropic_tools(mcp)
                async for event in backend.run_turn(
                    history=self.history,
                    system_prompt=self.system_prompt(),
                    user_content=content_blocks,
                    mcp=mcp,
                    tools=tools,
                    cancel_event=cancel_event,
                ):
                    yield event
        finally:
            self._cancel_event = None
            if saved_model is not _UNSET:
                backend.model = saved_model  # type: ignore[attr-defined]

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable summary of the session.

        Useful for autosave (future) and for surfacing the session
        state to the future UI's resource dock without exposing the
        live ``Resource`` objects.
        """
        return {
            "model": self.model,
            "history": list(self.history),
            "resources": [
                {"kind": r.kind(), "label": r.label()} for r in self.resources
            ],
        }


# ``run_turn`` is re-exported here for tests that monkeypatch it on
# this module (``psyneulink_agent.core.session.run_turn``). Keep the
# import alive even though ``Session`` itself goes through the
# backends now.
__all__ = ["Session", "run_turn", "DEFAULT_MODEL"]
