"""Unit tests for ``core.mcp_bridge.sse_mcp_session``.

We mock both ``sse_client`` and ``ClientSession`` — the only things
this helper does are wire the two together, call ``initialize()``, and
yield the session. Transport plumbing is the SDK's job.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from psyneulink_agent.core import mcp_bridge


def test_sse_mcp_session_uses_sse_client_and_initialises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sse_mcp_session(url)`` opens ``sse_client(url)``, wraps it in a
    ``ClientSession``, and calls ``initialize()`` before yielding."""
    url = "http://127.0.0.1:9999/sse"
    sse_calls: list[str] = []
    fake_read = object()
    fake_write = object()

    @asynccontextmanager
    async def _fake_sse_client(target: str) -> AsyncIterator[tuple[Any, Any]]:
        sse_calls.append(target)
        yield (fake_read, fake_write)

    fake_session = MagicMock(name="FakeClientSession")
    fake_session.initialize = AsyncMock()

    class _FakeClientSession:
        def __init__(self, read: Any, write: Any) -> None:
            self.read = read
            self.write = write

        async def __aenter__(self) -> Any:
            fake_session.read = self.read
            fake_session.write = self.write
            return fake_session

        async def __aexit__(self, *exc_info: Any) -> None:
            return None

    monkeypatch.setattr(mcp_bridge, "sse_client", _fake_sse_client)
    monkeypatch.setattr(mcp_bridge, "ClientSession", _FakeClientSession)

    seen: dict[str, Any] = {}

    async def _go() -> None:
        async with mcp_bridge.sse_mcp_session(url) as session:
            seen["session"] = session

    asyncio.run(_go())

    assert sse_calls == [url]
    assert seen["session"] is fake_session
    assert fake_session.read is fake_read
    assert fake_session.write is fake_write
    fake_session.initialize.assert_awaited_once()
