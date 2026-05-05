"""Shared pytest fixtures.

Most tests pre-date the LLM-backend split and assume the SDK code path
(passing a fake AsyncAnthropic, expecting stdio MCP semantics from
``Session.lifespan``). On dev machines with ``claude`` on ``$PATH`` but
no ``ANTHROPIC_API_KEY``, the auto-detect default would otherwise pick
``ClaudeCliBackend`` — which would try to spawn a real
``psyneulink-mcp --transport sse`` and break those tests.

The autouse fixture below pins ``PSYNEULINK_LLM_BACKEND=sdk`` for every
test; tests that need to exercise the auto-detect logic itself
explicitly delete the env var via ``monkeypatch.delenv``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_backend_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the default LLM backend to SDK for the duration of each test."""
    monkeypatch.setenv("PSYNEULINK_LLM_BACKEND", "sdk")
