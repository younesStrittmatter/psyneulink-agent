"""Unit tests for ``core.mcp_bridge`` translation helpers.

We stub out the real ``ClientSession``: these tests verify only the
shape conversion, not transport plumbing (covered in
``test_integration``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from psyneulink_agent.core.mcp_bridge import call_mcp_tool, list_anthropic_tools

# ---------------------------------------------------------------------------
# list_anthropic_tools
# ---------------------------------------------------------------------------


@dataclass
class _FakeTool:
    name: str
    description: str | None
    inputSchema: dict[str, Any] | None  # noqa: N815 — mirrors MCP camelCase


@dataclass
class _FakeListResult:
    tools: list[_FakeTool]


class _FakeSession:
    def __init__(
        self,
        tools: list[_FakeTool] | None = None,
        call_content: list[Any] | None = None,
    ) -> None:
        self._tools = tools or []
        self._call_content = call_content
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(tools=list(self._tools))

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))

        @dataclass
        class _Result:
            content: list[Any]

        return _Result(content=self._call_content or [])


def test_list_anthropic_tools_converts_input_schema_field() -> None:
    fake = _FakeSession(
        tools=[
            _FakeTool(
                name="create_x",
                description="Make an X.\nDetails follow.",
                inputSchema={"type": "object", "properties": {"a": {"type": "string"}}},
            ),
            _FakeTool(name="bare_tool", description=None, inputSchema=None),
        ]
    )

    result = asyncio.run(list_anthropic_tools(fake))

    assert len(result) == 2
    assert result[0]["name"] == "create_x"
    assert result[0]["description"].startswith("Make an X.")
    assert result[0]["input_schema"] == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    # Falsy description and schema get safe defaults.
    assert result[1]["description"] == ""
    assert result[1]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# call_mcp_tool
# ---------------------------------------------------------------------------


@dataclass
class _TextItem:
    text: str


@dataclass
class _OpaqueItem:
    payload: str

    def __str__(self) -> str:
        return f"<opaque {self.payload}>"


def test_call_mcp_tool_concatenates_text_blocks() -> None:
    fake = _FakeSession(call_content=[_TextItem("first"), _TextItem("second")])
    out = asyncio.run(call_mcp_tool(fake, "create_x", {"a": 1}))
    assert out == "first\nsecond"
    assert fake.calls == [("create_x", {"a": 1})]


def test_call_mcp_tool_falls_back_to_str_for_non_text_blocks() -> None:
    fake = _FakeSession(call_content=[_OpaqueItem("blob")])
    out = asyncio.run(call_mcp_tool(fake, "x", {}))
    assert out == "<opaque blob>"


def test_call_mcp_tool_returns_placeholder_when_empty() -> None:
    fake = _FakeSession(call_content=[])
    out = asyncio.run(call_mcp_tool(fake, "x", {}))
    assert out == "(no content)"
