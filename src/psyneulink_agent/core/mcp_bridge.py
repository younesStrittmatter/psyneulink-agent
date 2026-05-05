"""Glue between the MCP client and Anthropic's Messages tool format.

Three responsibilities, kept narrow on purpose so that ``loop.py`` can be
unit-tested with pure mocks:

* :func:`mcp_session` — async context manager that spawns the
  psyneulink-mcp server (resolved through
  :func:`psyneulink_agent.config.resolve_server_command`) and yields a
  connected ``ClientSession``.
* :func:`list_anthropic_tools` — translate ``ClientSession.list_tools()``
  output into the schema Anthropic's Messages API expects (the only
  meaningful difference is the JSON Schema field name:
  MCP uses ``inputSchema`` (camelCase), Anthropic uses ``input_schema``).
* :func:`call_mcp_tool` — invoke an MCP tool by name/args and flatten
  the resulting content blocks into a single string suitable for an
  Anthropic ``tool_result`` block.

This module is the only place that imports from ``mcp.*`` for the new
SDK path; the legacy ``client.py`` remains untouched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config import resolve_server_command


@asynccontextmanager
async def mcp_session(
    mcp_project: Path | None = None,
) -> AsyncIterator[ClientSession]:
    """Spawn psyneulink-mcp and yield a connected, initialised session."""
    cmd = resolve_server_command(mcp_project)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


async def list_anthropic_tools(session: ClientSession) -> list[dict[str, Any]]:
    """Return MCP tools translated to Anthropic Messages tool format."""
    result = await session.list_tools()
    out: list[dict[str, Any]] = []
    for tool in result.tools:
        # MCP uses `inputSchema` (camelCase JSON Schema), Anthropic
        # wants `input_schema`. Otherwise the schemas are interchangeable.
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        out.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            }
        )
    return out


async def call_mcp_tool(
    session: ClientSession,
    name: str,
    args: dict[str, Any],
) -> str:
    """Invoke an MCP tool and return one string for the LLM's tool_result block.

    MCP's ``CallToolResult.content`` is a list of content blocks; for
    text tools this is a list of one or more ``TextContent`` items. We
    concatenate the ``.text`` of every text block and ``str()`` anything
    else (images, embedded resources). Tools that return nothing get
    a literal ``"(no content)"`` so the model still sees a tool_result.
    """
    result = await session.call_tool(name, args)
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        if hasattr(item, "text"):
            parts.append(item.text)
        else:
            parts.append(str(item))
    return "\n".join(parts) if parts else "(no content)"
