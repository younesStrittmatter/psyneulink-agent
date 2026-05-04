"""Async helpers for talking to the psyneulink-mcp server.

This is the *only* place in psyneulink-agent that touches the MCP transport.
All other modules call ``connect_and_list`` / ``connect_and_call``; none of
them import from ``mcp`` directly or mention PsyNeuLink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

from .config import resolve_server_command


async def connect_and_list(mcp_project: Path | None = None) -> list[Tool]:
    """Spawn the MCP server, initialize a session, and return its tool list."""
    cmd = resolve_server_command(mcp_project)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.list_tools()
        return result.tools


async def connect_and_call(
    tool_name: str,
    arguments: dict[str, Any],
    mcp_project: Path | None = None,
) -> Any:
    """Spawn the MCP server, initialize a session, call *tool_name*, and return the result."""
    cmd = resolve_server_command(mcp_project)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await session.call_tool(tool_name, arguments)
