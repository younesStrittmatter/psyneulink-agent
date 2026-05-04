"""Unit tests for CLI argument parsing and output-mode helpers in ``main.py``.

No live MCP server is needed — we test the *logic around* the SDK.
"""

from __future__ import annotations

import sys

import pytest

from psyneulink_agent.main import _build_parser, _flatten, _parse_tool_args

# ---------------------------------------------------------------------------
# _build_parser / argparse behaviour
# ---------------------------------------------------------------------------


def test_list_tools_is_default_action() -> None:
    """When no action flag is given, ``--list-tools`` is effectively the default."""
    ns = _build_parser().parse_args([])
    assert ns.call is None
    assert ns.list_tools is False  # flag not set, but main() defaults to listing


def test_list_tools_flag_is_parsed() -> None:
    ns = _build_parser().parse_args(["--list-tools"])
    assert ns.list_tools is True


def test_call_arg_is_parsed() -> None:
    ns = _build_parser().parse_args(["--call", "my_tool"])
    assert ns.call == "my_tool"


def test_call_with_args_is_parsed() -> None:
    ns = _build_parser().parse_args(["--call", "foo", "--arg", "x=1", "--arg", 'y=hello'])
    assert ns.call == "foo"
    assert ns.arg == ["x=1", "y=hello"]


def test_json_flag_is_parsed() -> None:
    ns = _build_parser().parse_args(["--json"])
    assert ns.json is True


def test_json_is_false_by_default() -> None:
    ns = _build_parser().parse_args([])
    assert ns.json is False


def test_mcp_project_is_parsed() -> None:
    ns = _build_parser().parse_args(["--mcp-project", "/some/path"])
    assert ns.mcp_project == "/some/path"


# ---------------------------------------------------------------------------
# _parse_tool_args
# ---------------------------------------------------------------------------


def test_parse_tool_args_numeric_value() -> None:
    result = _parse_tool_args(["x=1"])
    assert result == {"x": 1}


def test_parse_tool_args_string_value() -> None:
    result = _parse_tool_args(["y=hello"])
    assert result == {"y": "hello"}


def test_parse_tool_args_mixed() -> None:
    result = _parse_tool_args(["x=1", "y=hello"])
    assert result == {"x": 1, "y": "hello"}


def test_parse_tool_args_json_bool() -> None:
    result = _parse_tool_args(["flag=true"])
    assert result == {"flag": True}


def test_parse_tool_args_missing_equals() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_tool_args(["noequals"])


def test_parse_tool_args_empty() -> None:
    assert _parse_tool_args([]) == {}


# ---------------------------------------------------------------------------
# _flatten (exception flattening for ExceptionGroup)
# ---------------------------------------------------------------------------


def test_flatten_passes_plain_exception_through() -> None:
    err = RuntimeError("boom")
    assert _flatten(err) == [err]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="ExceptionGroup is 3.11+ builtin")
def test_flatten_unwraps_nested_exception_group() -> None:
    from builtins import BaseExceptionGroup as _Grp

    inner = ValueError("inner")
    middle = _Grp("middle", [inner])
    outer = _Grp("outer", [middle, RuntimeError("other")])
    leaves = _flatten(outer)
    types = [type(e).__name__ for e in leaves]
    assert types == ["ValueError", "RuntimeError"]
