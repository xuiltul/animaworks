"""Tests for core/tools/_base.py — ToolResult, get_env_or_fail, auto_cli_guide."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.tools._base import (
    ToolConfigError,
    ToolResult,
    auto_cli_guide,
    get_env_or_fail,
)


# ── ToolConfigError ───────────────────────────────────────────────


class TestToolConfigError:
    def test_is_exception(self):
        assert issubclass(ToolConfigError, Exception)

    def test_message(self):
        exc = ToolConfigError("missing key")
        assert str(exc) == "missing key"


# ── ToolResult ────────────────────────────────────────────────────


class TestToolResult:
    def test_success_result(self):
        r = ToolResult(success=True, data={"key": "value"}, text="done")
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.text == "done"
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, error="something went wrong")
        assert r.success is False
        assert r.data is None
        assert r.text == ""
        assert r.error == "something went wrong"

    def test_defaults(self):
        r = ToolResult(success=True)
        assert r.data is None
        assert r.text == ""
        assert r.error is None


# ── get_env_or_fail ───────────────────────────────────────────────


class TestGetEnvOrFail:
    def test_returns_env_value(self):
        with patch.dict(os.environ, {"TEST_KEY_123": "test_value"}):
            result = get_env_or_fail("TEST_KEY_123", "test_tool")
            assert result == "test_value"

    def test_raises_on_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the key is not set
            os.environ.pop("MISSING_KEY_XYZ", None)
            with pytest.raises(ToolConfigError) as exc_info:
                get_env_or_fail("MISSING_KEY_XYZ", "my_tool")
            assert "my_tool" in str(exc_info.value)
            assert "MISSING_KEY_XYZ" in str(exc_info.value)

    def test_raises_on_empty_value(self):
        with patch.dict(os.environ, {"EMPTY_KEY": ""}):
            with pytest.raises(ToolConfigError):
                get_env_or_fail("EMPTY_KEY", "my_tool")


# ── auto_cli_guide ────────────────────────────────────────────────


class TestAutoCliGuide:
    def test_basic_schema(self):
        schemas = [
            {
                "name": "web_search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ]
        guide = auto_cli_guide("web_search", schemas)
        assert "### web_search" in guide
        assert "```bash" in guide
        assert "animaworks-tool web_search" in guide
        assert '"<query>"' in guide
        assert "--count" in guide
        assert "-j" in guide

    def test_boolean_flag(self):
        schemas = [
            {
                "name": "test_tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "verbose": {"type": "boolean"},
                    },
                    "required": [],
                },
            }
        ]
        guide = auto_cli_guide("test_tool", schemas)
        assert "[--verbose]" in guide

    def test_multiple_schemas(self):
        schemas = [
            {
                "name": "action_a",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
            {
                "name": "action_b",
                "input_schema": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": [],
                },
            },
        ]
        guide = auto_cli_guide("multi", schemas)
        lines = guide.strip().split("\n")
        # Should have header, opening ```, two command lines, closing ```
        assert lines[0] == "### multi"
        assert lines[1] == "```bash"
        assert lines[-1] == "```"

    def test_underscore_to_hyphen_in_flags(self):
        schemas = [
            {
                "name": "tool",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "max_results": {"type": "integer"},
                    },
                    "required": [],
                },
            }
        ]
        guide = auto_cli_guide("tool", schemas)
        assert "--max-results" in guide

    def test_parameters_key_fallback(self):
        """auto_cli_guide also checks 'parameters' key if 'input_schema' missing."""
        schemas = [
            {
                "name": "tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]
        guide = auto_cli_guide("tool", schemas)
        assert '"<query>"' in guide
