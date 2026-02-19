from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for A1 output guard functions.

Tests the PreToolUse updatedInput generation for Bash, Read, Grep, and Glob
tools to prevent context bloat in Agent SDK sessions.
"""

import shutil
from pathlib import Path

import pytest

from core.execution.agent_sdk import (
    _BASH_HEAD_BYTES,
    _BASH_TAIL_BYTES,
    _BASH_TRUNCATE_BYTES,
    _GLOB_DEFAULT_HEAD_LIMIT,
    _GREP_DEFAULT_HEAD_LIMIT,
    _READ_DEFAULT_LIMIT,
    _build_output_guard,
    _cleanup_tool_outputs,
    _guard_bash,
    _guard_glob,
    _guard_grep,
    _guard_read,
)


# ── _guard_bash tests ────────────────────────────────────────


class TestGuardBash:
    """Tests for Bash command wrapping."""

    def test_wraps_command(self, tmp_path: Path) -> None:
        """Non-empty command is wrapped with output capture."""
        result = _guard_bash({"command": "ls -la"}, tmp_path)
        assert result is not None
        assert "command" in result
        assert "ls -la" in result["command"]
        assert "mkdir -p" in result["command"]
        assert "_OUTF=" in result["command"]
        assert "exit $_EC" in result["command"]

    def test_preserves_other_keys(self, tmp_path: Path) -> None:
        """Other tool_input keys are preserved."""
        result = _guard_bash(
            {"command": "echo hi", "timeout": 30000, "description": "test"},
            tmp_path,
        )
        assert result["timeout"] == 30000
        assert result["description"] == "test"

    def test_empty_command_passthrough(self, tmp_path: Path) -> None:
        """Empty command returns tool_input unchanged."""
        inp = {"command": ""}
        result = _guard_bash(inp, tmp_path)
        assert result is inp

    def test_no_command_key(self, tmp_path: Path) -> None:
        """Missing command key returns tool_input unchanged."""
        inp = {"timeout": 5000}
        result = _guard_bash(inp, tmp_path)
        assert result is inp

    def test_output_dir_in_wrapped(self, tmp_path: Path) -> None:
        """Wrapped command uses anima_dir/shortterm/tool_outputs."""
        result = _guard_bash({"command": "echo test"}, tmp_path)
        expected_dir = str(tmp_path / "shortterm" / "tool_outputs")
        assert expected_dir in result["command"]

    def test_truncation_thresholds(self, tmp_path: Path) -> None:
        """Wrapped command uses correct truncation thresholds."""
        result = _guard_bash({"command": "cat bigfile"}, tmp_path)
        assert str(_BASH_TRUNCATE_BYTES) in result["command"]
        assert str(_BASH_HEAD_BYTES) in result["command"]
        assert str(_BASH_TAIL_BYTES) in result["command"]

    def test_compound_command(self, tmp_path: Path) -> None:
        """Compound commands (&&) are wrapped correctly."""
        result = _guard_bash(
            {"command": "echo a && echo b"},
            tmp_path,
        )
        assert "echo a && echo b" in result["command"]
        # Wrapped in { ... ; }
        assert "{ echo a && echo b ; }" in result["command"]

    def test_pipe_command(self, tmp_path: Path) -> None:
        """Piped commands are wrapped correctly."""
        result = _guard_bash(
            {"command": "cat file | grep pattern"},
            tmp_path,
        )
        assert "cat file | grep pattern" in result["command"]


# ── _guard_read tests ────────────────────────────────────────


class TestGuardRead:
    """Tests for Read limit injection."""

    def test_no_limit_injects_default(self) -> None:
        """When limit is not specified, default is injected."""
        result = _guard_read({"file_path": "/some/file"})
        assert result is not None
        assert result["limit"] == _READ_DEFAULT_LIMIT
        assert result["file_path"] == "/some/file"

    def test_explicit_limit_passthrough(self) -> None:
        """When limit is explicitly set, no modification."""
        result = _guard_read({"file_path": "/some/file", "limit": 1000})
        assert result is None

    def test_explicit_zero_limit_passthrough(self) -> None:
        """When limit is explicitly 0, no modification (agent's choice)."""
        # 0 is not None, so it's treated as explicit
        result = _guard_read({"file_path": "/some/file", "limit": 0})
        assert result is None

    def test_none_limit_injects_default(self) -> None:
        """When limit is explicitly None, default is injected."""
        result = _guard_read({"file_path": "/some/file", "limit": None})
        assert result is not None
        assert result["limit"] == _READ_DEFAULT_LIMIT

    def test_preserves_other_keys(self) -> None:
        """Other keys like offset are preserved."""
        result = _guard_read({"file_path": "/f", "offset": 100})
        assert result is not None
        assert result["offset"] == 100
        assert result["limit"] == _READ_DEFAULT_LIMIT


# ── _guard_grep tests ────────────────────────────────────────


class TestGuardGrep:
    """Tests for Grep head_limit injection."""

    def test_no_head_limit_injects_default(self) -> None:
        """When head_limit is not specified, default is injected."""
        result = _guard_grep({"pattern": "foo"})
        assert result is not None
        assert result["head_limit"] == _GREP_DEFAULT_HEAD_LIMIT

    def test_explicit_head_limit_passthrough(self) -> None:
        """When head_limit is explicitly set, no modification."""
        result = _guard_grep({"pattern": "foo", "head_limit": 500})
        assert result is None

    def test_none_head_limit_injects_default(self) -> None:
        """When head_limit is explicitly None, default is injected."""
        result = _guard_grep({"pattern": "foo", "head_limit": None})
        assert result is not None
        assert result["head_limit"] == _GREP_DEFAULT_HEAD_LIMIT

    def test_preserves_pattern(self) -> None:
        """Pattern and other keys are preserved."""
        result = _guard_grep({"pattern": "bar", "path": "/src"})
        assert result is not None
        assert result["pattern"] == "bar"
        assert result["path"] == "/src"


# ── _guard_glob tests ────────────────────────────────────────


class TestGuardGlob:
    """Tests for Glob head_limit injection."""

    def test_no_head_limit_injects_default(self) -> None:
        """When head_limit is not specified, default is injected."""
        result = _guard_glob({"pattern": "**/*.py"})
        assert result is not None
        assert result["head_limit"] == _GLOB_DEFAULT_HEAD_LIMIT

    def test_explicit_head_limit_passthrough(self) -> None:
        """When head_limit is explicitly set, no modification."""
        result = _guard_glob({"pattern": "*.py", "head_limit": 100})
        assert result is None

    def test_none_head_limit_injects_default(self) -> None:
        """When head_limit is explicitly None, default is injected."""
        result = _guard_glob({"pattern": "*.py", "head_limit": None})
        assert result is not None
        assert result["head_limit"] == _GLOB_DEFAULT_HEAD_LIMIT


# ── _build_output_guard tests ────────────────────────────────


class TestBuildOutputGuard:
    """Tests for the dispatch function."""

    def test_bash_dispatches(self, tmp_path: Path) -> None:
        """Bash tool dispatches to _guard_bash."""
        result = _build_output_guard("Bash", {"command": "ls"}, tmp_path)
        assert result is not None
        assert "mkdir -p" in result["command"]

    def test_read_dispatches(self, tmp_path: Path) -> None:
        """Read tool dispatches to _guard_read."""
        result = _build_output_guard(
            "Read", {"file_path": "/f"}, tmp_path,
        )
        assert result is not None
        assert result["limit"] == _READ_DEFAULT_LIMIT

    def test_grep_dispatches(self, tmp_path: Path) -> None:
        """Grep tool dispatches to _guard_grep."""
        result = _build_output_guard(
            "Grep", {"pattern": "x"}, tmp_path,
        )
        assert result is not None
        assert result["head_limit"] == _GREP_DEFAULT_HEAD_LIMIT

    def test_glob_dispatches(self, tmp_path: Path) -> None:
        """Glob tool dispatches to _guard_glob."""
        result = _build_output_guard(
            "Glob", {"pattern": "*.py"}, tmp_path,
        )
        assert result is not None
        assert result["head_limit"] == _GLOB_DEFAULT_HEAD_LIMIT

    def test_write_returns_none(self, tmp_path: Path) -> None:
        """Write tool returns None (not guarded)."""
        result = _build_output_guard(
            "Write", {"file_path": "/f", "content": "x"}, tmp_path,
        )
        assert result is None

    def test_edit_returns_none(self, tmp_path: Path) -> None:
        """Edit tool returns None (not guarded)."""
        result = _build_output_guard(
            "Edit", {"file_path": "/f"}, tmp_path,
        )
        assert result is None

    def test_unknown_tool_returns_none(self, tmp_path: Path) -> None:
        """Unknown tool returns None."""
        result = _build_output_guard(
            "SomeOtherTool", {}, tmp_path,
        )
        assert result is None

    def test_read_with_explicit_limit_returns_none(self, tmp_path: Path) -> None:
        """Read with explicit limit returns None (no modification needed)."""
        result = _build_output_guard(
            "Read", {"file_path": "/f", "limit": 100}, tmp_path,
        )
        assert result is None

    def test_grep_with_explicit_head_limit_returns_none(self, tmp_path: Path) -> None:
        """Grep with explicit head_limit returns None."""
        result = _build_output_guard(
            "Grep", {"pattern": "x", "head_limit": 50}, tmp_path,
        )
        assert result is None


# ── _cleanup_tool_outputs tests ──────────────────────────────


class TestCleanupToolOutputs:
    """Tests for session cleanup."""

    def test_removes_existing_directory(self, tmp_path: Path) -> None:
        """Existing tool_outputs directory is removed."""
        out_dir = tmp_path / "shortterm" / "tool_outputs"
        out_dir.mkdir(parents=True)
        (out_dir / "bash_12345.txt").write_text("some output")

        _cleanup_tool_outputs(tmp_path)

        assert not out_dir.exists()

    def test_noop_when_no_directory(self, tmp_path: Path) -> None:
        """No error when tool_outputs directory doesn't exist."""
        _cleanup_tool_outputs(tmp_path)
        # Should not raise

    def test_noop_when_shortterm_missing(self, tmp_path: Path) -> None:
        """No error when even shortterm directory doesn't exist."""
        _cleanup_tool_outputs(tmp_path / "nonexistent")
        # Should not raise
