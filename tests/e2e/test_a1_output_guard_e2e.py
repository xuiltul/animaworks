from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for A1 output guard (Bash wrapping and limit injection).

Tests actual bash command execution with the output guard wrapper to verify
file saving, truncation, and cleanup behavior.
"""

import os
import subprocess
from pathlib import Path

import pytest

from core.execution.agent_sdk import (
    _BASH_HEAD_BYTES,
    _BASH_TAIL_BYTES,
    _BASH_TRUNCATE_BYTES,
    _GLOB_DEFAULT_HEAD_LIMIT,
    _GREP_DEFAULT_HEAD_LIMIT,
    _READ_DEFAULT_LIMIT,
    _cleanup_tool_outputs,
    _guard_bash,
)


class TestBashGuardE2E:
    """E2E tests for Bash command wrapping with actual shell execution."""

    def test_small_output_no_file_kept(self, tmp_path: Path) -> None:
        """Output below threshold: displayed in full, temp file removed."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        wrapped = _guard_bash({"command": "echo hello"}, anima_dir)
        result = subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "hello" in result.stdout

        # Temp file should be cleaned up (output < 10KB)
        out_dir = anima_dir / "shortterm" / "tool_outputs"
        if out_dir.exists():
            files = list(out_dir.glob("bash_*.txt"))
            assert len(files) == 0, f"Temp file not cleaned up: {files}"

    def test_large_output_truncated_and_saved(self, tmp_path: Path) -> None:
        """Output above threshold: truncated display + file saved."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        # Generate 20KB of output (well above 10KB threshold)
        cmd = f"python3 -c \"print('A' * {_BASH_TRUNCATE_BYTES * 2})\""
        wrapped = _guard_bash({"command": cmd}, anima_dir)
        result = subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "truncated:" in result.stdout
        assert "Full output saved:" in result.stdout
        assert "Use Read tool" in result.stdout

        # Temp file should exist
        out_dir = anima_dir / "shortterm" / "tool_outputs"
        assert out_dir.exists()
        files = list(out_dir.glob("bash_*.txt"))
        assert len(files) == 1, f"Expected 1 temp file, got {len(files)}"

        # Verify full content is in the file
        content = files[0].read_text()
        assert len(content) > _BASH_TRUNCATE_BYTES

    def test_exit_code_preserved(self, tmp_path: Path) -> None:
        """Command exit code is preserved through the wrapper."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        wrapped = _guard_bash({"command": "exit 42"}, anima_dir)
        result = subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 42

    def test_exit_code_preserved_large_output(self, tmp_path: Path) -> None:
        """Exit code preserved even with truncated output."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        cmd = f"python3 -c \"print('X' * {_BASH_TRUNCATE_BYTES * 2}); import sys; sys.exit(7)\""
        wrapped = _guard_bash({"command": cmd}, anima_dir)
        result = subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 7

    def test_stderr_captured(self, tmp_path: Path) -> None:
        """stderr is captured in the output file (2>&1 redirect)."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        cmd = "echo stdout_msg && echo stderr_msg >&2"
        wrapped = _guard_bash({"command": cmd}, anima_dir)
        result = subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Both stdout and stderr should appear in stdout (due to 2>&1)
        assert "stdout_msg" in result.stdout
        assert "stderr_msg" in result.stdout

    def test_multiple_commands_unique_files(self, tmp_path: Path) -> None:
        """Multiple large-output commands create unique temp files."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        cmd = f"python3 -c \"print('B' * {_BASH_TRUNCATE_BYTES * 2})\""
        for _ in range(3):
            wrapped = _guard_bash({"command": cmd}, anima_dir)
            subprocess.run(
                ["bash", "-c", wrapped["command"]],
                capture_output=True,
                text=True,
                timeout=10,
            )

        out_dir = anima_dir / "shortterm" / "tool_outputs"
        files = list(out_dir.glob("bash_*.txt"))
        assert len(files) == 3, f"Expected 3 unique files, got {len(files)}"

    def test_cleanup_after_session(self, tmp_path: Path) -> None:
        """_cleanup_tool_outputs removes the tool_outputs directory."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "shortterm").mkdir()

        # Create large output to generate a temp file
        cmd = f"python3 -c \"print('C' * {_BASH_TRUNCATE_BYTES * 2})\""
        wrapped = _guard_bash({"command": cmd}, anima_dir)
        subprocess.run(
            ["bash", "-c", wrapped["command"]],
            capture_output=True,
            text=True,
            timeout=10,
        )

        out_dir = anima_dir / "shortterm" / "tool_outputs"
        assert out_dir.exists()

        # Simulate session end cleanup
        _cleanup_tool_outputs(anima_dir)
        assert not out_dir.exists()


class TestOutputGuardConstants:
    """Verify output guard constants have expected values."""

    def test_bash_truncate_bytes(self) -> None:
        assert _BASH_TRUNCATE_BYTES == 10_000

    def test_bash_head_bytes(self) -> None:
        assert _BASH_HEAD_BYTES == 5_000

    def test_bash_tail_bytes(self) -> None:
        assert _BASH_TAIL_BYTES == 3_000

    def test_read_default_limit(self) -> None:
        assert _READ_DEFAULT_LIMIT == 500

    def test_grep_default_head_limit(self) -> None:
        assert _GREP_DEFAULT_HEAD_LIMIT == 200

    def test_glob_default_head_limit(self) -> None:
        assert _GLOB_DEFAULT_HEAD_LIMIT == 500
