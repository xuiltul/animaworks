"""Tests for A1 mode security functions in core.execution.agent_sdk."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.execution.agent_sdk import (
    _BASH_BLOCKED_PATTERNS,
    _PROTECTED_FILES,
    _WRITE_COMMANDS,
    _check_a1_bash_command,
    _check_a1_file_access,
    _log_tool_use,
    _sanitise_tool_args,
    _summarise_tool_input,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def animas_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas"
    d.mkdir()
    return d


@pytest.fixture
def anima_dir(animas_dir: Path) -> Path:
    d = animas_dir / "test-anima"
    d.mkdir()
    (d / "permissions.md").write_text("", encoding="utf-8")
    (d / "identity.md").write_text("", encoding="utf-8")
    (d / "bootstrap.md").write_text("", encoding="utf-8")
    (d / "knowledge").mkdir()
    return d


@pytest.fixture
def other_anima_dir(animas_dir: Path) -> Path:
    d = animas_dir / "other-anima"
    d.mkdir()
    (d / "identity.md").write_text("secret", encoding="utf-8")
    (d / "episodes").mkdir()
    return d


# ── Constants ──────────────────────────────────────────────────


class TestConstants:
    def test_protected_files_contains_expected(self):
        assert "permissions.md" in _PROTECTED_FILES
        assert "identity.md" in _PROTECTED_FILES
        assert "bootstrap.md" in _PROTECTED_FILES

    def test_protected_files_is_frozenset(self):
        assert isinstance(_PROTECTED_FILES, frozenset)

    def test_write_commands_contains_expected(self):
        assert "cp" in _WRITE_COMMANDS
        assert "mv" in _WRITE_COMMANDS
        assert "rsync" in _WRITE_COMMANDS


# ── _check_a1_file_access ────────────────────────────────────


class TestCheckA1FileAccess:
    def test_empty_path_allowed(self, anima_dir: Path):
        assert _check_a1_file_access("", anima_dir, write=True) is None

    def test_write_to_own_permissions_blocked(self, anima_dir: Path):
        path = str(anima_dir / "permissions.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_write_to_own_identity_blocked(self, anima_dir: Path):
        path = str(anima_dir / "identity.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_write_to_own_bootstrap_blocked(self, anima_dir: Path):
        path = str(anima_dir / "bootstrap.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "protected file" in result

    def test_read_own_permissions_allowed(self, anima_dir: Path):
        """Reading protected files should be allowed."""
        path = str(anima_dir / "permissions.md")
        result = _check_a1_file_access(path, anima_dir, write=False)
        assert result is None

    def test_write_to_own_knowledge_allowed(self, anima_dir: Path):
        path = str(anima_dir / "knowledge" / "note.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None

    def test_read_other_anima_blocked(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        path = str(other_anima_dir / "identity.md")
        result = _check_a1_file_access(path, anima_dir, write=False)
        assert result is not None
        assert "other anima" in result.lower()

    def test_write_other_anima_blocked(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        path = str(other_anima_dir / "episodes" / "log.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is not None
        assert "other anima" in result.lower()

    def test_path_outside_animas_root_allowed(self, anima_dir: Path):
        """Paths outside the animas/ root are not restricted by this function."""
        result = _check_a1_file_access("/tmp/some_file.txt", anima_dir, write=True)
        assert result is None

    def test_write_to_own_heartbeat_allowed(self, anima_dir: Path):
        """heartbeat.md is not protected."""
        path = str(anima_dir / "heartbeat.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None

    def test_write_to_own_cron_allowed(self, anima_dir: Path):
        """cron.md is not protected."""
        path = str(anima_dir / "cron.md")
        result = _check_a1_file_access(path, anima_dir, write=True)
        assert result is None


# ── _check_a1_bash_command ───────────────────────────────────


class TestCheckA1BashCommand:
    def test_cp_to_other_anima_blocked(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        cmd = f"cp {other_anima_dir}/identity.md ./stolen.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "other anima" in result.lower()

    def test_mv_to_other_anima_blocked(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        cmd = f"mv ./file.md {other_anima_dir}/file.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "other anima" in result.lower()

    def test_normal_ls_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("ls -la", anima_dir)
        assert result is None

    def test_normal_python_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("python3 script.py", anima_dir)
        assert result is None

    def test_cp_within_own_dir_allowed(self, anima_dir: Path):
        cmd = f"cp {anima_dir}/file1.md {anima_dir}/file2.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is None

    def test_non_write_command_to_other_anima_allowed(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        """Non-write commands like 'cat' are not checked."""
        cmd = f"cat {other_anima_dir}/identity.md"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is None

    def test_empty_command_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("", anima_dir)
        assert result is None

    def test_malformed_command_allowed(self, anima_dir: Path):
        """Malformed commands that can't be parsed should not crash."""
        result = _check_a1_bash_command("echo 'unclosed", anima_dir)
        assert result is None

    def test_rsync_to_other_anima_blocked(
        self, anima_dir: Path, other_anima_dir: Path,
    ):
        cmd = f"rsync -a ./data/ {other_anima_dir}/data/"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None

    def test_flags_are_skipped(self, anima_dir: Path):
        """Arguments starting with - should be skipped."""
        result = _check_a1_bash_command("cp -r --verbose file1 file2", anima_dir)
        assert result is None


# ── _BASH_BLOCKED_PATTERNS / blocklist ──────────────────────


class TestBashBlockedPatterns:
    def test_chatwork_send_blocked(self, anima_dir: Path):
        result = _check_a1_bash_command("chatwork send room1 hello", anima_dir)
        assert result is not None
        assert "Chatwork send" in result

    def test_chatwork_cli_send_blocked(self, anima_dir: Path):
        result = _check_a1_bash_command("chatwork_cli.py send room1 hello", anima_dir)
        assert result is not None

    def test_curl_chatwork_api_blocked(self, anima_dir: Path):
        cmd = "curl -X POST https://api.chatwork.com/v2/rooms/123/messages -d 'body=hello'"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None

    def test_chatwork_send_case_insensitive(self, anima_dir: Path):
        result = _check_a1_bash_command("CHATWORK SEND room1 msg", anima_dir)
        assert result is not None

    def test_chatwork_messages_read_allowed(self, anima_dir: Path):
        """Read operations (no 'send') should not be blocked."""
        result = _check_a1_bash_command("chatwork messages room1", anima_dir)
        assert result is None

    def test_chatwork_unreplied_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("chatwork unreplied --json", anima_dir)
        assert result is None

    def test_normal_curl_allowed(self, anima_dir: Path):
        result = _check_a1_bash_command("curl https://example.com", anima_dir)
        assert result is None

    def test_wget_chatwork_api_blocked(self, anima_dir: Path):
        cmd = "wget --post-data='body=hello' https://api.chatwork.com/v2/rooms/123/messages"
        result = _check_a1_bash_command(cmd, anima_dir)
        assert result is not None
        assert "wget" in result

    def test_blocked_patterns_is_nonempty_list(self):
        assert len(_BASH_BLOCKED_PATTERNS) >= 4


# ── _summarise_tool_input ───────────────────────────────────


class TestSummariseToolInput:
    def test_bash_returns_command(self):
        result = _summarise_tool_input("Bash", {"command": "ls -la"})
        assert result == "ls -la"

    def test_bash_truncates_long_command(self):
        long_cmd = "x" * 500
        result = _summarise_tool_input("Bash", {"command": long_cmd})
        assert len(result) <= 300

    def test_read_returns_file_path(self):
        result = _summarise_tool_input("Read", {"file_path": "/tmp/foo"})
        assert result == "/tmp/foo"

    def test_grep_returns_pattern(self):
        result = _summarise_tool_input("Grep", {"pattern": "hello"})
        assert result == "hello"

    def test_glob_returns_pattern(self):
        result = _summarise_tool_input("Glob", {"pattern": "*.py"})
        assert result == "*.py"

    def test_empty_bash_command(self):
        result = _summarise_tool_input("Bash", {})
        assert result == "(empty)"


# ── _log_tool_use ───────────────────────────────────────────


class TestLogToolUse:
    def test_log_creates_activity_entry(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(anima_dir, "Bash", {"command": "ls -la"})

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert len(entries) == 1
        assert entries[0]["type"] == "tool_use"
        assert entries[0]["tool"] == "Bash"

    def test_log_blocked_includes_meta(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(anima_dir, "Bash", {"command": "chatwork send"}, blocked=True, block_reason="test")

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert entries[0]["meta"]["blocked"] is True
        assert entries[0]["meta"]["reason"] == "test"

    def test_log_never_raises(self):
        """Calling with an invalid anima_dir must not raise."""
        _log_tool_use(Path("/nonexistent"), "Bash", {"command": "echo hi"})

    def test_log_write_sanitises_content(self, tmp_path: Path):
        """Write tool content should be stripped, only length kept."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        big_content = "x" * 10_000
        _log_tool_use(anima_dir, "Write", {"file_path": "/tmp/f.py", "content": big_content})

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        args = entries[0]["meta"]["args"]
        assert "content" not in args
        assert args["content_length"] == 10_000
        assert args["file_path"] == "/tmp/f.py"

    def test_log_edit_truncates_strings(self, tmp_path: Path):
        """Edit tool old_string/new_string should be truncated to 200 chars."""
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        _log_tool_use(anima_dir, "Edit", {
            "file_path": "/tmp/f.py",
            "old_string": "a" * 500,
            "new_string": "b" * 500,
        })

        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        args = entries[0]["meta"]["args"]
        assert len(args["old_string"]) == 200
        assert len(args["new_string"]) == 200
        assert args["file_path"] == "/tmp/f.py"


# ── _sanitise_tool_args ──────────────────────────────────


class TestSanitiseToolArgs:
    def test_write_strips_content(self):
        result = _sanitise_tool_args("Write", {"file_path": "/f.py", "content": "x" * 5000})
        assert "content" not in result
        assert result["content_length"] == 5000
        assert result["file_path"] == "/f.py"

    def test_edit_truncates_old_new(self):
        result = _sanitise_tool_args("Edit", {
            "file_path": "/f.py",
            "old_string": "a" * 500,
            "new_string": "b" * 500,
        })
        assert len(result["old_string"]) == 200
        assert len(result["new_string"]) == 200

    def test_bash_passthrough(self):
        inp = {"command": "ls -la"}
        result = _sanitise_tool_args("Bash", inp)
        assert result is inp

    def test_read_passthrough(self):
        inp = {"file_path": "/tmp/foo"}
        result = _sanitise_tool_args("Read", inp)
        assert result is inp
