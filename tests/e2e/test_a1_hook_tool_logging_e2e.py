"""E2E: A1 hook tool logging and bash blocklist integration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.execution.agent_sdk import (
    _check_a1_bash_command,
    _log_tool_use,
    _BASH_BLOCKED_PATTERNS,
)


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "activity_log").mkdir()
    return d


class TestBlocklistAndLoggingIntegration:
    """Verify blocked commands are both denied and logged."""

    def test_blocked_chatwork_send_logged(self, anima_dir: Path):
        cmd = "chatwork send general hello"
        # Verify it's blocked
        reason = _check_a1_bash_command(cmd, anima_dir)
        assert reason is not None
        # Log the blocked call
        _log_tool_use(anima_dir, "Bash", {"command": cmd}, blocked=True, block_reason=reason)
        # Verify log entry
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["type"] == "tool_use"
        assert entry["tool"] == "Bash"
        assert entry["meta"]["blocked"] is True
        assert "Chatwork" in entry["meta"]["reason"]

    def test_allowed_command_logged_without_blocked(self, anima_dir: Path):
        cmd = "ls -la"
        reason = _check_a1_bash_command(cmd, anima_dir)
        assert reason is None
        _log_tool_use(anima_dir, "Bash", {"command": cmd})
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        entry = entries[0]
        assert entry["type"] == "tool_use"
        assert "blocked" not in entry.get("meta", {})

    def test_curl_chatwork_api_blocked_and_logged(self, anima_dir: Path):
        cmd = "curl -X POST https://api.chatwork.com/v2/rooms/123/messages -d 'body=test'"
        reason = _check_a1_bash_command(cmd, anima_dir)
        assert reason is not None
        _log_tool_use(anima_dir, "Bash", {"command": cmd}, blocked=True, block_reason=reason)
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert entries[0]["meta"]["blocked"] is True

    def test_multiple_tools_logged_sequentially(self, anima_dir: Path):
        _log_tool_use(anima_dir, "Read", {"file_path": "/tmp/a.txt"})
        _log_tool_use(anima_dir, "Bash", {"command": "echo hi"})
        _log_tool_use(anima_dir, "Grep", {"pattern": "foo"})
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        entries = [json.loads(line) for line in log_files[0].read_text().strip().split("\n")]
        assert len(entries) == 3
        tools = [e["tool"] for e in entries]
        assert tools == ["Read", "Bash", "Grep"]
