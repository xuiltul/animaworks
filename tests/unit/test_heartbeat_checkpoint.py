from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCheckpointAndRecovery:
    """Test heartbeat checkpoint/recovery note mechanics.

    These are unit-level tests for the file operations; full heartbeat
    integration requires mocking the agent cycle.
    """

    def test_recovery_note_write_and_read(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        recovery_path = state_dir / "recovery_note.md"

        # Simulate writing recovery note (as in anima.py exception handler)
        content = (
            "### エラー情報\n\n"
            "- エラー種別: CreditExhaustedError\n"
            "- エラー内容: Insufficient credits\n"
            "- 発生日時: 2026-02-19T10:00:00\n"
            "- 未処理メッセージ数: 3\n"
        )
        recovery_path.write_text(content, encoding="utf-8")
        assert recovery_path.exists()

        # Simulate reading and deleting (as in recovery note reader)
        read_content = recovery_path.read_text(encoding="utf-8")
        assert "CreditExhaustedError" in read_content
        recovery_path.unlink(missing_ok=True)
        assert not recovery_path.exists()

    def test_checkpoint_write_and_delete(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        checkpoint_path = state_dir / "heartbeat_checkpoint.json"

        # Simulate writing checkpoint
        checkpoint_data = {
            "ts": "2026-02-19T10:00:00",
            "trigger": "heartbeat",
            "unread_count": 5,
            "task_queue_summary": "2 pending tasks",
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint_data, ensure_ascii=False), encoding="utf-8",
        )
        assert checkpoint_path.exists()

        # Verify content
        loaded = json.loads(checkpoint_path.read_text())
        assert loaded["unread_count"] == 5

        # Simulate deletion on success
        checkpoint_path.unlink(missing_ok=True)
        assert not checkpoint_path.exists()
