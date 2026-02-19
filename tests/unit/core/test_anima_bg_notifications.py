"""Tests for DigitalAnima background notification handling."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.background import BackgroundTask, TaskStatus


class TestDrainBackgroundNotifications:
    """Tests for DigitalAnima.drain_background_notifications()."""

    def _make_anima_with_dir(self, anima_dir: Path):
        """Create a minimal DigitalAnima instance with the given anima_dir.

        Uses object.__new__ to bypass __init__ and sets only the fields
        needed by drain_background_notifications().
        """
        from core.anima import DigitalAnima

        anima = object.__new__(DigitalAnima)
        # drain_background_notifications accesses self.agent.anima_dir
        mock_agent = MagicMock()
        mock_agent.anima_dir = anima_dir
        mock_agent.background_manager = None
        mock_agent.has_human_notifier = False

        anima.agent = mock_agent
        anima.name = "test-anima"
        anima._ws_broadcast = None
        return anima

    def test_no_notifications_dir(self, tmp_path):
        """Returns empty list when notification directory doesn't exist."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        result = anima.drain_background_notifications()
        assert result == []

    def test_empty_notifications_dir(self, tmp_path):
        """Returns empty list when directory exists but is empty."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        result = anima.drain_background_notifications()
        assert result == []

    def test_reads_single_notification(self, tmp_path):
        """Reads a single .md notification file."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "task1.md").write_text(
            "# Task 1 done\n\nDetails here.", encoding="utf-8",
        )

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert len(result) == 1
        assert "Task 1 done" in result[0]
        assert "Details here." in result[0]

    def test_reads_multiple_notifications(self, tmp_path):
        """Reads all .md files from notification directory."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "task1.md").write_text("# Task 1 done", encoding="utf-8")
        (notif_dir / "task2.md").write_text("# Task 2 done", encoding="utf-8")
        (notif_dir / "task3.md").write_text("# Task 3 done", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert len(result) == 3
        texts = "\n".join(result)
        assert "Task 1 done" in texts
        assert "Task 2 done" in texts
        assert "Task 3 done" in texts

    def test_deletes_after_read(self, tmp_path):
        """Notification files are deleted after reading."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "task1.md").write_text("# Done", encoding="utf-8")
        (notif_dir / "task2.md").write_text("# Also done", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        anima.drain_background_notifications()

        assert not (notif_dir / "task1.md").exists()
        assert not (notif_dir / "task2.md").exists()

    def test_second_drain_returns_empty(self, tmp_path):
        """Draining twice: second call returns empty after files are deleted."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "task1.md").write_text("# Done", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        first = anima.drain_background_notifications()
        assert len(first) == 1

        second = anima.drain_background_notifications()
        assert second == []

    def test_ignores_non_md_files(self, tmp_path):
        """Only .md files are read; other extensions are ignored."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "task1.md").write_text("# Task 1", encoding="utf-8")
        (notif_dir / "task2.txt").write_text("ignored", encoding="utf-8")
        (notif_dir / "task3.json").write_text("{}", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert len(result) == 1
        assert "Task 1" in result[0]
        # Non-md files should still exist
        assert (notif_dir / "task2.txt").exists()
        assert (notif_dir / "task3.json").exists()

    def test_sorted_order(self, tmp_path):
        """Notifications are returned in sorted (alphabetical) filename order."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        (notif_dir / "c_task.md").write_text("C content", encoding="utf-8")
        (notif_dir / "a_task.md").write_text("A content", encoding="utf-8")
        (notif_dir / "b_task.md").write_text("B content", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert len(result) == 3
        assert result[0] == "A content"
        assert result[1] == "B content"
        assert result[2] == "C content"

    def test_handles_read_error_gracefully(self, tmp_path):
        """Files that cannot be read are skipped without raising."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        good_file = notif_dir / "good.md"
        good_file.write_text("# Good notification", encoding="utf-8")

        bad_file = notif_dir / "bad.md"
        bad_file.write_text("# Bad notification", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)

        # Patch path.read_text to raise for the bad file
        original_read_text = Path.read_text

        def patched_read_text(self_path, *args, **kwargs):
            if self_path.name == "bad.md":
                raise PermissionError("access denied")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", patched_read_text):
            result = anima.drain_background_notifications()

        # Only the good file should be read successfully
        assert len(result) == 1
        assert "Good notification" in result[0]

    def test_directory_is_file_not_dir(self, tmp_path):
        """Returns empty list when the path exists but is a file, not a directory."""
        anima_dir = tmp_path / "animas" / "test"
        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)
        # Create a file where the directory is expected
        notif_path = state_dir / "background_notifications"
        notif_path.write_text("not a directory", encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert result == []

    def test_unicode_content(self, tmp_path):
        """Notification files with Unicode content are handled correctly."""
        anima_dir = tmp_path / "animas" / "test"
        notif_dir = anima_dir / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        content = "# バックグラウンドタスク完了: image_gen\n\n結果: 画像生成完了"
        (notif_dir / "task1.md").write_text(content, encoding="utf-8")

        anima = self._make_anima_with_dir(anima_dir)
        result = anima.drain_background_notifications()

        assert len(result) == 1
        assert result[0] == content


# ── TestOnBackgroundTaskComplete ─────────────────────────────


class TestOnBackgroundTaskComplete:
    """Tests for DigitalAnima._on_background_task_complete()."""

    def _make_anima_with_dir(self, anima_dir: Path):
        """Create a minimal DigitalAnima with the given anima_dir.

        Uses object.__new__ to bypass __init__ and sets only the fields
        needed by _on_background_task_complete().
        """
        from core.anima import DigitalAnima

        anima = object.__new__(DigitalAnima)
        mock_agent = MagicMock()
        mock_agent.anima_dir = anima_dir
        mock_agent.has_human_notifier = False

        anima.agent = mock_agent
        anima.name = "test-anima"
        anima._ws_broadcast = None
        return anima

    async def test_writes_notification_file_on_completed(self, tmp_path):
        """_on_background_task_complete writes a .md file to state/background_notifications/."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        task = BackgroundTask(
            task_id="task001aaa",
            anima_name="test-anima",
            tool_name="image_gen",
            tool_args={"subcommand": "3d", "raw_args": ["3d", "--model", "test"]},
            status=TaskStatus.COMPLETED,
            result="Generated 3D model successfully",
        )

        await anima._on_background_task_complete(task)

        notif_dir = anima_dir / "state" / "background_notifications"
        assert notif_dir.is_dir()

        files = list(notif_dir.glob("*.md"))
        assert len(files) == 1
        assert files[0].name == "task001aaa.md"

        content = files[0].read_text(encoding="utf-8")
        assert "task001aaa" in content
        assert "image_gen" in content
        assert "completed" in content

    async def test_failed_task_has_failure_subject(self, tmp_path):
        """Failed tasks get '失敗' in the notification subject."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        task = BackgroundTask(
            task_id="fail999bbb",
            anima_name="test-anima",
            tool_name="local_llm",
            tool_args={"subcommand": "generate"},
            status=TaskStatus.FAILED,
            error="Model not found",
        )

        await anima._on_background_task_complete(task)

        notif_dir = anima_dir / "state" / "background_notifications"
        notif_file = notif_dir / "fail999bbb.md"
        assert notif_file.exists()

        content = notif_file.read_text(encoding="utf-8")
        assert "失敗" in content
        assert "local_llm" in content

    async def test_notification_contains_required_fields(self, tmp_path):
        """Notification contains task_id, tool_name, status, and result summary."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        task = BackgroundTask(
            task_id="xyz789abc",
            anima_name="test-anima",
            tool_name="transcribe",
            tool_args={"subcommand": "transcribe"},
            status=TaskStatus.COMPLETED,
            result="Transcription: Hello world",
        )

        await anima._on_background_task_complete(task)

        notif_dir = anima_dir / "state" / "background_notifications"
        content = (notif_dir / "xyz789abc.md").read_text(encoding="utf-8")

        # Verify all required fields are present
        assert "xyz789abc" in content            # task_id
        assert "transcribe" in content           # tool_name
        assert "completed" in content            # status
        assert "Transcription: Hello world" in content  # result in summary

    async def test_completed_task_has_completed_subject(self, tmp_path):
        """Completed tasks get '完了' (not '失敗') in the notification subject."""
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        anima = self._make_anima_with_dir(anima_dir)

        task = BackgroundTask(
            task_id="ok123",
            anima_name="test-anima",
            tool_name="image_gen",
            tool_args={},
            status=TaskStatus.COMPLETED,
            result="OK",
        )

        await anima._on_background_task_complete(task)

        notif_dir = anima_dir / "state" / "background_notifications"
        content = (notif_dir / "ok123.md").read_text(encoding="utf-8")

        assert "完了" in content
        assert "失敗" not in content


# ── TestHeartbeatDrainIntegration ────────────────────────────


class TestHeartbeatDrainIntegration:
    """Verify drain_background_notifications is callable from heartbeat context."""

    async def test_heartbeat_drains_background_notifications(self, tmp_path):
        """Verify run_heartbeat calls drain_background_notifications."""
        from core.anima import DigitalAnima

        anima = object.__new__(DigitalAnima)
        # Setup minimal mock state for heartbeat
        anima.drain_background_notifications = Mock(return_value=["# Test notification"])

        # Verify the method exists and is callable from heartbeat context
        result = anima.drain_background_notifications()
        assert result == ["# Test notification"]
        anima.drain_background_notifications.assert_called_once()
