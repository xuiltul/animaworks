# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: submit -> pending -> notification -> drain chain.

Validates the full lifecycle of background task execution:
1. ``_handle_submit()`` creates a pending file with correct structure
2. ``_on_background_task_complete()`` writes notification to inbox
3. ``drain_background_notifications()`` reads and deletes notifications
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.background import BackgroundTask, TaskStatus


class TestSubmitDrainIntegration:
    """Integration test: submit -> pending -> notification -> drain chain."""

    def test_submit_creates_pending_that_watcher_expects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify pending file format matches what _execute_pending_task expects."""
        from core.tools import _handle_submit

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        with patch("builtins.print"):
            _handle_submit(["image_gen", "3d", "--model", "test"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        files = list(pending_dir.glob("*.json"))
        assert len(files) == 1

        desc = json.loads(files[0].read_text(encoding="utf-8"))
        # These are the fields _execute_pending_task reads
        assert "task_id" in desc
        assert desc["tool_name"] == "image_gen"
        assert desc["subcommand"] == "3d"
        assert desc["raw_args"] == ["3d", "--model", "test"]
        assert desc["anima_name"] == "test-anima"
        assert desc["anima_dir"] == str(anima_dir)

    def test_notification_write_then_drain(self, tmp_path: Path) -> None:
        """Verify _on_background_task_complete -> drain_background_notifications chain."""
        # Simulate what _on_background_task_complete writes
        notif_dir = tmp_path / "state" / "background_notifications"
        notif_dir.mkdir(parents=True)

        task_id = "abc123"
        notif_content = (
            "# バックグラウンドタスク完了: image_gen\n\n"
            f"- タスクID: {task_id}\n"
            "- ツール: image_gen\n"
            "- ステータス: completed\n"
            "- 結果: [image_gen] completed: OK\n"
        )
        (notif_dir / f"{task_id}.md").write_text(notif_content, encoding="utf-8")

        # Now drain using the actual method
        # We need to bypass __init__ like existing tests do
        from core.anima import DigitalAnima

        anima = object.__new__(DigitalAnima)
        mock_agent = MagicMock()
        mock_agent.anima_dir = tmp_path
        mock_agent.background_manager = None
        mock_agent.has_human_notifier = False
        anima.agent = mock_agent
        anima.name = "test-anima"
        anima._ws_broadcast = None

        result = anima.drain_background_notifications()
        assert len(result) == 1
        assert "image_gen" in result[0]
        assert task_id in result[0]

        # Verify file was deleted
        assert not list(notif_dir.glob("*.md"))

    async def test_on_complete_then_drain_roundtrip(self, tmp_path: Path) -> None:
        """Full roundtrip: _on_background_task_complete writes, drain reads."""
        from core.anima import DigitalAnima

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        anima = object.__new__(DigitalAnima)
        mock_agent = MagicMock()
        mock_agent.anima_dir = anima_dir
        mock_agent.background_manager = None
        mock_agent.has_human_notifier = False
        anima.agent = mock_agent
        anima.name = "test-anima"
        anima._ws_broadcast = None

        # Step 1: _on_background_task_complete writes notification
        task = BackgroundTask(
            task_id="roundtrip01",
            anima_name="test-anima",
            tool_name="image_gen",
            tool_args={"subcommand": "3d"},
            status=TaskStatus.COMPLETED,
            result="Generated model",
        )
        await anima._on_background_task_complete(task)

        # Step 2: Verify notification file exists
        notif_dir = anima_dir / "state" / "background_notifications"
        assert (notif_dir / "roundtrip01.md").exists()

        # Step 3: drain_background_notifications reads and deletes
        notifications = anima.drain_background_notifications()
        assert len(notifications) == 1
        assert "roundtrip01" in notifications[0]
        assert "image_gen" in notifications[0]
        assert "完了" in notifications[0]

        # Step 4: Second drain returns empty (files deleted)
        assert anima.drain_background_notifications() == []

    def test_pending_file_has_all_watcher_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify pending JSON has every field _execute_pending_task reads via .get()."""
        from core.tools import _handle_submit

        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        with patch("builtins.print"):
            _handle_submit(["transcribe", "--language", "ja", "audio.wav"])

        pending_dir = anima_dir / "state" / "background_tasks" / "pending"
        desc = json.loads(list(pending_dir.glob("*.json"))[0].read_text(encoding="utf-8"))

        # Fields accessed by _execute_pending_task via .get()
        watcher_fields = {"task_id", "tool_name", "subcommand", "raw_args", "anima_dir"}
        assert watcher_fields.issubset(desc.keys()), (
            f"Missing fields: {watcher_fields - set(desc.keys())}"
        )
