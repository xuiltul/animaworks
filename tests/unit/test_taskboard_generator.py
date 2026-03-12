"""Tests for core.memory.taskboard_generator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.taskboard_generator import (
    _COMPLETED_DAYS,
    _collect_all_tasks,
    _format_deadline_short,
    _is_overdue,
    _sync_to_slack,
    _write_taskboard,
    generate_taskboard,
    regenerate_and_sync,
)
from core.schemas import TaskEntry

JST = timezone(timedelta(hours=9))


def _make_entry(
    task_id: str = "abc123",
    summary: str = "Test task",
    assignee: str = "kai",
    status: str = "pending",
    deadline: str | None = None,
    updated_at: str | None = None,
    meta: dict | None = None,
    source: str = "anima",
) -> TaskEntry:
    now_str = (updated_at or datetime(2026, 3, 12, 10, 0, tzinfo=JST).isoformat())
    return TaskEntry(
        task_id=task_id,
        ts=now_str,
        source=source,
        original_instruction="test instruction",
        assignee=assignee,
        status=status,
        summary=summary,
        deadline=deadline,
        relay_chain=[],
        updated_at=now_str,
        meta=meta or {},
    )


class TestFormatDeadlineShort:
    def test_none(self):
        assert _format_deadline_short(None) == "—"

    def test_valid_iso(self):
        assert _format_deadline_short("2026-03-15T10:00:00+09:00") == "3/15"

    def test_invalid(self):
        assert _format_deadline_short("invalid") == "—"


class TestIsOverdue:
    def test_no_deadline(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        assert _is_overdue(None, now) is False

    def test_future_deadline(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        assert _is_overdue("2026-03-15T10:00:00+09:00", now) is False

    def test_past_deadline(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        assert _is_overdue("2026-03-10T10:00:00+09:00", now) is True


class TestGenerateTaskboard:
    """Test Markdown generation from task data."""

    def test_empty_tasks(self):
        md = generate_taskboard(tasks={}, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "# タスクボード" in md
        assert "自動生成" in md
        assert "（なし）" in md

    def test_blocked_task(self):
        tasks = {
            "t1": _make_entry(task_id="t1", summary="GA4 blocked", status="blocked", assignee="kai"),
        }
        md = generate_taskboard(tasks=tasks, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "🔴 ブロック中" in md
        assert "GA4 blocked" in md
        assert "kai" in md

    def test_in_progress_task(self):
        tasks = {
            "t1": _make_entry(task_id="t1", summary="LP改善", status="in_progress", assignee="mio"),
        }
        md = generate_taskboard(tasks=tasks, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "🟡 進行中" in md
        assert "LP改善" in md

    def test_delegated_shows_as_in_progress(self):
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="SEO audit",
                status="delegated",
                assignee="kaede",
                meta={"delegated_to": "sena"},
            ),
        }
        md = generate_taskboard(tasks=tasks, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "🟡 進行中" in md
        assert "委任中→sena" in md

    def test_pending_task(self):
        tasks = {
            "t1": _make_entry(task_id="t1", summary="UTM設定", status="pending", assignee="hina"),
        }
        md = generate_taskboard(tasks=tasks, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "📋 未着手" in md
        assert "UTM設定" in md

    def test_done_within_window(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="Completed task",
                status="done",
                assignee="kai",
                updated_at=(now - timedelta(days=2)).isoformat(),
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        assert "✅" in md
        assert "Completed task" in md

    def test_done_outside_window_excluded(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="Old completed",
                status="done",
                assignee="kai",
                updated_at=(now - timedelta(days=_COMPLETED_DAYS + 1)).isoformat(),
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        assert "Old completed" not in md

    def test_overdue_marker(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="Overdue task",
                status="pending",
                assignee="ren",
                deadline="2026-03-10T10:00:00+09:00",
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        assert "⚠️期限超過" in md

    def test_cancelled_shows_note(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="Cancelled work",
                status="cancelled",
                assignee="kai",
                updated_at=(now - timedelta(days=1)).isoformat(),
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        assert "キャンセル" in md

    def test_auto_generated_notice(self):
        md = generate_taskboard(tasks={}, now=datetime(2026, 3, 12, 10, 0, tzinfo=JST))
        assert "直接編集しないでください" in md

    def test_sorting_by_deadline(self):
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "t1": _make_entry(
                task_id="t1",
                summary="Later task",
                status="pending",
                deadline="2026-03-20T10:00:00+09:00",
            ),
            "t2": _make_entry(
                task_id="t2",
                summary="Earlier task",
                status="pending",
                deadline="2026-03-14T10:00:00+09:00",
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        pos_earlier = md.index("Earlier task")
        pos_later = md.index("Later task")
        assert pos_earlier < pos_later

    def test_mixed_statuses(self):
        """All sections render correctly when tasks span all statuses."""
        now = datetime(2026, 3, 12, 10, 0, tzinfo=JST)
        tasks = {
            "b1": _make_entry(task_id="b1", summary="Blocked one", status="blocked"),
            "p1": _make_entry(task_id="p1", summary="In progress one", status="in_progress"),
            "n1": _make_entry(task_id="n1", summary="Pending one", status="pending"),
            "d1": _make_entry(
                task_id="d1",
                summary="Done one",
                status="done",
                updated_at=(now - timedelta(days=1)).isoformat(),
            ),
        }
        md = generate_taskboard(tasks=tasks, now=now)
        assert "Blocked one" in md
        assert "In progress one" in md
        assert "Pending one" in md
        assert "Done one" in md


class TestCollectAllTasks:
    """Test cross-agent task collection and deduplication."""

    def test_delegated_dedup(self, tmp_path: Path):
        """Delegated task on kaede should not duplicate executor's copy."""
        # kaede: delegated
        kaede_dir = tmp_path / "animas" / "kaede"
        (kaede_dir / "state").mkdir(parents=True)
        kaede_task = {
            "task_id": "shared123",
            "ts": "2026-03-12T10:00:00+09:00",
            "source": "anima",
            "original_instruction": "Do something",
            "assignee": "kaede",
            "status": "delegated",
            "summary": "Delegated task",
            "deadline": "2026-03-15T10:00:00+09:00",
            "relay_chain": [],
            "updated_at": "2026-03-12T10:00:00+09:00",
            "meta": {"delegated_to": "kai", "delegated_task_id": "shared123"},
        }
        (kaede_dir / "state" / "task_queue.jsonl").write_text(
            json.dumps(kaede_task) + "\n"
        )

        # kai: in_progress (same task_id)
        kai_dir = tmp_path / "animas" / "kai"
        (kai_dir / "state").mkdir(parents=True)
        kai_task = {
            "task_id": "shared123",
            "ts": "2026-03-12T10:00:00+09:00",
            "source": "anima",
            "original_instruction": "Do something",
            "assignee": "kai",
            "status": "in_progress",
            "summary": "Working on it",
            "deadline": "2026-03-15T10:00:00+09:00",
            "relay_chain": ["kaede"],
            "updated_at": "2026-03-12T11:00:00+09:00",
            "meta": {},
        }
        (kai_dir / "state" / "task_queue.jsonl").write_text(
            json.dumps(kai_task) + "\n"
        )

        with patch("core.memory.taskboard_generator.get_animas_dir", return_value=tmp_path / "animas"):
            tasks = _collect_all_tasks()

        # Should have exactly 1 task, with executor's status
        assert len(tasks) == 1
        assert tasks["shared123"].status == "in_progress"
        assert tasks["shared123"].assignee == "kai"

    def test_delegated_without_executor(self, tmp_path: Path):
        """Delegated task with no executor copy should still appear."""
        kaede_dir = tmp_path / "animas" / "kaede"
        (kaede_dir / "state").mkdir(parents=True)
        task = {
            "task_id": "orphan456",
            "ts": "2026-03-12T10:00:00+09:00",
            "source": "anima",
            "original_instruction": "Do something",
            "assignee": "kaede",
            "status": "delegated",
            "summary": "Orphan delegated",
            "deadline": "2026-03-15T10:00:00+09:00",
            "relay_chain": [],
            "updated_at": "2026-03-12T10:00:00+09:00",
            "meta": {},
        }
        (kaede_dir / "state" / "task_queue.jsonl").write_text(json.dumps(task) + "\n")

        with patch("core.memory.taskboard_generator.get_animas_dir", return_value=tmp_path / "animas"):
            tasks = _collect_all_tasks()

        assert "orphan456" in tasks
        assert tasks["orphan456"].status == "delegated"

    def test_empty_animas_dir(self, tmp_path: Path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        with patch("core.memory.taskboard_generator.get_animas_dir", return_value=animas_dir):
            tasks = _collect_all_tasks()

        assert tasks == {}

    def test_missing_animas_dir(self, tmp_path: Path):
        with patch("core.memory.taskboard_generator.get_animas_dir", return_value=tmp_path / "nonexistent"):
            tasks = _collect_all_tasks()

        assert tasks == {}


class TestWriteTaskboard:
    def test_writes_file(self, tmp_path: Path):
        with patch("core.memory.taskboard_generator.get_shared_dir", return_value=tmp_path / "shared"):
            path = _write_taskboard("# Test Board\n")

        assert path.exists()
        assert path.read_text() == "# Test Board\n"
        assert path.name == "task-board.md"


class TestSyncToSlack:
    def test_update_existing_message(self, tmp_path: Path):
        # Setup slack json
        slack_json = tmp_path / "task-board-slack.json"
        slack_json.write_text(json.dumps({"channel": "C123", "pinned_ts": "111.222"}))

        mock_client = MagicMock()

        with (
            patch("core.memory.taskboard_generator.get_shared_dir", return_value=tmp_path),
            patch("core.tools._base.get_credential", return_value="xoxb-test"),
            patch("core.tools.slack.SlackClient", return_value=mock_client),
            patch("core.tools.slack.taskboard_md_to_slack", return_value="slack text"),
        ):
            _sync_to_slack("# Board")

        mock_client.update_message.assert_called_once_with("C0AJ4J5KK46", "111.222", "slack text")

    def test_post_new_when_no_ts(self, tmp_path: Path):
        # No slack json
        mock_client = MagicMock()
        mock_client.post_message.return_value = {"ts": "999.888"}

        with (
            patch("core.memory.taskboard_generator.get_shared_dir", return_value=tmp_path),
            patch("core.tools._base.get_credential", return_value="xoxb-test"),
            patch("core.tools.slack.SlackClient", return_value=mock_client),
            patch("core.tools.slack.taskboard_md_to_slack", return_value="slack text"),
        ):
            _sync_to_slack("# Board")

        mock_client.post_message.assert_called_once()
        # Check ts was saved
        data = json.loads((tmp_path / "task-board-slack.json").read_text())
        assert data["pinned_ts"] == "999.888"

    def test_fallback_to_post_on_update_failure(self, tmp_path: Path):
        slack_json = tmp_path / "task-board-slack.json"
        slack_json.write_text(json.dumps({"channel": "C123", "pinned_ts": "old.ts"}))

        mock_client = MagicMock()
        mock_client.update_message.side_effect = Exception("message_not_found")
        mock_client.post_message.return_value = {"ts": "new.ts"}

        with (
            patch("core.memory.taskboard_generator.get_shared_dir", return_value=tmp_path),
            patch("core.tools._base.get_credential", return_value="xoxb-test"),
            patch("core.tools.slack.SlackClient", return_value=mock_client),
            patch("core.tools.slack.taskboard_md_to_slack", return_value="slack text"),
        ):
            _sync_to_slack("# Board")

        mock_client.post_message.assert_called_once()
        data = json.loads(slack_json.read_text())
        assert data["pinned_ts"] == "new.ts"

    def test_no_slack_token_skips(self, tmp_path: Path):
        """Should not raise when Slack token is unavailable."""
        with (
            patch("core.memory.taskboard_generator.get_shared_dir", return_value=tmp_path),
            patch("core.tools._base.get_credential", side_effect=Exception("no token")),
        ):
            _sync_to_slack("# Board")  # should not raise


class TestRegenerateAndSync:
    def test_end_to_end(self, tmp_path: Path):
        """Full pipeline: tasks → markdown → file → slack."""
        # Create agent with task
        kai_dir = tmp_path / "animas" / "kai"
        (kai_dir / "state").mkdir(parents=True)
        task = {
            "task_id": "e2e_test1",
            "ts": "2026-03-12T10:00:00+09:00",
            "source": "anima",
            "original_instruction": "Build LP",
            "assignee": "kai",
            "status": "in_progress",
            "summary": "LP改善実装",
            "deadline": "2026-03-15T10:00:00+09:00",
            "relay_chain": [],
            "updated_at": "2026-03-12T10:00:00+09:00",
            "meta": {},
        }
        (kai_dir / "state" / "task_queue.jsonl").write_text(json.dumps(task) + "\n")

        shared_dir = tmp_path / "shared"

        with (
            patch("core.memory.taskboard_generator.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.memory.taskboard_generator.get_shared_dir", return_value=shared_dir),
            patch("core.memory.taskboard_generator._sync_to_slack") as mock_slack,
        ):
            result = regenerate_and_sync()

        assert "LP改善実装" in result
        assert (shared_dir / "task-board.md").exists()
        board_content = (shared_dir / "task-board.md").read_text()
        assert "LP改善実装" in board_content
        mock_slack.assert_called_once()

    def test_handles_errors_gracefully(self):
        """Should not raise even if everything fails."""
        with patch("core.memory.taskboard_generator._collect_all_tasks", side_effect=Exception("boom")):
            result = regenerate_and_sync()
        assert result == ""
