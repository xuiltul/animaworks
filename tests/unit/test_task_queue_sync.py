from __future__ import annotations

import json
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.time_utils import now_iso


def _make_animas_dir(tmp_path: Path) -> Path:
    """Create a top-level animas directory with two animas."""
    animas_dir = tmp_path / "animas"
    for name in ("supervisor", "subordinate"):
        (animas_dir / name / "state").mkdir(parents=True)
    return animas_dir


def _add_delegated(sup_tqm: TaskQueueManager, sub_tqm: TaskQueueManager, target: str) -> tuple[str, str]:
    """Helper: create a delegated task on supervisor and a pending task on subordinate.

    Returns (supervisor_task_id, subordinate_task_id).
    """
    sub_entry = sub_tqm.add_task(
        source="anima",
        original_instruction="Do the work",
        assignee=target,
        summary="Review document",
        deadline="1d",
    )
    sup_entry = sup_tqm.add_delegated_task(
        original_instruction="Do the work",
        assignee=target,
        summary="Delegated: Review document",
        deadline="1d",
        meta={"delegated_to": target, "delegated_task_id": sub_entry.task_id},
    )
    return sup_entry.task_id, sub_entry.task_id


class TestSyncDelegated:
    """Tests for TaskQueueManager.sync_delegated()."""

    def test_subordinate_done_syncs_to_done(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "done", summary="Completed")

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 1
        task = sup_tqm.get_task_by_id(sup_id)
        assert task is not None
        assert task.status == "done"
        assert "subordinate" in task.summary
        assert "完了" in task.summary or "done" in task.summary

    def test_subordinate_failed_syncs_to_failed(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "failed", summary="Error occurred")

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 1
        task = sup_tqm.get_task_by_id(sup_id)
        assert task is not None
        assert task.status == "failed"
        assert "再委任" in task.summary or "re-delegation" in task.summary

    def test_subordinate_cancelled_syncs_to_done(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "cancelled")

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 1
        task = sup_tqm.get_task_by_id(sup_id)
        assert task.status == "done"

    def test_subordinate_still_pending_no_sync(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id, _sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 0
        task = sup_tqm.get_task_by_id(sup_id)
        assert task.status == "delegated"

    def test_subordinate_dir_missing_no_error(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")

        sup_tqm.add_delegated_task(
            original_instruction="Do something",
            assignee="nonexistent",
            summary="Task for missing anima",
            deadline="1d",
            meta={"delegated_to": "nonexistent", "delegated_task_id": "abc123"},
        )

        synced = sup_tqm.sync_delegated(animas_dir)
        assert synced == 0

    def test_archive_fallback_when_compacted(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "done", summary="All done")
        sub_tqm.compact()

        sub_task = sub_tqm.get_task_by_id(sub_id)
        assert sub_task is None  # compacted away

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 1
        task = sup_tqm.get_task_by_id(sup_id)
        assert task.status == "done"

    def test_no_meta_fields_skipped(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")

        sup_tqm.add_delegated_task(
            original_instruction="No meta",
            assignee="someone",
            summary="Missing meta fields",
            deadline="1d",
            meta={},
        )

        synced = sup_tqm.sync_delegated(animas_dir)
        assert synced == 0

    def test_multiple_delegated_tasks(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        sup_id1, sub_id1 = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sup_id2, sub_id2 = _add_delegated(sup_tqm, sub_tqm, "subordinate")

        sub_tqm.update_status(sub_id1, "done")
        # sub_id2 stays pending

        synced = sup_tqm.sync_delegated(animas_dir)

        assert synced == 1
        assert sup_tqm.get_task_by_id(sup_id1).status == "done"
        assert sup_tqm.get_task_by_id(sup_id2).status == "delegated"


class TestFormatDelegatedForPriming:
    """Tests for TaskQueueManager.format_delegated_for_priming()."""

    def test_no_delegated_returns_empty(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")

        result = sup_tqm.format_delegated_for_priming(animas_dir)
        assert result == ""

    def test_delegated_with_pending_subordinate(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        _add_delegated(sup_tqm, sub_tqm, "subordinate")

        result = sup_tqm.format_delegated_for_priming(animas_dir)
        assert "📌" in result
        assert "subordinate" in result
        assert "⏳" in result

    def test_delegated_with_done_subordinate(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        _sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "done")

        result = sup_tqm.format_delegated_for_priming(animas_dir)
        assert "✅" in result

    def test_delegated_with_failed_subordinate(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        _sup_id, sub_id = _add_delegated(sup_tqm, sub_tqm, "subordinate")
        sub_tqm.update_status(sub_id, "failed")

        result = sup_tqm.format_delegated_for_priming(animas_dir)
        assert "❌" in result

    def test_capped_at_five(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        for _i in range(8):
            _add_delegated(sup_tqm, sub_tqm, "subordinate")

        result = sup_tqm.format_delegated_for_priming(animas_dir, budget_chars=5000)
        lines = [ln for ln in result.split("\n") if ln.startswith("- 📌")]
        assert len(lines) <= 5

    def test_budget_respected(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        sup_tqm = TaskQueueManager(animas_dir / "supervisor")
        sub_tqm = TaskQueueManager(animas_dir / "subordinate")

        for _ in range(5):
            _add_delegated(sup_tqm, sub_tqm, "subordinate")

        result = sup_tqm.format_delegated_for_priming(animas_dir, budget_chars=100)
        assert len(result) <= 200  # some tolerance for last line


class TestSearchArchive:
    """Tests for TaskQueueManager._search_archive()."""

    def test_finds_task_in_archive(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        target_dir = animas_dir / "subordinate"
        archive_path = target_dir / "state" / "task_queue_archive.jsonl"

        archive_data = {"task_id": "abc123", "status": "done", "ts": now_iso()}
        archive_path.write_text(json.dumps(archive_data) + "\n", encoding="utf-8")

        result = TaskQueueManager._search_archive(target_dir, "abc123")
        assert result == "done"

    def test_not_in_archive_returns_none(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        target_dir = animas_dir / "subordinate"
        archive_path = target_dir / "state" / "task_queue_archive.jsonl"

        archive_data = {"task_id": "other", "status": "done", "ts": now_iso()}
        archive_path.write_text(json.dumps(archive_data) + "\n", encoding="utf-8")

        result = TaskQueueManager._search_archive(target_dir, "abc123")
        assert result is None

    def test_no_archive_file_returns_none(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        result = TaskQueueManager._search_archive(animas_dir / "subordinate", "abc123")
        assert result is None

    def test_corrupted_archive_handled(self, tmp_path):
        animas_dir = _make_animas_dir(tmp_path)
        target_dir = animas_dir / "subordinate"
        archive_path = target_dir / "state" / "task_queue_archive.jsonl"

        archive_path.write_text('not json\n{"task_id": "x", "status": "done"}\n', encoding="utf-8")

        result = TaskQueueManager._search_archive(target_dir, "x")
        assert result == "done"
