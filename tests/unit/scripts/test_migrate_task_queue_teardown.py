"""Unit tests for scripts/migrate_task_queue_teardown.py (temp dirs only)."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from scripts.migrate_task_queue_teardown import main, migrate_anima, plan_anima


def _write_legacy_row(queue_path: Path, *, task_id: str, status: str, summary: str = "s") -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": task_id,
        "ts": "2026-08-01T00:00:00+09:00",
        "source": "human",
        "original_instruction": "legacy",
        "assignee": "anima",
        "status": status,
        "summary": summary,
        "deadline": "2026-08-01T01:00:00+09:00",
        "relay_chain": [],
        "updated_at": "2026-08-01T00:30:00+09:00",
        "meta": {},
    }
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _animas_layout(tmp_path: Path) -> Path:
    animas_dir = tmp_path / "animas"

    sakura_queue = animas_dir / "sakura" / "state" / "task_queue.jsonl"
    _write_legacy_row(sakura_queue, task_id="t-blocked", status="blocked")
    _write_legacy_row(sakura_queue, task_id="t-failed", status="failed")
    _write_legacy_row(sakura_queue, task_id="t-pending", status="pending")

    hinata_queue = animas_dir / "hinata" / "state" / "task_queue.jsonl"
    _write_legacy_row(hinata_queue, task_id="h-done", status="done")

    # anima with no queue file at all
    (animas_dir / "empty" / "state").mkdir(parents=True)

    return animas_dir


def test_plan_anima_finds_only_blocked_and_failed(tmp_path: Path) -> None:
    animas_dir = _animas_layout(tmp_path)

    assert set(plan_anima(animas_dir / "sakura")) == {"t-blocked", "t-failed"}
    assert plan_anima(animas_dir / "hinata") == []
    assert plan_anima(animas_dir / "empty") == []


def test_migrate_anima_dry_run_does_not_write(tmp_path: Path) -> None:
    animas_dir = _animas_layout(tmp_path)
    sakura_dir = animas_dir / "sakura"
    queue_path = sakura_dir / "state" / "task_queue.jsonl"
    before = queue_path.read_text(encoding="utf-8")

    result = migrate_anima(sakura_dir, dry_run=True)

    assert set(result.retired_task_ids) == {"t-blocked", "t-failed"}
    assert queue_path.read_text(encoding="utf-8") == before


def test_migrate_anima_appends_pending_update_with_note(tmp_path: Path) -> None:
    animas_dir = _animas_layout(tmp_path)
    sakura_dir = animas_dir / "sakura"

    result = migrate_anima(sakura_dir, dry_run=False)
    assert set(result.retired_task_ids) == {"t-blocked", "t-failed"}

    manager = TaskQueueManager(sakura_dir)
    blocked_entry = manager.get_task_by_id("t-blocked")
    failed_entry = manager.get_task_by_id("t-failed")
    pending_entry = manager.get_task_by_id("t-pending")

    assert blocked_entry is not None
    assert blocked_entry.status == "pending"
    assert "blocked/failed 制度廃止" in blocked_entry.summary

    assert failed_entry is not None
    assert failed_entry.status == "pending"
    assert "blocked/failed 制度廃止" in failed_entry.summary

    # Untouched task must keep its original summary (append-only: existing
    # rows are never rewritten, and non-retired tasks get no new event).
    assert pending_entry is not None
    assert pending_entry.summary == "s"

    # Append-only: the original blocked/failed creation lines are still there,
    # plus one new "update" event per retired task.
    lines = [ln for ln in (sakura_dir / "state" / "task_queue.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 3 + 2


def test_migrate_anima_no_retired_tasks_is_noop(tmp_path: Path) -> None:
    animas_dir = _animas_layout(tmp_path)
    hinata_dir = animas_dir / "hinata"
    queue_path = hinata_dir / "state" / "task_queue.jsonl"
    before = queue_path.read_text(encoding="utf-8")

    result = migrate_anima(hinata_dir, dry_run=False)

    assert result.retired_task_ids == ()
    assert queue_path.read_text(encoding="utf-8") == before


def test_main_dry_run_reports_and_does_not_write(tmp_path: Path, capsys) -> None:
    animas_dir = _animas_layout(tmp_path)
    before = (animas_dir / "sakura" / "state" / "task_queue.jsonl").read_text(encoding="utf-8")

    rc = main(["--animas-dir", str(animas_dir), "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "sakura" in out
    assert "Total: 2 task(s) planned." in out
    assert (animas_dir / "sakura" / "state" / "task_queue.jsonl").read_text(encoding="utf-8") == before


def test_main_execute_writes_updates(tmp_path: Path, capsys) -> None:
    animas_dir = _animas_layout(tmp_path)

    rc = main(["--animas-dir", str(animas_dir)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "EXECUTE" in out
    assert "Total: 2 task(s) retired." in out

    manager = TaskQueueManager(animas_dir / "sakura")
    assert manager.get_task_by_id("t-blocked").status == "pending"
    assert manager.get_task_by_id("t-failed").status == "pending"


def test_main_missing_animas_dir_returns_error(tmp_path: Path) -> None:
    rc = main(["--animas-dir", str(tmp_path / "does-not-exist")])
    assert rc == 1
