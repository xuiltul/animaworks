from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the orphan task reaper."""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry
from core.supervisor.orphan_reaper import (
    find_orphan_tasks,
    notify_reaped,
    reap_orphan_tasks,
)
from core.time_utils import now_local


def _make_entry(task_id: str, summary: str = "summary") -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        ts=now_local().isoformat(),
        source="anima",
        original_instruction="inst",
        assignee="aoi",
        status="pending",
        summary=summary,
        updated_at=now_local().isoformat(),
        meta={},
    )


def _age_ledger(anima_dir: Path, task_id: str, hours: float) -> None:
    """Rewrite the ledger so ``task_id`` looks ``hours`` hours old."""
    path = anima_dir / "state" / "task_queue.jsonl"
    past = (now_local() - timedelta(hours=hours)).isoformat()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("task_id") == task_id:
            data["updated_at"] = past
            data["ts"] = past
            out.append(json.dumps(data, ensure_ascii=False))
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _manager(anima_dir: Path) -> TaskQueueManager:
    return TaskQueueManager(anima_dir)


def _write_descriptor(anima_dir: Path, task_id: str, *, subdir: str | None = None) -> None:
    base = anima_dir / "state" / "pending"
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{task_id}.json").write_text("{}", encoding="utf-8")


class TestReapOrphanTasks:
    def test_reaps_ledger_row_without_descriptor(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima",
            original_instruction="never runs",
            assignee="aoi",
            summary="kept title",
        )
        _age_ledger(anima_dir, entry.task_id, hours=2)

        reaped = reap_orphan_tasks(anima_dir, active_task_ids=set())

        assert [e.task_id for e in reaped] == [entry.task_id]
        after = manager.get_task_by_id(entry.task_id)
        assert after.status == "cancelled"
        assert after.meta["orphan_reason"] == "descriptor missing"
        # Title must be preserved, not overwritten by the cancel.
        assert after.summary == "kept title"

    def test_does_not_reap_delegated_rows(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        delegated = manager.add_delegated_task(
            original_instruction="handed off",
            assignee="rin",
            summary="delegated",
        )
        _age_ledger(anima_dir, delegated.task_id, hours=5)

        reaped = reap_orphan_tasks(anima_dir, active_task_ids=set())

        assert reaped == []
        assert manager.get_task_by_id(delegated.task_id).status == "delegated"

    def test_does_not_reap_row_with_descriptor_in_pending(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="s"
        )
        _write_descriptor(anima_dir, entry.task_id)
        _age_ledger(anima_dir, entry.task_id, hours=3)

        assert reap_orphan_tasks(anima_dir, active_task_ids=set()) == []
        assert manager.get_task_by_id(entry.task_id).status == "pending"

    def test_does_not_reap_row_with_descriptor_in_processing(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="s"
        )
        _write_descriptor(anima_dir, entry.task_id, subdir="processing")
        _age_ledger(anima_dir, entry.task_id, hours=3)

        assert reap_orphan_tasks(anima_dir, active_task_ids=set()) == []
        assert manager.get_task_by_id(entry.task_id).status == "pending"

    def test_does_not_reap_recent_row(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="s"
        )
        _age_ledger(anima_dir, entry.task_id, hours=5 / 60)  # 5 min ago

        assert reap_orphan_tasks(anima_dir, active_task_ids=set()) == []
        assert manager.get_task_by_id(entry.task_id).status == "pending"

    def test_does_not_reap_active_task_ids(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="s"
        )
        _age_ledger(anima_dir, entry.task_id, hours=3)

        reaped = reap_orphan_tasks(anima_dir, active_task_ids={entry.task_id})

        assert reaped == []
        assert manager.get_task_by_id(entry.task_id).status == "pending"

    def test_respects_max_per_sweep(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        ids: list[str] = []
        for i in range(60):
            e = manager.add_task(
                source="anima",
                original_instruction="x",
                assignee="aoi",
                summary=f"s{i}",
            )
            ids.append(e.task_id)
            _age_ledger(anima_dir, e.task_id, hours=3)

        reaped = reap_orphan_tasks(anima_dir, active_task_ids=set(), max_per_sweep=50)

        assert len(reaped) == 50
        cancelled = [t for t in manager.list_tasks(status="cancelled")]
        assert len(cancelled) == 50
        remaining = [t for t in manager.list_tasks(status="pending")]
        assert len(remaining) == 10

    def test_grace_seconds_env_override(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        entry = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="s"
        )
        _age_ledger(anima_dir, entry.task_id, hours=5 / 60)  # 5 min ago
        monkeypatch.setenv("ANIMAWORKS_ORPHAN_GRACE_SECONDS", "60")

        reaped = reap_orphan_tasks(anima_dir, active_task_ids=set())

        assert [e.task_id for e in reaped] == [entry.task_id]
        assert manager.get_task_by_id(entry.task_id).status == "cancelled"


class TestFindOrphanTasks:
    def test_returns_oldest_first(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        old = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="old"
        )
        newer = manager.add_task(
            source="anima", original_instruction="x", assignee="aoi", summary="newer"
        )
        _age_ledger(anima_dir, old.task_id, hours=5)
        _age_ledger(anima_dir, newer.task_id, hours=2)

        result = find_orphan_tasks(
            anima_dir,
            now=now_local(),
            active_task_ids=set(),
            grace_seconds=1800,
        )
        assert [e.task_id for e in result] == [old.task_id, newer.task_id]

    def test_scans_descriptor_tree_once(self, tmp_path, monkeypatch):
        """state/pending/ must be walked exactly once, not once per row."""
        import core.memory.task_queue as tq

        anima_dir = tmp_path / "aoi"
        manager = _manager(anima_dir)
        for i in range(4):
            e = manager.add_task(
                source="anima", original_instruction="x", assignee="aoi", summary=f"s{i}"
            )
            _age_ledger(anima_dir, e.task_id, hours=3)
        real = manager.add_task(
            source="anima", original_instruction="r", assignee="aoi", summary="real"
        )
        _write_descriptor(anima_dir, real.task_id)
        _age_ledger(anima_dir, real.task_id, hours=3)

        calls = {"n": 0}
        orig = tq._descriptor_ids

        def counting(adir):
            calls["n"] += 1
            return orig(adir)

        monkeypatch.setattr(tq, "_descriptor_ids", counting)

        result = find_orphan_tasks(
            anima_dir,
            now=now_local(),
            active_task_ids=set(),
            grace_seconds=1800,
        )
        assert calls["n"] == 1
        # 4 descriptor-less rows qualify; the one with a descriptor does not.
        assert len(result) == 4


class TestNotifyReaped:
    def test_notify_uses_system_source_and_single_message(self, tmp_path, monkeypatch):
        from core import messenger as messenger_mod

        send_mock = MagicMock()
        monkeypatch.setattr(messenger_mod.Messenger, "send", send_mock)

        reaped = [_make_entry(f"t{i}", summary=f"title {i}") for i in range(3)]
        notify_reaped("aoi", reaped)

        assert send_mock.call_count == 1
        _, kwargs = send_mock.call_args
        assert kwargs["source"] == "system"
        assert kwargs["to"] == "aoi"
        content = kwargs["content"]
        for i in range(3):
            assert f"t{i}" in content
        assert "submit_tasks" in content

    def test_notify_sends_nothing_when_empty(self, tmp_path, monkeypatch):
        from core import messenger as messenger_mod

        send_mock = MagicMock()
        monkeypatch.setattr(messenger_mod.Messenger, "send", send_mock)
        notify_reaped("aoi", [])
        assert send_mock.call_count == 0

    def test_notify_truncates_long_lists(self, tmp_path, monkeypatch):
        from core import messenger as messenger_mod

        send_mock = MagicMock()
        monkeypatch.setattr(messenger_mod.Messenger, "send", send_mock)

        reaped = [_make_entry(f"long{i}") for i in range(25)]
        notify_reaped("aoi", reaped)

        content = send_mock.call_args.kwargs["content"]
        # 20 listed ids; the 21st must not appear, and a trailing note must.
        assert "long0" in content
        assert "long19" in content
        assert "long20" not in content
        assert "ほか 5 件" in content


@pytest.mark.asyncio
async def test_watcher_loop_sweep_is_throttled(tmp_path, monkeypatch):
    """A sweep within the throttle interval must not re-run reap."""
    from core.supervisor.pending_executor import PendingTaskExecutor

    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    executor = PendingTaskExecutor(
        anima=MagicMock(),
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=MagicMock(),
    )

    calls = {"n": 0}

    def _counting():
        calls["n"] += 1
        return 0

    executor._run_orphan_sweep = MagicMock(side_effect=_counting)

    # First call runs (last sweep far in the past).
    await executor._maybe_run_orphan_sweep()
    assert calls["n"] == 1

    # Immediately-repeat is throttled: within the interval, no re-run.
    await executor._maybe_run_orphan_sweep()
    assert calls["n"] == 1

    # Rewinding the clock allows the next sweep.
    import time as time_mod

    executor._last_orphan_sweep = 0.0
    await executor._maybe_run_orphan_sweep()
    assert calls["n"] == 2
