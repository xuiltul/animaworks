from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for atomic canonical task publication.

Every executable task owns its complete input in the same SQLite transaction.
Backlog entries may remain intentionally unscheduled. No second ledger or
filesystem descriptor is published, and failures roll back without cancellation
compensation. Only the execution path can set in_progress through public tools.
"""

import argparse
import json

import pytest

from core.memory import MemoryManager
from core.memory.task_queue import TaskQueueManager
from core.tooling.handler import ToolHandler


def _parse_task_args(*argv: str) -> argparse.Namespace:
    """Build the real task subcommand parser and parse ``argv``."""
    from cli.commands.task_cmd import register_task_command

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register_task_command(sub)
    return parser.parse_args(argv)


def _make_handler(anima_dir) -> ToolHandler:
    for d in ["state", "episodes", "knowledge", "procedures", "skills"]:
        (anima_dir / d).mkdir(parents=True, exist_ok=True)
    memory = MemoryManager(anima_dir)
    return ToolHandler(anima_dir, memory)


class TestCliTaskAdd:
    def test_cli_task_add_atomically_persists_entry_and_complete_input(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from cli.commands.task_cmd import _cmd_add

        args = _parse_task_args(
            "task",
            "add",
            "--assignee",
            "testanima",
            "--instruction",
            "do the e2e check",
            "--summary",
            "e2e check",
        )
        manager = TaskQueueManager(anima_dir)
        _cmd_add(args, manager)

        tasks = manager.get_pending()
        assert len(tasks) == 1
        task_id = tasks[0].task_id
        assert tasks[0].status == "pending"

        desc_path = anima_dir / "state" / "pending" / f"{task_id}.json"
        assert not desc_path.exists()
        assert not manager.queue_path.exists()
        desc = manager.store.get_input(anima_dir.name, task_id)
        assert desc["task_id"] == task_id
        assert desc["description"] == tasks[0].original_instruction == "do the e2e check"

    def test_cli_task_add_rejects_foreign_assignee(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from cli.commands.task_cmd import _cmd_add

        args = _parse_task_args(
            "task",
            "add",
            "--assignee",
            "other",
            "--instruction",
            "x",
            "--summary",
            "x",
        )
        manager = TaskQueueManager(anima_dir)
        with pytest.raises(SystemExit) as exc:
            _cmd_add(args, manager)
        assert exc.value.code == 2

        # No ledger row and no descriptor left behind.
        assert not (anima_dir / "state" / "pending").exists() or not list(
            (anima_dir / "state" / "pending").rglob("*.json")
        )

    def test_cli_task_add_rolls_back_transaction_without_compensating_cancel(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        import cli.commands.task_cmd as task_cmd
        from core.taskboard.tasks import TaskStore

        original_submit = TaskStore.submit

        def _boom(store, *args, **kwargs):
            original_submit(store, *args, **kwargs)
            raise OSError("simulated failure before transaction commit")

        monkeypatch.setattr(TaskStore, "submit", _boom)

        args = _parse_task_args(
            "task",
            "add",
            "--assignee",
            "testanima",
            "--instruction",
            "x",
            "--summary",
            "x",
        )
        manager = TaskQueueManager(anima_dir)
        with pytest.raises(SystemExit) as exc:
            task_cmd._cmd_add(args, manager)
        assert exc.value.code == 3

        # Neither a partially executable task nor a compensating cancellation exists.
        assert not manager.store.read(anima_dir.name, archived=True)
        assert not manager.store.pending(anima_dir.name)


class TestInProgressRejection:
    def test_update_task_tool_rejects_in_progress(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)

        add = json.loads(
            handler.handle(
                "backlog_task",
                {
                    "source": "human",
                    "original_instruction": "test",
                    "assignee": "aoi",
                    "summary": "s",
                },
            )
        )
        task_id = add["task_id"]

        res = json.loads(handler.handle("update_task", {"task_id": task_id, "status": "in_progress"}))
        assert res["status"] == "error"
        assert res["error_type"] == "InvalidArguments"

        # Ledger status unchanged.
        manager = TaskQueueManager(anima_dir)
        assert manager.get_task_by_id(task_id).status == "pending"

    def test_cli_task_update_rejects_in_progress(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from types import SimpleNamespace

        from cli.commands.task_cmd import _cmd_update

        # Build the Namespace directly so the flow reaches _cmd_update's own
        # in_progress guard rather than argparse's choices check.
        args = SimpleNamespace(task_id="abc", status="in_progress", summary=None)
        manager = TaskQueueManager(anima_dir)
        with pytest.raises(SystemExit) as exc:
            _cmd_update(args, manager)
        assert exc.value.code == 2


class TestListTasksExecutability:
    def test_list_tasks_marks_ledger_only_rows_as_not_executable(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)
        manager = TaskQueueManager(anima_dir)

        # Ledger-only row: add_task but write no descriptor.
        ghost = manager.add_task(
            source="anima",
            original_instruction="will never run",
            assignee="aoi",
            summary="ghost",
        )
        # A complete canonical input is executable without a filesystem descriptor.
        real = manager.submit({"task_id": "real", "title": "real", "description": "will run"})

        data = json.loads(handler.handle("list_tasks", {}))
        by_id = {item["task_id"]: item for item in data}

        ghost_item = by_id[ghost.task_id]
        assert ghost_item["executable"] is False
        assert "executable_note" in ghost_item

        real_item = by_id[real.task_id]
        assert real_item["executable"] is True

    def test_mark_executability_queries_canonical_input_ids_once(self, tmp_path, monkeypatch):
        """One canonical input lookup serves the whole listing, without file scans."""
        import core.memory.task_queue as tq

        anima_dir = tmp_path / "aoi"
        (anima_dir / "state" / "pending").mkdir(parents=True)
        manager = TaskQueueManager(anima_dir)
        # Several ledger rows, only one with a real descriptor.
        for i in range(4):
            manager.add_task(source="anima", original_instruction="x", assignee="aoi", summary=f"s{i}")
        real = manager.submit({"task_id": "real", "title": "real", "description": "r"})

        calls = {"n": 0}
        orig = tq._descriptor_ids

        def counting(adir):
            calls["n"] += 1
            return orig(adir)

        monkeypatch.setattr(tq, "_descriptor_ids", counting)
        items = [e.model_dump() for e in manager.list_tasks()]
        tq.mark_executability(items, anima_dir)
        assert calls["n"] == 1
        assert [item["task_id"] for item in items if item["executable"]] == [real.task_id]

    def test_list_tasks_does_not_flag_delegated_rows(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)
        manager = TaskQueueManager(anima_dir)

        delegated = manager.add_delegated_task(
            original_instruction="handed off",
            assignee="rin",
            summary="delegated",
        )

        data = json.loads(handler.handle("list_tasks", {}))
        by_id = {item["task_id"]: item for item in data}
        item = by_id[delegated.task_id]
        assert "executable" not in item
        assert "executable_note" not in item


class TestSubmitTasksInvariant:
    def test_submission_retains_full_execution_input_beyond_display_limit(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)
        instruction = "Preserve every instruction\n" + "x" * 12_000
        task = {
            "task_id": "long-input",
            "title": "long task",
            "description": instruction,
            "context": "prior decision",
            "acceptance_criteria": ["verified"],
            "constraints": ["do not send"],
            "file_paths": ["src/example.py"],
        }
        result = json.loads(handler.handle("submit_tasks", {"batch_id": "long", "tasks": [task]}))
        assert result["status"] == "submitted"
        stored = TaskQueueManager(anima_dir).store.get_input(anima_dir.name, "long-input")
        for field in ("description", "context", "acceptance_criteria", "constraints", "file_paths"):
            assert stored[field] == task[field]

    def test_batch_failure_never_leaves_partial_work_or_compensating_cancel(self, tmp_path, monkeypatch):
        from core.taskboard.tasks import TaskStore

        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)
        original_submit = TaskStore.submit

        def fail_second(store, owner, entry, payload, **kwargs):
            original_submit(store, owner, entry, payload, **kwargs)
            if payload["task_id"] == "second":
                raise OSError("second input publication failed before commit")

        monkeypatch.setattr(TaskStore, "submit", fail_second)
        result = json.loads(
            handler.handle(
                "submit_tasks",
                {
                    "batch_id": "atomic",
                    "tasks": [
                        {"task_id": "first", "title": "first", "description": "first"},
                        {"task_id": "second", "title": "second", "description": "second", "depends_on": ["first"]},
                    ],
                },
            )
        )
        assert result["status"] == "error"
        assert not TaskQueueManager(anima_dir).store.read(anima_dir.name, archived=True)

    def test_submit_tasks_persists_complete_dag_input_without_descriptors(self, tmp_path):
        anima_dir = tmp_path / "aoi"
        handler = _make_handler(anima_dir)

        res = handler.handle(
            "submit_tasks",
            {
                "batch_id": "batch-1",
                "tasks": [
                    {
                        "task_id": "t-a",
                        "title": "A",
                        "description": "task A",
                    },
                    {
                        "task_id": "t-b",
                        "title": "B",
                        "description": "task B",
                        "depends_on": ["t-a"],
                    },
                ],
            },
        )
        assert json.loads(res)["status"] == "submitted"

        for tid in ("t-a", "t-b"):
            desc_path = anima_dir / "state" / "pending" / f"{tid}.json"
            assert not desc_path.exists()
            desc = TaskQueueManager(anima_dir).store.get_input(anima_dir.name, tid)
            assert desc["task_id"] == tid
            assert desc["description"] == ("task A" if tid == "t-a" else "task B")
            assert desc["depends_on"] == ([] if tid == "t-a" else ["t-a"])
