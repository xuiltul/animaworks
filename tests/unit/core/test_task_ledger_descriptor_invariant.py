from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ledger↔descriptor invariant.

Invariant under test
-------------------

台帳（task_queue.jsonl）に `pending` または `in_progress` の行を作るコード経路は、
必ず同じ task_id の descriptor を `state/pending/` に書かなければならない。
`delegated` はこの不変条件の対象外。

Concretely:
- ``animaworks-tool task add`` must write BOTH a ledger row AND a descriptor
  (and roll the ledger row back to ``cancelled`` if the descriptor write fails).
- ``animaworks-tool task update`` / ``update_task`` / ``/api/internal/update-task``
  may not set ``in_progress``; that status is written only by the running TaskExec
  calling ``TaskQueueManager.update_status`` directly.
- ``list_tasks`` must flag pending rows that have no descriptor as not executable.
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
    def test_cli_task_add_writes_both_ledger_row_and_descriptor(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from cli.commands.task_cmd import _cmd_add

        args = _parse_task_args(
            "task", "add",
            "--assignee", "testanima",
            "--instruction", "do the e2e check",
            "--summary", "e2e check",
        )
        manager = TaskQueueManager(anima_dir)
        _cmd_add(args, manager)

        tasks = manager.get_pending()
        assert len(tasks) == 1
        task_id = tasks[0].task_id
        assert tasks[0].status == "pending"

        desc_path = anima_dir / "state" / "pending" / f"{task_id}.json"
        assert desc_path.is_file()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        assert desc["task_id"] == task_id

    def test_cli_task_add_rejects_foreign_assignee(self, tmp_path, monkeypatch):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        from cli.commands.task_cmd import _cmd_add

        args = _parse_task_args(
            "task", "add",
            "--assignee", "other",
            "--instruction", "x",
            "--summary", "x",
        )
        manager = TaskQueueManager(anima_dir)
        with pytest.raises(SystemExit) as exc:
            _cmd_add(args, manager)
        assert exc.value.code == 2

        # No ledger row and no descriptor left behind.
        assert not (anima_dir / "state" / "pending").exists() or not list(
            (anima_dir / "state" / "pending").rglob("*.json")
        )

    def test_cli_task_add_rolls_back_ledger_row_when_descriptor_write_fails(
        self, tmp_path, monkeypatch
    ):
        anima_dir = tmp_path / "testanima"
        (anima_dir / "state").mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        import cli.commands.task_cmd as task_cmd
        import core.memory._io as memory_io

        def _boom(*a, **k):
            raise OSError("simulated descriptor write failure")

        # Patch at the source module; _cmd_add imports it lazily at call time.
        monkeypatch.setattr(memory_io, "atomic_write_text", _boom)

        args = _parse_task_args(
            "task", "add",
            "--assignee", "testanima",
            "--instruction", "x",
            "--summary", "x",
        )
        manager = TaskQueueManager(anima_dir)
        with pytest.raises(SystemExit) as exc:
            task_cmd._cmd_add(args, manager)
        assert exc.value.code == 3

        # The ledger row must have been cancelled, not left as pending.
        tasks = manager.list_tasks(status="cancelled")
        assert len(tasks) == 1
        assert tasks[0].status == "cancelled"


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

        res = json.loads(
            handler.handle("update_task", {"task_id": task_id, "status": "in_progress"})
        )
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
        # Row with a real descriptor.
        real_path = anima_dir / "state" / "pending"
        real_path.mkdir(parents=True)
        real = manager.add_task(
            source="anima",
            original_instruction="will run",
            assignee="aoi",
            summary="real",
        )
        (real_path / f"{real.task_id}.json").write_text("{}", encoding="utf-8")

        data = json.loads(handler.handle("list_tasks", {}))
        by_id = {item["task_id"]: item for item in data}

        ghost_item = by_id[ghost.task_id]
        assert ghost_item["executable"] is False
        assert "executable_note" in ghost_item

        real_item = by_id[real.task_id]
        assert real_item["executable"] is True

    def test_mark_executability_scans_pending_once(self, tmp_path, monkeypatch):
        """mark_executability should scan state/pending only once, not per row."""
        import core.memory.task_queue as tq

        anima_dir = tmp_path / "aoi"
        (anima_dir / "state" / "pending").mkdir(parents=True)
        manager = TaskQueueManager(anima_dir)
        # Several ledger rows, only one with a real descriptor.
        for i in range(4):
            manager.add_task(
                source="anima", original_instruction="x", assignee="aoi", summary=f"s{i}"
            )
        real = manager.add_task(
            source="anima", original_instruction="r", assignee="aoi", summary="real"
        )
        (anima_dir / "state" / "pending" / f"{real.task_id}.json").write_text("{}")

        calls = {"n": 0}
        orig = tq._descriptor_ids

        def counting(adir):
            calls["n"] += 1
            return orig(adir)

        monkeypatch.setattr(tq, "_descriptor_ids", counting)
        items = [e.model_dump() for e in manager.list_tasks()]
        tq.mark_executability(items, anima_dir)
        assert calls["n"] == 1

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
    def test_submit_tasks_writes_descriptor_for_every_active_row(self, tmp_path):
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
            assert desc_path.is_file()
            desc = json.loads(desc_path.read_text(encoding="utf-8"))
            assert desc["task_id"] == tid
