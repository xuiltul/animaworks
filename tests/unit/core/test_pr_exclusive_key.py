from __future__ import annotations

import json
from pathlib import Path

from core.blocked_recovery import regenerate_pending_json
from core.memory.task_queue import TaskQueueManager
from core.tasks_dispatch import dispatch_direct_task, pr_exclusive_key
from core.tooling.handler_delegation import _pr_key_from_text


def test_pr_exclusive_key_from_meta() -> None:
    assert pr_exclusive_key({"repo": "o/r", "number": 5008}) == "pr-5008"
    assert pr_exclusive_key({"repo": "o/r"}) == ""
    assert pr_exclusive_key(None) == ""


def test_pr_key_from_text() -> None:
    assert _pr_key_from_text("fix CI on PR #5008 now") == "pr-5008"
    assert _pr_key_from_text("see https://github.com/o/r/pull/4985 please") == "pr-4985"
    assert _pr_key_from_text("no reference here") == ""
    assert _pr_key_from_text("item #1 of the list") == ""


def test_dispatch_direct_task_sets_exclusive_key(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    assert dispatch_direct_task(
        target="natsume",
        task_id="gh-ci-o-r#5008-360895c3",
        summary="CI failed",
        instruction="fix it",
        meta={"repo": "o/r", "number": 5008, "sha": "360895c3"},
        animas_dir=tmp_path / "animas",
    )
    task_desc = json.loads((anima_dir / "state" / "pending" / "gh-ci-o-r#5008-360895c3.json").read_text())
    assert task_desc["exclusive_key"] == "pr-5008"
    entry = TaskQueueManager(anima_dir).get_task_by_id("gh-ci-o-r#5008-360895c3")
    assert entry is not None and entry.meta["exclusive_key"] == "pr-5008"


def test_regenerate_pending_json_restores_exclusive_key(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="fix PR",
        assignee="natsume",
        summary="fix PR",
        task_id="regen-1",
        meta={"exclusive_key": "pr-5008"},
    )
    entry = manager.get_task_by_id("regen-1")
    assert entry is not None
    assert regenerate_pending_json(anima_dir, "natsume", entry)
    task_desc = json.loads((anima_dir / "state" / "pending" / "regen-1.json").read_text())
    assert task_desc["exclusive_key"] == "pr-5008"
