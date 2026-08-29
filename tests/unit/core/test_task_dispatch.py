from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.task_queue import TaskQueueManager
from core.tasks_dispatch import dispatch_direct_task


def test_dispatch_direct_task_queues_and_publishes_pending(tmp_path: Path) -> None:
    target_dir = tmp_path / "natsume"
    target_dir.mkdir()

    assert dispatch_direct_task(
        target="natsume",
        task_id="gh-cmd-101",
        summary="Fix requested comment",
        instruction="Do the requested fix.",
        meta={"origin": "spoofed", "repo": "o/r"},
        animas_dir=tmp_path,
    )

    task = TaskQueueManager(target_dir).get_task_by_id("gh-cmd-101")
    assert task is not None
    assert task.assignee == "natsume"
    assert task.meta["origin"] == "github-event"
    assert task.meta["executor"] == "taskexec"
    assert task.meta["repo"] == "o/r"
    pending = json.loads((target_dir / "state" / "pending" / "gh-cmd-101.json").read_text(encoding="utf-8"))
    assert pending == {
        "task_type": "llm",
        "task_id": "gh-cmd-101",
        "title": "Fix requested comment",
        "description": "Do the requested fix.",
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": "github-event-dispatch",
        "submitted_at": pending["submitted_at"],
        "reply_to": "",
        "source": "delegation",
        "working_directory": "",
    }


def test_dispatch_direct_task_skips_active_duplicate_without_pending(tmp_path: Path) -> None:
    target_dir = tmp_path / "natsume"
    target_dir.mkdir()
    kwargs = {
        "target": "natsume",
        "task_id": "gh-ci-o-r#1-aaaaaaaa",
        "summary": "Fix CI",
        "instruction": "Fix it.",
        "animas_dir": tmp_path,
    }
    assert dispatch_direct_task(**kwargs)
    pending = target_dir / "state" / "pending" / "gh-ci-o-r#1-aaaaaaaa.json"
    pending.unlink()

    assert dispatch_direct_task(**kwargs) is False
    assert not pending.exists()


def test_dispatch_direct_task_stores_model_in_task_and_pending(tmp_path: Path) -> None:
    target_dir = tmp_path / "sumire"
    target_dir.mkdir()

    assert dispatch_direct_task(
        target="sumire",
        task_id="gh-ci-o-r#1-m-grok-grok-4-5",
        summary="Multi-pass review",
        instruction="Review it.",
        model="x:grok/grok-4.5",
        animas_dir=tmp_path,
    )

    task = TaskQueueManager(target_dir).get_task_by_id("gh-ci-o-r#1-m-grok-grok-4-5")
    assert task is not None
    assert task.meta["model"] == "x:grok/grok-4.5"
    pending = json.loads(
        (target_dir / "state" / "pending" / "gh-ci-o-r#1-m-grok-grok-4-5.json").read_text(encoding="utf-8")
    )
    assert pending["model"] == "x:grok/grok-4.5"


def test_dispatch_direct_task_without_model_omits_key(tmp_path: Path) -> None:
    target_dir = tmp_path / "sumire"
    target_dir.mkdir()
    dispatch_direct_task(
        target="sumire",
        task_id="gh-ci-o-r#1-aaaaaaaa",
        summary="Single review",
        instruction="Review it.",
        animas_dir=tmp_path,
    )
    pending = json.loads((target_dir / "state" / "pending" / "gh-ci-o-r#1-aaaaaaaa.json").read_text(encoding="utf-8"))
    assert "model" not in pending


def test_dispatch_direct_task_rejects_missing_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Anima directory not found: missing"):
        dispatch_direct_task(
            target="missing",
            task_id="gh-cmd-404",
            summary="Missing",
            instruction="No target.",
            animas_dir=tmp_path,
        )
