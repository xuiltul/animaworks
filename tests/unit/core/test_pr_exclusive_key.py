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


def _dispatch_with_ack(
    tmp_path: Path,
    monkeypatch,
    task_id: str,
    meta: dict,
) -> tuple[bool, list]:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True, exist_ok=True)
    calls: list = []

    def _fake_post(target_dir, repo, number, body):
        calls.append((repo, number, body))
        return True

    monkeypatch.setattr("core.tasks_dispatch._post_pr_comment", _fake_post)
    ok = dispatch_direct_task(
        target="natsume",
        task_id=task_id,
        summary="CI failed",
        instruction="fix it",
        meta=meta,
        animas_dir=tmp_path / "animas",
    )
    return ok, calls


def test_dispatch_posts_queue_ack_when_key_held(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="in-flight PR work",
        assignee="natsume",
        summary="in-flight PR work",
        task_id="holder-1",
        meta={"exclusive_key": "pr-5008"},
    )
    ok, calls = _dispatch_with_ack(
        tmp_path,
        monkeypatch,
        task_id="gh-ci-o-r#5008-360895c3",
        meta={"repo": "o/r", "number": 5008, "sha": "360895c3"},
    )
    assert ok
    assert len(calls) == 1
    repo, number, body = calls[0]
    assert repo == "o/r"
    assert number == 5008
    assert "gh-ci-o-r#5008-360895c3" in body
    assert "holder-1" in body
    # ack recorded to prevent double posting
    entry = manager.get_task_by_id("gh-ci-o-r#5008-360895c3")
    assert entry is not None and entry.meta["queue_ack_posted"] is True


def test_dispatch_no_ack_when_holder_is_blocked(tmp_path: Path, monkeypatch) -> None:
    """A blocked holder may never finish; do not promise a start after it."""
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="stalled PR work",
        assignee="natsume",
        summary="stalled PR work",
        task_id="holder-1",
        meta={"exclusive_key": "pr-5008"},
    )
    manager.update_status("holder-1", "blocked", summary="waiting on an external blocker")

    ok, calls = _dispatch_with_ack(
        tmp_path,
        monkeypatch,
        task_id="gh-ci-o-r#5008-360895c3",
        meta={"repo": "o/r", "number": 5008, "sha": "360895c3"},
    )
    assert ok
    assert calls == []


def test_dispatch_no_ack_when_key_free(tmp_path: Path, monkeypatch) -> None:
    ok, calls = _dispatch_with_ack(
        tmp_path,
        monkeypatch,
        task_id="gh-ci-o-r#5008-360895c3",
        meta={"repo": "o/r", "number": 5008, "sha": "360895c3"},
    )
    assert ok
    assert calls == []


def test_dispatch_ack_failure_does_not_block_dispatch(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="in-flight PR work",
        assignee="natsume",
        summary="in-flight PR work",
        task_id="holder-1",
        meta={"exclusive_key": "pr-5008"},
    )

    def _boom(target_dir, repo, number, body):
        raise RuntimeError("gh down")

    monkeypatch.setattr("core.tasks_dispatch._post_pr_comment", _boom)
    ok = dispatch_direct_task(
        target="natsume",
        task_id="gh-ci-o-r#5008-360895c3",
        summary="CI failed",
        instruction="fix it",
        meta={"repo": "o/r", "number": 5008, "sha": "360895c3"},
        animas_dir=tmp_path / "animas",
    )
    assert ok
    task_desc = json.loads((anima_dir / "state" / "pending" / "gh-ci-o-r#5008-360895c3.json").read_text())
    assert task_desc["exclusive_key"] == "pr-5008"


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


def test_dispatch_acks_once_per_pr_backlog(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "natsume"
    (anima_dir / "state" / "pending").mkdir(parents=True)
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="in-flight PR work",
        assignee="natsume",
        summary="in-flight PR work",
        task_id="holder-1",
        meta={"exclusive_key": "pr-5008"},
    )
    _, calls1 = _dispatch_with_ack(
        tmp_path,
        monkeypatch,
        task_id="gh-ci-o-r#5008-aaaa",
        meta={"repo": "o/r", "number": 5008, "sha": "aaaa"},
    )
    _, calls2 = _dispatch_with_ack(
        tmp_path,
        monkeypatch,
        task_id="gh-ci-o-r#5008-bbbb",
        meta={"repo": "o/r", "number": 5008, "sha": "bbbb"},
    )
    assert len(calls1) == 1
    assert len(calls2) == 0
