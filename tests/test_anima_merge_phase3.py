"""End-to-end tests for Anima merge REWRITE_REFS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.lifecycle.anima_merge import AnimaMergeService, MergePhase
from core.memory.task_queue import TaskQueueManager
from core.taskboard.store import TaskBoardStore
from tests.test_anima_merge import (
    _add_rewrite_refs_fixture,
    _setup_data_dir,
    _stub_rebuild_substeps,
    _write,
)


def test_anima_merge_rewrite_refs_updates_all_external_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, source, target = _setup_data_dir(tmp_path)
    _add_rewrite_refs_fixture(data_dir, source, target)
    _stub_rebuild_substeps(monkeypatch)
    source_before = {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }

    result = AnimaMergeService(data_dir, "source", "target").run(execute=True)

    source_after = {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    assert {k: v for k, v in source_after.items() if k != "status.json"} == {
        k: v for k, v in source_before.items() if k != "status.json"
    }
    assert json.loads(source_after["status.json"])["enabled"] is False

    worker_status = json.loads((data_dir / "animas" / "worker" / "status.json").read_text(encoding="utf-8"))
    assert worker_status["supervisor"] == "target"
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert config["animas"]["worker"]["supervisor"] == "target"
    assert config["external_messaging"]["slack"]["anima_mapping"] == {"C1": "target"}
    assert config["external_messaging"]["discord"]["channel_members"] == {"D1": ["target"]}
    assert config["external_messaging"]["zoom"]["meeting_mapping"] == {"M1": "target"}
    assert config["github_webhook"] == {"reviewer_anima": "target", "dispatcher_anima": "target"}
    assert config["external_messaging"]["slack"]["bot_token"] == "config-secret-must-not-be-journaled"

    channel = json.loads((data_dir / "shared" / "channels" / "team.meta.json").read_text(encoding="utf-8"))
    assert channel == {"members": ["target"], "created_by": "source"}
    meeting = json.loads((data_dir / "shared" / "meetings" / "open.json").read_text(encoding="utf-8"))
    assert meeting["participants"] == ["target"]
    assert meeting["chair"] == "target"
    assert meeting["conversation"][0]["speaker"] == "source"

    assert not list((data_dir / "shared" / "inbox" / "source").glob("*.json"))
    moved_message = json.loads(
        (data_dir / "shared" / "inbox" / "target" / "message__from_source.json").read_text(encoding="utf-8")
    )
    assert moved_message["id"] == "message__from_source"
    assert moved_message["thread_id"] == "message__from_source"
    assert moved_message["to_person"] == "target"
    assert moved_message["from_person"] == "source"
    assert moved_message["meta"]["task_id"] == "collision-task__from_source"
    moved_report = json.loads((data_dir / "shared" / "inbox" / "target" / "report.json").read_text(encoding="utf-8"))
    assert moved_report["to_person"] == "target"
    assert moved_report["meta"]["task_id"] == "collision-task"
    historical = json.loads((data_dir / "shared" / "inbox" / "worker" / "historical.json").read_text(encoding="utf-8"))
    assert historical["from_person"] == "source"

    merged_episode = target / "episodes" / "2026-07-15_source.md"
    assert "attachments/photo__from_source.png" in merged_episode.read_text(encoding="utf-8")
    assert "episodes/2026-07-15_source.md" in (target / "knowledge" / "linked.md").read_text(encoding="utf-8")
    qualified = (target / "knowledge" / "qualified.md").read_text(encoding="utf-8")
    assert "/api/animas/target/attachments/photo__from_source.png" in qualified
    assert "/api/animas/target/attachments/unique.png" in qualified

    mapping = {
        "collision-task": "collision-task__from_source",
        "terminal-result": "terminal-result",
        "unique-task": "unique-task",
    }
    target_queue = [
        entry.model_dump() for entry in TaskQueueManager(target).store.read("target", archived=True).values()
    ]
    assert {entry["task_id"] for entry in target_queue} == {
        "collision-task",
        "collision-task__from_source",
        "unique-task",
    }
    migrated = next(entry for entry in target_queue if entry["task_id"] == "collision-task__from_source")
    assert migrated["assignee"] == "target"
    worker_queue = next(
        iter(TaskQueueManager(data_dir / "animas" / "worker").store.read("worker").values())
    ).model_dump()
    assert worker_queue["assignee"] == "target"
    assert worker_queue["meta"]["delegated_to"] == "target"
    assert worker_queue["meta"]["delegated_task_id"] == "collision-task__from_source"
    assert worker_queue["meta"]["child_ref"] == {
        "anima_name": "target",
        "task_id": "collision-task__from_source",
    }
    with TaskQueueManager(target).store.reader() as database:
        pending = json.loads(
            database.execute(
                "SELECT input_json FROM tasks WHERE anima='target' AND task_id='collision-task__from_source'"
            ).fetchone()[0]
        )
    assert not (target / "state" / "pending" / "collision-task__from_source.json").exists()
    assert pending["task_id"] == "collision-task__from_source"
    assert pending["depends_on"] == ["unique-task"]
    assert pending["submitted_by"] == pending["reply_to"] == "target"
    assert (target / "state" / "task_results" / "unique-task.md").read_text(encoding="utf-8") == "unique result\n"
    assert (target / "state" / "task_results" / "terminal-result.md").read_text(encoding="utf-8") == "terminal result\n"

    board = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    assert {(item.anima_name, item.task_id) for item in board.list_metadata()} == {
        ("target", "collision-task"),
        ("target", "collision-task__from_source"),
        ("target", "unique-task"),
    }
    moved_board = board.get_metadata("target", "collision-task__from_source")
    assert moved_board is not None
    assert moved_board.source_ref == "task_queue:target:collision-task__from_source"
    worker_event = next(event for event in board.list_events() if event["anima_name"] == "worker")
    assert worker_event["payload"]["ref"] == {
        "anima_name": "target",
        "task_id": "collision-task__from_source",
    }

    assert (
        json.loads((data_dir / "run" / "notification_map.json").read_text(encoding="utf-8"))["thread"]["anima"]
        == "target"
    )
    assert (
        json.loads((data_dir / "run" / "discord_thread_map.json").read_text(encoding="utf-8"))["message"]["anima"]
        == "target"
    )
    usage = json.loads((data_dir / "usage_governor_state.json").read_text(encoding="utf-8"))
    assert usage["suspended_animas"] == ["target"]
    assert json.loads((data_dir / "animas" / ".bootstrap_retries.json").read_text(encoding="utf-8")) == {"target": 1}
    assert not (data_dir / "run" / "events" / "source").exists()
    assert not (data_dir / "run" / "animas" / "source.lock").exists()
    assert (data_dir / "run" / "inbox_wake" / "target").is_file()

    journal_text = result.journal_path.read_text(encoding="utf-8")
    journal = json.loads(journal_text)
    rewrite = journal["phases"][MergePhase.REWRITE_REFS.value]
    assert rewrite["status"] == "completed"
    assert rewrite["artifacts"]["task_id_mapping"] == mapping
    assert rewrite["substeps"]["memory_references"]["artifacts"]["dangling_references"] == 0
    candidates = rewrite["substeps"]["messaging"]["artifacts"]["credential_disable_candidates"]
    assert candidates == [{"storage": "shared/credentials.json", "key": "SLACK_BOT_TOKEN__source"}]
    assert "credential-secret-must-not-be-journaled" not in journal_text
    assert "target-secret-must-not-be-journaled" not in journal_text


def test_anima_merge_rewrite_refs_stops_on_organization_self_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, _source, target = _setup_data_dir(tmp_path)
    _write(target / "status.json", '{"enabled":true,"memory_backend":"legacy","supervisor":"source"}\n')
    _stub_rebuild_substeps(monkeypatch)
    status_before = (target / "status.json").read_bytes()

    service = AnimaMergeService(data_dir, "source", "target")
    with pytest.raises(RuntimeError, match="self-reference: target -> target"):
        service.run(execute=True)

    assert (target / "status.json").read_bytes() == status_before
    journal = json.loads(service.journal_path.read_text(encoding="utf-8"))
    assert journal["phases"][MergePhase.REWRITE_REFS.value]["status"] == "failed"
    assert journal["phases"][MergePhase.REWRITE_REFS.value]["substeps"]["organization"]["status"] == "failed"
    assert MergePhase.REBUILD_INDEXES.value not in journal["phases"]


def test_anima_merge_rewrite_refs_resume_reuses_mapping_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.lifecycle.anima_merge.external_refs import ExternalRefsRewriter

    data_dir, source, target = _setup_data_dir(tmp_path)
    _add_rewrite_refs_fixture(data_dir, source, target)
    _stub_rebuild_substeps(monkeypatch)
    original = ExternalRefsRewriter.rewrite_ancillary_state
    interrupted = False

    def interrupt_after_rewrite(self: ExternalRefsRewriter, *, wake_target: bool = False):
        nonlocal interrupted
        result = original(self, wake_target=wake_target)
        if not interrupted:
            interrupted = True
            raise RuntimeError("rewrite interruption")
        return result

    monkeypatch.setattr(ExternalRefsRewriter, "rewrite_ancillary_state", interrupt_after_rewrite)
    service = AnimaMergeService(data_dir, "source", "target")
    with pytest.raises(RuntimeError, match="rewrite interruption"):
        service.run(execute=True)

    failed = json.loads(service.journal_path.read_text(encoding="utf-8"))
    mapping = failed["phases"][MergePhase.REWRITE_REFS.value]["substeps"]["task_id_mapping"]["artifacts"][
        "task_id_mapping"
    ]
    queue_after_failure = (target / "state" / "task_queue.jsonl").read_bytes()
    board = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    events_after_failure = board.list_events()

    AnimaMergeService(data_dir, "source", "target").run(execute=True, resume=True)

    completed = json.loads(service.journal_path.read_text(encoding="utf-8"))
    assert completed["status"] == "done"
    assert completed["phases"][MergePhase.REWRITE_REFS.value]["artifacts"]["task_id_mapping"] == mapping
    assert (target / "state" / "task_queue.jsonl").read_bytes() == queue_after_failure
    assert TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3").list_events() == events_after_failure
    assert len(list((data_dir / "shared" / "inbox" / "target").glob("message__from_source*.json"))) == 1


def test_anima_merge_phase3_dry_run_leaves_external_state_and_db_unchanged(tmp_path: Path) -> None:
    data_dir, source, target = _setup_data_dir(tmp_path)
    _add_rewrite_refs_fixture(data_dir, source, target)
    observed = [
        data_dir / "config.json",
        data_dir / "shared" / "taskboard.sqlite3",
        data_dir / "shared" / "inbox" / "source" / "message.json",
        data_dir / "animas" / "worker" / "status.json",
        data_dir / "run" / "notification_map.json",
        data_dir / "usage_governor_state.json",
    ]
    before = {str(path): path.read_bytes() for path in observed}

    result = AnimaMergeService(data_dir, "source", "target").run()

    assert result.dry_run is True
    assert {str(path): path.read_bytes() for path in observed} == before
    assert not (data_dir / "state" / "merge_journal_source_target.json").exists()
