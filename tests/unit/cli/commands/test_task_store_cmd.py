from __future__ import annotations

import argparse
import json
import os
import sqlite3

import pytest

from cli.commands.task_store_cmd import run_maintenance
from core.schemas import TaskEntry
from core.taskboard.tasks import TaskStore, task_database_path


def entry(task_id: str, status: str = "pending") -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        ts="2026-09-08T00:00:00+00:00",
        updated_at="2026-09-08T00:00:00+00:00",
        source="human",
        original_instruction="Keep the full original instruction.",
        assignee="alice",
        status=status,
        summary=task_id,
    )


def seed(store: TaskStore, task_id: str) -> None:
    store.submit(
        "alice",
        entry(task_id),
        {
            "task_type": "llm",
            "task_id": task_id,
            "description": "Original input",
            "model": "openai/fixture",
            "working_directory": "/workspace",
            "acceptance_criteria": ["No duplicate effect"],
        },
    )


@pytest.fixture
def maintenance(data_dir, make_anima, monkeypatch):
    anima_dir = make_anima("alice")
    monkeypatch.setattr("cli.commands.server._find_server_pid_by_process", lambda: None)
    store = TaskStore(task_database_path(anima_dir))
    return data_dir, anima_dir, store


def test_pause_persists_and_prevents_new_claim(maintenance):
    _, _, store = maintenance
    seed(store, "first")
    result = run_maintenance(argparse.Namespace(anima="alice", task_store_action="quiesce"))
    assert result["quiesced"] is True
    assert TaskStore(store.db_path).claim("alice", "first", {}) is None
    run_maintenance(argparse.Namespace(anima="alice", task_store_action="resume"))
    assert store.claim("alice", "first", {})


def test_migration_refuses_live_server_and_requires_offline_state(maintenance, tmp_path):
    runtime, _, _ = maintenance
    (runtime / "server.pid").write_text(str(os.getpid()))
    with pytest.raises(RuntimeError):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "backup.db"))
    assert not (tmp_path / "backup.db").exists()


def test_export_uses_current_state_never_old_completed_descriptor(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    for task_id in ("finished", "ready", "uncertain"):
        seed(store, task_id)
    finished = store.claim("alice", "finished", {})
    assert store.finish(finished["_attempt_token"], status="done", stop_kind="completed")
    uncertain = store.claim("alice", "uncertain", {})
    assert store.finish(uncertain["_attempt_token"], status="pending", stop_kind="interrupted")
    # A stale descriptor on disk must not become runnable in the rollback view.
    pending = anima_dir / "state/pending"
    pending.mkdir(exist_ok=True)
    (pending / "finished.json").write_text(json.dumps({"task_type": "llm", "task_id": "finished"}))
    destination = tmp_path / "snapshot"
    result = run_maintenance(argparse.Namespace(anima="alice", task_store_action="export", destination=destination))
    assert result["claims_remain_paused"] is True
    exported = destination / "animas/alice/state"
    assert [p.stem for p in (exported / "pending").glob("*.json")] == ["ready"]
    payload = json.loads((exported / "pending/ready.json").read_text())
    assert payload["model"] == "openai/fixture"
    assert payload["acceptance_criteria"] == ["No duplicate effect"]
    ledger = {
        v["task_id"]: v for line in (exported / "task_queue.jsonl").read_text().splitlines() if (v := json.loads(line))
    }
    assert ledger["finished"]["status"] == "done"
    assert ledger["uncertain"]["status"] == "pending"
    assert (exported / "task_inputs_export.json").exists()
    # Import the current export into a separate DB, as a rollback/reupgrade rehearsal.
    recovered = TaskStore(tmp_path / "recovered.sqlite3")
    recovered.import_legacy(destination / "animas/alice")
    assert [p["task_id"] for p in recovered.pending("alice")] == ["ready"]


def test_export_rejects_active_attempt_and_leaves_claims_paused(maintenance, tmp_path):
    _, _, store = maintenance
    seed(store, "active")
    assert store.claim("alice", "active", {})
    with pytest.raises(RuntimeError):
        run_maintenance(
            argparse.Namespace(anima="alice", task_store_action="export", destination=tmp_path / "snapshot")
        )
    assert store.maintenance_status("alice")["quiesced"]
    assert not (tmp_path / "snapshot").exists()


def test_migration_rejects_conflicting_legacy_inputs(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    (anima_dir / "state/task_queue.jsonl").write_text(entry("work", "pending").model_dump_json() + "\n")
    pending = anima_dir / "state/pending"
    (pending / "failed").mkdir(parents=True)
    for path, description in ((pending / "work.json", "new input"), (pending / "failed/work.json", "old input")):
        path.write_text(json.dumps({"task_type": "llm", "task_id": "work", "description": description}))
    with pytest.raises(RuntimeError, match="conflicting legacy"):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "backup.db"))
    assert store.read("alice") == {}
    with store.reader() as db:
        assert db.execute("SELECT count(*) FROM task_imports").fetchone()[0] == 0


def test_backup_includes_wal_and_refuses_overwrite(maintenance, tmp_path):
    _, _, store = maintenance
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        seed(store, "wal-task")
        destination = tmp_path / "backup.sqlite3"
        store.backup(destination)
    assert (
        TaskStore(destination).read("alice")["wal-task"].original_instruction == entry("wal-task").original_instruction
    )
    with pytest.raises(FileExistsError):
        store.backup(destination)


def test_invalid_legacy_rows_rollback_import(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    (anima_dir / "state/task_queue.jsonl").write_text(json.dumps(entry("valid").model_dump()) + "\ninvalid\n")
    with pytest.raises(ValueError):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db"))
    assert store.read("alice") == {}
    assert store.maintenance_status("alice")["quiesced"]
    assert (tmp_path / "before.db").exists()


def test_migration_preserves_full_input_and_wakes_ambiguous_work(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    legacy = entry("ambiguous", "in_progress").model_dump()
    legacy["original_instruction"] = "Truncated"
    (anima_dir / "state/task_queue.jsonl").write_text(json.dumps(legacy) + "\n")
    pending = anima_dir / "state/pending"
    pending.mkdir(exist_ok=True)
    (pending / "ambiguous.json").write_text(
        json.dumps(
            {
                "task_type": "llm",
                "task_id": "ambiguous",
                "description": "Full source instruction including acceptance criteria.",
                "model": "openai/fixture",
            }
        )
    )
    report = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert report["migration"]["review_required"] == 1
    migrated = store.read("alice")["ambiguous"]
    assert migrated.original_instruction == "Full source instruction including acceptance criteria."
    assert migrated.meta["legacy_original_instruction"] == "Truncated"
    assert not store.pending("alice")
    assert store.wakeups("alice")[0]["reason"] == "migration_review"


@pytest.mark.parametrize("lease", [None, "malformed"])
def test_migration_refuses_unknown_processing_lease(maintenance, tmp_path, lease):
    _, anima_dir, _ = maintenance
    processing = anima_dir / "state/pending/processing"
    processing.mkdir(parents=True)
    descriptor = processing / "unknown.json"
    descriptor.write_text(json.dumps({"task_type": "llm", "task_id": "unknown", "description": "unconfirmed"}))
    if lease:
        descriptor.with_suffix(".json.lease").write_text(lease)
    with pytest.raises(RuntimeError):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db"))


@pytest.mark.parametrize("legacy_status", ["failed", "done", "cancelled", "completed"])
def test_archived_terminal_tasks_keep_history_without_review_or_runnable_input(maintenance, tmp_path, legacy_status):
    _, anima_dir, store = maintenance
    archived = entry("historical", legacy_status).model_dump()
    ledger = anima_dir / "state/task_queue_archive.jsonl"
    ledger.write_text(json.dumps(archived) + "\n")
    originals = {ledger: ledger.read_bytes()}
    # Two genuinely different historical inputs must not be arbitrarily selected.
    for container, description in (("failed", "first historical instruction"), ("suppressed", "different instruction")):
        descriptor = anima_dir / "state/pending" / container / "historical.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text(json.dumps({"task_type": "llm", "task_id": "historical", "description": description}))
        originals[descriptor] = descriptor.read_bytes()

    result = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert result["migration"]["terminal_descriptors"] == 2
    assert result["migration"]["review_required"] == 0
    assert result["migration"]["invalid_rows"] == 0
    assert store.read("alice") == {}
    history = store.read("alice", archived=True)["historical"]
    assert history.status == ("done" if legacy_status == "completed" else legacy_status)
    assert history.original_instruction == archived["original_instruction"]
    assert not history.meta.get("migration_review")
    assert store.get_input("alice", "historical") is None
    assert store.pending("alice") == []
    assert store.wakeups("alice") == []
    with store.reader() as db:
        row = db.execute("SELECT archived,ready,current_attempt FROM tasks WHERE task_id='historical'").fetchone()
        assert tuple(row) == (1, 0, None)
    for path, contents in originals.items():
        assert path.read_bytes() == contents


def test_active_ledger_redefinition_of_archived_failed_task_still_requires_review(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    (anima_dir / "state/task_queue_archive.jsonl").write_text(entry("reopened", "failed").model_dump_json() + "\n")
    active = entry("reopened", "failed").model_dump()
    active["original_instruction"] = "Explicitly redefined current instruction."
    (anima_dir / "state/task_queue.jsonl").write_text(json.dumps(active) + "\n")
    result = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert result["migration"]["review_required"] == 1
    task = store.read("alice")["reopened"]
    assert task.status == "pending"
    assert task.original_instruction == active["original_instruction"]
    assert task.meta["migration_review"] is True
    assert len(store.wakeups("alice")) == 1
    assert store.wakeups("alice")[0]["reason"] == "migration_review"
    assert store.pending("alice") == []


@pytest.mark.parametrize("legacy_status", ["done", "cancelled", "completed"])
def test_current_terminal_ledger_prevents_conflicting_descriptor_replay(maintenance, tmp_path, legacy_status):
    _, anima_dir, store = maintenance
    (anima_dir / "state/task_queue.jsonl").write_text(entry("terminal", legacy_status).model_dump_json() + "\n")
    originals = {}
    for relative, description in (("terminal.json", "newer file"), ("failed/terminal.json", "older file")):
        path = anima_dir / "state/pending" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task_type": "llm", "task_id": "terminal", "description": description}))
        originals[path] = path.read_bytes()
    result = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert result["migration"]["terminal_descriptors"] == 2
    assert result["migration"]["review_required"] == 0
    assert store.read("alice", archived=True)["terminal"].status == (
        "done" if legacy_status == "completed" else legacy_status
    )
    assert store.get_input("alice", "terminal") is None
    assert store.pending("alice") == []
    assert store.wakeups("alice") == []
    for path, contents in originals.items():
        assert path.read_bytes() == contents


def test_migration_ignores_arbitrary_evidence_subdirectories_not_task_containers(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    pending = anima_dir / "state/pending"
    evidence = {
        "review-evidence/reviews.json": "upstream request failed, not JSON",
        "review-evidence/work.json": json.dumps(
            {"task_type": "llm", "task_id": "evidence-only", "description": "Never execute evidence."}
        ),
        "failed/review-evidence/response.json": "",
    }
    originals = {}
    for relative, content in evidence.items():
        path = pending / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        originals[path] = path.read_bytes()
    live = pending / "live.json"
    live.write_text(json.dumps({"task_type": "llm", "task_id": "live", "description": "Current instruction."}))
    result = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert result["migration"]["ignored_artifacts"] == 3
    assert result["migration"]["invalid_rows"] == 0
    assert result["migration"]["descriptors"] == 1
    assert set(store.read("alice")) == {"live"}
    assert [item["task_id"] for item in store.pending("alice")] == ["live"]
    for path, contents in originals.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("container", ["", "failed", "suppressed"])
def test_malformed_json_in_real_task_containers_still_rolls_back(maintenance, tmp_path, container):
    _, anima_dir, store = maintenance
    path = anima_dir / "state/pending" / container / "malformed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("invalid JSON must not be silently skipped")
    with pytest.raises(ValueError):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db"))
    assert store.read("alice", archived=True) == {}
    assert store.maintenance_status("alice")["quiesced"]
    with store.reader() as db:
        assert db.execute("SELECT count(*) FROM task_imports").fetchone()[0] == 0


def test_known_completed_heartbeat_note_is_counted_and_preserved_not_migrated(maintenance, tmp_path):
    _, anima_dir, store = maintenance
    note = {
        "id": "historical-heartbeat-note",
        "type": "heartbeat",
        "status": "done",
        "created_at": "2026-07-24T19:20+09:00",
        "summary": "Historical observation, not executable work.",
    }
    ledger = anima_dir / "state/task_queue.jsonl"
    ledger.write_text(json.dumps(note) + "\n" + entry("ordinary", "done").model_dump_json() + "\n")
    original = ledger.read_bytes()
    result = run_maintenance(
        argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db")
    )
    assert result["migration"]["non_task_rows"] == 1
    assert result["migration"]["invalid_rows"] == 0
    assert result["migration"]["tasks"] == 1
    assert set(store.read("alice", archived=True)) == {"ordinary"}
    assert store.wakeups("alice") == []
    assert ledger.read_bytes() == original


@pytest.mark.parametrize(
    "difference",
    [{"type": "task"}, {"status": "pending"}, {"id": None}, {"created_at": None}],
)
def test_missing_task_id_is_not_broadly_exempted_as_a_non_task_note(maintenance, tmp_path, difference):
    _, anima_dir, store = maintenance
    note = {"id": "note", "type": "heartbeat", "status": "done", "created_at": "2026-07-24T19:20+09:00"}
    note.update(difference)
    (anima_dir / "state/task_queue.jsonl").write_text(json.dumps(note) + "\n")
    with pytest.raises(ValueError):
        run_maintenance(argparse.Namespace(anima="alice", task_store_action="migrate", backup=tmp_path / "before.db"))
    assert store.read("alice", archived=True) == {}
    with store.reader() as db:
        assert db.execute("SELECT count(*) FROM task_imports").fetchone()[0] == 0
