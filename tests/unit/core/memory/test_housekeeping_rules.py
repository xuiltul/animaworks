from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from core.config.models import HousekeepingConfig, InboxConfig
from core.memory.maintenance import housekeeping as current_housekeeping
from core.time_utils import now_local

# main/HEAD before any P3-4 edits; use the immutable commit so the differential
# check remains meaningful after this branch is merged and main advances.
# One-time migration guard: skipped when the commit is not available (shallow
# CI clones); delete it once housekeeping intentionally changes behavior.
_BASELINE_REVISION = "c9ced56b430817ff1509036d43ebe132d3aaef34"
_BASELINE_FILE = "core/memory/maintenance/housekeeping.py"


def _load_pre_refactor_housekeeping() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[4]
    try:
        source = subprocess.run(
            ["git", "show", f"{_BASELINE_REVISION}:{_BASELINE_FILE}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"baseline revision {_BASELINE_REVISION[:8]} is not available in this checkout")
    module = ModuleType("housekeeping_before_p3_4")
    module.__file__ = f"{_BASELINE_REVISION}:{_BASELINE_FILE}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _write_file(root: Path, relative_path: str, content: bytes, *, age_days: float | None = None) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if age_days is not None:
        timestamp = time.time() - age_days * 86_400
        os.utime(path, (timestamp, timestamp))
    return path


def _set_age(path: Path, age_days: float) -> None:
    timestamp = time.time() - age_days * 86_400
    os.utime(path, (timestamp, timestamp))


def _populate_cleanup_case(root: Path, *, now: datetime) -> list[Path]:
    old_superseded: list[Path] = []
    today = now.date()

    # Filename-date retention and mtime retention.
    prompt_logs = root / "animas/alice/prompt_logs"
    prompt_logs.mkdir(parents=True)
    (prompt_logs / f"{(today - timedelta(days=10)).isoformat()}.jsonl").write_text("old\n")
    (prompt_logs / f"{today.isoformat()}.jsonl").write_text("recent\n")

    cron_logs = root / "animas/alice/state/cron_logs"
    cron_logs.mkdir(parents=True)
    (cron_logs / f"{(today - timedelta(days=20)).isoformat()}.jsonl").write_text("old\n")
    (cron_logs / f"{(today - timedelta(days=2)).isoformat()}.jsonl").write_text("recent\n")

    curator = root / "animas/alice/state/skill_curator"
    curator.mkdir(parents=True)
    (curator / f"report-{(today - timedelta(days=40)).isoformat()}.json").write_text("old\n")
    (curator / f"report-{today.isoformat()}.json").write_text("recent\n")

    task_results = root / "animas/alice/state/task_results"
    _write_file(root, "animas/alice/state/task_results/old.md", b"old", age_days=10)
    _write_file(root, "animas/alice/state/task_results/recent.md", b"recent")
    assert task_results.is_dir()

    # Shortterm has independent archive retention and stale-thread collection.
    _write_file(root, "animas/alice/shortterm/chat/old.json", b"old", age_days=14)
    _write_file(root, "animas/alice/shortterm/chat/recent.json", b"recent")
    _write_file(root, "animas/alice/shortterm/chat/archive/old.json", b"old archive", age_days=31)
    _write_file(root, "animas/alice/shortterm/chat/archive/recent.json", b"recent archive", age_days=14)
    _write_file(root, "animas/alice/shortterm/chat/stale-thread/old.json", b"old thread", age_days=31)
    _write_file(root, "animas/alice/shortterm/chat/active-thread/current_session_chat.json", b"keep")

    # A stale facts lock is deletable only when empty and not held by another process.
    _write_file(root, "animas/alice/facts/stale.lock", b"", age_days=2)
    _write_file(root, "animas/alice/facts/recent.lock", b"")
    _write_file(root, "animas/alice/facts/nonempty.lock", b"locked", age_days=2)

    # Keep the newest timestamped corrupt vector DB generations.
    for stamp in ("20260101-010101", "20260201-010101", "20260301-010101"):
        _write_file(root, f"animas/alice/archive/vectordb-corrupt-{stamp}/data.bin", stamp.encode())

    # Size/copytruncate rotations, including existing numbered generations.
    _write_file(root, "logs/server-daemon.log", b"D" * (1024 * 1024 + 1))
    _write_file(root, "logs/server-daemon.log.1", b"daemon gen 1")
    _write_file(root, "logs/server-daemon.log.2", b"daemon gen 2")
    _write_file(root, "logs/server-daemon.log.3", b"daemon gen 3")
    _write_file(root, "animas/alice/state/suppressed_messages.jsonl", b"S" * (1024 * 1024 + 1))
    _write_file(root, "logs/sdk_bash_injection.jsonl", b"I" * (1024 * 1024 + 1))

    # Per-Anima log retention and total-size cap; active targets are protected.
    anima_logs = root / "logs/animas/alice"
    anima_logs.mkdir(parents=True)
    dated_old = anima_logs / f"{(today - timedelta(days=60)).strftime('%Y%m%d')}.log"
    dated_old.write_text("old dated log")
    current = anima_logs / f"{today.strftime('%Y%m%d')}.log"
    current.write_text("active log")
    stderr = anima_logs / "stderr.log"
    stderr.write_text("stderr")
    first = anima_logs / f"{(today - timedelta(days=2)).strftime('%Y%m%d')}.log"
    second = anima_logs / f"{(today - timedelta(days=1)).strftime('%Y%m%d')}.log"
    first.write_bytes(b"a" * 700_000)
    second.write_bytes(b"b" * 700_000)
    _set_age(first, 2)
    _set_age(second, 1)
    current_link = anima_logs / "current.log"
    try:
        current_link.symlink_to(current.name)
    except OSError:
        current_link.write_text(current.name)

    # Frontend backups are retained by filename order, not mtime.
    frontend = root / "logs/frontend"
    frontend.mkdir(parents=True)
    (frontend / "frontend.jsonl").write_text("active")
    for name in (
        "frontend.jsonl.20260610",
        "frontend.jsonl.20260609",
        "frontend.jsonl.20260301",
        "20260228.jsonl",
    ):
        (frontend / name).write_text(name)

    # Runtime tmp: remove old entries but keep recent descendants and reserved roots.
    _write_file(root, "tmp/old.tmp", b"old", age_days=20)
    _write_file(root, "tmp/recent.tmp", b"recent")
    _write_file(root, "tmp/attachments/old.bin", b"old", age_days=20)
    _write_file(root, "tmp/attachments/recent.bin", b"recent")
    _write_file(root, "tmp/old-dir-with-recent-child/recent.txt", b"recent")
    _set_age(root / "tmp/old-dir-with-recent-child", 20)

    # Only immediate Anima backup directories match this rule.
    old_backup = root / "animas/alice/assets_backup_old"
    old_backup.mkdir(parents=True)
    (old_backup / "asset.bin").write_bytes(b"old")
    _set_age(old_backup, 120)
    (root / "animas/alice/assets_backup_recent").mkdir()
    nested_backup = root / "animas/alice/knowledge/assets_backup_old"
    nested_backup.mkdir(parents=True)
    _set_age(nested_backup, 120)

    # Codex log DB bundles are removed as a unit; state/session DBs survive.
    _write_file(root, "animas/alice/.codex_home/logs_2.sqlite", b"L" * (2 * 1024 * 1024 + 1))
    _write_file(root, "animas/alice/.codex_home/logs_2.sqlite-wal", b"wal")
    _write_file(root, "animas/alice/.codex_home/logs_2.sqlite-shm", b"shm")
    _write_file(root, "animas/alice/.codex_home/state_5.sqlite", b"state")
    _write_file(root, "animas/alice/.codex_home/sessions/rollout.jsonl", b"session")

    # Codex tmp and per-Anima temporary git/log artifacts use age and open-file guards.
    _write_file(root, "animas/alice/.codex_home/.tmp/plugins/plugin.json", b"old", age_days=1)
    _set_age(root / "animas/alice/.codex_home/.tmp/plugins", 1)
    _write_file(root, "animas/alice/.codex_home/.tmp/recent/keep.txt", b"recent")
    _write_file(root, "animas/alice/tmp_gitdirs/work.git/pack", b"git temp", age_days=40)
    _set_age(root / "animas/alice/tmp_gitdirs/work.git", 40)
    _write_file(root, "animas/alice/logs/old.log", b"old", age_days=40)
    _write_file(root, "animas/alice/logs/recent.log", b"recent")

    # TaskBoard cleanup is state-aware and archives non-idle current state.
    running = root / "animas/alice/state/background_tasks/running.json"
    running.parent.mkdir(parents=True, exist_ok=True)
    running.write_text(json.dumps({"status": "running", "created_at": time.time() - 2 * 3600}))
    current_state = _write_file(root, "animas/alice/state/current_state.md", b"status: working\nnotes\n")
    _set_age(current_state, 2 / 24)

    # Archive/superseded uses ctime as the archive-entry time when it is newer than mtime.
    superseded = root / "animas/alice/archive/superseded"
    old_path = superseded / "old.md"
    _write_file(root, "animas/alice/archive/superseded/old.md", b"old", age_days=60)
    _write_file(root, "animas/alice/archive/superseded/recent.md", b"recent")
    old_superseded.append(old_path)

    # Archive versions are grouped by source and suffix, with filename timestamp ordering.
    _write_file(root, "animas/alice/archive/versions/knowledge__topic_v1_20260101_010101.md", b"v1")
    _write_file(root, "animas/alice/archive/versions/knowledge__topic_v2_20260201_010101.md", b"v2")

    # Shared inbox uses Messenger's TTL/state transitions plus per-state retention.
    from core.schemas import Message

    inbox = root / "shared/inbox/alice"
    inbox.mkdir(parents=True)
    old_message = Message(
        id="old-message",
        from_person="bob",
        to_person="alice",
        content="expired",
        timestamp=now - timedelta(hours=30),
    )
    protected_message = Message(
        id="delegation-message",
        from_person="bob",
        to_person="alice",
        content="delegation",
        timestamp=now - timedelta(hours=30),
        intent="delegation",
    )
    (inbox / "old.json").write_text(old_message.model_dump_json(), encoding="utf-8")
    (inbox / "delegation.json").write_text(protected_message.model_dump_json(), encoding="utf-8")
    _write_file(root, "shared/inbox/alice/processed/old.json", b"{}", age_days=40)
    _write_file(root, "shared/inbox/alice/quarantine/old.json", b"{}", age_days=40)

    return old_superseded


def _snapshot_files(root: Path) -> dict[str, tuple[str, bytes | str]]:
    snapshot: dict[str, tuple[str, bytes | str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _replace_ctime(stat_result: os.stat_result, ctime: float) -> os.stat_result:
    return os.stat_result(
        (
            stat_result.st_mode,
            stat_result.st_ino,
            stat_result.st_dev,
            stat_result.st_nlink,
            stat_result.st_uid,
            stat_result.st_gid,
            stat_result.st_size,
            stat_result.st_atime,
            stat_result.st_mtime,
            ctime,
        )
    )


def test_housekeeping_table_resolves_every_declared_setting() -> None:
    config = HousekeepingConfig()
    inbox = InboxConfig()

    assert current_housekeeping._HOUSEKEEPING_RULES
    assert len({rule.result_key for rule in current_housekeeping._HOUSEKEEPING_RULES}) == len(
        current_housekeeping._HOUSEKEEPING_RULES
    )
    for rule in current_housekeeping._HOUSEKEEPING_RULES:
        assert rule.path and rule.pattern and rule.method
        assert rule.method == "delete_old_files" or rule.method in current_housekeeping._HOUSEKEEPING_HANDLERS
        settings = inbox if rule.config_section == "inbox" else config
        assert all(hasattr(settings, key) for key in rule.settings)


@pytest.mark.asyncio
async def test_housekeeping_matches_pre_refactor_file_and_rotation_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare synthetic dry-runs with the immutable pre-P3-4 source snapshot."""
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    before_root.mkdir()
    after_root.mkdir()
    reference_now = now_local()
    old_paths = _populate_cleanup_case(before_root, now=reference_now) + _populate_cleanup_case(
        after_root, now=reference_now
    )

    real_stat = Path.stat
    old_ctime = time.time() - 90 * 86_400
    resolved_old_paths = {os.path.realpath(path) for path in old_paths}

    def stat_with_old_archive_ctime(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)
        if os.path.realpath(path) in resolved_old_paths:
            return _replace_ctime(result, old_ctime)
        return result

    monkeypatch.setattr(Path, "stat", stat_with_old_archive_ctime)

    before_module = _load_pre_refactor_housekeeping()

    def skip_report_generation(_animas_dir: Path) -> dict[str, int]:
        return {"generated_reports": 0}

    monkeypatch.setattr(before_module, "_run_skill_curator_reports", skip_report_generation)
    monkeypatch.setitem(current_housekeeping._HOUSEKEEPING_HANDLERS, "skill_curator", skip_report_generation)

    from core.tasks.board import housekeeping as taskboard_housekeeping

    monkeypatch.setattr(taskboard_housekeeping, "_has_active_visible_task", lambda *_args: False)

    config = HousekeepingConfig(
        prompt_log_retention_days=3,
        daemon_log_max_size_mb=1,
        daemon_log_keep_generations=2,
        anima_log_retention_days=30,
        anima_log_total_max_size_mb=1,
        frontend_log_backup_count=2,
        cron_log_retention_days=30,
        shortterm_retention_days=7,
        shortterm_archive_retention_days=30,
        shortterm_thread_gc_days=30,
        facts_lock_stale_hours=24,
        curator_report_retention_days=30,
        task_results_retention_days=7,
        corrupt_vectordb_keep_generations=2,
        tmp_retention_days=14,
        backup_retention_days=90,
        codex_log_max_size_mb=1,
        codex_tmp_retention_hours=12,
        anima_tmp_gitdirs_retention_days=14,
        anima_local_log_retention_days=30,
        pending_processing_stale_hours=24,
        background_running_stale_hours=1,
        current_state_stale_hours=1,
        suppressed_messages_max_size_mb=1,
        suppressed_messages_keep_generations=2,
        sdk_bash_injection_max_size_mb=1,
        archive_superseded_retention_days=7,
        archive_versions_keep_per_file=1,
    )
    inbox_config = InboxConfig()

    before_results = await before_module.run_housekeeping(before_root, housekeeping=config, inbox=inbox_config)
    before_results.pop("pending_failed", None)  # Retired legacy command-descriptor cleanup.
    after_results = await current_housekeeping.run_housekeeping(after_root, housekeeping=config, inbox=inbox_config)
    for retired_key in before_results.keys() - after_results.keys():
        before_results.pop(retired_key, None)
    for result_key in current_housekeeping._WORKSPACE_SWEEP_RULE_KEYS:
        after_results.pop(result_key, None)  # These rules are covered against the pre-refactor snapshot separately.

    assert before_results == after_results
    before_files = _snapshot_files(before_root)
    after_files = _snapshot_files(after_root)
    assert before_files.keys() == after_files.keys()
    assert before_files == after_files
