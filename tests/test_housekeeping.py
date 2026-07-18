from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for core.memory.housekeeping and related modules."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from core.time_utils import now_local, today_local

# ── HousekeepingConfig tests ────────────────────────────────────


class TestHousekeepingConfig:
    """Test HousekeepingConfig defaults and customization."""

    def test_default_values(self):
        from core.config.models import HousekeepingConfig

        cfg = HousekeepingConfig()
        assert cfg.enabled is True
        assert cfg.run_time == "05:30"
        assert cfg.prompt_log_retention_days == 3
        assert cfg.daemon_log_max_size_mb == 50
        assert cfg.daemon_log_keep_generations == 5
        assert cfg.anima_log_retention_days == 30
        assert cfg.anima_log_total_max_size_mb == 200
        assert cfg.frontend_log_backup_count == 7
        assert cfg.dm_log_archive_retention_days == 30
        assert cfg.cron_log_retention_days == 30
        assert cfg.shortterm_retention_days == 7
        assert cfg.shortterm_archive_retention_days == 30
        assert cfg.shortterm_thread_gc_days == 30
        assert cfg.facts_lock_stale_hours == 24
        assert cfg.task_results_retention_days == 7
        assert cfg.pending_failed_retention_days == 14
        assert cfg.corrupt_vectordb_keep_generations == 2
        assert cfg.tmp_retention_days == 14
        assert cfg.backup_retention_days == 90
        assert cfg.codex_log_max_size_mb == 200
        assert cfg.codex_tmp_retention_hours == 12
        assert cfg.anima_tmp_gitdirs_retention_days == 14
        assert cfg.anima_local_log_retention_days == 30
        assert cfg.suppressed_messages_max_size_mb == 10
        assert cfg.suppressed_messages_keep_generations == 5

    def test_custom_values(self):
        from core.config.models import HousekeepingConfig

        cfg = HousekeepingConfig(
            enabled=False,
            run_time="03:00",
            prompt_log_retention_days=7,
            daemon_log_max_size_mb=200,
            daemon_log_keep_generations=3,
            anima_log_retention_days=14,
            anima_log_total_max_size_mb=100,
            frontend_log_backup_count=14,
            dm_log_archive_retention_days=60,
            cron_log_retention_days=14,
            shortterm_retention_days=14,
            shortterm_archive_retention_days=60,
            shortterm_thread_gc_days=45,
            facts_lock_stale_hours=12,
            corrupt_vectordb_keep_generations=4,
            tmp_retention_days=21,
            backup_retention_days=120,
            codex_log_max_size_mb=128,
            codex_tmp_retention_hours=24,
            anima_tmp_gitdirs_retention_days=21,
            anima_local_log_retention_days=45,
            suppressed_messages_max_size_mb=20,
            suppressed_messages_keep_generations=2,
        )
        assert cfg.enabled is False
        assert cfg.run_time == "03:00"
        assert cfg.prompt_log_retention_days == 7
        assert cfg.daemon_log_max_size_mb == 200
        assert cfg.anima_log_retention_days == 14
        assert cfg.anima_log_total_max_size_mb == 100
        assert cfg.shortterm_archive_retention_days == 60
        assert cfg.shortterm_thread_gc_days == 45
        assert cfg.facts_lock_stale_hours == 12
        assert cfg.corrupt_vectordb_keep_generations == 4
        assert cfg.tmp_retention_days == 21
        assert cfg.backup_retention_days == 120
        assert cfg.codex_log_max_size_mb == 128
        assert cfg.codex_tmp_retention_hours == 24
        assert cfg.anima_tmp_gitdirs_retention_days == 21
        assert cfg.anima_local_log_retention_days == 45
        assert cfg.suppressed_messages_max_size_mb == 20
        assert cfg.suppressed_messages_keep_generations == 2

    def test_config_has_housekeeping_field(self):
        from core.config.models import AnimaWorksConfig

        cfg = AnimaWorksConfig()
        assert hasattr(cfg, "housekeeping")
        assert cfg.housekeeping.enabled is True

    def test_config_json_round_trip(self):
        from core.config.models import AnimaWorksConfig

        cfg = AnimaWorksConfig()
        data = cfg.model_dump()
        assert "housekeeping" in data
        assert data["housekeeping"]["enabled"] is True
        assert data["housekeeping"]["run_time"] == "05:30"

        restored = AnimaWorksConfig(**data)
        assert restored.housekeeping.prompt_log_retention_days == 3

    @pytest.mark.asyncio
    async def test_run_housekeeping_rotates_vector_worker_log(self, tmp_path: Path):
        from core.memory.housekeeping import run_housekeeping

        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "vector-worker.log").write_bytes(b"x" * 2048)

        results = await run_housekeeping(
            tmp_path,
            daemon_log_max_size_mb=0,
            daemon_log_keep_generations=5,
        )

        assert results["vector_worker_log"]["rotated"] is True
        assert (logs / "vector-worker.log.1").exists()

    @pytest.mark.asyncio
    async def test_run_housekeeping_rotates_suppressed_messages_log(self, tmp_path: Path):
        from core.memory.housekeeping import run_housekeeping

        state_dir = tmp_path / "animas" / "alice" / "state"
        state_dir.mkdir(parents=True)
        suppressed = state_dir / "suppressed_messages.jsonl"
        suppressed.write_bytes(b"x" * 2048)

        results = await run_housekeeping(
            tmp_path,
            suppressed_messages_max_size_mb=0,
            suppressed_messages_keep_generations=2,
        )

        assert results["suppressed_messages"]["files"] == 1
        assert results["suppressed_messages"]["rotated"] == 1
        assert (state_dir / "suppressed_messages.jsonl.1").exists()


class TestInboxConfig:
    """Test InboxConfig defaults and config roundtrip."""

    def test_default_values(self):
        from core.config.models import InboxConfig

        cfg = InboxConfig()
        assert cfg.ttl_hours == 24.0
        assert cfg.expired_retention_days == 7
        assert cfg.processed_retention_days == 30
        assert cfg.quarantine_retention_days == 30

    def test_config_json_round_trip(self):
        from core.config.models import AnimaWorksConfig

        cfg = AnimaWorksConfig()
        data = cfg.model_dump()
        assert "inbox" in data
        assert data["inbox"]["ttl_hours"] == 24.0

        data["inbox"]["ttl_hours"] = 12.0
        data["inbox"]["expired_retention_days"] = 3
        restored = AnimaWorksConfig(**data)
        assert restored.inbox.ttl_hours == 12.0
        assert restored.inbox.expired_retention_days == 3


# ── rotate_all_prompt_logs tests ────────────────────────────────


class TestRotateAllPromptLogs:
    """Test the rotate_all_prompt_logs function."""

    def test_deletes_old_logs(self, tmp_path: Path):
        from core._agent_prompt_log import rotate_all_prompt_logs

        anima_dir = tmp_path / "alice"
        log_dir = anima_dir / "prompt_logs"
        log_dir.mkdir(parents=True)

        old_date = (today_local() - timedelta(days=5)).isoformat()
        today = today_local().isoformat()
        (log_dir / f"{old_date}.jsonl").write_text("{}\n")
        (log_dir / f"{today}.jsonl").write_text("{}\n")

        result = rotate_all_prompt_logs(tmp_path, retention_days=3)
        assert "alice" in result
        assert result["alice"] == 1
        assert not (log_dir / f"{old_date}.jsonl").exists()
        assert (log_dir / f"{today}.jsonl").exists()

    def test_skips_non_directories(self, tmp_path: Path):
        from core._agent_prompt_log import rotate_all_prompt_logs

        (tmp_path / "not_a_dir.txt").write_text("hello")
        result = rotate_all_prompt_logs(tmp_path, retention_days=3)
        assert result == {}

    def test_skips_anima_without_prompt_logs(self, tmp_path: Path):
        from core._agent_prompt_log import rotate_all_prompt_logs

        (tmp_path / "bob").mkdir()
        result = rotate_all_prompt_logs(tmp_path, retention_days=3)
        assert result == {}

    def test_no_old_files(self, tmp_path: Path):
        from core._agent_prompt_log import rotate_all_prompt_logs

        anima_dir = tmp_path / "charlie"
        log_dir = anima_dir / "prompt_logs"
        log_dir.mkdir(parents=True)
        today = today_local().isoformat()
        (log_dir / f"{today}.jsonl").write_text("{}\n")

        result = rotate_all_prompt_logs(tmp_path, retention_days=3)
        assert result == {}

    def test_multiple_animas(self, tmp_path: Path):
        from core._agent_prompt_log import rotate_all_prompt_logs

        old_date = (today_local() - timedelta(days=10)).isoformat()

        for name in ("a1", "a2"):
            d = tmp_path / name / "prompt_logs"
            d.mkdir(parents=True)
            (d / f"{old_date}.jsonl").write_text("{}\n")

        result = rotate_all_prompt_logs(tmp_path, retention_days=3)
        assert len(result) == 2
        assert result["a1"] == 1
        assert result["a2"] == 1


# ── _rotate_daemon_log tests ───────────────────────────────────


class TestRotateDaemonLog:
    """Test daemon log rotation."""

    def test_skips_when_file_not_found(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        result = _rotate_daemon_log(tmp_path / "nonexistent.log", 100, 5)
        assert result["skipped"] is True
        assert result["reason"] == "file_not_found"

    def test_skips_when_under_size(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        log = tmp_path / "server-daemon.log"
        log.write_text("small content")
        result = _rotate_daemon_log(log, 100, 5)
        assert result["skipped"] is True

    def test_prunes_generations_when_current_under_size(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        log = tmp_path / "server-daemon.log"
        log.write_text("small content")
        (tmp_path / "server-daemon.log.1").write_text("gen1")
        (tmp_path / "server-daemon.log.2").write_text("gen2")
        (tmp_path / "server-daemon.log.3").write_text("gen3")

        result = _rotate_daemon_log(log, max_size_mb=100, keep_generations=2)

        assert result["skipped"] is True
        assert result["deleted_generations"] == 1
        assert (tmp_path / "server-daemon.log.1").exists()
        assert (tmp_path / "server-daemon.log.2").exists()
        assert not (tmp_path / "server-daemon.log.3").exists()

    def test_rotates_when_over_size(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        log = tmp_path / "server-daemon.log"
        log.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        result = _rotate_daemon_log(log, max_size_mb=1, keep_generations=3)
        assert result["rotated"] is True
        assert log.exists()  # copytruncate keeps the live file descriptor target
        assert log.stat().st_size == 0
        assert (tmp_path / "server-daemon.log.1").exists()

    def test_shifts_existing_generations(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        log = tmp_path / "server-daemon.log"
        log.write_bytes(b"x" * (2 * 1024 * 1024))
        (tmp_path / "server-daemon.log.1").write_text("gen1")
        (tmp_path / "server-daemon.log.2").write_text("gen2")

        result = _rotate_daemon_log(log, max_size_mb=1, keep_generations=3)
        assert result["rotated"] is True
        assert (tmp_path / "server-daemon.log.1").exists()
        assert (tmp_path / "server-daemon.log.2").read_text() == "gen1"
        assert (tmp_path / "server-daemon.log.3").read_text() == "gen2"

    def test_deletes_over_limit_generations(self, tmp_path: Path):
        from core.memory.housekeeping import _rotate_daemon_log

        log = tmp_path / "server-daemon.log"
        log.write_bytes(b"x" * (2 * 1024 * 1024))
        (tmp_path / "server-daemon.log.1").write_text("gen1")
        (tmp_path / "server-daemon.log.2").write_text("gen2")

        result = _rotate_daemon_log(log, max_size_mb=1, keep_generations=2)
        assert result["rotated"] is True
        assert not (tmp_path / "server-daemon.log.3").exists()


# ── _cleanup_dm_archives tests ─────────────────────────────────


class TestCleanupDmArchives:
    """Test DM archive cleanup."""

    def test_skips_when_dir_not_found(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_dm_archives

        result = _cleanup_dm_archives(tmp_path / "nonexistent", 30)
        assert result["skipped"] is True

    def test_deletes_old_archives(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_dm_archives

        old_archive = tmp_path / "alice-bob.20260101.archive.jsonl"
        old_archive.write_text("{}\n")
        old_time = time.time() - (60 * 86400)  # 60 days ago
        os.utime(old_archive, (old_time, old_time))

        recent_archive = tmp_path / "alice-charlie.20260304.archive.jsonl"
        recent_archive.write_text("{}\n")

        result = _cleanup_dm_archives(tmp_path, retention_days=30)
        assert result["deleted_files"] == 1
        assert not old_archive.exists()
        assert recent_archive.exists()

    def test_ignores_non_archive_files(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_dm_archives

        normal = tmp_path / "alice-bob.jsonl"
        normal.write_text("{}\n")
        old_time = time.time() - (60 * 86400)
        os.utime(normal, (old_time, old_time))

        result = _cleanup_dm_archives(tmp_path, retention_days=30)
        assert result["deleted_files"] == 0
        assert normal.exists()


# ── _cleanup_cron_logs tests ───────────────────────────────────


class TestCleanupCronLogs:
    """Test cron log cleanup."""

    def test_skips_when_dir_not_found(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_cron_logs

        result = _cleanup_cron_logs(tmp_path / "nonexistent", 30)
        assert result["skipped"] is True

    def test_deletes_old_cron_logs(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_cron_logs

        cron_dir = tmp_path / "alice" / "state" / "cron_logs"
        cron_dir.mkdir(parents=True)

        old_date = (today_local() - timedelta(days=40)).isoformat()
        today = today_local().isoformat()
        (cron_dir / f"{old_date}.jsonl").write_text("{}\n")
        (cron_dir / f"{today}.jsonl").write_text("{}\n")

        result = _cleanup_cron_logs(tmp_path, retention_days=30)
        assert result["deleted_files"] == 1
        assert not (cron_dir / f"{old_date}.jsonl").exists()
        assert (cron_dir / f"{today}.jsonl").exists()

    def test_skips_anima_without_cron_logs(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_cron_logs

        (tmp_path / "bob").mkdir()
        result = _cleanup_cron_logs(tmp_path, retention_days=30)
        assert result["deleted_files"] == 0


# ── _cleanup_shortterm tests ───────────────────────────────────


class TestCleanupShortterm:
    """Test shortterm cleanup."""

    def test_skips_when_dir_not_found(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        result = _cleanup_shortterm(tmp_path / "nonexistent", 7)
        assert result["skipped"] is True

    def test_deletes_old_session_files(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        chat_dir = tmp_path / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)

        old_file = chat_dir / "2026-01-01_session.json"
        old_file.write_text("{}")
        old_time = time.time() - (14 * 86400)  # 14 days ago
        os.utime(old_file, (old_time, old_time))

        recent_file = chat_dir / "2026-03-04_session.json"
        recent_file.write_text("{}")

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_preserves_current_session_files(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        chat_dir = tmp_path / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)

        protected = chat_dir / "current_session_chat.json"
        protected.write_text("{}")
        old_time = time.time() - (30 * 86400)
        os.utime(protected, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 0
        assert protected.exists()

    def test_preserves_streaming_journal_files(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "thread-123"
        thread_dir.mkdir(parents=True)

        protected = thread_dir / "streaming_journal.jsonl"
        protected.write_text("{}")
        old_time = time.time() - (14 * 86400)
        os.utime(protected, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 0
        assert protected.exists()
        assert thread_dir.exists()

    def test_cleans_both_chat_and_heartbeat(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        old_time = time.time() - (14 * 86400)
        for sub in ("chat", "heartbeat"):
            d = tmp_path / "alice" / "shortterm" / sub
            d.mkdir(parents=True)
            f = d / "old_session.json"
            f.write_text("{}")
            os.utime(f, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 2

    def test_cleans_thread_subdirectories(self, tmp_path: Path):
        # F13(c): per-thread subdirectories must be swept too.
        from core.memory.housekeeping import _cleanup_shortterm

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "thread-xyz"
        thread_dir.mkdir(parents=True)
        old_file = thread_dir / "old_session.json"
        old_file.write_text("{}")
        old_time = time.time() - (14 * 86400)
        os.utime(old_file, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 1
        assert not old_file.exists()

    def test_archive_uses_separate_retention_window(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        archive_dir = tmp_path / "alice" / "shortterm" / "chat" / "archive"
        archive_dir.mkdir(parents=True)
        expired = archive_dir / "expired.json"
        retained = archive_dir / "retained.json"
        expired.write_text("{}")
        retained.write_text("{}")
        expired_time = time.time() - (31 * 86400)
        retained_time = time.time() - (14 * 86400)
        os.utime(expired, (expired_time, expired_time))
        os.utime(retained, (retained_time, retained_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7, archive_retention_days=30)
        assert result["deleted_files"] == 1
        assert result["archive_deleted"] == 1
        assert not expired.exists()
        assert retained.exists()

    @pytest.mark.parametrize("archive_retention_days", [0, -1])
    def test_non_positive_archive_retention_skips_archive_cleanup(
        self,
        tmp_path: Path,
        archive_retention_days: int,
    ):
        from core.memory.housekeeping import _cleanup_shortterm

        archive_dir = tmp_path / "alice" / "shortterm" / "chat" / "archive"
        archive_dir.mkdir(parents=True)
        archived = archive_dir / "old.json"
        archived.write_text("{}")
        old_time = time.time() - (90 * 86400)
        os.utime(archived, (old_time, old_time))

        result = _cleanup_shortterm(
            tmp_path,
            retention_days=7,
            archive_retention_days=archive_retention_days,
        )

        assert archived.exists()
        assert result["archive_deleted"] == 0
        assert result["skipped_substeps"]["archive_cleanup"] == "archive_retention_days_must_be_positive"

    def test_cleans_cron_inbox_and_task(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        old_time = time.time() - (14 * 86400)
        files = []
        for sub in ("cron", "inbox", "task"):
            sub_dir = tmp_path / "alice" / "shortterm" / sub
            sub_dir.mkdir(parents=True)
            old_file = sub_dir / "old.json"
            old_file.write_text("{}")
            os.utime(old_file, (old_time, old_time))
            files.append(old_file)

        result = _cleanup_shortterm(tmp_path, retention_days=7)

        assert result["deleted_files"] == 3
        assert result["deleted_by_subdir"] == {
            "chat": 0,
            "heartbeat": 0,
            "cron": 1,
            "inbox": 1,
            "task": 1,
        }
        assert all(not path.exists() for path in files)

    def test_deletes_stale_thread_directory(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "stale-thread"
        thread_dir.mkdir(parents=True)
        protected = thread_dir / "current_session_chat.json"
        protected.write_text("{}")
        old_time = time.time() - (31 * 86400)
        os.utime(protected, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7, thread_gc_days=30)

        assert result["thread_dirs_deleted"] == 1
        assert result["deleted_files"] == 1
        assert not thread_dir.exists()

    def test_recent_current_session_protects_thread_directory(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_shortterm

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "active-thread"
        thread_dir.mkdir(parents=True)
        protected = thread_dir / "current_session_chat.json"
        protected.write_text("{}")

        result = _cleanup_shortterm(tmp_path, retention_days=7, thread_gc_days=30)

        assert result["thread_dirs_deleted"] == 0
        assert thread_dir.exists()
        assert protected.exists()

    @pytest.mark.parametrize("thread_gc_days", [0, -1])
    def test_non_positive_thread_gc_skips_directory_cleanup(
        self,
        tmp_path: Path,
        thread_gc_days: int,
    ):
        from core.memory.housekeeping import _cleanup_shortterm

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "stale-thread"
        thread_dir.mkdir(parents=True)
        protected = thread_dir / "current_session_chat.json"
        protected.write_text("{}")
        old_time = time.time() - (90 * 86400)
        os.utime(protected, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7, thread_gc_days=thread_gc_days)

        assert thread_dir.exists()
        assert result["thread_dirs_deleted"] == 0
        assert result["skipped_substeps"]["thread_gc"] == "thread_gc_days_must_be_positive"

    def test_thread_gc_rechecks_mtime_before_rmtree(self, tmp_path: Path, monkeypatch):
        import core.memory.housekeeping as housekeeping

        thread_dir = tmp_path / "alice" / "shortterm" / "chat" / "racing-thread"
        thread_dir.mkdir(parents=True)
        protected = thread_dir / "current_session_chat.json"
        protected.write_text("{}")
        old_time = time.time() - (31 * 86400)
        os.utime(protected, (old_time, old_time))

        original_scan = housekeeping._scan_shortterm_thread_for_gc
        scan_count = 0

        def update_after_first_scan(path, cutoff_ts):
            nonlocal scan_count
            scan_count += 1
            snapshot = original_scan(path, cutoff_ts)
            if scan_count == 1:
                os.utime(protected, None)
            return snapshot

        monkeypatch.setattr(housekeeping, "_scan_shortterm_thread_for_gc", update_after_first_scan)

        result = housekeeping._cleanup_shortterm(tmp_path, retention_days=7, thread_gc_days=30)

        assert scan_count == 2
        assert result["thread_dirs_deleted"] == 0
        assert thread_dir.exists()

    def test_episodifies_abandoned_chat_session_before_delete(self, tmp_path: Path, monkeypatch):
        # F13(b): an expired session_state.json is preserved as an episode
        # before it is deleted.
        from core.memory.housekeeping import _cleanup_shortterm

        chat_dir = tmp_path / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)
        state_file = chat_dir / "session_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "original_prompt": "調査してほしい",
                    "accumulated_response": "途中まで進めた作業内容",
                    "notes": "残タスクあり",
                }
            ),
            encoding="utf-8",
        )
        old_time = time.time() - (14 * 86400)
        os.utime(state_file, (old_time, old_time))

        appended: list[str] = []

        class _FakeMgr:
            def __init__(self, _anima_dir):
                pass

            def append_episode(self, entry, *, origin=""):
                appended.append(entry)

        monkeypatch.setattr("core.memory.manager.MemoryManager", _FakeMgr)

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["episodified_sessions"] == 1
        assert not state_file.exists()
        assert len(appended) == 1
        assert "途中まで進めた作業内容" in appended[0]

    def test_skips_delete_when_episode_save_fails(self, tmp_path: Path, monkeypatch):
        # F13(b): if the episode write fails, the state file is kept for retry.
        from core.memory.housekeeping import _cleanup_shortterm

        chat_dir = tmp_path / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)
        state_file = chat_dir / "session_state.json"
        state_file.write_text(
            json.dumps({"accumulated_response": "保存すべき内容"}),
            encoding="utf-8",
        )
        old_time = time.time() - (14 * 86400)
        os.utime(state_file, (old_time, old_time))

        class _FailingMgr:
            def __init__(self, _anima_dir):
                pass

            def append_episode(self, entry, *, origin=""):
                raise RuntimeError("disk full")

        monkeypatch.setattr("core.memory.manager.MemoryManager", _FailingMgr)

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["episodified_sessions"] == 0
        assert result["deleted_files"] == 0
        assert state_file.exists()  # kept for retry


# ── _cleanup_facts_locks tests ─────────────────────────────────


class TestCleanupFactsLocks:
    """Test stale empty facts lock cleanup."""

    def test_deletes_only_stale_empty_locks(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_facts_locks

        facts_dir = tmp_path / "alice" / "facts"
        facts_dir.mkdir(parents=True)
        stale_empty = facts_dir / "stale.lock"
        recent_empty = facts_dir / "recent.lock"
        stale_nonempty = facts_dir / "nonempty.lock"
        stale_empty.write_bytes(b"")
        recent_empty.write_bytes(b"")
        stale_nonempty.write_bytes(b"locked")
        stale_time = time.time() - (25 * 3600)
        recent_time = time.time() - 3600
        os.utime(stale_empty, (stale_time, stale_time))
        os.utime(recent_empty, (recent_time, recent_time))
        os.utime(stale_nonempty, (stale_time, stale_time))

        result = _cleanup_facts_locks(tmp_path, stale_hours=24)

        assert result == {
            "scanned_files": 3,
            "deleted_files": 1,
            "locked_files": 0,
            "lock_failures": 0,
        }
        assert not stale_empty.exists()
        assert recent_empty.exists()
        assert stale_nonempty.exists()

    def test_preserves_lock_held_by_another_file_descriptor(self, tmp_path: Path):
        fcntl = pytest.importorskip("fcntl")
        from core.memory.housekeeping import _cleanup_facts_locks

        facts_dir = tmp_path / "alice" / "facts"
        facts_dir.mkdir(parents=True)
        held_lock = facts_dir / "held.lock"
        held_lock.write_bytes(b"")
        stale_time = time.time() - (25 * 3600)
        os.utime(held_lock, (stale_time, stale_time))

        with held_lock.open("r+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = _cleanup_facts_locks(tmp_path, stale_hours=24)

            assert held_lock.exists()
            assert result["deleted_files"] == 0
            assert result["locked_files"] == 1

    @pytest.mark.parametrize("stale_hours", [0, -1])
    def test_non_positive_stale_hours_skips_cleanup(self, tmp_path: Path, stale_hours: int):
        from core.memory.housekeeping import _cleanup_facts_locks

        facts_dir = tmp_path / "alice" / "facts"
        facts_dir.mkdir(parents=True)
        stale_lock = facts_dir / "stale.lock"
        stale_lock.write_bytes(b"")
        old_time = time.time() - (90 * 3600)
        os.utime(stale_lock, (old_time, old_time))

        result = _cleanup_facts_locks(tmp_path, stale_hours=stale_hours)

        assert stale_lock.exists()
        assert result["skipped"] is True
        assert result["reason"] == "stale_hours_must_be_positive"

    def test_fcntl_unavailable_fails_closed(self, tmp_path: Path):
        from unittest.mock import patch

        from core.memory.housekeeping import _cleanup_facts_locks

        facts_dir = tmp_path / "alice" / "facts"
        facts_dir.mkdir(parents=True)
        stale_lock = facts_dir / "stale.lock"
        stale_lock.write_bytes(b"")
        old_time = time.time() - (25 * 3600)
        os.utime(stale_lock, (old_time, old_time))

        with patch.dict("sys.modules", {"fcntl": None}):
            result = _cleanup_facts_locks(tmp_path, stale_hours=24)

        assert stale_lock.exists()
        assert result["skipped"] is True
        assert result["reason"] == "fcntl_unavailable"


# ── run_housekeeping integration test ──────────────────────────


class TestRunHousekeeping:
    """Integration test for the run_housekeeping orchestrator."""

    @pytest.mark.asyncio
    async def test_runs_all_tasks(self, tmp_path: Path):
        data_dir = tmp_path
        animas_dir = data_dir / "animas"

        # Set up prompt_logs
        alice_logs = animas_dir / "alice" / "prompt_logs"
        alice_logs.mkdir(parents=True)
        old_date = (today_local() - timedelta(days=5)).isoformat()
        (alice_logs / f"{old_date}.jsonl").write_text("{}\n")

        # Set up cron_logs
        cron_dir = animas_dir / "alice" / "state" / "cron_logs"
        cron_dir.mkdir(parents=True)
        old_cron = (today_local() - timedelta(days=40)).isoformat()
        (cron_dir / f"{old_cron}.jsonl").write_text("{}\n")

        # Set up shortterm
        chat_dir = animas_dir / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)
        old_st = chat_dir / "old_session.json"
        old_st.write_text("{}")
        old_time = time.time() - (14 * 86400)
        os.utime(old_st, (old_time, old_time))

        # Set up dm_logs dir (empty for this test)
        (data_dir / "shared" / "dm_logs").mkdir(parents=True)

        # Set up logs dir (no daemon log for this test)
        (data_dir / "logs").mkdir(parents=True)

        # Set up task_results
        task_results_dir = animas_dir / "alice" / "state" / "task_results"
        task_results_dir.mkdir(parents=True, exist_ok=True)
        old_tr = task_results_dir / "old_task.md"
        old_tr.write_text("old result")
        os.utime(old_tr, (old_time, old_time))

        # Set up pending/failed
        pf_dir = animas_dir / "alice" / "state" / "pending" / "failed"
        pf_dir.mkdir(parents=True)
        old_pf = pf_dir / "failed_task.json"
        old_pf.write_text("{}")
        os.utime(old_pf, (old_time, old_time))

        from core.memory.housekeeping import run_housekeeping

        results = await run_housekeeping(
            data_dir,
            prompt_log_retention_days=3,
            cron_log_retention_days=30,
            shortterm_retention_days=7,
            task_results_retention_days=7,
            pending_failed_retention_days=7,
        )

        assert "prompt_logs" in results
        assert results["prompt_logs"]["deleted_files"] == 1
        assert "cron_logs" in results
        assert results["cron_logs"]["deleted_files"] == 1
        assert "shortterm" in results
        assert "facts_locks" in results
        assert results["shortterm"]["deleted_files"] == 1
        assert "daemon_log" in results
        assert results["daemon_log"]["skipped"] is True
        assert "anima_logs" in results
        assert "frontend_logs" in results
        assert "dm_archives" in results
        assert results["task_results"]["deleted_files"] == 1
        assert results["pending_failed"]["deleted_files"] == 1
        assert "corrupt_vectordb_archives" in results
        assert "runtime_tmp" in results
        assert "backup_dirs" in results
        assert "codex_execution_logs" in results
        assert "codex_tmp" in results
        assert "anima_runtime_artifacts" in results

    @pytest.mark.asyncio
    async def test_handles_missing_dirs_gracefully(self, tmp_path: Path):
        from core.memory.housekeeping import run_housekeeping

        results = await run_housekeeping(tmp_path)

        assert "prompt_logs" in results
        assert "daemon_log" in results
        assert "anima_logs" in results
        assert "frontend_logs" in results
        assert "dm_archives" in results
        assert "cron_logs" in results
        assert "shortterm" in results
        assert "facts_locks" in results
        assert "task_results" in results
        assert "pending_failed" in results
        assert "codex_execution_logs" in results
        assert "codex_tmp" in results
        assert "anima_runtime_artifacts" in results
        assert "shared_inbox" in results
        assert results["shared_inbox"]["skipped"] is True

    @pytest.mark.asyncio
    async def test_shared_inbox_cleanup(self, tmp_path: Path):
        data_dir = tmp_path
        inbox = data_dir / "shared" / "inbox" / "alice"
        inbox.mkdir(parents=True)

        from core.schemas import Message

        old = now_local() - timedelta(hours=30)
        stale = Message(from_person="bob", to_person="alice", content="old", timestamp=old)
        (inbox / "old.json").write_text(stale.model_dump_json(indent=2), encoding="utf-8")
        protected = Message(
            from_person="bob",
            to_person="alice",
            content="task",
            timestamp=old,
            intent="delegation",
        )
        (inbox / "delegation.json").write_text(protected.model_dump_json(indent=2), encoding="utf-8")

        processed = inbox / "processed"
        processed.mkdir()
        old_processed = processed / "processed_old.json"
        old_processed.write_text("{}", encoding="utf-8")
        old_mtime = time.time() - (40 * 86400)
        os.utime(old_processed, (old_mtime, old_mtime))

        from core.memory.housekeeping import run_housekeeping

        results = await run_housekeeping(
            data_dir,
            inbox_ttl_hours=24,
            inbox_expired_retention_days=7,
            inbox_processed_retention_days=30,
            inbox_quarantine_retention_days=30,
        )

        shared = results["shared_inbox"]
        assert shared["expired"] == 1
        assert shared["protected"] == 1
        assert shared["deleted_processed"] == 1
        assert (inbox / "expired" / "old.json").exists()
        assert (inbox / "delegation.json").exists()
        assert not old_processed.exists()


# ── Task results cleanup tests ──────────────────────────────────


class TestCleanupTaskResults:
    """Tests for _cleanup_task_results."""

    def test_deletes_old_files(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_task_results

        results_dir = tmp_path / "alice" / "state" / "task_results"
        results_dir.mkdir(parents=True)

        old_time = time.time() - (10 * 86400)
        old_file = results_dir / "task_old.md"
        old_file.write_text("old result")
        os.utime(old_file, (old_time, old_time))

        new_file = results_dir / "task_new.md"
        new_file.write_text("new result")

        result = _cleanup_task_results(tmp_path, retention_days=7)
        assert result["deleted_files"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_skips_missing_dir(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_task_results

        result = _cleanup_task_results(tmp_path / "nonexistent", retention_days=7)
        assert result["skipped"] is True

    def test_skips_anima_without_task_results(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_task_results

        (tmp_path / "alice" / "state").mkdir(parents=True)
        result = _cleanup_task_results(tmp_path, retention_days=7)
        assert result["deleted_files"] == 0

    def test_cleans_multiple_animas(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_task_results

        old_time = time.time() - (10 * 86400)
        for name in ("alice", "bob"):
            d = tmp_path / name / "state" / "task_results"
            d.mkdir(parents=True)
            f = d / "old_task.md"
            f.write_text("result")
            os.utime(f, (old_time, old_time))

        result = _cleanup_task_results(tmp_path, retention_days=7)
        assert result["deleted_files"] == 2


# ── Pending failed cleanup tests ────────────────────────────────


class TestCleanupPendingFailed:
    """Tests for _cleanup_pending_failed."""

    def test_deletes_old_llm_failed(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_pending_failed

        failed_dir = tmp_path / "alice" / "state" / "pending" / "failed"
        failed_dir.mkdir(parents=True)

        old_time = time.time() - (20 * 86400)
        old_file = failed_dir / "task_old.json"
        old_file.write_text("{}")
        os.utime(old_file, (old_time, old_time))

        new_file = failed_dir / "task_new.json"
        new_file.write_text("{}")

        result = _cleanup_pending_failed(tmp_path, retention_days=14)
        assert result["deleted_files"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_deletes_old_cmd_failed(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_pending_failed

        failed_dir = tmp_path / "alice" / "state" / "background_tasks" / "pending" / "failed"
        failed_dir.mkdir(parents=True)

        old_time = time.time() - (20 * 86400)
        old_file = failed_dir / "cmd_old.json"
        old_file.write_text("{}")
        os.utime(old_file, (old_time, old_time))

        result = _cleanup_pending_failed(tmp_path, retention_days=14)
        assert result["deleted_files"] == 1
        assert not old_file.exists()

    def test_cleans_both_failed_dirs(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_pending_failed

        old_time = time.time() - (20 * 86400)

        llm_failed = tmp_path / "alice" / "state" / "pending" / "failed"
        llm_failed.mkdir(parents=True)
        f1 = llm_failed / "llm_old.json"
        f1.write_text("{}")
        os.utime(f1, (old_time, old_time))

        cmd_failed = tmp_path / "alice" / "state" / "background_tasks" / "pending" / "failed"
        cmd_failed.mkdir(parents=True)
        f2 = cmd_failed / "cmd_old.json"
        f2.write_text("{}")
        os.utime(f2, (old_time, old_time))

        result = _cleanup_pending_failed(tmp_path, retention_days=14)
        assert result["deleted_files"] == 2

    def test_skips_missing_dir(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_pending_failed

        result = _cleanup_pending_failed(tmp_path / "nonexistent", retention_days=14)
        assert result["skipped"] is True


class TestRuntimeBloatRetention:
    """Tests for recurring runtime bloat retention helpers."""

    def test_keeps_latest_two_corrupt_vectordb_archives(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_corrupt_vectordb_archives

        archive = tmp_path / "sakura" / "archive"
        archive.mkdir(parents=True)
        names = [
            "vectordb-corrupt-20260101-010101",
            "vectordb-corrupt-20260201-010101",
            "vectordb-corrupt-20260301-010101",
            "vectordb-corrupt-20260401-010101",
            "corrupt-vectordb-20260501010101",
        ]
        for name in names:
            path = archive / name
            path.mkdir()
            (path / "data.bin").write_text(name)

        result = _cleanup_corrupt_vectordb_archives(tmp_path, keep_generations=2)

        assert result["deleted_dirs"] == 3
        assert not (archive / "vectordb-corrupt-20260101-010101").exists()
        assert not (archive / "vectordb-corrupt-20260201-010101").exists()
        assert not (archive / "vectordb-corrupt-20260301-010101").exists()
        assert (archive / "vectordb-corrupt-20260401-010101").exists()
        assert (archive / "corrupt-vectordb-20260501010101").exists()

    def test_runtime_tmp_deletes_old_top_level_entries(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_runtime_tmp

        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        old_file = tmp_dir / "old.tmp"
        old_file.write_text("old")
        old_time = time.time() - (20 * 86400)
        os.utime(old_file, (old_time, old_time))
        recent_file = tmp_dir / "recent.tmp"
        recent_file.write_text("recent")
        old_dir_with_recent_child = tmp_dir / "old-dir-with-recent-child"
        old_dir_with_recent_child.mkdir()
        recent_child = old_dir_with_recent_child / "recent.txt"
        recent_child.write_text("recent")
        os.utime(old_dir_with_recent_child, (old_time, old_time))
        attachments = tmp_dir / "attachments"
        attachments.mkdir()
        old_attachment = attachments / "old.bin"
        old_attachment.write_text("old")
        os.utime(old_attachment, (old_time, old_time))
        recent_attachment = attachments / "recent.bin"
        recent_attachment.write_text("recent")
        os.utime(attachments, (old_time, old_time))

        result = _cleanup_runtime_tmp(tmp_dir, retention_days=14)

        assert result["deleted_entries"] == 2
        assert not old_file.exists()
        assert recent_file.exists()
        assert old_dir_with_recent_child.exists()
        assert recent_child.exists()
        assert attachments.exists()
        assert not old_attachment.exists()
        assert recent_attachment.exists()

    def test_backup_dirs_delete_only_old_backup_patterns(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_backup_dirs

        anima = tmp_path / "sakura"
        old_backup = anima / "assets_backup_20260201"
        old_backup.mkdir(parents=True)
        (old_backup / "asset.png").write_text("old")
        old_time = time.time() - (120 * 86400)
        os.utime(old_backup, (old_time, old_time))

        recent_backup = anima / "assets_backup_20260601"
        recent_backup.mkdir()
        keep_memory = anima / "knowledge"
        keep_memory.mkdir()
        nested_memory_backup = keep_memory / "assets_backup_20260201"
        nested_memory_backup.mkdir()
        os.utime(nested_memory_backup, (old_time, old_time))

        result = _cleanup_backup_dirs(tmp_path, retention_days=90)

        assert result["deleted_dirs"] == 1
        assert not old_backup.exists()
        assert recent_backup.exists()
        assert keep_memory.exists()
        assert nested_memory_backup.exists()

    def test_anima_runtime_logs_delete_old_and_cap_directory(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_anima_runtime_logs

        logs_dir = tmp_path / "logs" / "animas" / "sakura"
        logs_dir.mkdir(parents=True)
        current = logs_dir / f"{today_local().strftime('%Y%m%d')}.log"
        current.write_text("current", encoding="utf-8")
        old_dated = logs_dir / "20260401.log"
        old_dated.write_text("old", encoding="utf-8")
        keep_stderr = logs_dir / "stderr.log"
        keep_stderr.write_text("stderr", encoding="utf-8")
        first_recent = logs_dir / f"{(today_local() - timedelta(days=2)).strftime('%Y%m%d')}.log"
        second_recent = logs_dir / f"{(today_local() - timedelta(days=1)).strftime('%Y%m%d')}.log"
        first_recent.write_bytes(b"a" * 700_000)
        second_recent.write_bytes(b"b" * 700_000)
        old_time = time.time() - (20 * 86400)
        os.utime(first_recent, (old_time, old_time))
        current_link = logs_dir / "current.log"
        try:
            current_link.symlink_to(current.name)
        except OSError:
            current_link.write_text(current.name, encoding="utf-8")

        result = _cleanup_anima_runtime_logs(tmp_path / "logs" / "animas", retention_days=30, max_total_size_mb=1)

        assert result["deleted_files"] == 2
        assert result["capped_files"] == 1
        assert not old_dated.exists()
        assert not first_recent.exists()
        assert second_recent.exists()
        assert current.exists()
        assert keep_stderr.exists()

    def test_codex_execution_logs_delete_only_oversized_log_db_bundle(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_codex_execution_logs

        codex_home = tmp_path / "sakura" / ".codex_home"
        codex_home.mkdir(parents=True)
        log_db = codex_home / "logs_2.sqlite"
        wal = codex_home / "logs_2.sqlite-wal"
        shm = codex_home / "logs_2.sqlite-shm"
        state_db = codex_home / "state_5.sqlite"
        session = codex_home / "sessions" / "2026" / "06" / "10" / "rollout.jsonl"
        log_db.write_bytes(b"x" * (2 * 1024 * 1024))
        wal.write_text("wal", encoding="utf-8")
        shm.write_text("shm", encoding="utf-8")
        state_db.write_text("state", encoding="utf-8")
        session.parent.mkdir(parents=True)
        session.write_text("session", encoding="utf-8")

        result = _cleanup_codex_execution_logs(tmp_path, max_size_mb=1)

        assert result["deleted_databases"] == 1
        assert result["deleted_files"] == 3
        assert not log_db.exists()
        assert not wal.exists()
        assert not shm.exists()
        assert state_db.exists()
        assert session.exists()

    def test_codex_tmp_deletes_old_temp_entries(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_codex_tmp_dirs

        codex_home = tmp_path / "sakura" / ".codex_home"
        old_tmp = codex_home / ".tmp" / "plugins"
        old_tmp.mkdir(parents=True)
        old_file = old_tmp / "plugin.json"
        old_file.write_text("old", encoding="utf-8")
        recent_tmp = codex_home / ".tmp" / "recent"
        recent_tmp.mkdir()
        (recent_tmp / "keep.txt").write_text("recent", encoding="utf-8")
        old_time = time.time() - (24 * 3600)
        os.utime(old_file, (old_time, old_time))
        os.utime(old_tmp, (old_time, old_time))

        result = _cleanup_codex_tmp_dirs(tmp_path, retention_hours=12)

        assert result["deleted_entries"] == 1
        assert not old_tmp.exists()
        assert recent_tmp.exists()

    def test_frontend_logs_keep_latest_backups(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_frontend_logs

        log_dir = tmp_path / "logs" / "frontend"
        log_dir.mkdir(parents=True)
        active = log_dir / "frontend.jsonl"
        active.write_text("active", encoding="utf-8")
        latest = log_dir / "frontend.jsonl.20260610"
        second = log_dir / "frontend.jsonl.20260609"
        old = log_dir / "frontend.jsonl.20260301"
        legacy_old = log_dir / "20260228.jsonl"
        for path in (latest, second, old, legacy_old):
            path.write_text(path.name, encoding="utf-8")

        result = _cleanup_frontend_logs(log_dir, backup_count=2)

        assert result["deleted_files"] == 2
        assert active.exists()
        assert latest.exists()
        assert second.exists()
        assert not old.exists()
        assert not legacy_old.exists()

    def test_anima_runtime_artifacts_delete_tmp_gitdirs_and_local_logs(self, tmp_path: Path):
        from core.memory.housekeeping import _cleanup_anima_runtime_artifacts

        anima = tmp_path / "sakura"
        tmp_gitdir = anima / "tmp_gitdirs" / "work.git"
        tmp_gitdir.mkdir(parents=True)
        (tmp_gitdir / "pack").write_text("git temp", encoding="utf-8")
        local_logs = anima / "logs"
        local_logs.mkdir()
        old_log = local_logs / "tool-errors.log"
        old_log.write_text("old", encoding="utf-8")
        recent_log = local_logs / "recent.log"
        recent_log.write_text("recent", encoding="utf-8")
        memory_dir = anima / "knowledge"
        memory_dir.mkdir()
        memory_file = memory_dir / "keep.md"
        memory_file.write_text("memory", encoding="utf-8")
        old_time = time.time() - (40 * 86400)
        os.utime(tmp_gitdir / "pack", (old_time, old_time))
        os.utime(tmp_gitdir, (old_time, old_time))
        os.utime(old_log, (old_time, old_time))

        result = _cleanup_anima_runtime_artifacts(
            tmp_path,
            tmp_gitdirs_retention_days=14,
            local_log_retention_days=30,
        )

        assert result["tmp_gitdirs_deleted"] == 1
        assert result["local_logs_deleted"] == 1
        assert not tmp_gitdir.exists()
        assert not old_log.exists()
        assert recent_log.exists()
        assert memory_file.exists()


class TestEpisodeifyReadErrors:
    """R3: transient read errors must not discard recoverable sessions."""

    def test_json_decode_error_is_deletable(self, tmp_path: Path):
        from core.memory.housekeeping import _episodeify_abandoned_session

        state = tmp_path / "session_state.json"
        state.write_text("{not json", encoding="utf-8")
        assert _episodeify_abandoned_session(tmp_path, state) is True

    def test_os_error_defers_deletion(self, tmp_path: Path, monkeypatch):
        from core.memory.housekeeping import _episodeify_abandoned_session

        state = tmp_path / "session_state.json"
        state.write_text("{}", encoding="utf-8")

        original_read = Path.read_text

        def flaky_read(self, *args, **kwargs):
            if self.name == "session_state.json":
                raise OSError(24, "Too many open files")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read)
        assert _episodeify_abandoned_session(tmp_path, state) is False
        assert state.exists()


class TestEpisodifiedUnlinkFailure:
    """R6: unlink failure after episodify must not re-episodify next run."""

    def test_unlink_failure_renames_state_file(self, tmp_path: Path, monkeypatch):
        from core.memory.housekeeping import _cleanup_shortterm

        chat_dir = tmp_path / "alice" / "shortterm" / "chat"
        chat_dir.mkdir(parents=True)
        state_file = chat_dir / "session_state.json"
        state_file.write_text(
            json.dumps({"accumulated_response": "内容"}),
            encoding="utf-8",
        )
        old_time = time.time() - (14 * 86400)
        os.utime(state_file, (old_time, old_time))

        class _FakeMgr:
            def __init__(self, _anima_dir):
                pass

            def append_episode(self, entry, *, origin=""):
                pass

        monkeypatch.setattr("core.memory.manager.MemoryManager", _FakeMgr)

        original_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self.name == "session_state.json":
                raise OSError(13, "Permission denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["episodified_sessions"] == 1
        assert result["deleted_files"] == 0
        renamed = chat_dir / "session_state.json.episodified.bak"
        assert renamed.exists()
        assert not state_file.exists()
