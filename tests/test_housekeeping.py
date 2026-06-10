from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for core.memory.housekeeping and related modules."""

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
        assert cfg.daemon_log_max_size_mb == 200
        assert cfg.daemon_log_keep_generations == 2
        assert cfg.frontend_log_backup_count == 7
        assert cfg.dm_log_archive_retention_days == 30
        assert cfg.cron_log_retention_days == 30
        assert cfg.shortterm_retention_days == 7
        assert cfg.task_results_retention_days == 7
        assert cfg.pending_failed_retention_days == 14
        assert cfg.corrupt_vectordb_keep_generations == 3
        assert cfg.tmp_retention_days == 14
        assert cfg.backup_retention_days == 90

    def test_custom_values(self):
        from core.config.models import HousekeepingConfig

        cfg = HousekeepingConfig(
            enabled=False,
            run_time="03:00",
            prompt_log_retention_days=7,
            daemon_log_max_size_mb=200,
            daemon_log_keep_generations=3,
            frontend_log_backup_count=14,
            dm_log_archive_retention_days=60,
            cron_log_retention_days=14,
            shortterm_retention_days=14,
            corrupt_vectordb_keep_generations=4,
            tmp_retention_days=21,
            backup_retention_days=120,
        )
        assert cfg.enabled is False
        assert cfg.run_time == "03:00"
        assert cfg.prompt_log_retention_days == 7
        assert cfg.daemon_log_max_size_mb == 200
        assert cfg.corrupt_vectordb_keep_generations == 4
        assert cfg.tmp_retention_days == 21
        assert cfg.backup_retention_days == 120

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
        assert not log.exists()  # original renamed
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

        hb_dir = tmp_path / "alice" / "shortterm" / "heartbeat"
        hb_dir.mkdir(parents=True)

        protected = hb_dir / "streaming_journal_heartbeat.jsonl"
        protected.write_text("{}")
        old_time = time.time() - (30 * 86400)
        os.utime(protected, (old_time, old_time))

        result = _cleanup_shortterm(tmp_path, retention_days=7)
        assert result["deleted_files"] == 0
        assert protected.exists()

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
        assert results["shortterm"]["deleted_files"] == 1
        assert "daemon_log" in results
        assert results["daemon_log"]["skipped"] is True
        assert "dm_archives" in results
        assert results["task_results"]["deleted_files"] == 1
        assert results["pending_failed"]["deleted_files"] == 1
        assert "corrupt_vectordb_archives" in results
        assert "runtime_tmp" in results
        assert "backup_dirs" in results

    @pytest.mark.asyncio
    async def test_handles_missing_dirs_gracefully(self, tmp_path: Path):
        from core.memory.housekeeping import run_housekeeping

        results = await run_housekeeping(tmp_path)

        assert "prompt_logs" in results
        assert "daemon_log" in results
        assert "dm_archives" in results
        assert "cron_logs" in results
        assert "shortterm" in results
        assert "task_results" in results
        assert "pending_failed" in results
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

    def test_keeps_latest_three_corrupt_vectordb_archives(self, tmp_path: Path):
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

        result = _cleanup_corrupt_vectordb_archives(tmp_path, keep_generations=3)

        assert result["deleted_dirs"] == 2
        assert not (archive / "vectordb-corrupt-20260101-010101").exists()
        assert not (archive / "vectordb-corrupt-20260201-010101").exists()
        assert (archive / "vectordb-corrupt-20260301-010101").exists()
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

        result = _cleanup_runtime_tmp(tmp_dir, retention_days=14)

        assert result["deleted_entries"] == 1
        assert not old_file.exists()
        assert recent_file.exists()

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

        result = _cleanup_backup_dirs(tmp_path, retention_days=90)

        assert result["deleted_dirs"] == 1
        assert not old_backup.exists()
        assert recent_backup.exists()
        assert keep_memory.exists()
