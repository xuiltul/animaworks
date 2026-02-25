from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for DM log archive rotation (_rotate_dm_logs_sync).

Covers:
- Old entries (older than max_age_days) archived to {stem}.{date}.archive.jsonl
- Recent entries retained in the original file
- Skip existing archive files (names containing .archive.)
- Empty or missing dm_logs dir returns empty dict
- Append to existing archive file
- Malformed JSON lines kept in recent file
- All entries old leaves empty main file
- No old entries does nothing (no results)
"""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.background import _rotate_dm_logs_sync
from core.time_utils import now_jst


def _write_dm_entries(dm_logs_dir: Path, pair_name: str, entries: list[dict]) -> Path:
    """Write DM log entries to a JSONL file."""
    dm_logs_dir.mkdir(parents=True, exist_ok=True)
    filepath = dm_logs_dir / f"{pair_name}.jsonl"
    with filepath.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return filepath


def _make_dm_entry(
    from_name: str = "alice",
    to_name: str = "bob",
    text: str = "hello",
    ts: str | None = None,
) -> dict:
    """Create a DM log entry dict."""
    return {
        "ts": ts or now_jst().isoformat(),
        "from": from_name,
        "to": to_name,
        "text": text,
    }


class TestRotateDmLogs:
    """Test dm_log archive rotation."""

    def test_archives_old_entries(self, tmp_path: Path) -> None:
        """7日超のエントリがアーカイブファイルに移動される。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        with patch("core.time_utils.now_jst", return_value=base):
            old_ts = (base - timedelta(days=8)).isoformat()
            recent_ts = (base - timedelta(days=3)).isoformat()
            _write_dm_entries(
                dm_logs,
                "alice-bob",
                [
                    _make_dm_entry("alice", "bob", "old msg", ts=old_ts),
                    _make_dm_entry("bob", "alice", "recent msg", ts=recent_ts),
                ],
            )

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result["alice-bob.jsonl"] == {"archived": 1, "kept": 1}
        archive_path = dm_logs / f"alice-bob.{base.strftime('%Y%m%d')}.archive.jsonl"
        assert archive_path.exists()
        lines = archive_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["text"] == "old msg"

        main_content = (dm_logs / "alice-bob.jsonl").read_text(encoding="utf-8")
        kept = json.loads(main_content.strip())
        assert kept["text"] == "recent msg"

    def test_keeps_recent_entries(self, tmp_path: Path) -> None:
        """7日以内のエントリは本体に残る。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        recent_ts = (base - timedelta(days=1)).isoformat()
        _write_dm_entries(
            dm_logs,
            "x-y",
            [_make_dm_entry("x", "y", "hi", ts=recent_ts)],
        )

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result == {}
        content = (dm_logs / "x-y.jsonl").read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        assert entry["text"] == "hi"
        assert not any(p.name.endswith(".archive.jsonl") for p in dm_logs.glob("*.jsonl"))

    def test_skips_archive_files(self, tmp_path: Path) -> None:
        """既存のarchiveファイルは処理されない。"""
        dm_logs = tmp_path / "dm_logs"
        dm_logs.mkdir(parents=True)
        archive_path = dm_logs / "alice-bob.20260201.archive.jsonl"
        archive_path.write_text('{"ts":"2026-01-01T00:00:00+09:00","from":"a","to":"b","text":"x"}\n')
        main_path = dm_logs / "bob-charlie.jsonl"
        old_ts = (now_jst() - timedelta(days=10)).isoformat()
        _write_dm_entries(dm_logs, "bob-charlie", [_make_dm_entry("bob", "charlie", "old", ts=old_ts)])

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert "bob-charlie.jsonl" in result
        assert "alice-bob.20260201.archive.jsonl" not in result
        assert archive_path.read_text() == '{"ts":"2026-01-01T00:00:00+09:00","from":"a","to":"b","text":"x"}\n'

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """dm_logsディレクトリが存在しない場合は空dictを返す。"""
        shared = tmp_path / "shared"
        shared.mkdir()
        assert not (shared / "dm_logs").exists()

        result = _rotate_dm_logs_sync(shared, max_age_days=7)

        assert result == {}

    def test_appends_to_existing_archive(self, tmp_path: Path) -> None:
        """既存archiveファイルに追記される。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        date_str = base.strftime("%Y%m%d")
        archive_path = dm_logs / f"alice-bob.{date_str}.archive.jsonl"
        dm_logs.mkdir(parents=True)
        archive_path.write_text('{"ts":"2026-01-01T00:00:00+09:00","from":"a","to":"b","text":"first"}\n')
        old_ts = (base - timedelta(days=8)).isoformat()
        main_path = dm_logs / "alice-bob.jsonl"
        main_path.write_text(json.dumps(_make_dm_entry("alice", "bob", "second", ts=old_ts), ensure_ascii=False) + "\n")

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result["alice-bob.jsonl"] == {"archived": 1, "kept": 0}
        lines = archive_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "first"
        assert json.loads(lines[1])["text"] == "second"

    def test_malformed_json_kept_in_recent(self, tmp_path: Path) -> None:
        """不正なJSONは最新ファイルに残る。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        old_ts = (base - timedelta(days=10)).isoformat()
        dm_logs.mkdir(parents=True)
        path = dm_logs / "pair.jsonl"
        path.write_text(
            json.dumps(_make_dm_entry("a", "b", "valid", ts=old_ts), ensure_ascii=False) + "\n"
            + "{not valid json}\n"
            + "{\"ts\":\"" + (base - timedelta(days=1)).isoformat() + "\",\"from\":\"x\",\"to\":\"y\",\"text\":\"ok\"}\n"
        )

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result["pair.jsonl"] == {"archived": 1, "kept": 2}
        recent_content = path.read_text(encoding="utf-8")
        assert "{not valid json}" in recent_content
        assert '"ok"' in recent_content

    def test_all_entries_old_leaves_empty_file(self, tmp_path: Path) -> None:
        """全エントリが古い場合、本体ファイルは空になる。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        old_ts = (base - timedelta(days=14)).isoformat()
        _write_dm_entries(
            dm_logs,
            "old-pair",
            [
                _make_dm_entry("a", "b", "old1", ts=old_ts),
                _make_dm_entry("b", "a", "old2", ts=old_ts),
            ],
        )

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result["old-pair.jsonl"] == {"archived": 2, "kept": 0}
        assert (dm_logs / "old-pair.jsonl").read_text() == ""

    def test_no_old_entries_does_nothing(self, tmp_path: Path) -> None:
        """古いエントリがない場合は何もしない。"""
        dm_logs = tmp_path / "dm_logs"
        base = now_jst()
        recent_ts = (base - timedelta(days=2)).isoformat()
        _write_dm_entries(dm_logs, "fresh", [_make_dm_entry("x", "y", "new", ts=recent_ts)])

        result = _rotate_dm_logs_sync(tmp_path, max_age_days=7)

        assert result == {}
        assert (dm_logs / "fresh.jsonl").read_text().strip() != ""
        assert not list(dm_logs.glob("*.archive.jsonl"))


class TestDmLogRotationSchedulerRegistration:
    """Verify rotate_dm_logs is registered in LifecycleManager system crons."""

    def test_scheduler_has_dm_log_rotation_job(self) -> None:
        """system_dm_log_rotation ジョブがスケジューラに登録される。"""
        from core.lifecycle import LifecycleManager

        lifecycle = LifecycleManager()
        lifecycle._setup_system_crons()
        job = lifecycle.scheduler.get_job("system_dm_log_rotation")
        assert job is not None, "system_dm_log_rotation job not found in scheduler"
        assert "DM Log Rotation" in job.name
