from __future__ import annotations

"""Tests for activity log rotation and tool_result recording."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.memory.activity import ActivityLogger


# ── Helpers ────────────────────────────────────────────────────────


def _make_log_file(log_dir: Path, date_str: str, size_bytes: int = 100) -> Path:
    """Create a fake JSONL activity log file of approximately *size_bytes*."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{date_str}.jsonl"
    # Fill with repeating JSON lines to reach the target size
    line = json.dumps({"ts": f"{date_str}T10:00:00+09:00", "type": "tool_use"}, ensure_ascii=False)
    lines_needed = max(1, size_bytes // (len(line) + 1))
    path.write_text("\n".join([line] * lines_needed) + "\n", encoding="utf-8")
    return path


# ── tool_result format ──────────────────────────────────────────


class TestToolResultFormat:
    """Test that tool_result type is properly formatted in priming output."""

    def test_format_entry_tool_result(self, tmp_path: Path) -> None:
        from core.memory.activity import ActivityEntry

        entry = ActivityEntry(
            ts="2026-02-22T10:00:00+09:00",
            type="tool_result",
            tool="web_search",
            content="Search results here",
        )
        formatted = ActivityLogger._format_entry(entry)
        assert "TRES" in formatted
        assert "tool:web_search" in formatted

    def test_log_tool_result_entry(self, tmp_path: Path) -> None:
        al = ActivityLogger(tmp_path)
        entry = al.log(
            "tool_result",
            tool="web_search",
            content="Full result text",
            meta={"tool_use_id": "tu_123", "is_error": False},
        )
        assert entry.type == "tool_result"
        assert entry.tool == "web_search"
        assert entry.meta["tool_use_id"] == "tu_123"

        # Verify written to disk
        today = date.today().isoformat()
        log_file = tmp_path / "activity_log" / f"{today}.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        raw = json.loads(lines[-1])
        assert raw["type"] == "tool_result"
        assert raw["tool"] == "web_search"
        assert raw["content"] == "Full result text"


# ── Rotation: time mode ──────────────────────────────────────────


class TestRotationTime:
    def test_deletes_old_files(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "activity_log"
        today = date.today()
        old_date = (today - timedelta(days=10)).isoformat()
        recent_date = (today - timedelta(days=2)).isoformat()
        today_str = today.isoformat()

        _make_log_file(log_dir, old_date, 500)
        _make_log_file(log_dir, recent_date, 500)
        _make_log_file(log_dir, today_str, 500)

        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="time", max_age_days=7)

        assert result["deleted_files"] == 1
        assert not (log_dir / f"{old_date}.jsonl").exists()
        assert (log_dir / f"{recent_date}.jsonl").exists()
        assert (log_dir / f"{today_str}.jsonl").exists()

    def test_preserves_today(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "activity_log"
        today_str = date.today().isoformat()
        _make_log_file(log_dir, today_str, 500)

        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="time", max_age_days=0)

        # Today's file must never be deleted
        assert result["deleted_files"] == 0
        assert (log_dir / f"{today_str}.jsonl").exists()


# ── Rotation: size mode ──────────────────────────────────────────


class TestRotationSize:
    def test_deletes_oldest_when_over_limit(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "activity_log"
        today = date.today()

        # Create 3 files, each ~600 bytes, total ~1800 bytes
        dates = [
            (today - timedelta(days=3)).isoformat(),
            (today - timedelta(days=2)).isoformat(),
            (today - timedelta(days=1)).isoformat(),
        ]
        for d in dates:
            _make_log_file(log_dir, d, 600)

        al = ActivityLogger(tmp_path)
        # Set limit to ~1200 bytes (should delete the oldest)
        result = al.rotate(mode="size", max_size_mb=0)  # 0 MB = force delete

        # Should delete files until under limit (0 means delete all except today)
        assert result["deleted_files"] == 3

    def test_preserves_today_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "activity_log"
        today_str = date.today().isoformat()
        _make_log_file(log_dir, today_str, 2_000_000)

        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="size", max_size_mb=1)  # 1MB limit, file is 2MB

        # Today's file protected
        assert result["deleted_files"] == 0
        assert (log_dir / f"{today_str}.jsonl").exists()


# ── Rotation: both mode ──────────────────────────────────────────


class TestRotationBoth:
    def test_applies_time_then_size(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "activity_log"
        today = date.today()

        # Old file (should be deleted by time)
        old_date = (today - timedelta(days=30)).isoformat()
        _make_log_file(log_dir, old_date, 500)

        # Recent but large files (should be trimmed by size)
        d1 = (today - timedelta(days=3)).isoformat()
        d2 = (today - timedelta(days=2)).isoformat()
        d3 = (today - timedelta(days=1)).isoformat()
        _make_log_file(log_dir, d1, 600)
        _make_log_file(log_dir, d2, 600)
        _make_log_file(log_dir, d3, 600)

        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="both", max_age_days=7, max_size_mb=0)

        # old_date deleted by time, then all remaining by size (0 MB limit)
        assert result["deleted_files"] == 4


# ── Rotation: empty / missing dir ────────────────────────────────


class TestRotationEdgeCases:
    def test_no_log_dir(self, tmp_path: Path) -> None:
        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="size", max_size_mb=100)
        assert result["deleted_files"] == 0
        assert result["freed_bytes"] == 0

    def test_empty_log_dir(self, tmp_path: Path) -> None:
        (tmp_path / "activity_log").mkdir()
        al = ActivityLogger(tmp_path)
        result = al.rotate(mode="size", max_size_mb=100)
        assert result["deleted_files"] == 0


# ── rotate_all ────────────────────────────────────────────────────


class TestRotateAll:
    def test_rotates_multiple_animas(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        today = date.today()
        old_date = (today - timedelta(days=30)).isoformat()

        for name in ["alice", "bob"]:
            log_dir = animas_dir / name / "activity_log"
            _make_log_file(log_dir, old_date, 500)

        results = ActivityLogger.rotate_all(
            animas_dir, mode="time", max_age_days=7,
        )
        assert "alice" in results
        assert "bob" in results
        assert results["alice"]["deleted_files"] == 1
        assert results["bob"]["deleted_files"] == 1

    def test_skips_anima_without_logs(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        (animas_dir / "empty_anima").mkdir(parents=True)

        results = ActivityLogger.rotate_all(
            animas_dir, mode="size", max_size_mb=100,
        )
        assert len(results) == 0
