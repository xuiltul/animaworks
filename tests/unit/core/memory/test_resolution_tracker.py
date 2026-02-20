"""Unit tests for core/memory/resolution_tracker.py — ResolutionTracker."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.memory.resolution_tracker import ResolutionTracker

_JST = ZoneInfo("Asia/Tokyo")


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    return d


@pytest.fixture
def rt(shared_dir: Path, monkeypatch: pytest.MonkeyPatch) -> ResolutionTracker:
    monkeypatch.setattr(
        "core.memory.resolution_tracker.get_shared_dir",
        lambda: shared_dir,
    )
    return ResolutionTracker()


# ── append_resolution ────────────────────────────────────


class TestAppendResolution:
    def test_creates_jsonl_file(self, rt: ResolutionTracker, shared_dir: Path) -> None:
        rt.append_resolution(issue="server crash", resolver="alice")
        path = shared_dir / "resolutions.jsonl"
        assert path.exists()

    def test_writes_correct_json_fields(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        rt.append_resolution(issue="disk full", resolver="bob")
        path = shared_dir / "resolutions.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["issue"] == "disk full"
        assert entry["resolver"] == "bob"
        assert "ts" in entry

    def test_appends_multiple_entries(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        rt.append_resolution(issue="issue-1", resolver="alice")
        rt.append_resolution(issue="issue-2", resolver="bob")
        path = shared_dir / "resolutions.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["issue"] == "issue-1"
        assert json.loads(lines[1])["issue"] == "issue-2"

    def test_creates_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "shared"
        monkeypatch.setattr(
            "core.memory.resolution_tracker.get_shared_dir",
            lambda: nested,
        )
        rt = ResolutionTracker()
        rt.append_resolution(issue="test", resolver="tester")
        assert (nested / "resolutions.jsonl").exists()


# ── read_resolutions ─────────────────────────────────────


class TestReadResolutions:
    def test_returns_recent_entries(self, rt: ResolutionTracker) -> None:
        rt.append_resolution(issue="resolved-issue", resolver="alice")
        results = rt.read_resolutions(days=7)
        assert len(results) == 1
        assert results[0]["issue"] == "resolved-issue"
        assert results[0]["resolver"] == "alice"

    def test_missing_file_returns_empty_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_shared = tmp_path / "empty_shared"
        empty_shared.mkdir()
        monkeypatch.setattr(
            "core.memory.resolution_tracker.get_shared_dir",
            lambda: empty_shared,
        )
        rt = ResolutionTracker()
        results = rt.read_resolutions(days=7)
        assert results == []

    def test_filters_old_entries(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        """Entries older than the requested day range are excluded."""
        path = shared_dir / "resolutions.jsonl"

        now = datetime.now(tz=_JST)

        # Recent entry (within 7 days)
        recent_ts = now.isoformat()
        recent_entry = json.dumps({
            "ts": recent_ts, "issue": "recent", "resolver": "alice",
        })

        # Old entry (30 days ago)
        old_ts = (now - timedelta(days=30)).isoformat()
        old_entry = json.dumps({
            "ts": old_ts, "issue": "old", "resolver": "bob",
        })

        path.write_text(old_entry + "\n" + recent_entry + "\n", encoding="utf-8")

        results = rt.read_resolutions(days=7)
        assert len(results) == 1
        assert results[0]["issue"] == "recent"

    def test_days_parameter_boundary(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        """Entries exactly at the boundary are included or excluded correctly."""
        path = shared_dir / "resolutions.jsonl"

        now = datetime.now(tz=_JST)

        # Entry from 1 day ago (should be included with days=2)
        one_day_ago = (now - timedelta(days=1)).isoformat()
        entry_1d = json.dumps({
            "ts": one_day_ago, "issue": "one-day-old", "resolver": "alice",
        })

        # Entry from 3 days ago (should be excluded with days=2)
        three_days_ago = (now - timedelta(days=3)).isoformat()
        entry_3d = json.dumps({
            "ts": three_days_ago, "issue": "three-days-old", "resolver": "bob",
        })

        path.write_text(entry_3d + "\n" + entry_1d + "\n", encoding="utf-8")

        results = rt.read_resolutions(days=2)
        issues = [r["issue"] for r in results]
        assert "one-day-old" in issues
        assert "three-days-old" not in issues

    def test_skips_malformed_json(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        """Malformed JSON lines are silently skipped."""
        path = shared_dir / "resolutions.jsonl"

        now = datetime.now(tz=_JST)
        valid_entry = json.dumps({
            "ts": now.isoformat(), "issue": "valid", "resolver": "alice",
        })
        path.write_text(
            "not valid json\n" + valid_entry + "\n", encoding="utf-8",
        )

        results = rt.read_resolutions(days=7)
        assert len(results) == 1
        assert results[0]["issue"] == "valid"

    def test_skips_blank_lines(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        """Blank lines in the JSONL file are skipped."""
        path = shared_dir / "resolutions.jsonl"

        now = datetime.now(tz=_JST)
        entry = json.dumps({
            "ts": now.isoformat(), "issue": "test", "resolver": "alice",
        })
        path.write_text(
            "\n" + entry + "\n\n", encoding="utf-8",
        )

        results = rt.read_resolutions(days=7)
        assert len(results) == 1

    def test_entry_without_ts_excluded(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        """Entries missing the 'ts' field are excluded by the cutoff comparison."""
        path = shared_dir / "resolutions.jsonl"

        now = datetime.now(tz=_JST)
        valid = json.dumps({
            "ts": now.isoformat(), "issue": "has-ts", "resolver": "alice",
        })
        no_ts = json.dumps({"issue": "no-ts", "resolver": "bob"})
        path.write_text(no_ts + "\n" + valid + "\n", encoding="utf-8")

        results = rt.read_resolutions(days=7)
        issues = [r["issue"] for r in results]
        assert "has-ts" in issues
        # Entry without ts has "" which is < cutoff, so excluded
        assert "no-ts" not in issues

    def test_empty_file_returns_empty_list(
        self, rt: ResolutionTracker, shared_dir: Path,
    ) -> None:
        path = shared_dir / "resolutions.jsonl"
        path.write_text("", encoding="utf-8")
        results = rt.read_resolutions(days=7)
        assert results == []
