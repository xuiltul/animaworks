from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.skills.usage — SkillUsageTracker."""

import json
from pathlib import Path

import pytest

from core.skills.models import SkillUsageEvent, SkillUsageEventType, SkillUsageStats
from core.skills.usage import SkillUsageTracker


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return tmp_path


@pytest.fixture
def tracker(anima_dir: Path) -> SkillUsageTracker:
    """Create a fresh tracker instance."""
    return SkillUsageTracker(anima_dir)


class TestSkillUsageEventModel:
    def test_create_event(self):
        event = SkillUsageEvent(
            ts="2026-05-06T21:00:00+09:00",
            skill_name="github-pr-review",
            event_type=SkillUsageEventType.view,
            is_common=False,
        )
        assert event.skill_name == "github-pr-review"
        assert event.event_type == SkillUsageEventType.view
        assert event.is_common is False
        assert event.notes is None
        assert event.source_origin is None

    def test_event_with_notes(self):
        event = SkillUsageEvent(
            ts="2026-05-06T21:05:00+09:00",
            skill_name="deploy",
            event_type=SkillUsageEventType.success,
            notes="PR #42 reviewed",
        )
        assert event.notes == "PR #42 reviewed"

    def test_event_serialization(self):
        event = SkillUsageEvent(
            ts="2026-05-06T21:00:00+09:00",
            skill_name="test-skill",
            event_type=SkillUsageEventType.failure,
            is_common=True,
        )
        data = json.loads(event.model_dump_json())
        assert data["skill_name"] == "test-skill"
        assert data["event_type"] == "failure"
        assert data["is_common"] is True
        assert data["is_procedure"] is False


class TestSkillUsageStatsModel:
    def test_defaults(self):
        stats = SkillUsageStats(skill_name="foo")
        assert stats.view_count == 0
        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.patch_count == 0
        assert stats.last_used_at is None
        assert stats.create_origins == {}
        assert stats.ref is None
        assert stats.is_procedure is False

    def test_populated(self):
        stats = SkillUsageStats(
            skill_name="bar",
            view_count=5,
            success_count=3,
            failure_count=1,
            last_used_at="2026-05-06T21:00:00+09:00",
        )
        assert stats.view_count == 5
        assert stats.success_count == 3


class TestSkillUsageTrackerRecord:
    def test_record_creates_file(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("my-skill", SkillUsageEventType.view)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        assert usage_file.exists()
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["skill_name"] == "my-skill"
        assert data["event_type"] == "view"

    def test_record_appends(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("skill-a", SkillUsageEventType.success)
        tracker.record("skill-b", SkillUsageEventType.failure)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_record_with_notes(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("deploy", SkillUsageEventType.success, notes="All green")
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        data = json.loads(usage_file.read_text().strip())
        assert data["notes"] == "All green"

    def test_record_is_common(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("shared-skill", SkillUsageEventType.view, is_common=True)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        data = json.loads(usage_file.read_text().strip())
        assert data["is_common"] is True

    def test_record_create_source_origin(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record(
            "new-skill",
            SkillUsageEventType.create,
            ref="skills/new-skill/SKILL.md",
            source_origin="manual",
        )
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        data = json.loads(usage_file.read_text().strip())
        assert data["source_origin"] == "manual"
        assert data["ref"] == "skills/new-skill/SKILL.md"


class TestSkillUsageTrackerDebounce:
    def test_view_debounce_same_skill(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("my-skill", SkillUsageEventType.view)
        tracker.record("my-skill", SkillUsageEventType.view)
        tracker.record("my-skill", SkillUsageEventType.view)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_view_debounce_different_skills(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("skill-a", SkillUsageEventType.view)
        tracker.record("skill-b", SkillUsageEventType.view)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_view_debounce_common_vs_personal(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("same-name", SkillUsageEventType.view, is_common=False)
        tracker.record("same-name", SkillUsageEventType.view, is_common=True)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_non_view_events_not_debounced(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("my-skill", SkillUsageEventType.success)
        tracker.record("my-skill", SkillUsageEventType.success)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_reset_session_views(self, tracker: SkillUsageTracker, anima_dir: Path):
        tracker.record("my-skill", SkillUsageEventType.view)
        tracker.reset_session_views()
        tracker.record("my-skill", SkillUsageEventType.view)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2


class TestSkillUsageTrackerStats:
    def test_get_stats_empty(self, tracker: SkillUsageTracker):
        stats = tracker.get_stats("nonexistent")
        assert stats.skill_name == "nonexistent"
        assert stats.view_count == 0
        assert stats.success_count == 0

    def test_get_stats_after_records(self, tracker: SkillUsageTracker):
        tracker.record("my-skill", SkillUsageEventType.view)
        tracker.record("my-skill", SkillUsageEventType.success)
        tracker.record("my-skill", SkillUsageEventType.success)
        tracker.record("my-skill", SkillUsageEventType.failure)

        stats = tracker.get_stats("my-skill")
        assert stats.view_count == 1
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.last_used_at is not None

    def test_get_all_stats(self, tracker: SkillUsageTracker):
        tracker.record("skill-a", SkillUsageEventType.view)
        tracker.record("skill-b", SkillUsageEventType.success)
        tracker.record("skill-a", SkillUsageEventType.patch)

        all_stats = tracker.get_all_stats()
        assert "skill-a" in all_stats
        assert "skill-b" in all_stats
        assert all_stats["skill-a"].view_count == 1
        assert all_stats["skill-a"].patch_count == 1
        assert all_stats["skill-b"].success_count == 1

    def test_get_all_stats_uses_ref_to_disambiguate_scopes(self, tracker: SkillUsageTracker):
        tracker.record("same", SkillUsageEventType.use, ref="skills/same/SKILL.md")
        tracker.record("same", SkillUsageEventType.use, is_common=True, ref="common_skills/same/SKILL.md")
        tracker.record("same", SkillUsageEventType.use, is_procedure=True, ref="procedures/same.md")

        all_stats = tracker.get_all_stats()
        assert all_stats["skills/same/SKILL.md"].use_count == 1
        assert all_stats["common_skills/same/SKILL.md"].use_count == 1
        assert all_stats["common_skills/same/SKILL.md"].is_common is True
        assert all_stats["procedures/same.md"].use_count == 1
        assert all_stats["procedures/same.md"].is_procedure is True

        aggregate = tracker.get_stats("same")
        assert aggregate.use_count == 3
        assert aggregate.is_common is True
        assert aggregate.is_procedure is True

    def test_get_stats_counts_create(self, tracker: SkillUsageTracker):
        tracker.record("new-skill", SkillUsageEventType.create, source_origin="manual")
        stats = tracker.get_stats("new-skill")
        assert stats.create_count == 1
        assert stats.create_origins == {"manual": 1}

    def test_get_stats_counts_legacy_create_origin_as_unknown(self, anima_dir: Path):
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        usage_file.write_text(
            '{"ts":"2026-01-01","skill_name":"legacy","event_type":"create","is_common":false}\n',
            encoding="utf-8",
        )
        stats = SkillUsageTracker(anima_dir).get_stats("legacy")
        assert stats.create_count == 1
        assert stats.create_origins == {"unknown": 1}

    def test_get_stats_counts_multiple_create_origins(self, tracker: SkillUsageTracker):
        tracker.record("new-skill", SkillUsageEventType.create, source_origin="manual")
        tracker.record("new-skill", SkillUsageEventType.create, source_origin="auto_created")
        tracker.record("new-skill", SkillUsageEventType.create)
        stats = tracker.get_stats("new-skill")
        assert stats.create_count == 3
        assert stats.create_origins == {
            "manual": 1,
            "auto_created": 1,
            "unknown": 1,
        }

    def test_malformed_lines_skipped(self, anima_dir: Path):
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        usage_file.write_text(
            '{"ts":"2026-01-01","skill_name":"good","event_type":"view","is_common":false}\n'
            "not valid json\n"
            '{"ts":"2026-01-02","skill_name":"good","event_type":"success","is_common":false}\n'
        )
        tracker = SkillUsageTracker(anima_dir)
        stats = tracker.get_stats("good")
        assert stats.view_count == 1
        assert stats.success_count == 1


class TestSkillUsageTrackerStale:
    def test_get_stale_candidates_all_stale(self, anima_dir: Path):
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        usage_file.write_text(
            '{"ts":"2025-01-01T00:00:00+00:00","skill_name":"old-skill","event_type":"view","is_common":false}\n'
        )
        tracker = SkillUsageTracker(anima_dir)
        stale = tracker.get_stale_candidates(days=90)
        assert "old-skill" in stale

    def test_get_stale_candidates_recent_not_stale(self, tracker: SkillUsageTracker):
        tracker.record("fresh-skill", SkillUsageEventType.success)
        stale = tracker.get_stale_candidates(days=90)
        assert "fresh-skill" not in stale

    def test_get_stale_candidates_empty(self, tracker: SkillUsageTracker):
        stale = tracker.get_stale_candidates(days=90)
        assert stale == []


class TestSkillUsageTrackerEdgeCases:
    def test_state_dir_autocreated(self, tmp_path: Path):
        anima_dir = tmp_path / "fresh-anima"
        anima_dir.mkdir()
        tracker = SkillUsageTracker(anima_dir)
        tracker.record("test", SkillUsageEventType.view)
        assert (anima_dir / "state" / "skill_usage.jsonl").exists()

    def test_concurrent_tracker_instances(self, anima_dir: Path):
        tracker1 = SkillUsageTracker(anima_dir)
        tracker2 = SkillUsageTracker(anima_dir)
        tracker1.record("skill-a", SkillUsageEventType.view)
        tracker2.record("skill-b", SkillUsageEventType.view)
        usage_file = anima_dir / "state" / "skill_usage.jsonl"
        lines = usage_file.read_text().strip().splitlines()
        assert len(lines) == 2
