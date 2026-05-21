from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Probation-specific Skill Curator tests."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.skills.curator import SkillCurator
from core.skills.models import SkillLifecycleState, SkillMetadata, SkillUsageEventType
from core.skills.usage import SkillUsageTracker


def test_probation_skill_has_stricter_review_and_archive_rules(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    failing = SkillMetadata(name="auto-failing", promotion_status="probation")
    unused = SkillMetadata(name="auto-unused", promotion_status="probation")
    tracker = SkillUsageTracker(anima_dir)
    for _ in range(3):
        tracker.record("auto-failing", SkillUsageEventType.failure)
    _append_usage_event(anima_dir, "auto-unused", SkillUsageEventType.create, datetime.now(UTC) - timedelta(days=91))

    suggestions = SkillCurator(anima_dir).suggest_lifecycle_transitions([failing, unused])
    by_name = {s.skill_name: s for s in suggestions}

    assert by_name["auto-failing"].suggested_state == SkillLifecycleState.review
    assert by_name["auto-failing"].reason == "probation_failure_rate_high"
    assert by_name["auto-unused"].suggested_state == SkillLifecycleState.archived
    assert by_name["auto-unused"].reason == "probation_unused_90d"


def test_fresh_never_used_probation_skill_is_not_archived_immediately(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    fresh = SkillMetadata(name="auto-fresh", promotion_status="probation")
    SkillUsageTracker(anima_dir).record("auto-fresh", SkillUsageEventType.create)

    suggestions = SkillCurator(anima_dir).suggest_lifecycle_transitions([fresh])

    assert suggestions == []


def _append_usage_event(anima_dir: Path, skill_name: str, event_type: SkillUsageEventType, ts: datetime) -> None:
    path = anima_dir / "state" / "skill_usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": ts.isoformat(),
        "skill_name": skill_name,
        "event_type": event_type.value,
        "is_common": False,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
