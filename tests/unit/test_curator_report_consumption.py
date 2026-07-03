from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Curator report consumption (heartbeat injection loop)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from core.skills.curator import (
    latest_unreviewed_report,
    summarize_curator_report,
)
from core.time_utils import now_iso


def _report_dir(anima_dir: Path) -> Path:
    d = anima_dir / "state" / "skill_curator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_report(anima_dir: Path, *, date: str = "2026-07-03", suggestions=None, duplicates=None) -> Path:
    report = {
        "states": {},
        "suggestions": suggestions if suggestions is not None else [],
        "metadata_gaps": {},
        "duplicates": duplicates if duplicates is not None else [],
    }
    path = _report_dir(anima_dir) / f"report-{date}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_reviewed_at(anima_dir: Path, when: datetime) -> None:
    path = _report_dir(anima_dir) / "last_reviewed.json"
    path.write_text(json.dumps({"reviewed_at": when.isoformat()}), encoding="utf-8")


_SUGGESTION = {"skill_name": "old-deploy", "suggested_state": "archived", "reason": "unused_180d", "metric": 200}
_DUPLICATE = {"skill_name": "deploy-a", "related_skill": "deploy-b", "score": 0.7, "signals": ["name_path_similarity"]}


def test_report_with_proposals_and_no_ack_is_returned(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[_SUGGESTION])
    report = latest_unreviewed_report(anima_dir)
    assert report is not None
    assert report["suggestions"][0]["skill_name"] == "old-deploy"


def test_report_older_than_ack_is_suppressed(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[_SUGGESTION])
    # Ack in the future -> the report predates the review -> suppressed.
    _write_reviewed_at(anima_dir, datetime.now(UTC) + timedelta(hours=1))
    assert latest_unreviewed_report(anima_dir) is None


def test_report_newer_than_ack_is_returned(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, duplicates=[_DUPLICATE])
    _write_reviewed_at(anima_dir, datetime.now(UTC) - timedelta(days=1))
    report = latest_unreviewed_report(anima_dir)
    assert report is not None
    assert report["duplicates"][0]["skill_name"] == "deploy-a"


def test_report_with_zero_proposals_is_not_returned(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[], duplicates=[])
    assert latest_unreviewed_report(anima_dir) is None


def test_corrupt_report_returns_none_without_raising(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (_report_dir(anima_dir) / "report-2026-07-03.json").write_text("{not json", encoding="utf-8")
    assert latest_unreviewed_report(anima_dir) is None


def test_missing_report_dir_returns_none(tmp_path: Path) -> None:
    assert latest_unreviewed_report(tmp_path / "nobody") is None


def test_latest_of_multiple_reports_is_used(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, date="2026-07-01", suggestions=[])
    _write_report(anima_dir, date="2026-07-03", suggestions=[_SUGGESTION])
    report = latest_unreviewed_report(anima_dir)
    assert report is not None
    assert report["suggestions"]


def test_summarize_curator_report_counts_and_breakdown() -> None:
    report = {
        "suggestions": [
            {"skill_name": "a", "suggested_state": "archived"},
            {"skill_name": "b", "suggested_state": "archived"},
            {"skill_name": "c", "suggested_state": "stale"},
        ],
        "duplicates": [{"skill_name": "d", "related_skill": "e"}],
    }
    count, breakdown, top_items = summarize_curator_report(report)
    assert count == 4
    assert "archived x2" in breakdown
    assert "stale x1" in breakdown
    assert "duplicates x1" in breakdown
    assert top_items.split(", ")[:3] == ["a", "b", "c"]


# ── Heartbeat injection (module-level helper, no full DigitalAnima) ──


def test_heartbeat_part_injected_when_unreviewed(tmp_path: Path) -> None:
    from core._anima_heartbeat import _build_curator_review_part

    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[_SUGGESTION])
    part = _build_curator_review_part(anima_dir, "alice")
    assert part is not None
    assert "curate_skills" in part
    assert "old-deploy" in part


def test_heartbeat_part_absent_after_ack(tmp_path: Path) -> None:
    from core._anima_heartbeat import _build_curator_review_part
    from core.tooling.handler_skills import SkillsToolsMixin

    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[_SUGGESTION])

    # Simulate curate_skills marking the report reviewed.
    marker_holder = SimpleNamespace(_anima_dir=anima_dir)
    SkillsToolsMixin._mark_curator_reviewed(marker_holder)
    marker = json.loads((anima_dir / "state" / "skill_curator" / "last_reviewed.json").read_text())
    assert "reviewed_at" in marker

    part = _build_curator_review_part(anima_dir, "alice")
    assert part is None


def test_heartbeat_part_absent_with_zero_proposals(tmp_path: Path) -> None:
    from core._anima_heartbeat import _build_curator_review_part

    anima_dir = tmp_path / "alice"
    _write_report(anima_dir, suggestions=[])
    assert _build_curator_review_part(anima_dir, "alice") is None


def test_heartbeat_part_absent_on_corrupt_report(tmp_path: Path) -> None:
    from core._anima_heartbeat import _build_curator_review_part

    anima_dir = tmp_path / "alice"
    (_report_dir(anima_dir) / "report-2026-07-03.json").write_text("broken", encoding="utf-8")
    assert _build_curator_review_part(anima_dir, "alice") is None


def test_mark_curator_reviewed_writes_iso_timestamp(tmp_path: Path) -> None:
    from core.tooling.handler_skills import SkillsToolsMixin

    anima_dir = tmp_path / "alice"
    holder = SimpleNamespace(_anima_dir=anima_dir)
    SkillsToolsMixin._mark_curator_reviewed(holder)
    data = json.loads((anima_dir / "state" / "skill_curator" / "last_reviewed.json").read_text())
    # Must parse as an ISO8601 timestamp (same format as now_iso()).
    assert datetime.fromisoformat(data["reviewed_at"].replace("Z", "+00:00"))
    assert data["reviewed_at"][:4] == now_iso()[:4]
