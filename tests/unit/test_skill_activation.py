from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pytest

from core.skills.activation import (
    build_active_skill_context,
    get_active_skill_refs,
    get_active_skill_state,
    list_skill_catalog,
    set_active_skill_refs,
    validate_thread_id,
)
from core.skills.usage import SkillUsageTracker


def _write_skill(root: Path, name: str, *, body: str = "Use this skill.", extra: str = "") -> Path:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: test skill\n"
        f"{extra}"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _write_common_skill(common_dir: Path, name: str, *, body: str = "Use common skill.") -> Path:
    skill_dir = common_dir / "community" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: common test skill\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def test_set_get_and_clear_active_skill_refs(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "daily-report", body="Write the daily report.")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        result = set_active_skill_refs(anima_dir, ["daily-report"], thread_id="default")
        assert [item.path for item in result.accepted] == ["skills/daily-report/SKILL.md"]
        assert get_active_skill_refs(anima_dir, "default") == ["skills/daily-report/SKILL.md"]

        set_active_skill_refs(anima_dir, [], thread_id="default")
        assert get_active_skill_refs(anima_dir, "default") == []


def test_active_skill_state_is_thread_scoped(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "thread-skill")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        set_active_skill_refs(anima_dir, ["thread-skill"], thread_id="thread_a")

    assert get_active_skill_refs(anima_dir, "thread_a") == ["skills/thread-skill/SKILL.md"]
    assert get_active_skill_refs(anima_dir, "thread_b") == []


def test_rejects_unsafe_and_requires_confirm_for_warn_or_risk(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "safe-skill")
    _write_skill(anima_dir, "warn-skill", extra="security:\n  verdict: warn\n")
    _write_skill(anima_dir, "danger-skill", extra="security:\n  verdict: dangerous\n")
    _write_skill(anima_dir, "blocked-skill", extra="trust_level: blocked\n")
    _write_skill(anima_dir, "quarantine-skill", extra="trust_level: quarantine\n")
    _write_skill(anima_dir, "send-skill", extra="risk:\n  external_send: true\n")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        result = set_active_skill_refs(
            anima_dir,
            ["safe-skill", "warn-skill", "danger-skill", "blocked-skill", "quarantine-skill", "send-skill"],
        )
        assert [item.name for item in result.accepted] == ["safe-skill"]
        assert {item.ref: item.reason for item in result.rejections} == {
            "warn-skill": "security_warn",
            "danger-skill": "security_dangerous",
            "blocked-skill": "trust_level_blocked",
            "quarantine-skill": "trust_level_quarantine",
            "send-skill": "risk_external_send",
        }

        confirmed = set_active_skill_refs(anima_dir, ["warn-skill", "send-skill"], confirm_risk=True)
        assert [item.name for item in confirmed.accepted] == ["warn-skill", "send-skill"]
        assert [(item.ref, item.reason) for item in confirmed.warnings] == [
            ("warn-skill", "security_warn_allowed"),
            ("send-skill", "risk_external_send_allowed"),
        ]


def test_build_active_skill_context_renders_body_and_records_use(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "writer", body="Follow writer rules.")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        set_active_skill_refs(anima_dir, ["writer"])
        result = build_active_skill_context(anima_dir)

    rendered = result.render()
    assert "## Trusted Skills" in rendered
    assert "### writer" in rendered
    assert "Follow writer rules." in rendered
    assert SkillUsageTracker(anima_dir).get_stats("writer").use_count == 1


def test_build_active_skill_context_truncates_without_rejecting(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "too-big", body="0123456789ABCDEF")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        set_active_skill_refs(anima_dir, ["too-big"])
        result = build_active_skill_context(anima_dir, max_skill_chars=10)

    rendered = result.render()
    assert "body: omitted" in rendered
    assert "max_skill_chars_exceeded" in rendered
    assert "0123456789ABCDEF" not in rendered


def test_catalog_marks_active_personal_and_common_skills(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "personal")
    _write_common_skill(common_dir, "shared")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        set_active_skill_refs(anima_dir, ["common_skills/community/shared/SKILL.md"])
        catalog = list_skill_catalog(anima_dir)

    by_name = {item["name"]: item for item in catalog}
    assert by_name["personal"]["active"] is False
    assert by_name["shared"]["active"] is True
    assert by_name["shared"]["path"] == "common_skills/community/shared/SKILL.md"


def test_deleted_active_skill_is_reported_on_state_read(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    common_dir = tmp_path / "common_skills"
    common_dir.mkdir(parents=True)
    _write_skill(anima_dir, "temp")

    with patch("core.paths.get_common_skills_dir", return_value=common_dir):
        set_active_skill_refs(anima_dir, ["temp"])
        (anima_dir / "skills" / "temp" / "SKILL.md").unlink()
        state = get_active_skill_state(anima_dir)

    assert not state.accepted
    assert [(item.ref, item.reason) for item in state.rejections] == [("skills/temp/SKILL.md", "not_found")]


def test_validate_thread_id_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        validate_thread_id("../bad")
