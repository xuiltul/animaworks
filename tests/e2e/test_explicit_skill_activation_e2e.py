from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.prompt.builder import build_system_prompt
from core.skills.activation import set_active_skill_refs


def _make_mock_memory(anima_dir: Path, tmp_path: Path) -> MagicMock:
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.common_skills_dir = tmp_path / "common_skills"
    memory.read_company_vision.return_value = ""
    memory.read_identity.return_value = "I am Alice"
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = ""
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_shared_users.return_value = []
    memory.load_recent_heartbeat_summary.return_value = ""
    memory.list_procedure_metas.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    return memory


def _fake_load_prompt(name: str, **kwargs) -> str:
    return ""


def _write_skill(anima_dir: Path, name: str, body: str, *, extra_frontmatter: str = "") -> Path:
    skill_dir = anima_dir / "skills" / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "description: active prompt skill\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _build(memory, common_skills_dir: Path, *, trigger: str, thread_id: str = "default") -> str:
    settings = SimpleNamespace(enabled=False, top_k=5, min_score=1.15, include_body=True)
    with (
        patch("core.paths.get_common_skills_dir", return_value=common_skills_dir),
        patch("core.prompt.builder._load_skill_catalog_router_settings", return_value=settings),
        patch("core.prompt.builder.load_prompt", side_effect=_fake_load_prompt),
        patch("core.prompt.builder._build_org_context", return_value=""),
        patch("core.prompt.builder._discover_other_animas", return_value=[]),
        patch("core.prompt.builder._build_messaging_section", return_value=""),
    ):
        return build_system_prompt(
            memory,
            message="use the writer skill",
            trigger=trigger,
            thread_id=thread_id,
        ).system_prompt


def test_active_skill_body_is_injected_for_matching_chat_thread(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    common_skills_dir = tmp_path / "common_skills"
    common_skills_dir.mkdir()
    _write_skill(anima_dir, "writer", "ACTIVE_WRITER_BODY")
    memory = _make_mock_memory(anima_dir, tmp_path)

    with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
        set_active_skill_refs(anima_dir, ["writer"], thread_id="thread_a")

    prompt = _build(memory, common_skills_dir, trigger="message:human", thread_id="thread_a")

    assert "## Trusted Skills" in prompt
    assert "skills/writer/SKILL.md" in prompt
    assert "ACTIVE_WRITER_BODY" in prompt


def test_probation_active_skill_is_rendered_as_candidate_hint(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    common_skills_dir = tmp_path / "common_skills"
    common_skills_dir.mkdir()
    _write_skill(
        anima_dir,
        "writer",
        "PROBATION_BODY_SHOULD_NOT_BE_INJECTED",
        extra_frontmatter=(
            "trust_level: community\n"
            "promotion_status: probation\n"
            "skill_policy:\n"
            "  use_mode: candidate_hint\n"
            "  injection: pointer_preferred\n"
        ),
    )
    memory = _make_mock_memory(anima_dir, tmp_path)

    with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
        set_active_skill_refs(anima_dir, ["writer"], thread_id="thread_a")

    prompt = _build(memory, common_skills_dir, trigger="message:human", thread_id="thread_a")

    assert "## Candidate Skill Hints" in prompt
    assert "skills/writer/SKILL.md" in prompt
    assert "Candidate hint only" in prompt
    assert "PROBATION_BODY_SHOULD_NOT_BE_INJECTED" not in prompt


def test_active_skill_body_is_not_injected_for_other_thread_or_background(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    common_skills_dir = tmp_path / "common_skills"
    common_skills_dir.mkdir()
    _write_skill(anima_dir, "writer", "ACTIVE_WRITER_BODY")
    memory = _make_mock_memory(anima_dir, tmp_path)

    with patch("core.paths.get_common_skills_dir", return_value=common_skills_dir):
        set_active_skill_refs(anima_dir, ["writer"], thread_id="thread_a")

    other_thread = _build(memory, common_skills_dir, trigger="message:human", thread_id="thread_b")
    heartbeat = _build(memory, common_skills_dir, trigger="heartbeat", thread_id="thread_a")
    cron = _build(memory, common_skills_dir, trigger="cron:daily", thread_id="thread_a")
    task = _build(memory, common_skills_dir, trigger="task:background", thread_id="thread_a")

    assert "ACTIVE_WRITER_BODY" not in other_thread
    assert "ACTIVE_WRITER_BODY" not in heartbeat
    assert "ACTIVE_WRITER_BODY" not in cron
    assert "ACTIVE_WRITER_BODY" not in task
