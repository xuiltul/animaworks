# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Japanese, English, and Korean locale template parity."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
_JA_DIR = _TEMPLATES_DIR / "ja"
_EN_DIR = _TEMPLATES_DIR / "en"
_KO_DIR = _TEMPLATES_DIR / "ko"


def _ja_files() -> list[str]:
    return sorted(path.relative_to(_JA_DIR).as_posix() for path in _JA_DIR.rglob("*") if path.is_file())


def _markdown_files() -> list[str]:
    return [path for path in _ja_files() if path.endswith(".md")]


_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_.]*)\}(?!\})")


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks, where placeholders are illustrative text."""
    return re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)


def _extract_placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(_strip_code_blocks(text)))


@pytest.mark.parametrize("locale", ["ja", "en", "ko"])
def test_environment_enforces_worktree_and_credential_resolver(locale: str) -> None:
    repo_rules = (_TEMPLATES_DIR / locale / "prompts" / "builder" / "repo_work_rules.md").read_text(encoding="utf-8")
    behavior = (_TEMPLATES_DIR / locale / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
    assert "git worktree" in repo_rules
    assert "main" in repo_rules and "clean" in repo_rules
    assert "ANIMAWORKS_SCRATCH_DIR" in repo_rules
    assert "UV_CACHE_DIR" in repo_rules
    assert "secrets.json" in behavior


class TestKoTemplateFilesExist:
    """Every Japanese source file must have exactly one Korean counterpart."""

    @pytest.mark.parametrize("rel_path", _ja_files())
    def test_expected_file_exists(self, rel_path: str) -> None:
        assert (_KO_DIR / rel_path).is_file(), f"Missing ko template: {rel_path}"

    def test_locale_file_sets_match_ja(self) -> None:
        ja_files = set(_ja_files())
        en_files = {path.relative_to(_EN_DIR).as_posix() for path in _EN_DIR.rglob("*") if path.is_file()}
        ko_files = {path.relative_to(_KO_DIR).as_posix() for path in _KO_DIR.rglob("*") if path.is_file()}
        assert ko_files == ja_files, (
            f"ko file set differs from ja. Extra: {ko_files - ja_files}; missing: {ja_files - ko_files}"
        )
        assert en_files == ja_files, (
            f"en file set differs from ja. Extra: {en_files - ja_files}; missing: {ja_files - en_files}"
        )


class TestKoDirectoryStructure:
    _EXPECTED_DIRS = [
        "prompts",
        "prompts/builder",
        "prompts/fragments",
        "prompts/memory",
        "prompts/tool_descriptions",
        "prompts/tool_guides",
        "common_knowledge",
        "common_knowledge/anatomy",
        "common_knowledge/communication",
        "common_knowledge/operations",
        "common_knowledge/organization",
        "common_knowledge/security",
        "common_skills",
        "roles",
        "reference",
        "reference/anatomy",
        "reference/communication",
        "reference/operations",
        "reference/organization",
        "reference/troubleshooting",
        "reference/usecases",
        "anima_templates",
        "company",
    ]

    @pytest.mark.parametrize("locale", ["en", "ko"])
    @pytest.mark.parametrize("subdir", _EXPECTED_DIRS)
    def test_directory_exists(self, locale: str, subdir: str) -> None:
        assert (_TEMPLATES_DIR / locale / subdir).is_dir(), f"Missing {locale} directory: {subdir}"


_HEADING_EXEMPT = {
    "anima_templates/_blank/IDENTITY.md",
    "anima_templates/_blank/SOUL.md",
    "anima_templates/_blank/AGENTS.md",
    "anima_templates/_blank/cron.md",
    "prompts/builder/fallbacks.md",
    "prompts/builder/heartbeat_tool_instruction.md",
    "prompts/builder/sections.md",
    "prompts/character_design_guide.md",
    "prompts/chat_message.md",
    "prompts/chat_message_with_history.md",
    "prompts/cron_task.md",
    "prompts/greet.md",
    "prompts/heartbeat_default_checklist.md",
    "prompts/inbox_message.md",
    "prompts/task_complete_notify.md",
    "prompts/task_exec.md",
    "prompts/fragments/bg_task_notification.md",
    "prompts/fragments/command_output.md",
    "prompts/fragments/cron_rejected_notice.md",
    "prompts/fragments/recent_reflections.md",
    "prompts/fragments/recovery_note_header.md",
    "prompts/fragments/stale_task_scoreboard.md",
    "prompts/memory/conversation_compression.md",
    "prompts/memory/knowledge_revision.md",
    "prompts/memory/procedure_revision.md",
    "prompts/memory/weekly_pattern.md",
    "prompts/tool_guides/s_builtin.md",
    "prompts/tool_guides/s_mcp.md",
    "prompts/builder/emotion_instruction.md",
}


@pytest.mark.parametrize(
    "rel_path",
    [
        path
        for path in _markdown_files()
        if path not in _HEADING_EXEMPT and not path.startswith("prompts/tool_descriptions/")
    ],
)
def test_ko_markdown_has_heading(rel_path: str) -> None:
    heading_prefixes = ("## ", "### ", "#### ", "##### ", "###### ")
    ja_text = (_JA_DIR / rel_path).read_text(encoding="utf-8")
    if not any(line.startswith(heading_prefixes) for line in ja_text.splitlines()):
        pytest.skip("ja source has no ## headings")
    text = (_KO_DIR / rel_path).read_text(encoding="utf-8")
    assert any(line.startswith(heading_prefixes) for line in text.splitlines()), (
        f"{rel_path} has no ## or deeper headings"
    )


@pytest.mark.parametrize("rel_path", _markdown_files())
def test_placeholders_match_ja(rel_path: str) -> None:
    ja_placeholders = _extract_placeholders((_JA_DIR / rel_path).read_text(encoding="utf-8"))
    ko_placeholders = _extract_placeholders((_KO_DIR / rel_path).read_text(encoding="utf-8"))
    assert ko_placeholders == ja_placeholders, (
        f"{rel_path}: ko placeholders differ from ja; missing={ja_placeholders - ko_placeholders}, "
        f"extra={ko_placeholders - ja_placeholders}"
    )


@pytest.mark.parametrize("rel_path", _markdown_files())
def test_en_placeholders_match_ja(rel_path: str) -> None:
    ja_placeholders = _extract_placeholders((_JA_DIR / rel_path).read_text(encoding="utf-8"))
    en_placeholders = _extract_placeholders((_EN_DIR / rel_path).read_text(encoding="utf-8"))
    assert en_placeholders == ja_placeholders, (
        f"{rel_path}: en placeholders differ from ja; missing={ja_placeholders - en_placeholders}, "
        f"extra={en_placeholders - ja_placeholders}"
    )
