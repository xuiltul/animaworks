# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Korean (ko) locale template completeness.

Validates:
1. File parity — every file in templates/en/ has a corresponding file in templates/ko/
2. Placeholder consistency — {placeholder} variables in ko/ match those in en/
3. Directory structure — expected subdirectories exist
4. Heading validation — each .md file has ## or deeper headings

Pattern follows tests/unit/core/test_system_reference_documents.py (ja template tests).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
_EN_DIR = _TEMPLATES_DIR / "en"
_KO_DIR = _TEMPLATES_DIR / "ko"

# All expected files (mirrors templates/en/ structure exactly)
_EXPECTED_FILES = [
    "anima_templates/_blank/cron.md",
    "anima_templates/_blank/heartbeat.md",
    "anima_templates/_blank/identity.md",
    "anima_templates/_blank/injection.md",
    "anima_templates/_blank/permissions.json",
    "anima_templates/_blank/skills/newstaff.md",
    "anima_templates/_blank/skills/worker_management.md",
    "bootstrap.md",
    "common_knowledge/00_index.md",
    "common_knowledge/anatomy/machine-tool-philosophy.md",
    "common_knowledge/anatomy/memory-system.md",
    "common_knowledge/anatomy/task-architecture.md",
    "common_knowledge/anatomy/what-is-anima.md",
    "common_knowledge/communication/board-guide.md",
    "common_knowledge/communication/call-human-guide.md",
    "common_knowledge/communication/instruction-patterns.md",
    "common_knowledge/communication/messaging-guide.md",
    "common_knowledge/communication/reporting-guide.md",
    "common_knowledge/communication/sending-limits.md",
    "common_knowledge/operations/background-tasks.md",
    "common_knowledge/operations/heartbeat-cron-guide.md",
    "common_knowledge/operations/task-board-guide.md",
    "common_knowledge/operations/task-delegation-guide.md",
    "common_knowledge/operations/task-management.md",
    "common_knowledge/operations/tool-usage-overview.md",
    "common_knowledge/operations/report-formats.md",
    "common_knowledge/operations/workspace-guide.md",
    "common_knowledge/organization/hierarchy-rules.md",
    "common_knowledge/organization/roles.md",
    "common_knowledge/security/prompt-injection-awareness.md",
    "common_knowledge/troubleshooting/common-issues.md",
    "common_knowledge/troubleshooting/escalation-flowchart.md",
    "common_knowledge/usecases/usecase-communication.md",
    "common_knowledge/usecases/usecase-customer-support.md",
    "common_knowledge/usecases/usecase-development.md",
    "common_knowledge/usecases/usecase-knowledge.md",
    "common_knowledge/usecases/usecase-monitoring.md",
    "common_knowledge/usecases/usecase-overview.md",
    "common_knowledge/usecases/usecase-research.md",
    "common_knowledge/usecases/usecase-secretary.md",
    "common_skills/agent-browser/SKILL.md",
    "common_skills/animaworks-guide/SKILL.md",
    "common_skills/aws-collector-tool/SKILL.md",
    "common_skills/chatwork-tool/SKILL.md",
    "common_skills/cron-management/SKILL.md",
    "common_skills/google-tasks-tool/SKILL.md",
    "common_skills/github-tool/SKILL.md",
    "common_skills/gmail-tool/SKILL.md",
    "common_skills/google-calendar-tool/SKILL.md",
    "common_skills/image-gen-tool/SKILL.md",
    "common_skills/image-posting/SKILL.md",
    "common_skills/local-llm-tool/SKILL.md",
    "common_skills/machine-tool/SKILL.md",
    "common_skills/notion-tool/SKILL.md",
    "common_skills/skill-creator/SKILL.md",
    "common_skills/slack-tool/SKILL.md",
    "common_skills/subagent-cli/SKILL.md",
    "common_skills/subordinate-management/SKILL.md",
    "common_skills/tool-creator/SKILL.md",
    "common_skills/transcribe-tool/SKILL.md",
    "common_skills/web-search-tool/SKILL.md",
    "common_skills/workspace-manager/SKILL.md",
    "common_skills/x-search-tool/SKILL.md",
    "company/vision.md",
    "prompts/a_reflection.md",
    "prompts/behavior_rules.md",
    "prompts/builder/common_knowledge_hint.md",
    "prompts/builder/emotion_instruction.md",
    "prompts/builder/external_tools_guide.md",
    "prompts/builder/fallbacks.md",
    "prompts/builder/heartbeat_tool_instruction.md",
    "prompts/builder/hiring_rules_other.md",
    "prompts/builder/hiring_rules_s.md",
    "prompts/builder/human_notification.md",
    "prompts/builder/human_notification_howto_other.md",
    "prompts/builder/human_notification_howto_s.md",
    "prompts/builder/light_tier_org.md",
    "prompts/builder/org_context_toplevel.md",
    "prompts/builder/reference_hint.md",
    "prompts/builder/resolution_registry.md",
    "prompts/builder/sections.md",
    "prompts/builder/task_in_progress.md",
    "prompts/builder/task_queue.md",
    "prompts/character_design_guide.md",
    "prompts/chat_message.md",
    "prompts/chat_message_with_history.md",
    "prompts/communication_rules.md",
    "prompts/communication_rules_s.md",
    "prompts/cron_task.md",
    "prompts/environment.md",
    "prompts/fragments/asset_synthesis_system.md",
    "prompts/fragments/asset_synthesis_system_realistic.md",
    "prompts/fragments/bg_task_notification.md",
    "prompts/fragments/command_output.md",
    "prompts/fragments/recent_dialogue.md",
    "prompts/fragments/recent_reflections.md",
    "prompts/fragments/recovery_note_header.md",
    "prompts/greet.md",
    "prompts/heartbeat.md",
    "prompts/heartbeat_default_checklist.md",
    "prompts/heartbeat_history.md",
    "prompts/heartbeat_subordinate_check.md",
    "prompts/hiring_context.md",
    "prompts/inbox_message.md",
    "prompts/memory/classification.md",
    "prompts/memory/consolidation_instruction.md",
    "prompts/memory/consolidation_retry.md",
    "prompts/memory/contradiction_detection.md",
    "prompts/memory/contradiction_merge.md",
    "prompts/memory/conversation_compression.md",
    "prompts/memory/daily_consolidation.md",
    "prompts/memory/daily_consolidation_related.md",
    "prompts/memory/daily_consolidation_resolved.md",
    "prompts/memory/daily_consolidation_task.md",
    "prompts/memory/episode_compression.md",
    "prompts/memory/forgetting_merge.md",
    "prompts/memory/knowledge_merge.md",
    "prompts/memory/knowledge_revision.md",
    "prompts/memory/knowledge_validation.md",
    "prompts/memory/knowledge_validity_review.md",
    "prompts/memory/procedure_from_resolved.md",
    "prompts/memory/procedure_revision.md",
    "prompts/memory/session_summary.md",
    "prompts/memory/weekly_consolidation_instruction.md",
    "prompts/memory/weekly_pattern.md",
    "prompts/memory_guide.md",
    "prompts/messaging.md",
    "prompts/messaging_s.md",
    "prompts/org_context.md",
    "prompts/session_continuation.md",
    "prompts/task_complete_notify.md",
    "prompts/task_exec.md",
    "prompts/tool_data_interpretation.md",
    "prompts/unread_messages.md",
    "reference/00_index.md",
    "reference/anatomy/anima-anatomy.md",
    "reference/anatomy/priming-channels.md",
    "reference/anatomy/working-memory.md",
    "reference/communication/slack-bot-token-guide.md",
    "reference/internals/common-knowledge-access-paths.md",
    "reference/operations/browser-automation-guide.md",
    "reference/operations/mode-s-auth-guide.md",
    "reference/operations/model-guide.md",
    "reference/operations/project-setup.md",
    "reference/operations/voice-chat-guide.md",
    "reference/organization/structure.md",
    "reference/troubleshooting/gmail-credential-setup.md",
    "roles/engineer/permissions.json",
    "roles/engineer/specialty_prompt.md",
    "roles/general/permissions.json",
    "roles/general/specialty_prompt.md",
    "roles/manager/permissions.json",
    "roles/manager/specialty_prompt.md",
    "roles/ops/permissions.json",
    "roles/ops/specialty_prompt.md",
    "roles/researcher/permissions.json",
    "roles/researcher/specialty_prompt.md",
    "roles/writer/permissions.json",
    "roles/writer/specialty_prompt.md",
]

# Regex to extract {placeholder} variables (ignoring {{ escaped braces }})
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks (``` ... ```) from text.

    Placeholders inside code blocks are example text, not runtime variables.
    Both en and ko localize these freely (e.g., en: {task_name} → ko: {태스크명}).
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _extract_placeholders(text: str) -> set[str]:
    """Extract all {placeholder} names from template text, excluding code blocks."""
    return set(_PLACEHOLDER_RE.findall(_strip_code_blocks(text)))


# ── 1. File existence ────────────────────────────────────────


class TestKoTemplateFilesExist:
    """Verify all expected Korean template files are present."""

    @pytest.mark.parametrize("rel_path", _EXPECTED_FILES)
    def test_expected_file_exists(self, rel_path: str):
        full_path = _KO_DIR / rel_path
        assert full_path.exists(), f"Missing ko template: {rel_path}"

    def test_total_file_count(self):
        """File count in ko/ should match _EXPECTED_FILES."""
        all_files = sorted(
            str(f.relative_to(_KO_DIR))
            for f in _KO_DIR.rglob("*")
            if f.is_file()
        )
        assert len(all_files) == len(_EXPECTED_FILES), (
            f"Expected {len(_EXPECTED_FILES)} files, found {len(all_files)}. "
            f"Extra: {set(all_files) - set(_EXPECTED_FILES)}, "
            f"Missing: {set(_EXPECTED_FILES) - set(all_files)}"
        )

    def test_no_extra_files(self):
        """ko/ should not contain files that are not in en/."""
        en_files = {
            str(f.relative_to(_EN_DIR))
            for f in _EN_DIR.rglob("*")
            if f.is_file()
        }
        ko_files = {
            str(f.relative_to(_KO_DIR))
            for f in _KO_DIR.rglob("*")
            if f.is_file()
        }
        extra = ko_files - en_files
        assert not extra, f"ko/ has extra files not in en/: {extra}"


# ── 2. Directory structure ───────────────────────────────────


class TestKoDirectoryStructure:
    """Expected subdirectories should exist."""

    _EXPECTED_DIRS = [
        "prompts",
        "prompts/builder",
        "prompts/fragments",
        "prompts/memory",
        "common_knowledge",
        "common_knowledge/anatomy",
        "common_knowledge/communication",
        "common_knowledge/operations",
        "common_knowledge/organization",
        "common_knowledge/security",
        "common_knowledge/troubleshooting",
        "common_knowledge/usecases",
        "common_skills",
        "roles",
        "reference",
        "anima_templates",
        "company",
    ]

    @pytest.mark.parametrize("subdir", _EXPECTED_DIRS)
    def test_directory_exists(self, subdir: str):
        assert (_KO_DIR / subdir).is_dir(), f"Missing ko directory: {subdir}"


# ── 3. Heading validation ───────────────────────────────────


# Files that may legitimately have no ## headings (short fragments, placeholders)
# Files without ## headings in en/ source — exempt from heading check.
# Generated by: for f in $(find templates/en -name "*.md"); do grep -qL "^##" "$f" && echo; done
_HEADING_EXEMPT = {
    "anima_templates/_blank/identity.md",
    "anima_templates/_blank/injection.md",
    "anima_templates/_blank/cron.md",
    "prompts/builder/fallbacks.md",
    "prompts/builder/heartbeat_tool_instruction.md",
    "prompts/builder/human_notification_howto_other.md",
    "prompts/builder/human_notification_howto_s.md",
    "prompts/builder/light_tier_org.md",
    "prompts/builder/sections.md",
    "prompts/chat_message.md",
    "prompts/chat_message_with_history.md",
    "prompts/cron_task.md",
    "prompts/greet.md",
    "prompts/inbox_message.md",
    "prompts/session_continuation.md",
    "prompts/task_complete_notify.md",
    "prompts/task_exec.md",
    "prompts/fragments/bg_task_notification.md",
    "prompts/fragments/command_output.md",
    "prompts/fragments/recent_dialogue.md",
    "prompts/fragments/recent_reflections.md",
    "prompts/fragments/recovery_note_header.md",
    "prompts/memory/contradiction_detection.md",
    "prompts/memory/contradiction_merge.md",
    "prompts/memory/conversation_compression.md",
    "prompts/memory/daily_consolidation.md",
    "prompts/memory/daily_consolidation_related.md",
    "prompts/memory/daily_consolidation_resolved.md",
    "prompts/memory/forgetting_merge.md",
    "prompts/memory/knowledge_revision.md",
    "prompts/memory/knowledge_validation.md",
    "prompts/memory/knowledge_validity_review.md",
    "prompts/memory/procedure_revision.md",
    "prompts/memory/weekly_pattern.md",
}


class TestKoTemplateHeadings:
    """Each template file should have ## level headings."""

    @pytest.mark.parametrize(
        "rel_path",
        [f for f in _EXPECTED_FILES if f not in _HEADING_EXEMPT and f.endswith(".md")],
    )
    def test_file_has_headings(self, rel_path: str):
        full_path = _KO_DIR / rel_path
        if not full_path.exists():
            pytest.skip(f"File not found: {rel_path}")
        content = full_path.read_text(encoding="utf-8")
        has_heading = any(
            line.startswith("## ") or line.startswith("### ") or line.startswith("#### ")
            for line in content.splitlines()
        )
        assert has_heading, f"{rel_path} has no ## or deeper headings"


# ── 4. Placeholder consistency ───────────────────────────────


class TestKoPlaceholderConsistency:
    """Placeholder variables in ko/ should match those in en/.

    Only checks files under prompts/ — these are loaded by load_prompt() and
    have real runtime placeholders substituted via _SafeFormatDict.
    Files under common_knowledge/, reference/, etc. contain example/template
    placeholders (e.g., {task description}) that are meant to be localized.
    """

    _PROMPT_FILES = [f for f in _EXPECTED_FILES if f.startswith("prompts/")]

    @pytest.mark.parametrize("rel_path", _PROMPT_FILES)
    def test_placeholders_match(self, rel_path: str):
        en_path = _EN_DIR / rel_path
        ko_path = _KO_DIR / rel_path
        if not ko_path.exists():
            pytest.skip(f"ko file not found: {rel_path}")

        en_text = en_path.read_text(encoding="utf-8")
        ko_text = ko_path.read_text(encoding="utf-8")

        en_placeholders = _extract_placeholders(en_text)
        ko_placeholders = _extract_placeholders(ko_text)

        missing = en_placeholders - ko_placeholders
        extra = ko_placeholders - en_placeholders

        assert not missing, (
            f"{rel_path}: ko is missing placeholders from en: {missing}"
        )
        assert not extra, (
            f"{rel_path}: ko has extra placeholders not in en: {extra}"
        )
