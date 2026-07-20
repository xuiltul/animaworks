# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Korean (ko) locale template completeness.

Validates:
1. File inventory — every file under templates/ko/ is listed (sorted); ko may omit some en-only files
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

# All expected files (sorted; must match templates/ko/ exactly)
_EXPECTED_FILES = [
    "anima_templates/_blank/cron.md",
    "anima_templates/_blank/heartbeat.md",
    "anima_templates/_blank/identity.md",
    "anima_templates/_blank/injection.md",
    "anima_templates/_blank/permissions.json",
    "anima_templates/_blank/skills/newstaff/SKILL.md",
    "anima_templates/_blank/skills/worker_management/SKILL.md",
    "bootstrap.md",
    "common_knowledge/00_index.md",
    "common_knowledge/anatomy/essentials.md",
    "common_knowledge/anatomy/machine-tool-philosophy.md",
    "common_knowledge/anatomy/task-architecture.md",
    "common_knowledge/anatomy/what-is-anima.md",
    "common_knowledge/communication/board-guide.md",
    "common_knowledge/communication/call-human-guide.md",
    "common_knowledge/communication/sending-limits.md",
    "common_knowledge/operations/action-rules-guide.md",
    "common_knowledge/operations/background-tasks.md",
    "common_knowledge/operations/machine/tool-usage.md",
    "common_knowledge/operations/machine/workflow-engineer.md",
    "common_knowledge/operations/machine/workflow-pdm.md",
    "common_knowledge/operations/machine/workflow-reviewer.md",
    "common_knowledge/operations/machine/workflow-tester.md",
    "common_knowledge/operations/report-formats.md",
    "common_knowledge/operations/task-board-guide.md",
    "common_knowledge/operations/task-delegation-guide.md",
    "common_knowledge/operations/workspace-guide.md",
    "common_knowledge/organization/hierarchy-rules.md",
    "common_knowledge/security/prompt-injection-awareness.md",
    "common_knowledge/team-design/coo/coo/checklist.md",
    "common_knowledge/team-design/coo/coo/injection.template.md",
    "common_knowledge/team-design/coo/coo/machine.md",
    "common_knowledge/team-design/coo/team.md",
    "common_knowledge/team-design/corporate-planning/analyst/checklist.md",
    "common_knowledge/team-design/corporate-planning/analyst/injection.template.md",
    "common_knowledge/team-design/corporate-planning/analyst/machine.md",
    "common_knowledge/team-design/corporate-planning/coordinator/checklist.md",
    "common_knowledge/team-design/corporate-planning/coordinator/injection.template.md",
    "common_knowledge/team-design/corporate-planning/coordinator/machine.md",
    "common_knowledge/team-design/corporate-planning/strategist/checklist.md",
    "common_knowledge/team-design/corporate-planning/strategist/injection.template.md",
    "common_knowledge/team-design/corporate-planning/strategist/machine.md",
    "common_knowledge/team-design/corporate-planning/team.md",
    "common_knowledge/team-design/customer-success/cs-lead/checklist.md",
    "common_knowledge/team-design/customer-success/cs-lead/injection.template.md",
    "common_knowledge/team-design/customer-success/cs-lead/machine.md",
    "common_knowledge/team-design/customer-success/support/checklist.md",
    "common_knowledge/team-design/customer-success/support/injection.template.md",
    "common_knowledge/team-design/customer-success/team.md",
    "common_knowledge/team-design/development/engineer/checklist.md",
    "common_knowledge/team-design/development/engineer/injection.template.md",
    "common_knowledge/team-design/development/engineer/machine.md",
    "common_knowledge/team-design/development/pdm/checklist.md",
    "common_knowledge/team-design/development/pdm/injection.template.md",
    "common_knowledge/team-design/development/pdm/machine.md",
    "common_knowledge/team-design/development/reviewer/checklist.md",
    "common_knowledge/team-design/development/reviewer/injection.template.md",
    "common_knowledge/team-design/development/reviewer/machine.md",
    "common_knowledge/team-design/development/team.md",
    "common_knowledge/team-design/development/tester/checklist.md",
    "common_knowledge/team-design/development/tester/injection.template.md",
    "common_knowledge/team-design/development/tester/machine.md",
    "common_knowledge/team-design/finance/analyst/checklist.md",
    "common_knowledge/team-design/finance/analyst/injection.template.md",
    "common_knowledge/team-design/finance/auditor/checklist.md",
    "common_knowledge/team-design/finance/auditor/injection.template.md",
    "common_knowledge/team-design/finance/auditor/machine.md",
    "common_knowledge/team-design/finance/collector/checklist.md",
    "common_knowledge/team-design/finance/collector/injection.template.md",
    "common_knowledge/team-design/finance/director/checklist.md",
    "common_knowledge/team-design/finance/director/injection.template.md",
    "common_knowledge/team-design/finance/director/machine.md",
    "common_knowledge/team-design/finance/team.md",
    "common_knowledge/team-design/guide.md",
    "common_knowledge/team-design/infrastructure/director/checklist.md",
    "common_knowledge/team-design/infrastructure/director/injection.template.md",
    "common_knowledge/team-design/infrastructure/monitor/checklist.md",
    "common_knowledge/team-design/infrastructure/monitor/injection.template.md",
    "common_knowledge/team-design/infrastructure/team.md",
    "common_knowledge/team-design/legal/director/checklist.md",
    "common_knowledge/team-design/legal/director/injection.template.md",
    "common_knowledge/team-design/legal/director/machine.md",
    "common_knowledge/team-design/legal/researcher/checklist.md",
    "common_knowledge/team-design/legal/researcher/injection.template.md",
    "common_knowledge/team-design/legal/team.md",
    "common_knowledge/team-design/legal/verifier/checklist.md",
    "common_knowledge/team-design/legal/verifier/injection.template.md",
    "common_knowledge/team-design/legal/verifier/machine.md",
    "common_knowledge/team-design/org-chart-template.md",
    "common_knowledge/team-design/sales-marketing/creator/checklist.md",
    "common_knowledge/team-design/sales-marketing/creator/injection.template.md",
    "common_knowledge/team-design/sales-marketing/creator/machine.md",
    "common_knowledge/team-design/sales-marketing/director/checklist.md",
    "common_knowledge/team-design/sales-marketing/director/injection.template.md",
    "common_knowledge/team-design/sales-marketing/director/machine.md",
    "common_knowledge/team-design/sales-marketing/researcher/checklist.md",
    "common_knowledge/team-design/sales-marketing/researcher/injection.template.md",
    "common_knowledge/team-design/sales-marketing/sdr/checklist.md",
    "common_knowledge/team-design/sales-marketing/sdr/injection.template.md",
    "common_knowledge/team-design/sales-marketing/sdr/machine.md",
    "common_knowledge/team-design/sales-marketing/team.md",
    "common_knowledge/team-design/secretary/secretary/checklist.md",
    "common_knowledge/team-design/secretary/secretary/injection.template.md",
    "common_knowledge/team-design/secretary/secretary/machine.md",
    "common_knowledge/team-design/secretary/team.md",
    "common_knowledge/team-design/trading/analyst/checklist.md",
    "common_knowledge/team-design/trading/analyst/injection.template.md",
    "common_knowledge/team-design/trading/analyst/machine.md",
    "common_knowledge/team-design/trading/auditor/checklist.md",
    "common_knowledge/team-design/trading/auditor/injection.template.md",
    "common_knowledge/team-design/trading/auditor/machine.md",
    "common_knowledge/team-design/trading/director/checklist.md",
    "common_knowledge/team-design/trading/director/injection.template.md",
    "common_knowledge/team-design/trading/director/machine.md",
    "common_knowledge/team-design/trading/engineer/checklist.md",
    "common_knowledge/team-design/trading/engineer/injection.template.md",
    "common_knowledge/team-design/trading/engineer/machine.md",
    "common_knowledge/team-design/trading/team.md",
    "common_skills/agent-browser/SKILL.md",
    "common_skills/animaworks-guide/SKILL.md",
    "common_skills/aws-collector-tool/SKILL.md",
    "common_skills/chatwork-tool/SKILL.md",
    "common_skills/cron-management/SKILL.md",
    "common_skills/github-tool/SKILL.md",
    "common_skills/gmail-tool/SKILL.md",
    "common_skills/google-calendar-tool/SKILL.md",
    "common_skills/google-tasks-tool/SKILL.md",
    "common_skills/image-gen-tool/SKILL.md",
    "common_skills/image-posting/SKILL.md",
    "common_skills/local-llm-tool/SKILL.md",
    "common_skills/machine-tool/SKILL.md",
    "common_skills/notion-tool/SKILL.md",
    "common_skills/skill-creator/SKILL.md",
    "common_skills/skill-creator/references/description_guide.md",
    "common_skills/skill-creator/scripts/lint_skill.py",
    "common_skills/skill-creator/templates/skill_template.md",
    "common_skills/slack-tool/SKILL.md",
    "common_skills/subagent-cli/SKILL.md",
    "common_skills/subordinate-management/SKILL.md",
    "common_skills/tool-creator/SKILL.md",
    "common_skills/transcribe-tool/SKILL.md",
    "common_skills/web-search-tool/SKILL.md",
    "common_skills/workspace-manager/SKILL.md",
    "common_skills/x-search-tool/SKILL.md",
    "common_skills/zoom-meeting-scribe/SKILL.md",
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
    "prompts/fragments/curator_report_review.md",
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
    "prompts/memory/contradiction_detection.md",
    "prompts/memory/contradiction_merge.md",
    "prompts/memory/conversation_compression.md",
    "prompts/memory/episode_extraction.md",
    "prompts/memory/forgetting_merge.md",
    "prompts/memory/knowledge_revision.md",
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
    "prompts/tool_descriptions/Bash.md",
    "prompts/tool_descriptions/Edit.md",
    "prompts/tool_descriptions/Glob.md",
    "prompts/tool_descriptions/Grep.md",
    "prompts/tool_descriptions/Read.md",
    "prompts/tool_descriptions/WebFetch.md",
    "prompts/tool_descriptions/WebSearch.md",
    "prompts/tool_descriptions/Write.md",
    "prompts/tool_descriptions/archive_memory_file.md",
    "prompts/tool_descriptions/backlog_task.md",
    "prompts/tool_descriptions/call_human.md",
    "prompts/tool_descriptions/create_anima.md",
    "prompts/tool_descriptions/list_tasks.md",
    "prompts/tool_descriptions/post_channel.md",
    "prompts/tool_descriptions/read_channel.md",
    "prompts/tool_descriptions/read_dm_history.md",
    "prompts/tool_descriptions/read_memory_file.md",
    "prompts/tool_descriptions/refresh_tools.md",
    "prompts/tool_descriptions/report_knowledge_outcome.md",
    "prompts/tool_descriptions/report_procedure_outcome.md",
    "prompts/tool_descriptions/search_memory.md",
    "prompts/tool_descriptions/send_message.md",
    "prompts/tool_descriptions/share_tool.md",
    "prompts/tool_descriptions/update_task.md",
    "prompts/tool_descriptions/write_memory_file.md",
    "prompts/tool_guides/non_s.md",
    "prompts/tool_guides/s_builtin.md",
    "prompts/tool_guides/s_mcp.md",
    "prompts/unread_messages.md",
    "reference/00_index.md",
    "reference/anatomy/anima-anatomy.md",
    "reference/anatomy/memory-system.md",
    "reference/anatomy/priming-channels.md",
    "reference/anatomy/working-memory.md",
    "reference/communication/instruction-patterns.md",
    "reference/communication/messaging-guide.md",
    "reference/communication/reporting-guide.md",
    "reference/communication/slack-bot-token-guide.md",
    "reference/internals/common-knowledge-access-paths.md",
    "reference/operations/browser-automation-guide.md",
    "reference/operations/heartbeat-cron-guide.md",
    "reference/operations/mode-s-auth-guide.md",
    "reference/operations/model-guide.md",
    "reference/operations/project-setup.md",
    "reference/operations/task-management.md",
    "reference/operations/tool-usage-overview.md",
    "reference/operations/voice-chat-guide.md",
    "reference/organization/roles.md",
    "reference/organization/structure.md",
    "reference/troubleshooting/common-issues.md",
    "reference/troubleshooting/escalation-flowchart.md",
    "reference/troubleshooting/gmail-credential-setup.md",
    "reference/usecases/usecase-communication.md",
    "reference/usecases/usecase-customer-support.md",
    "reference/usecases/usecase-development.md",
    "reference/usecases/usecase-knowledge.md",
    "reference/usecases/usecase-monitoring.md",
    "reference/usecases/usecase-overview.md",
    "reference/usecases/usecase-research.md",
    "reference/usecases/usecase-secretary.md",
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
        all_files = sorted(str(f.relative_to(_KO_DIR)) for f in _KO_DIR.rglob("*") if f.is_file())
        assert len(all_files) == len(_EXPECTED_FILES), (
            f"Expected {len(_EXPECTED_FILES)} files, found {len(all_files)}. "
            f"Extra: {set(all_files) - set(_EXPECTED_FILES)}, "
            f"Missing: {set(_EXPECTED_FILES) - set(all_files)}"
        )

    def test_no_extra_files(self):
        """ko/ should not contain files that are not in en/."""
        en_files = {str(f.relative_to(_EN_DIR)) for f in _EN_DIR.rglob("*") if f.is_file()}
        ko_files = {str(f.relative_to(_KO_DIR)) for f in _KO_DIR.rglob("*") if f.is_file()}
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
    "prompts/memory/forgetting_merge.md",
    "prompts/memory/knowledge_revision.md",
    "prompts/memory/procedure_revision.md",
    "prompts/memory/weekly_pattern.md",
    "prompts/tool_guides/s_builtin.md",
}


class TestKoTemplateHeadings:
    """Each template file should have ## level headings."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            f
            for f in _EXPECTED_FILES
            if f not in _HEADING_EXEMPT
            and not f.startswith("prompts/tool_descriptions/")
            and f.endswith(".md")
        ],
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

        assert not missing, f"{rel_path}: ko is missing placeholders from en: {missing}"
        assert not extra, f"{rel_path}: ko has extra placeholders not in en: {extra}"
