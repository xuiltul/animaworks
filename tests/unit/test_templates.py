from __future__ import annotations

import re
from pathlib import Path

TEMPLATES_ROOT = Path(__file__).parent.parent.parent / "templates"
TEMPLATES_DIR = TEMPLATES_ROOT / "ja" / "prompts"
LOCALES = ("ja", "en", "ko")


class TestCommunicationRulesTemplate:
    def test_has_delegation_keyword(self):
        path = TEMPLATES_DIR / "communication_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "委任" in content

    def test_delegation_protocol_rules(self):
        path = TEMPLATES_DIR / "communication_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "原文引用" in content
        assert "完了条件" in content
        assert "パラフレーズ確認" in content

    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "communication_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "## コミュニケーションルール" in content


class TestBehaviorRulesTemplate:
    def test_task_recording_rules(self):
        behavior = (TEMPLATES_DIR / "behavior_rules.md").read_text(encoding="utf-8")
        assert "backlog_task" in behavior
        assert "完了条件の証跡" in behavior
        assert "resume" in behavior

    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "behavior_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "## 行動ルール" in content

    def test_artifact_selection_rules_all_locales(self):
        required = [
            "[ACTION-RULE]",
            "knowledge/",
            "procedures/",
            "skills/{name}/SKILL.md",
            "HEARTBEAT.md",
            "cron.md",
            "current_state.md",
            "create_skill",
        ]
        for locale in LOCALES:
            locale_root = TEMPLATES_ROOT / locale
            content = "\n".join(
                (
                    (locale_root / "prompts" / "behavior_rules.md").read_text(encoding="utf-8"),
                    (locale_root / "reference" / "operations" / "memory-writing-guide.md").read_text(encoding="utf-8"),
                    (
                        locale_root / "common_knowledge" / "operations" / "action-rule-memory-write-destination.md"
                    ).read_text(encoding="utf-8"),
                )
            )
            for token in required:
                assert token in content, f"{locale} behavior_rules missing {token}"
            behavior = (locale_root / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
            assert "backlog_task" in behavior
            assert "update_task" in behavior
            assert 'update_task(status="in_progress")' not in content
            assert "must be registered with `submit_tasks`" not in content
            assert "必ず `submit_tasks` でタスクキューに登録" not in content
            assert "반드시 `submit_tasks`로 태스크 큐에 등록" not in content


class TestActionRulesGuideTemplate:
    def test_action_rule_guides_current_across_locales(self):
        required_tools = {
            "call_human",
            "send_message",
            "post_channel",
            "write_memory_file",
            "gmail_draft",
            "gmail_send",
            "chatwork_send",
            "slack_send",
            "discord_send",
        }
        cli_mappings = [
            "animaworks-tool gmail draft",
            "animaworks-tool gmail send",
            "animaworks-tool chatwork send",
            "animaworks-tool slack send",
            "animaworks-tool discord send",
            "animaworks-tool call_human",
        ]
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "common_knowledge" / "operations" / "action-rules-guide.md").read_text(
                encoding="utf-8"
            )
            assert "[ACTION-RULE]" in content
            assert "trigger_tools" in content
            assert "read_memory_file(path=" in content
            assert "0.80" in content
            assert "fail-open" in content or "fail open" in content
            assert "slack_post" not in content
            for tool in required_tools:
                assert f"`{tool}`" in content, f"{locale} action-rules-guide missing {tool}"
            for cli in cli_mappings:
                assert cli in content, f"{locale} action-rules-guide missing {cli}"

    def test_indexes_and_hints_point_to_action_rules_without_skill_creator_duplication(self):
        for locale in LOCALES:
            index = (TEMPLATES_ROOT / locale / "common_knowledge" / "00_index.md").read_text(encoding="utf-8")
            hint = (TEMPLATES_ROOT / locale / "prompts" / "memory_guide.md").read_text(encoding="utf-8")
            assert "operations/action-rules-guide.md" in index
            assert "operations/action-rules-guide.md" in hint
            assert "common_skills/skill-creator/SKILL.md" not in hint

    def test_current_priming_docs_have_no_obsolete_channel_or_skill_tool_references(self):
        paths = list(Path(".").glob("README*.md"))
        historical_dirs = {
            "analysis",
            "drafts",
            "implemented",
            "investigations",
            "legacy",
            "plans",
            "records",
            "reports",
            "research",
        }
        paths.extend(
            path
            for root in (Path("docs"), Path("templates"))
            for path in root.rglob("*.md")
            if not any(part in historical_dirs for part in path.parts)
        )
        pattern = re.compile(
            r"Channel D|channel D|channel_d|D:\s*Skill Match|"
            r"`skill`\s*(?:/|\||tool|ツール|도구)|`skill` tool|skill tool"
        )
        for path in paths:
            content = path.read_text(encoding="utf-8")
            assert not pattern.search(content), f"{path} contains obsolete priming/skill-tool wording"


class TestHeartbeatToolInstructionTemplate:
    def test_heartbeat_uses_explicit_resume_and_authoritative_task_read_all_locales(self):
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "prompts" / "builder" / "heartbeat_tool_instruction.md").read_text(
                encoding="utf-8"
            )
            assert "submit_tasks" in content
            assert "resume" in content
            assert "list_tasks(detail=true)" in content


class TestSkillCreatorTemplate:
    def test_skill_creator_docs_match_create_skill_schema_metadata(self):
        fields = [
            "allowed_tools",
            "trust_level",
            "source_type",
            "source_origin",
            "category",
            "promotion_status",
            "skill_policy",
            "use_when",
            "trigger_phrases",
            "negative_phrases",
            "domains",
            "routing_examples",
        ]
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "common_skills" / "skill-creator" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            assert "create_skill" in content
            for field in fields:
                assert field in content, f"{locale} skill-creator missing {field}"


class TestUnreadMessagesTemplate:
    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "## 未読メッセージ" in content

    def test_inbox_closing_line_appears_once_for_all_locales(self):
        """B2: the inbox user_message must contain the closing instruction once."""
        from core.paths import load_prompt

        for locale in LOCALES:
            # The outer inbox template wraps the unread-messages block.
            unread = load_prompt("unread_messages", locale=locale, summary="dummy-summary")
            prompt = load_prompt("inbox_message", locale=locale, messages=unread)
            # The consolidated delegate portion should appear exactly once.
            assert "delegate_task" in prompt
            assert prompt.count("delegate_task") == 1
            # No stale duplicate closing uses "直接答え" (removed inbox wording) or
            # the old unread-messages trailing instruction as a second close.
            assert prompt.count("thread_id") == 1

    def test_unread_messages_has_no_independent_closing(self):
        """B2: unread_messages is a pure content block with no second closing."""
        from core.paths import load_prompt_text

        for locale in LOCALES:
            content = load_prompt_text("unread_messages", locale=locale)
            assert "delegate_task" not in content
            assert "thread_id" not in content


class TestRuntimeTemplateAccuracy:
    def test_ja_templates_have_no_retired_or_invalid_runtime_references(self):
        forbidden = (
            "deadline=",
            "sync_delegated",
            "OVERDUE",
            "dynamic_budget",
            "budget_greeting",
            "export-sections",
            "animaworks slack ",
            "migrate --resync-db",
            "max_chains",
            "max_turns",
            "Max Chains",
            "Max Turns",
        )
        for path in (TEMPLATES_ROOT / "ja").rglob("*.md"):
            content = path.read_text(encoding="utf-8")
            for term in forbidden:
                assert term not in content, f"{path} contains retired reference {term!r}"

    def test_tool_usage_overview_lists_every_exposed_mcp_tool(self):
        from core.tooling.policy.surface import MCP_TOOL_NAMES

        overview = (TEMPLATES_ROOT / "ja" / "reference" / "operations" / "tool-usage-overview.md").read_text(
            encoding="utf-8"
        )
        missing = sorted(name for name in MCP_TOOL_NAMES if name not in overview)
        assert not missing, f"MCP tools missing from the reference: {missing}"

    def test_ja_templates_do_not_contain_organization_specific_anima_names(self):
        ja_root = TEMPLATES_ROOT / "ja"
        for path in ja_root.rglob("*.md"):
            if "anima_templates" in path.relative_to(ja_root).parts:
                continue
            content = path.read_text(encoding="utf-8")
            assert not re.search(r"\btaka\b", content), f"{path} contains an organization-specific name"

    def test_delegate_task_required_arguments_match_the_subordinate_guide(self):
        from core.tooling.policy.schemas.supervisor import _supervisor_tools

        schema = next(tool for tool in _supervisor_tools() if tool["name"] == "delegate_task")
        required = set(schema["parameters"]["required"])
        guide = (TEMPLATES_ROOT / "ja" / "common_skills" / "subordinate-management" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        required_line = next(line for line in guide.splitlines() if line.startswith("# 必須:"))
        required_text = required_line.split("任意:", maxsplit=1)[0]
        documented = set(re.findall(r"`([a-z_]+)`", required_text))
        assert documented == required
