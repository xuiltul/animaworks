from __future__ import annotations

import re
from pathlib import Path

from core.memory.priming.constants import (
    _BUDGET_GRAPH_CONTEXT,
    _BUDGET_GREETING,
    _BUDGET_HEARTBEAT,
    _BUDGET_IMPORTANT_KNOWLEDGE,
    _BUDGET_PENDING_TASKS,
    _BUDGET_QUESTION,
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_EPISODES,
    _BUDGET_RELATED_KNOWLEDGE,
    _BUDGET_REQUEST,
    _BUDGET_SENDER_PROFILE,
)

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
    def test_has_task_recording_section(self):
        path = TEMPLATES_DIR / "behavior_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "### タスク記録と報告" in content

    def test_task_recording_rules(self):
        path = TEMPLATES_DIR / "behavior_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "通常チャットでは `submit_tasks` を使わない" in content
        assert "解決済み案件の再報告禁止" in content

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
            "heartbeat.md",
            "cron.md",
            "current_state.md",
            "create_skill",
        ]
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "prompts" / "behavior_rules.md").read_text(encoding="utf-8")
            for token in required:
                assert token in content, f"{locale} behavior_rules missing {token}"
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
            assert "fail-open" in content
            assert "slack_post" not in content
            for tool in required_tools:
                assert f"`{tool}`" in content, f"{locale} action-rules-guide missing {tool}"
            for cli in cli_mappings:
                assert cli in content, f"{locale} action-rules-guide missing {cli}"

    def test_indexes_and_hints_point_to_action_rules_and_skill_creator(self):
        for locale in LOCALES:
            index = (TEMPLATES_ROOT / locale / "common_knowledge" / "00_index.md").read_text(encoding="utf-8")
            hint = (TEMPLATES_ROOT / locale / "prompts" / "builder" / "common_knowledge_hint.md").read_text(
                encoding="utf-8"
            )
            for content in (index, hint):
                assert "operations/action-rules-guide.md" in content
                assert "common_skills/skill-creator/SKILL.md" in content


class TestPrimingChannelsReference:
    channel_budget_constants = {
        "A": _BUDGET_SENDER_PROFILE,
        "B": _BUDGET_RECENT_ACTIVITY,
        "C": _BUDGET_RELATED_KNOWLEDGE,
        "C0": _BUDGET_IMPORTANT_KNOWLEDGE,
        "E": _BUDGET_PENDING_TASKS,
        "F": _BUDGET_RELATED_EPISODES,
        "G": _BUDGET_GRAPH_CONTEXT,
    }
    message_budget_constants = {
        "priming.budget_greeting": _BUDGET_GREETING,
        "priming.budget_question": _BUDGET_QUESTION,
        "priming.budget_request": _BUDGET_REQUEST,
        "priming.budget_heartbeat": _BUDGET_HEARTBEAT,
    }

    def test_channel_overview_budgets_match_constants_all_locales(self):
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "reference" / "anatomy" / "priming-channels.md").read_text(
                encoding="utf-8"
            )
            rows = dict(re.findall(r"^\| (A|B|C|C0|E|F|G): [^|]+ \| (\d+) \|", content, re.MULTILINE))
            assert rows == {channel: str(budget) for channel, budget in self.channel_budget_constants.items()}, (
                f"{locale} priming channel overview budget drift"
            )

    def test_channel_section_budgets_match_constants_all_locales(self):
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "reference" / "anatomy" / "priming-channels.md").read_text(
                encoding="utf-8"
            )
            for channel, expected in self.channel_budget_constants.items():
                section = content.split(f"## Channel {channel}:", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
                match = re.search(r"\*\*[^*]*(?:Budget|バジェット|버짓)[^*]*\*\*:\s*(\d+)", section)
                assert match, f"{locale} Channel {channel} missing detailed budget"
                assert int(match.group(1)) == expected, f"{locale} Channel {channel} detailed budget drift"

    def test_message_type_budgets_match_constants_all_locales(self):
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "reference" / "anatomy" / "priming-channels.md").read_text(
                encoding="utf-8"
            )
            for config_key, expected in self.message_budget_constants.items():
                match = re.search(rf"^\| [^|]+ \| (\d+) \| `{re.escape(config_key)}` \|", content, re.MULTILINE)
                assert match, f"{locale} missing {config_key} budget row"
                assert int(match.group(1)) == expected, f"{locale} {config_key} budget drift"

    def test_current_priming_docs_have_no_obsolete_channel_or_skill_tool_references(self):
        paths = list(Path(".").glob("README*.md"))
        historical_dirs = {
            "analysis",
            "drafts",
            "implemented",
            "investigations",
            "legacy",
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
    def test_heartbeat_mentions_lightweight_skill_authoring_all_locales(self):
        for locale in LOCALES:
            content = (TEMPLATES_ROOT / locale / "prompts" / "builder" / "heartbeat_tool_instruction.md").read_text(
                encoding="utf-8"
            )
            assert "create_skill" in content
            assert "submit_tasks" in content


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
    def test_has_task_delegation_receipt(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "### タスク委任の受領" in content

    def test_delegation_receipt_steps(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "パラフレーズ確認" in content
        assert "不明点の質問" in content

    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "## 未読メッセージ" in content
