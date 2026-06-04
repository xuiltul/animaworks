from __future__ import annotations

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
            content = (
                TEMPLATES_ROOT / locale / "common_knowledge" / "operations" / "action-rules-guide.md"
            ).read_text(encoding="utf-8")
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
            hint = (
                TEMPLATES_ROOT / locale / "prompts" / "builder" / "common_knowledge_hint.md"
            ).read_text(encoding="utf-8")
            for content in (index, hint):
                assert "operations/action-rules-guide.md" in content
                assert "common_skills/skill-creator/SKILL.md" in content


class TestHeartbeatToolInstructionTemplate:
    def test_heartbeat_mentions_lightweight_skill_authoring_all_locales(self):
        for locale in LOCALES:
            content = (
                TEMPLATES_ROOT / locale / "prompts" / "builder" / "heartbeat_tool_instruction.md"
            ).read_text(encoding="utf-8")
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
            content = (
                TEMPLATES_ROOT / locale / "common_skills" / "skill-creator" / "SKILL.md"
            ).read_text(encoding="utf-8")
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
