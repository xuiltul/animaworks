from __future__ import annotations

from pathlib import Path

import pytest


TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates" / "prompts"


class TestCommunicationRulesTemplate:
    def test_has_task_delegation_section(self):
        path = TEMPLATES_DIR / "communication_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "### タスク委任プロトコル" in content

    def test_delegation_protocol_rules(self):
        path = TEMPLATES_DIR / "communication_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "原文引用" in content
        assert "完了条件" in content
        assert "確認応答" in content

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
        assert "add_task" in content
        assert "解決済み案件の再報告禁止" in content

    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "behavior_rules.md"
        content = path.read_text(encoding="utf-8")
        assert "## 行動ルール" in content


class TestUnreadMessagesTemplate:
    def test_has_task_delegation_receipt(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "### タスク委任の受領" in content

    def test_delegation_receipt_steps(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "Ack返答" in content or "Ack" in content
        assert "パラフレーズ" in content

    def test_existing_sections_preserved(self):
        path = TEMPLATES_DIR / "unread_messages.md"
        content = path.read_text(encoding="utf-8")
        assert "## 未読メッセージ" in content
