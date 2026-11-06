from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import ConfigError


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "mei"
    (d / "knowledge").mkdir(parents=True)
    (d / "procedures").mkdir()
    (d / "procedures" / "check.md").write_text("# Check\n\nConfirm before action.\n", encoding="utf-8")
    return d


@pytest.fixture
def handler(anima_dir: Path):
    from core.tooling.handler import ToolHandler

    memory = MagicMock()
    memory.search_memory_text.return_value = []
    with patch("core.config.models.load_config", side_effect=ConfigError("skip subordinate cache")):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=["gmail"])
    h._external.dispatch = MagicMock(return_value="draft ok")
    return h


def test_find_action_rules_returns_only_above_threshold(anima_dir: Path, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    results = [
        FakeRule("r-high", "## [ACTION-RULE] high", 0.95),
        FakeRule("r-low", "## [ACTION-RULE] low", 0.79),
    ]
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: results)

    rules = action_gate.find_action_rules(anima_dir, "gmail_send", {"to": "a@example.com"})

    assert [r.rule_id for r in rules] == ["r-high"]


def test_find_action_rules_sorts_by_score_desc(anima_dir: Path, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    results = [
        FakeRule("r1", "body", 0.81),
        FakeRule("r3", "body", 0.99),
        FakeRule("r2", "body", 0.85),
    ]
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: results)

    rules = action_gate.find_action_rules(anima_dir, "gmail_send", {})

    assert [r.rule_id for r in rules] == ["r3", "r2", "r1"]


def test_find_action_rules_caps_at_three(anima_dir: Path, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    results = [FakeRule(f"r{i}", "body", 0.90 + i * 0.01) for i in range(6)]
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: results)

    rules = action_gate.find_action_rules(anima_dir, "gmail_send", {})

    assert len(rules) == 3


def test_find_action_rules_truncates_body_to_2000(anima_dir: Path, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    long_body = "x" * 5000
    monkeypatch.setattr(
        action_gate,
        "_search_action_rules",
        lambda *args, **kwargs: [FakeRule("r-long", long_body, 0.98)],
    )

    rules = action_gate.find_action_rules(anima_dir, "gmail_send", {})

    assert len(rules[0].content) == 2000


def test_find_action_rules_returns_empty_on_search_exception(anima_dir: Path, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    def raise_search(*args, **kwargs):
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    assert action_gate.find_action_rules(anima_dir, "gmail_send", {}) == []


def test_find_action_rules_empty_tool_name(anima_dir: Path) -> None:
    from core.tooling.policy import action_gate

    assert action_gate.find_action_rules(anima_dir, "", {}) == []


def test_format_action_rules_contains_tag_and_body() -> None:
    from core.tooling.policy.action_gate import ActionRule, format_action_rules

    rendered = format_action_rules(
        [ActionRule(rule_id="mei/knowledge/rule.md#0", content="## [ACTION-RULE] check", score=0.87)]
    )

    assert '<action-rule path="mei/knowledge/rule.md#0" score="0.87">' in rendered
    assert "## [ACTION-RULE] check" in rendered
    assert "</action-rule>" in rendered


def test_format_action_rules_empty_returns_empty_string() -> None:
    from core.tooling.policy.action_gate import format_action_rules

    assert format_action_rules([]) == ""


def test_handler_appends_rules_to_result(anima_dir: Path, handler, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    rule = FakeRule(
        "rule-1",
        "## [ACTION-RULE] Gmail draft check\ntrigger_tools: gmail_draft\n---\nConfirm the recipient.",
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    result = handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"})

    assert result.startswith("draft ok")
    assert '<action-rule path="rule-1"' in result
    assert "Confirm the recipient." in result
    handler._external.dispatch.assert_called_once()


def test_handler_without_action_rules_returns_result_unchanged(anima_dir: Path, handler, monkeypatch) -> None:
    from core.tooling.policy import action_gate

    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [])

    result = handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"})

    assert result == "draft ok"


def test_handler_attaches_for_core_side_effect_tool(anima_dir: Path, handler, monkeypatch) -> None:
    from core.tooling.handler import ToolHandler
    from core.tooling.policy import action_gate

    memory = MagicMock()
    memory.search_memory_text.return_value = []
    with patch("core.config.models.load_config", side_effect=ConfigError("skip subordinate cache")):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=["gmail"])
    h._dispatch = {"call_human": lambda _args: "called human"}
    monkeypatch.setattr(
        action_gate,
        "_search_action_rules",
        lambda *args, **kwargs: [FakeRule("rule-call", "## [ACTION-RULE] confirm\n---\nConfirm first.", 0.96)],
    )

    result = h.handle("call_human", {"message": "ping"})

    assert result.startswith("called human")
    assert '<action-rule path="rule-call"' in result


def test_cli_argv_mapping() -> None:
    from core.tooling.policy.action_gate import action_tool_name_from_cli_argv

    assert action_tool_name_from_cli_argv(["gmail", "draft", "--to", "a@example.com"]) == "gmail_draft"
    assert action_tool_name_from_cli_argv(["gmail", "draft-update", "draft-id"]) == "gmail_draft_update"
    assert action_tool_name_from_cli_argv(["gmail", "send", "--to", "a@example.com"]) == "gmail_send"
    assert action_tool_name_from_cli_argv(["chatwork", "send", "room", "body"]) == "chatwork_send"
    assert action_tool_name_from_cli_argv(["board", "post", "general", "body"]) == "post_channel"
    assert action_tool_name_from_cli_argv(["board", "post", "general", "-", "--as", "operator"]) == "post_channel"
    assert action_tool_name_from_cli_argv(["slack", "send", "#ops", "body"]) == "slack_send"
    assert action_tool_name_from_cli_argv(["discord", "send", "general", "body"]) == "discord_send"
    assert action_tool_name_from_cli_argv(["call_human", "subject", "body"]) == "call_human"
    assert action_tool_name_from_cli_argv(["gmail", "unread"]) is None
    assert action_tool_name_from_cli_argv(["submit", "gmail", "send"]) is None


def test_handler_action_tool_names_are_current() -> None:
    from core.tooling.policy.action_gate import ACTION_TOOL_NAMES, action_tool_name_for_handler

    assert {
        "call_human",
        "send_message",
        "post_channel",
        "write_memory_file",
        "create_skill",
        "gmail_draft",
        "gmail_draft_update",
        "gmail_send",
        "chatwork_send",
        "slack_send",
        "discord_send",
    } == ACTION_TOOL_NAMES
    assert action_tool_name_for_handler("slack_post") is None


def test_sdk_tool_name_normalization() -> None:
    from core.tooling.policy.action_gate import action_tool_name_for_sdk

    assert action_tool_name_for_sdk("mcp__aw__send_message") == "send_message"
    assert action_tool_name_for_sdk("mcp__aw__write_memory_file") == "write_memory_file"
    assert action_tool_name_for_sdk("mcp__aw__gmail_draft") == "gmail_draft"
    assert action_tool_name_for_sdk("slack_post") is None
