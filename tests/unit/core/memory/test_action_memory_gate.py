from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


def test_extract_required_memory_paths_normalizes_and_deduplicates(tmp_path: Path) -> None:
    from core.memory.action_gate import extract_required_memory_paths

    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    content = (
        "## [ACTION-RULE] test\n"
        "trigger_tools: call_human\n"
        "---\n"
        'read_memory_file(path="./procedures/check.md")\n'
        "read_memory_file(path='procedures/check.md')\n"
    )

    assert extract_required_memory_paths(content, anima_dir) == ["procedures/check.md"]


def test_extract_required_memory_paths_normalizes_absolute_and_shared_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from core.memory.action_gate import extract_required_memory_paths

    anima_dir = tmp_path / "animas" / "mei"
    own_file = anima_dir / "procedures" / "check.md"
    common_dir = tmp_path / "shared" / "common_knowledge"
    common_file = common_dir / "ops" / "rules.md"
    own_file.parent.mkdir(parents=True)
    common_file.parent.mkdir(parents=True)
    own_file.write_text("# Check\n", encoding="utf-8")
    common_file.write_text("# Rules\n", encoding="utf-8")
    monkeypatch.setattr("core.paths.get_common_knowledge_dir", lambda: common_dir)
    monkeypatch.setattr("core.paths.get_reference_dir", lambda: tmp_path / "shared" / "reference")
    monkeypatch.setattr("core.paths.get_common_skills_dir", lambda: tmp_path / "shared" / "common_skills")

    content = (
        f'read_memory_file(path="{own_file}")\n'
        f'read_memory_file(path="{common_file}")\n'
    )

    assert extract_required_memory_paths(content, anima_dir) == [
        "procedures/check.md",
        "common_knowledge/ops/rules.md",
    ]


def test_required_read_blocks_until_memory_read(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    rule = FakeRule(
        "mei/knowledge/rule.md#0",
        '## [ACTION-RULE] before notify\ntrigger_tools: call_human\n---\nread_memory_file(path="procedures/check.md")',
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    blocked = action_gate.check_action(anima_dir, "call_human", {"body": "notify"}, session_key="s1")

    assert blocked.allowed is False
    assert blocked.reason == "missing_required_memory"
    assert blocked.missing_paths == ["procedures/check.md"]

    action_gate.record_memory_read(anima_dir, "procedures/check.md", session_key="s1")
    allowed = action_gate.check_action(anima_dir, "call_human", {"body": "notify"}, session_key="s1")

    assert allowed.allowed is True
    assert allowed.reason == "required_memory_satisfied"


def test_rule_without_required_read_blocks_once_then_allows(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    rule = FakeRule("rule-1", "## [ACTION-RULE] check context\ntrigger_tools: post_channel\n---\nConfirm context.")
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    first = action_gate.check_action(anima_dir, "post_channel", {"text": "FYI"}, session_key="s2")
    second = action_gate.check_action(anima_dir, "post_channel", {"text": "FYI"}, session_key="s2")

    assert first.allowed is False
    assert first.reason == "review_rule_before_retry"
    assert second.allowed is True
    assert second.reason == "rule_already_shown"


def test_lower_ranked_required_rule_blocks_before_review_only_rule(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    rules = [
        FakeRule("rule-review", "## [ACTION-RULE] review\ntrigger_tools: gmail_send\n---\nReview context.", 0.97),
        FakeRule(
            "rule-required",
            '## [ACTION-RULE] duplicate check\ntrigger_tools: gmail_send\n---\nread_memory_file(path="procedures/check.md")',
            0.96,
        ),
    ]
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: rules)

    decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s4")

    assert decision.allowed is False
    assert decision.reason == "missing_required_memory"
    assert decision.rule_id == "rule-required"
    assert decision.missing_paths == ["procedures/check.md"]


def test_empty_tool_and_no_matching_rule_are_allowed(tmp_path: Path) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"

    empty_tool = action_gate.check_action(anima_dir, "", {}, session_key="s-empty")
    no_match = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-no-match")

    assert empty_tool.allowed is True
    assert no_match.allowed is True
    assert no_match.reason == "no_matching_rule"


def test_below_threshold_rule_is_allowed(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    monkeypatch.setattr(
        action_gate,
        "_search_action_rules",
        lambda *args, **kwargs: [FakeRule("rule-low", "## [ACTION-RULE]\ntrigger_tools: gmail_send", 0.79)],
    )

    decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-low")

    assert decision.allowed is True
    assert decision.reason == "below_threshold"
    assert decision.score == 0.79


def test_search_failure_fails_open(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)

    def raise_search(*args, **kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    decision = action_gate.check_action(anima_dir, "chatwork_send", {"message": "hello"}, session_key="s3")

    assert decision.allowed is True
    assert decision.reason == "search_failed"


def test_retriever_initialization_failure_fails_open(tmp_path: Path, monkeypatch) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)

    def raise_vector_store(*args, **kwargs):
        raise RuntimeError("vector init failed")

    monkeypatch.setattr("core.memory.rag.singleton.get_vector_store", raise_vector_store)

    decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-retriever")

    assert decision.allowed is True
    assert decision.reason == "no_matching_rule"


def test_corrupt_state_is_treated_as_empty(tmp_path: Path) -> None:
    from core.memory import action_gate

    anima_dir = tmp_path / "animas" / "mei"
    state_file = anima_dir / "run" / "action_memory_gate" / "s-corrupt.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("[]", encoding="utf-8")

    state = action_gate._load_state(anima_dir, "s-corrupt")

    assert state == {"read_paths": [], "shown_rules": []}


def test_session_key_uses_env_when_runtime_context_missing(monkeypatch) -> None:
    from core.memory import action_gate

    monkeypatch.setenv("ANIMAWORKS_TOOL_SESSION_ID", "tool session/1")

    assert action_gate._session_key() == "tool_session_1"


def test_cli_argv_mapping() -> None:
    from core.memory.action_gate import action_tool_name_from_cli_argv

    assert action_tool_name_from_cli_argv(["gmail", "draft", "--to", "a@example.com"]) == "gmail_draft"
    assert action_tool_name_from_cli_argv(["gmail", "send", "--to", "a@example.com"]) == "gmail_send"
    assert action_tool_name_from_cli_argv(["chatwork", "send", "room", "body"]) == "chatwork_send"
    assert action_tool_name_from_cli_argv(["call_human", "subject", "body"]) == "call_human"
    assert action_tool_name_from_cli_argv(["gmail", "unread"]) is None
    assert action_tool_name_from_cli_argv(["submit", "gmail", "send"]) is None
