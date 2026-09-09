"""Unit tests for Action Memory Gate fail_mode (pi-fix3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


def _patch_fail_mode(monkeypatch, mode: str, cooldown: int = 21600) -> None:
    from core.memory import action_gate

    monkeypatch.setattr(
        action_gate,
        "_resolve_fail_mode",
        lambda: (mode, cooldown),
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "mei"
    (d / "knowledge").mkdir(parents=True)
    return d


# ── fail_mode=close ──────────────────────────────────────────


def test_close_search_failed_blocks(anima_dir: Path, monkeypatch) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "close")

    def raise_search(*args, **kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    decision = action_gate.check_action(anima_dir, "chatwork_send", {"message": "hello"}, session_key="s-close-sf")

    assert decision.allowed is False
    assert decision.reason == "search_failed"
    assert decision.fail_mode == "close"
    assert decision.would_block is True


def test_close_no_matching_rule_holds_and_notifies(
    anima_dir: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "close", cooldown=0)
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [])

    notify_calls: list[tuple] = []

    def fake_notify(ad, tool, *, cooldown_seconds):
        notify_calls.append((ad, tool, cooldown_seconds))
        return True

    monkeypatch.setattr(action_gate, "_maybe_notify_no_matching_rule", fake_notify)

    with caplog.at_level(logging.WARNING, logger="animaworks.action_memory_gate"):
        decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-close-nmr")

    assert decision.allowed is False
    assert decision.reason == "no_matching_rule"
    assert decision.fail_mode == "close"
    assert decision.would_block is True
    assert len(notify_calls) == 1
    assert notify_calls[0][1] == "gmail_send"
    assert any("action_gate_soft_fail" in r.message for r in caplog.records)
    assert any("no_matching_rule" in r.message for r in caplog.records)


def test_close_no_matching_rule_explicit_allow_releases(
    anima_dir: Path,
    monkeypatch,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "close")
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [])

    blocked = action_gate.check_action(anima_dir, "slack_send", {"text": "hi"}, session_key="s-allow")
    assert blocked.allowed is False
    assert blocked.reason == "no_matching_rule"

    action_gate.grant_no_rule_allow(anima_dir, "slack_send", session_key="s-allow")

    allowed = action_gate.check_action(anima_dir, "slack_send", {"text": "hi"}, session_key="s-allow")
    assert allowed.allowed is True
    assert allowed.reason == "no_matching_rule_allowed"


def test_close_rule_read_still_allows(anima_dir: Path, monkeypatch) -> None:
    """Existing happy path: correct rule read still passes under close."""
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "close")
    rule = FakeRule(
        "rule-1",
        '## [ACTION-RULE]\ntrigger_tools: call_human\n---\nread_memory_file(path="procedures/check.md")',
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [rule])

    blocked = action_gate.check_action(anima_dir, "call_human", {"body": "x"}, session_key="s-ok")
    assert blocked.allowed is False
    assert blocked.reason == "missing_required_memory"

    action_gate.record_memory_read(anima_dir, "procedures/check.md", session_key="s-ok")
    allowed = action_gate.check_action(anima_dir, "call_human", {"body": "x"}, session_key="s-ok")
    assert allowed.allowed is True
    assert allowed.reason == "required_memory_satisfied"


def test_close_below_threshold_does_not_pass_through(
    anima_dir: Path,
    monkeypatch,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "close")
    rule = FakeRule(
        "rule-low",
        '## [ACTION-RULE]\ntrigger_tools: gmail_send\n---\nread_memory_file(path="procedures/check.md")',
        0.50,
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [rule])

    decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-bt")

    assert decision.allowed is False
    assert decision.reason == "missing_required_memory"
    assert decision.missing_paths == ["procedures/check.md"]


# ── fail_mode=middle ─────────────────────────────────────────


def test_middle_search_failed_blocks(anima_dir: Path, monkeypatch) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "middle")

    def raise_search(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    decision = action_gate.check_action(anima_dir, "discord_send", {"message": "x"}, session_key="s-mid-sf")
    assert decision.allowed is False
    assert decision.reason == "search_failed"
    assert decision.fail_mode == "middle"


def test_middle_no_matching_rule_allows_with_log(
    anima_dir: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "middle")
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [])

    with caplog.at_level(logging.WARNING, logger="animaworks.action_memory_gate"):
        decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "x"}, session_key="s-mid-nmr")

    assert decision.allowed is True
    assert decision.reason == "no_matching_rule"
    assert decision.would_block is True  # close would have blocked
    assert any("action_gate_soft_fail" in r.message for r in caplog.records)


def test_middle_below_threshold_enforces_read_review(
    anima_dir: Path,
    monkeypatch,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "middle")
    rule = FakeRule(
        "rule-low",
        "## [ACTION-RULE]\ntrigger_tools: post_channel\n---\nConfirm context.",
        0.40,
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [rule])

    first = action_gate.check_action(anima_dir, "post_channel", {"text": "FYI"}, session_key="s-mid-bt")
    second = action_gate.check_action(anima_dir, "post_channel", {"text": "FYI"}, session_key="s-mid-bt")

    assert first.allowed is False
    assert first.reason == "review_rule_before_retry"
    assert second.allowed is True
    assert second.reason == "rule_already_shown"


# ── fail_mode=open (legacy + observability) ──────────────────


def test_open_search_failed_allows_with_structured_log(
    anima_dir: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "open")

    def raise_search(*args, **kwargs):
        raise RuntimeError("vector down")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    with caplog.at_level(logging.WARNING, logger="animaworks.action_memory_gate"):
        decision = action_gate.check_action(anima_dir, "chatwork_send", {"message": "hello"}, session_key="s-open-sf")

    assert decision.allowed is True
    assert decision.reason == "search_failed"
    assert decision.would_block is True
    assert decision.fail_mode == "open"
    assert any("action_gate_soft_fail" in r.message for r in caplog.records)
    assert any("search_failed" in r.message for r in caplog.records)
    assert any("would_block=True" in r.message for r in caplog.records)


def test_open_no_matching_rule_allows_with_log(
    anima_dir: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "open")
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *a, **k: [])

    with caplog.at_level(logging.WARNING, logger="animaworks.action_memory_gate"):
        decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-open-nmr")

    assert decision.allowed is True
    assert decision.reason == "no_matching_rule"
    assert decision.would_block is True
    assert any("action_gate_soft_fail" in r.message for r in caplog.records)
    assert any("no_matching_rule" in r.message for r in caplog.records)


def test_open_below_threshold_allows_with_log(
    anima_dir: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.memory import action_gate

    _patch_fail_mode(monkeypatch, "open")
    monkeypatch.setattr(
        action_gate,
        "_search_action_rules",
        lambda *a, **k: [FakeRule("rule-low", "## [ACTION-RULE]\n", 0.79)],
    )

    with caplog.at_level(logging.WARNING, logger="animaworks.action_memory_gate"):
        decision = action_gate.check_action(anima_dir, "gmail_send", {"body": "hello"}, session_key="s-open-bt")

    assert decision.allowed is True
    assert decision.reason == "below_threshold"
    assert decision.score == 0.79
    assert decision.would_block is True
    assert any("below_threshold" in r.message for r in caplog.records)


# ── notify dedup ─────────────────────────────────────────────


def test_no_rule_notify_dedup_per_action_anima(
    anima_dir: Path,
    monkeypatch,
) -> None:
    from core.memory import action_gate

    # Avoid HumanNotifier; exercise trail + cooldown only
    monkeypatch.setattr(
        action_gate,
        "_resolve_fail_mode",
        lambda: ("close", 3600),
    )
    # Force notify path without real config load for HumanNotifier
    monkeypatch.setattr(
        "core.config.load_config",
        MagicMock(side_effect=RuntimeError("no config")),
    )

    first = action_gate._maybe_notify_no_matching_rule(anima_dir, "gmail_send", cooldown_seconds=3600)
    second = action_gate._maybe_notify_no_matching_rule(anima_dir, "gmail_send", cooldown_seconds=3600)
    other_tool = action_gate._maybe_notify_no_matching_rule(anima_dir, "slack_send", cooldown_seconds=3600)

    assert first is True
    assert second is False  # suppressed by cooldown
    assert other_tool is True  # different tool not suppressed

    trail = anima_dir / "run" / "action_memory_gate" / "no_rule_holds.jsonl"
    assert trail.is_file()
    lines = trail.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # gmail first + slack (second gmail suppressed)


# ── config schema ────────────────────────────────────────────


def test_action_gate_config_default_is_open() -> None:
    from core.config.schemas import ActionGateConfig, AnimaWorksConfig

    assert ActionGateConfig().fail_mode == "open"
    assert AnimaWorksConfig().action_gate.fail_mode == "open"


def test_decision_payload_includes_fail_fields() -> None:
    from core.memory.action_gate import ActionMemoryGateDecision

    blocked = ActionMemoryGateDecision(
        allowed=False,
        tool="gmail_send",
        reason="search_failed",
        fail_mode="close",
        would_block=True,
    )
    payload = blocked.to_payload()
    assert payload["error_type"] == "ActionMemoryGate"
    assert payload["fail_mode"] == "close"
    assert payload["would_block"] is True
    assert "infrastructure" in payload["message"].lower() or "search" in payload["message"].lower()


def test_common_memory_write_rule_is_returned_before_write(
    anima_dir: Path,
    monkeypatch,
) -> None:
    """The packaged common rule participates in the write-memory gate."""
    from core.memory import action_gate
    from core.memory.rag.retriever import MemoryRetriever

    rule_path = (
        Path(__file__).resolve().parents[4]
        / "templates"
        / "ja"
        / "common_knowledge"
        / "operations"
        / "action-rule-memory-write-destination.md"
    )
    rule_content = rule_path.read_text(encoding="utf-8")
    retriever = MemoryRetriever.__new__(MemoryRetriever)

    def fake_vector_search(query, collection_name, top_k, filter_metadata):
        del query, top_k
        assert filter_metadata == {"type": "action_rule"}
        if collection_name == "shared_common_knowledge":
            return [
                (
                    "shared/action-rule-memory-write-destination.md#0",
                    rule_content,
                    0.99,
                    {"type": "action_rule", "trigger_tools": "write_memory_file,create_skill"},
                )
            ]
        return []

    retriever._vector_search_collection = fake_vector_search
    monkeypatch.setattr(action_gate, "_get_retriever", lambda _path: retriever)
    _patch_fail_mode(monkeypatch, "close")

    decision = action_gate.check_action(
        anima_dir,
        "write_memory_file",
        {"path": "knowledge/example.md"},
        session_key="common-memory-rule",
    )

    assert decision.allowed is False
    assert decision.rule_id == "shared/action-rule-memory-write-destination.md#0"
    assert decision.missing_paths == ["reference/operations/memory-writing-guide.md"]
