from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, format_priming_section
from core.memory.priming.engine import PrimingResult
from core.memory.priming.gate import (
    MemoryCandidate,
    PrimingRenderMode,
    apply_priming_plan,
    build_candidates_from_result,
    build_priming_plan,
    classify_risk_tags,
    evidence_needed,
)


def test_japanese_explicit_recall_enables_evidence_mode() -> None:
    candidates = [MemoryCandidate(channel="episodes", content="raw episode body")]
    tags = classify_risk_tags("前に森村さんへ送信しかけた件を教えて", candidates)

    assert "explicit_recall" in tags
    assert evidence_needed("前に森村さんへ送信しかけた件を教えて", "chat", tags, candidates)


def test_english_external_action_requires_search_before_action() -> None:
    candidates = [MemoryCandidate(channel="related_knowledge", content="Gmail draft workflow")]
    plan = build_priming_plan("Please prepare a Gmail draft reply", "chat", "", candidates)

    assert plan.evidence_mode is True
    assert plan.require_search_before_action is True
    assert "external_action" in plan.risk_tags


def test_pointer_related_memory_remains_visible_without_evidence_mode() -> None:
    candidates = [
        MemoryCandidate(
            channel="related_knowledge",
            content='--- Result 1 [personal] ---\nCue\n  -> read_memory_file(path="knowledge/x.md")',
        )
    ]
    plan = build_priming_plan("こんにちは", "chat", "", candidates)
    decision = plan.channel_decisions["related_knowledge"]

    assert plan.evidence_mode is False
    assert decision.visible is True
    assert decision.render_mode == PrimingRenderMode.POINTER


def test_pointer_content_terms_do_not_enable_evidence_mode_by_themselves() -> None:
    candidates = [
        MemoryCandidate(
            channel="related_knowledge",
            content=(
                "--- Result 1 [personal] ---\n"
                "Gmail下書きは承認確認が必要\n"
                '  -> read_memory_file(path="knowledge/gmail-approval.md")'
            ),
        )
    ]
    plan = build_priming_plan("こんにちは", "chat", "", candidates)
    decision = plan.channel_decisions["related_knowledge"]

    assert plan.evidence_mode is False
    assert plan.require_search_before_action is False
    assert decision.visible is True
    assert decision.render_mode == PrimingRenderMode.POINTER


def test_non_pointer_related_memory_is_suppressed_without_evidence_mode() -> None:
    result = PrimingResult(related_knowledge="raw knowledge payload")
    plan = build_priming_plan("こんにちは", "chat", "", build_candidates_from_result(result))
    gated = apply_priming_plan(result, plan)

    assert plan.channel_decisions["related_knowledge"].visible is False
    assert gated.related_knowledge == ""


def test_candidate_content_terms_enable_guardrail_and_evidence_mode() -> None:
    result = PrimingResult(related_knowledge="Gmail下書きは送信前に承認を確認する")
    plan = build_priming_plan("こんにちは", "chat", "", build_candidates_from_result(result))
    gated = apply_priming_plan(result, plan)

    decision = plan.channel_decisions["related_knowledge"]
    assert plan.evidence_mode is True
    assert plan.require_search_before_action is True
    assert decision.visible is True
    assert decision.render_mode == PrimingRenderMode.GUARDRAIL
    assert gated.related_knowledge == "Gmail下書きは送信前に承認を確認する"


def test_non_pointer_related_memory_is_visible_in_evidence_mode() -> None:
    result = PrimingResult(related_knowledge="raw knowledge payload")
    plan = build_priming_plan("前に話した件の根拠を教えて", "chat", "", build_candidates_from_result(result))
    gated = apply_priming_plan(result, plan)

    decision = plan.channel_decisions["related_knowledge"]
    assert decision.visible is True
    assert decision.render_mode == PrimingRenderMode.EVIDENCE
    assert gated.related_knowledge == "raw knowledge payload"


def test_guardrail_external_send_cue_remains_visible() -> None:
    candidate = MemoryCandidate(
        channel="related_knowledge",
        content="Gmail下書きは送信前に承認を確認し、重複 draft を確認する",
    )
    plan = build_priming_plan("メール下書きを作って", "chat", "", [candidate])
    decision = plan.channel_decisions["related_knowledge"]

    assert decision.visible is True
    assert decision.render_mode == PrimingRenderMode.GUARDRAIL
    assert decision.require_search_before_action is True


def test_pending_task_external_action_requires_search_without_evidence_mode() -> None:
    result = PrimingResult(pending_tasks="Slack返信タスク: 送信前に相手と内容を確認する")
    plan = build_priming_plan("こんにちは", "chat", "", build_candidates_from_result(result))
    decision = plan.channel_decisions["pending_tasks"]

    assert plan.evidence_mode is False
    assert plan.require_search_before_action is True
    assert decision.visible is True
    assert decision.require_search_before_action is True


def test_apply_plan_preserves_untrusted_split() -> None:
    result = PrimingResult(
        related_knowledge='Cue\n  -> read_memory_file(path="knowledge/a.md")',
        related_knowledge_untrusted='External cue\n  -> read_memory_file(path="knowledge/b.md")',
    )
    plan = build_priming_plan("こんにちは", "chat", "", build_candidates_from_result(result))
    gated = apply_priming_plan(result, plan)

    assert gated.related_knowledge
    assert gated.related_knowledge_untrusted
    assert plan.channel_decisions["related_knowledge"].render_mode == PrimingRenderMode.POINTER
    assert plan.channel_decisions["related_knowledge_untrusted"].render_mode == PrimingRenderMode.POINTER


def test_format_priming_section_collapses_pointer_mode() -> None:
    body = (
        "--- Result 1 [personal] ---\n"
        "Gmail下書きは送信前に承認を確認する\n"
        '  -> read_memory_file(path="knowledge/gmail-approval.md")\n'
        "RAW_DETAIL_SHOULD_NOT_BE_IN_POINTER_MODE\n"
    )
    result = PrimingResult(related_knowledge=body)
    result.gate_plan = build_priming_plan("こんにちは", "chat", "", build_candidates_from_result(result))

    formatted = format_priming_section(result)

    assert 'render_mode="pointer"' in formatted
    assert '- Gmail下書きは送信前に承認を確認する -> read_memory_file(path="knowledge/gmail-approval.md")' in formatted
    assert "RAW_DETAIL_SHOULD_NOT_BE_IN_POINTER_MODE" not in formatted


def test_format_priming_section_marks_evidence_mode() -> None:
    result = PrimingResult(related_knowledge="RAW_EVIDENCE_PAYLOAD")
    result.gate_plan = build_priming_plan(
        "前に話した件の根拠を教えて", "chat", "", build_candidates_from_result(result)
    )

    formatted = format_priming_section(result)

    assert 'render_mode="evidence"' in formatted
    assert "RAW_EVIDENCE_PAYLOAD" in formatted


@pytest.mark.asyncio
async def test_engine_suppresses_raw_related_payload_before_truncation(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    engine = PrimingEngine(anima_dir)

    async def empty_str(*args, **kwargs):
        return ""

    async def raw_related(*args, **kwargs):
        return ("raw knowledge payload", "")

    async def raw_episode(*args, **kwargs):
        return "raw episode payload"

    monkeypatch.setattr(engine, "_channel_a_sender_profile", empty_str)
    monkeypatch.setattr(engine, "_channel_b_recent_activity", empty_str)
    monkeypatch.setattr(engine, "_channel_c0_important_knowledge", empty_str)
    monkeypatch.setattr(engine, "_channel_c_related_knowledge", raw_related)
    monkeypatch.setattr(engine, "_channel_e_pending_tasks", empty_str)
    monkeypatch.setattr(engine, "_collect_recent_outbound", empty_str)
    monkeypatch.setattr(engine, "_channel_f_episodes", raw_episode)
    monkeypatch.setattr(engine, "_collect_pending_human_notifications", empty_str)
    monkeypatch.setattr(engine, "_channel_g_graph_context", empty_str)

    result = await engine.prime_memories("こんにちは", channel="chat", enable_dynamic_budget=False)

    assert result.related_knowledge == ""
    assert result.episodes == ""
    assert result.gate_plan is not None
    assert result.gate_plan.channel_decisions["related_knowledge"].reason == (
        "non_pointer_related_memory_without_evidence_mode"
    )


@pytest.mark.asyncio
async def test_engine_keeps_raw_related_payload_in_evidence_mode(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    engine = PrimingEngine(anima_dir)

    async def empty_str(*args, **kwargs):
        return ""

    async def raw_related(*args, **kwargs):
        return ("raw knowledge payload", "")

    monkeypatch.setattr(engine, "_channel_a_sender_profile", empty_str)
    monkeypatch.setattr(engine, "_channel_b_recent_activity", empty_str)
    monkeypatch.setattr(engine, "_channel_c0_important_knowledge", empty_str)
    monkeypatch.setattr(engine, "_channel_c_related_knowledge", raw_related)
    monkeypatch.setattr(engine, "_channel_e_pending_tasks", empty_str)
    monkeypatch.setattr(engine, "_collect_recent_outbound", empty_str)
    monkeypatch.setattr(engine, "_channel_f_episodes", empty_str)
    monkeypatch.setattr(engine, "_collect_pending_human_notifications", empty_str)
    monkeypatch.setattr(engine, "_channel_g_graph_context", empty_str)

    result = await engine.prime_memories("前に話した件の根拠を教えて", channel="chat", enable_dynamic_budget=False)

    assert result.related_knowledge == "raw knowledge payload"
    assert result.gate_plan is not None
    assert result.gate_plan.evidence_mode is True
