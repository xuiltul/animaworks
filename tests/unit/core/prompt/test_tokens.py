from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.priming.constants import (
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_KNOWLEDGE,
)
from core.memory.priming.engine import PrimingEngine
from core.prompt.tokens import estimate_tokens, tokens_to_chars_hint, truncate_to_tokens


def test_estimate_ascii_calibration() -> None:
    assert estimate_tokens("a" * 1000) == pytest.approx(310, rel=0.05)


def test_estimate_japanese_calibration() -> None:
    assert estimate_tokens("あ漢" * 500) == pytest.approx(1140, rel=0.05)


def test_truncate_mixed_text_keeps_complete_head_lines() -> None:
    first = "日本語の一行目です。\n"
    second = "ASCII second line.\n"
    third = "混在 third line です。\n"
    fourth = "末尾の行"
    budget = estimate_tokens(first + second + "...")

    result = truncate_to_tokens(first + second + third + fourth, budget)

    assert result == first + second + "..."
    assert estimate_tokens(result) <= budget


def test_truncate_tail_preserves_last_lines() -> None:
    first = "先頭の行\n"
    second = "middle line\n"
    third = "末尾から二行目\n"
    fourth = "最後の行"
    budget = estimate_tokens("..." + third + fourth)

    result = truncate_to_tokens(first + second + third + fourth, budget, keep="tail")

    assert result == "..." + third + fourth
    assert result.endswith(fourth)
    assert estimate_tokens(result) <= budget


def test_single_line_is_cut_by_character() -> None:
    result = truncate_to_tokens("日本語" * 100, 20)

    assert result.endswith("...")
    assert estimate_tokens(result) <= 20


def test_empty_text_estimates_zero() -> None:
    assert estimate_tokens("") == 0
    assert truncate_to_tokens("", 100) == ""


def test_tokens_to_chars_hint_uses_sample_language_mix() -> None:
    assert tokens_to_chars_hint(114, "日本語") == 100
    assert tokens_to_chars_hint(31, "ascii") == 100
    assert tokens_to_chars_hint(100) > 0


@pytest.mark.asyncio
async def test_prime_memories_keeps_japanese_channels_within_token_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir()
    engine = PrimingEngine(anima_dir)

    async def empty(*args, **kwargs):
        return ""

    async def activity(*args, **kwargs):
        return "活動記録です。\n" * 250

    async def knowledge(*args, **kwargs):
        return ("根拠となる知識です。\n" * 200, "")

    monkeypatch.setattr(engine, "_channel_a_sender_profile", empty)
    monkeypatch.setattr(engine, "_channel_b_recent_activity", activity)
    monkeypatch.setattr(engine, "_channel_c0_important_knowledge", empty)
    monkeypatch.setattr(engine, "_channel_c_related_knowledge", knowledge)
    monkeypatch.setattr(engine, "_channel_e_pending_tasks", empty)
    monkeypatch.setattr(engine, "_collect_recent_outbound", empty)
    monkeypatch.setattr(engine, "_channel_f_episodes", empty)
    monkeypatch.setattr(engine, "_collect_pending_human_notifications", empty)
    monkeypatch.setattr(engine, "_channel_g_graph_context", empty)

    result = await engine.prime_memories("根拠を教えて", enable_dynamic_budget=False)

    assert result.recent_activity
    assert result.related_knowledge
    assert estimate_tokens(result.recent_activity) <= _BUDGET_RECENT_ACTIVITY
    assert estimate_tokens(result.related_knowledge) <= _BUDGET_RELATED_KNOWLEDGE
