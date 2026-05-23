from __future__ import annotations

import pytest

from core.memory.retrieval.confidence_gate import GateResult, apply_confidence_gate


class TestApplyConfidenceGate:
    def test_empty_candidates_abstains(self) -> None:
        result = apply_confidence_gate([], threshold=0.35)
        assert result == GateResult(candidates=[], abstain=True, reason="low_confidence")

    def test_single_below_threshold_abstains(self) -> None:
        result = apply_confidence_gate([{"score": 0.1, "content": "x"}], threshold=0.35)
        assert result.abstain is True
        assert result.candidates == []

    def test_single_at_threshold_passes(self) -> None:
        result = apply_confidence_gate([{"score": 0.35, "content": "x"}], threshold=0.35)
        assert result.abstain is False
        assert len(result.candidates) == 1

    def test_single_above_threshold_passes(self) -> None:
        result = apply_confidence_gate([{"score": 0.9, "content": "x"}], threshold=0.35)
        assert result.abstain is False

    def test_max_score_wins_among_many(self) -> None:
        items = [
            {"score": 0.1, "content": "low"},
            {"score": 0.8, "content": "high"},
            {"score": 0.2, "content": "mid"},
        ]
        result = apply_confidence_gate(items, threshold=0.35)
        assert result.abstain is False
        assert result.candidates == items

    def test_one_high_one_low_still_passes(self) -> None:
        items = [{"score": 0.5, "content": "a"}, {"score": 0.01, "content": "b"}]
        result = apply_confidence_gate(items, threshold=0.35)
        assert result.abstain is False

    def test_all_below_threshold_abstains(self) -> None:
        items = [{"score": 0.01, "content": "a"}, {"score": 0.02, "content": "b"}]
        result = apply_confidence_gate(items, threshold=0.35)
        assert result.abstain is True
        assert result.candidates == []

    def test_rrf_low_threshold_boundary(self) -> None:
        result = apply_confidence_gate([{"score": 0.019, "content": "x"}], threshold=0.02)
        assert result.abstain is True

    def test_rrf_threshold_pass(self) -> None:
        result = apply_confidence_gate([{"score": 0.02, "content": "x"}], threshold=0.02)
        assert result.abstain is False

    def test_missing_score_treated_as_zero(self) -> None:
        result = apply_confidence_gate([{"content": "no score"}], threshold=0.01)
        assert result.abstain is True

    def test_none_score_treated_as_zero(self) -> None:
        result = apply_confidence_gate([{"score": None, "content": "x"}], threshold=0.01)
        assert result.abstain is True

    def test_custom_score_field(self) -> None:
        result = apply_confidence_gate(
            [{"ce_score": 0.9, "content": "x"}],
            threshold=0.35,
            score_field="ce_score",
        )
        assert result.abstain is False

    @pytest.mark.parametrize(
        ("score", "threshold", "expected_abstain"),
        [
            (0.34, 0.35, True),
            (0.35, 0.35, False),
            (0.0, 0.35, True),
        ],
    )
    def test_boundary_values(
        self,
        score: float,
        threshold: float,
        expected_abstain: bool,
    ) -> None:
        result = apply_confidence_gate([{"score": score}], threshold=threshold)
        assert result.abstain is expected_abstain
