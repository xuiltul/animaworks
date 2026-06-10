from __future__ import annotations

from benchmarks.locomo.answer_prompt import (
    LOCOMO_CONFIDENCE_ADVERSARIAL,
    LOCOMO_CONFIDENCE_DEFAULT,
    LOCOMO_CONFIDENCE_MULTI_HOP_FACT,
    LOCOMO_RRF_CONFIDENCE_MULTI_HOP_FACT,
    build_answer_user_content,
    confidence_gate_for_category,
    merge_pipeline_gate_settings,
    normalize_locomo_answer,
)


class TestNormalizeLoCoMoAnswer:
    def test_strips_chain_of_thought(self) -> None:
        raw = "We need to find when Melanie painted a sunrise. From context she painted last year in 2022."
        assert normalize_locomo_answer(raw, category=2, enable_category_normalization=True) == "2022"

    def test_default_does_not_extract_temporal_year_from_category(self) -> None:
        raw = "Melanie painted a lake sunrise last year, which happened during 2022."

        assert normalize_locomo_answer(raw, category=2) != "2022"

    def test_extracts_which_is_clause(self) -> None:
        raw = 'Based on the conversation, Melanie painted "last year," which is 2022.'
        assert normalize_locomo_answer(raw, category=2) == "2022"

    def test_open_domain_does_not_collapse_to_year(self) -> None:
        raw = (
            'We are asked: "What book did Caroline recommend to Melanie?" '
            "The conversation happened in 2023. The answer is Becoming Nicole."
        )
        assert normalize_locomo_answer(raw, category=4) != "2023"

    def test_temporal_verbose_answer_can_extract_year(self) -> None:
        raw = "Melanie painted a lake sunrise last year, which is 2022."
        assert normalize_locomo_answer(raw, category=2, enable_category_normalization=True) == "2022"

    def test_single_line_reasoning_answer_clause(self) -> None:
        raw = (
            'We are asked: "What was grandma\'s gift to Caroline?" '
            "The context mentions her grandma's gift. The answer is necklace."
        )
        assert normalize_locomo_answer(raw, category=4) == "necklace"

    def test_adversarial_abstain_canonical(self) -> None:
        assert normalize_locomo_answer("not mentioned in the logs", category=5) == ("No information available.")

    def test_empty_returns_abstain(self) -> None:
        assert normalize_locomo_answer("", category=1) == "No information available."


class TestConfidenceGateForCategory:
    def test_default_matches_config_threshold(self) -> None:
        gate = confidence_gate_for_category(4)
        assert gate["confidence_threshold"] == LOCOMO_CONFIDENCE_DEFAULT
        assert gate["confidence_threshold"] == LOCOMO_CONFIDENCE_ADVERSARIAL

    def test_adversarial_uses_same_default_threshold(self) -> None:
        gate = confidence_gate_for_category(5)
        assert gate["confidence_threshold"] == LOCOMO_CONFIDENCE_ADVERSARIAL


class TestMergePipelineGateSettings:
    def test_category_1_fact_evidence_relaxes_multihop_gate(self) -> None:
        gate = merge_pipeline_gate_settings(
            {"confidence_threshold": 0.35, "rrf_confidence_threshold": 0.02},
            category=1,
            fact_evidence=True,
        )

        assert gate["confidence_threshold"] == LOCOMO_CONFIDENCE_MULTI_HOP_FACT
        assert gate["rrf_confidence_threshold"] == LOCOMO_RRF_CONFIDENCE_MULTI_HOP_FACT

    def test_other_categories_keep_configured_gate_even_with_fact_evidence(self) -> None:
        gate = merge_pipeline_gate_settings(
            {"confidence_threshold": 0.35, "rrf_confidence_threshold": 0.02},
            category=5,
            fact_evidence=True,
        )

        assert gate["confidence_threshold"] == 0.35
        assert gate["rrf_confidence_threshold"] == 0.02


class TestBuildAnswerUserContent:
    def test_no_category_specific_hint(self) -> None:
        text = build_answer_user_content("Q?", "ctx", category=4)
        assert "Open-domain" not in text
        assert "never abstain when context exists" not in text
        assert "Q?" in text
