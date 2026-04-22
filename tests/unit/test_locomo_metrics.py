from __future__ import annotations

import pytest

from benchmarks.locomo.metrics import (
    CATEGORY_NAMES,
    _multi_hop_f1,
    _normalize_answer,
    _stemmed_tokens,
    compute_summary,
    eval_by_category,
    f1_score,
)

# ── Normalize ──────────


class TestNormalizeAnswer:
    def test_lowercase_and_strip(self):
        assert _normalize_answer("  Hello World  ") == "hello world"

    def test_remove_articles(self):
        assert _normalize_answer("The cat and a dog") == "cat dog"

    def test_remove_punctuation(self):
        assert _normalize_answer("It's here!") == "its here"

    def test_remove_commas(self):
        assert _normalize_answer("red, green, blue") == "red green blue"

    def test_empty(self):
        assert _normalize_answer("") == ""


# ── Stemmed tokens ──────────


class TestStemmedTokens:
    def test_basic(self):
        tokens = _stemmed_tokens("running quickly")
        assert len(tokens) == 2
        assert all(isinstance(t, str) for t in tokens)

    def test_empty(self):
        assert _stemmed_tokens("") == []


# ── F1 ──────────


class TestF1Score:
    def test_exact_match(self):
        assert f1_score("hello world", "hello world") == pytest.approx(1.0)

    def test_partial(self):
        score = f1_score("hello world foo", "hello world")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert f1_score("alpha", "beta") == 0.0

    def test_empty_prediction(self):
        assert f1_score("", "hello") == 0.0

    def test_empty_reference(self):
        assert f1_score("hello", "") == 0.0

    def test_case_insensitive(self):
        assert f1_score("Hello World", "hello world") == pytest.approx(1.0)


# ── Multi-hop F1 ──────────


class TestMultiHopF1:
    def test_single_part(self):
        score = _multi_hop_f1("paris", "paris")
        assert score == pytest.approx(1.0)

    def test_multi_part_match(self):
        score = _multi_hop_f1("paris, london", "paris, london")
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        score = _multi_hop_f1("paris", "paris, london")
        assert 0.0 < score <= 1.0

    def test_empty(self):
        assert _multi_hop_f1("", "paris") == 0.0


# ── Category eval ──────────


class TestEvalByCategory:
    def test_cat1_multi_hop(self):
        score = eval_by_category("paris, london", "paris, london", 1)
        assert score == pytest.approx(1.0)

    def test_cat2_temporal(self):
        score = eval_by_category("7 May 2023", "7 May 2023", 2)
        assert score == pytest.approx(1.0)

    def test_cat3_semicolon_trim(self):
        score = eval_by_category("paris", "paris; france", 3)
        assert score == pytest.approx(1.0)

    def test_cat4_open_domain(self):
        score = eval_by_category("running", "running", 4)
        assert score == pytest.approx(1.0)

    def test_cat5_adversarial_abstain(self):
        assert eval_by_category("No information available.", "anything", 5) == 1.0

    def test_cat5_adversarial_not_mentioned(self):
        assert eval_by_category("That was not mentioned in the conversation.", "x", 5) == 1.0

    def test_cat5_adversarial_wrong(self):
        assert eval_by_category("She went to Paris.", "anything", 5) == 0.0

    def test_unknown_category_fallback(self):
        score = eval_by_category("hello", "hello", 99)
        assert score == pytest.approx(1.0)


# ── Summary ──────────


class TestComputeSummary:
    def test_empty(self):
        s = compute_summary([])
        assert s["overall_f1"] == 0.0
        assert s["overall_judge"] is None
        assert s["by_category"] == {}

    def test_basic(self):
        results = [
            {"category": 1, "f1": 0.8, "judge_score": None},
            {"category": 1, "f1": 0.6, "judge_score": None},
            {"category": 2, "f1": 0.5, "judge_score": 1.0},
        ]
        s = compute_summary(results)
        assert s["overall_f1"] == pytest.approx((0.8 + 0.6 + 0.5) / 3)
        assert s["overall_judge"] == pytest.approx(1.0)
        cat = s["by_category"]
        assert cat["multi_hop"]["f1"] == pytest.approx(0.7)
        assert cat["multi_hop"]["count"] == 2
        assert cat["temporal"]["f1"] == pytest.approx(0.5)
        assert cat["temporal"]["judge"] == pytest.approx(1.0)

    def test_all_categories_present(self):
        for cat_id, name in CATEGORY_NAMES.items():
            results = [{"category": cat_id, "f1": 0.5, "judge_score": None}]
            s = compute_summary(results)
            assert name in s["by_category"]
