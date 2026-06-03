from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.locomo.llm_config import default_answer_model, default_baseline_path, llm_routing_configured

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "benchmarks/locomo/data/locomo10.json"


def _baseline_path() -> Path:
    return default_baseline_path(os.environ.get("LOCOMO_ANSWER_MODEL") or default_answer_model())


def _llm_configured() -> bool:
    return llm_routing_configured()


def test_phase13_regression_guardrails_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Representative Run 3 failures are blocked before live smoke."""
    from benchmarks.locomo.adapter import (
        locomo_entity_aware_graph_enabled,
        locomo_entity_boost_enabled,
        locomo_fact_index_enabled,
        locomo_temporal_boost_enabled,
    )
    from benchmarks.locomo.answer_prompt import (
        LOCOMO_ANSWER_MAX_TOKENS,
        build_answer_user_content,
        normalize_locomo_answer,
    )

    monkeypatch.delenv("LOCOMO_TEMPORAL_BOOST", raising=False)
    monkeypatch.delenv("LOCOMO_ENTITY_BOOST", raising=False)
    monkeypatch.delenv("LOCOMO_ENTITY_AWARE_GRAPH", raising=False)
    monkeypatch.delenv("LOCOMO_FACT_INDEX", raising=False)
    prompt = build_answer_user_content("What book did Caroline recommend?", "ctx", category=4)
    assert "never abstain when context exists" not in prompt
    assert LOCOMO_ANSWER_MAX_TOKENS == 512
    assert locomo_temporal_boost_enabled() is False
    assert locomo_entity_boost_enabled() is False
    assert locomo_entity_aware_graph_enabled() is False
    assert locomo_fact_index_enabled() is False

    open_domain_raw = (
        'We are asked: "What book did Caroline recommend to Melanie?" '
        "The conversation happened in 2023. The answer is Becoming Nicole."
    )
    assert normalize_locomo_answer(open_domain_raw, category=4) == "Becoming Nicole"
    assert normalize_locomo_answer("not mentioned in the memories", category=5) == (
        "No information available."
    )


@pytest.mark.locomo
def test_legacy_baseline_file_shape() -> None:
    """Baseline JSON exists and contains required metrics."""
    baseline_path = _baseline_path()
    assert baseline_path.is_file(), f"missing baseline: {baseline_path}"
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data["overall_f1"] > 0.0
    assert data["by_category"]["open_domain"]["f1"] > 0.0
    assert "thresholds" in data
    model = (data.get("answer_model") or "").lower()
    if "deepseek" in model:
        assert data["overall_f1"] == pytest.approx(0.447, rel=0.02)
        assert data["by_category"]["open_domain"]["f1"] == pytest.approx(0.403, rel=0.02)
    elif "qwen" in model:
        assert data["overall_f1"] == pytest.approx(0.637, rel=0.01)
        assert data["by_category"]["open_domain"]["f1"] == pytest.approx(0.708, rel=0.01)


@pytest.mark.locomo
def test_legacy_scope_all_within_baseline() -> None:
    """Run 1-conversation Legacy scope_all and compare to fixed baseline."""
    if not _llm_configured():
        pytest.skip("LLM routing not configured — skipping live LoCoMo smoke")
    if not DATA_PATH.is_file():
        pytest.fail(
            f"dataset missing: {DATA_PATH}\n"
            "wget -O benchmarks/locomo/data/locomo10.json "
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        )

    from argparse import Namespace

    from benchmarks.locomo.runner import run_benchmark

    args = Namespace(
        data=DATA_PATH,
        output=ROOT / "benchmarks/locomo/results/smoke_pytest",
        mode="scope_all",
        conversations=1,
        top_k=10,
        judge=False,
        judge_model="gpt-4o",
        answer_model=os.environ.get("LOCOMO_ANSWER_MODEL", default_answer_model()),
        exclude_cat5=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    all_results, errors = run_benchmark(args)
    assert errors == 0, f"benchmark had {errors} errors"
    summary = all_results["scope_all"]["summary"]
    baseline_path = _baseline_path()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    overall = float(summary["overall_f1"])
    open_dom = float(summary["by_category"]["open_domain"]["f1"])
    b_overall = float(baseline["overall_f1"])
    b_open = float(baseline["by_category"]["open_domain"]["f1"])
    d_overall = float(baseline["thresholds"]["overall_f1_min_delta"])
    d_open = float(baseline["thresholds"]["open_domain_f1_min_delta"])

    assert overall >= b_overall - d_overall, f"overall_f1 regression: {overall:.4f} < {b_overall - d_overall:.4f}"
    assert open_dom >= b_open - d_open, f"open_domain regression: {open_dom:.4f} < {b_open - d_open:.4f}"
