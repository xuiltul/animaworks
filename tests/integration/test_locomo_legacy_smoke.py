from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "benchmarks/results/baselines/legacy_scope_all_20260522.json"
DATA_PATH = ROOT / "benchmarks/locomo/data/locomo10.json"


def _llm_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"))


@pytest.mark.locomo
def test_legacy_baseline_file_shape() -> None:
    """Baseline JSON exists and contains required metrics."""
    assert BASELINE_PATH.is_file(), f"missing baseline: {BASELINE_PATH}"
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert data["overall_f1"] == pytest.approx(0.637, rel=0.01)
    assert data["by_category"]["open_domain"]["f1"] == pytest.approx(0.708, rel=0.01)
    assert "thresholds" in data


@pytest.mark.locomo
def test_legacy_scope_all_within_baseline() -> None:
    """Run 1-conversation Legacy scope_all and compare to fixed baseline."""
    if not _llm_configured():
        pytest.skip("OPENAI_API_BASE not set — skipping live LoCoMo smoke")
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
        answer_model=os.environ.get("LOCOMO_ANSWER_MODEL", "gpt-4o-mini"),
        exclude_cat5=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    all_results, errors = run_benchmark(args)
    assert errors == 0, f"benchmark had {errors} errors"
    summary = all_results["scope_all"]["summary"]
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    overall = float(summary["overall_f1"])
    open_dom = float(summary["by_category"]["open_domain"]["f1"])
    b_overall = float(baseline["overall_f1"])
    b_open = float(baseline["by_category"]["open_domain"]["f1"])
    d_overall = float(baseline["thresholds"]["overall_f1_min_delta"])
    d_open = float(baseline["thresholds"]["open_domain_f1_min_delta"])

    assert overall >= b_overall - d_overall, (
        f"overall_f1 regression: {overall:.4f} < {b_overall - d_overall:.4f}"
    )
    assert open_dom >= b_open - d_open, (
        f"open_domain regression: {open_dom:.4f} < {b_open - d_open:.4f}"
    )
