from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark: compare Legacy vs Neo4j (5 runs)
"""Run 5 benchmark configurations and generate a comparison summary.

Runs:
  1. legacy/scope_all (best legacy mode)
  2. neo4j/full (all features)
  3. neo4j/no_reranker (ablation)
  4. neo4j/no_bfs (ablation)
  5. neo4j/no_invalidation (ablation)

Usage:
    python -m benchmarks.locomo.compare_modes [--conversations N] [--answer-model MODEL]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import SEARCH_MODES, AnimaWorksLoCoMoAdapter, load_dataset
from benchmarks.locomo.metrics import CATEGORY_NAMES, compute_summary
from benchmarks.locomo.neo4j_adapter import AblationFlags, Neo4jLoCoMoAdapter
from benchmarks.locomo.runner import EMBEDDING_MODEL_DEFAULT, _resolve_path, _run_qa_loop

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks" / "results"

# 5 run configurations
RUNS: list[dict[str, Any]] = [
    {"name": "legacy_scope_all", "backend": "legacy", "mode": "scope_all"},
    {"name": "neo4j_full", "backend": "neo4j", "ablation": AblationFlags()},
    {"name": "neo4j_no_reranker", "backend": "neo4j", "ablation": AblationFlags(reranker=False)},
    {"name": "neo4j_no_bfs", "backend": "neo4j", "ablation": AblationFlags(bfs=False)},
    {"name": "neo4j_no_invalidation", "backend": "neo4j", "ablation": AblationFlags(invalidation=False)},
]

AC_F1_THRESHOLD = 0.45


# ── Main ──────────


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Execute all 5 runs and collect results."""
    data_path = _resolve_path(Path(args.data))
    if not data_path.is_file():
        print(f"Error: dataset not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(data_path)
    n_conv = int(args.conversations)
    samples = dataset[:n_conv]
    if not samples:
        print("No samples to process.", file=sys.stderr)
        sys.exit(1)

    out_dir = _resolve_path(Path(args.output))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"locomo-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, Any] = {}
    total_errors = 0

    print(f"\n{'#' * 70}")
    print(f"  LoCoMo Comparison: {len(RUNS)} runs × {n_conv} conversations")
    print(f"  Answer model: {args.answer_model}")
    print(f"  Output: {run_dir}")
    print(f"{'#' * 70}\n")

    t0 = time.perf_counter()

    for run_idx, run_cfg in enumerate(RUNS, 1):
        run_name = run_cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"  Run {run_idx}/{len(RUNS)}: {run_name}")
        print(f"{'=' * 60}")

        config_block: dict[str, Any] = {
            "run_name": run_name,
            "backend": run_cfg["backend"],
            "top_k": int(args.top_k),
            "answer_model": str(args.answer_model),
            "judge_enabled": bool(args.judge),
            "conversations": n_conv,
        }

        mode_results: list[dict[str, Any]] = []
        errors = 0

        try:
            if run_cfg["backend"] == "legacy":
                mode = run_cfg.get("mode", "scope_all")
                config_block["search_mode"] = mode
                with AnimaWorksLoCoMoAdapter(search_mode=mode, top_k=int(args.top_k)) as adapter:
                    mode_results, errors = _run_qa_loop(
                        adapter=adapter,
                        samples=samples,
                        args=args,
                        mode_label=run_name,
                    )
            else:
                ablation: AblationFlags = run_cfg.get("ablation", AblationFlags())
                config_block["ablation"] = ablation.label()
                config_block["ablation_flags"] = {
                    "reranker": ablation.reranker,
                    "bfs": ablation.bfs,
                    "community": ablation.community,
                    "invalidation": ablation.invalidation,
                }
                with Neo4jLoCoMoAdapter(
                    top_k=int(args.top_k),
                    ablation=ablation,
                    answer_model=str(args.answer_model),
                ) as adapter:
                    mode_results, errors = _run_qa_loop(
                        adapter=adapter,
                        samples=samples,
                        args=args,
                        mode_label=run_name,
                    )
        except Exception as exc:
            errors += 1
            logger.exception("Run %s failed: %s", run_name, exc)
            print(f"  FATAL: {exc}", file=sys.stderr)

        total_errors += errors
        summary = compute_summary(mode_results)
        all_summaries[run_name] = {
            "summary": summary,
            "config": config_block,
            "question_count": len(mode_results),
            "errors": errors,
        }

        result_path = run_dir / f"{run_name}.json"
        result_path.write_text(
            json.dumps(
                {"config": config_block, "summary": summary, "results": mode_results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        of1 = summary.get("overall_f1", 0.0) * 100.0
        print(f"\n  → {run_name}: F1 = {of1:.1f}% ({len(mode_results)} questions, {errors} errors)")

    elapsed = time.perf_counter() - t0

    # Generate summary
    _write_summary(run_dir, all_summaries, elapsed, args)

    print(f"\n{'#' * 70}")
    print(f"  All runs complete in {elapsed:.0f}s. Results: {run_dir}")
    print(f"  Total errors: {total_errors}")
    print(f"{'#' * 70}")

    return all_summaries


def _write_summary(
    run_dir: Path,
    all_summaries: dict[str, Any],
    elapsed: float,
    args: argparse.Namespace,
) -> None:
    """Write summary.md with comparison table and AC judgment."""
    lines: list[str] = [
        "# LoCoMo Benchmark Comparison",
        "",
        f"**Date**: {datetime.now().isoformat(timespec='seconds')}",
        f"**Answer Model**: {args.answer_model}",
        f"**Conversations**: {args.conversations}",
        f"**Top-K**: {args.top_k}",
        f"**Total Time**: {elapsed:.0f}s",
        "",
        "## Results",
        "",
        "| Run | Overall F1 | multi_hop | temporal | open_domain | complex | adversarial | AC |",
        "|-----|-----------|-----------|----------|-------------|---------|-------------|-----|",
    ]

    for run_name, data in all_summaries.items():
        summ = data["summary"]
        of1 = summ.get("overall_f1", 0.0)
        by_cat = summ.get("by_category", {})

        cells: list[str] = [run_name, f"{of1 * 100:.1f}%"]
        for cat_name in ["multi_hop", "temporal", "open_domain", "complex", "adversarial"]:
            cat_data = by_cat.get(cat_name, {})
            cf1 = cat_data.get("f1", 0.0) if cat_data else 0.0
            cells.append(f"{cf1 * 100:.1f}%")

        ac = "PASS" if of1 >= AC_F1_THRESHOLD else "FAIL"
        cells.append(ac)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Acceptance Criteria",
        "",
        f"**Threshold**: F1 ≥ {AC_F1_THRESHOLD * 100:.0f}%",
        "",
    ])

    neo4j_full = all_summaries.get("neo4j_full", {})
    if neo4j_full:
        nf1 = neo4j_full.get("summary", {}).get("overall_f1", 0.0)
        status = "PASS ✅" if nf1 >= AC_F1_THRESHOLD else "FAIL ❌"
        lines.append(f"**Neo4j Full Result**: F1 = {nf1 * 100:.1f}% → **{status}**")

    legacy = all_summaries.get("legacy_scope_all", {})
    if legacy and neo4j_full:
        lf1 = legacy.get("summary", {}).get("overall_f1", 0.0)
        nf1 = neo4j_full.get("summary", {}).get("overall_f1", 0.0)
        diff = (nf1 - lf1) * 100.0
        sign = "+" if diff >= 0 else ""
        lines.append(f"**Neo4j vs Legacy**: {sign}{diff:.1f}pp")

    lines.extend(["", "## Ablation Analysis", ""])
    full_f1 = neo4j_full.get("summary", {}).get("overall_f1", 0.0) if neo4j_full else 0.0
    for run_name in ["neo4j_no_reranker", "neo4j_no_bfs", "neo4j_no_invalidation"]:
        data = all_summaries.get(run_name, {})
        if data:
            af1 = data.get("summary", {}).get("overall_f1", 0.0)
            drop = (full_f1 - af1) * 100.0
            lines.append(f"- **{run_name}**: F1 = {af1 * 100:.1f}% (Δ = -{drop:.1f}pp from full)")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Summary written: {summary_path}")

    meta_path = run_dir / "summary.json"
    meta_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "answer_model": str(args.answer_model),
                "conversations": int(args.conversations),
                "ac_threshold": AC_F1_THRESHOLD,
                "runs": {
                    name: {
                        "overall_f1": d["summary"].get("overall_f1", 0.0),
                        "by_category": d["summary"].get("by_category", {}),
                        "question_count": d["question_count"],
                        "errors": d["errors"],
                    }
                    for name, d in all_summaries.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── CLI ──────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.locomo.compare_modes",
        description="Compare Legacy vs Neo4j on LoCoMo (5 runs).",
    )
    p.add_argument(
        "--data", type=Path, default=_DEFAULT_DATA,
        help="path to locomo10.json",
    )
    p.add_argument(
        "--conversations", type=int, default=10, metavar="N",
        help="number of conversations (default: 10)",
    )
    p.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="retrieval top-k (default: 5)",
    )
    p.add_argument(
        "--answer-model", type=str, default="openai/qwen3.6-35b-a3b",
        dest="answer_model",
        help="answer generation model (default: openai/qwen3.6-35b-a3b)",
    )
    p.add_argument(
        "--judge", action="store_true",
        help="enable LLM judge",
    )
    p.add_argument(
        "--judge-model", type=str, default="gpt-4o", dest="judge_model",
        help="LLM judge model",
    )
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="results directory",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_comparison(args)


if __name__ == "__main__":
    main()
