#!/usr/bin/env python3
"""Run LoCoMo benchmark with Neo4j backend (single run).

Usage:
    python -m benchmarks.locomo.run_neo4j [--conversations N] [--answer-model MODEL] [--top-k K] [--judge] [--exclude-cat5]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import load_dataset
from benchmarks.locomo.metrics import CATEGORY_NAMES, compute_summary, eval_by_category, llm_judge_sync
from benchmarks.locomo.neo4j_adapter import AblationFlags, Neo4jLoCoMoAdapter

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks" / "locomo" / "results"

EMBEDDING_MODEL_DEFAULT = "intfloat/multilingual-e5-small"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoCoMo Neo4j benchmark (single run)")
    p.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    p.add_argument("--conversations", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--answer-model", type=str, default="openai/qwen3.6-35b-a3b", dest="answer_model")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge-model", type=str, default="gpt-4o", dest="judge_model")
    p.add_argument("--exclude-cat5", action="store_true", dest="exclude_cat5")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        print(f"Error: dataset not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(data_path)
    samples = dataset[: int(args.conversations)]
    if not samples:
        print("No samples.", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  LoCoMo Neo4j Benchmark")
    print(f"  Conversations: {len(samples)}, top_k: {args.top_k}")
    print(f"  Answer model: {args.answer_model}")
    print(f"  Exclude cat5: {args.exclude_cat5}")
    print(f"{'=' * 60}\n")

    config: dict[str, Any] = {
        "backend": "neo4j",
        "ablation": "full",
        "top_k": int(args.top_k),
        "answer_model": str(args.answer_model),
        "judge_model": str(args.judge_model),
        "judge_enabled": bool(args.judge),
        "conversations": len(samples),
        "embedding_model": EMBEDDING_MODEL_DEFAULT,
        "exclude_cat5": bool(args.exclude_cat5),
        "ablation_flags": {"reranker": True, "bfs": True, "community": True, "invalidation": True},
    }

    results: list[dict[str, Any]] = []
    errors = 0
    t0 = time.perf_counter()

    with Neo4jLoCoMoAdapter(
        top_k=int(args.top_k),
        ablation=AblationFlags(),
        answer_model=str(args.answer_model),
    ) as adapter:
        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"conv-{i}")
            print(f"\n[{i + 1}/{len(samples)}] {sample_id}")
            tc0 = time.perf_counter()

            try:
                adapter.reset()
                chunks = adapter.ingest_conversation(sample)
            except Exception as exc:
                errors += 1
                logger.exception("Ingest failed for %s: %s", sample_id, exc)
                print(f"  Skipped: {exc}", file=sys.stderr)
                continue

            print(f"  Ingested: {chunks} chunks")

            qa_list = sample.get("qa", [])
            if not isinstance(qa_list, list):
                qa_list = []

            for j, qa in enumerate(qa_list):
                if not isinstance(qa, dict):
                    continue

                if args.exclude_cat5:
                    try:
                        cat = int(qa.get("category", 0) or 0)
                    except (TypeError, ValueError):
                        cat = 0
                    if cat == 5:
                        continue

                try:
                    question = str(qa.get("question", "") or "")
                    answer = str(qa.get("answer", "") or "")
                    try:
                        category = int(qa.get("category", 0) or 0)
                    except (TypeError, ValueError):
                        category = 0
                    if not question or not answer:
                        continue

                    try:
                        context = adapter.retrieve(question)
                    except Exception as exc:
                        errors += 1
                        logger.exception("Retrieve failed: %s", exc)
                        context = []

                    prediction = ""
                    try:
                        prediction = adapter.answer(question, context)
                    except Exception as exc:
                        errors += 1
                        logger.exception("Answer failed: %s", exc)

                    f1 = float(eval_by_category(prediction, answer, category))

                    judge_score: float | None = None
                    if args.judge:
                        try:
                            jr = llm_judge_sync(question, answer, prediction, model=str(args.judge_model))
                            judge_score = float(jr["score"])
                        except Exception as exc:
                            errors += 1
                            logger.exception("Judge failed: %s", exc)

                    results.append({
                        "sample_id": str(sample_id),
                        "question_index": j,
                        "category": category,
                        "question": question,
                        "reference": answer,
                        "prediction": prediction,
                        "f1": f1,
                        "judge_score": judge_score,
                        "context_count": len(context),
                    })

                    if (j + 1) % 50 == 0:
                        print(f"  Questions: {j + 1}/{len(qa_list)}")
                except Exception as exc:
                    errors += 1
                    logger.exception("QA error: %s", exc)

            elapsed = time.perf_counter() - tc0
            print(f"  Done: {len(qa_list)} questions ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - t0
    summary = compute_summary(results)

    out_path = out_dir / f"{ts}_neo4j_full.json"
    out_path.write_text(
        json.dumps({"config": config, "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"  Results: {out_path}")
    print(f"  Overall F1: {summary.get('overall_f1', 0) * 100:.1f}%")
    print(f"  Questions: {len(results)}, Errors: {errors}")
    print(f"  Elapsed: {total_elapsed:.1f}s")
    by_cat = summary.get("by_category", {})
    for cat_name, cat_data in by_cat.items():
        f1_pct = cat_data.get("f1", 0) * 100
        cnt = cat_data.get("count", 0)
        jdg = cat_data.get("judge")
        jdg_str = f", judge={jdg * 100:.1f}%" if jdg is not None else ""
        print(f"  {cat_name:15s}: F1={f1_pct:5.1f}% (n={cnt}){jdg_str}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
