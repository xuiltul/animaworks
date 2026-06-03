from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""LoCoMo retrieval-only diagnostics for Legacy search modes."""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import (
    SEARCH_MODES,
    AnimaWorksLoCoMoAdapter,
    load_dataset,
    locomo_fact_index_enabled,
)
from benchmarks.locomo.metrics import CATEGORY_NAMES, _stemmed_tokens

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks" / "locomo" / "results" / "retrieval_diagnostics"


def answer_token_recall(answer: str, contexts: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Return token recall and all-present flag for an answer against contexts."""
    answer_tokens = _stemmed_tokens(answer)
    if not answer_tokens:
        return None, None

    context_text = "\n".join(str(item.get("content", "") or "") for item in contexts)
    context_tokens = _stemmed_tokens(context_text)
    if not context_tokens:
        return 0.0, 0.0

    answer_counts: Counter[str] = Counter(answer_tokens)
    context_counts: Counter[str] = Counter(context_tokens)
    matched = sum((answer_counts & context_counts).values())
    recall = matched / len(answer_tokens)
    return recall, 1.0 if matched == len(answer_tokens) else 0.0


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval diagnostics overall and by LoCoMo category."""

    def eligible(row: dict[str, Any]) -> bool:
        return int(row.get("category", 0) or 0) != 5 and row.get("answer_token_recall_at_10") is not None

    rows = [row for row in results if eligible(row)]
    summary = {
        "answer_token_recall_at_10": _mean(rows, "answer_token_recall_at_10"),
        "answer_token_recall_at_50": _mean(rows, "answer_token_recall_at_50"),
        "all_answer_tokens_present_at_10": _mean(rows, "all_answer_tokens_present_at_10"),
        "all_answer_tokens_present_at_50": _mean(rows, "all_answer_tokens_present_at_50"),
        "count": len(rows),
        "excluded_adversarial": sum(1 for row in results if int(row.get("category", 0) or 0) == 5),
        "by_category": {},
    }

    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[int(row.get("category", 0) or 0)].append(row)

    for category in sorted(by_cat):
        cat_rows = by_cat[category]
        summary["by_category"][CATEGORY_NAMES.get(category, str(category))] = {
            "count": len(cat_rows),
            "answer_token_recall_at_10": _mean(cat_rows, "answer_token_recall_at_10"),
            "answer_token_recall_at_50": _mean(cat_rows, "answer_token_recall_at_50"),
            "all_answer_tokens_present_at_10": _mean(cat_rows, "all_answer_tokens_present_at_10"),
            "all_answer_tokens_present_at_50": _mean(cat_rows, "all_answer_tokens_present_at_50"),
        }
    return summary


def run_retrieval_diagnostics(
    *,
    samples: list[dict[str, Any]],
    mode: str,
    top_k: int,
    ceiling_top_k: int,
    temporal_boost: bool = False,
    entity_boost: bool = False,
    entity_aware_graph: bool = False,
    fact_index: bool | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Run retrieval-only LoCoMo diagnostics and return results plus error count."""
    results: list[dict[str, Any]] = []
    errors = 0
    adapter = AnimaWorksLoCoMoAdapter(search_mode=mode, top_k=top_k)
    try:
        with _temporary_ablation_boosts(
            temporal_boost=temporal_boost,
            entity_boost=entity_boost,
            entity_aware_graph=entity_aware_graph,
            fact_index=fact_index,
        ):
            for sample_index, sample in enumerate(samples):
                sample_id = str(sample.get("sample_id", f"conv-{sample_index}"))
                print(f"\n[{sample_index + 1}/{len(samples)}] retrieval | {sample_id}")
                try:
                    adapter.reset()
                    chunks = adapter.ingest_conversation(sample)
                    fact_count = int(getattr(adapter, "_last_fact_count", 0) or 0)
                    print(f"  Ingested: {chunks} chunks; facts={fact_count}")
                except Exception:
                    errors += 1
                    continue

                qa_list = sample.get("qa", [])
                if not isinstance(qa_list, list):
                    continue
                for question_index, qa in enumerate(qa_list):
                    if not isinstance(qa, dict):
                        continue
                    question = str(qa.get("question", "") or "")
                    answer = str(qa.get("answer", "") or "")
                    try:
                        category = int(qa.get("category", 0) or 0)
                    except (TypeError, ValueError):
                        category = 0
                    if not question or not answer:
                        continue
                    try:
                        top_context = _retrieve_at_k(adapter, question, category=category, top_k=top_k)
                        ceiling_context = (
                            top_context
                            if ceiling_top_k == top_k
                            else _retrieve_at_k(adapter, question, category=category, top_k=ceiling_top_k)
                        )
                    except Exception:
                        errors += 1
                        top_context = []
                        ceiling_context = []

                    recall_10, all_10 = answer_token_recall(answer, top_context)
                    recall_50, all_50 = answer_token_recall(answer, ceiling_context)
                    top = _top_context(top_context)
                    top_meta = top.get("metadata") if isinstance(top.get("metadata"), dict) else {}
                    fact_count = int(getattr(adapter, "_last_fact_count", 0) or 0)
                    excluded = category == 5
                    results.append(
                        {
                            "sample_id": sample_id,
                            "question_index": question_index,
                            "category": category,
                            "question": question,
                            "reference": answer,
                            "excluded_from_recall": excluded,
                            "context_count": len(top_context),
                            "context_count_at_50": len(ceiling_context),
                            "fact_count": fact_count,
                            "top_retrieval_score": _top_score(top_context),
                            "top_memory_type": str(top_meta.get("memory_type", "") or ""),
                            "top_event_time_iso": str(top_meta.get("event_time_iso", "") or ""),
                            "top_entity_boost": top_meta.get("entity_boost"),
                            "top_entity_overlap": top_meta.get("entity_overlap", []),
                            "answer_token_recall_at_10": None if excluded else recall_10,
                            "answer_token_recall_at_50": None if excluded else recall_50,
                            "all_answer_tokens_present_at_10": None if excluded else all_10,
                            "all_answer_tokens_present_at_50": None if excluded else all_50,
                        },
                    )
                    if (question_index + 1) % 25 == 0:
                        print(f"  Questions: {question_index + 1}/{len(qa_list)}")
    finally:
        adapter.cleanup()
    return results, errors


def write_diagnostics_json(
    output: Path,
    *,
    mode: str,
    conversations: int,
    top_k: int,
    ceiling_top_k: int,
    temporal_boost: bool,
    entity_boost: bool,
    entity_aware_graph: bool = False,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    errors: int,
    temporal_ablation: dict[str, Any] | None = None,
    entity_ablation: dict[str, Any] | None = None,
    entity_aware_graph_ablation: dict[str, Any] | None = None,
    fact_index: bool = False,
    fact_ablation: dict[str, Any] | None = None,
) -> Path:
    """Write diagnostics JSON to a directory or explicit JSON path."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "config": {
            "mode": mode,
            "conversations": conversations,
            "top_k": top_k,
            "ceiling_top_k": ceiling_top_k,
            "temporal_boost": temporal_boost,
            "entity_boost": entity_boost,
            "entity_aware_graph": entity_aware_graph,
            "fact_index": fact_index,
        },
        "summary": summary,
        "results": results,
        "errors": errors,
    }
    if temporal_ablation is not None:
        payload["temporal_ablation"] = temporal_ablation
    if entity_ablation is not None:
        payload["entity_ablation"] = entity_ablation
    if entity_aware_graph_ablation is not None:
        payload["entity_aware_graph_ablation"] = entity_aware_graph_ablation
    if fact_ablation is not None:
        payload["fact_ablation"] = fact_ablation

    if output.suffix == ".json":
        path = output
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        file_ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = output / f"{file_ts}_{mode}_retrieval_diagnostics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _retrieve_at_k(
    adapter: AnimaWorksLoCoMoAdapter,
    question: str,
    *,
    category: int,
    top_k: int,
) -> list[dict[str, Any]]:
    previous_top_k = adapter._top_k
    adapter._top_k = top_k
    try:
        return adapter.retrieve(question, category=category)
    finally:
        adapter._top_k = previous_top_k


def _top_context(context: list[dict[str, Any]]) -> dict[str, Any]:
    return {} if not context else max(context, key=lambda item: float(item.get("score", 0.0) or 0.0))


def _top_score(context: list[dict[str, Any]]) -> float | None:
    return None if not context else float(_top_context(context).get("score", 0.0) or 0.0)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _ablation_delta(base: dict[str, Any], boosted: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "answer_token_recall_at_10",
        "answer_token_recall_at_50",
        "all_answer_tokens_present_at_10",
        "all_answer_tokens_present_at_50",
    )
    delta: dict[str, Any] = {}
    for key in keys:
        delta[key] = _numeric_delta(base.get(key), boosted.get(key))
    delta["by_category"] = {}
    base_by = base.get("by_category", {})
    boosted_by = boosted.get("by_category", {})
    if isinstance(base_by, dict) and isinstance(boosted_by, dict):
        for category in sorted(set(base_by) | set(boosted_by)):
            base_cat = base_by.get(category, {}) if isinstance(base_by.get(category, {}), dict) else {}
            boosted_cat = boosted_by.get(category, {}) if isinstance(boosted_by.get(category, {}), dict) else {}
            delta["by_category"][category] = {
                key: _numeric_delta(base_cat.get(key), boosted_cat.get(key)) for key in keys
            }
    return delta


def _numeric_delta(base: Any, boosted: Any) -> float | None:
    return None if base is None or boosted is None else float(boosted) - float(base)


@contextmanager
def _temporary_temporal_boost(enabled: bool) -> Iterator[None]:
    with _temporary_env_flag("LOCOMO_TEMPORAL_BOOST", enabled):
        yield


@contextmanager
def _temporary_entity_boost(enabled: bool) -> Iterator[None]:
    with _temporary_env_flag("LOCOMO_ENTITY_BOOST", enabled):
        yield


@contextmanager
def _temporary_entity_aware_graph(enabled: bool) -> Iterator[None]:
    with _temporary_env_flag("LOCOMO_ENTITY_AWARE_GRAPH", enabled):
        yield


@contextmanager
def _temporary_fact_index(enabled: bool | None) -> Iterator[None]:
    if enabled is None:
        yield
        return
    with _temporary_env_flag("LOCOMO_FACT_INDEX", enabled):
        yield


@contextmanager
def _temporary_ablation_boosts(
    *,
    temporal_boost: bool,
    entity_boost: bool,
    entity_aware_graph: bool,
    fact_index: bool | None,
) -> Iterator[None]:
    with (
        _temporary_temporal_boost(temporal_boost),
        _temporary_entity_boost(entity_boost),
        _temporary_entity_aware_graph(entity_aware_graph),
        _temporary_fact_index(fact_index),
    ):
        yield


@contextmanager
def _temporary_env_flag(name: str, enabled: bool) -> Iterator[None]:
    previous = os.environ.get(name)
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoCoMo retrieval-only diagnostics")
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=SEARCH_MODES, default="scope_all")
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ceiling-top-k", type=int, default=50)
    parser.add_argument("--temporal-ablation", action="store_true")
    parser.add_argument("--entity-ablation", action="store_true")
    parser.add_argument("--entity-aware-graph-ablation", action="store_true")
    parser.add_argument("--fact-ablation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ceiling_top_k < args.top_k:
        raise SystemExit("--ceiling-top-k must be >= --top-k")

    data_path = args.data if args.data.is_absolute() else (_ROOT / args.data).resolve()
    samples = load_dataset(data_path)[: max(0, int(args.conversations))]
    primary_fact_index = locomo_fact_index_enabled()
    if args.fact_ablation:
        primary_fact_index = False

    started = time.perf_counter()
    results, errors = run_retrieval_diagnostics(
        samples=samples,
        mode=str(args.mode),
        top_k=int(args.top_k),
        ceiling_top_k=int(args.ceiling_top_k),
        temporal_boost=False,
        entity_boost=False,
        entity_aware_graph=False,
        fact_index=primary_fact_index,
    )
    summary = summarize_results(results)

    temporal_ablation: dict[str, Any] | None = None
    entity_ablation: dict[str, Any] | None = None
    entity_aware_graph_ablation: dict[str, Any] | None = None
    fact_ablation: dict[str, Any] | None = None
    if args.temporal_ablation:
        boosted_results, boosted_errors = run_retrieval_diagnostics(
            samples=samples,
            mode=str(args.mode),
            top_k=int(args.top_k),
            ceiling_top_k=int(args.ceiling_top_k),
            temporal_boost=True,
            entity_boost=False,
            entity_aware_graph=False,
            fact_index=primary_fact_index,
        )
        boosted_summary = summarize_results(boosted_results)
        temporal_ablation = {
            "config": {"temporal_boost": True},
            "summary": boosted_summary,
            "results": boosted_results,
            "errors": boosted_errors,
            "deltas": _ablation_delta(summary, boosted_summary),
        }
        errors += boosted_errors
    if args.entity_ablation:
        boosted_results, boosted_errors = run_retrieval_diagnostics(
            samples=samples,
            mode=str(args.mode),
            top_k=int(args.top_k),
            ceiling_top_k=int(args.ceiling_top_k),
            temporal_boost=False,
            entity_boost=True,
            entity_aware_graph=False,
            fact_index=primary_fact_index,
        )
        boosted_summary = summarize_results(boosted_results)
        entity_ablation = {
            "config": {"entity_boost": True},
            "summary": boosted_summary,
            "results": boosted_results,
            "errors": boosted_errors,
            "deltas": _ablation_delta(summary, boosted_summary),
        }
        errors += boosted_errors
    if args.entity_aware_graph_ablation:
        boosted_results, boosted_errors = run_retrieval_diagnostics(
            samples=samples,
            mode=str(args.mode),
            top_k=int(args.top_k),
            ceiling_top_k=int(args.ceiling_top_k),
            temporal_boost=False,
            entity_boost=False,
            entity_aware_graph=True,
            fact_index=primary_fact_index,
        )
        boosted_summary = summarize_results(boosted_results)
        entity_aware_graph_ablation = {
            "config": {"entity_aware_graph": True},
            "summary": boosted_summary,
            "results": boosted_results,
            "errors": boosted_errors,
            "deltas": _ablation_delta(summary, boosted_summary),
        }
        errors += boosted_errors
    if args.fact_ablation:
        boosted_results, boosted_errors = run_retrieval_diagnostics(
            samples=samples,
            mode=str(args.mode),
            top_k=int(args.top_k),
            ceiling_top_k=int(args.ceiling_top_k),
            temporal_boost=False,
            entity_boost=False,
            entity_aware_graph=False,
            fact_index=True,
        )
        boosted_summary = summarize_results(boosted_results)
        fact_ablation = {
            "config": {"fact_index": True},
            "summary": boosted_summary,
            "results": boosted_results,
            "errors": boosted_errors,
            "deltas": _ablation_delta(summary, boosted_summary),
        }
        errors += boosted_errors

    out = write_diagnostics_json(
        args.output,
        mode=str(args.mode),
        conversations=len(samples),
        top_k=int(args.top_k),
        ceiling_top_k=int(args.ceiling_top_k),
        temporal_boost=False,
        entity_boost=False,
        entity_aware_graph=False,
        fact_index=primary_fact_index,
        summary=summary,
        results=results,
        errors=errors,
        temporal_ablation=temporal_ablation,
        entity_ablation=entity_ablation,
        entity_aware_graph_ablation=entity_aware_graph_ablation,
        fact_ablation=fact_ablation,
    )
    elapsed = time.perf_counter() - started
    print(f"\nWrote {out}")
    print(f"Retrieval diagnostics complete in {elapsed:.1f}s; errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
