from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark CLI runner
import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import SEARCH_MODES, AnimaWorksLoCoMoAdapter, load_dataset
from benchmarks.locomo.llm_config import default_answer_model
from benchmarks.locomo.metrics import CATEGORY_NAMES, compute_summary, eval_by_category, llm_judge_sync
from benchmarks.locomo.protocol import (
    PROTOCOL_VERSION,
    apply_standard_protocol_args,
    attach_standard_protocol_summary,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_DATA = _ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
_DEFAULT_OUTPUT = _ROOT / "benchmarks" / "locomo" / "results"

EMBEDDING_MODEL_DEFAULT = "intfloat/multilingual-e5-small"
_REFERENCE_SCORES: dict[str, dict[str, float]] = {
    "Mem0": {
        "overall": 66.88,
        "multi_hop": 51.15,
        "temporal": 55.51,
        "open_domain": 72.93,
    },
    "Zep": {
        "overall": 75.14,
        "multi_hop": 66.04,
        "temporal": 79.79,
        "open_domain": 67.71,
    },
}

# ── Path helpers ──────────


def _resolve_path(p: Path | str) -> Path:
    """Resolve a path; treat relative paths as relative to repo root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (_ROOT / path).resolve()


# ── Formatting ──────────


def _print_summary(mode: str, summary: dict[str, Any], *, total_count: int) -> None:
    """Print ASCII table for one mode's aggregate summary."""
    by_cat: dict[str, Any] = summary.get("by_category") or {}
    lines = [
        "+---------------------------------------------+",
        f"|  LoCoMo Results: {mode:<27}|",
        "+--------------+--------+-------+---------+",
        "| Category     | F1     | Judge | Count   |",
        "+--------------+--------+-------+---------+",
    ]
    for _cid in sorted(CATEGORY_NAMES):
        cname = CATEGORY_NAMES[_cid]
        if cname not in by_cat:
            continue
        row = by_cat[cname]
        f1v = float(row.get("f1", 0.0))
        jv = row.get("judge")
        cnt = int(row.get("count", 0))
        jstr = f"{float(jv) * 100.0:5.1f}%" if jv is not None else "   -  "
        lines.append(
            f"| {cname:<12} | {f1v * 100.0:5.1f}% | {jstr:>7} | {cnt:7d} |",
        )
    ojf = summary.get("overall_judge")
    oj_str = f"{float(ojf) * 100.0:5.1f}%" if ojf is not None else "   -  "
    of1 = float(summary.get("overall_f1", 0.0))
    lines.append("+--------------+--------+-------+---------+")
    lines.append(
        f"| {'OVERALL':<12} | {of1 * 100.0:5.1f}% | {oj_str:>7} | {total_count:7d} |",
    )
    lines.append("+--------------+--------+-------+---------+")
    error_rate = float(summary.get("error_rate", 0.0) or 0.0)
    if error_rate:
        lines.append(f"Error rate: {error_rate * 100.0:.1f}%")
    protocol = summary.get("standard_protocol")
    if isinstance(protocol, dict):
        primary = protocol.get("primary_value")
        primary_text = f"{float(primary) * 100.0:.1f}%" if primary is not None else "n/a"
        lines.append(
            f"Standard primary ({protocol.get('primary_metric', 'cat5_excluded.overall_judge')}): {primary_text}"
        )
    print("\n" + "\n".join(lines))


def _print_comparison(all_results: dict[str, Any]) -> None:
    """Print cross-mode F1 comparison with Mem0 reference column.

    Assumes the standard three ``SEARCH_MODES`` when ``--mode all`` completed.
    """
    modes: list[str] = [m for m in SEARCH_MODES if m in all_results]
    if not modes:
        return
    mem0 = _REFERENCE_SCORES.get("Mem0", {})
    print("\n+-------------------------------------------------------------+")
    print("|  LoCoMo Comparison (F1 %)                                   |")
    print("+--------------+---------+--------------+-----------+---------+")
    print(
        f"| {'Category':<12} | {'vector':>7} | {'vector_graph':>12} | {'scope_all':>9} | {'Mem0*':>7} |",
    )
    print("+--------------+---------+--------------+-----------+---------+")
    cat_order = [CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES)]
    for cname in cat_order:
        cells: list[str] = []
        for m in modes:
            summ = all_results[m].get("summary", {})
            bc = (summ.get("by_category") or {}).get(cname, {})
            f1 = float(bc.get("f1", 0.0)) * 100.0
            cells.append(f"{f1:5.1f}")
        while len(cells) < 3:
            cells.append("  -  ")
        m0v = mem0.get(cname)
        ref_s = f"{m0v:5.1f}" if m0v is not None else "  -  "
        print(
            f"| {cname:<12} | {cells[0]:>7} | {cells[1]:>12} | {cells[2]:>9} | {ref_s:>7} |",
        )
    o_cells: list[str] = []
    for m in modes:
        summ = all_results[m].get("summary", {})
        o_cells.append(f"{float(summ.get('overall_f1', 0.0)) * 100.0:5.1f}")
    while len(o_cells) < 3:
        o_cells.append("  -  ")
    mo = mem0.get("overall")
    ref_o = f"{mo:5.1f}" if mo is not None else "  -  "
    print("+--------------+---------+--------------+-----------+---------+")
    print(
        f"| {'OVERALL':<12} | {o_cells[0]:>7} | {o_cells[1]:>12} | {o_cells[2]:>9} | {ref_o:>7} |",
    )
    print("+--------------+---------+--------------+-----------+---------+")
    print(
        "* Reference scores from memobase benchmark (LLM Judge, cat5 excluded)",
    )


# ── Persistence ──────────


def _write_result_json(
    out_dir: Path,
    file_ts: str,
    mode: str,
    *,
    timestamp_iso: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    errors: int = 0,
) -> Path:
    """Write one mode's JSON results; creates ``out_dir`` on first use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{file_ts}_{mode}.json"
    payload: dict[str, Any] = {
        "mode": mode,
        "timestamp": timestamp_iso,
        "config": config,
        "summary": summary,
        "results": results,
        "errors": errors,
        "error_rate": summary.get("error_rate", 0.0),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _write_checkpoint_json(
    out_dir: Path,
    file_ts: str,
    mode: str,
    *,
    timestamp_iso: str,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    errors: int,
) -> Path:
    """Write an incremental checkpoint for long-running LoCoMo runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{file_ts}_{mode}.partial.json"
    payload: dict[str, Any] = {
        "mode": mode,
        "timestamp": timestamp_iso,
        "checkpoint": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": compute_summary(results),
        "results": results,
        "errors": errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Core ──────────


def _run_qa_loop(
    *,
    adapter: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    mode_label: str = "",
    checkpoint_writer: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Run ingest → retrieve → answer → score loop for each sample/question.

    Args:
        adapter: An adapter with ``reset()``, ``ingest_conversation()``,
            ``retrieve()``, and ``answer()`` methods.
        samples: LoCoMo samples to process.
        args: CLI namespace (``answer_model``, ``judge``, ``judge_model``,
            ``exclude_cat5``).
        mode_label: Display label for progress messages.

    Returns:
        ``(results, error_count)``
    """
    results: list[dict[str, Any]] = []
    errors = 0
    litellm_warned = False
    max_questions = int(getattr(args, "max_questions", 0) or 0)

    def _warn_litellm(exc: Exception) -> None:
        nonlocal litellm_warned
        if litellm_warned:
            return
        litellm_warned = True
        print(
            f"\nWarning: LiteLLM/API call failed ({exc!r}). "
            "Set API keys or provider credentials. "
            "Continuing with explicit error rows excluded from score aggregates.\n",
            file=sys.stderr,
        )

    for i, sample in enumerate(samples):
        if max_questions > 0 and len(results) >= max_questions:
            break
        sample_id = sample.get("sample_id", f"conv-{i}")
        print(f"\n[{i + 1}/{len(samples)}] {mode_label} | {sample_id}")
        t_conv0 = time.perf_counter()
        sample_result_start = len(results)

        try:
            adapter.reset()
            chunks = adapter.ingest_conversation(sample)
        except Exception as exc:
            errors += 1
            logger.exception("ingest_conversation failed for %s: %s", sample_id, exc)
            _warn_litellm(exc)
            print(f"  Skipped sample after ingest error: {exc}", file=sys.stderr)
            continue

        print(f"  Ingested: {chunks} chunks")

        qa_list = sample.get("qa", [])
        if not isinstance(qa_list, list):
            qa_list = []

        for j, qa in enumerate(qa_list):
            if max_questions > 0 and len(results) >= max_questions:
                break
            if not isinstance(qa, dict):
                continue
            if getattr(args, "exclude_cat5", False):
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

                benchmark_category = category if bool(getattr(args, "enable_locomo_category_branches", False)) else None
                base_result: dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "question_index": j,
                    "category": category,
                    "question": question,
                    "reference": answer,
                    "benchmark_category_used": benchmark_category is not None,
                }

                try:
                    context = adapter.retrieve(question, category=benchmark_category)
                except Exception as exc:
                    errors += 1
                    logger.exception("retrieve failed: %s", exc)
                    _warn_litellm(exc)
                    results.append(
                        {
                            **base_result,
                            "status": "error",
                            "error_stage": "retrieve",
                            "error_message": str(exc),
                            "prediction": "",
                            "f1": None,
                            "judge_score": None,
                            "context_count": 0,
                        }
                    )
                    continue

                prediction = ""
                try:
                    prediction = adapter.answer(
                        question,
                        context,
                        model=str(args.answer_model),
                        category=benchmark_category,
                    )
                except Exception as exc:
                    errors += 1
                    logger.exception("answer failed: %s", exc)
                    _warn_litellm(exc)
                    results.append(
                        {
                            **base_result,
                            "status": "error",
                            "error_stage": "answer",
                            "error_message": str(exc),
                            "prediction": "",
                            "f1": None,
                            "judge_score": None,
                            "context_count": len(context),
                        }
                    )
                    continue

                f1 = float(eval_by_category(prediction, answer, category))

                judge_score: float | None = None
                judge_error = ""
                if bool(getattr(args, "judge", False)):
                    try:
                        judge_result = llm_judge_sync(
                            question,
                            answer,
                            prediction,
                            model=str(args.judge_model),
                        )
                        if judge_result.get("verdict") == "error":
                            raise RuntimeError("llm_judge returned verdict=error")
                        judge_score = float(judge_result["score"])
                    except Exception as exc:
                        errors += 1
                        logger.exception("llm_judge failed: %s", exc)
                        _warn_litellm(exc)
                        judge_score = None
                        judge_error = str(exc)

                result: dict[str, Any] = {
                    **base_result,
                    "status": "ok",
                    "prediction": prediction,
                    "f1": f1,
                    "judge_score": judge_score,
                    "context_count": len(context),
                }
                if judge_error:
                    result["judge_error"] = judge_error
                raw_prediction = getattr(adapter, "_last_raw_answer", None)
                normalized_prediction = getattr(adapter, "_last_normalized_answer", None)
                abstain_reason = getattr(adapter, "_last_abstain_reason", "")
                top_score = getattr(adapter, "_last_top_score", None)
                top_event_time_iso = getattr(adapter, "_last_top_event_time_iso", "")
                if isinstance(raw_prediction, str):
                    result["raw_prediction"] = raw_prediction
                if isinstance(normalized_prediction, str):
                    result["normalized_prediction"] = normalized_prediction
                if isinstance(abstain_reason, str) and abstain_reason:
                    result["abstain_reason"] = abstain_reason
                if isinstance(top_score, int | float):
                    result["top_retrieval_score"] = float(top_score)
                if isinstance(top_event_time_iso, str) and top_event_time_iso:
                    result["top_event_time_iso"] = top_event_time_iso
                results.append(result)
                checkpoint_every = int(getattr(args, "checkpoint_every", 0) or 0)
                if checkpoint_writer is not None and checkpoint_every > 0 and len(results) % checkpoint_every == 0:
                    try:
                        checkpoint_writer(results, errors)
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        logger.exception("checkpoint write failed: %s", exc)
                if (j + 1) % 50 == 0:
                    print(f"  Questions: {j + 1}/{len(qa_list)}")
            except Exception as exc:
                errors += 1
                logger.exception("question loop error: %s", exc)
                _warn_litellm(exc)

        conv_elapsed = time.perf_counter() - t_conv0
        answered = len(results) - sample_result_start
        print(f"  Done: {answered}/{len(qa_list)} questions (elapsed: {conv_elapsed:.1f}s)")

    return results, errors


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Run LoCoMo for one or all search modes, aggregate metrics, and write JSON.

    For each mode, ingests each conversation, retrieves context per question, scores
    predictions with F1 and optionally LLM-judge, then saves ``{ts}_{mode}.json`` under
    the configured output directory.

    Args:
        args: Namespace from :func:`_build_arg_parser` (``data``, ``output``, ``mode``,
            ``conversations``, ``top_k``, ``judge``, models, etc.).

    Returns:
        ``(all_results, error_count)`` where ``all_results[mode]`` has ``summary``,
        ``results``, and ``config``; ``error_count`` is the number of recoverable failures
        (ingest, retrieve, answer, judge, or per-question).
    """
    apply_standard_protocol_args(args)
    data_path = _resolve_path(Path(args.data))
    if not data_path.is_file():
        print(
            f"Error: dataset file not found: {data_path}\n"
            "Download `locomo10.json` (LoCoMo dataset) and place it at:\n"
            f"  {data_path}\n"
            "See: https://github.com/snap-research/locomo and the LoCoMo paper for data.",
            file=sys.stderr,
        )
        return {}, 1

    dataset = load_dataset(data_path)
    n_conv = int(args.conversations)
    samples = dataset[:n_conv]
    if not samples:
        logger.warning("No conversations to process (empty dataset or conversations=0).")
        return {}, 0

    modes: tuple[str, ...] = SEARCH_MODES if args.mode == "all" else (args.mode,)
    all_results: dict[str, Any] = {}
    error_count = 0
    timestamp_iso = datetime.now().isoformat(timespec="seconds")
    file_ts = timestamp_iso.replace(":", "-")

    out_dir = _resolve_path(Path(args.output))

    t_total0 = time.perf_counter()
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}")

        mode_results: list[dict[str, Any]] = []
        enable_locomo_alias = bool(getattr(args, "enable_locomo_alias", False))
        enable_category_branches = bool(getattr(args, "enable_locomo_category_branches", False))
        config_block: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "search_mode": mode,
            "top_k": int(args.top_k),
            "exclude_cat5": bool(getattr(args, "exclude_cat5", False)),
            "answer_model": str(args.answer_model),
            "answer_timeout": getattr(args, "answer_timeout", None),
            "answer_max_retries": int(getattr(args, "answer_max_retries", 2) or 0),
            "checkpoint_every": int(getattr(args, "checkpoint_every", 0) or 0),
            "max_questions": int(getattr(args, "max_questions", 0) or 0),
            "judge_model": str(args.judge_model),
            "judge_enabled": bool(args.judge),
            "conversations": n_conv,
            "embedding_model": EMBEDDING_MODEL_DEFAULT,
            "leakage_alias_map_enabled": enable_locomo_alias,
            "category_branches_enabled": enable_category_branches,
            "category_dependent_normalization_enabled": enable_category_branches,
            "standard_protocol_requested": bool(getattr(args, "standard_protocol", False)),
        }
        try:

            def checkpoint_writer(
                current_results: list[dict[str, Any]],
                current_errors: int,
                *,
                checkpoint_mode: str = mode,
                checkpoint_config: dict[str, Any] = config_block,
            ) -> None:
                _write_checkpoint_json(
                    out_dir,
                    file_ts,
                    checkpoint_mode,
                    timestamp_iso=timestamp_iso,
                    config=checkpoint_config,
                    results=current_results,
                    errors=current_errors,
                )

            with AnimaWorksLoCoMoAdapter(
                search_mode=mode,
                top_k=int(args.top_k),
                answer_timeout=getattr(args, "answer_timeout", None),
                answer_max_retries=int(getattr(args, "answer_max_retries", 2) or 0),
                enable_locomo_alias=enable_locomo_alias,
                enable_locomo_category_branches=enable_category_branches,
            ) as adapter:
                mode_results, mode_errors = _run_qa_loop(
                    adapter=adapter,
                    samples=samples,
                    args=args,
                    mode_label=mode,
                    checkpoint_writer=checkpoint_writer,
                )
                error_count += mode_errors
        except Exception as exc:
            error_count += 1
            logger.exception("Mode %s failed: %s", mode, exc)
            print(f"Fatal error in mode {mode}: {exc}", file=sys.stderr)
            continue

        summary = compute_summary(mode_results)
        attach_standard_protocol_summary(summary, config_block)
        all_results[mode] = {
            "summary": summary,
            "results": mode_results,
            "config": config_block,
        }
        total_q = len(mode_results)
        _print_summary(mode, summary, total_count=total_q)

        _write_result_json(
            out_dir,
            file_ts,
            mode,
            timestamp_iso=timestamp_iso,
            config=config_block,
            summary=summary,
            results=mode_results,
            errors=mode_errors,
        )

    t_elapsed = time.perf_counter() - t_total0
    print(f"\nBenchmark completed in {t_elapsed:.1f}s total (wall).")
    return all_results, error_count


# ── CLI ──────────


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the LoCoMo runner."""
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.locomo.runner",
        description="Run LoCoMo memory benchmark with AnimaWorks RAG adapter.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help=f"path to locomo10.json (default: {_DEFAULT_DATA})",
    )
    p.add_argument(
        "--mode",
        choices=[*SEARCH_MODES, "all"],
        default="all",
        help="search mode: vector, vector_graph, scope_all, or all (default: all)",
    )
    p.add_argument(
        "--conversations",
        type=int,
        default=10,
        metavar="N",
        help="number of conversations to process (default: 10 = all in locomo10)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top_k",
        metavar="K",
        help="retrieval top-k (default: 10)",
    )
    p.add_argument(
        "--exclude-cat5",
        action="store_true",
        dest="exclude_cat5",
        help="exclude adversarial (category 5) questions from evaluation",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        help="enable LLM judge (default: off, F1 only)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        dest="judge_model",
        help="LLM judge model (default: gpt-4o)",
    )
    p.add_argument(
        "--answer-model",
        type=str,
        default=default_answer_model(),
        dest="answer_model",
        help=f"answer generation model (default: {default_answer_model()} via LiteLLM proxy)",
    )
    p.add_argument(
        "--answer-timeout",
        type=float,
        default=None,
        dest="answer_timeout",
        help="optional answer generation timeout in seconds (default: provider/client default)",
    )
    p.add_argument(
        "--answer-max-retries",
        type=int,
        default=2,
        dest="answer_max_retries",
        help="answer retries after the first attempt (default: 2 = 3 total attempts)",
    )
    p.add_argument(
        "--enable-locomo-alias",
        action="store_true",
        dest="enable_locomo_alias",
        help="enable historical LoCoMo alias-map expansion; default off because it contains test-set leakage",
    )
    p.add_argument(
        "--enable-locomo-category-branches",
        action="store_true",
        dest="enable_locomo_category_branches",
        help="enable gold-category-specific retrieval and answer normalization branches; default off",
    )
    p.add_argument(
        "--standard-protocol",
        action="store_true",
        help="run the canonical integrity protocol: 10 conversations, judge on, cat5 excluded, leakage flags off",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        dest="checkpoint_every",
        metavar="N",
        help="write a .partial.json checkpoint every N answered questions (default: disabled)",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=0,
        dest="max_questions",
        metavar="N",
        help="stop after N answered questions across selected conversations (default: all)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"results directory (default: {_DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    return p


def main() -> None:
    """CLI entry: parse args, run benchmark, print totals."""
    args = _build_arg_parser().parse_args()
    apply_standard_protocol_args(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    all_results, err = run_benchmark(args)
    if err and not all_results:
        sys.exit(1)
    print(f"\nTotal errors (recoverable + skipped): {err}")
    if args.mode == "all" and all_results:
        _print_comparison(all_results)


if __name__ == "__main__":
    main()
