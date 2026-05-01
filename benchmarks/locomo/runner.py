from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# LoCoMo benchmark CLI runner
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapter import SEARCH_MODES, AnimaWorksLoCoMoAdapter, load_dataset
from benchmarks.locomo.metrics import CATEGORY_NAMES, compute_summary, eval_by_category, llm_judge_sync

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
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── Core ──────────


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

    litellm_warned = False

    def _warn_litellm(first_exc: Exception) -> None:
        nonlocal litellm_warned
        if litellm_warned:
            return
        litellm_warned = True
        print(
            f"\nWarning: LiteLLM/API call failed ({first_exc!r}). "
            "Set API keys (e.g. OPENAI_API_KEY) or provider credentials. "
            "Continuing with empty predictions / F1-only where applicable.\n",
            file=sys.stderr,
        )

    t_total0 = time.perf_counter()
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}")

        mode_results: list[dict[str, Any]] = []
        config_block: dict[str, Any] = {
            "search_mode": mode,
            "top_k": int(args.top_k),
            "answer_model": str(args.answer_model),
            "judge_model": str(args.judge_model),
            "judge_enabled": bool(args.judge),
            "conversations": n_conv,
            "embedding_model": EMBEDDING_MODEL_DEFAULT,
        }
        try:
            with AnimaWorksLoCoMoAdapter(search_mode=mode, top_k=int(args.top_k)) as adapter:
                for i, sample in enumerate(samples):
                    sample_id = sample.get("sample_id", f"conv-{i}")
                    print(f"\n[{i + 1}/{len(samples)}] {sample_id}")
                    t_conv0 = time.perf_counter()

                    try:
                        adapter.reset()
                        chunks = adapter.ingest_conversation(sample)
                    except Exception as exc:
                        error_count += 1
                        logger.exception("ingest_conversation failed for %s: %s", sample_id, exc)
                        _warn_litellm(exc)
                        print(f"  Skipped sample after ingest error: {exc}", file=sys.stderr)
                        continue

                    print(f"  Ingested: {chunks} chunks")

                    qa_list = sample.get("qa", [])
                    if not isinstance(qa_list, list):
                        qa_list = []

                    for j, qa in enumerate(qa_list):
                        if not isinstance(qa, dict):
                            continue
                        # Skip adversarial questions if --exclude-cat5
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

                            try:
                                context = adapter.retrieve(question)
                            except Exception as exc:
                                error_count += 1
                                logger.exception("retrieve failed: %s", exc)
                                _warn_litellm(exc)
                                context = []

                            prediction = ""
                            try:
                                prediction = adapter.answer(
                                    question,
                                    context,
                                    model=str(args.answer_model),
                                )
                            except Exception as exc:
                                error_count += 1
                                logger.exception("answer failed: %s", exc)
                                _warn_litellm(exc)

                            f1 = float(eval_by_category(prediction, answer, category))

                            judge_score: float | None = None
                            if bool(args.judge):
                                try:
                                    judge_result = llm_judge_sync(
                                        question,
                                        answer,
                                        prediction,
                                        model=str(args.judge_model),
                                    )
                                    judge_score = float(judge_result["score"])
                                except Exception as exc:
                                    error_count += 1
                                    logger.exception("llm_judge failed: %s", exc)
                                    _warn_litellm(exc)
                                    judge_score = None

                            result: dict[str, Any] = {
                                "sample_id": str(sample_id),
                                "question_index": j,
                                "category": category,
                                "question": question,
                                "reference": answer,
                                "prediction": prediction,
                                "f1": f1,
                                "judge_score": judge_score,
                                "context_count": len(context),
                            }
                            mode_results.append(result)
                            if (j + 1) % 50 == 0:
                                print(f"  Questions: {j + 1}/{len(qa_list)}")
                        except Exception as exc:
                            error_count += 1
                            logger.exception("question loop error: %s", exc)
                            _warn_litellm(exc)

                    conv_elapsed = time.perf_counter() - t_conv0
                    print(f"  Done: {len(qa_list)} questions (elapsed: {conv_elapsed:.1f}s)")
        except Exception as exc:
            error_count += 1
            logger.exception("Mode %s failed: %s", mode, exc)
            print(f"Fatal error in mode {mode}: {exc}", file=sys.stderr)
            continue

        summary = compute_summary(mode_results)
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
        default="gpt-4o-mini",
        dest="answer_model",
        help="answer generation model (default: gpt-4o-mini)",
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
