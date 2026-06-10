from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Compare LoCoMo benchmark result JSON files."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.locomo.metrics import CATEGORY_NAMES, compute_summary


@dataclass(frozen=True)
class ResultBundle:
    """Loaded LoCoMo result payload plus derived aggregates."""

    path: Path
    payload: dict[str, Any]
    results: list[dict[str, Any]]
    summary: dict[str, Any]
    cat5_excluded_summary: dict[str, Any]
    blank_predictions: int
    missing_summary: bool


def load_result_bundle(path: Path) -> ResultBundle:
    """Load a LoCoMo result JSON and derive missing summary fields."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LoCoMo result must be a JSON object: {path}")
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"LoCoMo result has non-list results: {path}")
    rows = [row for row in results if isinstance(row, dict)]
    raw_summary = payload.get("summary")
    missing_summary = not isinstance(raw_summary, dict)
    summary = raw_summary if isinstance(raw_summary, dict) else compute_summary(rows)
    cat5_excluded = [row for row in rows if int(row.get("category", 0) or 0) != 5]
    return ResultBundle(
        path=path,
        payload=payload,
        results=rows,
        summary=summary,
        cat5_excluded_summary=compute_summary(cat5_excluded),
        blank_predictions=sum(1 for row in rows if _is_blank_prediction(row)),
        missing_summary=missing_summary,
    )


def compare_result_files(
    before_path: Path,
    after_path: Path,
    *,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Compare two LoCoMo result JSON files."""
    before = load_result_bundle(before_path)
    after = load_result_bundle(after_path)
    before_by_key = {_question_key(row): row for row in before.results}
    after_by_key = {_question_key(row): row for row in after.results}
    common_keys = sorted(set(before_by_key) & set(after_by_key))

    deltas = {
        "overall_f1": _numeric_delta(before.summary.get("overall_f1"), after.summary.get("overall_f1")),
        "cat5_excluded_overall_f1": _numeric_delta(
            before.cat5_excluded_summary.get("overall_f1"),
            after.cat5_excluded_summary.get("overall_f1"),
        ),
        "blank_predictions": after.blank_predictions - before.blank_predictions,
        "by_category": _category_deltas(before.summary, after.summary),
        "cat5_excluded_by_category": _category_deltas(before.cat5_excluded_summary, after.cat5_excluded_summary),
    }
    per_question = _per_question_deltas(before_by_key, after_by_key, common_keys, max_examples=max_examples)
    return {
        "before": _bundle_summary(before),
        "after": _bundle_summary(after),
        "deltas": deltas,
        "questions": {
            "common_count": len(common_keys),
            "before_only_count": len(set(before_by_key) - set(after_by_key)),
            "after_only_count": len(set(after_by_key) - set(before_by_key)),
            **per_question,
        },
        "warnings": _condition_warnings(before, after),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for humans."""
    before = report["before"]
    after = report["after"]
    deltas = report["deltas"]
    questions = report["questions"]
    lines = [
        "# LoCoMo Result Comparison",
        "",
        "| Metric | Before | After | Delta |",
        "|--------|--------|-------|-------|",
        (
            "| Overall F1 | "
            f"{_pct(before['overall_f1'])} | {_pct(after['overall_f1'])} | {_signed_pct(deltas['overall_f1'])} |"
        ),
        (
            "| Cat5-excluded F1 | "
            f"{_pct(before['cat5_excluded_overall_f1'])} | "
            f"{_pct(after['cat5_excluded_overall_f1'])} | "
            f"{_signed_pct(deltas['cat5_excluded_overall_f1'])} |"
        ),
        f"| Blank predictions | {before['blank_predictions']} | {after['blank_predictions']} | {deltas['blank_predictions']:+d} |",
        f"| Questions | {before['question_count']} | {after['question_count']} | |",
        "",
        "## Category Deltas",
        "",
        "| Category | Before F1 | After F1 | Delta |",
        "|----------|-----------|----------|-------|",
    ]
    for category, row in sorted(deltas["by_category"].items()):
        lines.append(
            f"| {category} | {_pct(row.get('before_f1'))} | {_pct(row.get('after_f1'))} | "
            f"{_signed_pct(row.get('f1_delta'))} |",
        )
    lines.extend(
        [
            "",
            "## Question Deltas",
            "",
            f"- Common questions: {questions['common_count']}",
            f"- Before-only questions: {questions['before_only_count']}",
            f"- After-only questions: {questions['after_only_count']}",
            f"- Changed predictions: {questions['changed_prediction_count']}",
            "",
            "### Worst Regressions",
            "",
        ],
    )
    lines.extend(_question_lines(questions.get("worst_regressions", [])))
    lines.extend(["", "### Best Improvements", ""])
    lines.extend(_question_lines(questions.get("best_improvements", [])))
    warnings = list(report.get("warnings", []) or [])
    if before["missing_summary"]:
        warnings.append(f"Before summary was missing and recomputed from results: {before['path']}")
    if after["missing_summary"]:
        warnings.append(f"After summary was missing and recomputed from results: {after['path']}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _bundle_summary(bundle: ResultBundle) -> dict[str, Any]:
    return {
        "path": str(bundle.path),
        "question_count": len(bundle.results),
        "blank_predictions": bundle.blank_predictions,
        "overall_f1": bundle.summary.get("overall_f1"),
        "cat5_excluded_overall_f1": bundle.cat5_excluded_summary.get("overall_f1"),
        "summary": bundle.summary,
        "cat5_excluded_summary": bundle.cat5_excluded_summary,
        "missing_summary": bundle.missing_summary,
        "errors": bundle.payload.get("errors"),
        "config": bundle.payload.get("config", {}),
    }


def _condition_warnings(before: ResultBundle, after: ResultBundle) -> list[str]:
    before_config = before.payload.get("config") if isinstance(before.payload.get("config"), dict) else {}
    after_config = after.payload.get("config") if isinstance(after.payload.get("config"), dict) else {}
    keys = (
        "protocol_version",
        "leakage_alias_map_enabled",
        "category_branches_enabled",
        "category_dependent_normalization_enabled",
        "judge_enabled",
        "exclude_cat5",
        "answer_model",
        "judge_model",
        "conversations",
    )
    warnings: list[str] = []
    for key in keys:
        before_value = before_config.get(key)
        after_value = after_config.get(key)
        if before_value != after_value:
            warnings.append(f"Benchmark condition differs for `{key}`: before={before_value!r}, after={after_value!r}")
    return warnings


def _question_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- None"]
    lines: list[str] = []
    for row in rows:
        lines.append(
            "- "
            f"{row['sample_id']} q{row['question_index']} "
            f"({row['category_name']}): {_signed_pct(row['f1_delta'])} "
            f"- {row['question']}",
        )
    return lines


def _per_question_deltas(
    before_by_key: dict[tuple[str, int, str], dict[str, Any]],
    after_by_key: dict[tuple[str, int, str], dict[str, Any]],
    common_keys: list[tuple[str, int, str]],
    *,
    max_examples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    changed_predictions = 0
    for key in common_keys:
        before = before_by_key[key]
        after = after_by_key[key]
        before_prediction = _prediction_text(before)
        after_prediction = _prediction_text(after)
        if before_prediction != after_prediction:
            changed_predictions += 1
        category = int(after.get("category", before.get("category", 0)) or 0)
        before_f1 = float(before.get("f1", 0.0) or 0.0)
        after_f1 = float(after.get("f1", 0.0) or 0.0)
        rows.append(
            {
                "sample_id": key[0],
                "question_index": key[1],
                "question": key[2],
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, f"category_{category}"),
                "before_f1": before_f1,
                "after_f1": after_f1,
                "f1_delta": after_f1 - before_f1,
                "before_prediction": before_prediction,
                "after_prediction": after_prediction,
                "before_blank": _is_blank_prediction(before),
                "after_blank": _is_blank_prediction(after),
            },
        )
    regressions = [row for row in rows if row["f1_delta"] < 0]
    improvements = [row for row in rows if row["f1_delta"] > 0]
    return {
        "changed_prediction_count": changed_predictions,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "worst_regressions": sorted(regressions, key=lambda row: row["f1_delta"])[:max_examples],
        "best_improvements": sorted(improvements, key=lambda row: row["f1_delta"], reverse=True)[:max_examples],
    }


def _category_deltas(before_summary: dict[str, Any], after_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    before_by = before_summary.get("by_category", {})
    after_by = after_summary.get("by_category", {})
    if not isinstance(before_by, dict) or not isinstance(after_by, dict):
        return out
    for category in sorted(set(before_by) | set(after_by)):
        before_row = before_by.get(category, {}) if isinstance(before_by.get(category), dict) else {}
        after_row = after_by.get(category, {}) if isinstance(after_by.get(category), dict) else {}
        before_f1 = before_row.get("f1")
        after_f1 = after_row.get("f1")
        out[category] = {
            "before_f1": before_f1,
            "after_f1": after_f1,
            "f1_delta": _numeric_delta(before_f1, after_f1),
            "before_count": before_row.get("count"),
            "after_count": after_row.get("count"),
        }
    return out


def _question_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sample_id", "")),
        int(row.get("question_index", 0) or 0),
        str(row.get("question", "") or ""),
    )


def _is_blank_prediction(row: dict[str, Any]) -> bool:
    return _prediction_text(row) == ""


def _prediction_text(row: dict[str, Any]) -> str:
    value = row.get("prediction")
    if value is None:
        value = row.get("normalized_prediction")
    return str(value or "").strip()


def _numeric_delta(before: Any, after: Any) -> float | None:
    return None if before is None or after is None else float(after) - float(before)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"


def _signed_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:+.2f}pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two LoCoMo result JSON files")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, help="write comparison JSON to this path")
    parser.add_argument("--markdown", type=Path, help="write Markdown report to this path")
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare_result_files(args.before, args.after, max_examples=max(0, int(args.max_examples)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if not args.output and not args.markdown:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
