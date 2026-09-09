"""Read-only, content-free activity/prompt/token report for a fixed date window.

Usage: python scripts/slim_runtime_report.py DATA_DIR --start 2026-09-01 --end 2026-09-07
Prints JSON to stdout; never reads config, credentials, or message bodies.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def summarize(data_dir: Path, start: date, end: date) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    task_ends: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    correlations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    invalid: Counter[str] = Counter()
    days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    files = 0
    agents = sorted(p for p in (data_dir / "animas").iterdir() if (p / "status.json").is_file())
    for index, anima in enumerate(agents):
        anonymous_name = f"agent-{index + 1:02d}"
        for family in ("activity_log", "prompt_logs", "token_usage"):
            for day in days:
                path = anima / family / f"{day}.jsonl"
                if not path.is_file():
                    continue
                files += 1
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise ValueError("not an object")
                        except (ValueError, TypeError):
                            invalid[family] += 1
                            continue
                        meta = event.get("meta") or {}
                        for field in ("task_id", "attempt_id", "request_id", "event_id"):
                            value = event.get(field) or meta.get(field)
                            if value:
                                correlations[field].add((anonymous_name, str(value)))
                        if family == "activity_log":
                            kind = event.get("type", "unknown")
                            counts[kind] += 1
                            if kind == "task_exec_end":
                                task_ends[meta.get("status", "unknown")] += 1
                            if kind == "tool_use" and event.get("tool") in {"search_memory", "read_memory_file"}:
                                counts[f"explicit_{event['tool']}"] += 1
                        else:
                            # Collapse task titles; no customer text in the output.
                            trigger = str(event.get("trigger", "unknown")).split(":", 1)[0]
                            key = "/".join(
                                (
                                    anonymous_name,
                                    family,
                                    trigger,
                                    str(event.get("mode", "unknown")),
                                    str(event.get("model", "unknown")),
                                )
                            )
                            group = groups[key]
                            group["records"] += 1
                            if family == "prompt_logs":
                                group[f"type:{event.get('type', 'unknown')}"] += 1
                                length = event.get("system_prompt_length")
                                if isinstance(length, (float, int)):
                                    group["system_prompt_characters"] += int(length)
                            else:
                                for field in ("input_tokens", "output_tokens", "cache_read_tokens", "duration_ms"):
                                    value = event.get(field)
                                    if isinstance(value, (float, int)):
                                        group[field] += value
                                    else:
                                        group[f"missing:{field}"] += 1
                                cost = event.get("estimated_cost_usd")
                                if isinstance(cost, (float, int)) and cost > 0:
                                    group["positive_estimated_cost_usd"] += cost
                                else:
                                    group["unknown_or_zero_estimated_cost_records"] += 1
    return {
        "window": {"start": start.isoformat(), "end_inclusive": end.isoformat()},
        "agents": len(agents),
        "files": files,
        "invalid_lines": invalid,
        "activity_events": counts,
        "task_end_events_by_status": task_ends,
        "distinct_available_correlation_ids": {k: len(v) for k, v in correlations.items()},
        "groups": dict(groups),
        "limitations": [
            "Events and session completions are not unique jobs or business acceptance.",
            "Correlation IDs are counted when present; missing IDs are never inferred from timestamps.",
            "Prompt lengths are characters, not measured tokens; auto-recall search counts may be unlogged.",
            "Zero/missing estimated cost is unknown; no total cost or ROI is inferred.",
            "Human correction time, acceptance and duplicate external effects require separate review.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("end must not precede start")
    rendered = json.dumps(summarize(args.data_dir, args.start, args.end), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
