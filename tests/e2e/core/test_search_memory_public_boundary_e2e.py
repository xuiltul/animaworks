from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.memory.manager import MemoryManager
from core.tooling.handler import ToolHandler


def _handler(anima_dir: Path) -> ToolHandler:
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.json").write_text(
        '{"version":1,"file_roots":["/"],"commands":{"allow_all":true,"allow":[],"deny":[]},'
        '"external_tools":{"allow_all":true},"tool_creation":{"personal":true,"shared":false}}',
        encoding="utf-8",
    )
    return ToolHandler(anima_dir, MemoryManager(anima_dir), tool_registry=[], superuser=True)


def test_public_search_reads_fresh_mixed_script_write_and_abstains(tmp_path: Path) -> None:
    handler = _handler(tmp_path / "animas" / "test")

    handler.handle(
        "write_memory_file",
        {"path": "knowledge/retention.md", "content": "Dashboard集計問題とMetrics保管方針"},
    )

    hit = handler.handle("search_memory", {"query": "Metrics", "scope": "knowledge"})
    miss = handler.handle("search_memory", {"query": "qzxv_nonexistent_847291", "scope": "knowledge"})

    assert "knowledge/retention.md" in hit
    assert "Metrics保管方針" in hit
    assert miss == "No results for 'qzxv_nonexistent_847291'"

    handler.handle("read_memory_file", {"path": "knowledge/retention.md"})
    handler.handle(
        "write_memory_file",
        {"path": "knowledge/retention.md", "content": "Meridian device policy", "mode": "overwrite"},
    )
    assert "No results" in handler.handle("search_memory", {"query": "Metrics", "scope": "knowledge"})
    assert "knowledge/retention.md" in handler.handle("search_memory", {"query": "Meridian", "scope": "knowledge"})

    handler.handle(
        "archive_memory_file",
        {"path": "knowledge/retention.md", "reason": "superseded in boundary test"},
    )
    assert "No results" in handler.handle("search_memory", {"query": "Meridian", "scope": "knowledge"})


def test_public_activity_search_applies_exact_time_range(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    handler = _handler(anima_dir)
    now = datetime.now(UTC)
    day = now.date().isoformat()
    early = now.replace(hour=8, minute=0, second=0, microsecond=0)
    late = early + timedelta(hours=2)
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir()
    (log_dir / f"{day}.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": ts.isoformat(), "type": "message_received", "content": content})
            for ts, content in ((early, "Meridian early"), (late, "Meridian late"))
        )
        + "\n",
        encoding="utf-8",
    )

    result = handler.handle(
        "search_memory",
        {
            "query": "Meridian",
            "scope": "activity_log",
            "time_range": {
                "after": (early + timedelta(minutes=30)).isoformat(),
                "before": (late + timedelta(minutes=30)).isoformat(),
            },
        },
    )

    assert "Meridian late" in result
    assert "Meridian early" not in result


def test_public_search_fixed_gold_set_meets_hit_targets(tmp_path: Path) -> None:
    handler = _handler(tmp_path / "animas" / "test")
    gold = {
        "Widget": ("knowledge/retention.md", "Widget metrics retention policy"),
        "espresso": ("knowledge/coffee.md", "Alice prefers espresso for morning focus"),
        "ZephyrNova": ("knowledge/project.md", "ZephyrNova launch checklist and telemetry"),
        "請求書": ("knowledge/accounting.md", "請求書の月次照合と承認手順"),
        "incident": ("knowledge/ops.md", "production incident response procedure"),
    }
    for source, content in gold.values():
        handler.handle("write_memory_file", {"path": source, "content": content})

    hit_at_1 = 0
    for query, (source, _content) in gold.items():
        result = handler.handle("search_memory", {"query": query, "scope": "knowledge"})
        assert source in result
        first_result = next(line for line in result.splitlines() if line.startswith("[1]"))
        hit_at_1 += source in first_result

    assert hit_at_1 / len(gold) >= 0.8
    assert "No results" in handler.handle("search_memory", {"query": "qzxv_negative_gold_9384", "scope": "knowledge"})
