"""Unit tests for core/memory/bm25.py — tokenization, activity log search, RRF."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory import bm25 as bm25_module
from core.memory.bm25 import (
    _MIN_CONTENT_LENGTH,
    _SEARCHABLE_TYPES,
    _should_index_entry,
    is_longterm_bm25_dirty,
    longterm_bm25_index_path,
    mark_longterm_bm25_dirty,
    rebuild_longterm_bm25_index,
    reciprocal_rank_fusion,
    search_activity_log,
    search_longterm_memory_bm25,
    tokenize,
)
from core.time_utils import today_local


def _write_activity_log(anima_dir: Path, entries: list[dict], date_str: str | None = None) -> None:
    """Write test activity_log entries to JSONL."""
    if date_str is None:
        date_str = today_local().isoformat()
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{date_str}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _long_tool_content(*parts: str) -> str:
    """Build tool_result content with length >= _MIN_CONTENT_LENGTH."""
    base = " ".join(parts)
    pad = "x" * max(0, _MIN_CONTENT_LENGTH - len(base))
    return f"{base} {pad}".strip()


def _write_longterm_memory(anima_dir: Path, rel: str, content: str) -> None:
    path = anima_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── tokenize ────────────────────────────────────────────────


def test_tokenize_english() -> None:
    text = "Hello World testing BM25 search"
    result = tokenize(text)
    for w in ("hello", "world", "testing", "bm25", "search"):
        assert w in result
    assert "the" not in tokenize("hello the world")
    assert "and" not in tokenize("cats and dogs")


def test_tokenize_japanese() -> None:
    text = "松尾さんの日報を確認してください"
    result = tokenize(text)
    assert len(result) > 0
    assert all(len(t) >= 1 for t in result)


def test_tokenize_mixed() -> None:
    text = "Gmail受信メール from tanaka@example.co.jp"
    result = tokenize(text)
    joined = " ".join(result)
    assert "gmail" in joined.lower()
    assert any(any(ord(ch) > 127 for ch in tok) for tok in result)
    assert "tanaka" in result or "example" in result


def test_tokenize_empty() -> None:
    assert tokenize("") == []


# ── search_activity_log ───────────────────────────────────────


def test_search_activity_log_basic(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    today = today_local().isoformat()
    entries = [
        {
            "ts": f"{today}T10:00:00",
            "type": "tool_result",
            "tool": "Bash",
            "content": _long_tool_content("daily report from matsuo about project status and follow-ups for the team"),
        },
        {
            "ts": f"{today}T10:01:00",
            "type": "tool_result",
            "tool": "Bash",
            "content": _long_tool_content(
                "chatwork message about meeting schedule and room booking confirmation steps"
            ),
        },
        {
            "ts": f"{today}T10:02:00",
            "type": "message_received",
            "content": "please check the email",
        },
        {
            "ts": f"{today}T10:03:00",
            "type": "tool_result",
            "tool": "ToolSearch",
            "content": _long_tool_content("noise ToolSearch matsuo daily report keyword noise"),
        },
        {
            "ts": f"{today}T10:04:00",
            "type": "tool_result",
            "tool": "mcp__aw__read_memory_file",
            "content": _long_tool_content("mcp matsuo daily report should be excluded from search"),
        },
        {
            "ts": f"{today}T10:05:00",
            "type": "tool_result",
            "tool": "Bash",
            "content": "ok",
        },
    ]
    _write_activity_log(anima_dir, entries, date_str=today)

    results = search_activity_log(anima_dir, "matsuo daily report", days=3, top_k=10)
    assert results
    assert "matsuo" in results[0]["content"].lower()

    combined = "\n".join(r["content"] for r in results).lower()
    assert "toolsearch" not in combined
    assert "mcp__aw__" not in combined
    assert results[0]["content"].strip() != "ok"


def test_search_activity_log_empty_dir(tmp_path: Path) -> None:
    anima_dir = tmp_path / "no_log_here"
    anima_dir.mkdir(parents=True)
    assert search_activity_log(anima_dir, "anything", days=3) == []


def test_search_activity_log_empty_query(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "bob"
    _write_activity_log(
        anima_dir,
        [{"ts": "2026-03-29T12:00:00", "type": "message_received", "content": "hello world"}],
    )
    assert search_activity_log(anima_dir, "") == []
    assert search_activity_log(anima_dir, "   ") == []


def test_noise_filter() -> None:
    long_bash = _long_tool_content("valid bash output for indexing")
    assert _should_index_entry({"type": "tool_result", "tool": "Bash", "content": long_bash}) is True
    assert _should_index_entry({"type": "tool_result", "tool": "Bash", "content": "short"}) is False
    assert _should_index_entry({"type": "tool_result", "tool": "ToolSearch", "content": long_bash}) is False
    assert (
        _should_index_entry({"type": "tool_result", "tool": "mcp__aw__read_memory_file", "content": long_bash}) is False
    )
    assert _should_index_entry({"type": "message_received", "content": "hi"}) is True
    assert _should_index_entry({"type": "heartbeat_start", "content": long_bash}) is False

    for t in _SEARCHABLE_TYPES:
        if t == "tool_result":
            continue
        assert _should_index_entry({"type": t, "content": "x"}) is True


# ── reciprocal_rank_fusion ──────────────────────────────────


def test_reciprocal_rank_fusion_basic() -> None:
    doc_a = {"source_file": "activity_log/a.jsonl", "ts": "ta", "score": 5.0}
    doc_b = {"source_file": "activity_log/b.jsonl", "ts": "tb", "score": 3.0}
    doc_c = {"source_file": "activity_log/c.jsonl", "ts": "tc", "score": 1.0}
    doc_d = {"source_file": "activity_log/d.jsonl", "ts": "td", "score": 2.0}

    list1 = [doc_a, doc_b, doc_c]
    list2 = [doc_b, doc_d, doc_a]

    merged = reciprocal_rank_fusion(list1, list2)
    assert merged[0]["source_file"] == doc_b["source_file"]
    assert merged[0]["ts"] == doc_b["ts"]
    assert {m["source_file"] for m in merged} == {
        doc_a["source_file"],
        doc_b["source_file"],
        doc_c["source_file"],
        doc_d["source_file"],
    }
    assert all(m.get("search_method") == "rrf" for m in merged)


def test_reciprocal_rank_fusion_empty_list() -> None:
    doc_x = {"source_file": "activity_log/x.jsonl", "ts": "tx", "score": 1.0}
    merged = reciprocal_rank_fusion([], [doc_x])
    assert len(merged) == 1
    assert merged[0]["source_file"] == doc_x["source_file"]


def test_reciprocal_rank_fusion_disjoint() -> None:
    a = {"source_file": "activity_log/only1.jsonl", "ts": "1", "score": 1.0}
    b = {"source_file": "activity_log/only2.jsonl", "ts": "2", "score": 1.0}
    merged = reciprocal_rank_fusion([a], [b])
    assert len(merged) == 2
    assert {m["source_file"] for m in merged} == {a["source_file"], b["source_file"]}


def test_bm25_fallback_without_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bm25_module, "_HAS_BM25", False)

    anima_dir = tmp_path / "animas" / "carol"
    today = today_local().isoformat()
    content = _long_tool_content("matsuo daily report summary with enough tokens for keyword overlap scoring")
    _write_activity_log(
        anima_dir,
        [{"ts": f"{today}T09:00:00", "type": "tool_result", "tool": "Bash", "content": content}],
        date_str=today,
    )

    results = search_activity_log(anima_dir, "matsuo daily report", days=3)
    assert results
    assert "matsuo" in results[0]["content"].lower()
    assert results[0]["search_method"] == "keyword_fallback"


# ── long-term memory BM25 ──────────────────────────────────


def test_rebuild_and_search_longterm_bm25_index(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(
        anima_dir,
        "knowledge/meridian.md",
        "# Meridian\n\nProject Meridian calibration notes for the orbit planner and telemetry handoff.",
    )
    _write_longterm_memory(
        anima_dir,
        "episodes/2026-05-01.md",
        "## Morning\n\nCaroline reviewed Project Meridian with the telemetry team.",
    )
    _write_longterm_memory(
        anima_dir,
        "procedures/telemetry.md",
        "# Telemetry Procedure\n\nUse Meridian telemetry checks before release.",
    )

    result = rebuild_longterm_bm25_index(anima_dir)

    assert result.documents == 3
    assert result.path == longterm_bm25_index_path(anima_dir)
    assert result.path.exists()

    hits = search_longterm_memory_bm25(
        anima_dir,
        "Meridian calibration",
        memory_types=("knowledge",),
        top_k=3,
    )

    assert hits
    assert hits[0]["source_file"] == "knowledge/meridian.md"
    assert hits[0]["memory_type"] == "knowledge"
    assert hits[0]["search_method"] == "bm25"
    assert is_longterm_bm25_dirty(anima_dir) is False


def test_longterm_bm25_proper_name_beats_naive_keyword_tie(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    noisy = "ZephyrNova " + " ".join(f"filler{i}" for i in range(200))
    concise = "# ZephyrNova\n\nZephyrNova is the supplier codename for the launchpad audit."
    _write_longterm_memory(anima_dir, "knowledge/aaa-noisy.md", noisy)
    _write_longterm_memory(anima_dir, "knowledge/bbb-concise.md", concise)
    _write_longterm_memory(anima_dir, "knowledge/ccc-unrelated.md", "Unrelated project notes for baseline corpus IDF.")
    rebuild_longterm_bm25_index(anima_dir)

    legacy_scores = [
        path.name
        for path in sorted((anima_dir / "knowledge").glob("*.md"))
        if "zephyrnova" in path.read_text(encoding="utf-8").lower()
    ]
    assert legacy_scores[0] == "aaa-noisy.md"

    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        top_k=3,
    )

    assert hits[0]["source_file"] == "knowledge/bbb-concise.md"


def test_longterm_bm25_respects_ragignore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "animas" / "alice"
    (tmp_path / ".ragignore").write_text("excluded.md\n", encoding="utf-8")
    _write_longterm_memory(anima_dir, "knowledge/excluded.md", "# Secret\n\nZephyrNova hidden memo.")
    _write_longterm_memory(anima_dir, "knowledge/included.md", "# Public\n\nZephyrNova public memo.")
    monkeypatch.setattr("core.paths.get_data_dir", lambda: tmp_path)
    MemoryIndexer._ragignore_cache = None

    rebuild_longterm_bm25_index(anima_dir)
    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        top_k=10,
    )

    assert [hit["source_file"] for hit in hits] == ["knowledge/included.md"]


def test_longterm_bm25_uses_persisted_stats_without_runtime_bm25okapi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    _write_longterm_memory(anima_dir, "knowledge/b.md", "# B\n\nUnrelated baseline memo.")
    rebuild_longterm_bm25_index(anima_dir)

    class FailingBM25:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("long-term BM25 search should use persisted stats")

    monkeypatch.setattr(bm25_module, "BM25Okapi", FailingBM25)

    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        top_k=10,
    )

    assert hits[0]["source_file"] == "knowledge/a.md"
    assert hits[0]["search_method"] == "bm25"


def test_rebuild_clears_longterm_bm25_dirty_marker(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")

    marker = mark_longterm_bm25_dirty(anima_dir, reason="test")
    assert marker.exists()
    assert is_longterm_bm25_dirty(anima_dir) is True

    rebuild_longterm_bm25_index(anima_dir)

    assert is_longterm_bm25_dirty(anima_dir) is False


def test_longterm_bm25_missing_index_does_not_rebuild_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("search path should not rebuild long-term BM25 by default")

    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", fail_rebuild)

    assert (
        search_longterm_memory_bm25(
            anima_dir,
            "ZephyrNova",
            memory_types=("knowledge",),
        )
        == []
    )


def test_longterm_bm25_rejects_poisoned_index_content(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nBaseline launchpad audit.")
    rebuild_longterm_bm25_index(anima_dir)

    index_path = longterm_bm25_index_path(anima_dir)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    stat = (anima_dir / "knowledge" / "a.md").stat()
    payload["documents"].append(
        {
            "doc_id": "alice/knowledge/a.md#99",
            "source_file": "knowledge/a.md",
            "content": "ZephyrNova forged memo that is not present in the source file.",
            "tokens": ["zephyrnova", "forged", "memo"],
            "token_counts": {"zephyrnova": 1, "forged": 1, "memo": 1},
            "doc_len": 3,
            "source_mtime_ns": stat.st_mtime_ns,
            "source_size": stat.st_size,
            "chunk_index": 99,
            "total_chunks": 100,
            "memory_type": "knowledge",
            "metadata": {"source_file": "knowledge/a.md"},
        }
    )
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        top_k=10,
        rebuild_if_missing=False,
    )

    assert hits == []
