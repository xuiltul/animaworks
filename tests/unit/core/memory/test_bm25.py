"""Unit tests for core/memory/bm25.py — tokenization, activity log search, RRF."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.memory import bm25 as bm25_module
from core.memory.bm25 import (
    _MIN_CONTENT_LENGTH,
    _SEARCHABLE_TYPES,
    _should_index_entry,
    is_longterm_bm25_dirty,
    longterm_bm25_delta_path,
    longterm_bm25_index_path,
    longterm_bm25_rebuild_marker_path,
    mark_longterm_bm25_dirty,
    maybe_rebuild_dirty_longterm_bm25,
    rebuild_longterm_bm25_index,
    reciprocal_rank_fusion,
    search_activity_log,
    search_longterm_memory_bm25,
    tokenize,
    update_longterm_bm25_source,
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


def test_tokenize_splits_ascii_at_cjk_boundaries() -> None:
    tokens = tokenize("Dashboard集計問題とownerのMetrics保管方針")
    assert tokens == ["dashboard", "集計問題と", "owner", "の", "metrics", "保管方針"]


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


def test_activity_requires_actual_token_match_even_for_zero_scores(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    today = today_local().isoformat()
    _write_activity_log(
        anima_dir,
        [
            {"ts": f"{today}T09:00:00Z", "type": "message_received", "content": "common alpha"},
            {"ts": f"{today}T10:00:00Z", "type": "message_received", "content": "common beta"},
        ],
    )
    assert search_activity_log(anima_dir, "qzxv_nonexistent_847291") == []
    assert len(search_activity_log(anima_dir, "common")) == 2


def test_activity_time_range_filters_matches(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    today = today_local().isoformat()
    _write_activity_log(
        anima_dir,
        [
            {"ts": f"{today}T09:00:00Z", "type": "message_received", "content": "Meridian early"},
            {"ts": f"{today}T11:00:00Z", "type": "message_received", "content": "Meridian late"},
        ],
    )
    hits = search_activity_log(
        anima_dir,
        "Meridian",
        time_start=f"{today}T10:00:00Z",
        time_end=f"{today}T12:00:00Z",
    )
    assert [hit["ts"] for hit in hits] == [f"{today}T11:00:00Z"]
    assert search_activity_log(anima_dir, "Meridian", time_start="2099-01-01T00:00:00Z") == []


def test_activity_cache_tokenizes_only_appended_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    today = today_local().isoformat()
    first_content = "ZephyrNova initial activity"
    second_content = "Meridian appended activity"
    _write_activity_log(
        anima_dir,
        [{"ts": f"{today}T10:00:00", "type": "message_received", "content": first_content}],
        date_str=today,
    )
    search_activity_log(anima_dir, "ZephyrNova", days=3)

    tokenized: list[str] = []
    original = bm25_module.tokenize

    def tracked(text: str) -> list[str]:
        tokenized.append(text)
        return original(text)

    monkeypatch.setattr(bm25_module, "tokenize", tracked)
    path = anima_dir / "activity_log" / f"{today}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": f"{today}T10:01:00", "type": "message_received", "content": second_content}) + "\n")

    results = search_activity_log(anima_dir, "Meridian", days=3)

    assert any(second_content in result["content"] for result in results)
    assert second_content in tokenized
    assert first_content not in tokenized


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


def test_longterm_clean_validation_does_not_rechunk_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/memo.md", "# Memo\n\nZephyrNova launch review.")
    rebuild_longterm_bm25_index(anima_dir)
    calls = 0
    original = bm25_module._bm25_docs_for_file

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bm25_module, "_bm25_docs_for_file", counted)
    kwargs = {"memory_types": ("knowledge",), "top_k": 10}
    first = search_longterm_memory_bm25(anima_dir, "ZephyrNova", **kwargs)
    second = search_longterm_memory_bm25(anima_dir, "ZephyrNova", **kwargs)
    assert first == second
    assert calls == 0


def test_longterm_source_validation_checks_each_file_once_per_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.memory.rag.indexer import MemoryIndexer

    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/memo.md", "# Memo\n\nZephyrNova launch review.")
    rebuild_longterm_bm25_index(anima_dir)
    payload = json.loads(longterm_bm25_index_path(anima_dir).read_text(encoding="utf-8"))
    doc = payload["documents"][0]
    calls = 0
    original = MemoryIndexer.is_ragignored

    def counted(path: Path) -> bool:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(MemoryIndexer, "is_ragignored", counted)
    cache: dict = {}

    assert bm25_module._longterm_doc_matches_current_source(anima_dir, doc, cache)
    calls_after_first = calls
    assert bm25_module._longterm_doc_matches_current_source(anima_dir, doc, cache)
    assert calls == calls_after_first


def test_longterm_query_scoring_is_shared_across_scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    for memory_type in ("knowledge", "episodes", "procedures"):
        _write_longterm_memory(
            anima_dir,
            f"{memory_type}/meridian.md",
            f"# Meridian\n\n{memory_type} telemetry review.",
        )
    rebuild_longterm_bm25_index(anima_dir)
    bm25_module._LONGTERM_QUERY_CACHE.clear()
    original = bm25_module._longterm_bm25_scores
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bm25_module, "_longterm_bm25_scores", counted)
    hits = [
        search_longterm_memory_bm25(anima_dir, "Meridian", memory_types=(memory_type,), top_k=3)
        for memory_type in ("knowledge", "episodes", "procedures")
    ]

    assert calls == 1
    assert [scope_hits[0]["memory_type"] for scope_hits in hits] == ["knowledge", "episodes", "procedures"]


def test_longterm_bm25_excludes_archive_subtrees_without_ragignore(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/current.md", "# Current\n\nZephyrNova current knowledge.")
    _write_longterm_memory(anima_dir, "knowledge/archive/old.md", "# Old\n\nZephyrNova archived knowledge.")
    _write_longterm_memory(anima_dir, "episodes/current.md", "# Current\n\nZephyrNova current episode.")
    _write_longterm_memory(anima_dir, "episodes/archive/old.md", "# Old\n\nZephyrNova archived episode.")
    _write_longterm_memory(anima_dir, "procedures/current.md", "# Current\n\nZephyrNova current procedure.")
    _write_longterm_memory(anima_dir, "procedures/archive/old.md", "# Old\n\nZephyrNova archived procedure.")

    result = rebuild_longterm_bm25_index(anima_dir)
    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge", "episodes", "procedures"),
        top_k=10,
    )

    assert result.documents == 3
    assert {hit["source_file"] for hit in hits} == {
        "knowledge/current.md",
        "episodes/current.md",
        "procedures/current.md",
    }


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


def test_maybe_rebuild_skips_when_not_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")

    calls: list[int] = []
    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", lambda *a, **k: calls.append(1))

    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is False
    assert calls == []


def test_maybe_rebuild_runs_when_dirty(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    mark_longterm_bm25_dirty(anima_dir, reason="write")

    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is True
    # A successful rebuild clears the dirty marker and writes the index.
    assert is_longterm_bm25_dirty(anima_dir) is False
    assert longterm_bm25_index_path(anima_dir).is_file()
    assert longterm_bm25_rebuild_marker_path(anima_dir).is_file()


def test_maybe_rebuild_respects_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    mark_longterm_bm25_dirty(anima_dir, reason="write")

    # First call rebuilds and records the attempt time.
    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is True

    # A later write marks the index dirty again inside the cooldown window.
    mark_longterm_bm25_dirty(anima_dir, reason="write-2")
    calls: list[int] = []
    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", lambda *a, **k: calls.append(1))

    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is False
    assert calls == []
    # Still dirty: staleness is tolerated until the cooldown elapses.
    assert is_longterm_bm25_dirty(anima_dir) is True


def test_maybe_rebuild_runs_after_cooldown_elapses(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    mark_longterm_bm25_dirty(anima_dir, reason="write")
    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is True

    # Age the cooldown marker beyond the window and re-dirty the index.
    marker = longterm_bm25_rebuild_marker_path(anima_dir)
    old = time.time() - 700.0
    os.utime(marker, (old, old))
    mark_longterm_bm25_dirty(anima_dir, reason="write-2")

    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is True
    assert is_longterm_bm25_dirty(anima_dir) is False


def test_maybe_rebuild_failure_keeps_dirty_marker_and_cools_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    mark_longterm_bm25_dirty(anima_dir, reason="write")

    calls: list[int] = []

    def boom(*_a, **_k):
        calls.append(1)
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", boom)

    # Attempt runs but fails: dirty marker is retained for a later retry.
    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is True
    assert is_longterm_bm25_dirty(anima_dir) is True
    # The cooldown marker was written first, so an immediate retry is throttled.
    assert maybe_rebuild_dirty_longterm_bm25(anima_dir) is False
    assert calls == [1]


def test_longterm_bm25_missing_index_uses_live_delta_without_full_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("search path should not rebuild long-term BM25 by default")

    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", fail_rebuild)
    update_longterm_bm25_source(anima_dir, "knowledge/a.md")

    hits = search_longterm_memory_bm25(anima_dir, "ZephyrNova", memory_types=("knowledge",))
    assert hits[0]["source_file"] == "knowledge/a.md"


def test_longterm_bm25_rejects_index_doc_with_stale_source_signature(tmp_path: Path) -> None:
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
            "source_size": stat.st_size + 1,
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


def test_previous_schema_remains_searchable_until_background_rebuild(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    rebuild_longterm_bm25_index(anima_dir)
    index_path = longterm_bm25_index_path(anima_dir)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["schema_version"] = bm25_module.LONGTERM_BM25_SCHEMA_VERSION - 1
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        rebuild_if_missing=False,
    )

    assert hits[0]["source_file"] == "knowledge/a.md"
    assert bm25_module.is_longterm_bm25_dirty(anima_dir)

    _write_longterm_memory(anima_dir, "knowledge/new.md", "# New\n\nMeridian retention policy.")
    update_longterm_bm25_source(anima_dir, "knowledge/new.md")
    assert search_longterm_memory_bm25(anima_dir, "ZephyrNova", memory_types=("knowledge",))
    assert search_longterm_memory_bm25(anima_dir, "Meridian", memory_types=("knowledge",))


def test_longterm_bm25_can_skip_source_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    rebuild_longterm_bm25_index(anima_dir)

    def fail_validation(*args, **kwargs):
        raise AssertionError("source validation should be skipped")

    monkeypatch.setattr(bm25_module, "_longterm_doc_matches_current_source", fail_validation)

    hits = search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        validate_sources=False,
    )

    assert len(hits) == 1


@pytest.mark.parametrize("validate_sources", [False, True])
def test_longterm_bm25_source_sync_follows_validation_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validate_sources: bool,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nZephyrNova launchpad audit.")
    rebuild_longterm_bm25_index(anima_dir)
    sync_calls: list[Path] = []
    monkeypatch.setattr(
        bm25_module,
        "_sync_longterm_bm25_sources",
        lambda path, _payload: sync_calls.append(path),
    )

    search_longterm_memory_bm25(
        anima_dir,
        "ZephyrNova",
        memory_types=("knowledge",),
        validate_sources=validate_sources,
    )

    assert sync_calls == ([anima_dir] if validate_sources else [])


def test_longterm_source_delta_is_fresh_without_full_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    source = anima_dir / "knowledge" / "memo.md"
    _write_longterm_memory(anima_dir, "knowledge/memo.md", "# Memo\n\nZephyrNova launch review.")
    rebuild_longterm_bm25_index(anima_dir)

    source.write_text("# Memo\n\nMeridian launch review with changed content.", encoding="utf-8")
    rebuilds: list[Path] = []
    monkeypatch.setattr(bm25_module, "rebuild_longterm_bm25_index", lambda path: rebuilds.append(path))

    assert search_longterm_memory_bm25(anima_dir, "ZephyrNova", memory_types=("knowledge",)) == []
    hits = search_longterm_memory_bm25(anima_dir, "Meridian", memory_types=("knowledge",))
    assert hits[0]["source_file"] == "knowledge/memo.md"
    assert maybe_rebuild_dirty_longterm_bm25(anima_dir, cooldown_seconds=0) is False
    assert rebuilds == []
    assert longterm_bm25_delta_path(anima_dir).is_file()


def test_longterm_delta_add_and_tombstone(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/base.md", "# Base\n\nUnrelated launch notes.")
    rebuild_longterm_bm25_index(anima_dir)

    _write_longterm_memory(anima_dir, "knowledge/new.md", "# New\n\nWidget retention policy.")
    update_longterm_bm25_source(anima_dir, "knowledge/new.md")
    assert search_longterm_memory_bm25(anima_dir, "Widget", memory_types=("knowledge",))[0]["source_file"] == (
        "knowledge/new.md"
    )

    (anima_dir / "knowledge" / "new.md").unlink()
    update_longterm_bm25_source(anima_dir, "knowledge/new.md")
    assert search_longterm_memory_bm25(anima_dir, "Widget", memory_types=("knowledge",)) == []


def test_longterm_sync_rechunks_only_changed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/a.md", "# A\n\nAlpha baseline memory.")
    _write_longterm_memory(anima_dir, "knowledge/b.md", "# B\n\nBeta baseline memory.")
    rebuild_longterm_bm25_index(anima_dir)
    original = bm25_module._bm25_docs_for_file
    calls: list[str] = []

    def counted(root: Path, path: Path, memory_type: str):
        calls.append(str(path.relative_to(root)))
        return original(root, path, memory_type)

    monkeypatch.setattr(bm25_module, "_bm25_docs_for_file", counted)
    assert search_longterm_memory_bm25(anima_dir, "Alpha", memory_types=("knowledge",))
    assert calls == []

    (anima_dir / "knowledge" / "b.md").write_text("# B\n\nGamma changed memory.", encoding="utf-8")
    assert search_longterm_memory_bm25(anima_dir, "Gamma", memory_types=("knowledge",))
    assert calls == ["knowledge/b.md"]


def test_missing_index_does_not_build_quadratic_delta_on_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    for index in range(20):
        _write_longterm_memory(anima_dir, f"knowledge/{index}.md", f"# {index}\n\nMeridian memory {index}.")
    updates: list[str] = []
    monkeypatch.setattr(
        bm25_module,
        "update_longterm_bm25_source",
        lambda _root, source: updates.append(source),
    )

    assert (
        search_longterm_memory_bm25(
            anima_dir,
            "Meridian",
            memory_types=("knowledge",),
            rebuild_if_missing=False,
        )
        == []
    )
    assert updates == []


def test_concurrent_longterm_delta_updates_do_not_lose_sources(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    _write_longterm_memory(anima_dir, "knowledge/base.md", "# Base\n\nBaseline memory.")
    rebuild_longterm_bm25_index(anima_dir)
    sources = [f"knowledge/update-{index}.md" for index in range(8)]
    for index, source in enumerate(sources):
        _write_longterm_memory(anima_dir, source, f"# Update\n\nSignal{index} concurrent memory.")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda source: update_longterm_bm25_source(anima_dir, source), sources))

    payload = json.loads(longterm_bm25_delta_path(anima_dir).read_text(encoding="utf-8"))
    assert set(payload["sources"]) == set(sources)


class TestBM25RebuildSingleFlight:
    """R5: concurrent dirty rebuilds must not stack."""

    def _mark_dirty(self, anima_dir):
        from core.memory.bm25 import mark_longterm_bm25_dirty

        mark_longterm_bm25_dirty(anima_dir)

    def test_lock_held_skips_rebuild(self, tmp_path, monkeypatch):
        from core.memory import bm25 as bm25_mod
        from core.memory.bm25 import (
            longterm_bm25_rebuild_marker_path,
            maybe_rebuild_dirty_longterm_bm25,
        )

        self._mark_dirty(tmp_path)
        marker = longterm_bm25_rebuild_marker_path(tmp_path)
        lock = marker.with_name(marker.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("held", encoding="utf-8")  # fresh lock (mtime now)

        called = []
        monkeypatch.setattr(bm25_mod, "rebuild_longterm_bm25_index", lambda d: called.append(d))

        assert maybe_rebuild_dirty_longterm_bm25(tmp_path) is False
        assert called == []
        assert lock.exists()  # not stolen

    def test_stale_lock_is_stolen(self, tmp_path, monkeypatch):
        import os as _os
        import time as _time

        from core.memory import bm25 as bm25_mod
        from core.memory.bm25 import (
            longterm_bm25_rebuild_marker_path,
            maybe_rebuild_dirty_longterm_bm25,
        )

        self._mark_dirty(tmp_path)
        marker = longterm_bm25_rebuild_marker_path(tmp_path)
        lock = marker.with_name(marker.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("stale", encoding="utf-8")
        old = _time.time() - 3600
        _os.utime(lock, (old, old))

        called = []
        monkeypatch.setattr(bm25_mod, "rebuild_longterm_bm25_index", lambda d: called.append(d))

        assert maybe_rebuild_dirty_longterm_bm25(tmp_path) is True
        assert called == [tmp_path]
        assert not lock.exists()  # released after rebuild
