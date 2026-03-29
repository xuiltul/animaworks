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
    reciprocal_rank_fusion,
    search_activity_log,
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
    text = "Gmail受信メール from nogizaka@tatemono.co.jp"
    result = tokenize(text)
    joined = " ".join(result)
    assert "gmail" in joined.lower()
    assert any(any(ord(ch) > 127 for ch in tok) for tok in result)
    assert "nogizaka" in result or "tatemono" in result


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
