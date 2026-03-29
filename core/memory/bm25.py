from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""BM25 keyword search over activity_log JSONL files.

Indexes recent activity entries and ranks them against a query using
``rank_bm25.BM25Okapi`` when available, with a token-overlap fallback.
"""

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from core.time_utils import today_local

logger = logging.getLogger("animaworks.memory")

try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# ── Constants ───────────────────────────────────────────────

_SEARCHABLE_TYPES: frozenset[str] = frozenset(
    {
        "tool_result",
        "message_received",
        "response_sent",
        "message_sent",
        "human_notify",
    }
)

_EXCLUDED_TOOL_PREFIXES: tuple[str, ...] = ("mcp__aw__",)

_EXCLUDED_TOOLS: frozenset[str] = frozenset(
    {
        "ToolSearch",
        "read_memory_file",
        "search_memory",
        "skill",
        "write_memory_file",
        "post_channel",
        "send_message",
        "call_human",
        "update_task",
        "archive_memory_file",
    }
)

_MIN_CONTENT_LENGTH = 100

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "this",
        "that",
        "with",
        "from",
        "have",
        "has",
        "was",
        "were",
        "will",
        "been",
        "not",
        "but",
        "they",
        "their",
        "what",
        "which",
        "when",
        "where",
        "who",
        "how",
        "can",
        "all",
        "each",
        "its",
        "than",
        "other",
        "into",
        "could",
        "your",
        "about",
        "would",
        "there",
        "these",
        "some",
        "them",
        "then",
        "also",
    }
)

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
    (0x0E00, 0x0E7F),
)

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

# ── Tokenizer ───────────────────────────────────────────────


def _char_in_cjk_ranges(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _token_is_cjk_class(tok: str) -> bool:
    return bool(tok) and all(_char_in_cjk_ranges(c) for c in tok)


def tokenize(text: str) -> list[str]:
    """Split *text* into filtered lowercase tokens for BM25 indexing."""
    out: list[str] = []
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        t = raw.lower()
        if t in _STOPWORDS:
            continue
        if _token_is_cjk_class(t) or len(t) >= 3:
            out.append(t)
    return out


# ── Activity log loading & filtering ────────────────────────


def _entry_tool_name(entry: dict[str, Any]) -> str:
    tool = entry.get("tool") or ""
    meta = entry.get("meta")
    if isinstance(meta, dict):
        tool = tool or meta.get("tool_name") or ""
    return str(tool) if tool else ""


def _should_index_entry(entry: dict[str, Any]) -> bool:
    etype = entry.get("type")
    if etype not in _SEARCHABLE_TYPES:
        return False
    if etype == "tool_result":
        tool = _entry_tool_name(entry)
        if tool in _EXCLUDED_TOOLS:
            return False
        if any(tool.startswith(p) for p in _EXCLUDED_TOOL_PREFIXES):
            return False
        content = entry.get("content") or ""
        if len(content) < _MIN_CONTENT_LENGTH:
            return False
    return True


def _activity_log_dates(days: int) -> list[date]:
    today = today_local()
    return [today - timedelta(days=i) for i in range(days)]


def _load_activity_entries(anima_dir: Path, days: int) -> list[tuple[str, dict[str, Any]]]:
    base = anima_dir / "activity_log"
    rows: list[tuple[str, dict[str, Any]]] = []
    for d in _activity_log_dates(days):
        date_str = d.isoformat()
        path = base / f"{date_str}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append((date_str, obj))
    return rows


def _fallback_scores(corpus_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    if not query_tokens:
        return [0.0] * len(corpus_tokens)
    qset = set(query_tokens)
    scores: list[float] = []
    for doc_tokens in corpus_tokens:
        if not doc_tokens:
            scores.append(0.0)
            continue
        doc_set = set(doc_tokens)
        matched = len(qset & doc_set)
        scores.append(matched / max(1, len(doc_tokens)))
    return scores


def _bm25_scores(corpus_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    if _HAS_BM25:
        bm25 = BM25Okapi(corpus_tokens)
        raw = bm25.get_scores(query_tokens)
        return [float(x) for x in raw]
    return _fallback_scores(corpus_tokens, query_tokens)


# ── Public API ──────────────────────────────────────────────


def search_activity_log(
    anima_dir: Path,
    query: str,
    *,
    days: int = 3,
    top_k: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """BM25 search over recent ``activity_log`` JSONL entries."""
    try:
        if not query.strip():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        rows = _load_activity_entries(anima_dir, days)
        corpus_tokens: list[list[str]] = []
        kept: list[tuple[str, dict[str, Any]]] = []
        for date_str, entry in rows:
            if not _should_index_entry(entry):
                continue
            content = entry.get("content") or ""
            doc_tokens = tokenize(content)
            if not doc_tokens:
                continue
            corpus_tokens.append(doc_tokens)
            kept.append((date_str, entry))

        if not corpus_tokens:
            return []

        scores = _bm25_scores(corpus_tokens, query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        window = order[offset : offset + top_k]

        search_method = "bm25" if _HAS_BM25 else "keyword_fallback"
        results: list[dict[str, Any]] = []
        for i in window:
            date_str, entry = kept[i]
            entry_content = entry.get("content") or ""
            etype = entry.get("type")
            entry_type = str(etype) if etype is not None else ""
            results.append(
                {
                    "source_file": f"activity_log/{date_str}.jsonl",
                    "content": entry_content[:2000],
                    "score": scores[i],
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "memory_type": "activity_log",
                    "search_method": search_method,
                    "ts": entry.get("ts"),
                    "tool": _entry_tool_name(entry),
                    "entry_type": entry_type,
                }
            )
        return results
    except Exception as exc:
        logger.debug("search_activity_log failed: %s", exc, exc_info=True)
        return []


def reciprocal_rank_fusion(*ranked_lists: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """Merge ranked result lists with reciprocal rank fusion (RRF)."""
    scores: dict[str, float] = {}
    first_row: dict[str, dict[str, Any]] = {}

    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            sf = str(item.get("source_file", ""))
            ts = item.get("ts")
            doc_id = f"{sf}\x00{ts}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_row:
                first_row[doc_id] = dict(item)

    merged: list[dict[str, Any]] = []
    for doc_id, rrf in scores.items():
        row = dict(first_row[doc_id])
        row["score"] = rrf
        row["search_method"] = "rrf"
        merged.append(row)

    merged.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return merged
