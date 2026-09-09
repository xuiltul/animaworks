from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Priming utilities: retriever cache, dual-query, search, keyword extraction."""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.memory.priming.constants import _MAX_KEYWORD_INPUT_LEN, _MINIMAL_STOPWORDS, _RE_UNICODE_WORDS
from core.prompt.tokens import truncate_to_tokens

if TYPE_CHECKING:
    from core.memory.rag.retriever import MemoryRetriever

logger = logging.getLogger("animaworks.priming")

# TTL (seconds) before a failed lazy initialization is retried.  Prevents a
# transient failure from permanently degrading priming until process restart.
_INIT_RETRY_TTL_SECONDS = 300.0


# ── RetrieverCache ────────────────────────────────────────────────


class RetrieverCache:
    """Holds retriever instance and initialization state for lazy creation."""

    def __init__(self) -> None:
        self._retriever: MemoryRetriever | None = None
        self._lock = threading.Lock()
        # Monotonic timestamp of the last failed init; ``None`` means never failed.
        self._failed_at: float | None = None

    def get_or_create(self, anima_dir: Path, knowledge_dir: Path) -> MemoryRetriever | None:
        """Return a MemoryRetriever instance, creating one lazily if needed.

        Shared for Channel C (related knowledge).  Returns ``None`` when RAG dependencies are unavailable
        or the knowledge directory does not exist.

        A failed initialization is latched for ``_INIT_RETRY_TTL_SECONDS`` and then
        retried once, so a transient outage does not silently disable priming forever.
        """
        if self._retriever is not None:
            return self._retriever

        with self._lock:
            return self._get_or_create_locked(anima_dir, knowledge_dir)

    def _get_or_create_locked(self, anima_dir: Path, knowledge_dir: Path) -> MemoryRetriever | None:
        if self._retriever is not None:
            return self._retriever

        if self._failed_at is not None and (time.monotonic() - self._failed_at) < _INIT_RETRY_TTL_SECONDS:
            return None

        first_failure = self._failed_at is None

        if not knowledge_dir.is_dir():
            self._failed_at = time.monotonic()
            return None

        try:
            from core.memory.rag import MemoryRetriever
            from core.memory.rag.indexer import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            anima_name = anima_dir.name
            vector_store = get_vector_store(anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, retriever disabled")
                self._failed_at = time.monotonic()
                return None
            indexer = MemoryIndexer(vector_store, anima_name, anima_dir)
            self._retriever = MemoryRetriever(
                vector_store,
                indexer,
                knowledge_dir,
            )
            self._failed_at = None
            logger.debug("Shared MemoryRetriever initialized for %s", anima_name)
        except ImportError:
            logger.debug("RAG dependencies not installed, retriever unavailable")
            self._failed_at = time.monotonic()
        except Exception as e:
            if first_failure:
                logger.warning("Failed to initialize MemoryRetriever: %s", e)
            else:
                logger.debug("Failed to initialize MemoryRetriever (retry): %s", e)
            self._failed_at = time.monotonic()

        return self._retriever


# ── Trigger normalization ─────────────────────────────────────────


def normalize_trigger(trigger: str) -> str:
    """Normalize a priming trigger/channel to a ``TRIGGER_POLICIES`` key.

    Splits ``"inbox:sender"``-style triggers on ``:`` and lowercases so the
    downstream policy lookup is deterministic.  Unknown keys fall back to
    ``"chat"`` (matching ``UnifiedMemorySearch._policy_for``).
    """
    from core.memory.retrieval.unified_search import TRIGGER_POLICIES

    key = str(trigger or "chat").split(":", 1)[0].strip().lower()
    return key if key in TRIGGER_POLICIES else "chat"


# ── Unified searcher builder (shared by Channel C and Channel F) ──


def build_unified_searcher(
    anima_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
    unified_search_cls: Any,
) -> Any:
    """Build a UnifiedMemorySearch, reusing an injected retriever and indexer when available.

    Shared by Channel C and Channel F so both reuse the runtime/test retriever
    cache instead of constructing a cold searcher on every priming run.  The
    concrete ``UnifiedMemorySearch`` class is passed in by the caller so module
    level patches in tests continue to take effect.
    """
    try:
        retriever = get_retriever()
        indexer = getattr(retriever, "indexer", None) if retriever is not None else None
        if indexer is not None:
            from core.memory.rag_search import RAGMemorySearch
            from core.paths import get_common_knowledge_dir, get_common_skills_dir

            rag_search = RAGMemorySearch(
                anima_dir,
                get_common_knowledge_dir(),
                get_common_skills_dir(),
            )
            rag_search._indexer = indexer
            rag_search._retriever = retriever
            rag_search._indexer_initialized = True
            return unified_search_cls(anima_dir, rag_search=rag_search)
    except Exception:
        logger.debug("Priming: failed to reuse injected retriever", exc_info=True)
    return unified_search_cls(anima_dir)


# ── Dual-query helpers ──────────────────────────────────────────


def _build_context_query(recent_msgs: list[str], max_chars: int = 500) -> str:
    """Build a context query from recent messages (newest-first, max_chars cutoff)."""
    parts: list[str] = []
    total = 0
    for msg in recent_msgs:
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(msg) <= remaining:
            parts.append(msg)
            total += len(msg)
        else:
            parts.append(msg[:remaining])
            total += remaining
            break
    return "\n".join(parts)


def build_queries(
    message: str,
    keywords: list[str],
    recent_human_messages: list[str] | None = None,
) -> list[str]:
    """Build 1–3 queries: message, keywords, and optional conversation context."""
    queries: list[str] = []
    msg_query = message[:300].strip() if message else ""
    kw_query = " ".join(keywords[:5]).strip() if keywords else ""
    if msg_query:
        queries.append(msg_query)
    if kw_query and kw_query != msg_query:
        queries.append(kw_query)
    if recent_human_messages:
        ctx = _build_context_query(recent_human_messages, max_chars=500)
        if ctx and ctx != msg_query:
            queries.append(ctx)
    return queries


def build_dual_queries(message: str, keywords: list[str]) -> list[str]:
    """Backward-compatible alias for build_queries."""
    return build_queries(message, keywords)


def search_and_merge(
    retriever: MemoryRetriever,
    queries: list[str],
    anima_name: str,
    *,
    memory_type: str,
    top_k: int,
    include_shared: bool = False,
    min_score: float | None = None,
) -> list:
    """Execute multiple queries and merge by max-score deduplication."""
    best: dict[str, object] = {}

    for query in queries:
        results = retriever.search(
            query=query,
            anima_name=anima_name,
            memory_type=memory_type,
            top_k=top_k,
            include_shared=include_shared,
            min_score=min_score,
        )
        for r in results:
            existing = best.get(r.doc_id)
            if existing is None or r.score > existing.score:  # type: ignore[union-attr]
                best[r.doc_id] = r

    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]  # type: ignore[union-attr]


# ── Keyword extraction ───────────────────────────────────────────


def extract_keywords(message: str, knowledge_dir: Path) -> list[str]:
    """Language-agnostic keyword extraction.

    1. Known entity matching (knowledge/ filenames — top priority)
    2. Unicode-aware tokenization
    3. Character-category min-length filter + minimal stopwords
    4. Length-descending sort (longer = more specific)

    Input is truncated to ``_MAX_KEYWORD_INPUT_LEN`` to bound regex cost.
    """
    text = message[:_MAX_KEYWORD_INPUT_LEN] if len(message) > _MAX_KEYWORD_INPUT_LEN else message

    known_entities: set[str] = set()
    if knowledge_dir.is_dir():
        known_entities = {f.stem.lower() for f in knowledge_dir.glob("*.md")}

    tokens = _RE_UNICODE_WORDS.findall(text)

    filtered = [t for t in tokens if meets_min_length(t) and t.lower() not in _MINIMAL_STOPWORDS]

    entity_matches = [t for t in filtered if t.lower() in known_entities]
    entity_set = {t.lower() for t in entity_matches}

    general = [t for t in filtered if t.lower() not in entity_set]
    general.sort(key=len, reverse=True)

    seen: set[str] = set()
    combined: list[str] = []
    for w in entity_matches + general:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            combined.append(w)

    return combined[:10]


def meets_min_length(token: str) -> bool:
    """Check minimum length based on Unicode character category.

    CJK characters carry meaning even as a single character (e.g. 裏, 金, 型).
    Latin and other scripts need at least 3 characters to be meaningful.
    """
    for c in token:
        cp = ord(c)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3040 <= cp <= 0x309F
            or 0x30A0 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
            or 0x0E00 <= cp <= 0x0E7F
        ):
            return len(token) >= 1
    return len(token) >= 3


# ── Truncation helpers ──────────────────────────────────────────


def truncate_head(text: str, max_tokens: int) -> str:
    """Truncate text keeping the head (front), cutting from the tail.

    Suitable for sender profiles (basic info at the top) and
    ripgrep results (best matches first).
    """
    return truncate_to_tokens(text, max_tokens, keep="head")


def truncate_tail(text: str, max_tokens: int) -> str:
    """Truncate text keeping the tail (end), cutting from the head.

    Suitable for recent episodes where newest entries are most relevant.
    """
    return truncate_to_tokens(text, max_tokens, keep="tail")
