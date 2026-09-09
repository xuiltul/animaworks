from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel C: Related knowledge and important knowledge search."""

import asyncio
import json
import logging
import re
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.file_access_policy import load_denied_roots, memory_source_is_allowed
from core.memory.priming.constants import _BUDGET_IMPORTANT_KNOWLEDGE
from core.memory.priming.items import ItemizedMemory, MemoryItem, render_items, select_within_budget
from core.memory.priming.utils import build_queries, build_unified_searcher, normalize_trigger
from core.memory.retrieval.unified_search import UnifiedMemorySearch
from core.prompt.tokens import estimate_tokens

if TYPE_CHECKING:
    from core.memory.rag.retriever import MemoryRetriever

logger = logging.getLogger("animaworks.priming")


class KnowledgeSearchCache:
    """Single-flight search results owned by one prime_memories invocation.

    C0 and C run in separate worker threads but can request the exact same
    search. Never retain this cache on an engine or across conversations.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.results: dict[tuple, tuple[UnifiedMemorySearch, list[dict]]] = {}


def _single_line(text: str, limit: int = 160) -> str:
    """Collapse prompt cue text to one bounded line."""
    collapsed = " ".join(str(text or "").split())
    return collapsed[:limit]


def _quote_path(path: str) -> str:
    """Return a JSON string literal for read_memory_file path examples."""
    return json.dumps(path, ensure_ascii=False)


def extract_summary(content: str, metadata: dict) -> tuple[str, str]:
    """Extract title and body summary from an [IMPORTANT] search result.

    Returns:
        (title, body_summary) where body_summary is the first meaningful
        line after the H1 heading, truncated to 100 chars. Empty string
        if no body is available.
    """
    title = ""
    body = ""

    fm_summary = metadata.get("summary")
    if fm_summary:
        return (str(fm_summary).strip(), "")

    match = re.search(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        after_h1 = content[match.end() :].lstrip("\n")
        for line in after_h1.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                body = stripped[:100]
                break
    else:
        source = metadata.get("source_file", "")
        if source:
            title = Path(source).stem.replace("-", " ").replace("_", " ")

    return (title, body)


def _usable_summary_body(body: str) -> str:
    """Reject structural Markdown lines that do not summarize a document."""
    stripped = body.strip()
    if stripped.startswith("|") or re.fullmatch(r"[-|:\s]+", stripped):
        return ""
    if re.fullmatch(r"#+", stripped):
        return ""
    return stripped


def _timestamp_rank(value: str) -> float:
    """Convert an ISO timestamp into a sortable numeric rank."""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _path_from_doc_id(doc_id: str, memory_type: str = "knowledge") -> str:
    """Best-effort conversion from retriever doc_id to read_memory_file path."""
    if not doc_id:
        return ""
    doc_path = str(doc_id).split("#", 1)[0]
    for marker in ("companies/", "common_knowledge/", "knowledge/", "procedures/", "episodes/"):
        if marker in doc_path:
            return marker + doc_path.split(marker, 1)[1]
    marker = f"{memory_type}/"
    if marker in doc_path:
        return marker + doc_path.split(marker, 1)[1]
    if doc_path.endswith(".md"):
        return f"{memory_type}/{Path(doc_path).name}"
    return ""


def to_read_memory_path(metadata: dict, anima_name: str, doc_id: str = "") -> str:
    """Convert chunk metadata to read_memory_file path."""
    source = metadata.get("source_file", "") or _path_from_doc_id(doc_id)
    if not source:
        return ""
    if metadata.get("anima") == "shared":
        # companies/ paths are directly readable via read_memory_file; prefixing
        # common_knowledge/ would make them unresolvable
        if source.startswith(("common_knowledge/", "companies/")):
            return source
        return f"common_knowledge/{source}"
    return source


def format_pointer_result(
    *,
    content: str,
    metadata: dict,
    path: str,
) -> str:
    """Format a retrieval result as a pointer cue instead of raw payload."""
    title, _ = extract_summary(content, metadata)
    summary = title or Path(path).stem.replace("-", " ").replace("_", " ")
    summary = _single_line(summary)
    return f"📌 {summary} → read_memory_file(path={_quote_path(path)})"


def _is_action_rule(path: str, content: str) -> bool:
    """Return whether a chunk belongs to the separately primed action gate."""
    return Path(path).name.startswith("action-rule-") or "[ACTION-RULE]" in content


def _updated_from_metadata(metadata: dict) -> str:
    return str(metadata.get("updated_at") or metadata.get("updated") or metadata.get("created_at") or "")


def _item_from_chunk(
    *,
    content: str,
    metadata: dict,
    path: str,
    rank: float,
) -> MemoryItem:
    title, _ = extract_summary(content, metadata)
    title = _single_line(title or Path(path).stem.replace("-", " ").replace("_", " "))
    return MemoryItem(
        source="important_knowledge",
        key=path,
        text=f"📌 {title} → read_memory_file(path={_quote_path(path)})",
        ref=path,
        updated=_updated_from_metadata(metadata),
        rank=rank,
    )


def _always_prime_chunks(retriever: MemoryRetriever, anima_name: str) -> list:
    """Fetch opt-in resident chunks without reusing the all-IMPORTANT query."""
    vector_store = getattr(retriever, "vector_store", None)
    if vector_store is None:
        return []

    results = list(
        vector_store.get_by_metadata(
            f"{anima_name}_knowledge",
            {"always_prime": True},
            limit=20,
        )
    )
    results.extend(
        vector_store.get_by_metadata(
            "shared_common_knowledge",
            {"always_prime": True},
            limit=20,
        )
    )
    return results


def _static_c0_chunks(
    get_retriever: Callable[[], MemoryRetriever | None],
    anima_name: str,
    *,
    include_fallback: bool,
    anima_dir: Path,
    queries: list[str],
    trigger: str,
    min_score: float,
    resident_only: bool = False,
    search_cache: KnowledgeSearchCache | None = None,
) -> tuple[list, list[dict]]:
    """Load all C0 sources in one worker-thread transaction."""
    retriever = get_retriever()
    if retriever is None:
        return [], []
    always = _always_prime_chunks(retriever, anima_name)
    if resident_only:
        return always, []
    if not include_fallback:
        searcher, relevant = _search_related_knowledge(
            anima_dir,
            get_retriever,
            queries,
            trigger=normalize_trigger(trigger),
            min_score=min_score,
            search_cache=search_cache,
        )
        if bool(searcher.last_search_meta.get("abstain", False)):
            relevant = []
        return always, relevant

    fallback = retriever.get_important_chunks(anima_name, include_shared=True)
    relevant = [
        {
            "doc_id": str(getattr(r.document, "id", "") or getattr(r, "doc_id", "") or ""),
            "content": r.document.content,
            "score": float(getattr(r, "score", 0.0) or 0.0),
            **r.document.metadata,
            "importance": r.document.metadata.get("importance", "important"),
        }
        for r in fallback
    ]
    return always, relevant


def _unknown_origin_is_internal(anima_dir: Path, path: str) -> bool:
    """Treat only clearly owned, legacy origin-less knowledge as medium trust."""
    parts = Path(path).parts
    if parts and parts[0] in {"knowledge", "procedures"}:
        return True
    if len(parts) < 3 or parts[0] != "companies" or parts[2] not in {"knowledge", "procedures"}:
        return False
    try:
        from core.company import get_company

        company = get_company(anima_dir.name, animas_dir=anima_dir.parent)
    except Exception:
        logger.debug("Channel C: failed to resolve company membership", exc_info=True)
        return False
    return bool(company and parts[1] == company)


def _build_unified_searcher(
    anima_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
) -> UnifiedMemorySearch:
    """Build UnifiedMemorySearch, preserving injected retriever state for tests/runtime caches.

    Thin wrapper over the shared helper that binds this module's ``UnifiedMemorySearch``
    reference so test-time patches on ``channel_c.UnifiedMemorySearch`` keep working.
    """
    return build_unified_searcher(anima_dir, get_retriever, UnifiedMemorySearch)


def _search_related_knowledge(
    anima_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
    queries: list[str],
    *,
    trigger: str,
    min_score: float,
    search_cache: KnowledgeSearchCache | None = None,
) -> tuple[UnifiedMemorySearch, list[dict]]:
    """Build and execute Channel C search without blocking the event loop."""
    if search_cache is not None:
        key = (str(anima_dir), tuple(queries), normalize_trigger(trigger), min_score)
        with search_cache.lock:
            if key not in search_cache.results:
                search_cache.results[key] = _search_related_knowledge(
                    anima_dir,
                    get_retriever,
                    queries,
                    trigger=trigger,
                    min_score=min_score,
                )
            return search_cache.results[key]
    searcher = _build_unified_searcher(anima_dir, get_retriever)
    results = searcher.search_many(
        queries,
        scope="common_knowledge",
        limit=5,
        trigger=normalize_trigger(trigger),
        min_score=min_score,
    )
    return searcher, results


async def channel_c0_important_knowledge(
    anima_dir: Path,
    knowledge_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
    queries: list[str] | None = None,
    trigger: str = "chat",
    resident_only: bool = False,
    search_cache: KnowledgeSearchCache | None = None,
) -> str:
    """Channel C0: opt-in resident and query-relevant important pointers."""
    if not knowledge_dir.is_dir():
        return ""
    try:
        denied_roots = load_denied_roots(anima_dir)
        anima_name = anima_dir.name
        # Explicit residency is opt-in and bounded; [IMPORTANT] by itself only
        # protects retention and must not inject unrelated recent knowledge.
        effective_queries = [query for query in (queries or []) if str(query).strip()]
        _min_score: float | None = None
        try:
            from core.config.models import load_config as _load_cfg

            _min_score = _load_cfg().rag.min_retrieval_score
        except Exception:
            logger.debug("Failed to load rag.min_retrieval_score from config, using default")
        always_results, relevant_rows = await asyncio.to_thread(
            _static_c0_chunks,
            get_retriever,
            anima_name,
            include_fallback=not effective_queries,
            anima_dir=anima_dir,
            queries=effective_queries,
            trigger=trigger,
            min_score=float(_min_score) if _min_score is not None else 0.0,
            resident_only=resident_only,
            search_cache=search_cache,
        )
        newest_always_by_path: dict[str, tuple[MemoryItem, float]] = {}
        for r in always_results:
            meta = r.document.metadata
            doc_id = str(getattr(r.document, "id", "") or getattr(r, "doc_id", "") or "")
            rel_path = to_read_memory_path(meta, anima_name, doc_id)
            if not rel_path or not memory_source_is_allowed(anima_dir, rel_path, denied_roots):
                continue
            content = r.document.content
            if _is_action_rule(rel_path, content):
                continue
            timestamp = _timestamp_rank(_updated_from_metadata(meta))
            item = _item_from_chunk(content=content, metadata=meta, path=rel_path, rank=timestamp)
            previous = newest_always_by_path.get(rel_path)
            if previous is None or timestamp > previous[1]:
                newest_always_by_path[rel_path] = (item, timestamp)

        always_items = [
            pair[0] for pair in sorted(newest_always_by_path.values(), key=lambda pair: pair[1], reverse=True)[:3]
        ]

        # Background turns can have no usable query; only then the worker
        # transaction above supplies the legacy bounded fallback.

        relevant_by_path: dict[str, tuple[MemoryItem, float]] = {}
        always_paths = {item.key for item in always_items}
        for row in relevant_rows:
            metadata = _metadata_from_unified_result(row)
            if metadata.get("importance") != "important":
                continue
            content = str(row.get("content", "") or "")
            rel_path = to_read_memory_path(metadata, anima_name, str(row.get("doc_id", "") or ""))
            if (
                not rel_path
                or rel_path in always_paths
                or _is_action_rule(rel_path, content)
                or not memory_source_is_allowed(anima_dir, rel_path, denied_roots)
            ):
                continue
            score = float(row.get("score", 0.0) or 0.0)
            item = _item_from_chunk(content=content, metadata=metadata, path=rel_path, rank=score)
            previous = relevant_by_path.get(rel_path)
            if previous is None or score > previous[1]:
                relevant_by_path[rel_path] = (item, score)

        relevant_items = [
            pair[0]
            for pair in sorted(
                relevant_by_path.values(),
                key=lambda pair: (pair[1], _timestamp_rank(pair[0].updated)),
                reverse=True,
            )[:3]
        ]
        ordered_items = always_items + relevant_items
        # Stable synthetic ranks preserve each category's required ordering when
        # the engine reapplies its item budget later in the pipeline.
        ranked_items = [
            _replace_item_rank(item, len(ordered_items) - index) for index, item in enumerate(ordered_items)
        ]

        header = "### [IMPORTANT] Knowledge (summary pointers)"
        available = _BUDGET_IMPORTANT_KNOWLEDGE - estimate_tokens(header)
        if available <= 0:
            return ""
        selected = select_within_budget(ranked_items, available)
        while selected and estimate_tokens(render_items(selected, header)) > _BUDGET_IMPORTANT_KNOWLEDGE:
            selected.pop()
        if not selected:
            return ""
        return ItemizedMemory(render_items(selected, header), selected)
    except Exception as e:
        logger.debug("Channel C0: get_important_chunks failed: %s", e)
        return ""


def _replace_item_rank(item: MemoryItem, rank: float) -> MemoryItem:
    """Return a MemoryItem with a pipeline-stable C0 ordering rank."""
    return MemoryItem(
        source=item.source,
        key=item.key,
        text=item.text,
        ref=item.ref,
        updated=item.updated,
        rank=rank,
    )


async def channel_c_related_knowledge(
    anima_dir: Path,
    knowledge_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
    keywords: list[str],
    message: str = "",
    recent_human_messages: list[str] | None = None,
    trigger: str = "chat",
    search_cache: KnowledgeSearchCache | None = None,
) -> tuple[ItemizedMemory, ItemizedMemory]:
    """Channel C: Related knowledge search through unified Legacy retrieval.

    Searches both personal knowledge and shared common_knowledge,
    merging results by score.

    ``trigger`` selects the retrieval policy (rerank/pool/scopes); it is
    normalized to a ``TRIGGER_POLICIES`` key before use.

    Returns a ``(medium, untrusted)`` tuple whose string-compatible values
    retain one indivisible item per readable source path.
    """
    if not knowledge_dir.is_dir():
        logger.debug("Channel C: No knowledge dir")
        return (ItemizedMemory(""), ItemizedMemory(""))

    try:
        denied_roots = load_denied_roots(anima_dir)
        queries = build_queries(message, keywords, recent_human_messages)
        if not queries:
            logger.debug("Channel C: No keywords and no message")
            return (ItemizedMemory(""), ItemizedMemory(""))
        anima_name = anima_dir.name

        _min_score: float | None = None
        try:
            from core.config.models import load_config as _load_cfg

            _min_score = _load_cfg().rag.min_retrieval_score
        except Exception:
            logger.debug("Failed to load rag.min_retrieval_score from config, using default")

        searcher, results = await asyncio.to_thread(
            _search_related_knowledge,
            anima_dir,
            get_retriever,
            queries,
            trigger=normalize_trigger(trigger),
            min_score=float(_min_score) if _min_score is not None else 0.0,
            search_cache=search_cache,
        )
        if bool(searcher.last_search_meta.get("abstain", False)):
            logger.debug("Channel C: unified search abstained")
            return (ItemizedMemory(""), ItemizedMemory(""))

        if results:
            from core.execution._sanitize import ORIGIN_UNKNOWN, resolve_trust

            medium_by_path: dict[str, MemoryItem] = {}
            untrusted_by_path: dict[str, MemoryItem] = {}
            for result in results:
                metadata = _metadata_from_unified_result(result)
                chunk_origin = metadata.get("origin", "")
                chunk_trust = resolve_trust(chunk_origin or ORIGIN_UNKNOWN)
                rel_path = to_read_memory_path(metadata, anima_name, str(result.get("doc_id", "")))
                if not rel_path:
                    logger.debug("Channel C: skipping result without readable path: %s", result.get("doc_id", ""))
                    continue
                if not memory_source_is_allowed(anima_dir, rel_path, denied_roots):
                    logger.debug("Channel C: skipping result from denied or ambiguous source: %s", rel_path)
                    continue
                line = format_pointer_result(
                    content=str(result.get("content", "") or ""),
                    metadata=metadata,
                    path=rel_path,
                )
                item_kwargs = {
                    "key": rel_path,
                    "text": line,
                    "ref": rel_path,
                    "updated": _updated_from_metadata(metadata),
                    "rank": float(result.get("score", 0.0) or 0.0),
                }
                # Old internal files predate origin metadata. Elevate only paths
                # whose ownership is unambiguous; the global sanitizer default
                # remains conservative for every other origin-less payload.
                if chunk_trust == "untrusted" and not chunk_origin and _unknown_origin_is_internal(anima_dir, rel_path):
                    target = medium_by_path
                    source = "related_knowledge"
                elif chunk_trust == "untrusted":
                    target = untrusted_by_path
                    source = "related_knowledge_untrusted"
                else:
                    target = medium_by_path
                    source = "related_knowledge"
                item = MemoryItem(source=source, **item_kwargs)
                previous = target.get(rel_path)
                if previous is None or (item.rank, item.updated) > (previous.rank, previous.updated):
                    target[rel_path] = item

            medium_items = tuple(
                sorted(medium_by_path.values(), key=lambda item: (item.rank, item.updated), reverse=True)
            )
            untrusted_items = tuple(
                sorted(untrusted_by_path.values(), key=lambda item: (item.rank, item.updated), reverse=True)
            )
            medium_output = ItemizedMemory(render_items(medium_items, ""), medium_items)
            untrusted_output = ItemizedMemory(render_items(untrusted_items, ""), untrusted_items)

            logger.debug(
                "Channel C: Vector search returned %d results (medium=%d, untrusted=%d)",
                len(results),
                len(medium_items),
                len(untrusted_items),
            )
            return (medium_output, untrusted_output)
        else:
            logger.debug("Channel C: Vector search found no results")
            return (ItemizedMemory(""), ItemizedMemory(""))

    except Exception as e:
        logger.warning("Channel C: Vector search failed: %s", e)
        return (ItemizedMemory(""), ItemizedMemory(""))


def _metadata_from_unified_result(result: dict) -> dict:
    metadata = {
        key: value
        for key, value in result.items()
        if key not in ("content", "score") and isinstance(value, (str, int, float, bool, list))
    }
    if "source_file" not in metadata and result.get("source"):
        metadata["source_file"] = result["source"]
    return metadata
