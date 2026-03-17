from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel C: Related knowledge and important knowledge search."""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from core.memory.priming.constants import _BUDGET_IMPORTANT_KNOWLEDGE, _CHARS_PER_TOKEN
from core.memory.priming.utils import build_dual_queries, search_and_merge

if TYPE_CHECKING:
    from core.memory.rag.retriever import MemoryRetriever

logger = logging.getLogger("animaworks.priming")


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

    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
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


def to_read_memory_path(metadata: dict, anima_name: str) -> str:
    """Convert chunk metadata to read_memory_file path."""
    source = metadata.get("source_file", "")
    if not source:
        return ""
    if metadata.get("anima") == "shared":
        return f"common_knowledge/{source}" if not source.startswith("common_knowledge/") else source
    return source


async def channel_c0_important_knowledge(
    anima_dir: Path,
    knowledge_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
) -> str:
    """Channel C0: Always-prime [IMPORTANT] chunks (summary pointers only)."""
    if not knowledge_dir.is_dir():
        return ""
    try:
        retriever = get_retriever()
        if retriever is None:
            return ""
        anima_name = anima_dir.name
        results = retriever.get_important_chunks(anima_name, include_shared=True)
        if not results:
            return ""
        budget_chars = _BUDGET_IMPORTANT_KNOWLEDGE * _CHARS_PER_TOKEN
        lines: list[tuple[int, str]] = []
        for r in results:
            content = r.document.content
            meta = r.document.metadata
            title, body = extract_summary(content, meta)
            rel_path = to_read_memory_path(meta, anima_name)
            if not rel_path:
                continue
            if body:
                line = f'📌 {title} — {body}\n  → read_memory_file(path="{rel_path}")'
            else:
                line = f'📌 {title} → read_memory_file(path="{rel_path}")'
            lines.append((len(line), line))
        lines.sort(key=lambda x: x[0])
        out: list[str] = []
        used = 0
        header = "### [IMPORTANT] Knowledge (summary pointers)"
        header_len = len(header) + 1
        if header_len > budget_chars:
            return ""
        out.append(header)
        used += header_len
        for _, line in lines:
            if used + len(line) + 1 > budget_chars:
                break
            out.append(line)
            used += len(line) + 1
        if len(out) <= 1:
            return ""
        return "\n".join(out)
    except Exception as e:
        logger.debug("Channel C0: get_important_chunks failed: %s", e)
        return ""


async def channel_c_related_knowledge(
    anima_dir: Path,
    knowledge_dir: Path,
    get_retriever: Callable[[], MemoryRetriever | None],
    keywords: list[str],
    restrict_to: list[str] | None = None,
    message: str = "",
) -> tuple[str, str]:
    """Channel C: Related knowledge search (vector search).

    Uses dense vector retrieval via MemoryRetriever.
    Searches both personal knowledge and shared common_knowledge,
    merging results by score.

    Returns a ``(medium_text, untrusted_text)`` tuple where results
    are split by their provenance-derived trust level.
    """
    if not knowledge_dir.is_dir():
        logger.debug("Channel C: No knowledge dir")
        return ("", "")

    try:
        retriever = get_retriever()
        if retriever is None:
            logger.debug("Channel C: Retriever unavailable")
            return ("", "")

        queries = build_dual_queries(message, keywords)
        if not queries:
            logger.debug("Channel C: No keywords and no message")
            return ("", "")
        anima_name = anima_dir.name

        _min_score: float | None = None
        try:
            from core.config.models import load_config as _load_cfg

            _min_score = _load_cfg().rag.min_retrieval_score
        except Exception:
            logger.debug("Failed to load rag.min_retrieval_score from config, using default")

        results = search_and_merge(
            retriever,
            queries,
            anima_name,
            memory_type="knowledge",
            top_k=5,
            include_shared=True,
            min_score=_min_score,
        )

        if restrict_to is not None and results:
            restrict_set = set(restrict_to)
            results = [r for r in results if Path(str(r.metadata.get("source_file", r.doc_id))).stem in restrict_set]

        if results:
            from core.execution._sanitize import ORIGIN_UNKNOWN, resolve_trust

            retriever.record_access(results, anima_name)

            medium_parts: list[str] = []
            untrusted_parts: list[str] = []

            for i, result in enumerate(results):
                chunk_origin = result.metadata.get("origin", "")
                chunk_trust = resolve_trust(chunk_origin or ORIGIN_UNKNOWN)
                source_label = result.metadata.get("anima", anima_name)
                label = "shared" if source_label == "shared" else "personal"
                line = f"--- Result {i + 1} [{label}] (score: {result.score:.3f}) ---\n{result.content}\n"
                if chunk_trust == "untrusted":
                    untrusted_parts.append(line)
                else:
                    medium_parts.append(line)

            medium_output = "\n".join(medium_parts)
            untrusted_output = "\n".join(untrusted_parts)

            logger.debug(
                "Channel C: Vector search returned %d results (medium=%d, untrusted=%d)%s",
                len(results),
                len(medium_parts),
                len(untrusted_parts),
                f" (restricted to {len(restrict_to)} overflow files)" if restrict_to else "",
            )
            return (medium_output, untrusted_output)
        else:
            logger.debug("Channel C: Vector search found no results")
            return ("", "")

    except Exception as e:
        logger.warning("Channel C: Vector search failed: %s", e)
        return ("", "")
