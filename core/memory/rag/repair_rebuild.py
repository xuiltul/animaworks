from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quarantine and reindex helpers for RAG auto-repair."""

import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("animaworks.rag.repair")


def quarantine_vectordb(anima_name: str) -> Path | None:
    import gc

    from core.memory.rag.singleton import reset_vector_store
    from core.paths import get_anima_vectordb_dir

    reset_vector_store(anima_name)
    gc.collect()

    source = get_anima_vectordb_dir(anima_name)
    if not source.exists():
        return None

    archive_dir = source.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dest = archive_dir / f"vectordb-corrupt-{stamp}"
    suffix = 1
    while dest.exists():
        suffix += 1
        dest = archive_dir / f"vectordb-corrupt-{stamp}-{suffix}"
    shutil.move(str(source), str(dest))
    return dest


def full_reindex(anima_name: str, *, include_shared: bool) -> int:
    from core.memory.bm25 import rebuild_longterm_bm25_index
    from core.memory.rag import MemoryIndexer
    from core.memory.rag.singleton import get_vector_store
    from core.paths import get_animas_dir, get_common_knowledge_dir, get_common_skills_dir, get_data_dir

    anima_dir = get_animas_dir() / anima_name
    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        raise RuntimeError("RAG reindex requires ANIMAWORKS_VECTOR_URL; run it through the vector worker")
    vector_store = get_vector_store(anima_name)
    if vector_store is None:
        raise RuntimeError(f"Vector store unavailable for {anima_name}")

    total_chunks = 0
    indexer = MemoryIndexer(vector_store, anima_name, anima_dir)
    for memory_type in ("knowledge", "episodes", "procedures", "skills"):
        memory_dir = anima_dir / memory_type
        if memory_dir.is_dir():
            total_chunks += indexer.index_directory(memory_dir, memory_type, force=True)

    state_dir = anima_dir / "state"
    if (state_dir / "conversation.json").is_file():
        total_chunks += indexer.index_conversation_summary(state_dir, anima_name, force=True)

    bm25_result = rebuild_longterm_bm25_index(anima_dir)
    logger.info("Rebuilt long-term BM25 index for %s: documents=%d", anima_name, bm25_result.documents)

    if include_shared:
        base_dir = get_data_dir()
        shared_indexer = MemoryIndexer(
            vector_store,
            anima_name="shared",
            anima_dir=base_dir,
            collection_prefix="shared",
        )
        for label, src_dir, glob, meta_key in (
            ("common_knowledge", get_common_knowledge_dir(), "*.md", "shared_common_knowledge_hash"),
            ("common_skills", get_common_skills_dir(), "SKILL.md", "shared_common_skills_hash"),
        ):
            if not src_dir.is_dir():
                continue
            total_chunks += shared_indexer.index_directory(src_dir, label, force=True)
            write_shared_hash(anima_dir / "index_meta.json", src_dir, glob, meta_key)
    return total_chunks


def write_shared_hash(meta_path: Path, src_dir: Path, glob: str, meta_key: str) -> None:
    try:
        from core.memory.rag_search import _compute_dir_hash, _write_shared_hash

        _write_shared_hash(meta_path, meta_key, _compute_dir_hash(src_dir, glob))
    except Exception:
        logger.debug("Failed to update shared hash for %s", meta_key, exc_info=True)
