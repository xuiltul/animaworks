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


def reset_worker_vector_store(anima_name: str) -> bool:
    """Reset the vector worker's cached store for an anima when configured."""
    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        return False
    try:
        from core.memory.rag.http_store import HttpVectorStore
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(anima_name)
        if isinstance(store, HttpVectorStore):
            return store.reset_store()
    except Exception:
        logger.debug("Failed to reset vector worker store for %s", anima_name, exc_info=True)
    return False


def quarantine_vectordb(anima_name: str) -> Path | None:
    import gc

    from core.memory.rag.singleton import reset_vector_store
    from core.paths import get_anima_vectordb_dir

    reset_worker_vector_store(anima_name)
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

    # Recreate an empty vectordb dir and drop any worker handle that a concurrent
    # read may have pinned during the move. The pre-move reset above only releases
    # handles so the OS move succeeds; it does not stop a priming read arriving
    # mid-move from lazily re-creating the worker's cached store against the
    # now-missing path. Such a store opens a schema-less stub ("no such table:
    # collections") that the upcoming reindex would reuse, writing an empty DB and
    # leaving reads broken indefinitely. Resetting here forces the reindex to
    # rebuild a clean store bound to the recreated directory.
    source.mkdir(parents=True, exist_ok=True)
    reset_worker_vector_store(anima_name)
    reset_vector_store(anima_name)
    return dest


class RebuildVerificationError(RuntimeError):
    """Raised when a just-rebuilt vector DB is still missing its data.

    A rebuild that reports indexed chunks but whose collections are absent left
    a schema-less stub (e.g. upserts silently failed under worker contention).
    Treating this as a failure lets the caller mark the repair failed so the
    cooldown engages instead of reporting a false success that immediately
    re-triggers another repair.
    """


def verify_rebuilt_vectordb(anima_name: str, *, expected_chunks: int) -> None:
    """Confirm a freshly rebuilt vector DB actually holds its collections.

    Raises ``RebuildVerificationError`` when ``expected_chunks`` is positive but
    the store has no collections (or cannot be listed) — the signature of a stub
    left behind by failed upserts. A genuinely empty anima (``expected_chunks``
    == 0) is considered healthy.
    """
    if expected_chunks <= 0:
        return
    from core.memory.rag.singleton import get_vector_store

    store = get_vector_store(anima_name)
    if store is None:
        raise RebuildVerificationError(
            f"vector store unavailable for {anima_name} after rebuild (indexed {expected_chunks} chunks)"
        )
    try:
        collections = store.list_collections()
    except Exception as exc:  # noqa: BLE001 — any failure here means the DB is unusable
        raise RebuildVerificationError(
            f"rebuilt vector DB for {anima_name} is unreadable (indexed {expected_chunks} chunks): {exc}"
        ) from exc
    if not collections:
        raise RebuildVerificationError(
            f"rebuilt vector DB for {anima_name} has no collections despite indexing {expected_chunks} chunks "
            "(stub left by failed upserts)"
        )


def _reindex_into_store(
    vector_store,
    anima_name: str,
    *,
    include_shared: bool,
    anima_dir: Path | None = None,
) -> int:
    """Index an anima's memory (and optionally shared collections) into a store."""
    from core.company_resources import get_company_resources
    from core.memory.bm25 import rebuild_longterm_bm25_index
    from core.memory.rag import MemoryIndexer
    from core.paths import get_animas_dir, get_common_knowledge_dir, get_common_skills_dir, get_data_dir

    anima_dir = Path(anima_dir) if anima_dir is not None else get_animas_dir() / anima_name
    total_chunks = 0
    indexer = MemoryIndexer(vector_store, anima_name, anima_dir)
    for memory_type in ("knowledge", "episodes", "procedures", "skills", "facts"):
        memory_dir = anima_dir / memory_type
        if memory_dir.is_dir():
            total_chunks += indexer.index_directory(memory_dir, memory_type, force=True).chunks_indexed

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
        shared_sources = [
            ("common_knowledge", get_common_knowledge_dir(), "*.md", "shared_common_knowledge_hash"),
            ("common_skills", get_common_skills_dir(), "SKILL.md", "shared_common_skills_hash"),
        ]
        company_resources = get_company_resources(anima_dir, data_dir=base_dir)
        if company_resources is not None:
            shared_sources.extend(
                (
                    (
                        "common_knowledge",
                        company_resources.knowledge_dir,
                        "*.md",
                        "shared_company_knowledge_hash",
                    ),
                    (
                        "common_skills",
                        company_resources.skills_dir,
                        "SKILL.md",
                        "shared_company_skills_hash",
                    ),
                )
            )
        for label, src_dir, glob, meta_key in shared_sources:
            if not src_dir.is_dir():
                continue
            total_chunks += shared_indexer.index_directory(src_dir, label, force=True).chunks_indexed
            write_shared_hash(anima_dir / "index_meta.json", src_dir, glob, meta_key)
    return total_chunks


def full_reindex(anima_name: str, *, include_shared: bool) -> int:
    """Reindex an anima in place via the vector worker (legacy path)."""
    from core.memory.rag.singleton import get_vector_store

    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        raise RuntimeError("RAG reindex requires ANIMAWORKS_VECTOR_URL; run it through the vector worker")
    vector_store = get_vector_store(anima_name)
    if vector_store is None:
        raise RuntimeError(f"Vector store unavailable for {anima_name}")
    return _reindex_into_store(vector_store, anima_name, include_shared=include_shared)


def atomic_rebuild_vectordb(
    anima_name: str,
    *,
    include_shared: bool,
    anima_dir: Path | None = None,
) -> tuple[int, Path | None]:
    """Build a fresh vector DB in a staging dir and atomically swap it in.

    Unlike the in-place rebuild (quarantine the live DB, then reindex into the
    now-empty live path), this keeps the live DB intact and queryable for the
    whole slow reindex and only swaps at the end. Benefits:

    - A failed rebuild leaves the live DB untouched (no data loss on failure).
    - The worker is reset only twice (at the swap) instead of for the whole
      rebuild, and never serves a half-built DB.
    - The build uses a process-local direct ChromaDB client whose system cache
      is isolated from the worker's, so it cannot be corrupted by — or corrupt —
      live worker traffic.

    Embeddings are still generated via the server (``ANIMAWORKS_EMBED_URL``);
    only the vector writes go to the local staging store. Returns
    ``(chunks_indexed, archive_path)``.
    """
    import gc

    from core.memory.rag.singleton import reset_vector_store
    from core.memory.rag.store import create_chroma_vector_store
    from core.paths import get_anima_vectordb_dir

    resolved_anima_dir = Path(anima_dir) if anima_dir is not None else None
    live = resolved_anima_dir / "vectordb" if resolved_anima_dir is not None else get_anima_vectordb_dir(anima_name)
    staging = live.parent / f"vectordb.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    prev_allow = os.environ.get("ANIMAWORKS_ALLOW_DIRECT_CHROMA")
    os.environ["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] = "1"
    try:
        store = create_chroma_vector_store(persist_dir=staging, anima_name=anima_name)
        try:
            chunks = _reindex_into_store(
                store,
                anima_name,
                include_shared=include_shared,
                anima_dir=resolved_anima_dir,
            )
            if chunks > 0 and not store.list_collections():
                raise RebuildVerificationError(
                    f"staged vector DB for {anima_name} has no collections despite indexing {chunks} chunks"
                )
        finally:
            store.close()
            gc.collect()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # never leave a half-built staging dir
        raise
    finally:
        if prev_allow is None:
            os.environ.pop("ANIMAWORKS_ALLOW_DIRECT_CHROMA", None)
        else:
            os.environ["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] = prev_allow

    # Atomic swap. Reset the worker so it releases the live handle, archive the
    # old DB, move staging into place, then reset again so the worker reopens the
    # new DB. The sibling-drop reset fix keeps these resets from poisoning others.
    archive: Path | None = None
    reset_worker_vector_store(anima_name)
    reset_vector_store(anima_name)
    if live.exists():
        archive_dir = live.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        archive = archive_dir / f"vectordb-corrupt-{stamp}"
        suffix = 1
        while archive.exists():
            suffix += 1
            archive = archive_dir / f"vectordb-corrupt-{stamp}-{suffix}"
        shutil.move(str(live), str(archive))
    shutil.move(str(staging), str(live))
    reset_worker_vector_store(anima_name)
    reset_vector_store(anima_name)
    return chunks, archive


def write_shared_hash(meta_path: Path, src_dir: Path, glob: str, meta_key: str) -> None:
    try:
        from core.memory.rag_search import _compute_dir_hash, _write_shared_hash

        _write_shared_hash(meta_path, meta_key, _compute_dir_hash(src_dir, glob))
    except Exception:
        logger.debug("Failed to update shared hash for %s", meta_key, exc_info=True)
