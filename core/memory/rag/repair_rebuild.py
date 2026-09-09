from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Quarantine and reindex helpers for RAG auto-repair."""

import argparse
import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("animaworks.rag.repair")


def _has_active_repair_fence(anima_name: str, *, anima_dir: Path | None = None) -> bool:
    from core.memory.rag import repair_state

    animas_dir = anima_dir.parent if anima_dir is not None else None
    state = repair_state.read_state(anima_name, animas_dir=animas_dir)
    return state.get("status") in repair_state.ACTIVE_REPAIR_STATUSES


def reset_worker_vector_store(anima_name: str) -> bool:
    """Reset the vector worker's cached store for an anima when configured."""
    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        logger.warning(
            "Vector worker reset skipped for %s: ANIMAWORKS_VECTOR_URL is unset",
            anima_name,
        )
        return False
    try:
        from core.memory.rag.http_store import HttpVectorStore
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(anima_name)
        if not isinstance(store, HttpVectorStore):
            logger.warning(
                "Vector worker reset failed for %s: store type mismatch (got %s, need HttpVectorStore)",
                anima_name,
                type(store).__name__ if store is not None else "None",
            )
            return False
        if not store.reset_store():
            logger.warning(
                "Vector worker reset failed for %s: HTTP /reset-store returned failure",
                anima_name,
            )
            return False
        return True
    except Exception:
        logger.warning(
            "Vector worker reset failed for %s: exception during reset",
            anima_name,
            exc_info=True,
        )
        return False


def verify_worker_vector_store(anima_name: str, *, expected_chunks: int) -> bool:
    """Verify the swapped DB through the fenced worker using the repair nonce."""
    repair_nonce = os.environ.get("ANIMAWORKS_RAG_REPAIR_NONCE")
    if not repair_nonce:
        logger.warning(
            "Vector worker verify skipped for %s: ANIMAWORKS_RAG_REPAIR_NONCE is unset",
            anima_name,
        )
        return False
    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        logger.warning(
            "Vector worker verify skipped for %s: ANIMAWORKS_VECTOR_URL is unset",
            anima_name,
        )
        return False
    try:
        from core.memory.rag.http_store import HttpVectorStore
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(anima_name)
        if not isinstance(store, HttpVectorStore):
            logger.warning(
                "Vector worker verify failed for %s: store type mismatch (got %s, need HttpVectorStore)",
                anima_name,
                type(store).__name__ if store is not None else "None",
            )
            return False
        if not store.verify_repair(repair_nonce, expected_chunks=expected_chunks):
            logger.warning(
                "Vector worker verify failed for %s: HTTP /verify-repair returned failure (expected_chunks=%s)",
                anima_name,
                expected_chunks,
            )
            return False
        return True
    except Exception:
        logger.warning(
            "Vector worker verify failed for %s: exception during verify",
            anima_name,
            exc_info=True,
        )
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
    collections = store.list_collections_checked()
    if collections is None:
        raise RebuildVerificationError(
            f"rebuilt vector DB for {anima_name} is unreadable (indexed {expected_chunks} chunks)"
        )
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
    rebuild_bm25: bool = True,
    source_data_dir: Path | None = None,
    source_file_stats: dict | None = None,
) -> tuple[int, dict[str, str]]:
    """Index an anima's memory (and optionally shared collections) into a store."""
    from core.company_resources import get_company_resources
    from core.memory.bm25 import rebuild_longterm_bm25_index
    from core.memory.rag import MemoryIndexer
    from core.paths import get_animas_dir, get_common_knowledge_dir, get_common_skills_dir, get_data_dir

    anima_dir = Path(anima_dir) if anima_dir is not None else get_animas_dir() / anima_name
    total_chunks = 0
    shared_hashes: dict[str, str] = {}
    source_options = (
        {"source_data_dir": source_data_dir, "source_file_stats": source_file_stats}
        if source_data_dir is not None
        else {}
    )
    indexer = MemoryIndexer(vector_store, anima_name, anima_dir, **source_options)
    for memory_type in ("knowledge", "episodes", "procedures", "skills", "facts"):
        memory_dir = anima_dir / memory_type
        if memory_dir.is_dir():
            result = indexer.index_directory(memory_dir, memory_type, force=True)
            if result.files_failed or result.files_unprocessed:
                raise RebuildVerificationError(
                    f"failed to fully rebuild {memory_type} for {anima_name}: "
                    f"failed={result.files_failed} unprocessed={result.files_unprocessed}"
                )
            total_chunks += result.chunks_indexed

    state_dir = anima_dir / "state"
    if (state_dir / "conversation.json").is_file():
        total_chunks += indexer.index_conversation_summary(state_dir, anima_name, force=True)

    from core.memory.entity_index import load_entity_registry, rebuild_entity_collection

    # Resolve one registry snapshot for both the write and its expected count.
    # Each entry creates one entity document. Omitting these documents from the
    # count made otherwise successful full repairs fail verification.
    registry = load_entity_registry(anima_dir)
    entity_count = len(registry.get("entities", {}))
    if entity_count:
        if not rebuild_entity_collection(anima_dir, registry=registry, vector_store=vector_store):
            raise RebuildVerificationError(f"failed to fully rebuild entities for {anima_name}")
        total_chunks += entity_count

    if rebuild_bm25:
        bm25_result = rebuild_longterm_bm25_index(anima_dir)
        logger.info("Rebuilt long-term BM25 index for %s: documents=%d", anima_name, bm25_result.documents)

    if include_shared:
        base_dir = source_data_dir if source_data_dir is not None else get_data_dir()
        shared_indexer = MemoryIndexer(
            vector_store,
            anima_name="shared",
            anima_dir=base_dir,
            collection_prefix="shared",
            **source_options,
        )
        shared_sources = [
            (
                "common_knowledge",
                base_dir / "common_knowledge" if source_data_dir else get_common_knowledge_dir(),
                "*.md",
                "shared_common_knowledge_hash",
            ),
            (
                "common_skills",
                base_dir / "common_skills" if source_data_dir else get_common_skills_dir(),
                "SKILL.md",
                "shared_common_skills_hash",
            ),
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
            result = shared_indexer.index_directory(src_dir, label, force=True)
            if result.files_failed or result.files_unprocessed:
                raise RebuildVerificationError(
                    f"failed to fully rebuild {meta_key} for {anima_name}: "
                    f"failed={result.files_failed} unprocessed={result.files_unprocessed}"
                )
            total_chunks += result.chunks_indexed
            from core.memory.rag_search import _compute_dir_hash

            shared_hashes[meta_key] = _compute_dir_hash(src_dir, glob)
    return total_chunks, shared_hashes


def build_staging_vectordb(
    anima_name: str,
    *,
    include_shared: bool,
    anima_dir: Path,
    staging: Path,
    rebuild_bm25: bool = True,
) -> tuple[int, dict[str, str]]:
    """Build from private inputs without publishing live DB or metadata.

    ``rebuild_bm25`` is retained for call compatibility; BM25 is now rebuilt by
    the promotion owner only after the staged vector data passes verification.
    """
    del rebuild_bm25
    import gc

    from core.memory.rag.store import create_chroma_vector_store

    anima_dir = anima_dir.resolve()
    staging = staging.resolve()
    if staging.parent != anima_dir or not staging.name.startswith("vectordb.staging-"):
        raise ValueError(f"direct Chroma rebuild path must be an anima staging directory: {staging}")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("ANIMAWORKS_ALLOW_DIRECT_CHROMA")
    os.environ["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] = "1"
    try:
        from core.memory.rag.repair_snapshot import save_rebuild_artifacts, snapshot_inputs, validate_rebuild_sources

        with snapshot_inputs(anima_dir, include_shared=include_shared) as inputs:
            store = create_chroma_vector_store(persist_dir=staging, anima_name=anima_name)
            try:
                chunks, shared_hashes = _reindex_into_store(
                    store,
                    anima_name,
                    include_shared=include_shared,
                    anima_dir=inputs.anima_dir,
                    rebuild_bm25=False,
                    source_data_dir=inputs.data_dir,
                    source_file_stats=inputs.file_stats,
                )
                store.verify_rebuilt_data(expected_chunks=chunks)
                save_rebuild_artifacts(staging, inputs)
                validate_rebuild_sources(staging, anima_dir)
                return chunks, shared_hashes
            finally:
                store.close()
                gc.collect()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if previous is None:
            os.environ.pop("ANIMAWORKS_ALLOW_DIRECT_CHROMA", None)
        else:
            os.environ["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] = previous


def full_reindex(anima_name: str, *, include_shared: bool) -> int:
    """Reindex an anima in place via the vector worker (legacy path)."""
    from core.memory.rag.singleton import get_vector_store
    from core.paths import get_animas_dir

    if not os.environ.get("ANIMAWORKS_VECTOR_URL"):
        raise RuntimeError("RAG reindex requires ANIMAWORKS_VECTOR_URL; run it through the vector worker")
    vector_store = get_vector_store(anima_name)
    if vector_store is None:
        raise RuntimeError(f"Vector store unavailable for {anima_name}")
    chunks, shared_hashes = _reindex_into_store(vector_store, anima_name, include_shared=include_shared)
    if shared_hashes:
        from core.memory.rag.shared_meta import write_shared_hashes

        write_shared_hashes(get_animas_dir() / anima_name, shared_hashes)
    return chunks


def _repair_metadata_paths(anima_dir: Path) -> tuple[Path, ...]:
    from core.memory.bm25 import longterm_bm25_delta_path, longterm_bm25_dirty_path, longterm_bm25_index_path
    from core.memory.rag.shared_meta import shared_index_meta_path

    return (
        anima_dir / "index_meta.json",
        shared_index_meta_path(anima_dir),
        longterm_bm25_index_path(anima_dir),
        longterm_bm25_dirty_path(anima_dir),
        longterm_bm25_delta_path(anima_dir),
    )


def _backup_repair_metadata(anima_dir: Path) -> tuple[Path, set[str]]:
    state_dir = anima_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".rag-repair-metadata-", dir=state_dir))
    existing: set[str] = set()
    for path in _repair_metadata_paths(anima_dir):
        if path.is_file():
            relative = path.relative_to(anima_dir)
            existing.add(str(relative))
            (backup_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / relative)
    return backup_dir, existing


def _restore_repair_metadata(anima_dir: Path, backup_dir: Path, existing: set[str]) -> None:
    for path in _repair_metadata_paths(anima_dir):
        relative = path.relative_to(anima_dir)
        backup = backup_dir / relative
        if str(relative) in existing:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, path)
        elif path.exists():
            path.unlink()


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
    from core.memory.rag.repair_snapshot import publish_rebuild_metadata, validate_rebuild_sources
    from core.memory.rag.singleton import reset_vector_store
    from core.paths import get_anima_vectordb_dir

    resolved_anima_dir = Path(anima_dir) if anima_dir is not None else get_anima_vectordb_dir(anima_name).parent
    live = resolved_anima_dir / "vectordb"
    staging = live.parent / f"vectordb.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    metadata_backup: Path | None = None
    metadata_existing: set[str] = set()
    archive: Path | None = None
    swap_started = False
    succeeded = False
    rollback_incomplete = False

    try:
        chunks, shared_hashes = build_staging_vectordb(
            anima_name,
            include_shared=include_shared,
            anima_dir=resolved_anima_dir,
            staging=staging,
        )

        if not _has_active_repair_fence(anima_name, anima_dir=resolved_anima_dir):
            raise RebuildVerificationError(
                f"active RAG repair access fence missing for {anima_name}; refusing vector DB swap"
            )
        validate_rebuild_sources(staging, resolved_anima_dir)
        metadata_backup, metadata_existing = _backup_repair_metadata(resolved_anima_dir)
        if not reset_worker_vector_store(anima_name):
            raise RebuildVerificationError(f"vector worker reset failed before swap for {anima_name}")
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
        # Only roll back a DB swap after the original live directory has
        # actually been retired. A failed first rename leaves it untouched.
        swap_started = True
        shutil.move(str(staging), str(live))

        if not reset_worker_vector_store(anima_name):
            raise RebuildVerificationError(f"vector worker reset failed after swap for {anima_name}")
        reset_vector_store(anima_name)
        if not verify_worker_vector_store(anima_name, expected_chunks=chunks):
            raise RebuildVerificationError(f"vector worker verification failed after swap for {anima_name}")

        from core.memory.bm25 import rebuild_longterm_bm25_index

        rebuild_longterm_bm25_index(resolved_anima_dir)
        if shared_hashes:
            from core.memory.rag.shared_meta import write_shared_hashes

            write_shared_hashes(resolved_anima_dir, shared_hashes)
        # Migrate legacy shared keys before replacing personal file hashes.
        publish_rebuild_metadata(resolved_anima_dir)
        from core.memory.rag.shared_check_registry import invalidate_shared_checks

        invalidate_shared_checks(anima_name)
        succeeded = True
        return chunks, archive
    except BaseException as exc:
        rollback_errors: list[str] = []
        if swap_started:
            try:
                reset_worker_vector_store(anima_name)
                reset_vector_store(anima_name)
                if live.exists():
                    failed_dir = live.parent / "archive"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    failed_stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    failed_live = failed_dir / f"vectordb-rebuild-failed-{failed_stamp}"
                    suffix = 1
                    while failed_live.exists():
                        suffix += 1
                        failed_live = failed_dir / f"vectordb-rebuild-failed-{failed_stamp}-{suffix}"
                    shutil.move(str(live), str(failed_live))
                if archive is not None and archive.exists():
                    shutil.move(str(archive), str(live))
                reset_vector_store(anima_name)
                if not reset_worker_vector_store(anima_name):
                    rollback_errors.append("worker reset failed after rollback")
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        try:
            if swap_started and metadata_backup is not None:
                _restore_repair_metadata(resolved_anima_dir, metadata_backup, metadata_existing)
        except Exception as rollback_exc:
            rollback_errors.append(f"metadata rollback failed: {rollback_exc}")
        if rollback_errors:
            rollback_incomplete = True
            raise RebuildVerificationError(f"{exc}; rollback incomplete: {'; '.join(rollback_errors)}") from exc
        raise
    finally:
        if not succeeded:
            shutil.rmtree(staging, ignore_errors=True)
        if metadata_backup is not None and not rollback_incomplete:
            shutil.rmtree(metadata_backup, ignore_errors=True)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build a phase3 RAG repair staging database")
    parser.add_argument("--build-staging", action="store_true", required=True)
    parser.add_argument("--anima", required=True)
    parser.add_argument("--anima-dir", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    chunks, shared_hashes = build_staging_vectordb(
        args.anima,
        include_shared=args.shared,
        anima_dir=args.anima_dir,
        staging=args.staging,
        rebuild_bm25=False,
    )
    args.result.write_text(
        json.dumps({"chunks": chunks, "shared_hashes": shared_hashes}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
