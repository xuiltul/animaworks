from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-file deletion helpers for MemoryIndexer."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.rag.indexer")


def delete_indexed_file(indexer: Any, file_path: Path, memory_type: str) -> int:
    """Delete indexed chunks for a source file and clear index metadata."""
    collection_name = f"{indexer.collection_prefix}_{memory_type}"
    try:
        file_key = str(file_path.relative_to(indexer.anima_dir))
    except ValueError:
        file_key = str(file_path)

    deleted_count = delete_indexed_file_documents(indexer, collection_name, file_key)
    if deleted_count is None:
        return 0

    if file_key in indexer.index_meta:
        indexer.index_meta.pop(file_key, None)
        indexer._save_index_meta()
    return deleted_count


def get_indexed_file_document_ids(indexer: Any, collection_name: str, source_file: str) -> list[str] | None:
    """Return indexed document IDs for a source file, or None when lookup fails."""
    documents = get_indexed_file_documents(indexer, collection_name, source_file)
    return None if documents is None else [document.id for document in documents]


def get_indexed_file_documents(indexer: Any, collection_name: str, source_file: str) -> list[Any] | None:
    """Read existing source chunks through the configured owner, without vectors."""
    try:
        results = indexer.vector_store.get_by_metadata(collection_name, {"source_file": source_file}, limit=10_000)
    except Exception:
        logger.debug("Failed to find indexed documents for %s/%s", collection_name, source_file, exc_info=True)
        return None
    return [result.document for result in results]


def delete_document_ids(indexer: Any, collection_name: str, source_file: str, ids: list[str]) -> bool:
    """Delete known document IDs for a source file."""
    if not ids:
        return True
    try:
        if indexer.vector_store.delete_documents(collection_name, ids):
            return True
        logger.warning("Vector delete failed for %s/%s", collection_name, source_file)
    except Exception:
        logger.debug("Failed to delete indexed documents for %s/%s", collection_name, source_file, exc_info=True)
    return False


def upsert_file_documents(
    indexer: Any,
    collection_name: str,
    source_file: str,
    file_path: Path,
    documents: list[Any],
    *,
    existing_ids: list[str] | None = None,
    metadata_only: list[Any] | None = None,
) -> bool:
    """Upsert current chunks and remove obsolete chunks for the same source file."""
    if existing_ids is None:
        existing_ids = get_indexed_file_document_ids(indexer, collection_name, source_file)
    if existing_ids is None:
        logger.warning("Could not inspect existing chunks for %s, skipping index", file_path)
        return False

    if (documents or not metadata_only) and not indexer.vector_store.upsert(collection_name, documents):
        transient_probe = getattr(indexer.vector_store, "is_transient_write_failure", None)
        log = logger.debug if callable(transient_probe) and transient_probe(collection_name) else logger.warning
        log("Upsert failed for %s, skipping index_meta update", file_path)
        indexer._record_upsert_failure(collection_name, source_file, file_path)
        return False

    if metadata_only and not indexer.vector_store.update_metadata(
        collection_name,
        [document.id for document in metadata_only],
        [document.metadata for document in metadata_only],
    ):
        logger.warning("Metadata refresh failed for %s, skipping index_meta update", file_path)
        return False

    indexer._mark_collection_known(collection_name)
    indexer._clear_upsert_failure(source_file, collection_name)
    current_ids = {document.id for document in documents}
    current_ids.update(document.id for document in metadata_only or [])
    stale_ids = [document_id for document_id in existing_ids if document_id not in current_ids]
    if not delete_document_ids(indexer, collection_name, source_file, stale_ids):
        logger.warning("Stale chunk cleanup failed for %s, skipping index_meta update", file_path)
        return False
    if metadata_only:
        # Metadata updates do not fail when an ID disappeared concurrently.
        # Verify the owner-visible body, signature and source hash before
        # committing the source ledger; a partial write must remain retryable.
        actual = get_indexed_file_documents(indexer, collection_name, source_file)
        if actual is None:
            return False
        by_id = {document.id: document for document in actual}
        for document in [*documents, *metadata_only]:
            stored = by_id.get(document.id)
            if (
                stored is None
                or stored.content != document.content
                or any(
                    stored.metadata.get(key) != document.metadata.get(key)
                    for key in ("source_hash", "embedding_signature")
                )
            ):
                logger.warning("Incremental chunk verification failed for %s, skipping index_meta update", file_path)
                return False
        if set(by_id) != current_ids:
            return False
    return True


def delete_indexed_file_documents(indexer: Any, collection_name: str, source_file: str) -> int | None:
    """Best-effort removal of stale chunks for a file that must not be indexed."""
    ids = get_indexed_file_document_ids(indexer, collection_name, source_file)
    if ids is None:
        return None
    if not delete_document_ids(indexer, collection_name, source_file, ids):
        return None
    return len(ids)
