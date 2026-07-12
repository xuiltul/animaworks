from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for MemoryIndexer source-file deletion cleanup."""

from pathlib import Path

from core.memory.rag.indexer import MemoryChunk, MemoryIndexer
from core.memory.rag.store import Document, SearchResult


class MockVectorStore:
    """Minimal vector store for deletion helper tests."""

    def __init__(self, hits: list[SearchResult] | None = None, *, delete_ok: bool = True) -> None:
        self.hits = hits or []
        self.delete_ok = delete_ok
        self.metadata_calls: list[tuple[str, dict[str, str], int]] = []
        self.deleted: list[tuple[str, list[str]]] = []
        self.created: list[str] = []
        self.upserts: list[tuple[str, list[Document]]] = []

    def get_by_metadata(self, collection: str, where: dict[str, str], limit: int = 20) -> list[SearchResult]:
        self.metadata_calls.append((collection, where, limit))
        return self.hits

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        self.deleted.append((collection, ids))
        return self.delete_ok

    def create_collection(self, collection: str) -> bool:
        self.created.append(collection)
        return True

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        self.upserts.append((collection, documents))
        return True

    def list_collections(self) -> list[str]:
        return self.created


def _make_indexer(tmp_path: Path, vector_store: MockVectorStore) -> MemoryIndexer:
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.vector_store = vector_store
    indexer.anima_name = "test_anima"
    indexer.anima_dir = tmp_path
    indexer.collection_prefix = "test_anima"
    indexer.meta_path = tmp_path / "index_meta.json"
    indexer.index_meta = {}
    indexer._known_collections = set()
    return indexer


def _hit(doc_id: str, source_file: str) -> SearchResult:
    return SearchResult(
        document=Document(
            id=doc_id,
            content="content",
            metadata={"source_file": source_file},
        ),
        score=1.0,
    )


def test_delete_indexed_file_deletes_matching_source_chunks(tmp_path: Path) -> None:
    """delete_indexed_file removes vector docs by exact source_file metadata."""
    source_file = "knowledge/deleted.md"
    vector_store = MockVectorStore(
        [
            _hit("test_anima/knowledge/deleted.md#0", source_file),
            _hit("test_anima/knowledge/deleted.md#1", source_file),
        ]
    )
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 2}

    deleted_count = indexer.delete_indexed_file(tmp_path / source_file, "knowledge")

    assert deleted_count == 2
    assert vector_store.metadata_calls == [
        ("test_anima_knowledge", {"source_file": source_file}, 10_000)
    ]
    assert vector_store.deleted == [
        (
            "test_anima_knowledge",
            [
                "test_anima/knowledge/deleted.md#0",
                "test_anima/knowledge/deleted.md#1",
            ],
        )
    ]
    assert source_file not in indexer.index_meta
    assert source_file not in indexer.meta_path.read_text(encoding="utf-8")


def test_delete_indexed_file_clears_index_meta_when_no_hits(tmp_path: Path) -> None:
    """A deleted source with no vector hits still clears stale index metadata."""
    source_file = "knowledge/missing.md"
    vector_store = MockVectorStore([])
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 1}

    deleted_count = indexer.delete_indexed_file(tmp_path / source_file, "knowledge")

    assert deleted_count == 0
    assert vector_store.deleted == []
    assert source_file not in indexer.index_meta
    assert source_file not in indexer.meta_path.read_text(encoding="utf-8")


def test_delete_indexed_file_preserves_index_meta_when_vector_delete_fails(tmp_path: Path) -> None:
    """Vector delete failures leave source metadata for later repair/retry."""
    source_file = "knowledge/deleted.md"
    vector_store = MockVectorStore([_hit("test_anima/knowledge/deleted.md#0", source_file)], delete_ok=False)
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 1}
    indexer._save_index_meta()

    deleted_count = indexer.delete_indexed_file(tmp_path / source_file, "knowledge")

    assert deleted_count == 0
    assert vector_store.deleted == [("test_anima_knowledge", ["test_anima/knowledge/deleted.md#0"])]
    assert indexer.index_meta[source_file] == {"hash": "old", "chunks": 1}
    assert source_file in indexer.meta_path.read_text(encoding="utf-8")


def test_delete_indexed_file_uses_absolute_source_for_outside_path(tmp_path: Path) -> None:
    """Paths outside anima_dir use the same absolute key fallback as index_file."""
    outside = tmp_path.parent / "outside.md"
    vector_store = MockVectorStore([])
    indexer = _make_indexer(tmp_path, vector_store)

    deleted_count = indexer.delete_indexed_file(outside, "knowledge")

    assert deleted_count == 0
    assert vector_store.metadata_calls == [
        ("test_anima_knowledge", {"source_file": str(outside)}, 10_000)
    ]


def test_index_file_removes_stale_chunks_after_successful_upsert(tmp_path: Path) -> None:
    """Re-indexing a shorter file deletes old chunks not present in the new chunk set."""
    source_file = "knowledge/shrink.md"
    file_path = tmp_path / source_file
    file_path.parent.mkdir()
    file_path.write_text("# Shrink\n\nCurrent", encoding="utf-8")
    vector_store = MockVectorStore(
        [
            _hit("test_anima/knowledge/shrink.md#0", source_file),
            _hit("test_anima/knowledge/shrink.md#1", source_file),
        ]
    )
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 2}
    indexer.is_ragignored = lambda _path: False
    indexer._compute_file_hash = lambda _path: "new"
    indexer._chunk_file = lambda _path, _content, _memory_type, origin="": [
        MemoryChunk(
            id="test_anima/knowledge/shrink.md#0",
            content="current",
            metadata={"source_file": source_file},
        )
    ]
    indexer._generate_embeddings = lambda contents: [[0.1] for _content in contents]

    indexed_count = indexer.index_file(file_path, "knowledge")

    assert indexed_count == 1
    assert vector_store.upserts[0][1][0].id == "test_anima/knowledge/shrink.md#0"
    assert vector_store.deleted == [("test_anima_knowledge", ["test_anima/knowledge/shrink.md#1"])]
    assert indexer.index_meta[source_file]["hash"] == "new"
    assert indexer.index_meta[source_file]["chunks"] == 1


def test_index_file_preserves_old_meta_when_stale_cleanup_fails(tmp_path: Path) -> None:
    """A partial re-index does not mark metadata current if stale chunk cleanup fails."""
    source_file = "knowledge/shrink.md"
    file_path = tmp_path / source_file
    file_path.parent.mkdir()
    file_path.write_text("# Shrink\n\nCurrent", encoding="utf-8")
    vector_store = MockVectorStore(
        [
            _hit("test_anima/knowledge/shrink.md#0", source_file),
            _hit("test_anima/knowledge/shrink.md#1", source_file),
        ],
        delete_ok=False,
    )
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 2}
    indexer.is_ragignored = lambda _path: False
    indexer._compute_file_hash = lambda _path: "new"
    indexer._chunk_file = lambda _path, _content, _memory_type, origin="": [
        MemoryChunk(
            id="test_anima/knowledge/shrink.md#0",
            content="current",
            metadata={"source_file": source_file},
        )
    ]
    indexer._generate_embeddings = lambda contents: [[0.1] for _content in contents]
    failure_state = tmp_path / "state" / "rag_upsert_failures.json"
    failure_state.parent.mkdir()
    failure_state.write_text(
        '{"failures": {"knowledge/shrink.md": {"consecutive_failures": 2}}, "quarantined": []}',
        encoding="utf-8",
    )

    indexed_count = indexer.index_file(file_path, "knowledge")

    assert indexed_count == 0
    assert vector_store.upserts
    assert vector_store.deleted == [("test_anima_knowledge", ["test_anima/knowledge/shrink.md#1"])]
    assert indexer.index_meta[source_file] == {"hash": "old", "chunks": 2}
    state = failure_state.read_text(encoding="utf-8")
    assert source_file not in state


def test_index_file_deletes_stale_chunks_when_new_content_has_no_chunks(tmp_path: Path) -> None:
    """An emptied or non-chunkable file removes old indexed chunks for that source."""
    source_file = "knowledge/emptied.md"
    file_path = tmp_path / source_file
    file_path.parent.mkdir()
    file_path.write_text("", encoding="utf-8")
    vector_store = MockVectorStore([_hit("test_anima/knowledge/emptied.md#0", source_file)])
    indexer = _make_indexer(tmp_path, vector_store)
    indexer.index_meta[source_file] = {"hash": "old", "chunks": 1}
    indexer.is_ragignored = lambda _path: False
    indexer._compute_file_hash = lambda _path: "empty"
    indexer._chunk_file = lambda _path, _content, _memory_type, origin="": []

    indexed_count = indexer.index_file(file_path, "knowledge")

    assert indexed_count == 0
    assert vector_store.upserts == []
    assert vector_store.deleted == [("test_anima_knowledge", ["test_anima/knowledge/emptied.md#0"])]
    assert source_file not in indexer.index_meta
