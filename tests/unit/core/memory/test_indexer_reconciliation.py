from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.indexer import MemoryIndexer, _IndexFileOutcome
from core.memory.rag.store import Document, SearchResult


class ReconciliationVectorStore:
    def __init__(self, hits: dict[str, list[str]] | None = None, *, delete_ok: bool = True) -> None:
        self.hits = hits or {}
        self.delete_ok = delete_ok
        self.metadata_calls: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, list[str]]] = []

    def get_by_metadata(self, collection: str, where: dict[str, str], limit: int = 20) -> list[SearchResult]:
        source_file = where["source_file"]
        self.metadata_calls.append((collection, source_file))
        return [
            SearchResult(
                document=Document(id=document_id, content="old", metadata={"source_file": source_file}),
                score=1.0,
            )
            for document_id in self.hits.get(source_file, [])
        ]

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        self.deleted.append((collection, ids))
        return self.delete_ok


def _make_indexer(tmp_path: Path, store: ReconciliationVectorStore) -> tuple[MemoryIndexer, Path]:
    anima_dir = tmp_path / "anima"
    knowledge_dir = anima_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.vector_store = store
    indexer.anima_name = "sora"
    indexer.anima_dir = anima_dir
    indexer.collection_prefix = "sora"
    indexer.meta_path = anima_dir / "index_meta.json"
    indexer.index_meta = {}
    indexer.is_ragignored = lambda _path: False  # type: ignore[method-assign]

    def unchanged(_path: Path, _memory_type: str, force: bool = False) -> int:
        del force
        indexer._last_index_file_outcome = _IndexFileOutcome("unchanged")
        return 0

    indexer.index_file = unchanged  # type: ignore[method-assign]
    return indexer, knowledge_dir


def test_reconciliation_deletes_missing_source_chunks_and_meta(tmp_path: Path) -> None:
    source = "knowledge/deleted.md"
    store = ReconciliationVectorStore({source: ["sora/knowledge/deleted.md#0"]})
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    indexer.index_meta[source] = {"hash": "old", "chunks": 1}

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 1
    assert store.deleted == [("sora_knowledge", ["sora/knowledge/deleted.md#0"])]
    assert source not in indexer.index_meta


def test_reconciliation_removes_newly_ragignored_source(tmp_path: Path) -> None:
    source = "knowledge/archive/old.md"
    store = ReconciliationVectorStore({source: ["sora/knowledge/archive/old.md#0"]})
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    archived_file = knowledge_dir / "archive" / "old.md"
    archived_file.parent.mkdir()
    archived_file.write_text("old", encoding="utf-8")
    indexer.index_meta[source] = {"hash": "old", "chunks": 1}
    ragignore = tmp_path / ".ragignore"
    ragignore.write_text("*/knowledge/archive/*\n", encoding="utf-8")
    MemoryIndexer._ragignore_cache = None
    index_file = MagicMock(side_effect=indexer.index_file)
    indexer.index_file = index_file

    with patch("core.paths.get_data_dir", return_value=tmp_path):
        indexer.is_ragignored = MemoryIndexer.is_ragignored  # type: ignore[method-assign]
        result = indexer.index_directory(knowledge_dir, "knowledge")

    index_file.assert_called_once_with(archived_file, "knowledge", force=False)
    assert result.files_unchanged == 1
    assert result.files_reconciled == 1
    assert source not in indexer.index_meta
    MemoryIndexer._ragignore_cache = None


def test_reconciliation_does_not_match_archive_in_ancestor_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "archive" / "runtime"
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(data_dir, store)
    source = "knowledge/live.md"
    live_file = knowledge_dir / "live.md"
    live_file.write_text("live", encoding="utf-8")
    indexer.index_meta[source] = {"hash": "current"}
    (data_dir / ".ragignore").write_text("*/knowledge/archive/*\n", encoding="utf-8")
    MemoryIndexer._ragignore_cache = None

    with patch("core.paths.get_data_dir", return_value=data_dir):
        indexer.is_ragignored = MemoryIndexer.is_ragignored  # type: ignore[method-assign]
        result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 0
    assert source in indexer.index_meta
    assert store.metadata_calls == []
    MemoryIndexer._ragignore_cache = None


def test_reconciliation_does_not_touch_other_directory_entries(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    target_source = "knowledge/missing.md"
    other_source = "episodes/missing.md"
    prefix_source = "knowledge-old/missing.md"
    indexer.index_meta = {
        target_source: {"hash": "old"},
        other_source: {"hash": "old"},
        prefix_source: {"hash": "old"},
    }

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 1
    assert target_source not in indexer.index_meta
    assert other_source in indexer.index_meta
    assert prefix_source in indexer.index_meta
    assert store.metadata_calls == [("sora_knowledge", target_source)]


def test_reconciliation_preserves_meta_when_vector_delete_fails(tmp_path: Path) -> None:
    source = "knowledge/deleted.md"
    store = ReconciliationVectorStore({source: ["sora/knowledge/deleted.md#0"]}, delete_ok=False)
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    indexer.index_meta[source] = {"hash": "old", "chunks": 1}

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 0
    assert source in indexer.index_meta
    assert store.deleted == [("sora_knowledge", ["sora/knowledge/deleted.md#0"])]


def test_reconciliation_continues_after_one_delete_raises(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    indexer.index_meta = {
        "knowledge/first.md": {"hash": "old"},
        "knowledge/second.md": {"hash": "old"},
    }

    def delete(_collection_name: str, source_file: str) -> int:
        if source_file.endswith("first.md"):
            raise RuntimeError("delete failed")
        return 0

    indexer._delete_indexed_file_documents = delete  # type: ignore[method-assign]

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 1
    assert "knowledge/first.md" in indexer.index_meta
    assert "knowledge/second.md" not in indexer.index_meta


def test_reconciliation_caps_each_run_at_500_entries(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    indexer.index_meta = {f"knowledge/missing-{number:03d}.md": {"hash": "old"} for number in range(501)}
    delete_calls: list[str] = []

    def delete(_collection_name: str, source_file: str) -> int:
        delete_calls.append(source_file)
        return 0

    indexer._delete_indexed_file_documents = delete  # type: ignore[method-assign]

    result = indexer.index_directory(knowledge_dir, "knowledge", force=True)

    assert result.files_reconciled == 500
    assert len(delete_calls) == 500
    assert list(indexer.index_meta) == ["knowledge/missing-500.md"]


def test_reconciliation_saves_meta_once_after_batch(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    indexer.index_meta = {
        "knowledge/first.md": {"hash": "old"},
        "knowledge/second.md": {"hash": "old"},
    }
    indexer._save_index_meta = MagicMock()

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 2
    assert indexer.index_meta == {}
    indexer._save_index_meta.assert_called_once_with()


def test_reconciliation_uses_lexical_membership_for_symlinked_parent(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    outside = tmp_path / "outside"
    outside.mkdir()
    (knowledge_dir / "linked").symlink_to(outside, target_is_directory=True)
    source = "knowledge/linked/missing.md"
    indexer.index_meta[source] = {"hash": "old"}

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 1
    assert source not in indexer.index_meta
    assert store.metadata_calls == [("sora_knowledge", source)]


def test_reconciliation_treats_broken_symlink_as_existing(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    source = "knowledge/broken.md"
    (knowledge_dir / "broken.md").symlink_to(tmp_path / "missing-target.md")
    indexer.index_meta[source] = {"hash": "old"}

    result = indexer.index_directory(knowledge_dir, "knowledge")

    assert result.files_reconciled == 0
    assert source in indexer.index_meta
    assert store.metadata_calls == []


def test_shared_indexer_skips_reconciliation(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    source = "knowledge/missing.md"
    indexer.collection_prefix = "shared"
    indexer.index_meta[source] = {"hash": "old"}
    indexer._delete_indexed_file_documents = MagicMock()
    indexer._save_index_meta = MagicMock()

    result = indexer.index_directory(knowledge_dir, "common_knowledge")

    assert result.files_reconciled == 0
    assert source in indexer.index_meta
    indexer._delete_indexed_file_documents.assert_not_called()
    indexer._save_index_meta.assert_not_called()


def test_reconciliation_runs_only_after_successful_directory_scan(tmp_path: Path) -> None:
    store = ReconciliationVectorStore()
    indexer, knowledge_dir = _make_indexer(tmp_path, store)
    active = knowledge_dir / "active.md"
    active.write_text("active", encoding="utf-8")
    indexer.index_meta["knowledge/deleted.md"] = {"hash": "old"}
    indexer.index_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("index failed"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="index failed"):
        indexer.index_directory(knowledge_dir, "knowledge")

    assert "knowledge/deleted.md" in indexer.index_meta
    assert store.metadata_calls == []
