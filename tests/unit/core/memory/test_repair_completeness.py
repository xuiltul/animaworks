from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.memory.rag.indexer import IndexDirectoryResult
from core.memory.rag.repair_rebuild import RebuildVerificationError, _reindex_into_store
from core.memory.rag.store import ChromaVectorStore


def _entity_rebuild_fixture(tmp_path, monkeypatch, *, entities):
    anima_dir = tmp_path / "alice"
    (anima_dir / "knowledge").mkdir(parents=True)
    indexer = MagicMock()
    indexer.index_directory.return_value = IndexDirectoryResult(chunks_indexed=3, files_indexed=1)
    monkeypatch.setattr("core.memory.rag.MemoryIndexer", MagicMock(return_value=indexer))
    registry = {"version": 1, "entities": entities}
    monkeypatch.setattr("core.memory.entity_index.load_entity_registry", lambda _path: registry)
    monkeypatch.setattr(
        "core.memory.rag.singleton.generate_embeddings",
        lambda texts, **kwargs: [[0.1, 0.2] for _ in texts],
    )
    store = MagicMock()
    store.create_collection.return_value = True
    store.upsert.return_value = True
    return anima_dir, store


def test_rebuild_expected_chunks_includes_actual_entity_documents(tmp_path, monkeypatch):
    anima_dir, store = _entity_rebuild_fixture(
        tmp_path,
        monkeypatch,
        entities={"ada": {"canonical": "Ada"}, "bert": {"canonical": "Bert"}},
    )
    chunks, hashes = _reindex_into_store(
        store,
        "alice",
        include_shared=False,
        anima_dir=anima_dir,
        rebuild_bm25=False,
    )
    assert chunks == 5  # Three knowledge chunks plus two actual entity upserts.
    assert hashes == {}
    store.create_collection.assert_called_once_with("alice_entities")
    collection, documents = store.upsert.call_args.args
    assert collection == "alice_entities"
    assert {document.id for document in documents} == {"alice/entity/ada", "alice/entity/bert"}
    assert all(document.embedding == [0.1, 0.2] for document in documents)


@pytest.mark.parametrize("failed_operation", ["create_collection", "upsert"])
def test_required_entity_write_failure_is_not_successful_repair(tmp_path, monkeypatch, failed_operation):
    anima_dir, store = _entity_rebuild_fixture(tmp_path, monkeypatch, entities={"ada": {"canonical": "Ada"}})
    getattr(store, failed_operation).return_value = False
    with pytest.raises(RebuildVerificationError, match="fully rebuild entities"):
        _reindex_into_store(store, "alice", include_shared=False, anima_dir=anima_dir, rebuild_bm25=False)
    if failed_operation == "create_collection":
        store.upsert.assert_not_called()


def test_empty_registry_needs_no_entity_collection(tmp_path, monkeypatch):
    anima_dir, store = _entity_rebuild_fixture(tmp_path, monkeypatch, entities={})
    chunks, _ = _reindex_into_store(
        store,
        "alice",
        include_shared=False,
        anima_dir=anima_dir,
        rebuild_bm25=False,
    )
    assert chunks == 3
    store.create_collection.assert_not_called()
    store.upsert.assert_not_called()


def _verification_store():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    collections = {}
    for name in ("alice_knowledge", "shared_common_knowledge", "alice_entities", "empty"):
        collection = MagicMock()
        collection.count.return_value = 0 if name == "empty" else 1
        collection.get.return_value = {"ids": [name + "/doc"], "embeddings": [[0.1, 0.2]]}
        collection.query.return_value = {"ids": [[name + "/doc"]]}
        collections[name] = collection
    store.client.list_collections.return_value = [SimpleNamespace(name=name) for name in collections]
    store.client.get_collection.side_effect = lambda name: collections[name]
    return store, collections


def test_rebuild_verifies_each_nonempty_collection_and_skips_only_empty():
    store, collections = _verification_store()
    assert store.verify_rebuilt_data(expected_chunks=3) == {"collections": 4, "chunks": 3, "query_results": 3}
    for name, collection in collections.items():
        if name == "empty":
            collection.get.assert_not_called()
            collection.query.assert_not_called()
        else:
            collection.query.assert_called_once_with(query_embeddings=[[0.1, 0.2]], n_results=1)


@pytest.mark.parametrize("failure", ["missing_embedding", "empty_results", "query_error"])
def test_first_collection_success_cannot_hide_later_collection_failure(failure):
    store, collections = _verification_store()
    failed = collections["alice_entities"]
    if failure == "missing_embedding":
        failed.get.return_value = {"ids": [], "embeddings": []}
    elif failure == "empty_results":
        failed.query.return_value = {"ids": [[]]}
    else:
        failed.query.side_effect = RuntimeError("synthetic native query failure")
    with pytest.raises(RuntimeError):
        store.verify_rebuilt_data(expected_chunks=3)
    # Two successful queries precede this failure; verification must not have
    # returned success after either of them.
    collections["alice_knowledge"].query.assert_called_once()
    collections["shared_common_knowledge"].query.assert_called_once()
