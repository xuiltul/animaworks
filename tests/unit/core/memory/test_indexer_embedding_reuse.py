"""Incremental embedding reuse keeps owner-visible chunks and source hashes exact."""

from __future__ import annotations

import copy
import hashlib
from unittest.mock import MagicMock

import pytest

from core.memory.rag.indexer import MemoryIndexer
from core.memory.rag.store import Document, SearchResult


class Store:
    def __init__(self):
        self.docs: dict[str, Document] = {}
        self.upsert_ok = self.metadata_ok = self.delete_ok = True
        self.metadata_calls = 0
        self.after_update = None
        self.lookup_limit = None

    def create_collection(self, collection):
        return True

    def list_collections_checked(self):
        return ["test_episodes", "shared_common_knowledge"]

    def get_by_metadata(self, collection, where, limit=20):
        self.lookup_limit = limit
        return [
            SearchResult(document=copy.deepcopy(doc), score=1)
            for doc in self.docs.values()
            if doc.metadata.get("source_file") == where["source_file"]
        ][:limit]

    def upsert(self, collection, documents):
        if not self.upsert_ok:
            return False
        self.docs.update({doc.id: copy.deepcopy(doc) for doc in documents})
        return True

    def update_metadata(self, collection, ids, metadatas):
        self.metadata_calls += 1
        if not self.metadata_ok:
            return False
        for doc_id, metadata in zip(ids, metadatas, strict=True):
            if doc_id in self.docs:
                self.docs[doc_id].metadata.update(metadata)
        if self.after_update:
            self.after_update()
        return True

    def delete_documents(self, collection, ids):
        if not self.delete_ok:
            return False
        for doc_id in ids:
            self.docs.pop(doc_id, None)
        return True


def make_indexer(tmp_path, monkeypatch, *, shared=False):
    store = Store()
    indexer = MemoryIndexer(
        store,
        "test",
        tmp_path,
        collection_prefix="shared" if shared else "test",
        embedding_model=object(),
        upsert_quarantine_failure_threshold=3,
    )
    indexer._generate_embeddings = MagicMock(side_effect=lambda texts: [[float(len(text)), 1.0] for text in texts])
    monkeypatch.setattr(indexer, "_document_embedding_signature", lambda: "known-policy")
    memory_type = "common_knowledge" if shared else "episodes"
    path = tmp_path / memory_type / "2026-09-08.md"
    path.parent.mkdir()
    return indexer, store, path, memory_type


def body(count):
    return "# Daily record\n\n" + "\n\n".join(
        f"## 12:00 Event {i}\n\n" + f"Observed event number {i}. " * 5 for i in range(count)
    )


@pytest.mark.parametrize("shared", [False, True])
def test_append_embeds_only_new_chunk_and_updates_every_hash(tmp_path, monkeypatch, shared):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch, shared=shared)
    path.write_text(body(600))
    assert idx.index_file(path, kind) == 600
    old_vectors = {key: doc.embedding for key, doc in store.docs.items()}
    first = next(iter(store.docs.values()))
    first.metadata["access_count"] = 7
    idx._generate_embeddings.reset_mock()
    path.write_text(body(601))

    assert idx.index_file(path, kind) == 601
    assert len(idx._generate_embeddings.call_args.args[0]) == 1
    assert store.metadata_calls == 1
    assert len(store.docs) == 601
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(doc.metadata["source_hash"] == digest for doc in store.docs.values())
    assert all(store.docs[key].embedding == vector for key, vector in old_vectors.items())
    assert store.docs[first.id].metadata["access_count"] == 7
    expected = idx._chunk_file(path, path.read_text(), kind)
    assert {doc.id: doc.content for doc in store.docs.values()} == {chunk.id: chunk.content for chunk in expected}


@pytest.mark.parametrize("reason", ["force", "unknown", "policy", "title", "removed_metadata"])
def test_unsafe_reuse_reembeds(tmp_path, monkeypatch, reason):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    assert idx.index_file(path, kind) == 2
    idx._generate_embeddings.reset_mock()
    if reason == "unknown":
        for doc in store.docs.values():
            doc.metadata.pop("embedding_signature")
    elif reason == "policy":
        monkeypatch.setattr(idx, "_document_embedding_signature", lambda: "changed-policy")
    elif reason == "removed_metadata":
        for doc in store.docs.values():
            doc.metadata["trigger_tools"] = "send_message"
    path.write_text(body(2).replace("Daily record", "New title") if reason == "title" else body(2) + "\n")

    assert idx.index_file(path, kind, force=reason == "force") == 2
    assert len(idx._generate_embeddings.call_args.args[0]) == 2
    assert store.metadata_calls == 0


def test_policy_change_reembeds_even_when_source_hash_unchanged(tmp_path, monkeypatch):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    idx.index_file(path, kind)
    idx._generate_embeddings.reset_mock()
    monkeypatch.setattr(idx, "_document_embedding_signature", lambda: "changed-policy")
    assert idx.index_file(path, kind) == 2
    assert len(idx._generate_embeddings.call_args.args[0]) == 2


def test_edit_one_chunk_and_shrink_removes_stale_ids(tmp_path, monkeypatch):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(3))
    idx.index_file(path, kind)
    idx._generate_embeddings.reset_mock()
    path.write_text(body(2).replace("Observed event number 1", "Updated event number 1"))
    assert idx.index_file(path, kind) == 2
    assert len(idx._generate_embeddings.call_args.args[0]) == 1
    assert len(store.docs) == 2


def test_whitespace_only_source_edit_needs_no_embedding(tmp_path, monkeypatch):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    idx.index_file(path, kind)
    idx._generate_embeddings.reset_mock()
    path.write_text(body(2) + "\n\n")
    assert idx.index_file(path, kind) == 2
    idx._generate_embeddings.assert_not_called()
    assert store.metadata_calls == 1


def test_lookup_at_limit_disables_reuse(tmp_path, monkeypatch):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    idx.index_file(path, kind)
    idx._generate_embeddings.reset_mock()
    existing = list(store.docs.values())
    monkeypatch.setattr(
        store,
        "get_by_metadata",
        lambda *args, **kwargs: [SearchResult(document=existing[i % 2], score=1) for i in range(10_000)],
    )
    path.write_text(body(2) + "\n")
    assert idx.index_file(path, kind) == 2
    assert len(idx._generate_embeddings.call_args.args[0]) == 2
    assert store.metadata_calls == 0


@pytest.mark.parametrize("kind", ["common_skills", "shared_users"])
def test_whole_file_shared_collections_refresh_metadata(tmp_path, monkeypatch, kind):
    idx, store, _, _ = make_indexer(tmp_path, monkeypatch, shared=True)
    path = tmp_path / kind / "record.md"
    path.parent.mkdir()
    path.write_text(body(2))
    assert idx.index_file(path, kind) == 1
    idx._generate_embeddings.reset_mock()
    path.write_text(body(2) + "\n")
    assert idx.index_file(path, kind) == 1
    idx._generate_embeddings.assert_not_called()
    assert store.metadata_calls == 1
    doc = next(iter(store.docs.values()))
    assert doc.id == f"shared/{kind}/record.md#0"
    assert doc.metadata["source_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_owner_metadata_update_keeps_embedding_and_refreshes_hash(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from core.memory.rag.store import ChromaVectorStore

    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")
    idx, _, path, kind = make_indexer(tmp_path, monkeypatch)
    store = ChromaVectorStore(persist_dir=tmp_path / "test-vectors")
    idx.vector_store = store
    try:
        path.write_text(body(2))
        assert idx.index_file(path, kind) == 2
        coll = store.client.get_collection("test_episodes")
        before = coll.get(include=["embeddings"])
        old = dict(zip(before["ids"], before["embeddings"], strict=True))
        idx._generate_embeddings.reset_mock()
        path.write_text(body(3))
        assert idx.index_file(path, kind) == 3
        assert len(idx._generate_embeddings.call_args.args[0]) == 1
        after = coll.get(include=["embeddings", "metadatas"])
        vectors = dict(zip(after["ids"], after["embeddings"], strict=True))
        assert all(list(vectors[key]) == list(value) for key, value in old.items())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(metadata["source_hash"] == digest for metadata in after["metadatas"])
        monkeypatch.setattr(idx, "_document_embedding_signature", lambda: None)
        path.write_text(body(4))
        assert idx.index_file(path, kind) == 4
        unsigned = coll.get(include=["metadatas"])
        assert all(metadata["embedding_signature"] == "" for metadata in unsigned["metadatas"])
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["upsert", "metadata", "delete", "missing_id", "wrong_body", "source_changed"])
def test_partial_failure_does_not_commit_source_hash(tmp_path, monkeypatch, failure):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(3))
    idx.index_file(path, kind)
    old_meta = copy.deepcopy(idx.index_meta)
    old_disk = idx.meta_path.read_bytes()
    path.write_text(body(2).replace("Observed event number 1", "Updated event number 1"))
    if failure in {"upsert", "metadata", "delete"}:
        setattr(store, failure + "_ok", False)
    elif failure == "missing_id":
        store.after_update = lambda: store.docs.pop(next(iter(store.docs)))
    elif failure == "wrong_body":
        store.after_update = lambda: setattr(next(iter(store.docs.values())), "content", "concurrent replacement")
    else:
        store.after_update = lambda: path.write_text(body(4))

    assert idx.index_file(path, kind) == 0
    assert idx._last_index_file_outcome.status == "failed"
    assert idx.index_meta == old_meta
    assert idx.meta_path.read_bytes() == old_disk


def test_source_change_during_embedding_does_not_write(tmp_path, monkeypatch):
    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    idx.index_file(path, kind)
    old_docs = copy.deepcopy(store.docs)
    path.write_text(body(3))

    def embed(texts):
        path.write_text(body(4))
        return [[1.0, 1.0] for _ in texts]

    idx._generate_embeddings.side_effect = embed
    assert idx.index_file(path, kind) == 0
    assert idx._last_index_file_outcome.transient is True
    assert store.docs == old_docs


@pytest.mark.parametrize("changed", ["model", "enabled", "document_prefix", "query_prefix", "max_seq_length"])
def test_signature_tracks_actual_encoder_policy(monkeypatch, changed):
    from types import SimpleNamespace

    from core import config
    from core.memory.rag import singleton

    rag = SimpleNamespace(
        embedding_e5_prefix_enabled=True,
        embedding_query_prefix="query: ",
        embedding_document_prefix="passage: ",
        embedding_max_seq_length=512,
    )
    monkeypatch.setattr(config, "load_config", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(singleton, "get_embedding_model_name", lambda: "model-a")
    initial = MemoryIndexer._document_embedding_signature()
    assert initial is not None
    if changed == "model":
        monkeypatch.setattr(singleton, "get_embedding_model_name", lambda: "model-b")
    elif changed == "enabled":
        rag.embedding_e5_prefix_enabled = False
    elif changed == "max_seq_length":
        rag.embedding_max_seq_length = 256
    else:
        setattr(rag, "embedding_" + changed, "different: ")
    assert MemoryIndexer._document_embedding_signature() != initial


def test_unknown_policy_reembeds_without_signing_guessed_defaults(tmp_path, monkeypatch):
    from core import config

    idx, store, path, kind = make_indexer(tmp_path, monkeypatch)
    path.write_text(body(2))
    idx.index_file(path, kind)
    idx._generate_embeddings.reset_mock()
    monkeypatch.setattr(config, "load_config", MagicMock(side_effect=RuntimeError("unreadable config")))
    monkeypatch.setattr(idx, "_document_embedding_signature", MemoryIndexer._document_embedding_signature)
    assert idx._document_embedding_signature() is None
    path.write_text(body(2) + "\n")
    assert idx.index_file(path, kind) == 2
    assert len(idx._generate_embeddings.call_args.args[0]) == 2
    assert store.metadata_calls == 0
    assert all(doc.metadata["embedding_signature"] == "" for doc in store.docs.values())
