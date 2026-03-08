from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for ChromaDB cosine similarity integration.

Verifies that collections are created with cosine distance and
that query scores fall within [0.0, 1.0].
"""

import math
import shutil
import tempfile
from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb", reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'")


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def temp_vectordb():
    """Create a temporary ChromaDB directory and store."""
    tmpdir = Path(tempfile.mkdtemp())
    vectordb_dir = tmpdir / "vectordb"
    vectordb_dir.mkdir()

    from core.memory.rag.store import ChromaVectorStore

    store = ChromaVectorStore(persist_dir=vectordb_dir)
    yield store
    shutil.rmtree(tmpdir)


# ── Collection metric tests ────────────────────────────────


def test_create_collection_sets_cosine(temp_vectordb):
    """New collections must use cosine distance metric."""
    store = temp_vectordb
    store.create_collection("cosine_test", dimension=3)

    coll = store.client.get_collection(name="cosine_test")
    assert coll.metadata.get("hnsw:space") == "cosine"


def test_get_or_create_collection_sets_cosine(temp_vectordb):
    """get_or_create_collection via upsert must use cosine."""
    from core.memory.rag.store import Document

    store = temp_vectordb
    doc = Document(id="d1", content="hello", embedding=[1.0, 0.0, 0.0], metadata={"t": "test"})
    store.upsert("upsert_cosine_test", [doc])

    coll = store.client.get_collection(name="upsert_cosine_test")
    assert coll.metadata.get("hnsw:space") == "cosine"


# ── Score range tests ──────────────────────────────────────


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def test_cosine_scores_in_valid_range(temp_vectordb):
    """Query scores must be in [0.0, 1.0] for cosine collections."""
    from core.memory.rag.store import Document

    store = temp_vectordb
    store.create_collection("score_range_test", dimension=3)

    docs = [
        Document(id="same", content="identical direction", embedding=_normalize([1.0, 0.0, 0.0]), metadata={"t": "a"}),
        Document(id="ortho", content="orthogonal", embedding=_normalize([0.0, 1.0, 0.0]), metadata={"t": "b"}),
        Document(id="similar", content="somewhat similar", embedding=_normalize([0.9, 0.4, 0.1]), metadata={"t": "c"}),
    ]
    store.upsert("score_range_test", docs)

    results = store.query(
        collection="score_range_test",
        embedding=_normalize([1.0, 0.0, 0.0]),
        top_k=3,
    )

    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1] for doc {r.document.id}"


def test_cosine_identical_vector_scores_near_one(temp_vectordb):
    """Querying with the same vector should yield score ~1.0."""
    from core.memory.rag.store import Document

    store = temp_vectordb
    store.create_collection("identical_test", dimension=3)

    vec = _normalize([0.5, 0.5, 0.5])
    store.upsert(
        "identical_test",
        [Document(id="d1", content="test", embedding=vec, metadata={"t": "test"})],
    )

    results = store.query(collection="identical_test", embedding=vec, top_k=1)
    assert len(results) == 1
    assert results[0].score >= 0.99, f"Expected ~1.0 for identical vector, got {results[0].score}"


def test_cosine_ordering(temp_vectordb):
    """More similar vectors should have higher scores."""
    from core.memory.rag.store import Document

    store = temp_vectordb
    store.create_collection("ordering_test", dimension=3)

    query = _normalize([1.0, 0.0, 0.0])
    docs = [
        Document(id="close", content="close", embedding=_normalize([0.95, 0.05, 0.0]), metadata={"t": "a"}),
        Document(id="mid", content="mid", embedding=_normalize([0.5, 0.5, 0.0]), metadata={"t": "b"}),
        Document(id="far", content="far", embedding=_normalize([0.0, 0.0, 1.0]), metadata={"t": "c"}),
    ]
    store.upsert("ordering_test", docs)

    results = store.query(collection="ordering_test", embedding=query, top_k=3)
    assert results[0].document.id == "close"
    assert results[0].score > results[1].score > results[2].score


# ── needs_cosine_migration tests ───────────────────────────


def test_needs_cosine_migration_empty(temp_vectordb):
    """No collections → empty migration list."""
    assert temp_vectordb.needs_cosine_migration() == []


def test_needs_cosine_migration_all_cosine(temp_vectordb):
    """Cosine collections → empty migration list."""
    store = temp_vectordb
    store.create_collection("coll_a", dimension=3)
    store.create_collection("coll_b", dimension=3)

    assert store.needs_cosine_migration() == []


def test_needs_cosine_migration_detects_l2(temp_vectordb):
    """Manually created L2 collection should be detected."""
    store = temp_vectordb

    store.client.create_collection(name="legacy_l2", metadata={"dimension": 3})
    store.create_collection("new_cosine", dimension=3)

    l2_list = store.needs_cosine_migration()
    assert "legacy_l2" in l2_list
    assert "new_cosine" not in l2_list
