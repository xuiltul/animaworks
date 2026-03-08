"""Unit tests for ChromaDB cosine similarity configuration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Score clamping ────────────────────────────────────────


class TestScoreClamp:
    """Test that scores are clamped to [0.0, 1.0]."""

    def _make_store_and_query(self, distances: list[float]) -> list:
        """Create a ChromaVectorStore with mocked client and query it."""
        with patch("chromadb.PersistentClient"):
            from core.memory.rag.store import ChromaVectorStore

            store = ChromaVectorStore.__new__(ChromaVectorStore)
            store.client = MagicMock()
            store.persist_dir = MagicMock()

        mock_coll = MagicMock()
        mock_coll.query.return_value = {
            "ids": [["d1", "d2", "d3"][: len(distances)]],
            "documents": [["doc"] * len(distances)],
            "metadatas": [[{}] * len(distances)],
            "distances": [distances],
        }
        store.client.get_collection.return_value = mock_coll

        return store.query(collection="test", embedding=[0.0], top_k=len(distances))

    def test_normal_cosine_score(self):
        results = self._make_store_and_query([0.2])
        assert results[0].score == pytest.approx(0.8)

    def test_identical_vectors(self):
        results = self._make_store_and_query([0.0])
        assert results[0].score == pytest.approx(1.0)

    def test_negative_raw_score_clamped_to_zero(self):
        results = self._make_store_and_query([1.5])
        assert results[0].score == pytest.approx(0.0)

    def test_slight_overshoot_clamped_to_one(self):
        results = self._make_store_and_query([-0.001])
        assert results[0].score == pytest.approx(1.0)

    def test_all_scores_in_range(self):
        results = self._make_store_and_query([0.0, 0.5, 1.0, 1.5, 2.0])
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ── Cosine metric in create_collection ────────────────────


class TestCosineMetric:
    """Test that cosine metric is specified when creating collections."""

    @pytest.fixture
    def mock_store(self):
        with patch("chromadb.PersistentClient") as mock_client_cls:
            from core.memory.rag.store import ChromaVectorStore

            store = ChromaVectorStore.__new__(ChromaVectorStore)
            store.client = mock_client_cls.return_value
            store.persist_dir = MagicMock()
            yield store

    def test_create_collection_uses_cosine(self, mock_store):
        mock_store.create_collection("test_coll", dimension=384)

        mock_store.client.create_collection.assert_called_once_with(
            name="test_coll",
            metadata={"hnsw:space": "cosine", "dimension": 384},
        )

    def test_get_or_create_collection_uses_cosine(self, mock_store):
        from core.memory.rag.store import Document

        doc = Document(id="d1", content="hello", embedding=[0.1, 0.2, 0.3])
        mock_store.upsert("test_coll", [doc])

        mock_store.client.get_or_create_collection.assert_called_once_with(
            name="test_coll",
            metadata={"hnsw:space": "cosine"},
        )


# ── needs_cosine_migration ────────────────────────────────


class TestNeedsCosinesMigration:
    """Test detection of L2 collections."""

    @pytest.fixture
    def mock_store(self):
        with patch("chromadb.PersistentClient"):
            from core.memory.rag.store import ChromaVectorStore

            store = ChromaVectorStore.__new__(ChromaVectorStore)
            store.client = MagicMock()
            store.persist_dir = MagicMock()
            yield store

    def test_detects_l2_collection(self, mock_store):
        l2_coll = MagicMock()
        l2_coll.name = "old_l2"
        l2_coll.metadata = {"dimension": 384}

        mock_store.client.list_collections.return_value = [l2_coll]
        mock_store.client.get_collection.return_value = l2_coll

        result = mock_store.needs_cosine_migration()
        assert result == ["old_l2"]

    def test_cosine_collection_not_flagged(self, mock_store):
        cos_coll = MagicMock()
        cos_coll.name = "good_cosine"
        cos_coll.metadata = {"hnsw:space": "cosine", "dimension": 384}

        mock_store.client.list_collections.return_value = [cos_coll]
        mock_store.client.get_collection.return_value = cos_coll

        result = mock_store.needs_cosine_migration()
        assert result == []

    def test_mixed_collections(self, mock_store):
        l2_coll = MagicMock()
        l2_coll.name = "legacy_l2"
        l2_coll.metadata = {}

        cos_coll = MagicMock()
        cos_coll.name = "new_cosine"
        cos_coll.metadata = {"hnsw:space": "cosine"}

        mock_store.client.list_collections.return_value = [l2_coll, cos_coll]
        mock_store.client.get_collection.side_effect = lambda name: l2_coll if name == "legacy_l2" else cos_coll

        result = mock_store.needs_cosine_migration()
        assert result == ["legacy_l2"]

    def test_empty_metadata_treated_as_l2(self, mock_store):
        coll = MagicMock()
        coll.name = "no_meta"
        coll.metadata = None

        mock_store.client.list_collections.return_value = [coll]
        mock_store.client.get_collection.return_value = coll

        result = mock_store.needs_cosine_migration()
        assert result == ["no_meta"]
