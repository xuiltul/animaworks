"""E2E tests for IPC streaming residual issues.

Problem A: setting_sources=[] prevents CLI hook loading
Problem B: Per-anima ChromaDB creates truly isolated vector stores
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Problem A: setting_sources verified in source ──────────────────


class TestSettingSourcesE2E:
    """E2E: Verify setting_sources=[] is present in actual code."""

    def test_both_methods_have_setting_sources(self):
        """Both execute() and execute_streaming() must set setting_sources=[]."""
        import inspect
        from core.execution.agent_sdk import AgentSDKExecutor

        execute_src = inspect.getsource(AgentSDKExecutor.execute)
        streaming_src = inspect.getsource(AgentSDKExecutor.execute_streaming)

        assert "setting_sources=[]" in execute_src, (
            "execute() missing setting_sources=[]"
        )
        assert "setting_sources=[]" in streaming_src, (
            "execute_streaming() missing setting_sources=[]"
        )


# ── Problem B: Per-anima ChromaDB isolation ────────────────────────


class TestPerAnimaChromaDBIsolation:
    """E2E: Real ChromaDB stores per anima are isolated from each other."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path, monkeypatch):
        """Set up a temporary ANIMAWORKS_DATA_DIR."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        return tmp_path

    @pytest.fixture(autouse=True)
    def _reset(self):
        from core.memory.rag.singleton import _reset_for_testing
        _reset_for_testing()
        yield
        _reset_for_testing()

    def test_different_animas_have_separate_directories(self, data_dir: Path):
        """Each anima's vectordb is at {data_dir}/animas/{name}/vectordb."""
        from core.paths import get_anima_vectordb_dir

        dir_a = get_anima_vectordb_dir("alice")
        dir_b = get_anima_vectordb_dir("bob")

        assert dir_a != dir_b
        assert dir_a == data_dir / "animas" / "alice" / "vectordb"
        assert dir_b == data_dir / "animas" / "bob" / "vectordb"

    def test_real_chromadb_isolation(self, data_dir: Path):
        """Data written to anima A's store is NOT visible in anima B's store.

        This is the key E2E test: uses real ChromaDB PersistentClient instances
        to verify that per-anima databases are truly isolated.
        """
        from core.memory.rag.singleton import get_vector_store

        store_a = get_vector_store("alice")
        store_b = get_vector_store("bob")

        # Verify different persist directories
        assert store_a.persist_dir != store_b.persist_dir
        assert "alice" in str(store_a.persist_dir)
        assert "bob" in str(store_b.persist_dir)

        # Create a collection and insert data in store_a
        store_a.create_collection("test_collection", dimension=3)
        from core.memory.rag.store import Document
        doc = Document(
            id="doc1",
            content="Hello from Alice",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "alice"},
        )
        store_a.upsert("test_collection", [doc])

        # Verify data exists in store_a
        results_a = store_a.query(
            "test_collection",
            embedding=[1.0, 0.0, 0.0],
            top_k=5,
        )
        assert len(results_a) == 1
        assert results_a[0].document.content == "Hello from Alice"

        # Verify store_b does NOT have the collection at all
        collections_b = store_b.list_collections()
        assert "test_collection" not in collections_b

        # Verify store_b returns empty results even after creating same-named collection
        store_b.create_collection("test_collection", dimension=3)
        results_b = store_b.query(
            "test_collection",
            embedding=[1.0, 0.0, 0.0],
            top_k=5,
        )
        assert len(results_b) == 0

    def test_none_anima_uses_shared_legacy_dir(self, data_dir: Path):
        """get_vector_store(None) uses the shared legacy vectordb directory."""
        from core.memory.rag.singleton import get_vector_store

        shared_store = get_vector_store(None)
        assert shared_store.persist_dir == data_dir / "vectordb"

    def test_singleton_per_anima(self, data_dir: Path):
        """Same anima_name returns the same ChromaDB instance."""
        from core.memory.rag.singleton import get_vector_store

        s1 = get_vector_store("charlie")
        s2 = get_vector_store("charlie")
        assert s1 is s2

    def test_different_animas_different_instances(self, data_dir: Path):
        """Different anima_names return different ChromaDB instances."""
        from core.memory.rag.singleton import get_vector_store

        sa = get_vector_store("alice")
        sb = get_vector_store("bob")
        assert sa is not sb
