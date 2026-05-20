# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for get_vector_store() when ANIMAWORKS_VECTOR_URL is set."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.http_store import HttpVectorStore
from core.memory.rag.singleton import _reset_for_testing, get_vector_store
from core.memory.rag.store import ChromaVectorStore


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before and after each test for isolation."""
    _reset_for_testing()
    yield
    _reset_for_testing()


def test_returns_http_store_when_env_set():
    """When ANIMAWORKS_VECTOR_URL is set, get_vector_store returns HttpVectorStore."""
    with patch.dict(os.environ, {"ANIMAWORKS_VECTOR_URL": "http://localhost:18500/api/internal/vector"}):
        store = get_vector_store("test_anima")
        assert store is not None
        assert isinstance(store, HttpVectorStore)
        assert store._anima_name == "test_anima"
        assert store._base_url == "http://localhost:18500/api/internal/vector"


def test_returns_none_when_vector_url_and_direct_allow_are_not_set(monkeypatch):
    """Production callers without vector URL must not fall back to direct Chroma."""
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)
    monkeypatch.delenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", raising=False)

    with patch("core.memory.rag.store.create_chroma_vector_store") as create_store:
        store = get_vector_store("test_anima")

    assert store is None
    create_store.assert_not_called()


def test_returns_chroma_store_when_direct_allow_set(monkeypatch):
    """Worker/test callers with explicit allow can use the guarded Chroma factory."""
    os.environ.pop("ANIMAWORKS_VECTOR_URL", None)
    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")
    mock_chroma = MagicMock()

    with patch("core.memory.rag.store.create_chroma_vector_store", return_value=mock_chroma):
        store = get_vector_store("test_anima")

    assert store is not None
    assert not isinstance(store, HttpVectorStore)
    assert store is mock_chroma


def test_returns_none_when_chroma_init_fails(monkeypatch):
    """When ChromaVectorStore init fails, get_vector_store returns None."""
    os.environ.pop("ANIMAWORKS_VECTOR_URL", None)
    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")

    with patch("core.memory.rag.store.create_chroma_vector_store", side_effect=RuntimeError("ChromaDB init failed")):
        store = get_vector_store("test_anima")

    assert store is None


def test_http_store_cache_is_keyed_by_base_url():
    """Changing vector URLs must not reuse an HttpVectorStore for another worker."""
    with patch.dict(os.environ, {"ANIMAWORKS_VECTOR_URL": "http://localhost:1111/vector"}):
        store1 = get_vector_store("test_anima")
    with patch.dict(os.environ, {"ANIMAWORKS_VECTOR_URL": "http://localhost:2222/vector"}):
        store2 = get_vector_store("test_anima")

    assert isinstance(store1, HttpVectorStore)
    assert isinstance(store2, HttpVectorStore)
    assert store1 is not store2
    assert store1._base_url == "http://localhost:1111/vector"
    assert store2._base_url == "http://localhost:2222/vector"


def test_chroma_vector_store_constructor_requires_direct_allow(monkeypatch, tmp_path):
    """The low-level store guard prevents accidental production direct construction."""
    monkeypatch.delenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", raising=False)

    with pytest.raises(RuntimeError, match="Direct ChromaDB access is disabled"):
        ChromaVectorStore(persist_dir=tmp_path / "vectordb")
