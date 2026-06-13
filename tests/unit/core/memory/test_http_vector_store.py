# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core/memory/rag/http_store.py — HttpVectorStore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from core.memory.rag.http_store import HttpVectorStore
from core.memory.rag.store import Document, SearchResult


def _make_store(
    base_url: str = "http://localhost:18500/api/internal/vector", anima_name: str = "rin"
) -> HttpVectorStore:
    return HttpVectorStore(base_url=base_url, anima_name=anima_name)


# ── test_query_returns_search_results ─────────────────────────────


def test_query_returns_search_results():
    """Mock httpx.Client.post to return search results. Assert query returns SearchResult list."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"id": "doc1", "content": "hello", "score": 0.9, "metadata": {"type": "knowledge"}},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        results = store.query("rin_knowledge", [0.1, 0.2], top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].document.id == "doc1"
    assert results[0].document.content == "hello"
    assert results[0].document.metadata == {"type": "knowledge"}
    assert results[0].score == 0.9
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "/query"
    assert call_args[1]["json"]["collection"] == "rin_knowledge"
    assert call_args[1]["json"]["embedding"] == [0.1, 0.2]
    assert call_args[1]["json"]["top_k"] == 5


# ── test_query_empty_on_error ─────────────────────────────────────


def test_query_empty_on_error():
    """Mock httpx to raise httpx.HTTPError. Assert query returns []."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.HTTPError("Connection refused")

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        results = store.query("rin_knowledge", [0.1, 0.2], top_k=5)

    assert results == []


# ── test_upsert_sends_documents ───────────────────────────────────


def test_upsert_sends_documents():
    """Mock httpx.Client.post to return 200. Call upsert with 2 documents. Assert correct payload."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    docs = [
        Document(id="d1", content="c1", embedding=[0.1, 0.2], metadata={"k": "v1"}),
        Document(id="d2", content="c2", embedding=[0.3, 0.4], metadata={"k": "v2"}),
    ]

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.upsert("rin_knowledge", docs)

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["anima_name"] == "rin"
    assert payload["collection"] == "rin_knowledge"
    assert len(payload["documents"]) == 2
    assert payload["documents"][0] == {"id": "d1", "content": "c1", "embedding": [0.1, 0.2], "metadata": {"k": "v1"}}
    assert payload["documents"][1] == {"id": "d2", "content": "c2", "embedding": [0.3, 0.4], "metadata": {"k": "v2"}}


# ── test_upsert_batches_large_payloads ────────────────────────────


def test_upsert_batches_large_payloads():
    """Create 600 documents. Mock httpx.Client.post. Call upsert. Assert post called twice (500 + 100)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    docs = [Document(id=f"doc{i}", content=f"content{i}", embedding=[0.1, 0.2], metadata={}) for i in range(600)]

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.upsert("rin_knowledge", docs)

    assert mock_client.post.call_count == 2
    first_payload = mock_client.post.call_args_list[0][1]["json"]
    second_payload = mock_client.post.call_args_list[1][1]["json"]
    assert len(first_payload["documents"]) == 500
    assert len(second_payload["documents"]) == 100


# ── test_update_metadata ──────────────────────────────────────────


def test_update_metadata():
    """Mock httpx, call update_metadata with ids and metadatas, assert correct payload."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.update_metadata(
            "rin_knowledge",
            ids=["id1", "id2"],
            metadatas=[{"k1": "v1"}, {"k2": "v2"}],
        )

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["anima_name"] == "rin"
    assert payload["collection"] == "rin_knowledge"
    assert payload["ids"] == ["id1", "id2"]
    assert payload["metadatas"] == [{"k1": "v1"}, {"k2": "v2"}]


# ── test_get_by_ids ───────────────────────────────────────────────


def test_get_by_ids():
    """Mock httpx to return documents. Assert returns list of Document."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "documents": [
            {"id": "doc1", "content": "hello", "metadata": {"key": "val"}},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        docs = store.get_by_ids("rin_knowledge", ["doc1"])

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].id == "doc1"
    assert docs[0].content == "hello"
    assert docs[0].metadata == {"key": "val"}
    payload = mock_client.post.call_args[1]["json"]
    assert payload["ids"] == ["doc1"]


# ── test_get_by_metadata ───────────────────────────────────────────


def test_get_by_metadata():
    """Similar to query test but for get_by_metadata endpoint."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"id": "doc1", "content": "hello", "score": 1.0, "metadata": {"type": "knowledge"}},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        results = store.get_by_metadata("rin_knowledge", {"type": "knowledge"}, limit=10)

    assert len(results) == 1
    assert results[0].document.id == "doc1"
    assert results[0].document.content == "hello"
    assert results[0].score == 1.0
    payload = mock_client.post.call_args[1]["json"]
    assert payload["where"] == {"type": "knowledge"}
    assert payload["limit"] == 10


# ── test_delete_documents ─────────────────────────────────────────


def test_delete_documents():
    """Mock httpx, call delete_documents, assert correct payload."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.delete_documents("rin_knowledge", ["id1", "id2"])

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["anima_name"] == "rin"
    assert payload["collection"] == "rin_knowledge"
    assert payload["ids"] == ["id1", "id2"]


# ── test_list_collections ─────────────────────────────────────────


def test_list_collections():
    """Mock httpx to return collections. Assert returns list."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"collections": ["col1", "col2"]}
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        colls = store.list_collections()

    assert colls == ["col1", "col2"]


# ── test_create_collection_and_delete_collection ──────────────────


def test_create_collection():
    """Simple POST test for create_collection."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.create_collection("new_col")

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["anima_name"] == "rin"
    assert payload["collection"] == "new_col"
    assert mock_client.post.call_args[0][0] == "/create-collection"


def test_delete_collection():
    """Simple POST test for delete_collection."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.delete_collection("old_col")

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["anima_name"] == "rin"
    assert payload["collection"] == "old_col"
    assert mock_client.post.call_args[0][0] == "/delete-collection"


def test_reset_store_posts_owner_and_clears_local_circuit():
    """reset_store asks the worker to drop cached handles for this anima."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_client.post.return_value = mock_response

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store._write_circuit_retry_at["rin_knowledge"] = 999.0
        assert store.reset_store() is True

    assert store._write_circuit_retry_at == {}
    mock_client.post.assert_called_once_with("/reset-store", json={"anima_name": "rin"})


def test_reset_store_returns_false_on_old_or_unavailable_worker():
    """reset_store must not raise when the worker lacks /reset-store."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.HTTPError("404 Not Found")

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        assert store.reset_store() is False


# ── test_close ─────────────────────────────────────────────────────


def test_close():
    """Call close(), assert client.close() was called."""
    mock_client = MagicMock()

    with patch("httpx.Client", return_value=mock_client):
        store = _make_store()
        store.query("col", [0.1])  # trigger client creation
        store.close()

    mock_client.close.assert_called_once()
    assert store._client is None


def test_close_no_client():
    """Close without ever creating client should not raise."""
    store = _make_store()
    store.close()
    assert store._client is None
