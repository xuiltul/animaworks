# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.memory.rag.store import ChromaVectorStore, Document


class _FakeChromaVectorStore(ChromaVectorStore):
    clients: list[MagicMock] = []

    def __init__(self, persist_dir: Path | None = None, anima_name: str | None = None) -> None:
        self.client = self.clients.pop(0)
        self.persist_dir = persist_dir or Path("/tmp/vectordb")
        self.anima_name = anima_name
        self._closed = False


def _failing_upsert_client(message: str) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.side_effect = RuntimeError(message)
    return client


def _successful_upsert_client() -> MagicMock:
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def test_chroma_store_self_heals_corruption_and_retries_once(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("disk I/O error (code: 522)")
    good_client = _successful_upsert_client()
    _FakeChromaVectorStore.clients = [bad_client, good_client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is True
    reset.assert_called_once_with("sora")
    assert bad_client.get_or_create_collection.call_count == 1
    assert good_client.get_or_create_collection.call_count == 1


def test_chroma_store_self_heal_returns_false_after_retry_failure(tmp_path: Path) -> None:
    first_bad = _failing_upsert_client("Failed to get segments")
    second_bad = _failing_upsert_client("Failed to get segments")
    _FakeChromaVectorStore.clients = [first_bad, second_bad]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is False
    reset.assert_called_once_with("sora")
    assert first_bad.get_or_create_collection.call_count == 1
    assert second_bad.get_or_create_collection.call_count == 1


def test_chroma_store_does_not_retry_non_corruption_errors(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("permission denied")
    _FakeChromaVectorStore.clients = [bad_client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is False
    reset.assert_not_called()
    assert bad_client.get_or_create_collection.call_count == 1
