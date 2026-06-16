# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.sqlite_health import SQLiteHealthResult
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


def _upsert_client_with_side_effects(*effects: object) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.side_effect = effects
    return client


def _sqlite_health(tmp_path: Path, *, status: str, ok: bool) -> SQLiteHealthResult:
    return SQLiteHealthResult(db_path=tmp_path / "chroma.sqlite3", ok=ok, status=status)


def test_chroma_store_self_heals_corruption_and_retries_once(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("database disk image is malformed")
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
    bad_client.close.assert_called_once()
    good_client.close.assert_called_once()


def test_chroma_store_retries_transient_error_without_reset(tmp_path: Path) -> None:
    collection = MagicMock()
    client = _upsert_client_with_side_effects(RuntimeError("Failed to get segments"), collection)
    _FakeChromaVectorStore.clients = [client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
        patch(
            "core.memory.rag.sqlite_health.quick_check_chroma_sqlite",
            return_value=_sqlite_health(tmp_path, status="ok", ok=True),
        ),
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is True
    reset.assert_not_called()
    assert client.get_or_create_collection.call_count == 2
    client.close.assert_not_called()


def test_chroma_store_retries_sqlite_refutable_error_when_quick_check_ok(tmp_path: Path) -> None:
    collection = MagicMock()
    client = _upsert_client_with_side_effects(RuntimeError("no such table: collections"), collection)
    _FakeChromaVectorStore.clients = [client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
        patch(
            "core.memory.rag.sqlite_health.quick_check_chroma_sqlite",
            return_value=_sqlite_health(tmp_path, status="ok", ok=True),
        ) as quick_check,
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is True
    reset.assert_not_called()
    quick_check.assert_called_once_with(tmp_path)
    assert client.get_or_create_collection.call_count == 2
    client.close.assert_not_called()


def test_chroma_store_self_heal_returns_false_after_lightweight_retry_failure(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("Failed to get segments")
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
    assert bad_client.get_or_create_collection.call_count == 2
    bad_client.close.assert_not_called()


def test_chroma_store_lightweight_retry_failure_raises_from_self_heal(tmp_path: Path) -> None:
    client = MagicMock()
    _FakeChromaVectorStore.clients = [client]
    calls = 0

    def action(_store: ChromaVectorStore) -> bool:
        nonlocal calls
        calls += 1
        raise RuntimeError("Failed to get segments")

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        with pytest.raises(RuntimeError, match="Failed to get segments"):
            store._with_self_heal("upsert", "sora_knowledge", action)

    assert calls == 2
    reset.assert_not_called()
    client.close.assert_not_called()


def test_chroma_store_resets_when_chroma_corruption_fails_quick_check(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("no such table: collections")
    good_client = _successful_upsert_client()
    _FakeChromaVectorStore.clients = [bad_client, good_client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
        patch(
            "core.memory.rag.sqlite_health.quick_check_chroma_sqlite",
            return_value=_sqlite_health(tmp_path, status="corrupt", ok=False),
        ),
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
    bad_client.close.assert_called_once()
    good_client.close.assert_called_once()


def test_chroma_store_resets_hnsw_corruption_without_sqlite_gate(tmp_path: Path) -> None:
    bad_client = _failing_upsert_client("hnsw index panic: corrupt graph")
    good_client = _successful_upsert_client()
    _FakeChromaVectorStore.clients = [bad_client, good_client]

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        patch("core.memory.rag.repair.record_chroma_error"),
        patch(
            "core.memory.rag.sqlite_health.quick_check_chroma_sqlite",
            return_value=_sqlite_health(tmp_path, status="ok", ok=True),
        ) as quick_check,
    ):
        store = _FakeChromaVectorStore(persist_dir=tmp_path, anima_name="sora")
        ok = store.upsert(
            "sora_knowledge",
            [Document(id="doc1", content="hello", embedding=[0.1], metadata={})],
        )

    assert ok is True
    reset.assert_called_once_with("sora")
    quick_check.assert_not_called()
    assert bad_client.get_or_create_collection.call_count == 1
    assert good_client.get_or_create_collection.call_count == 1
    bad_client.close.assert_called_once()
    good_client.close.assert_called_once()


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
    bad_client.close.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "hnsw segment reader: Too many open files (os error 24)",
        "unable to open database file",
    ],
)
def test_chroma_store_does_not_retry_resource_exhaustion(
    tmp_path: Path,
    message: str,
) -> None:
    bad_client = _failing_upsert_client(message)
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
    bad_client.close.assert_not_called()
