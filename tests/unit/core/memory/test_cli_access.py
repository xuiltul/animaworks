from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.rag.owner_lock import VectorOwnerBusy, VectorOwnerLock
from core.memory.rag.store import Document, SearchResult, VectorStore


class _MemoryStore(VectorStore):
    def __init__(self) -> None:
        self.documents: dict[str, list[Document]] = {}

    def create_collection(self, name: str) -> bool:
        self.documents.setdefault(name, [])
        return True

    def delete_collection(self, name: str) -> bool:
        self.documents.pop(name, None)
        return True

    def list_collections(self) -> list[str]:
        return list(self.documents)

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        current = {item.id: item for item in self.documents.setdefault(collection, [])}
        current.update({item.id: item for item in documents})
        self.documents[collection] = list(current.values())
        return True

    def query(self, collection: str, embedding: list[float], top_k: int = 10, filter_metadata=None):
        return [SearchResult(document=item, score=1.0) for item in self.documents.get(collection, [])[:top_k]]

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        self.documents[collection] = [item for item in self.documents.get(collection, []) if item.id not in ids]
        return True

    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict]) -> bool:
        return True

    def get_by_metadata(self, collection: str, where: dict, limit: int = 20):
        return []

    def get_all(self, collection: str, limit: int = 100_000):
        return [SearchResult(document=item, score=1.0) for item in self.documents.get(collection, [])[:limit]]

    def count(self, collection: str) -> int:
        return len(self.documents.get(collection, []))

    def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        return [item for item in self.documents.get(collection, []) if item.id in ids]

    # MemoryService dispatches to the single-attempt ``_*_once`` methods.
    _create_collection_once = create_collection
    _delete_collection_once = delete_collection
    _list_collections_once = list_collections
    _upsert_once = upsert
    _query_once = query
    _delete_documents_once = delete_documents
    _update_metadata_once = update_metadata
    _get_by_metadata_once = get_by_metadata
    _get_all_once = get_all
    _count_once = count
    _get_by_ids_once = get_by_ids

    def close(self) -> None:
        return None


def test_open_vector_access_uses_temporary_owner_when_lock_is_free(tmp_path: Path, monkeypatch) -> None:
    import core.memory.rag.cli_access as cli_access
    from core.runtime.memory_service import MemoryService

    anima_dir = tmp_path / "animas" / "sora"
    store = _MemoryStore()
    monkeypatch.setattr(cli_access, "detect_server", lambda: cli_access.ServerInfo(False, "http://127.0.0.1:18500/api"))
    monkeypatch.setattr(MemoryService, "_open_native_store", lambda _self: store)

    async def repair(_self, *, include_shared: bool):
        assert include_shared is True
        return {"ok": True, "status": "success"}

    monkeypatch.setattr(MemoryService, "repair", repair)

    with cli_access.open_vector_access("sora", anima_dir, purpose="test") as access:
        assert access.mode == "owner"
        assert access.store.create_collection("sora_knowledge")
        assert access.store.upsert("sora_knowledge", [Document("doc-1", "hello", [0.1])])
        result = access.store.query("sora_knowledge", [0.1])
        assert result[0].document.content == "hello"
        assert access.repair(include_shared=False) == {"ok": True, "status": "success"}

    assert not cli_access.is_owner_lock_held(anima_dir)


def test_open_vector_access_uses_server_when_owner_lock_is_held(tmp_path: Path, monkeypatch) -> None:
    import core.memory.rag.cli_access as cli_access

    anima_dir = tmp_path / "animas" / "sora"
    lock = VectorOwnerLock(anima_dir, "root")
    lock.acquire()
    monkeypatch.setattr(
        cli_access,
        "detect_server",
        lambda: cli_access.ServerInfo(True, "http://127.0.0.1:18500/api"),
    )
    repair = MagicMock(return_value={"ok": True, "status": "healthy"})
    monkeypatch.setattr(cli_access, "request_repair_and_wait", repair)
    monkeypatch.setattr(cli_access, "_repair_timeout_seconds", lambda: 123.0)
    try:
        with cli_access.open_vector_access("sora", anima_dir, purpose="test") as access:
            assert access.mode == "server"
            assert access.store._base_url == "http://127.0.0.1:18500/api/internal/vector"
            assert access.repair(include_shared=False) == {"ok": True, "status": "healthy"}
            repair.assert_called_once_with(
                "sora",
                include_shared=True,
                reason="cli_test",
                timeout_seconds=123.0,
            )
    finally:
        lock.release()


def test_open_vector_access_rejects_orphaned_owner_lock(tmp_path: Path, monkeypatch) -> None:
    import core.memory.rag.cli_access as cli_access

    anima_dir = tmp_path / "animas" / "sora"
    lock = VectorOwnerLock(anima_dir, "root")
    lock.acquire()
    monkeypatch.setattr(cli_access, "detect_server", lambda: cli_access.ServerInfo(False, "http://unused/api"))
    try:
        with (
            pytest.raises(VectorOwnerBusy, match="animaworks stop"),
            cli_access.open_vector_access("sora", anima_dir, purpose="test"),
        ):
            pytest.fail("access must not open")
    finally:
        lock.release()


@pytest.mark.parametrize(
    ("state", "expected_ok"),
    [({"status": "healthy", "last_chunks_indexed": 4}, True), ({"status": "failed", "last_error": "bad"}, False)],
)
def test_request_repair_and_wait_returns_terminal_state(monkeypatch, state: dict, expected_ok: bool) -> None:
    import core.memory.rag.cli_access as cli_access
    from core.memory.rag.repair import state as repair_state

    reads = iter([{}, state])
    monkeypatch.setattr(repair_state, "read_state", lambda _name: next(reads))
    writer = MagicMock()
    monkeypatch.setattr(repair_state, "write_repair_request_state", writer)

    result = cli_access.request_repair_and_wait(
        "sora",
        include_shared=True,
        reason="test",
        timeout_seconds=2,
    )

    assert result["ok"] is expected_ok
    assert result["status"] == state["status"]
    writer.assert_called_once()


def test_request_repair_and_wait_times_out(monkeypatch) -> None:
    import core.memory.rag.cli_access as cli_access
    from core.memory.rag.repair import state as repair_state

    monkeypatch.setattr(repair_state, "read_state", lambda _name: {"status": "requested"})
    monkeypatch.setattr(repair_state, "write_repair_request_state", MagicMock())
    monkeypatch.setattr(cli_access.time, "monotonic", lambda: 10.0)

    result = cli_access.request_repair_and_wait(
        "sora",
        include_shared=True,
        reason="test",
        timeout_seconds=0,
    )

    assert result["ok"] is False
    assert result["status"] == "timeout"


@pytest.mark.parametrize(
    ("server_url", "expected"),
    [(None, "http://127.0.0.1:18500/api"), ("http://localhost:18610", "http://127.0.0.1:18610/api")],
)
def test_detect_server_uses_server_url_port(
    monkeypatch: pytest.MonkeyPatch, server_url: str | None, expected: str
) -> None:
    from core.memory.rag import cli_access

    if server_url is None:
        monkeypatch.delenv("ANIMAWORKS_SERVER_URL", raising=False)
    else:
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", server_url)
    monkeypatch.setattr("core.platform.pid.read_server_pid", lambda: None)
    assert cli_access.detect_server().base_url == expected
