from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.rag.ipc_store import IpcVectorStore, root_memory_requester
from core.memory.rag.singleton import _reset_for_testing, configure_ipc_vector_requester, get_vector_store
from core.memory.rag.store import CollectionExistence
from core.supervisor.memory_service import MemoryService, MemoryServiceUnavailable
from core.supervisor.runner import AnimaRunner


@pytest.fixture(autouse=True)
def clean_singletons():
    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.mark.asyncio
async def test_root_requester_wins_over_http_and_waits_for_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://never-call.invalid/internal/vector")
    monkeypatch.delenv("ANIMAWORKS_MEMORY_VIA_ROOT", raising=False)
    native = MagicMock()
    native._list_collections_once.return_value = ["alice_knowledge"]
    service = MemoryService("alice", tmp_path / "alice", opener=lambda: native)
    supervisor = SimpleNamespace(_memory_service=service, start=service.start, handle_memory=service.handle)
    runner = AnimaRunner("alice", tmp_path / "ipc", tmp_path, tmp_path)
    runner._scheduler_mgr = SimpleNamespace(_task_runner_supervisor=supervisor)
    runner._configure_root_memory_requester()
    try:
        store = get_vector_store("alice")
        assert isinstance(store, IpcVectorStore)
        assert await asyncio.to_thread(store.list_collections_checked) == ["alice_knowledge"]
        native._list_collections_once.assert_called_once()
        # An owner-local requester must not reinterpret another Anima's DB.
        assert get_vector_store("bob") is None
        assert get_vector_store(None) is None
        # Shutdown must not bootstrap another service or bypass the queue.
        runner.shutdown_event.set()
        assert await asyncio.to_thread(store.list_collections_checked) is None
        native._list_collections_once.assert_called_once()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_root_requester_uses_owning_loop_even_from_other_loop():
    owning_loop = asyncio.get_running_loop()
    owning_thread = threading.get_ident()

    async def handler(method, params):
        assert asyncio.get_running_loop() is owning_loop
        assert threading.get_ident() == owning_thread
        return {"method": method, **params}

    requester = root_memory_requester(handler)

    def other_loop():
        async def invoke():
            return requester("memory.query", {"value": 1})

        return asyncio.run(invoke())

    assert await asyncio.to_thread(other_loop) == {"method": "memory.query", "value": 1}
    with pytest.raises(RuntimeError, match="root event loop"):
        requester("memory.query", {})


def test_root_requester_does_not_schedule_on_closed_loop():
    async def construct():
        return root_memory_requester(AsyncMock())

    requester = asyncio.run(construct())
    # A different thread must receive the lifecycle error without creating an
    # unawaited coroutine or opening HTTP/native fallback storage.
    from concurrent.futures import ThreadPoolExecutor

    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="event loop is unavailable"),
    ):
        pool.submit(requester, "memory.query", {}).result(timeout=2)


@pytest.mark.asyncio
async def test_closed_memory_service_stays_unavailable(tmp_path):
    opener = MagicMock(return_value=MagicMock())
    service = MemoryService("alice", tmp_path / "alice", opener=opener)
    requester = root_memory_requester(service.handle)
    await service.close()
    with pytest.raises(MemoryServiceUnavailable):
        await asyncio.to_thread(requester, "memory.list_collections_checked", {})
    opener.assert_not_called()


def test_missing_requester_cannot_fall_back_to_http(monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_MEMORY_VIA_ROOT", "1")
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://never-call.invalid/internal/vector")
    configure_ipc_vector_requester(None)
    assert get_vector_store("alice") is None


@pytest.mark.asyncio
async def test_mcp_entry_uses_host_proxy_not_parent_python_requester(monkeypatch):
    from contextlib import asynccontextmanager

    from core.mcp import server as mcp

    monkeypatch.setenv("ANIMAWORKS_MEMORY_VIA_ROOT", "1")
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://host/api/internal/vector")
    monkeypatch.setenv("ANIMAWORKS_TASK_IPC_PATH", "/parent-only.sock")

    @asynccontextmanager
    async def stdio():
        yield None, None

    monkeypatch.setattr(mcp, "stdio_server", stdio)
    monkeypatch.setattr(mcp.server, "run", AsyncMock())
    await mcp.main()
    from core.memory.rag.http_store import HttpVectorStore

    store = get_vector_store("alice")
    assert type(store) is HttpVectorStore
    assert store._base_url == "http://host/api/internal/vector"
    # Preserve isolation metadata (and cold-auto-index suppression) for tools.
    import os

    assert os.environ["ANIMAWORKS_TASK_IPC_PATH"] == "/parent-only.sock"


@pytest.mark.parametrize("existence", [CollectionExistence.MISSING, CollectionExistence.UNAVAILABLE])
def test_optional_entity_collection_does_not_embed_or_query_when_absent(tmp_path, existence):
    from core.memory.entity_index import _match_query_entities_from_collection

    store = MagicMock()
    store.collection_exists.return_value = existence
    embed = MagicMock()
    assert (
        _match_query_entities_from_collection(
            tmp_path / "alice",
            "synthetic lookup",
            vector_store=store,
            embedding_fn=embed,
            top_k=5,
            min_score=0.3,
        )
        == set()
    )
    store.collection_exists.assert_called_once_with("alice_entities")
    store.query.assert_not_called()
    embed.assert_not_called()
