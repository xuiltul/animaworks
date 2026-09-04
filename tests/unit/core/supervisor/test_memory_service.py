from __future__ import annotations

import asyncio
import shutil
import threading
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.rag.ipc_store import IpcVectorStore
from core.memory.rag.store import CollectionExistence, Document, SearchResult
from core.supervisor.ipc_v2 import IPCV2Connection, IPCV2ConnectionState, IPCV2Identity, ipc_v2_error
from core.supervisor.memory_service import MemoryService, MemoryServiceUnavailable
from core.supervisor.task_runner import _MemoryRpcClient


def _store() -> MagicMock:
    store = MagicMock()
    store._create_collection_once.return_value = True
    store._delete_collection_once.return_value = True
    store._upsert_once.return_value = True
    store._delete_documents_once.return_value = True
    store._update_metadata_once.return_value = True
    store._query_once.return_value = [SearchResult(Document("doc-1", "hello", metadata={"kind": "knowledge"}), 0.9)]
    store._list_collections_once.return_value = ["sakura_knowledge"]
    store._get_by_metadata_once.return_value = [
        SearchResult(Document("doc-2", "matched", metadata={"kind": "knowledge"}), 1.0)
    ]
    store._get_by_ids_once.return_value = [Document("doc-1", "hello", metadata={"kind": "knowledge"})]
    return store


@pytest.mark.asyncio
async def test_memory_service_checked_reads(tmp_path: Path) -> None:
    store = _store()
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)

    query = await service.handle(
        "memory.query",
        {"collection": "sakura_knowledge", "embedding": [0.1, 0.2], "top_k": 3},
    )
    listed = await service.handle("memory.list_collections_checked", {})
    metadata = await service.handle(
        "memory.get_by_metadata",
        {"collection": "sakura_knowledge", "where": {"kind": "knowledge"}, "limit": 2},
    )
    documents = await service.handle(
        "memory.get_by_ids",
        {"collection": "sakura_knowledge", "ids": ["doc-1"]},
    )

    assert query["results"][0]["document"]["id"] == "doc-1"
    assert listed == {"collections": ["sakura_knowledge"]}
    assert metadata["results"][0]["document"]["content"] == "matched"
    assert documents["documents"][0]["metadata"] == {"kind": "knowledge"}
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_logs_queue_and_execution_times(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    service = MemoryService("sakura", tmp_path / "sakura", opener=_store)

    with caplog.at_level("INFO", logger="core.supervisor.memory_service"):
        await service.handle(
            "memory.query",
            {"collection": "sakura_knowledge", "embedding": [0.1], "top_k": 1},
        )

    assert "method=memory.query" in caplog.text
    assert "queue_wait=" in caplog.text
    assert "execute=" in caplog.text
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_rejects_invalid_request_as_protocol_error(tmp_path: Path) -> None:
    service = MemoryService("sakura", tmp_path / "sakura", opener=_store)

    with pytest.raises(ValueError, match="embedding"):
        await service.handle(
            "memory.query",
            {"collection": "sakura_knowledge", "embedding": "not-a-vector"},
        )
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_supports_all_vector_store_writes(tmp_path: Path) -> None:
    store = _store()
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)
    document = {"id": "doc-1", "content": "hello", "embedding": [0.1], "metadata": {"kind": "knowledge"}}

    assert await service.handle("memory.create_collection", {"collection": "sakura_knowledge"}) == {"ok": True}
    assert await service.handle(
        "memory.upsert",
        {"collection": "sakura_knowledge", "documents": [document]},
    ) == {"ok": True}
    assert await service.handle(
        "memory.update_metadata",
        {"collection": "sakura_knowledge", "ids": ["doc-1"], "metadatas": [{"kind": "episode"}]},
    ) == {"ok": True}
    assert await service.handle(
        "memory.delete_documents",
        {"collection": "sakura_knowledge", "ids": ["doc-1"]},
    ) == {"ok": True}
    assert await service.handle("memory.delete_collection", {"collection": "sakura_knowledge"}) == {"ok": True}

    store._create_collection_once.assert_called_once_with("sakura_knowledge")
    store._upsert_once.assert_called_once()
    store._update_metadata_once.assert_called_once_with("sakura_knowledge", ["doc-1"], [{"kind": "episode"}])
    store._delete_documents_once.assert_called_once_with("sakura_knowledge", ["doc-1"])
    store._delete_collection_once.assert_called_once_with("sakura_knowledge")
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_defers_and_atomically_applies_access_deltas(tmp_path: Path) -> None:
    store = _store()
    metadata: dict[str, object] = {
        "access_count": 1.0,
        "retrieved_count": 2,
        "used_count": 3,
    }
    entered = threading.Event()
    release = threading.Event()

    def get_by_ids(_collection: str, _ids: list[str]) -> list[Document]:
        return [Document("doc-1", "hello", metadata=dict(metadata))]

    def update_metadata(_collection: str, _ids: list[str], metadatas: list[dict]) -> bool:
        entered.set()
        release.wait(timeout=2)
        metadata.update(metadatas[0])
        return True

    store._get_by_ids_once.side_effect = get_by_ids
    store._update_metadata_once.side_effect = update_metadata
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)
    operation = {
        "collection": "sakura_knowledge",
        "doc_id": "doc-1",
        "access_delta": 0.2,
        "retrieved_delta": 1,
        "used_delta": 0,
        "last_accessed_at": "2026-08-14T00:00:00+00:00",
        "last_retrieved_at": "2026-08-14T00:00:00+00:00",
    }

    assert await service.handle("memory.apply_access_updates", {"operations": [operation]}) == {"ok": True}
    assert await asyncio.to_thread(entered.wait, 1)
    assert service._pending == 1
    release.set()
    for _ in range(100):
        if service._pending == 0:
            break
        await asyncio.sleep(0.01)

    assert metadata["access_count"] == pytest.approx(1.2)
    assert metadata["retrieved_count"] == 3
    assert metadata["used_count"] == 3
    assert service._pending == 0

    assert await service.handle("memory.apply_access_updates", {"operations": [operation]}) == {"ok": True}
    for _ in range(100):
        if service._pending == 0:
            break
        await asyncio.sleep(0.01)

    assert metadata["access_count"] == pytest.approx(1.4)
    assert metadata["retrieved_count"] == 4
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_close_drains_deferred_access_updates(tmp_path: Path) -> None:
    store = _store()
    metadata: dict[str, object] = {"access_count": 1.0, "last_accessed_at": "2026-08-14T01:00:00+00:00"}
    entered = threading.Event()
    release = threading.Event()

    store._get_by_ids_once.return_value = [Document("doc-1", "hello", metadata=dict(metadata))]

    def update_metadata(_collection: str, _ids: list[str], metadatas: list[dict]) -> bool:
        entered.set()
        release.wait(timeout=2)
        metadata.update(metadatas[0])
        return True

    store._update_metadata_once.side_effect = update_metadata
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)
    operation = {
        "collection": "sakura_knowledge",
        "doc_id": "doc-1",
        "access_delta": 0.2,
        "retrieved_delta": 1,
        "used_delta": 0,
        "last_accessed_at": "2026-08-14T00:00:00+00:00",
    }

    assert await service.handle("memory.apply_access_updates", {"operations": [operation]}) == {"ok": True}
    assert await asyncio.to_thread(entered.wait, 1)
    close_task = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    release.set()
    await close_task

    assert metadata["access_count"] == pytest.approx(1.2)
    assert metadata["last_accessed_at"] == "2026-08-14T01:00:00+00:00"
    store.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [True, False])
async def test_memory_service_unavailable_is_not_an_empty_success(tmp_path: Path, fenced: bool) -> None:
    opener = (lambda: _store()) if fenced else MagicMock(side_effect=RuntimeError("broken DB"))
    service = MemoryService(
        "sakura",
        tmp_path / "sakura",
        opener=opener,
        repair_fenced=lambda: fenced,
    )

    with pytest.raises(MemoryServiceUnavailable):
        await service.handle("memory.list_collections_checked", {})
    await service.close()


@pytest.mark.asyncio
async def test_phase3_memory_does_not_use_cross_process_repair_fence(tmp_path: Path) -> None:
    from core.memory.rag import repair_state

    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    repair_state.write_repair_request_state(
        "sakura",
        reason="hnsw_corruption",
        collection=None,
        source="test",
        include_shared=False,
        animas_dir=anima_dir.parent,
    )
    service = MemoryService("sakura", anima_dir, opener=_store)

    assert await service.handle("memory.list_collections_checked", {}) == {"collections": ["sakura_knowledge"]}
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_queue_overflow_is_unavailable(tmp_path: Path) -> None:
    store = _store()
    entered = threading.Event()
    release = threading.Event()

    def blocked_upsert(_collection: str, _documents: list[Document]) -> bool:
        entered.set()
        release.wait(timeout=2)
        return True

    store._upsert_once.side_effect = blocked_upsert
    service = MemoryService("sakura", tmp_path / "sakura", queue_limit=1, opener=lambda: store)
    params = {
        "collection": "sakura_knowledge",
        "documents": [{"id": "doc-1", "content": "hello", "embedding": [0.1], "metadata": {}}],
    }
    first = asyncio.create_task(service.handle("memory.upsert", params))
    await asyncio.to_thread(entered.wait, 1)

    with pytest.raises(MemoryServiceUnavailable, match="queue is full"):
        await service.handle("memory.upsert", params)

    release.set()
    assert await first == {"ok": True}
    await service.close()


@pytest.mark.asyncio
async def test_memory_service_serializes_parallel_writes(tmp_path: Path) -> None:
    store = _store()
    first_entered = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def ordered_upsert(_collection: str, documents: list[Document]) -> bool:
        order.append(documents[0].id)
        if documents[0].id == "first":
            first_entered.set()
            release_first.wait(timeout=2)
        return True

    store._upsert_once.side_effect = ordered_upsert
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)

    def params(doc_id: str) -> dict:
        return {
            "collection": "sakura_knowledge",
            "documents": [{"id": doc_id, "content": doc_id, "embedding": [0.1], "metadata": {}}],
        }

    first = asyncio.create_task(service.handle("memory.upsert", params("first")))
    await asyncio.to_thread(first_entered.wait, 1)
    second = asyncio.create_task(service.handle("memory.upsert", params("second")))
    await asyncio.sleep(0)
    release_first.set()

    assert await asyncio.gather(first, second) == [{"ok": True}, {"ok": True}]
    assert order == ["first", "second"]
    await service.close()


@pytest.mark.asyncio
async def test_memory_ipc_round_trip_and_checked_unavailable(tmp_path: Path) -> None:
    native = _store()
    written: dict[str, Document] = {}

    def upsert(_collection: str, documents: list[Document]) -> bool:
        written.update((document.id, document) for document in documents)
        return True

    def query(_collection: str, _embedding: list[float], _top_k: int, _where) -> list[SearchResult]:
        return [SearchResult(document, 1.0) for document in written.values()]

    native._upsert_once.side_effect = upsert
    native._query_once.side_effect = query
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: native)
    identity = IPCV2Identity(
        job_id="job-memory",
        root_epoch=str(uuid.uuid4()),
        attempt=1,
        lane="cron",
        display_lane="background",
    )
    server_state = IPCV2ConnectionState(identity)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = IPCV2Connection(reader, writer, server_state)
        try:
            for _ in range(3):
                request = await connection.receive()
                try:
                    result = await service.handle(request.body["method"], request.body["params"])
                except MemoryServiceUnavailable as exc:
                    await connection.send_response(
                        request.body["request_id"],
                        error=ipc_v2_error("UNAVAILABLE", str(exc), retryable=True),
                    )
                else:
                    await connection.send_response(request.body["request_id"], result=result)
        finally:
            await connection.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    connection = IPCV2Connection(reader, writer, IPCV2ConnectionState(identity))
    rpc = _MemoryRpcClient(connection)
    store = IpcVectorStore("http://vector.invalid", "sakura", rpc.request)

    async def receive_responses() -> None:
        for _ in range(3):
            assert rpc.accept_response(await connection.receive())

    receiver = asyncio.create_task(receive_responses())
    assert await asyncio.to_thread(
        store.upsert,
        "sakura_knowledge",
        [Document("written", "through root", embedding=[0.1], metadata={"kind": "knowledge"})],
    )
    results = await asyncio.to_thread(store.query, "sakura_knowledge", [0.1])
    assert results[0].document.id == "written"

    service._repair_fenced = lambda: True
    assert await asyncio.to_thread(store.list_collections_checked) is None
    await receiver

    rpc.close()
    await connection.close()
    server.close()
    await server.wait_closed()
    await service.close()


def test_ipc_vector_store_routes_writes_to_root() -> None:
    requester = MagicMock(return_value={"ok": True})
    store = IpcVectorStore("http://vector.invalid", "sakura", requester)

    assert store.create_collection("sakura_knowledge") is True
    requester.assert_called_once_with("memory.create_collection", {"collection": "sakura_knowledge"})
    assert store._client is None


def test_ipc_vector_store_marks_unavailable_write_as_transient() -> None:
    responses = iter([RuntimeError("root busy"), {"ok": True}])

    def requester(_method: str, _params: dict) -> dict:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    store = IpcVectorStore("http://vector.invalid", "sakura", requester)

    assert store.create_collection("sakura_knowledge") is False
    assert store.is_transient_write_failure("sakura_knowledge") is True
    assert store.create_collection("sakura_knowledge") is True
    assert store.is_transient_write_failure("sakura_knowledge") is False


def test_ipc_vector_store_collection_existence_is_three_state() -> None:
    available = IpcVectorStore(
        "http://vector.invalid",
        "sakura",
        lambda _method, _params: {"collections": ["sakura_knowledge"]},
    )
    unavailable = IpcVectorStore(
        "http://vector.invalid",
        "sakura",
        MagicMock(side_effect=RuntimeError("root down")),
    )

    assert available.collection_exists("sakura_knowledge") is CollectionExistence.EXISTS
    assert available.collection_exists("missing") is CollectionExistence.MISSING
    assert unavailable.collection_exists("sakura_knowledge") is CollectionExistence.UNAVAILABLE


@pytest.mark.asyncio
async def test_root_repair_builds_staging_in_subprocess(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def create_subprocess(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        result_path = Path(cmd[cmd.index("--result") + 1])
        staging = Path(cmd[cmd.index("--staging") + 1])
        staging.mkdir()
        result_path.write_text('{"chunks": 2, "shared_hashes": {}}', encoding="utf-8")
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    service = MemoryService("sakura", anima_dir, opener=_store)

    staging, chunks, hashes = await service._build_staging_subprocess(True)

    cmd = captured["cmd"]
    assert isinstance(cmd, tuple)
    assert cmd[1:3] == ("-m", "core.memory.rag.repair_rebuild")
    assert "--shared" in cmd
    assert staging.name.startswith("vectordb.staging-root-")
    assert chunks == 2
    assert hashes == {}
    shutil.rmtree(staging)
    await service.close()


@pytest.mark.asyncio
async def test_root_repair_swaps_reopens_and_queries(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    live = anima_dir / "vectordb"
    live.mkdir()
    (live / "old.bin").write_text("old", encoding="utf-8")
    staging = anima_dir / "vectordb.staging-test"
    staging.mkdir()
    (staging / "new.bin").write_text("new", encoding="utf-8")
    old_store = _store()
    reopened = _store()
    reopened.verify_rebuilt_data.return_value = {"collections": 1, "chunks": 1, "query_results": 1}
    service = MemoryService("sakura", anima_dir, opener=MagicMock(side_effect=[old_store, reopened]))
    service._build_staging_subprocess = AsyncMock(return_value=(staging, 1, {}))  # type: ignore[method-assign]
    monkeypatch.setattr(service, "_rebuild_bm25_sync", lambda: None)

    await service.start()
    result = await service.repair(include_shared=False)

    assert result["ok"] is True
    assert (live / "new.bin").read_text(encoding="utf-8") == "new"
    assert list((anima_dir / "archive").glob("vectordb-corrupt-*/old.bin"))
    old_store.close.assert_called_once()
    reopened.verify_rebuilt_data.assert_called_once_with(expected_chunks=1)
    await service.close()


@pytest.mark.asyncio
async def test_root_repair_verification_failure_rolls_back_vector_and_bm25(tmp_path: Path, monkeypatch) -> None:
    from core.memory.bm25 import longterm_bm25_index_path

    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    live = anima_dir / "vectordb"
    live.mkdir()
    (live / "old.bin").write_text("old", encoding="utf-8")
    bm25 = longterm_bm25_index_path(anima_dir)
    bm25.write_text("old-bm25", encoding="utf-8")
    staging = anima_dir / "vectordb.staging-test"
    staging.mkdir()
    (staging / "new.bin").write_text("new", encoding="utf-8")
    reopened = _store()
    reopened.verify_rebuilt_data.side_effect = RuntimeError("query failed")
    service = MemoryService(
        "sakura",
        anima_dir,
        opener=MagicMock(side_effect=[_store(), reopened, _store()]),
    )
    service._build_staging_subprocess = AsyncMock(return_value=(staging, 1, {}))  # type: ignore[method-assign]
    monkeypatch.setattr(service, "_rebuild_bm25_sync", lambda: bm25.write_text("new-bm25", encoding="utf-8"))

    await service.start()
    with pytest.raises(RuntimeError, match="query failed"):
        await service.repair(include_shared=False)

    assert (live / "old.bin").read_text(encoding="utf-8") == "old"
    assert bm25.read_text(encoding="utf-8") == "old-bm25"
    assert list((anima_dir / "archive").glob("vectordb-rebuild-failed-*/new.bin"))
    await service.close()


@pytest.mark.asyncio
async def test_root_repair_swap_failure_reopens_untouched_store(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    live = anima_dir / "vectordb"
    live.mkdir()
    (live / "old.bin").write_text("old", encoding="utf-8")
    missing_staging = anima_dir / "vectordb.staging-missing"
    reopened = _store()
    service = MemoryService(
        "sakura",
        anima_dir,
        opener=MagicMock(side_effect=[_store(), reopened]),
    )
    service._build_staging_subprocess = AsyncMock(return_value=(missing_staging, 1, {}))  # type: ignore[method-assign]
    monkeypatch.setattr(service, "_rebuild_bm25_sync", lambda: None)

    await service.start()
    with pytest.raises(FileNotFoundError):
        await service.repair(include_shared=False)

    assert (live / "old.bin").read_text(encoding="utf-8") == "old"
    assert await service.handle("memory.list_collections_checked", {}) == {"collections": ["sakura_knowledge"]}
    await service.close()


@pytest.mark.asyncio
async def test_root_memory_is_explicitly_unavailable_during_repair(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    service = MemoryService("sakura", tmp_path / "sakura", opener=_store)

    async def build(_include_shared: bool):
        entered.set()
        await release.wait()
        raise RuntimeError("stop test repair")

    service._build_staging_subprocess = build  # type: ignore[method-assign]
    repair = asyncio.create_task(service.repair(include_shared=False))
    await entered.wait()

    with pytest.raises(MemoryServiceUnavailable, match="repair in progress"):
        await service.handle("memory.list_collections_checked", {})

    release.set()
    with pytest.raises(RuntimeError, match="stop test repair"):
        await repair
    await service.close()


@pytest.mark.asyncio
async def test_root_open_failure_marks_background_repair_without_failing_startup(tmp_path: Path) -> None:
    from core.memory.rag import repair_state

    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    service = MemoryService("sakura", anima_dir, opener=MagicMock(side_effect=RuntimeError("broken DB")))

    await service.start()

    state = repair_state.read_state("sakura", animas_dir=anima_dir.parent)
    assert state["status"] == "requested"
    assert state["reason"] == "store_init_failed"
    assert state["source"] == "phase3_root_startup"
    await service.close()


async def test_repeated_collection_initialization_is_idempotent_through_root(tmp_path: Path) -> None:
    from core.memory.rag.store import ChromaVectorStore

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    store.client.create_collection.side_effect = RuntimeError("Collection [sakura_knowledge] already exists")
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)
    try:
        assert await service.handle("memory.create_collection", {"collection": "sakura_knowledge"}) == {"ok": True}
        # Genuine failures still surface; only the already-existing case is benign.
        store.client.create_collection.side_effect = RuntimeError("disk unavailable")
        with pytest.raises(MemoryServiceUnavailable, match="disk unavailable"):
            await service.handle("memory.create_collection", {"collection": "sakura_knowledge"})
    finally:
        await service.close()


async def test_metadata_recall_before_initial_indexing_is_empty_through_root(tmp_path: Path) -> None:
    from core.memory.rag.store import ChromaVectorStore

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    store.client.get_collection.side_effect = RuntimeError("Collection [shared_common_knowledge] does not exist")
    service = MemoryService("sakura", tmp_path / "sakura", opener=lambda: store)
    params = {"collection": "shared_common_knowledge", "where": {}}
    try:
        assert await service.handle("memory.get_by_metadata", params) == {"results": []}
        store.client.get_collection.side_effect = RuntimeError("disk unavailable")
        with pytest.raises(MemoryServiceUnavailable, match="disk unavailable"):
            await service.handle("memory.get_by_metadata", params)
    finally:
        await service.close()
