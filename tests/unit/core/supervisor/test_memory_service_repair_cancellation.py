"""Request cancellation waits for native RAG promotion or rollback to finish."""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.memory_service import MemoryService, MemoryServiceUnavailable


@pytest.mark.parametrize("fail_verification", [False, True])
@pytest.mark.parametrize("cancel_twice", [False, True])
async def test_cancelled_repair_keeps_fence_until_consistent_generation(
    tmp_path, monkeypatch, fail_verification, cancel_twice
):
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    anima = tmp_path / "animas" / "synthetic"
    live = anima / "vectordb"
    live.mkdir(parents=True)
    (live / "old.marker").write_text("old")
    (anima / "index_meta.json").write_text('{"generation":"old"}')
    staging = anima / "vectordb.staging-test"
    artifacts = staging / ".rebuild"
    artifacts.mkdir(parents=True)
    (staging / "new.marker").write_text("new")
    (artifacts / "sources.json").write_text(json.dumps({"owner": str(anima.resolve()), "sources": {}}))
    (artifacts / "index_meta.json").write_text('{"generation":"new"}')
    service = MemoryService("synthetic", anima, opener=MagicMock(return_value=MagicMock()))
    await service.start()
    service._build_staging_subprocess = AsyncMock(return_value=(staging, 1, {}))
    monkeypatch.setattr(service, "_rebuild_bm25_sync", lambda: None)
    monkeypatch.setattr(service, "_invalidate_shared_checks_sync", lambda: None)
    verification = MagicMock(return_value={"chunks": 1})
    if fail_verification:
        verification.side_effect = RuntimeError("synthetic verification failure")
    monkeypatch.setattr(service, "_verify_store_sync", verification)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    promote = service._promote_staging_sync

    def delayed_promote(path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return promote(path)

    monkeypatch.setattr(service, "_promote_staging_sync", delayed_promote)
    running = asyncio.create_task(service.repair(include_shared=True))
    try:
        await asyncio.wait_for(started.wait(), 5)
        running.cancel()
        await asyncio.sleep(0)
        if cancel_twice:
            running.cancel()
            await asyncio.sleep(0)
        assert not running.done()
        assert service._repairing is True
        assert service._repair_lock.locked()
        with pytest.raises(MemoryServiceUnavailable, match="repair in progress"):
            await service.handle("memory.list_collections_checked", {})
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
        expected = "old" if fail_verification else "new"
        assert (live / f"{expected}.marker").exists()
        assert json.loads((anima / "index_meta.json").read_text())["generation"] == expected
        assert service._repairing is False
        assert not service._repair_lock.locked()
        assert service._store is not None
        assert service._open_error is None
        assert not list((anima / "state").glob(".rag-repair-metadata-*"))
        verification.assert_called_once()
    finally:
        release.set()
        await service.close()


async def test_child_cancellation_does_not_loop_forever(tmp_path, monkeypatch):
    service = MemoryService("synthetic", tmp_path, opener=MagicMock())

    async def child_cancelled(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "_promote_and_verify_consistent", child_cancelled)
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(service._promote_and_verify(tmp_path / "stage", 0, {}), 1)
    finally:
        await service.close()
