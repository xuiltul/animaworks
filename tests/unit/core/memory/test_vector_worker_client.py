from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.rag.http_store import HttpVectorStore
from core.memory.rag.store import Document
from core.memory.rag.vector_worker_client import VectorWorkerManager, start_temporary_vector_worker


class _ExitedProcess:
    returncode = -11

    def poll(self) -> int:
        return self.returncode


def test_vector_worker_segfault_records_rag_corruption(tmp_path: Path) -> None:
    manager = VectorWorkerManager(
        enabled=True,
        host="127.0.0.1",
        port=0,
        log_dir=tmp_path,
    )
    manager.process = _ExitedProcess()  # type: ignore[assignment]

    with patch("core.memory.rag.repair.record_chroma_error") as record:
        manager._record_crash_if_exited(  # noqa: SLF001
            {"anima_name": "sora", "collection": "sora_knowledge"}
        )

    record.assert_called_once_with(
        anima_name="sora",
        collection="sora_knowledge",
        error=-11,
        source="vector_worker",
    )
    assert manager.process is None
    assert manager.native_crash_detected is True


def test_vector_worker_config_defaults_do_not_direct_fallback(tmp_path: Path) -> None:
    manager = VectorWorkerManager.from_config(
        SimpleNamespace(rag=SimpleNamespace()),
        log_dir=tmp_path,
    )

    assert manager.fallback_direct is False
    assert manager.shutdown_timeout == 30.0


def test_vector_worker_config_reads_shutdown_timeout(tmp_path: Path) -> None:
    manager = VectorWorkerManager.from_config(
        SimpleNamespace(rag=SimpleNamespace(vector_worker_shutdown_timeout_seconds=45.0)),
        log_dir=tmp_path,
    )

    assert manager.shutdown_timeout == 45.0


def test_vector_worker_stop_waits_for_graceful_shutdown(tmp_path: Path) -> None:
    manager = VectorWorkerManager(
        enabled=True,
        host="127.0.0.1",
        port=0,
        log_dir=tmp_path,
        shutdown_timeout=12.0,
    )
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    manager.process = proc
    manager.base_url = "http://127.0.0.1:12345"

    import asyncio

    asyncio.run(manager.stop())

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once_with(timeout=12.0)
    proc.kill.assert_not_called()


def test_vector_worker_stop_kills_after_shutdown_timeout(tmp_path: Path) -> None:
    manager = VectorWorkerManager(
        enabled=True,
        host="127.0.0.1",
        port=0,
        log_dir=tmp_path,
        shutdown_timeout=0.1,
    )
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("vector-worker", 0.1), 0]
    manager.process = proc
    manager.base_url = "http://127.0.0.1:12345"

    import asyncio

    asyncio.run(manager.stop())

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_args_list[0].kwargs == {"timeout": 0.1}
    assert proc.wait.call_args_list[1].kwargs == {"timeout": 5}


def test_vector_worker_subprocess_env_allows_direct_chroma(tmp_path: Path) -> None:
    manager = VectorWorkerManager(
        enabled=True,
        host="127.0.0.1",
        port=12345,
        log_dir=tmp_path,
    )

    async def fake_wait() -> None:
        return None

    with (
        patch("subprocess.Popen") as popen,
        patch.object(manager, "_wait_until_healthy", side_effect=fake_wait),
    ):
        import asyncio

        asyncio.run(manager._start_process())  # noqa: SLF001

    env = popen.call_args.kwargs["env"]
    assert env["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] == "1"
    assert "ANIMAWORKS_VECTOR_URL" not in env


def test_start_temporary_vector_worker_sets_and_restores_vector_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://previous/vector")

    async def fake_ensure_running(self, *, payload=None) -> None:
        self.base_url = "http://127.0.0.1:45678"

    with (
        patch.object(VectorWorkerManager, "_ensure_running", new=fake_ensure_running),
        patch.object(VectorWorkerManager, "stop", new=AsyncMock()),
    ):
        worker = start_temporary_vector_worker(
            config=SimpleNamespace(rag=SimpleNamespace(vector_worker_enabled=True)),
            log_dir=tmp_path,
        )

        assert os.environ["ANIMAWORKS_VECTOR_URL"] == "http://127.0.0.1:45678"
        worker.stop()

    assert os.environ["ANIMAWORKS_VECTOR_URL"] == "http://previous/vector"


def test_http_vector_store_suspends_writes_after_worker_circuit_429(monkeypatch) -> None:
    class FakeResponse:
        status_code = 429
        text = "circuit open"
        headers = {"Retry-After": "60"}

        def raise_for_status(self) -> None:
            import httpx

            raise httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock(status_code=429))

        def json(self) -> dict[str, str]:
            return {"detail": "Vector write circuit breaker open"}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, path, json):
            self.calls += 1
            return FakeResponse()

    client = FakeClient()
    store = HttpVectorStore("http://vector", anima_name="sora")
    store._client = client  # noqa: SLF001

    assert store.upsert("sora_knowledge", [Document(id="d1", content="c", embedding=[0.1])]) is False
    assert store.upsert("sora_knowledge", [Document(id="d2", content="c", embedding=[0.2])]) is False
    assert client.calls == 1
