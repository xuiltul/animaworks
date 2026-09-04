"""Phase 3 cascade-isolation and recovery acceptance tests."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil
import pytest

from core.memory.bm25 import rebuild_longterm_bm25_index
from core.memory.rag.sqlite_health import quick_check_chroma_sqlite
from core.memory.rag_search import RAGMemorySearch
from core.memory.task_queue import TaskQueueManager
from core.platform.processing_lease import write_processing_lease
from core.schemas import CronTask
from core.supervisor.memory_service import MemoryService
from core.supervisor.process_handle import ProcessHandle
from core.supervisor.task_runner_supervisor import TaskRunnerSupervisor

pytestmark = [
    pytest.mark.timeout(90),
    pytest.mark.skipif(os.name != "posix", reason="process fault injection requires POSIX signals"),
]

logger = logging.getLogger(__name__)
_VECTOR = [1.0, 0.0, 0.0]


class _EmbeddingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _EmbeddingHandler)
        self.repair_started = threading.Event()
        self.release_repair = threading.Event()
        self.release_repair.set()

    def arm_repair_gate(self) -> None:
        self.repair_started.clear()
        self.release_repair.clear()


class _EmbeddingHandler(BaseHTTPRequestHandler):
    server: _EmbeddingServer

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        texts = payload.get("texts", [])
        self.server.repair_started.set()
        if not self.server.release_repair.wait(timeout=30):
            self.send_error(504)
            return
        body = json.dumps({"embeddings": [_VECTOR for _ in texts]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_embed_url() -> tuple[str, _EmbeddingServer]:
    server = _EmbeddingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/internal/embed", server
    finally:
        server.release_repair.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def phase3_animas(
    data_dir: Path,
    make_anima: Callable[..., Path],
) -> dict[str, Path]:
    animas = {name: make_anima(name) for name in ("cascade-a", "cascade-b", "kill-root", "drain-root")}
    for anima_dir in animas.values():
        status_path = anima_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            {
                "process_model": "phase3",
                "heartbeat_enabled": False,
                "consolidation_enabled": False,
            }
        )
        status_path.write_text(json.dumps(status), encoding="utf-8")
    return animas


def _root(data_dir: Path, name: str, embed_url: str) -> ProcessHandle:
    return ProcessHandle(
        anima_name=name,
        socket_path=data_dir / "run" / "sockets" / f"{name}.sock",
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        log_dir=data_dir / "logs",
        child_env_urls={
            "ANIMAWORKS_EMBED_URL": embed_url,
            "ANIMAWORKS_RERANK_URL": "http://127.0.0.1:9/api/internal/rerank",
        },
        startup_ready_timeout=30,
    )


async def _memory(handle: ProcessHandle, method: str, params: dict[str, Any]) -> Any:
    return await handle.send_request("memory", {"method": method, "params": params}, timeout=30)


async def _seed(handle: ProcessHandle, content: str) -> None:
    params = {
        "collection": f"{handle.anima_name}_knowledge",
        "documents": [
            {
                "id": "sentinel",
                "content": content,
                "embedding": _VECTOR,
                "metadata": {"source_file": "knowledge/sentinel.md", "memory_type": "knowledge"},
            }
        ],
    }
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        response = await _memory(handle, "memory.upsert", params)
        if response.error is None:
            assert response.result == {"ok": True}
            return
        if asyncio.get_running_loop().time() >= deadline or "not open" not in response.error["message"]:
            raise AssertionError(response.error)
        await asyncio.sleep(0.05)


async def _query(handle: ProcessHandle) -> Any:
    return await _memory(
        handle,
        "memory.query",
        {
            "collection": f"{handle.anima_name}_knowledge",
            "embedding": _VECTOR,
            "top_k": 10,
        },
    )


async def _query_ready(handle: ProcessHandle) -> Any:
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        response = await _query(handle)
        if response.error is None:
            return response
        if asyncio.get_running_loop().time() >= deadline or "not open" not in response.error["message"]:
            return response
        await asyncio.sleep(0.05)


async def _wait_for(probe: Callable[[], Any], timeout: float = 10) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = probe()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting for phase3 condition")


def _record_event(path: Path, event: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _events(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


async def _drain_root_harness(anima_dir: Path, event_path: Path) -> int:
    service = MemoryService(anima_dir.name, anima_dir)
    stopping = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
    try:
        await service.start()
        await service.handle(
            "memory.upsert",
            {
                "collection": f"{anima_dir.name}_knowledge",
                "documents": [
                    {
                        "id": "sentinel",
                        "content": "drain durable sentinel",
                        "embedding": _VECTOR,
                        "metadata": {"source_file": "knowledge/sentinel.md", "memory_type": "knowledge"},
                    }
                ],
            },
        )
        _record_event(event_path, "ready")
        await stopping.wait()
    finally:
        await service.close()
    _record_event(event_path, "closed")
    return 0


def _db_snapshot(anima_dir: Path) -> tuple[frozenset[str], int, str]:
    persist_dir = anima_dir / "vectordb"
    health = quick_check_chroma_sqlite(persist_dir)
    assert health.status == "ok"
    with sqlite3.connect(persist_dir / "chroma.sqlite3") as connection:
        collections = frozenset(row[0] for row in connection.execute("SELECT name FROM collections"))
        chunks = int(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0])
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert journal_mode == "wal"
    return collections, chunks, journal_mode


def _bm25_fallback(anima_dir: Path, query: str) -> list[dict[str, Any]]:
    search = RAGMemorySearch(
        anima_dir,
        anima_dir.parents[1] / "common_knowledge",
        anima_dir.parents[1] / "common_skills",
    )
    return search._keyword_search_fallback(
        query,
        "knowledge",
        0,
        knowledge_dir=anima_dir / "knowledge",
        episodes_dir=anima_dir / "episodes",
        procedures_dir=anima_dir / "procedures",
        common_knowledge_dir=anima_dir.parents[1] / "common_knowledge",
    )


@pytest.mark.asyncio
async def test_corruption_isolated_and_reads_continue_during_repair(
    data_dir: Path,
    phase3_animas: dict[str, Path],
    fake_embed_url: tuple[str, _EmbeddingServer],
) -> None:
    embed_url, embed_server = fake_embed_url
    root_a = _root(data_dir, "cascade-a", embed_url)
    root_b = _root(data_dir, "cascade-b", embed_url)
    await asyncio.gather(root_a.start(), root_b.start())
    try:
        await _seed(root_a, "cascade-a original vector")
        await _seed(root_b, "cascade-b remains healthy")
        assert (await _query(root_a)).result["results"][0]["document"]["content"] == "cascade-a original vector"
        assert (await _query(root_b)).result["results"][0]["document"]["content"] == "cascade-b remains healthy"

        anima_a = phase3_animas["cascade-a"]
        fallback_text = "# Phase 3 survival\n\n" + ("phase3 bm25 survival sentinel remains searchable. " * 40)
        (anima_a / "knowledge" / "sentinel.md").write_text(fallback_text, encoding="utf-8")
        rebuild_longterm_bm25_index(anima_a)

        await root_a.kill()
        db_path = anima_a / "vectordb" / "chroma.sqlite3"
        db_path.write_bytes(b"deliberate phase3 corruption")
        db_path.with_name(f"{db_path.name}-wal").unlink(missing_ok=True)
        db_path.with_name(f"{db_path.name}-shm").unlink(missing_ok=True)
        await root_a.start()

        unavailable = await _query(root_a)
        assert unavailable.error and "unavailable" in unavailable.error["message"].lower()
        fallback = _bm25_fallback(anima_a, "bm25 survival sentinel")
        assert fallback and "phase3 bm25 survival sentinel" in fallback[0]["content"]
        assert fallback[0]["search_method"] == "bm25"

        await _seed(root_b, "cascade-b indexed while sibling corrupt")
        healthy = await _query(root_b)
        assert healthy.error is None
        assert healthy.result["results"][0]["document"]["content"] == "cascade-b indexed while sibling corrupt"

        embed_server.arm_repair_gate()
        repairing = asyncio.create_task(root_a.send_request("repair_memory", {"include_shared": False}, timeout=60))
        assert await asyncio.to_thread(embed_server.repair_started.wait, 10)
        during_repair = await _query(root_a)
        assert during_repair.error and "repair in progress" in during_repair.error["message"].lower()
        assert _bm25_fallback(anima_a, "bm25 survival sentinel")[0]["search_method"] == "bm25"

        embed_server.release_repair.set()
        repaired = await repairing
        assert repaired.error is None and repaired.result["status"] == "success"
        repaired_query = await _query(root_a)
        assert repaired_query.error is None
        assert "phase3 bm25 survival sentinel" in repaired_query.result["results"][0]["document"]["content"]
        assert (await _query(root_b)).error is None
    finally:
        embed_server.release_repair.set()
        await asyncio.gather(root_a.stop(drain_streams=False), root_b.stop(drain_streams=False), return_exceptions=True)


@pytest.mark.asyncio
async def test_root_sigkill_respawn_preserves_db_and_recovers_lease(
    data_dir: Path,
    phase3_animas: dict[str, Path],
    fake_embed_url: tuple[str, _EmbeddingServer],
) -> None:
    embed_url, _ = fake_embed_url
    anima_dir = phase3_animas["kill-root"]
    root = _root(data_dir, "kill-root", embed_url)
    await root.start()
    try:
        await _seed(root, "sigkill durable sentinel")
        before = _db_snapshot(anima_dir)
        assert before[0] == {"kill-root_knowledge"} and before[1] == 1
        killed_pid = root.process.pid
        killed_create_time = psutil.Process(killed_pid).create_time()
        await root.kill()

        queue = TaskQueueManager(anima_dir)
        entry = queue.add_task(
            source="anima",
            original_instruction="recover after root SIGKILL",
            assignee="kill-root",
            summary="interrupted phase3 task",
            status="in_progress",
        )
        processing = anima_dir / "state" / "background_tasks" / "pending" / "processing"
        failed = processing.parent / "failed"
        processing.mkdir(parents=True, exist_ok=True)
        descriptor = processing / f"{entry.task_id}.json"
        descriptor.write_text(json.dumps({"task_id": entry.task_id}), encoding="utf-8")
        write_processing_lease(
            descriptor,
            anima="kill-root",
            task_id=entry.task_id,
            pid=killed_pid,
            job_id="killed-job",
            task_pid=killed_pid,
            pgid=killed_pid,
            root_epoch="killed-root-epoch",
            attempt=1,
            process_start_time=killed_create_time,
        )

        await root.start()
        await _wait_for(lambda: not descriptor.exists())
        assert not (failed / descriptor.name).exists()
        assert _db_snapshot(anima_dir) == before
        query = await _query_ready(root)
        assert query.error is None
        assert query.result["results"][0]["document"]["content"] == "sigkill durable sentinel"
        recovered = queue.get_task_by_id(entry.task_id)
        assert recovered.status == "pending"
        assert recovered.summary == "interrupted phase3 task"
        assert "INTERRUPTED" in recovered.meta["last_run_note"]
        assert recovered.meta["last_run_stop_kind"] == "crash"
    finally:
        await root.stop(drain_streams=False)


@pytest.mark.asyncio
async def test_root_sigterm_clean_close_is_bounded(
    data_dir: Path,
    phase3_animas: dict[str, Path],
) -> None:
    anima_dir = phase3_animas["drain-root"]
    event_path = data_dir / "drain-events.jsonl"
    env = {
        **os.environ,
        "ANIMAWORKS_DATA_DIR": str(data_dir),
        "ANIMAWORKS_PHASE3_DRAIN_HARNESS": str(anima_dir),
        "ANIMAWORKS_PHASE3_DRAIN_EVENTS": str(event_path),
    }
    root = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__)), env=env, start_new_session=True)
    try:
        ready = await _wait_for(
            lambda: next((event for event in _events(event_path) if event["event"] == "ready"), None),
            timeout=15,
        )
        assert ready["pid"] == root.pid
        started = time.monotonic()
        os.kill(root.pid, signal.SIGTERM)
        returncode = await asyncio.wait_for(root.wait(), timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if root.returncode is None:
            os.killpg(root.pid, signal.SIGKILL)
            await root.wait()

    logger.info("phase3 root drain_seconds=%.3f", elapsed)
    assert returncode == 0
    assert elapsed < 10
    assert [event["event"] for event in _events(event_path)] == ["ready", "closed"]
    assert _db_snapshot(anima_dir) == ({"drain-root_knowledge"}, 1, "wal")


@pytest.mark.asyncio
async def test_phase3_task_runner_process_smoke(
    data_dir: Path,
    phase3_animas: dict[str, Path],
    fake_embed_url: tuple[str, _EmbeddingServer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed_url, _ = fake_embed_url
    monkeypatch.setenv("ANIMAWORKS_EMBED_URL", embed_url)
    supervisor = TaskRunnerSupervisor(
        "cascade-a",
        phase3_animas["cascade-a"],
        data_dir / "shared",
        memory_via_root=True,
    )
    try:
        result = await supervisor.run_cron(
            CronTask(
                name="phase3-process-smoke",
                schedule="0 0 1 1 *",
                type="command",
                command="printf phase3-task-runner-ok",
                trigger_heartbeat=False,
            )
        )
        assert result["success"] is True
        assert result["result"]["stdout"] == "phase3-task-runner-ok"
    finally:
        await supervisor.close()


def test_direct_chroma_construction_stays_inside_approved_boundaries() -> None:
    repo = Path(__file__).resolve().parents[3]
    boundary_allowlist = {
        "constructor": {"core/memory/rag/store.py"},
        "root": {"core/supervisor/memory_service.py"},
        "staging": {"core/memory/rag/repair_rebuild.py"},
        "worker": {"core/memory/rag/vector_worker.py"},
        "direct_env": {
            "core/memory/rag/repair_rebuild.py",
            "core/memory/rag/repair_service.py",
            "core/memory/rag/vector_worker_client.py",
        },
    }
    constructors: set[str] = set()
    direct_access_enablers: set[str] = set()
    direct_env_writers: set[str] = set()

    for path in (repo / "core").rglob("*.py"):
        relative = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        chromadb_aliases = {"chromadb"}
        persistent_aliases: set[str] = set()
        enable_aliases = {"enable_direct_chroma_for_process"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                chromadb_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "chromadb")
            elif isinstance(node, ast.ImportFrom) and node.module == "chromadb":
                persistent_aliases.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "PersistentClient"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "core.memory.rag.direct_access":
                enable_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "enable_direct_chroma_for_process"
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                named_constructor = isinstance(node.func, ast.Name) and node.func.id in persistent_aliases
                qualified_constructor = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "PersistentClient"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in chromadb_aliases
                )
                if named_constructor or qualified_constructor:
                    constructors.add(relative)
                elif isinstance(node.func, ast.Name) and node.func.id in enable_aliases:
                    direct_access_enablers.add(relative)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "ANIMAWORKS_ALLOW_DIRECT_CHROMA"
            ):
                direct_env_writers.add(relative)

    assert constructors == boundary_allowlist["constructor"]
    assert direct_access_enablers == boundary_allowlist["root"] | boundary_allowlist["worker"]
    assert direct_env_writers == boundary_allowlist["direct_env"]
    assert boundary_allowlist["staging"] <= direct_env_writers


if __name__ == "__main__" and os.environ.get("ANIMAWORKS_PHASE3_DRAIN_HARNESS"):
    raise SystemExit(
        asyncio.run(
            _drain_root_harness(
                Path(os.environ["ANIMAWORKS_PHASE3_DRAIN_HARNESS"]),
                Path(os.environ["ANIMAWORKS_PHASE3_DRAIN_EVENTS"]),
            )
        )
    )
