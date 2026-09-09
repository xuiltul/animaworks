from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core import startup_progress
from core.auth.models import AuthConfig

_LOCAL_TRUST_AUTH = AuthConfig(auth_mode="local_trust")


@pytest.fixture(autouse=True)
def _reset_startup_progress():
    startup_progress._reset_for_testing()
    yield
    startup_progress._reset_for_testing()


def _make_app(data_dir: Path):
    from core.config.models import AnimaWorksConfig, invalidate_cache, save_config
    from server.app import create_app

    config = AnimaWorksConfig(setup_complete=True)
    save_config(config, data_dir / "config.json")
    invalidate_cache()

    animas_dir = data_dir / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    with (
        patch("server.app.load_config", return_value=config),
        patch("server.app.ProcessSupervisor") as mock_supervisor_cls,
    ):
        supervisor = MagicMock()
        supervisor.get_process_status.return_value = {"status": "stopped"}
        mock_supervisor_cls.return_value = supervisor
        return create_app(animas_dir, shared_dir)


@pytest.mark.asyncio
async def test_startup_gate_returns_html_for_browser_requests(data_dir: Path):
    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("indexing", detail="sora/knowledge/topic.md", done_count=2, total_count=5)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/", headers={"accept": "text/html"})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "5"
    assert "text/html" in resp.headers["content-type"]
    assert "sora/knowledge/topic.md" in resp.text
    assert "2/5" in resp.text


@pytest.mark.asyncio
async def test_startup_gate_returns_json_for_api_requests(data_dir: Path):
    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("repairing", detail="sora", done_count=1, total_count=3)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/animas", headers={"accept": "application/json"})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "5"
    data = resp.json()
    assert data["status"] == "starting"
    assert data["phase"] == "repairing"
    assert data["detail"] == "sora"
    assert data["progress"]["done_count"] == 1
    assert data["progress"]["total_count"] == 3


@pytest.mark.asyncio
async def test_startup_status_returns_progress_snapshot(data_dir: Path):
    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("preflight", detail="checking vector DBs")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/startup-status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "starting"
    assert data["phase"] == "preflight"
    assert data["detail"] == "checking vector DBs"


@pytest.mark.asyncio
async def test_startup_ready_snapshot_includes_ready_at(data_dir: Path):
    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("ready", detail="ready")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/startup-status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert isinstance(data["ready_at"], float)


@pytest.mark.asyncio
async def test_startup_gate_allows_normal_routes_after_ready(data_dir: Path):
    app = _make_app(data_dir)
    startup_progress.set_phase("ready")

    transport = ASGITransport(app=app)
    with patch("server.app.load_auth", return_value=_LOCAL_TRUST_AUTH):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_spawning_workers_can_embed_before_public_readiness(data_dir: Path):
    app = _make_app(data_dir)
    app.state.worker_services_ready = True
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("spawning_animas")

    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    with (
        patch("server.app.load_auth", return_value=_LOCAL_TRUST_AUTH),
        patch("core.memory.rag.singleton.thread_safe_encode", return_value=[[0.1, 0.2]]) as encode,
    ):
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            resp = await client.post("/api/internal/embed", json={"texts": ["memory"]})
            public = await client.get("/api/animas")

    assert resp.status_code == 200
    assert resp.json() == {"embeddings": [[0.1, 0.2]]}
    encode.assert_called_once_with(["memory"], purpose="document", priority="interactive")
    assert public.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["starting", "preflight", "repairing", "indexing", "spawning_animas", "failed"])
async def test_internal_endpoints_remain_gated_before_worker_startup(data_dir: Path, phase: str):
    app = _make_app(data_dir)
    assert app.state.worker_services_ready is False
    startup_progress.begin_startup("booting")
    startup_progress.set_phase(phase)
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    with patch("core.memory.rag.singleton.thread_safe_encode") as encode:
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            resp = await client.post("/api/internal/embed", json={"texts": ["memory"]})
    assert resp.status_code == 503
    encode.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("peer", "base_url", "headers"),
    [
        ("203.0.113.1", "http://127.0.0.1", {}),
        ("127.0.0.1", "http://attacker.example", {}),
        ("127.0.0.1", "http://127.0.0.1", {"origin": "https://attacker.example"}),
    ],
)
async def test_spawning_internal_exception_requires_safe_local_request(
    data_dir: Path, peer: str, base_url: str, headers: dict[str, str]
):
    app = _make_app(data_dir)
    app.state.worker_services_ready = True
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("spawning_animas")
    transport = ASGITransport(app=app, client=(peer, 12345))
    with patch("core.memory.rag.singleton.thread_safe_encode") as encode:
        async with AsyncClient(transport=transport, base_url=base_url) as client:
            resp = await client.post("/api/internal/embed", json={"texts": ["memory"]}, headers=headers)
    assert resp.status_code == 503
    encode.assert_not_called()


@pytest.mark.asyncio
async def test_spawning_internal_exception_preserves_authentication(data_dir: Path):
    app = _make_app(data_dir)
    app.state.worker_services_ready = True
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("spawning_animas")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    auth = AuthConfig(auth_mode="password", trust_localhost=False)
    with (
        patch("server.app.load_auth", return_value=auth),
        patch("core.memory.rag.singleton.thread_safe_encode") as encode,
    ):
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            resp = await client.post("/api/internal/embed", json={"texts": ["memory"]})
    assert resp.status_code == 401
    encode.assert_not_called()


@pytest.mark.asyncio
async def test_catchup_indexing_cannot_close_ready_worker_services(data_dir: Path):
    from core.memory.rag.indexer import MemoryIndexer
    from server.app import _run_startup_initialization

    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    empty = data_dir / "empty-knowledge"
    empty.mkdir()
    # Exercise the real directory-indexing progress updates without opening a
    # vector DB, loading an embedding model, or accessing runtime memory.
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.collection_prefix = "shared"

    def preflight(**_kwargs):
        assert app.state.worker_services_ready is False

    app.state.startup_preflight_runner = preflight

    async def spawn_and_catchup(_app, **_kwargs):
        assert app.state.worker_services_ready is True
        assert startup_progress.snapshot()["phase"] == "spawning_animas"
        transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            before = await client.post("/api/internal/embed", json={"texts": ["memory"]})
            await asyncio.to_thread(indexer.index_directory, empty, "knowledge")
            assert startup_progress.snapshot()["phase"] == "indexing"
            after = await client.post("/api/internal/embed", json={"texts": ["memory"]})
            public = await client.get("/api/animas")
        assert before.status_code == after.status_code == 200
        assert public.status_code == 503

    with (
        patch("server.app._prepare_startup_vector_worker", new_callable=AsyncMock),
        patch("server.app._startup_animas_background", side_effect=spawn_and_catchup) as spawn,
        patch("server.app._start_usage_governor_if_enabled", new_callable=AsyncMock),
        patch("server.app.load_auth", return_value=_LOCAL_TRUST_AUTH),
        patch("core.memory.rag.singleton.thread_safe_encode", return_value=[[0.1, 0.2]]) as encode,
    ):
        await _run_startup_initialization(app)
    spawn.assert_awaited_once()
    assert encode.call_count == 2
    assert startup_progress.snapshot()["phase"] == "ready"
    assert app.state.worker_services_ready is True


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_post_preflight_failure_revokes_worker_service_readiness(data_dir: Path, cancel: bool):
    from server.app import _run_startup_initialization

    app = _make_app(data_dir)
    startup_progress.begin_startup("booting")
    app.state.startup_preflight_runner = lambda **_kwargs: None

    async def fail_after_preflight(_app, **_kwargs):
        assert app.state.worker_services_ready is True
        if cancel:
            raise asyncio.CancelledError
        raise RuntimeError("worker startup failed")

    with (
        patch("server.app._prepare_startup_vector_worker", new_callable=AsyncMock),
        patch("server.app._startup_animas_background", side_effect=fail_after_preflight),
    ):
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await _run_startup_initialization(app)
        else:
            await _run_startup_initialization(app)
            assert startup_progress.snapshot()["phase"] == "failed"
    assert app.state.worker_services_ready is False

    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        response = await client.post("/api/internal/embed", json={"texts": ["memory"]})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_failed_phase_cannot_use_a_stale_worker_ready_flag(data_dir: Path):
    app = _make_app(data_dir)
    app.state.worker_services_ready = True
    startup_progress.begin_startup("booting")
    startup_progress.set_phase("failed")
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    with patch("core.memory.rag.singleton.thread_safe_encode") as encode:
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.post("/api/internal/embed", json={"texts": ["memory"]})
    assert response.status_code == 503
    encode.assert_not_called()


@pytest.mark.asyncio
async def test_startup_initialization_failure_sets_failed_and_server_survives(data_dir: Path):
    from server.app import _run_startup_initialization

    app = _make_app(data_dir)
    app.state.vector_worker = None

    def fail_preflight(*, force_all_vectordb: bool = False) -> None:
        raise RuntimeError("preflight exploded")

    app.state.startup_preflight_runner = fail_preflight
    app.state.force_startup_repair_all_vectordb = False

    startup_progress.begin_startup("booting")
    await _run_startup_initialization(app)

    snapshot = startup_progress.snapshot()
    assert app.state.worker_services_ready is False
    assert snapshot["status"] == "failed"
    assert snapshot["phase"] == "failed"
    assert "preflight exploded" in str(snapshot["error"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/startup-status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
