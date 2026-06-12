from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    assert snapshot["status"] == "failed"
    assert snapshot["phase"] == "failed"
    assert "preflight exploded" in str(snapshot["error"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/startup-status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
