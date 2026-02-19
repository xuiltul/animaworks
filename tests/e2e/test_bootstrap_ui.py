# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for bootstrap UI feature — anima list and start endpoints.

Tests the API integration points that drive the frontend bootstrap UI:
1. GET /api/animas returns bootstrapping flag per anima
2. POST /api/animas/{name}/start triggers anima startup
3. Start endpoint rejects already-running animas
4. Bootstrapping anima transitions to idle after completion
5. Multiple animas can have independent states simultaneously
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────────


def _create_app(
    tmp_path: Path,
    anima_names: list[str] | None = None,
    supervisor: MagicMock | None = None,
) -> "FastAPI":  # noqa: F821
    """Build a real FastAPI app via create_app with mocked externals.

    Args:
        tmp_path: Temporary directory for anima/shared data.
        anima_names: Override discovered anima names.
        supervisor: Optional pre-configured mock supervisor. When not
            supplied a default mock is created.
    """
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        sup = supervisor or MagicMock()
        if supervisor is None:
            sup.get_all_status.return_value = {}
            sup.get_process_status.return_value = {
                "status": "stopped",
                "pid": None,
                "bootstrapping": False,
            }
        mock_sup_cls.return_value = sup

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


def _create_anima_on_disk(animas_dir: Path, name: str) -> Path:
    """Create a minimal anima directory on disk."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("episodes", "knowledge", "procedures", "state", "shortterm"):
        (anima_dir / subdir).mkdir(exist_ok=True)
    (anima_dir / "identity.md").write_text(
        f"# {name}\nTest anima.", encoding="utf-8",
    )
    (anima_dir / "injection.md").write_text("", encoding="utf-8")
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    return anima_dir


def _make_supervisor_mock(
    statuses: dict[str, dict] | None = None,
) -> MagicMock:
    """Create a mock supervisor with configurable per-anima status.

    Args:
        statuses: Mapping from anima name to the dict returned by
            ``get_process_status(name)``.  Names not in this mapping
            return a default ``not_found`` status.
    """
    statuses = statuses or {}
    sup = MagicMock()
    sup.get_all_status.return_value = statuses

    def _get_process_status(name: str) -> dict:
        return statuses.get(name, {"status": "not_found", "bootstrapping": False})

    sup.get_process_status = MagicMock(side_effect=_get_process_status)
    sup.start_anima = AsyncMock()
    return sup


# ── Tests ────────────────────────────────────────────────────


class TestAnimaListBootstrapIntegration:
    """Test GET /api/animas returns correct bootstrap status."""

    async def test_anima_list_shows_bootstrapping_status(
        self, tmp_path: Path,
    ) -> None:
        """When supervisor reports an anima as bootstrapping, the
        GET /api/animas response includes ``bootstrapping: True``."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "bootstrapping",
                    "pid": 12345,
                    "uptime_sec": 5.0,
                    "bootstrapping": True,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        alice = data[0]
        assert alice["name"] == "alice"
        assert alice["bootstrapping"] is True
        assert alice["status"] == "bootstrapping"

    async def test_anima_start_triggers_bootstrap(
        self, tmp_path: Path,
    ) -> None:
        """POST /api/animas/{name}/start calls supervisor.start_anima
        and returns ``{status: started}``."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "stopped",
                    "pid": None,
                    "uptime_sec": None,
                    "bootstrapping": False,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "started"
        assert body["name"] == "alice"

        # Verify supervisor.start_anima was called exactly once
        supervisor.start_anima.assert_awaited_once_with("alice")

    async def test_start_endpoint_rejects_running_anima(
        self, tmp_path: Path,
    ) -> None:
        """When an anima is already running, POST /start returns
        ``{status: already_running}`` without calling start_anima."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "running",
                    "pid": 99999,
                    "uptime_sec": 120.0,
                    "bootstrapping": False,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "already_running"
        assert body["current_status"] == "running"

        # start_anima should NOT have been called
        supervisor.start_anima.assert_not_awaited()

    async def test_bootstrapping_anima_transitions_to_idle(
        self, tmp_path: Path,
    ) -> None:
        """After bootstrap completes, the anima status should reflect
        'running' (idle) rather than 'bootstrapping'.

        This simulates two successive GET /api/animas calls: the first
        during bootstrap, the second after bootstrap completes.
        """
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        # Phase 1: bootstrapping
        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "bootstrapping",
                    "pid": 12345,
                    "uptime_sec": 3.0,
                    "bootstrapping": True,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.get("/api/animas")

        data1 = resp1.json()
        assert data1[0]["bootstrapping"] is True
        assert data1[0]["status"] == "bootstrapping"

        # Phase 2: bootstrap complete — update supervisor mock to return
        # running status.  Since the route calls supervisor.get_process_status
        # on each request, changing the mock is equivalent to the bootstrap
        # finishing between the two requests.
        app.state.supervisor.get_process_status = MagicMock(
            return_value={
                "status": "running",
                "pid": 12345,
                "uptime_sec": 30.0,
                "bootstrapping": False,
            },
        )

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp2 = await client.get("/api/animas")

        data2 = resp2.json()
        assert data2[0]["bootstrapping"] is False
        assert data2[0]["status"] == "running"

    async def test_multiple_animas_independent_states(
        self, tmp_path: Path,
    ) -> None:
        """Two animas can have different states simultaneously — one
        sleeping (stopped), one bootstrapping."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")
        _create_anima_on_disk(animas_dir, "bob")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "stopped",
                    "pid": None,
                    "uptime_sec": None,
                    "bootstrapping": False,
                },
                "bob": {
                    "status": "bootstrapping",
                    "pid": 54321,
                    "uptime_sec": 2.0,
                    "bootstrapping": True,
                },
            },
        )

        app = _create_app(
            tmp_path, anima_names=["alice", "bob"], supervisor=supervisor,
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        animas = {p["name"]: p for p in data}

        # Alice is sleeping (stopped)
        assert animas["alice"]["status"] == "stopped"
        assert animas["alice"]["bootstrapping"] is False

        # Bob is bootstrapping
        assert animas["bob"]["status"] == "bootstrapping"
        assert animas["bob"]["bootstrapping"] is True

    async def test_start_unknown_anima_returns_404(
        self, tmp_path: Path,
    ) -> None:
        """POST /api/animas/{name}/start for a name not in anima_names
        returns 404."""
        supervisor = _make_supervisor_mock()
        app = _create_app(tmp_path, anima_names=[], supervisor=supervisor)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/nobody/start")

        assert resp.status_code == 404

    async def test_start_stopped_anima_is_accepted(
        self, tmp_path: Path,
    ) -> None:
        """POST /start with status 'not_found' (never started) is accepted."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "not_found",
                    "bootstrapping": False,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/start")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "started"
        supervisor.start_anima.assert_awaited_once_with("alice")

    async def test_anima_list_non_bootstrapping_anima(
        self, tmp_path: Path,
    ) -> None:
        """A running anima that is NOT bootstrapping has
        ``bootstrapping: False``."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        supervisor = _make_supervisor_mock(
            statuses={
                "alice": {
                    "status": "running",
                    "pid": 11111,
                    "uptime_sec": 600.0,
                    "bootstrapping": False,
                },
            },
        )

        app = _create_app(tmp_path, anima_names=["alice"], supervisor=supervisor)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["bootstrapping"] is False
        assert data[0]["status"] == "running"
