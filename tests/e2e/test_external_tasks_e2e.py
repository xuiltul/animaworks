"""E2E tests for GET /api/external-tasks (snapshot store backed, no external I/O)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.external_tasks.models import ExternalTask, Snapshot, SourceHealth
from core.external_tasks.store import ExternalTaskStore
from core.paths import get_external_tasks_store_path


# ── Helpers ──────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app with mocked externals (same style as activity e2e)."""
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
        # Disable background external-tasks collection in lifespan.
        cfg.external_tasks = MagicMock()
        cfg.external_tasks.enabled = False
        mock_cfg.return_value = cfg
        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor
        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager
        from server.app import create_app

        app = create_app(animas_dir, shared_dir)
    import server.app as _sa

    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth
    if anima_names is not None:
        app.state.anima_names = anima_names
    return app


def _task(
    *,
    id: str,
    title: str = "Task",
    status: str = "open",
    source_type: str = "github",
    priority: int = 50,
    created_at: str | None = None,
    last_updated_at: str | None = None,
) -> ExternalTask:
    base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    return ExternalTask(
        id=id,
        title=title,
        status=status,
        source_type=source_type,
        source_icon=source_type,
        source_url=f"https://example.com/{id}",
        created_at=created_at or base.isoformat(),
        last_updated_at=last_updated_at or created_at or base.isoformat(),
        priority=priority,
    )


def _write_snapshot(snapshot: Snapshot) -> Path:
    path = get_external_tasks_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ExternalTaskStore(path).save(snapshot)
    return path


# ── Tests ────────────────────────────────────────────────


class TestExternalTasksApiE2E:
    async def test_snapshot_with_three_sources_and_unavailable(
        self, data_dir: Path
    ) -> None:
        """Real store + real app: data count, meta.sources, last_collected_at."""
        base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
        collected = base.isoformat()
        prev_slack = "2026-07-19T08:00:00+00:00"
        snapshot = Snapshot(
            version=1,
            last_collected_at=collected,
            sources={
                "github": SourceHealth(
                    status="ok", collected_at=collected, error=None
                ),
                "slack": SourceHealth(
                    status="unavailable",
                    collected_at=prev_slack,
                    error="credential_missing",
                ),
                "gmail": SourceHealth(
                    status="ok", collected_at=collected, error=None
                ),
            },
            tasks=[
                _task(
                    id="github-1",
                    title="GH PR",
                    source_type="github",
                    priority=90,
                    created_at=(base - timedelta(hours=1)).isoformat(),
                ),
                _task(
                    id="slack-1",
                    title="Slack mention",
                    source_type="slack",
                    priority=80,
                    created_at=(base - timedelta(hours=2)).isoformat(),
                ),
                _task(
                    id="gmail-1",
                    title="Unread mail",
                    source_type="gmail",
                    status="open",
                    priority=70,
                    created_at=(base - timedelta(hours=3)).isoformat(),
                ),
                _task(
                    id="github-2",
                    title="Done issue",
                    source_type="github",
                    status="done",
                    priority=40,
                    created_at=(base - timedelta(days=1)).isoformat(),
                ),
            ],
        )
        _write_snapshot(snapshot)

        app = _create_app(data_dir, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/external-tasks")

        assert resp.status_code == 200
        body = resp.json()
        assert body["meta"]["total_count"] == 4
        assert body["meta"]["last_collected_at"] == collected
        assert body["meta"]["sources"]["github"]["status"] == "ok"
        assert body["meta"]["sources"]["slack"]["status"] == "unavailable"
        assert body["meta"]["sources"]["slack"]["error"] == "credential_missing"
        assert body["meta"]["sources"]["slack"]["collected_at"] == prev_slack
        assert body["meta"]["sources"]["gmail"]["status"] == "ok"
        assert len(body["data"]) == 4
        ids = {t["id"] for t in body["data"]}
        assert ids == {"github-1", "slack-1", "gmail-1", "github-2"}

    async def test_status_and_source_type_filters(self, data_dir: Path) -> None:
        base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
        _write_snapshot(
            Snapshot(
                version=1,
                last_collected_at=base.isoformat(),
                sources={
                    "github": SourceHealth(status="ok", collected_at=base.isoformat()),
                    "slack": SourceHealth(status="ok", collected_at=base.isoformat()),
                },
                tasks=[
                    _task(id="github-open", source_type="github", status="open", priority=90),
                    _task(id="github-done", source_type="github", status="done", priority=40),
                    _task(id="slack-open", source_type="slack", status="open", priority=80),
                ],
            )
        )
        app = _create_app(data_dir, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            by_status = await client.get(
                "/api/external-tasks", params={"status": "open"}
            )
            by_source = await client.get(
                "/api/external-tasks", params={"source_type": "github"}
            )

        assert by_status.status_code == 200
        status_ids = {t["id"] for t in by_status.json()["data"]}
        assert status_ids == {"github-open", "slack-open"}
        assert by_status.json()["meta"]["total_count"] == 2

        assert by_source.status_code == 200
        source_ids = {t["id"] for t in by_source.json()["data"]}
        assert source_ids == {"github-open", "github-done"}

    async def test_pagination(self, data_dir: Path) -> None:
        base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
        tasks = [
            _task(
                id=f"github-{i}",
                source_type="github",
                priority=100 - i,
                created_at=(base - timedelta(minutes=i)).isoformat(),
            )
            for i in range(5)
        ]
        _write_snapshot(
            Snapshot(
                version=1,
                last_collected_at=base.isoformat(),
                sources={"github": SourceHealth(status="ok", collected_at=base.isoformat())},
                tasks=tasks,
            )
        )
        app = _create_app(data_dir, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            page1 = await client.get(
                "/api/external-tasks",
                params={"limit": 2, "offset": 0, "sort": "priority", "order": "desc"},
            )
            page2 = await client.get(
                "/api/external-tasks",
                params={"limit": 2, "offset": 2, "sort": "priority", "order": "desc"},
            )
            page3 = await client.get(
                "/api/external-tasks",
                params={"limit": 2, "offset": 4, "sort": "priority", "order": "desc"},
            )

        assert page1.status_code == 200
        b1 = page1.json()
        assert b1["meta"]["total_count"] == 5
        assert b1["meta"]["has_more"] is True
        assert len(b1["data"]) == 2
        assert b1["data"][0]["id"] == "github-0"

        assert page2.json()["meta"]["has_more"] is True
        assert len(page2.json()["data"]) == 2

        b3 = page3.json()
        assert b3["meta"]["has_more"] is False
        assert len(b3["data"]) == 1

    async def test_corrupt_json_returns_200_empty(self, data_dir: Path) -> None:
        path = get_external_tasks_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")

        app = _create_app(data_dir, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/external-tasks")

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["meta"]["total_count"] == 0
        assert body["meta"]["last_collected_at"] is None
        assert body["meta"]["sources"] == {}
