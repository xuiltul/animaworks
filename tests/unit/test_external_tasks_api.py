# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GET /api/external-tasks (snapshot store backed)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.external_tasks.models import ExternalTask, Snapshot, SourceHealth
from core.external_tasks.store import ExternalTaskStore
from server.routes.external_tasks import create_external_tasks_router


def _task(
    *,
    id: str,
    title: str = "Task",
    status: str = "open",
    source_type: str = "github",
    source_icon: str | None = None,
    source_url: str | None = "https://example.com/t",
    created_at: str,
    last_updated_at: str | None = None,
    priority: int = 50,
) -> ExternalTask:
    # Use None sentinel for "default to created_at"; allow explicit "" for bad-ts tests.
    resolved_updated = created_at if last_updated_at is None else last_updated_at
    return ExternalTask(
        id=id,
        title=title,
        status=status,
        source_type=source_type,
        source_icon=source_icon or source_type,
        source_url=source_url,
        created_at=created_at,
        last_updated_at=resolved_updated,
        priority=priority,
    )


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "external_tasks.json"


@pytest.fixture
def client(store_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(create_external_tasks_router(), prefix="/api")
    with patch(
        "server.routes.external_tasks.get_external_tasks_store_path",
        return_value=store_path,
    ):
        yield TestClient(app)


def _save_snapshot(path: Path, snapshot: Snapshot) -> None:
    ExternalTaskStore(path).save(snapshot)


# ── empty / corrupt store ─────────────────────────


def test_empty_store_returns_empty_data(client: TestClient) -> None:
    resp = client.get("/api/external-tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["meta"]["total_count"] == 0
    assert body["meta"]["limit"] == 20
    assert body["meta"]["offset"] == 0
    assert body["meta"]["has_more"] is False
    assert body["meta"]["last_collected_at"] is None
    assert body["meta"]["sources"] == {}


def test_corrupt_json_returns_200_empty(client: TestClient, store_path: Path) -> None:
    store_path.write_text("{not valid json", encoding="utf-8")
    resp = client.get("/api/external-tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["meta"]["total_count"] == 0
    assert body["meta"]["last_collected_at"] is None


# ── meta from snapshot ────────────────────────────


def test_meta_last_collected_at_and_sources(client: TestClient, store_path: Path) -> None:
    collected = "2026-07-20T12:00:00+00:00"
    _save_snapshot(
        store_path,
        Snapshot(
            version=1,
            last_collected_at=collected,
            sources={
                "github": SourceHealth(
                    status="ok",
                    collected_at=collected,
                    error=None,
                ),
                "slack": SourceHealth(
                    status="unavailable",
                    collected_at=collected,
                    error="missing token",
                ),
            },
            tasks=[],
        ),
    )
    resp = client.get("/api/external-tasks")
    assert resp.status_code == 200
    meta = resp.json()["meta"]
    assert meta["last_collected_at"] == collected
    assert meta["sources"]["github"]["status"] == "ok"
    assert meta["sources"]["github"]["error"] is None
    assert meta["sources"]["slack"]["status"] == "unavailable"
    assert meta["sources"]["slack"]["error"] == "missing token"


# ── filter / sort / page ──────────────────────────


@pytest.fixture
def seeded_client(client: TestClient, store_path: Path) -> TestClient:
    base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    tasks = [
        _task(
            id="github-1",
            title="High GH",
            status="open",
            source_type="github",
            created_at=(base - timedelta(hours=1)).isoformat(),
            last_updated_at=(base - timedelta(minutes=10)).isoformat(),
            priority=90,
        ),
        _task(
            id="slack-1",
            title="Mid Slack",
            status="open",
            source_type="slack",
            created_at=(base - timedelta(hours=2)).isoformat(),
            last_updated_at=(base - timedelta(hours=1)).isoformat(),
            priority=70,
        ),
        _task(
            id="gmail-1",
            title="Done Gmail",
            status="done",
            source_type="gmail",
            created_at=(base - timedelta(days=1)).isoformat(),
            last_updated_at=(base - timedelta(hours=5)).isoformat(),
            priority=40,
        ),
        _task(
            id="chatwork-1",
            title="In progress CW",
            status="in_progress",
            source_type="chatwork",
            created_at=(base - timedelta(hours=3)).isoformat(),
            last_updated_at=(base - timedelta(minutes=30)).isoformat(),
            priority=80,
        ),
        _task(
            id="github-2",
            title="Old GH",
            status="open",
            source_type="github",
            created_at=(base - timedelta(days=3)).isoformat(),
            last_updated_at=(base - timedelta(days=2)).isoformat(),
            priority=50,
        ),
    ]
    _save_snapshot(
        store_path,
        Snapshot(
            version=1,
            last_collected_at=base.isoformat(),
            sources={
                "github": SourceHealth(status="ok", collected_at=base.isoformat()),
            },
            tasks=tasks,
        ),
    )
    return client


def test_filter_by_status(seeded_client: TestClient) -> None:
    resp = seeded_client.get("/api/external-tasks", params={"status": "open"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 3
    assert all(t["status"] == "open" for t in data)


def test_filter_by_source_type(seeded_client: TestClient) -> None:
    resp = seeded_client.get("/api/external-tasks", params={"source_type": "github"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2
    assert all(t["source_type"] == "github" for t in data)


def test_filter_by_since(seeded_client: TestClient) -> None:
    # Only tasks updated after 2026-07-20T10:00:00Z (within ~2h of base 12:00)
    since = "2026-07-20T10:00:00+00:00"
    resp = seeded_client.get("/api/external-tasks", params={"since": since})
    assert resp.status_code == 200
    data = resp.json()["data"]
    ids = {t["id"] for t in data}
    assert "github-1" in ids
    assert "slack-1" in ids
    assert "chatwork-1" in ids
    assert "github-2" not in ids  # updated 2 days ago
    assert "gmail-1" not in ids  # updated 5 hours before base = 07:00


def test_filter_by_since_skips_unparseable_last_updated(
    client: TestClient, store_path: Path
) -> None:
    base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    _save_snapshot(
        store_path,
        Snapshot(
            version=1,
            last_collected_at=base.isoformat(),
            sources={"github": SourceHealth(status="ok", collected_at=base.isoformat())},
            tasks=[
                _task(
                    id="github-ok",
                    created_at=base.isoformat(),
                    last_updated_at=base.isoformat(),
                    priority=90,
                ),
                _task(
                    id="github-empty-ts",
                    created_at=base.isoformat(),
                    last_updated_at="",
                    priority=80,
                ),
                _task(
                    id="github-bad-ts",
                    created_at=base.isoformat(),
                    last_updated_at="not-a-date",
                    priority=70,
                ),
            ],
        ),
    )
    resp = client.get(
        "/api/external-tasks",
        params={"since": "2026-07-20T00:00:00+00:00"},
    )
    assert resp.status_code == 200
    ids = {t["id"] for t in resp.json()["data"]}
    assert ids == {"github-ok"}


def test_filter_by_since_naive_param_returns_200(
    client: TestClient, store_path: Path
) -> None:
    base = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    _save_snapshot(
        store_path,
        Snapshot(
            version=1,
            last_collected_at=base.isoformat(),
            sources={},
            tasks=[
                _task(
                    id="github-1",
                    created_at=base.isoformat(),
                    last_updated_at=base.isoformat(),
                )
            ],
        ),
    )
    # Naive since (no timezone) must not 500; ensure_aware aligns it.
    resp = client.get(
        "/api/external-tasks",
        params={"since": "2026-07-20T00:00:00"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1


def test_sort_by_priority_desc(seeded_client: TestClient) -> None:
    resp = seeded_client.get(
        "/api/external-tasks",
        params={"sort": "priority", "order": "desc"},
    )
    assert resp.status_code == 200
    priorities = [t["priority"] for t in resp.json()["data"]]
    assert priorities == sorted(priorities, reverse=True)
    assert resp.json()["data"][0]["id"] == "github-1"


def test_sort_by_created_at_asc(seeded_client: TestClient) -> None:
    resp = seeded_client.get(
        "/api/external-tasks",
        params={"sort": "created_at", "order": "asc"},
    )
    assert resp.status_code == 200
    created = [t["created_at"] for t in resp.json()["data"]]
    assert created == sorted(created)
    assert resp.json()["data"][0]["id"] == "github-2"


def test_pagination_limit_offset_has_more(seeded_client: TestClient) -> None:
    resp = seeded_client.get(
        "/api/external-tasks",
        params={"limit": 2, "offset": 0, "sort": "priority", "order": "desc"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["total_count"] == 5
    assert body["meta"]["limit"] == 2
    assert body["meta"]["offset"] == 0
    assert body["meta"]["has_more"] is True
    assert len(body["data"]) == 2

    resp2 = seeded_client.get(
        "/api/external-tasks",
        params={"limit": 2, "offset": 4, "sort": "priority", "order": "desc"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert len(body2["data"]) == 1
    assert body2["meta"]["has_more"] is False
    assert body2["meta"]["offset"] == 4


# ── invalid parameters ────────────────────────────


def test_invalid_sort_returns_400(client: TestClient) -> None:
    resp = client.get("/api/external-tasks", params={"sort": "unknown"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_PARAMETER"
    assert err["details"][0]["field"] == "sort"


def test_invalid_order_returns_400(client: TestClient) -> None:
    resp = client.get("/api/external-tasks", params={"order": "sideways"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_PARAMETER"
    assert err["details"][0]["field"] == "order"


def test_invalid_status_filter_returns_422(client: TestClient) -> None:
    resp = client.get("/api/external-tasks", params={"status": "bogus"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_FILTER"
    assert err["details"][0]["field"] == "status"


def test_invalid_source_type_filter_returns_422(client: TestClient) -> None:
    resp = client.get("/api/external-tasks", params={"source_type": "bogus"})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_FILTER"
    assert err["details"][0]["field"] == "source_type"


def test_invalid_since_returns_400(client: TestClient) -> None:
    resp = client.get("/api/external-tasks", params={"since": "not-a-date"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_PARAMETER"
    assert err["details"][0]["field"] == "since"


def test_limit_out_of_range_returns_422(client: TestClient) -> None:
    # FastAPI Query ge/le validation
    resp = client.get("/api/external-tasks", params={"limit": 0})
    assert resp.status_code == 422
    resp2 = client.get("/api/external-tasks", params={"limit": 101})
    assert resp2.status_code == 422


def test_chatwork_is_valid_source_type(seeded_client: TestClient) -> None:
    resp = seeded_client.get("/api/external-tasks", params={"source_type": "chatwork"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "chatwork-1"
