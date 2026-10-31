"""Route tests for the facts-backed memory connections API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.time_utils import now_local
from server.routes.memory_routes import create_memory_router


def _make_app(tmp_path: Path) -> tuple[FastAPI, Path, Path]:
    data_dir = tmp_path / "data"
    animas_dir = data_dir / "animas"
    shared_dir = data_dir / "shared"
    anima_dir = animas_dir / "alice"
    (anima_dir / "facts").mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("Alice", encoding="utf-8")
    (animas_dir / "bob").mkdir(parents=True)
    (animas_dir / "bob" / "IDENTITY.md").write_text("Bob", encoding="utf-8")
    (shared_dir / "users" / "taka").mkdir(parents=True)
    (data_dir / "config.json").write_text(
        json.dumps({"external_messaging": {"user_aliases": {"guest": {"slack_user_id": "U123"}}}}),
        encoding="utf-8",
    )
    timestamp = now_local().isoformat()
    fact = {
        "fact_id": "fact-1",
        "text": "Taka discussed Alpha with Bob.",
        "entities": ["taka（代表）", "Alpha", "bob"],
        "recorded_at": timestamp,
        "valid_at": timestamp,
        "valid_until": "",
    }
    (anima_dir / "facts" / f"{timestamp[:10]}.jsonl").write_text(
        json.dumps(fact, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir
    app.include_router(create_memory_router(), prefix="/api")
    return app, animas_dir, anima_dir


async def _get(app: FastAPI, url: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(url)


class TestMemoryConnectionsRoutes:
    async def test_connections_route_returns_expected_shape(self, tmp_path: Path) -> None:
        app, _animas_dir, _anima_dir = _make_app(tmp_path)

        response = await _get(app, "/api/animas/alice/memory/connections")

        assert response.status_code == 200
        data = response.json()
        assert data["anima"] == "alice"
        assert data["totals"] == {"facts": 1, "entities": 3}
        assert data["days"] is None
        assert {node["label"] for node in data["nodes"]} == {"taka", "Alpha", "bob"}
        assert all({"id", "label", "kind", "count", "last_seen"} <= set(node) for node in data["nodes"])
        assert data["links"]
        assert all({"source", "target", "weight"} == set(link) for link in data["links"])

    async def test_entity_route_returns_facts_and_unknown_entity_as_empty(self, tmp_path: Path) -> None:
        app, _animas_dir, _anima_dir = _make_app(tmp_path)

        response = await _get(app, "/api/animas/alice/memory/connections/entity?name=Alpha")
        missing = await _get(app, "/api/animas/alice/memory/connections/entity?name=Unknown")

        assert response.status_code == 200
        assert response.json()["entity"] == "Alpha"
        assert response.json()["facts"][0]["fact_id"] == "fact-1"
        assert {"entity", "kind", "count", "last_seen", "facts", "related", "files"} <= set(response.json())
        assert missing.status_code == 200
        assert missing.json()["facts"] == []
        assert missing.json()["count"] == 0

    @pytest.mark.parametrize("query", ["limit=9", "limit=81"])
    async def test_connections_limit_bounds(self, tmp_path: Path, query: str) -> None:
        app, _animas_dir, _anima_dir = _make_app(tmp_path)

        response = await _get(app, f"/api/animas/alice/memory/connections?{query}")

        assert response.status_code == 422

    async def test_both_routes_return_404_for_missing_anima(self, tmp_path: Path) -> None:
        app, _animas_dir, _anima_dir = _make_app(tmp_path)

        connections = await _get(app, "/api/animas/missing/memory/connections")
        entity = await _get(app, "/api/animas/missing/memory/connections/entity?name=Alpha")

        assert connections.status_code == 404
        assert entity.status_code == 404

    async def test_connections_accepts_only_supported_day_windows(self, tmp_path: Path) -> None:
        app, _animas_dir, _anima_dir = _make_app(tmp_path)

        valid = await _get(app, "/api/animas/alice/memory/connections?days=7&limit=10")
        invalid = await _get(app, "/api/animas/alice/memory/connections?days=14")

        assert valid.status_code == 200
        assert valid.json()["days"] == 7
        assert invalid.status_code == 422
