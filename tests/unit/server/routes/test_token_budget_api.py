from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.routes.system import create_system_router


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_system_router(), prefix="/api")
    return app


async def test_token_budget_endpoint_returns_each_anima_current_month_status(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "animas" / "alice").mkdir(parents=True)
    (data_dir / "animas" / "bob").mkdir(parents=True)

    def budget_status(anima_dir, *, now):
        if anima_dir.name == "alice":
            return SimpleNamespace(budget=1_000, consumed=1_100, remaining=0, exceeded=True)
        return SimpleNamespace(budget=None, consumed=25, remaining=None, exceeded=False)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("server.routes.system.now_local", return_value=datetime(2026, 7, 22, tzinfo=UTC)),
        patch("core.memory.token_budget.read_token_budget_status", side_effect=budget_status),
    ):
        transport = ASGITransport(app=_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/system/token-budget")

    assert response.status_code == 200
    assert response.json() == {
        "alice": {
            "month": "2026-07",
            "budget": 1_000,
            "consumed": 1_100,
            "remaining": 0,
            "exceeded": True,
        },
        "bob": {
            "month": "2026-07",
            "budget": None,
            "consumed": 25,
            "remaining": None,
            "exceeded": False,
        },
    }


async def test_token_budget_endpoint_can_filter_one_anima(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "animas" / "alice").mkdir(parents=True)
    status = SimpleNamespace(budget=1_000, consumed=500, remaining=500, exceeded=False)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.memory.token_budget.read_token_budget_status", return_value=status),
    ):
        transport = ASGITransport(app=_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/system/token-budget?anima=alice")

    assert response.status_code == 200
    assert list(response.json()) == ["alice"]


async def test_token_budget_endpoint_reports_missing_anima(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "animas").mkdir(parents=True)

    with patch("core.paths.get_data_dir", return_value=data_dir):
        transport = ASGITransport(app=_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/system/token-budget?anima=missing")

    assert response.status_code == 200
    assert response.json() == {"error": "Anima 'missing' not found"}


async def test_token_budget_endpoint_rejects_parent_path(tmp_path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "animas").mkdir(parents=True)
    (data_dir / "outside").mkdir()

    with patch("core.paths.get_data_dir", return_value=data_dir):
        transport = ASGITransport(app=_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/system/token-budget", params={"anima": "../outside"})

    assert response.status_code == 200
    assert response.json() == {"error": "Anima '../outside' not found"}
    assert not (data_dir / "outside" / "token_usage").exists()
