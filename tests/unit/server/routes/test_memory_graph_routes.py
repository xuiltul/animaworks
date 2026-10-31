"""Unit tests for the memory graph and episode calendar APIs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.time_utils import get_app_timezone
from server.routes.memory_routes import create_memory_router


def _make_test_app(animas_dir: Path) -> FastAPI:
    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.include_router(create_memory_router(), prefix="/api")
    return app


async def _get(app: FastAPI, url: str):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.get(url)


def _make_anima(tmp_path: Path) -> tuple[Path, Path]:
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    for directory in ("episodes", "knowledge", "procedures"):
        (anima_dir / directory).mkdir(parents=True, exist_ok=True)
    return animas_dir, anima_dir


class TestMemoryGraph:
    async def test_explicit_graph_includes_frontmatter_and_resolved_links(
        self,
        tmp_path: Path,
    ) -> None:
        animas_dir, anima_dir = _make_anima(tmp_path)
        alpha = anima_dir / "knowledge" / "alpha.md"
        beta = anima_dir / "knowledge" / "beta.md"
        episode = anima_dir / "episodes" / "2026-07-01.md"
        alpha.write_text(
            "---\n"
            "created_at: '2026-07-01T10:00:00+09:00'\n"
            "updated_at: '2026-07-02T10:00:00+09:00'\n"
            "confidence: 0.9\n"
            "source_episodes: [2026-06-30.md]\n"
            "---\n\nAlpha links to [[beta]] and [[missing]].",
            encoding="utf-8",
        )
        beta.write_text("Beta", encoding="utf-8")
        episode.write_text("Episode", encoding="utf-8")

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/memory/graph",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["partial"] is True
        assert data["edges_capped"] is False
        assert {node["id"] for node in data["nodes"]} == {"alpha", "beta"}
        alpha_node = next(node for node in data["nodes"] if node["id"] == "alpha")
        assert alpha_node == {
            "id": "alpha",
            "memory_type": "knowledge",
            "stem": "alpha",
            "created_at": "2026-07-01T10:00:00+09:00",
            "updated_at": "2026-07-02T10:00:00+09:00",
            "confidence": 0.9,
            "source_episodes": ["2026-06-30.md"],
        }
        assert data["edges"] == [
            {
                "source": "alpha",
                "target": "beta",
                "link_type": "explicit",
                "similarity": 1.0,
            },
        ]

    async def test_explicit_graph_resolves_links_without_cache(self, tmp_path: Path) -> None:
        animas_dir, anima_dir = _make_anima(tmp_path)
        alpha = anima_dir / "knowledge" / "alpha.md"
        alpha.write_text(
            "Links to [[beta|Beta note]], [[deploy]], and [[missing]].",
            encoding="utf-8",
        )
        beta = anima_dir / "knowledge" / "beta.md"
        beta.write_text(
            "---\nconfidence: 0.7\n---\n\nBeta",
            encoding="utf-8",
        )
        deploy = anima_dir / "procedures" / "deploy.md"
        deploy.write_text("Deploy steps", encoding="utf-8")
        fallback_mtime = 1_720_000_000
        os.utime(alpha, (fallback_mtime, fallback_mtime))

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/memory/graph",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["partial"] is True
        assert data["edges_capped"] is False
        assert {node["id"] for node in data["nodes"]} == {
            "alpha",
            "beta",
            "procedures:deploy",
        }
        assert {(edge["source"], edge["target"]) for edge in data["edges"]} == {
            ("alpha", "beta"),
            ("alpha", "procedures:deploy"),
        }
        alpha_node = next(node for node in data["nodes"] if node["id"] == "alpha")
        assert (
            alpha_node["created_at"]
            == datetime.fromtimestamp(
                fallback_mtime,
                tz=get_app_timezone(),
            ).isoformat()
        )

    async def test_empty_memory_returns_empty_partial_graph(self, tmp_path: Path) -> None:
        animas_dir, _ = _make_anima(tmp_path)

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/memory/graph",
        )

        assert response.status_code == 200
        assert response.json() == {
            "nodes": [],
            "edges": [],
            "partial": True,
            "edges_capped": False,
        }



class TestEpisodeCalendar:
    async def test_month_with_episodes(self, tmp_path: Path) -> None:
        animas_dir, anima_dir = _make_anima(tmp_path)
        first = anima_dir / "episodes" / "2026-07-01.md"
        second = anima_dir / "episodes" / "2026-07-20.md"
        first.write_text("one", encoding="utf-8")
        second.write_text("twenty", encoding="utf-8")
        (anima_dir / "episodes" / "2026-08-01.md").write_text(
            "other month",
            encoding="utf-8",
        )

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/episodes/calendar?year=2026&month=7",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["year"] == 2026
        assert data["month"] == 7
        assert len(data["days"]) == 31
        days = {day["date"]: day for day in data["days"]}
        assert days["2026-07-01"] == {
            "date": "2026-07-01",
            "has_episode": True,
            "size_bytes": first.stat().st_size,
        }
        assert days["2026-07-20"]["size_bytes"] == second.stat().st_size
        assert days["2026-07-02"] == {
            "date": "2026-07-02",
            "has_episode": False,
            "size_bytes": 0,
        }

    async def test_empty_month(self, tmp_path: Path) -> None:
        animas_dir, _ = _make_anima(tmp_path)

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/episodes/calendar?year=2026&month=2",
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["days"]) == 28
        assert all(not day["has_episode"] for day in data["days"])
        assert all(day["size_bytes"] == 0 for day in data["days"])

    @pytest.mark.parametrize(
        "query",
        [
            "year=2026&month=0",
            "year=2026&month=13",
            "year=invalid&month=7",
            "year=2026",
        ],
    )
    async def test_invalid_parameters_return_422(
        self,
        tmp_path: Path,
        query: str,
    ) -> None:
        animas_dir, _ = _make_anima(tmp_path)

        response = await _get(
            _make_test_app(animas_dir),
            f"/api/animas/alice/episodes/calendar?{query}",
        )

        assert response.status_code == 422
