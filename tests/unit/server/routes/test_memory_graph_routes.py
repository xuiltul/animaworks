"""Unit tests for the memory graph and episode calendar APIs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from networkx.readwrite import json_graph

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


def _write_graph_cache(anima_dir: Path, graph: nx.DiGraph) -> None:
    cache_dir = anima_dir / "vectordb"
    cache_dir.mkdir()
    (cache_dir / "knowledge_graph.json").write_text(
        json.dumps(json_graph.node_link_data(graph)),
        encoding="utf-8",
    )


class TestMemoryGraph:
    async def test_cached_graph_includes_frontmatter_and_excludes_dangling_edges(
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
            "---\n\nAlpha",
            encoding="utf-8",
        )
        beta.write_text("Beta", encoding="utf-8")
        episode.write_text("Episode", encoding="utf-8")

        graph = nx.DiGraph()
        graph.add_node(
            "alpha",
            node_type="memory_file",
            memory_type="knowledge",
            stem="alpha",
            rel_key="alpha",
            path=str(alpha),
        )
        graph.add_node(
            "beta",
            node_type="memory_file",
            memory_type="knowledge",
            stem="beta",
            rel_key="beta",
            path=str(beta),
        )
        graph.add_node(
            "deleted",
            node_type="memory_file",
            memory_type="knowledge",
            stem="deleted",
            rel_key="deleted",
            path=str(anima_dir / "knowledge" / "deleted.md"),
        )
        graph.add_node(
            "episodes:2026-07-01",
            node_type="memory_file",
            memory_type="episodes",
            stem="2026-07-01",
            rel_key="2026-07-01",
            path=str(episode),
        )
        graph.add_edge("alpha", "beta", link_type="explicit", similarity=1.0)
        graph.add_edge("beta", "alpha", link_type="implicit", similarity=0.83)
        graph.add_edge("alpha", "deleted", link_type="explicit", similarity=1.0)
        graph.add_edge("alpha", "episodes:2026-07-01", link_type="implicit", similarity=0.9)
        _write_graph_cache(anima_dir, graph)

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/memory/graph",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["partial"] is False
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
            {
                "source": "beta",
                "target": "alpha",
                "link_type": "implicit",
                "similarity": 0.83,
            },
        ]

    async def test_missing_cache_builds_partial_explicit_graph(self, tmp_path: Path) -> None:
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

    async def test_similarity_edges_are_sorted_and_capped(self, tmp_path: Path) -> None:
        animas_dir, anima_dir = _make_anima(tmp_path)
        shared_file = anima_dir / "knowledge" / "shared.md"
        shared_file.write_text("Shared", encoding="utf-8")
        graph = nx.DiGraph()
        graph.add_node(
            "source",
            node_type="memory_file",
            memory_type="knowledge",
            path=str(shared_file),
        )
        for index in range(501):
            node_id = f"target-{index}"
            graph.add_node(
                node_id,
                node_type="memory_file",
                memory_type="knowledge",
                path=str(shared_file),
            )
            graph.add_edge(
                "source",
                node_id,
                link_type="implicit",
                similarity=index / 1000,
            )
        _write_graph_cache(anima_dir, graph)

        response = await _get(
            _make_test_app(animas_dir),
            "/api/animas/alice/memory/graph",
        )

        data = response.json()
        assert response.status_code == 200
        assert data["edges_capped"] is True
        assert len(data["edges"]) == 500
        assert data["edges"][0]["similarity"] == 0.5
        assert data["edges"][-1]["similarity"] == 0.001


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
