"""E2E coverage for memory graph and episode calendar APIs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from httpx import ASGITransport, AsyncClient
from networkx.readwrite import json_graph

pytestmark = pytest.mark.e2e


def _create_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    for directory in ("episodes", "knowledge", "procedures"):
        (anima_dir / directory).mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text("# Alice\n", encoding="utf-8")
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    with (
        patch("server.app.ProcessSupervisor") as supervisor_class,
        patch("server.app.load_config") as load_config,
        patch("server.app.WebSocketManager") as websocket_manager_class,
        patch("server.app.load_auth") as load_auth,
    ):
        config = MagicMock()
        config.setup_complete = True
        load_config.return_value = config

        auth = MagicMock()
        auth.auth_mode = "local_trust"
        load_auth.return_value = auth

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        supervisor_class.return_value = supervisor

        websocket_manager = MagicMock()
        websocket_manager.active_connections = []
        websocket_manager_class.return_value = websocket_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as server_app

    monkeypatch.setattr(server_app, "load_auth", lambda: auth)
    return app, anima_dir


async def test_graph_calendar_and_episode_content_through_full_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, anima_dir = _create_app(tmp_path, monkeypatch)
    (anima_dir / "knowledge" / "alpha.md").write_text(
        "---\nsource_episodes: [2026-07-20.md]\n---\n\nAlpha links to [[beta]].",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "beta.md").write_text("Beta body", encoding="utf-8")
    (anima_dir / "episodes" / "2026-07-20.md").write_text(
        "# July 20 episode",
        encoding="utf-8",
    )
    graph = nx.DiGraph()
    for node_id, memory_type, stem in (
        ("alpha", "knowledge", "alpha"),
        ("beta", "knowledge", "beta"),
        ("episodes:2026-07-20", "episodes", "2026-07-20"),
    ):
        graph.add_node(
            node_id,
            node_type="memory_file",
            memory_type=memory_type,
            stem=stem,
            rel_key=stem,
            path=str(anima_dir / memory_type / f"{stem}.md"),
        )
    graph.add_edge("alpha", "beta", link_type="explicit", similarity=1.0)
    graph.add_edge(
        "alpha",
        "episodes:2026-07-20",
        link_type="implicit",
        similarity=0.9,
    )
    cache_dir = anima_dir / "vectordb"
    cache_dir.mkdir()
    (cache_dir / "knowledge_graph.json").write_text(
        json.dumps(json_graph.node_link_data(graph)),
        encoding="utf-8",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        graph_response = await client.get("/api/animas/alice/memory/graph")
        calendar_response = await client.get(
            "/api/animas/alice/episodes/calendar",
            params={"year": 2026, "month": 7},
        )
        episode_response = await client.get("/api/animas/alice/episodes/2026-07-20")

    assert graph_response.status_code == 200
    graph = graph_response.json()
    assert graph["partial"] is False
    assert {node["id"] for node in graph["nodes"]} == {"alpha", "beta"}
    assert graph["edges"] == [
        {
            "source": "alpha",
            "target": "beta",
            "link_type": "explicit",
            "similarity": 1.0,
        }
    ]
    alpha = next(node for node in graph["nodes"] if node["id"] == "alpha")
    assert alpha["source_episodes"] == ["2026-07-20.md"]

    assert calendar_response.status_code == 200
    calendar_days = {day["date"]: day for day in calendar_response.json()["days"]}
    assert calendar_days["2026-07-20"]["has_episode"] is True
    assert calendar_days["2026-07-20"]["size_bytes"] > 0
    assert calendar_days["2026-07-19"]["has_episode"] is False

    assert episode_response.status_code == 200
    assert episode_response.json()["content"] == "# July 20 episode"
