from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from server.routes.animas import create_animas_router
from server.routes.memory_routes import create_memory_router
from server.routes.sessions import create_sessions_router


def _snapshot_tree(root: Path) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        snapshot[relative] = path.read_bytes() if path.is_file() else None
    return snapshot


@pytest.mark.asyncio
async def test_read_only_get_apis_do_not_create_or_migrate_memory_files(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    state_dir = anima_dir / "state"
    legacy_shortterm = anima_dir / "shortterm"
    state_dir.mkdir(parents=True)
    legacy_shortterm.mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("Alice identity\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("Alice instructions\n", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (anima_dir / "permissions.md").write_text("## File Operations\n", encoding="utf-8")
    (state_dir / "current_task.md").write_text("legacy task\n", encoding="utf-8")
    (state_dir / "pending.md").write_text("legacy pending\n", encoding="utf-8")
    (legacy_shortterm / "session_state.md").write_text("legacy session\n", encoding="utf-8")
    before = _snapshot_tree(anima_dir)

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.anima_names = ["alice"]
    app.state.supervisor = MagicMock()
    app.state.supervisor.get_process_status.return_value = {"status": "stopped"}
    app.include_router(create_animas_router(), prefix="/api")
    app.include_router(create_sessions_router(), prefix="/api")
    app.include_router(create_memory_router(), prefix="/api")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/animas/alice")).status_code == 200
        assert (await client.get("/api/animas/alice/sessions")).status_code == 200
        assert (await client.get("/api/animas/alice/episodes")).status_code == 200
        assert (await client.get("/api/animas/alice/knowledge")).status_code == 200

    assert _snapshot_tree(anima_dir) == before
