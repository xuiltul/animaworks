"""Unit tests for POST /api/animas/{name}/bootstrap/repair."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient


def _make_anima(animas_dir: Path, name: str, *, defined: bool) -> Path:
    anima_dir = animas_dir / name
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "shortterm").mkdir()
    body = "Defined" if defined else "未定義"
    (anima_dir / "IDENTITY.md").write_text(f"# Identity\n\n{body}\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text(f"# Injection\n\n{body}\n", encoding="utf-8")
    (anima_dir / "character_sheet.md").write_text("# Sheet\n", encoding="utf-8")
    (anima_dir / "state" / "bootstrap_state.json").write_text(
        json.dumps({"version": 1, "state": "needs_repair", "validation_errors": ["character_sheet_unprocessed"]}),
        encoding="utf-8",
    )
    return anima_dir


def _make_app(animas_dir: Path, names: list[str]):
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.anima_names = list(names)
    supervisor = MagicMock()
    supervisor._bootstrap_retries_file = animas_dir / ".bootstrap_retries.json"
    supervisor._bootstrap_retry_counts = {name: 2 for name in names}
    supervisor.restart_anima = AsyncMock()
    supervisor._broadcast_event = AsyncMock()
    app.state.supervisor = supervisor
    app.include_router(create_animas_router(), prefix="/api")
    return app, supervisor


async def _post(app, name: str, action: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(f"/api/animas/{name}/bootstrap/repair", json={"action": action})


async def test_complete_marks_defined_anima_completed(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "hina", defined=True)
    app, supervisor = _make_app(tmp_path, ["hina"])

    resp = await _post(app, "hina", "complete")

    assert resp.status_code == 200
    assert resp.json()["bootstrap_state"]["state"] == "completed"
    assert not (anima_dir / "character_sheet.md").exists()
    supervisor.restart_anima.assert_awaited_once_with("hina")
    supervisor._broadcast_event.assert_awaited_once()
    assert "hina" not in supervisor._bootstrap_retry_counts


async def test_complete_refuses_undefined_identity(tmp_path: Path) -> None:
    _make_anima(tmp_path, "sora", defined=False)
    app, supervisor = _make_app(tmp_path, ["sora"])

    resp = await _post(app, "sora", "complete")

    assert resp.status_code == 409
    supervisor.restart_anima.assert_not_awaited()


async def test_retry_prepares_new_attempt(tmp_path: Path) -> None:
    anima_dir = _make_anima(tmp_path, "kaito", defined=False)
    app, supervisor = _make_app(tmp_path, ["kaito"])

    resp = await _post(app, "kaito", "retry")

    assert resp.status_code == 200
    state = json.loads((anima_dir / "state" / "bootstrap_state.json").read_text(encoding="utf-8"))
    assert state["state"] == "pending_user_input"
    supervisor.restart_anima.assert_awaited_once_with("kaito")


async def test_rejects_unknown_action(tmp_path: Path) -> None:
    _make_anima(tmp_path, "hina", defined=True)
    app, _ = _make_app(tmp_path, ["hina"])

    resp = await _post(app, "hina", "nuke")

    assert resp.status_code == 400
