# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the image colour exposed by GET /api/animas."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from server.routes.animas import _read_image_color


def _make_test_app(animas_dir: Path, anima_names: list[str]):
    from fastapi import FastAPI

    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = animas_dir
    app.state.anima_names = anima_names

    supervisor = MagicMock()
    supervisor.get_process_status.return_value = {"status": "running", "pid": 1234}
    supervisor.send_request = AsyncMock(return_value={"action": "skip", "summary": "ok"})
    app.state.supervisor = supervisor

    app.include_router(create_animas_router(), prefix="/api")
    return app


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("- **イメージカラー**: 紅蓮 (#C41E3A)\n", "#C41E3A"),
        ("| イメージカラー | 薄曇りの水色 (#9FC3D9) |\n", "#9FC3D9"),
        ("image_color: #123456\n", "#123456"),
    ],
)
def test_read_image_color_formats(tmp_path: Path, identity: str, expected: str) -> None:
    (tmp_path / "IDENTITY.md").write_text(identity, encoding="utf-8")

    assert _read_image_color(tmp_path) == expected


def test_read_image_color_missing_file(tmp_path: Path) -> None:
    assert _read_image_color(tmp_path) is None


def test_read_image_color_without_declaration(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text(
        "# Alice\n\n- 好きなカラー: 青\n- 役割: 進行役\n",
        encoding="utf-8",
    )

    assert _read_image_color(tmp_path) is None


def test_read_image_color_ignores_bare_color_label(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("カラー: #ABCDEF\n", encoding="utf-8")

    assert _read_image_color(tmp_path) is None


async def test_list_animas_includes_image_color(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    (animas_dir / "alice").mkdir(parents=True)
    (animas_dir / "alice" / "IDENTITY.md").write_text(
        "- **イメージカラー**: 紅蓮 (#C41E3A)\n",
        encoding="utf-8",
    )
    (animas_dir / "bob").mkdir(parents=True)
    (animas_dir / "bob" / "IDENTITY.md").write_text("# Bob\n", encoding="utf-8")

    app = _make_test_app(animas_dir, ["alice", "bob"])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/animas")

    assert response.status_code == 200
    rows = {row["name"]: row for row in response.json()}
    assert rows["alice"]["image_color"] == "#C41E3A"
    assert rows["bob"]["image_color"] is None
