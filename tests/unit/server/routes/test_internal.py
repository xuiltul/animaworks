"""Unit tests for server/routes/internal.py — Internal API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient


def _make_test_app():
    from fastapi import FastAPI

    from server.routes.internal import create_internal_router

    app = FastAPI()
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_internal_router()
    app.include_router(router, prefix="/api")
    return app


# ── POST /internal/message-sent ──────────────────────────


class TestInternalMessageSent:
    async def test_message_sent_broadcasts(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "alice",
                    "to_person": "bob",
                    "content": "Hello Bob",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        ws = app.state.ws_manager
        ws.broadcast.assert_awaited_once()
        call_data = ws.broadcast.call_args[0][0]
        assert call_data["type"] == "anima.interaction"
        assert call_data["data"]["from_person"] == "alice"
        assert call_data["data"]["to_person"] == "bob"

    async def test_message_sent_no_anima_match(self):
        """Non-managed anima as sender should not crash."""
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "external",
                    "to_person": "alice",
                    "content": "Hi",
                },
            )
        assert resp.status_code == 200

    async def test_message_sent_truncates_content(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        long_content = "x" * 500
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={
                    "from_person": "alice",
                    "to_person": "bob",
                    "content": long_content,
                },
            )
        assert resp.status_code == 200
        ws = app.state.ws_manager
        call_data = ws.broadcast.call_args[0][0]
        # The summary should be truncated to 200 chars
        assert len(call_data["data"]["summary"]) <= 200

    async def test_message_sent_missing_fields(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={"from_person": "alice"},
            )
        # Pydantic validation error
        assert resp.status_code == 422

    async def test_message_sent_default_content(self):
        """Content field has a default of empty string."""
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/message-sent",
                json={"from_person": "alice", "to_person": "bob"},
            )
        assert resp.status_code == 200


# ── POST /internal/anima/create ──────────────────────────

_VALID_SHEET = """\
# Character: yoru

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | yoru |
| 日本語名 | ヨル |
| 役職/専門 | テスト |
| 上司 | (なし) |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-6 |
| credential | anthropic |

## 人格

夜勤担当。

## 役割・行動方針

テスト業務を担当します。
"""


class TestInternalAnimaCreate:
    async def test_create_success(self, data_dir):
        from pathlib import Path
        from unittest.mock import patch

        app = _make_test_app()
        transport = ASGITransport(app=app)
        animas_dir = Path(data_dir) / "animas"

        with patch("cli.commands.init_cmd._register_anima_in_config"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/internal/anima/create",
                    json={
                        "character_sheet_content": _VALID_SHEET,
                        "name": "yoru",
                        "calling_anima": "taka",
                    },
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "yoru" in body["anima_dir"]
        assert (animas_dir / "yoru").is_dir()
        # supervisor fallback from calling_anima
        status = json.loads((animas_dir / "yoru" / "status.json").read_text(encoding="utf-8"))
        assert status.get("supervisor") == "taka"

    async def test_create_duplicate_409(self, data_dir):
        from pathlib import Path
        from unittest.mock import patch

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with patch("cli.commands.init_cmd._register_anima_in_config"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                first = await client.post(
                    "/api/internal/anima/create",
                    json={"character_sheet_content": _VALID_SHEET, "name": "yoru"},
                )
                assert first.status_code == 200
                second = await client.post(
                    "/api/internal/anima/create",
                    json={"character_sheet_content": _VALID_SHEET, "name": "yoru"},
                )

        assert second.status_code == 409
        assert "detail" in second.json()
        # ensure first creation still on disk
        assert (Path(data_dir) / "animas" / "yoru").is_dir()

    async def test_create_invalid_sheet_422(self, data_dir):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/anima/create",
                json={"character_sheet_content": "# not a valid sheet\n"},
            )
        assert resp.status_code == 422
        assert "detail" in resp.json()

    async def test_create_missing_content_422(self, data_dir):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/internal/anima/create",
                json={"name": "yoru"},
            )
        assert resp.status_code == 422
