"""Unit tests for error handling in server/routes/animas.py — trigger endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _make_test_app(supervisor=None):
    from fastapi import FastAPI
    from server.routes.animas import create_animas_router

    app = FastAPI()
    app.state.animas_dir = Path("/tmp/fake/animas")
    app.state.anima_names = ["alice"]
    sup = supervisor or MagicMock()
    app.state.supervisor = sup
    router = create_animas_router()
    app.include_router(router, prefix="/api")
    return app


class TestTriggerErrorHandling:
    async def test_trigger_timeout_returns_504(self):
        sup = MagicMock()
        sup.send_request = AsyncMock(side_effect=asyncio.TimeoutError())
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/trigger")
        assert resp.status_code == 504
        data = resp.json()
        assert "error" in data

    async def test_trigger_runtime_error_returns_500(self):
        sup = MagicMock()
        sup.send_request = AsyncMock(side_effect=RuntimeError("process crashed"))
        app = _make_test_app(supervisor=sup)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/animas/alice/trigger")
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
