"""Unit tests for global exception handler in server/app.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse


def _make_test_app_with_error_route():
    """Create minimal FastAPI app with global exception handler and error route."""
    app = FastAPI()

    logger = logging.getLogger("animaworks.server")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return StarletteJSONResponse(
            {"error": "Internal server error"}, status_code=500,
        )

    @app.get("/api/test-unhandled-error")
    async def raise_error():
        raise RuntimeError("test unhandled error")

    return app


class TestGlobalExceptionHandler:
    """Tests for the global exception handler."""

    async def test_unhandled_exception_returns_500_json(self):
        app = _make_test_app_with_error_route()
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/test-unhandled-error")
        assert resp.status_code == 500
        assert resp.json() == {"error": "Internal server error"}

    async def test_unhandled_exception_logs_traceback(self, caplog):
        app = _make_test_app_with_error_route()
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        with caplog.at_level(logging.ERROR):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.get("/api/test-unhandled-error")
        assert any("Unhandled exception" in r.message for r in caplog.records)
