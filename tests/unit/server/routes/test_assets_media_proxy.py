"""Unit tests for media proxy endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.config.models import MediaProxyConfig


def _make_test_app(animas_dir: Path | None = None):
    from fastapi import FastAPI

    from server.routes.assets import create_assets_router

    app = FastAPI()
    app.state.animas_dir = animas_dir or Path("/tmp/fake/animas")
    app.state.ws_manager = MagicMock()
    app.state.ws_manager.broadcast = AsyncMock()
    router = create_assets_router()
    app.include_router(router, prefix="/api")
    return app


class TestMediaProxy:
    @pytest.fixture(autouse=True)
    def _reset_rate_limit_state(self):
        from server.routes import media_proxy as media_proxy_module

        media_proxy_module._PROXY_RATE_LIMIT_BUCKETS.clear()

    async def test_proxy_rejects_non_https(self, tmp_path):
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "http://example.com/a.png"})
        assert resp.status_code == 400

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    async def test_proxy_allows_non_allowlist_domain_in_open_mode(
        self, mock_get, mock_getaddrinfo, tmp_path,
    ):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "content-length": "8"}
        mock_resp.content = b"\x89PNG\r\n\x1a\n"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
        assert resp.status_code == 200

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.load_config")
    async def test_proxy_blocks_non_allowlisted_domain_in_allowlist_mode(
        self, mock_load_config, mock_getaddrinfo, tmp_path,
    ):
        mock_load_config.return_value = MagicMock(
            server=MagicMock(
                media_proxy=MediaProxyConfig(mode="allowlist", allowed_domains=["images.unsplash.com"]),
            ),
        )
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
        assert resp.status_code == 403

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    async def test_proxy_success(self, mock_get, mock_getaddrinfo, tmp_path):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "content-length": "8"}
        mock_resp.content = b"\x89PNG\r\n\x1a\n"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request(
                "GET",
                "/api/media/proxy",
                params={"url": "https://images.unsplash.com/photo.png"},
            )
        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    async def test_proxy_redirect_revalidates_host(self, mock_get, mock_getaddrinfo, tmp_path):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        first = MagicMock()
        first.status_code = 302
        first.headers = {"location": "https://example.com/not-allowed.png"}
        first.content = b""
        second = MagicMock()
        second.status_code = 200
        second.headers = {"content-type": "image/png", "content-length": "8"}
        second.content = b"\x89PNG\r\n\x1a\n"
        mock_get.side_effect = [first, second]

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request(
                "GET",
                "/api/media/proxy",
                params={"url": "https://images.unsplash.com/photo.png"},
            )
        assert resp.status_code == 200

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    @patch("server.routes.media_proxy.load_config")
    async def test_proxy_redirect_revalidates_host_in_allowlist_mode(
        self, mock_load_config, mock_get, mock_getaddrinfo, tmp_path,
    ):
        mock_load_config.return_value = MagicMock(
            server=MagicMock(
                media_proxy=MediaProxyConfig(mode="allowlist", allowed_domains=["images.unsplash.com"]),
            ),
        )
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        first = MagicMock()
        first.status_code = 302
        first.headers = {"location": "https://example.com/not-allowed.png"}
        first.content = b""
        mock_get.return_value = first

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request(
                "GET",
                "/api/media/proxy",
                params={"url": "https://images.unsplash.com/photo.png"},
            )
        assert resp.status_code == 403

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    async def test_proxy_rejects_header_magic_mismatch(self, mock_get, mock_getaddrinfo, tmp_path):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "content-length": "11"}
        mock_resp.content = b"not-an-image"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request(
                "GET",
                "/api/media/proxy",
                params={"url": "https://images.unsplash.com/photo.png"},
            )
        assert resp.status_code == 415

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    @patch("server.routes.media_proxy.load_config")
    async def test_proxy_rate_limit_returns_429(
        self, mock_load_config, mock_get, mock_getaddrinfo, tmp_path,
    ):
        mock_load_config.return_value = MagicMock(
            server=MagicMock(
                media_proxy=MediaProxyConfig(rate_limit_requests=1, rate_limit_window_s=60),
            ),
        )
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "content-length": "8"}
        mock_resp.content = b"\x89PNG\r\n\x1a\n"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
            second = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.headers.get("retry-after") is not None

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    async def test_proxy_rejects_svg_content_type(self, mock_get, mock_getaddrinfo, tmp_path):
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/svg+xml", "content-length": "32"}
        mock_resp.content = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.svg"})
        assert resp.status_code == 415

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    @patch("server.routes.media_proxy.load_config")
    async def test_proxy_rejects_declared_oversize(
        self, mock_load_config, mock_get, mock_getaddrinfo, tmp_path,
    ):
        mock_load_config.return_value = MagicMock(
            server=MagicMock(
                media_proxy=MediaProxyConfig(max_bytes=4),
            ),
        )
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "content-length": "8"}
        mock_resp.content = b"\x89PNG\r\n\x1a\n"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
        assert resp.status_code == 413

    @patch("server.routes.media_proxy.socket.getaddrinfo")
    @patch("server.routes.media_proxy.httpx.AsyncClient.get")
    @patch("server.routes.media_proxy.load_config")
    async def test_proxy_rejects_actual_oversize(
        self, mock_load_config, mock_get, mock_getaddrinfo, tmp_path,
    ):
        mock_load_config.return_value = MagicMock(
            server=MagicMock(
                media_proxy=MediaProxyConfig(max_bytes=8),
            ),
        )
        mock_getaddrinfo.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNG\r\n\x1a\nextra"
        mock_get.return_value = mock_resp

        app = _make_test_app(animas_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.request("GET", "/api/media/proxy", params={"url": "https://example.com/a.png"})
        assert resp.status_code == 413
