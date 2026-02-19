# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the setup wizard flow.

Tests the complete setup wizard lifecycle including:
- Setup guard middleware (route blocking/allowing)
- Setup API endpoints (environment, locale, validate-key)
- Full wizard flow (fresh config → setup → route switching)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.config import AnimaWorksConfig, invalidate_cache, save_config


# ── Helpers ──────────────────────────────────────────────────


def _write_config(data_dir: Path, *, setup_complete: bool = False, **overrides: Any) -> None:
    """Write a config.json with the given setup_complete flag."""
    invalidate_cache()
    config = AnimaWorksConfig(setup_complete=setup_complete, **overrides)
    save_config(config, data_dir / "config.json")
    invalidate_cache()


def _create_app(data_dir: Path) -> Any:
    """Create a FastAPI app pointing at the test data_dir."""
    from server.app import create_app

    animas_dir = data_dir / "animas"
    shared_dir = data_dir / "shared"
    return create_app(animas_dir, shared_dir)


@pytest.fixture
def setup_app(data_dir: Path):
    """Create a fresh app in setup mode (setup_complete=False)."""
    _write_config(data_dir, setup_complete=False)
    return _create_app(data_dir)


@pytest.fixture
def completed_app(data_dir: Path):
    """Create an app where setup is already complete."""
    _write_config(data_dir, setup_complete=True)
    return _create_app(data_dir)


# ── 1. Setup Guard Tests ────────────────────────────────────


class TestSetupGuard:
    """Test the setup guard middleware controls route access."""

    @pytest.mark.asyncio
    async def test_setup_routes_accessible_during_setup(self, setup_app):
        """Setup API routes are accessible when setup_complete=False."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_non_setup_api_blocked_during_setup(self, setup_app):
        """Non-setup API routes return 503 during setup."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas")
            assert resp.status_code == 503
            body = resp.json()
            assert "Setup not yet complete" in body["error"]

    @pytest.mark.asyncio
    async def test_root_redirects_to_setup_during_setup(self, setup_app):
        """Root / redirects to /setup/ when setup is not complete."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 307
            assert "/setup/" in resp.headers["location"]

    @pytest.mark.asyncio
    async def test_setup_api_blocked_after_completion(self, completed_app):
        """Setup API routes return 403 after setup is complete."""
        transport = ASGITransport(app=completed_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 403
            body = resp.json()
            assert "Setup already completed" in body["error"]

    @pytest.mark.asyncio
    async def test_setup_page_redirects_after_completion(self, completed_app):
        """Accessing /setup/ after completion redirects to /."""
        transport = ASGITransport(app=completed_app)
        async with AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            resp = await client.get("/setup/")
            assert resp.status_code == 307
            assert resp.headers["location"] == "/"

    @pytest.mark.asyncio
    async def test_multiple_api_routes_blocked_during_setup(self, setup_app):
        """Various non-setup API routes all return 503."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for path in ["/api/animas", "/api/system/health", "/api/sessions"]:
                resp = await client.get(path)
                assert resp.status_code == 503, f"{path} should return 503 during setup"


# ── 2. Setup Flow Integration ───────────────────────────────


class TestSetupEndpoints:
    """Test individual setup API endpoints."""

    @pytest.mark.asyncio
    async def test_get_environment(self, setup_app):
        """GET /api/setup/environment returns correct structure."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 200
            body = resp.json()

            assert "claude_code_available" in body
            assert isinstance(body["claude_code_available"], bool)
            assert "locale" in body
            assert "providers" in body
            assert isinstance(body["providers"], list)
            assert len(body["providers"]) > 0
            assert "available_locales" in body
            assert "ja" in body["available_locales"]
            assert "en" in body["available_locales"]

    @pytest.mark.asyncio
    async def test_get_environment_providers_structure(self, setup_app):
        """Providers in environment response have correct fields."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/environment")
            body = resp.json()

            for provider in body["providers"]:
                assert "id" in provider
                assert "name" in provider
                assert "models" in provider
                assert isinstance(provider["models"], list)
                # env_key can be None (e.g. for Ollama)
                assert "env_key" in provider

    @pytest.mark.asyncio
    async def test_detect_locale_japanese(self, setup_app):
        """GET /api/setup/detect-locale detects Japanese from Accept-Language."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "ja,en-US;q=0.9,en;q=0.8"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "ja"
            assert "available" in body

    @pytest.mark.asyncio
    async def test_detect_locale_english(self, setup_app):
        """GET /api/setup/detect-locale detects English from Accept-Language."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "en"

    @pytest.mark.asyncio
    async def test_detect_locale_fallback(self, setup_app):
        """GET /api/setup/detect-locale falls back to 'ja' for unknown locales."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "sw,cy;q=0.9"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "ja"

    @pytest.mark.asyncio
    async def test_detect_locale_no_header(self, setup_app):
        """GET /api/setup/detect-locale defaults to 'ja' without Accept-Language."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "ja"

    @pytest.mark.asyncio
    async def test_detect_locale_chinese_simplified(self, setup_app):
        """GET /api/setup/detect-locale detects zh-CN from Accept-Language."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "zh-CN,en;q=0.9"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "zh-CN"

    @pytest.mark.asyncio
    async def test_detect_locale_chinese_traditional(self, setup_app):
        """GET /api/setup/detect-locale detects zh-TW from Accept-Language."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "zh-TW,en;q=0.9"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "zh-TW"

    @pytest.mark.asyncio
    async def test_detect_locale_zh_hant_variant(self, setup_app):
        """GET /api/setup/detect-locale maps zh-Hant to zh-TW."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "zh-Hant,en;q=0.9"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["detected"] == "zh-TW"

    @pytest.mark.asyncio
    async def test_validate_key_ollama(self, setup_app):
        """POST /api/setup/validate-key for Ollama always returns valid."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/validate-key",
                json={"provider": "ollama", "api_key": ""},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_key_unknown_provider(self, setup_app):
        """POST /api/setup/validate-key for unknown provider returns invalid."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/validate-key",
                json={"provider": "nonexistent", "api_key": "sk-test"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["valid"] is False
            assert "Unknown provider" in body["message"]

    @pytest.mark.asyncio
    async def test_validate_key_anthropic_mocked(self, setup_app):
        """POST /api/setup/validate-key for Anthropic with mocked HTTP call."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            transport = ASGITransport(app=setup_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "anthropic", "api_key": "sk-ant-test123"},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_key_anthropic_invalid(self, setup_app):
        """POST /api/setup/validate-key for Anthropic with invalid key."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 401

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            transport = ASGITransport(app=setup_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "anthropic", "api_key": "invalid-key"},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_key_openai_mocked(self, setup_app):
        """POST /api/setup/validate-key for OpenAI with mocked HTTP call."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            transport = ASGITransport(app=setup_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "openai", "api_key": "sk-test"},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_key_google_mocked(self, setup_app):
        """POST /api/setup/validate-key for Google with mocked HTTP call."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            transport = ASGITransport(app=setup_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "google", "api_key": "AIza-test"},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["valid"] is True


# ── 3. Setup Completion & Full Wizard Flow ──────────────────


class TestSetupComplete:
    """Test POST /api/setup/complete and the full wizard flow."""

    @pytest.mark.asyncio
    async def test_complete_setup_minimal(self, data_dir: Path):
        """Complete setup with minimal payload (no anima)."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={"locale": "en"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"

        # Verify config was updated
        invalidate_cache()
        config_raw = json.loads((data_dir / "config.json").read_text("utf-8"))
        assert config_raw["setup_complete"] is True
        assert config_raw["locale"] == "en"

    @pytest.mark.asyncio
    async def test_complete_setup_with_chinese_locale(self, data_dir: Path):
        """Complete setup with zh-CN locale and verify it's stored in config."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={"locale": "zh-CN"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"

        # Verify config was updated with zh-CN locale
        invalidate_cache()
        config_raw = json.loads((data_dir / "config.json").read_text("utf-8"))
        assert config_raw["setup_complete"] is True
        assert config_raw["locale"] == "zh-CN"

    @pytest.mark.asyncio
    async def test_complete_setup_with_credentials(self, data_dir: Path):
        """Complete setup saves credentials to config."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "credentials": {
                        "anthropic": {"api_key": "sk-ant-test"},
                        "openai": {"api_key": "sk-oai-test"},
                    },
                },
            )
            assert resp.status_code == 200

        invalidate_cache()
        config_raw = json.loads((data_dir / "config.json").read_text("utf-8"))
        assert config_raw["credentials"]["anthropic"]["api_key"] == "sk-ant-test"
        assert config_raw["credentials"]["openai"]["api_key"] == "sk-oai-test"

    @pytest.mark.asyncio
    async def test_complete_setup_creates_blank_anima(self, data_dir: Path):
        """Complete setup with anima creates a blank anima directory."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        # Mock start_all to prevent spawning real child processes
        app.state.supervisor.start_all = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "anima": {"name": "hinata"},
                },
            )
            assert resp.status_code == 200

        # Verify anima directory was created
        anima_dir = data_dir / "animas" / "hinata"
        assert anima_dir.exists()
        assert (anima_dir / "state").is_dir()

        # Verify anima was added to config
        invalidate_cache()
        config_raw = json.loads((data_dir / "config.json").read_text("utf-8"))
        assert "hinata" in config_raw["animas"]

    @pytest.mark.asyncio
    async def test_route_switching_after_completion(self, data_dir: Path):
        """After setup completion, middleware switches: setup blocked, API open."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        # Mock start_all to prevent spawning real child processes
        app.state.supervisor.start_all = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Before completion: setup accessible, API blocked
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 200

            resp = await client.get("/api/animas")
            assert resp.status_code == 503

            # Complete setup
            resp = await client.post(
                "/api/setup/complete",
                json={"locale": "ja"},
            )
            assert resp.status_code == 200

            # After completion: setup blocked, API accessible
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 403

            # /api/animas should no longer return 503
            resp = await client.get("/api/animas")
            assert resp.status_code != 503

    @pytest.mark.asyncio
    async def test_full_wizard_flow(self, data_dir: Path):
        """End-to-end flow: fresh config → detect → configure → complete → verify."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        # Mock start_all to prevent spawning real child processes
        app.state.supervisor.start_all = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Root redirects to setup
            resp = await client.get("/", follow_redirects=False)
            assert resp.status_code == 307
            assert "/setup/" in resp.headers["location"]

            # Step 2: Detect environment
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 200
            env_data = resp.json()
            assert "providers" in env_data

            # Step 3: Detect locale
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"Accept-Language": "ja,en;q=0.9"},
            )
            assert resp.status_code == 200
            assert resp.json()["detected"] == "ja"

            # Step 4: Validate key (Ollama — no external call)
            resp = await client.post(
                "/api/setup/validate-key",
                json={"provider": "ollama", "api_key": ""},
            )
            assert resp.status_code == 200
            assert resp.json()["valid"] is True

            # Step 5: Complete setup with anima creation
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "credentials": {
                        "anthropic": {"api_key": "sk-ant-test"},
                    },
                    "anima": {"name": "aoi"},
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # Step 6: Verify setup_complete in config
            invalidate_cache()
            config_raw = json.loads((data_dir / "config.json").read_text("utf-8"))
            assert config_raw["setup_complete"] is True

            # Step 7: Verify anima was created
            assert (data_dir / "animas" / "aoi").is_dir()
            assert "aoi" in config_raw["animas"]

            # Step 8: Verify route switching
            resp = await client.get("/api/setup/environment")
            assert resp.status_code == 403

            resp = await client.get("/api/animas")
            assert resp.status_code != 503

    @pytest.mark.asyncio
    async def test_duplicate_anima_handled_gracefully(self, data_dir: Path, make_anima):
        """Completing setup with an existing anima name doesn't crash."""
        make_anima("existing")
        _write_config(data_dir, setup_complete=False)
        # Ensure the anima dir survives the config rewrite
        app = _create_app(data_dir)

        # Mock start_all to prevent spawning real child processes
        app.state.supervisor.start_all = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "anima": {"name": "existing"},
                },
            )
            # Should succeed (logs warning but does not fail)
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_setup_complete_idempotent_config(self, data_dir: Path):
        """Calling complete twice (via direct config) keeps setup_complete=True."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        # Mock start_all to prevent spawning real child processes
        app.state.supervisor.start_all = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First completion
            resp = await client.post(
                "/api/setup/complete",
                json={"locale": "ja"},
            )
            assert resp.status_code == 200

            # Second call should be blocked by middleware (403)
            resp = await client.post(
                "/api/setup/complete",
                json={"locale": "en"},
            )
            assert resp.status_code == 403


# ── 4. Setup Cache Control & i18n ──────────────────────────


class TestSetupCacheControl:
    """Test Cache-Control headers for setup static files."""

    @pytest.mark.asyncio
    async def test_setup_static_js_has_no_cache(self, setup_app):
        """Setup JS files served with Cache-Control: no-cache."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/setup.js")

        if resp.status_code == 200:
            cc = resp.headers.get("cache-control", "")
            assert "no-cache" in cc
            assert "no-store" in cc
            assert "must-revalidate" in cc

    @pytest.mark.asyncio
    async def test_setup_static_css_has_no_cache(self, setup_app):
        """Setup CSS files served with Cache-Control: no-cache."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/setup.css")

        if resp.status_code == 200:
            cc = resp.headers.get("cache-control", "")
            assert "no-cache" in cc

    @pytest.mark.asyncio
    async def test_setup_i18n_json_has_no_cache(self, setup_app):
        """i18n JSON files served with Cache-Control: no-cache."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for locale in ["ja", "en"]:
                resp = await client.get(f"/setup/i18n/{locale}.json")
                if resp.status_code == 200:
                    cc = resp.headers.get("cache-control", "")
                    assert "no-cache" in cc, f"i18n/{locale}.json missing no-cache"

    @pytest.mark.asyncio
    async def test_setup_api_routes_no_extra_cache_header(self, setup_app):
        """Setup API routes should NOT have the no-store cache-control header."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")

        assert resp.status_code == 200
        cc = resp.headers.get("cache-control", "")
        assert "no-store" not in cc

    @pytest.mark.asyncio
    async def test_i18n_ja_has_userinfo_keys(self, setup_app):
        """Japanese i18n file contains all userinfo translation keys."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/i18n/ja.json")

        if resp.status_code == 200:
            data = resp.json()
            required_keys = [
                "userinfo.title",
                "userinfo.desc",
                "userinfo.name",
                "userinfo.name.placeholder",
                "userinfo.name.hint",
                "userinfo.displayname",
                "userinfo.bio",
                "userinfo.bio.placeholder",
                "userinfo.error",
            ]
            for key in required_keys:
                assert key in data, f"Missing i18n key: {key}"

    @pytest.mark.asyncio
    async def test_i18n_en_has_userinfo_keys(self, setup_app):
        """English i18n file contains all userinfo translation keys."""
        transport = ASGITransport(app=setup_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/setup/i18n/en.json")

        if resp.status_code == 200:
            data = resp.json()
            required_keys = [
                "userinfo.title",
                "userinfo.desc",
                "userinfo.name",
                "userinfo.name.placeholder",
                "userinfo.displayname",
                "userinfo.bio",
                "userinfo.bio.placeholder",
            ]
            for key in required_keys:
                assert key in data, f"Missing i18n key: {key}"


# ── 5. Setup with User Info & Anima Auto-start ─────────


class TestSetupWithUserInfo:
    """Test POST /api/setup/complete with user info and anima auto-start."""

    @pytest.mark.asyncio
    async def test_complete_setup_with_user_info(self, data_dir: Path):
        """Complete setup with user info creates auth.json with owner."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "user": {
                        "username": "taro",
                        "display_name": "Taro Yamada",
                        "bio": "A software engineer who loves AI.",
                    },
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ok"

        # Verify auth.json was created
        auth_path = data_dir / "auth.json"
        assert auth_path.exists(), "auth.json should be created"

        auth_data = json.loads(auth_path.read_text("utf-8"))
        assert auth_data["auth_mode"] == "local_trust"
        assert auth_data["owner"]["username"] == "taro"
        assert auth_data["owner"]["display_name"] == "Taro Yamada"
        assert auth_data["owner"]["bio"] == "A software engineer who loves AI."

    @pytest.mark.asyncio
    async def test_complete_setup_creates_user_profile(self, data_dir: Path):
        """Complete setup with user info creates a user profile markdown file."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "user": {
                        "username": "hanako",
                        "display_name": "Hanako Sato",
                        "bio": "Product manager and cat lover.",
                    },
                },
            )
            assert resp.status_code == 200

        # Verify user profile was created
        profile_path = data_dir / "shared" / "users" / "hanako" / "index.md"
        assert profile_path.exists(), "User profile index.md should be created"

        content = profile_path.read_text("utf-8")
        assert content.startswith("# Hanako Sato")
        assert "Product manager and cat lover." in content

    @pytest.mark.asyncio
    async def test_complete_setup_user_profile_no_bio(self, data_dir: Path):
        """User profile created without bio does not contain bio content."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "ja",
                    "user": {
                        "username": "jiro",
                        "display_name": "Jiro",
                    },
                },
            )
            assert resp.status_code == 200

        # Verify profile exists but has no bio
        profile_path = data_dir / "shared" / "users" / "jiro" / "index.md"
        assert profile_path.exists()

        content = profile_path.read_text("utf-8")
        assert content.startswith("# Jiro")
        # With no bio, the content should only be the heading line
        lines = [line for line in content.strip().splitlines() if line.strip()]
        assert len(lines) == 1, "Profile without bio should only have the heading"

    @pytest.mark.asyncio
    async def test_complete_setup_with_user_and_anima(self, data_dir: Path):
        """Complete setup with both user and anima creates all artifacts."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        with patch.object(app.state.supervisor, "start_all", new_callable=AsyncMock):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "en",
                        "user": {
                            "username": "alice",
                            "display_name": "Alice",
                            "bio": "Researcher.",
                        },
                        "anima": {"name": "sakura"},
                    },
                )
                assert resp.status_code == 200

                # Verify auth.json
                auth_path = data_dir / "auth.json"
                assert auth_path.exists()
                auth_data = json.loads(auth_path.read_text("utf-8"))
                assert auth_data["owner"]["username"] == "alice"

                # Verify anima directory
                anima_dir = data_dir / "animas" / "sakura"
                assert anima_dir.exists()
                assert (anima_dir / "identity.md").exists()

                # Verify user profile
                profile_path = data_dir / "shared" / "users" / "alice" / "index.md"
                assert profile_path.exists()

                # Verify anima_names was updated (accessible via /api/animas)
                resp = await client.get("/api/animas")
                assert resp.status_code != 503, "API should be accessible after setup"

    @pytest.mark.asyncio
    async def test_complete_setup_anima_autostart(self, data_dir: Path):
        """After setup with an anima, anima_names includes the new anima."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        with patch.object(
            app.state.supervisor, "start_all", new_callable=AsyncMock
        ) as mock_start:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "anima": {"name": "kaede"},
                    },
                )
                assert resp.status_code == 200

            # Verify anima_names state was updated
            assert "kaede" in app.state.anima_names

            # Verify start_all was called with the new anima name
            mock_start.assert_awaited_once()
            call_args = mock_start.call_args[0][0]
            assert "kaede" in call_args

    @pytest.mark.asyncio
    async def test_full_wizard_flow_with_user_info(self, data_dir: Path):
        """End-to-end wizard flow including user info, anima, and route switching."""
        _write_config(data_dir, setup_complete=False)
        app = _create_app(data_dir)

        with patch.object(app.state.supervisor, "start_all", new_callable=AsyncMock):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Step 1: Detect locale
                resp = await client.get(
                    "/api/setup/detect-locale",
                    headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                assert resp.status_code == 200
                assert resp.json()["detected"] == "en"

                # Step 2: Complete setup with locale, credentials, anima, and user
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "en",
                        "credentials": {
                            "anthropic": {"api_key": "sk-ant-test-full"},
                        },
                        "anima": {"name": "miku"},
                        "user": {
                            "username": "bob",
                            "display_name": "Bob Smith",
                            "bio": "Team lead.",
                        },
                    },
                )
                assert resp.status_code == 200
                assert resp.json()["status"] == "ok"

                # Step 3: Verify config.json
                invalidate_cache()
                config_raw = json.loads(
                    (data_dir / "config.json").read_text("utf-8")
                )
                assert config_raw["setup_complete"] is True
                assert config_raw["locale"] == "en"

                # Step 4: Verify auth.json
                auth_path = data_dir / "auth.json"
                assert auth_path.exists()
                auth_data = json.loads(auth_path.read_text("utf-8"))
                assert auth_data["owner"]["username"] == "bob"
                assert auth_data["owner"]["display_name"] == "Bob Smith"
                assert auth_data["auth_mode"] == "local_trust"

                # Step 5: Verify anima directory
                assert (data_dir / "animas" / "miku").is_dir()
                assert "miku" in config_raw["animas"]

                # Step 6: Verify user profile
                profile_path = data_dir / "shared" / "users" / "bob" / "index.md"
                assert profile_path.exists()
                profile_content = profile_path.read_text("utf-8")
                assert profile_content.startswith("# Bob Smith")
                assert "Team lead." in profile_content

                # Step 7: Verify route switching — setup blocked, API accessible
                resp = await client.get("/api/setup/environment")
                assert resp.status_code == 403

                resp = await client.get("/api/animas")
                assert resp.status_code != 503
