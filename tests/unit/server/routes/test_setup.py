"""Unit tests for server/routes/setup.py — Setup wizard API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from server.routes.setup import (
    AVAILABLE_LOCALES,
    AVAILABLE_PROVIDERS,
    _normalize_locale,
    _parse_accept_language,
    create_setup_router,
)


# ── Helper to build a minimal FastAPI app with setup router ──


def _make_test_app(
    setup_complete: bool = False,
    animas_dir: Path | None = None,
    supervisor: object | None = None,
):
    from fastapi import FastAPI

    app = FastAPI()
    app.state.setup_complete = setup_complete
    app.state.animas_dir = animas_dir or Path("/tmp/_test_nonexistent_animas")
    app.state.supervisor = supervisor or AsyncMock()
    router = create_setup_router()
    app.include_router(router)
    return app


# ── _normalize_locale ─────────────────────────────────────

_ZH_SIMPLIFIED = {"cn", "hans", "sg"}
_ZH_TRADITIONAL = {"tw", "hant", "hk", "mo"}


class TestNormalizeLocale:
    def test_bare_zh_defaults_to_simplified(self):
        assert _normalize_locale("zh", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-CN"

    def test_zh_cn(self):
        assert _normalize_locale("zh-cn", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-CN"

    def test_zh_tw(self):
        assert _normalize_locale("zh-tw", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-TW"

    def test_zh_hans(self):
        assert _normalize_locale("zh-hans", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-CN"

    def test_zh_hant(self):
        assert _normalize_locale("zh-hant", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-TW"

    def test_zh_hk(self):
        assert _normalize_locale("zh-hk", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-TW"

    def test_zh_mo(self):
        assert _normalize_locale("zh-mo", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-TW"

    def test_zh_sg(self):
        assert _normalize_locale("zh-sg", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-CN"

    def test_en_us_normalizes_to_primary(self):
        assert _normalize_locale("en-us", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "en"

    def test_fr_passthrough(self):
        assert _normalize_locale("fr", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "fr"

    def test_zh_tw_underscore(self):
        assert _normalize_locale("zh_tw", _ZH_SIMPLIFIED, _ZH_TRADITIONAL) == "zh-TW"


# ── _parse_accept_language ───────────────────────────────


class TestParseAcceptLanguage:
    def test_empty_returns_ja(self):
        assert _parse_accept_language("") == "ja"

    def test_ja_header(self):
        assert _parse_accept_language("ja") == "ja"

    def test_en_header(self):
        assert _parse_accept_language("en") == "en"

    def test_en_us_normalizes_to_en(self):
        assert _parse_accept_language("en-US") == "en"

    def test_weighted_prefers_higher_quality(self):
        assert _parse_accept_language("en;q=0.8,ja;q=0.9") == "ja"

    def test_weighted_en_first(self):
        assert _parse_accept_language("en;q=0.9,ja;q=0.8") == "en"

    def test_no_quality_defaults_to_1(self):
        # "ja" without q= has quality 1.0, "en;q=0.5" has 0.5
        assert _parse_accept_language("ja,en;q=0.5") == "ja"

    def test_unknown_locale_falls_back_to_ja(self):
        assert _parse_accept_language("sw,cy") == "ja"

    def test_mixed_known_unknown(self):
        assert _parse_accept_language("fr;q=1.0,en;q=0.8") == "fr"

    def test_complex_header(self):
        header = "ja;q=0.9,en-US;q=0.8,en;q=0.7,fr;q=0.5"
        assert _parse_accept_language(header) == "ja"

    def test_invalid_quality_ignored(self):
        # "en;q=notanumber" → q=0.0, "ja" → q=1.0
        assert _parse_accept_language("en;q=notanumber,ja") == "ja"

    def test_zh_cn_header(self):
        assert _parse_accept_language("zh-CN") == "zh-CN"

    def test_zh_tw_header(self):
        assert _parse_accept_language("zh-TW") == "zh-TW"

    def test_zh_hans_maps_to_simplified(self):
        assert _parse_accept_language("zh-Hans") == "zh-CN"

    def test_zh_hant_maps_to_traditional(self):
        assert _parse_accept_language("zh-Hant") == "zh-TW"

    def test_zh_hk_maps_to_traditional(self):
        assert _parse_accept_language("zh-HK") == "zh-TW"

    def test_bare_zh_defaults_to_simplified(self):
        assert _parse_accept_language("zh") == "zh-CN"

    def test_korean_header(self):
        assert _parse_accept_language("ko") == "ko"

    def test_spanish_header(self):
        assert _parse_accept_language("es") == "es"


# ── GET /api/setup/environment ───────────────────────────


class TestGetEnvironment:
    async def test_returns_environment_info(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("shutil.which", return_value="/usr/bin/claude"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/setup/environment")

        assert resp.status_code == 200
        data = resp.json()
        assert data["claude_code_available"] is True
        assert data["locale"] == "ja"
        assert data["providers"] == AVAILABLE_PROVIDERS
        assert data["available_locales"] == AVAILABLE_LOCALES

    async def test_claude_not_available(self):
        mock_config = MagicMock()
        mock_config.locale = "en"

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("shutil.which", return_value=None),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/setup/environment")

        data = resp.json()
        assert data["claude_code_available"] is False
        assert data["locale"] == "en"


# ── GET /api/setup/detect-locale ─────────────────────────


class TestDetectLocale:
    async def test_ja_header(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"accept-language": "ja"},
            )

        data = resp.json()
        assert data["detected"] == "ja"
        assert data["available"] == AVAILABLE_LOCALES

    async def test_en_us_header(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/setup/detect-locale",
                headers={"accept-language": "en-US,en;q=0.9"},
            )

        data = resp.json()
        assert data["detected"] == "en"

    async def test_no_header_defaults_ja(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/setup/detect-locale")

        data = resp.json()
        assert data["detected"] == "ja"


# ── POST /api/setup/validate-key ─────────────────────────


class TestValidateKey:
    async def test_anthropic_valid(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with patch("server.routes.setup._validate_anthropic_key", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = {"valid": True, "message": "API key is valid"}
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "anthropic", "api_key": "sk-test"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True

    async def test_anthropic_invalid(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with patch("server.routes.setup._validate_anthropic_key", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = {"valid": False, "message": "Invalid API key"}
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "anthropic", "api_key": "bad-key"},
                )

        data = resp.json()
        assert data["valid"] is False

    async def test_openai_validation(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with patch("server.routes.setup._validate_openai_key", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = {"valid": True, "message": "API key is valid"}
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "openai", "api_key": "sk-openai"},
                )

        data = resp.json()
        assert data["valid"] is True

    async def test_google_validation(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with patch("server.routes.setup._validate_google_key", new_callable=AsyncMock) as mock_val:
            mock_val.return_value = {"valid": True, "message": "API key is valid"}
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/validate-key",
                    json={"provider": "google", "api_key": "google-key"},
                )

        data = resp.json()
        assert data["valid"] is True

    async def test_ollama_no_key_needed(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/validate-key",
                json={"provider": "ollama", "api_key": ""},
            )

        data = resp.json()
        assert data["valid"] is True
        assert "does not require" in data["message"]

    async def test_unknown_provider(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/validate-key",
                json={"provider": "unknown_provider", "api_key": "key"},
            )

        data = resp.json()
        assert data["valid"] is False
        assert "Unknown provider" in data["message"]


# ── POST /api/setup/complete ─────────────────────────────


class TestCompleteSetup:
    async def test_basic_complete(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config") as mock_save,
            patch("core.config.invalidate_cache") as mock_invalidate,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={"locale": "en", "credentials": {}},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert mock_config.locale == "en"
        assert mock_config.setup_complete is True
        mock_save.assert_called_once_with(mock_config)
        mock_invalidate.assert_called_once()

    async def test_complete_with_credentials(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {
                            "anthropic": {"api_key": "sk-test-key"},
                        },
                    },
                )

        assert resp.status_code == 200
        assert "anthropic" in mock_config.credentials

    async def test_complete_with_blank_anima(self, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.anima_factory.create_blank", return_value=anima_dir),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "anima": {"name": "alice"},
                    },
                )

        assert resp.status_code == 200

    async def test_complete_updates_app_state(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app(setup_complete=False)
        assert app.state.setup_complete is False

        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={"locale": "ja", "credentials": {}},
                )

        assert resp.status_code == 200
        assert app.state.setup_complete is True

    async def test_complete_anima_already_exists(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.paths.get_animas_dir", return_value=Path("/tmp/test/animas")),
            patch("core.anima_factory.create_blank", side_effect=FileExistsError("already exists")),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "anima": {"name": "alice"},
                    },
                )

        # Should succeed despite FileExistsError (anima creation is skipped)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_complete_anima_creation_fails(self):
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.paths.get_animas_dir", return_value=Path("/tmp/test/animas")),
            patch("core.anima_factory.create_blank", side_effect=RuntimeError("disk error")),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "anima": {"name": "alice"},
                    },
                )

        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data

    async def test_complete_with_leader_name_only(self):
        """Test that complete_setup works with just a name (no template, no identity_md)."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.paths.get_animas_dir", return_value=Path("/tmp/test/animas")),
            patch("core.anima_factory.create_blank") as mock_create,
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "anima": {"name": "sakura"},
                    },
                )

        assert resp.status_code == 200
        mock_create.assert_called_once_with(Path("/tmp/test/animas"), "sakura")
        assert "sakura" in mock_config.animas

    async def test_complete_with_user_info(self, tmp_path: Path):
        """POST with user info should call save_auth and create user profile dir."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        shared_dir = tmp_path / "shared"
        app = _make_test_app(animas_dir=tmp_path / "animas")
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.auth.manager.save_auth") as mock_save_auth,
            patch("core.paths.get_shared_dir", return_value=shared_dir),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "user": {
                            "username": "taro",
                            "display_name": "Taro",
                            "bio": "test bio",
                        },
                    },
                )

        assert resp.status_code == 200
        mock_save_auth.assert_called_once()
        saved_config = mock_save_auth.call_args[0][0]
        assert saved_config.owner.username == "taro"
        assert saved_config.owner.display_name == "Taro"

        # User profile directory should have been created
        user_dir = shared_dir / "users" / "taro"
        assert user_dir.exists()

    async def test_complete_with_user_creates_profile(self, tmp_path: Path):
        """Verify shared/users/{username}/index.md is created with correct content."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        shared_dir = tmp_path / "shared"
        app = _make_test_app(animas_dir=tmp_path / "animas")
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.auth.manager.save_auth"),
            patch("core.paths.get_shared_dir", return_value=shared_dir),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "user": {
                            "username": "taro",
                            "display_name": "Taro",
                            "bio": "test bio",
                        },
                    },
                )

        assert resp.status_code == 200
        profile_path = shared_dir / "users" / "taro" / "index.md"
        assert profile_path.exists()
        content = profile_path.read_text(encoding="utf-8")
        assert "# Taro" in content
        assert "test bio" in content

    async def test_complete_with_user_no_bio(self, tmp_path: Path):
        """Verify profile is created without bio section when bio is empty."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        shared_dir = tmp_path / "shared"
        app = _make_test_app(animas_dir=tmp_path / "animas")
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.auth.manager.save_auth"),
            patch("core.paths.get_shared_dir", return_value=shared_dir),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "user": {
                            "username": "hanako",
                            "display_name": "Hanako",
                            "bio": "",
                        },
                    },
                )

        assert resp.status_code == 200
        profile_path = shared_dir / "users" / "hanako" / "index.md"
        assert profile_path.exists()
        content = profile_path.read_text(encoding="utf-8")
        assert "# Hanako" in content
        # Bio section should not be present
        assert content.strip() == "# Hanako"

    async def test_complete_triggers_anima_start(self, tmp_path: Path):
        """Verify supervisor.start_all() is called after creating an anima."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        # Create an anima directory with identity.md so rescan picks it up
        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "sakura"
        anima_dir.mkdir(parents=True)
        (anima_dir / "identity.md").write_text("# Sakura", encoding="utf-8")

        mock_supervisor = AsyncMock()
        app = _make_test_app(animas_dir=animas_dir, supervisor=mock_supervisor)
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config"),
            patch("core.config.invalidate_cache"),
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.anima_factory.create_blank"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={
                        "locale": "ja",
                        "credentials": {},
                        "anima": {"name": "sakura"},
                    },
                )

        assert resp.status_code == 200
        mock_supervisor.start_all.assert_called_once_with(["sakura"])

    async def test_complete_without_user(self):
        """Verify old behavior still works without user field."""
        mock_config = MagicMock()
        mock_config.locale = "ja"
        mock_config.credentials = {}
        mock_config.animas = {}

        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.config.save_config") as mock_save,
            patch("core.config.invalidate_cache"),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/setup/complete",
                    json={"locale": "en", "credentials": {}},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_save.assert_called_once()
        # save_auth should NOT have been called (no user field)
        assert mock_config.setup_complete is True
