"""Unit tests for server/routes/config_routes.py config endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from core.config.models import AnimaWorksConfig, CredentialConfig
from server.routes.config_routes import _mask_secrets

# ── Helper ──────────────────────────────────────────────────


def _make_test_app():
    from fastapi import FastAPI

    from server.routes.config_routes import create_config_router

    app = FastAPI()
    router = create_config_router()
    app.include_router(router, prefix="/api")
    return app


# ── _mask_secrets ───────────────────────────────────────────


class TestMaskSecrets:
    def test_mask_key_containing_key(self):
        result = _mask_secrets({"api_key": "abcdefghij"})
        assert result["api_key"] == "abc...ghij"

    def test_mask_key_containing_token(self):
        result = _mask_secrets({"auth_token": "1234567890"})
        assert result["auth_token"] == "123...7890"

    def test_mask_key_containing_secret(self):
        result = _mask_secrets({"client_secret": "abcdefghijkl"})
        assert result["client_secret"] == "abc...ijkl"

    def test_mask_key_containing_password(self):
        result = _mask_secrets({"db_password": "supersecretpw"})
        assert result["db_password"] == "sup...etpw"

    def test_short_secret_gets_triple_star(self):
        result = _mask_secrets({"api_key": "short"})
        assert result["api_key"] == "***"

    def test_exactly_eight_chars_gets_triple_star(self):
        result = _mask_secrets({"api_key": "12345678"})
        assert result["api_key"] == "***"

    def test_empty_secret_stays_empty(self):
        result = _mask_secrets({"api_key": ""})
        assert result["api_key"] == ""

    def test_nine_chars_gets_masked(self):
        result = _mask_secrets({"api_key": "123456789"})
        assert result["api_key"] == "123...6789"

    def test_non_secret_keys_not_masked(self):
        result = _mask_secrets({"name": "alice", "model": "gpt-4o"})
        assert result["name"] == "alice"
        assert result["model"] == "gpt-4o"

    def test_nested_dicts(self):
        result = _mask_secrets({"providers": {"anthropic": {"api_key": "sk-ant-1234567890"}}})
        assert result["providers"]["anthropic"]["api_key"] == "sk-...7890"

    def test_nested_lists(self):
        result = _mask_secrets(
            {
                "items": [
                    {"api_key": "abcdefghij", "name": "test"},
                    {"token": "xyz"},
                ]
            }
        )
        assert result["items"][0]["api_key"] == "abc...ghij"
        assert result["items"][0]["name"] == "test"
        assert result["items"][1]["token"] == "***"

    def test_non_string_secret_value_not_masked(self):
        result = _mask_secrets({"api_key": 12345})
        assert result["api_key"] == 12345

    def test_plain_value_passthrough(self):
        assert _mask_secrets("hello") == "hello"
        assert _mask_secrets(42) == 42
        assert _mask_secrets(None) is None


# ── GET /system/config ──────────────────────────────────────


class TestGetConfig:
    async def test_404_when_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Config file not found"

    async def test_returns_masked_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        config_dir = tmp_path
        config = {
            "model": "claude-sonnet-4-6",
            "providers": {"anthropic": {"api_key": "sk-ant-1234567890"}},
        }
        (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "claude-sonnet-4-6"
        # Secret should be masked
        assert data["providers"]["anthropic"]["api_key"] != "sk-ant-1234567890"
        assert "..." in data["providers"]["anthropic"]["api_key"]

    async def test_500_on_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        (tmp_path / "config.json").write_text("not valid json {{{", encoding="utf-8")

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 500
        assert "Invalid config JSON" in resp.json()["detail"]


class TestAnthropicAuthSettings:
    """API-level checks for settings-page Anthropic auth (migrated from SPA #/setup)."""

    async def test_get_anthropic_auth_uses_config_and_runtime_status(self, monkeypatch):
        config = AnimaWorksConfig(
            credentials={
                "anthropic": CredentialConfig(type="claude_code_login"),
            }
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.is_claude_code_available", return_value=True),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/settings/anthropic-auth")

        assert resp.status_code == 200
        data = resp.json()
        assert data["auth_mode"] == "claude_code_login"
        assert data["config_present"] is True
        assert data["claude_code_available"] is True
        assert data["configured"] is True
        assert data["env_api_key_configured"] is False

    async def test_put_anthropic_auth_saves_api_key(self):
        config = AnimaWorksConfig()
        saved = {}
        app = _make_test_app()
        transport = ASGITransport(app=app)

        def _update_config(fn, *args, **kwargs):
            fn(config)
            saved["config"] = config
            return config

        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.update_config", side_effect=_update_config),
            patch("server.routes.config_routes.is_claude_code_available", return_value=False),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/settings/anthropic-auth",
                    json={"auth_mode": "api_key", "api_key": "sk-ant-test-key"},
                )

        assert resp.status_code == 200
        saved_config = saved["config"]
        assert saved_config.credentials["anthropic"].type == "api_key"
        assert saved_config.credentials["anthropic"].api_key == "sk-ant-test-key"

    async def test_put_anthropic_auth_rejects_missing_api_key(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("server.routes.config_routes.load_config", return_value=AnimaWorksConfig()),
            patch("server.routes.config_routes.is_claude_code_available", return_value=False),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/settings/anthropic-auth",
                    json={"auth_mode": "api_key", "api_key": ""},
                )

        assert resp.status_code == 400


class TestOpenAIAuthSettings:
    async def test_get_openai_auth_uses_config_and_runtime_status(self, monkeypatch):
        config = AnimaWorksConfig(
            credentials={
                "openai": CredentialConfig(type="codex_login"),
            }
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.is_codex_cli_available", return_value=True),
            patch("server.routes.config_routes.is_codex_login_available", return_value=True),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/settings/openai-auth")

        assert resp.status_code == 200
        data = resp.json()
        assert data["auth_mode"] == "codex_login"
        assert data["config_present"] is True
        assert data["codex_cli_available"] is True
        assert data["codex_login_available"] is True
        assert data["configured"] is True

    async def test_put_openai_auth_saves_api_key(self):
        config = AnimaWorksConfig()
        saved = {}
        app = _make_test_app()
        transport = ASGITransport(app=app)

        def _update_config(fn, *args, **kwargs):
            fn(config)
            saved["config"] = config
            return config

        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.update_config", side_effect=_update_config),
            patch("server.routes.config_routes.is_codex_cli_available", return_value=True),
            patch("server.routes.config_routes.is_codex_login_available", return_value=False),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/settings/openai-auth",
                    json={"auth_mode": "api_key", "api_key": "sk-test-openai"},
                )

        assert resp.status_code == 200
        saved_config = saved["config"]
        assert saved_config.credentials["openai"].type == "api_key"
        assert saved_config.credentials["openai"].api_key == "sk-test-openai"

    async def test_put_openai_auth_rejects_missing_codex_login(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)

        with (
            patch("server.routes.config_routes.load_config", return_value=AnimaWorksConfig()),
            patch("server.routes.config_routes.is_codex_cli_available", return_value=True),
            patch("server.routes.config_routes.is_codex_login_available", return_value=False),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/settings/openai-auth",
                    json={"auth_mode": "codex_login"},
                )

        assert resp.status_code == 400
        assert resp.status_code == 400

    async def test_available_models_include_codex_subscription_models(self):
        config = AnimaWorksConfig(credentials={"openai": CredentialConfig(type="codex_login")})
        app = _make_test_app()
        transport = ASGITransport(app=app)
        from core.config.model_discovery import DiscoveredModel

        stub = [
            DiscoveredModel("c:codex/gpt-5.4", "c", "codex/gpt-5.4", "GPT-5.4", "Codex", note="", source="codex-cli"),
            DiscoveredModel(
                "c:codex/gpt-5.4-mini", "c", "codex/gpt-5.4-mini", "GPT-5.4-Mini", "Codex", note="", source="codex-cli"
            ),
            DiscoveredModel(
                "c:codex/gpt-5.3-codex",
                "c",
                "codex/gpt-5.3-codex",
                "GPT-5.3-Codex",
                "Codex",
                note="",
                source="codex-cli",
            ),
        ]
        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.discover_models", return_value=stub),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/available-models")

        assert resp.status_code == 200
        data = resp.json()
        models = data["models"]
        ids = {item["id"] for item in models}

        assert "c:codex/gpt-5.4" in ids
        assert "c:codex/gpt-5.4-mini" in ids
        assert "c:codex/gpt-5.3-codex" in ids
        # New response contract
        assert data["groups"] == ["Codex"]
        assert "generated_at" in data
        assert all(item["mode"] == "c" for item in models)
        assert all(item["credential"] == "codex" for item in models)

    async def test_available_models_include_grok_build_models(self):
        config = AnimaWorksConfig()
        app = _make_test_app()
        transport = ASGITransport(app=app)
        from core.config.model_discovery import DiscoveredModel

        stub = [
            DiscoveredModel("x:grok/grok-4.5", "x", "grok/grok-4.5", "grok-4.5", "Grok", source="grok-cli"),
            DiscoveredModel(
                "x:grok/grok-composer-2.5-fast",
                "x",
                "grok/grok-composer-2.5-fast",
                "grok-composer-2.5-fast",
                "Grok",
                source="grok-cli",
            ),
        ]
        with (
            patch("server.routes.config_routes.load_config", return_value=config),
            patch("server.routes.config_routes.discover_models", return_value=stub),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/system/available-models")

        assert resp.status_code == 200
        models = resp.json()["models"]
        grok_models = {item["id"]: item for item in models if item["group"] == "Grok"}

        assert set(grok_models) == {"x:grok/grok-4.5", "x:grok/grok-composer-2.5-fast"}
        assert all(item["label"] == item["model"].removeprefix("grok/") for item in grok_models.values())


class TestRemovedLocalLLMRoutes:
    async def test_local_llm_routes_return_404(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = [
                await client.get("/api/settings/local-llm"),
                await client.put("/api/settings/local-llm", json={}),
                await client.post("/api/settings/local-llm/apply-role-presets"),
            ]
        assert [response.status_code for response in responses] == [404, 404, 404]


# ── Removed init-status endpoint ────────────────────────────


class TestRemovedInitStatus:
    async def test_init_status_returns_404(self):
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/system/init-status")
        assert response.status_code == 404


class TestRootOwnedConfigWrites:
    async def test_config_value_route_updates_config_through_root(self, tmp_path, monkeypatch):
        from core.config.io import get_config_path, invalidate_cache

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        invalidate_cache()
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/system/config/value",
                json={"key": "system.timezone", "value": "Asia/Tokyo"},
            )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert json.loads(get_config_path().read_text(encoding="utf-8"))["system"]["timezone"] == "Asia/Tokyo"

    async def test_per_anima_heartbeat_interval_routes_reload_to_root_scheduler(self, tmp_path, monkeypatch):
        from core.config.io import invalidate_cache
        from core.paths import get_animas_dir

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        invalidate_cache()
        anima_dir = get_animas_dir() / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("Alice\n", encoding="utf-8")
        (anima_dir / "status.json").write_text("{}", encoding="utf-8")
        app = _make_test_app()
        app.state.supervisor = MagicMock()
        app.state.supervisor.processes = {"alice": MagicMock()}
        app.state.supervisor.send_request = AsyncMock(return_value={})

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/system/config/value",
                json={"key": "animas.alice.heartbeat_interval_minutes", "value": 60},
            )

        assert response.status_code == 200
        assert json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))["heartbeat_interval_minutes"] == 60
        app.state.supervisor.send_request.assert_awaited_once_with("alice", "reschedule_heartbeat", {}, timeout=10.0)

    async def test_legacy_model_config_value_uses_status_store(self, tmp_path, monkeypatch):
        from core.config.io import invalidate_cache
        from core.paths import get_animas_dir

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        invalidate_cache()
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/system/config/value",
                json={"key": "animas.alice.model", "value": "claude-sonnet-4-6"},
            )
        assert response.status_code == 200
        status = json.loads((get_animas_dir() / "alice" / "status.json").read_text(encoding="utf-8"))
        assert status["model"] == "claude-sonnet-4-6"
