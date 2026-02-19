"""Unit tests for server/routes/config_routes.py — Config & init-status endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

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

    def test_nine_chars_gets_masked(self):
        result = _mask_secrets({"api_key": "123456789"})
        assert result["api_key"] == "123...6789"

    def test_non_secret_keys_not_masked(self):
        result = _mask_secrets({"name": "alice", "model": "gpt-4o"})
        assert result["name"] == "alice"
        assert result["model"] == "gpt-4o"

    def test_nested_dicts(self):
        result = _mask_secrets({
            "providers": {
                "anthropic": {"api_key": "sk-ant-1234567890"}
            }
        })
        assert result["providers"]["anthropic"]["api_key"] == "sk-...7890"

    def test_nested_lists(self):
        result = _mask_secrets({
            "items": [
                {"api_key": "abcdefghij", "name": "test"},
                {"token": "xyz"},
            ]
        })
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
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Config file not found"

    async def test_returns_masked_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".animaworks"
        config_dir.mkdir()
        config = {
            "model": "claude-sonnet-4-20250514",
            "providers": {
                "anthropic": {"api_key": "sk-ant-1234567890"}
            },
        }
        (config_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "claude-sonnet-4-20250514"
        # Secret should be masked
        assert data["providers"]["anthropic"]["api_key"] != "sk-ant-1234567890"
        assert "..." in data["providers"]["anthropic"]["api_key"]

    async def test_500_on_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".animaworks"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            "not valid json {{{", encoding="utf-8"
        )

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/config")
        assert resp.status_code == 500
        assert "Invalid config JSON" in resp.json()["detail"]


# ── GET /system/init-status ─────────────────────────────────


class TestInitStatus:
    async def test_nothing_initialized(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Remove API keys from environment
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        assert resp.status_code == 200
        data = resp.json()
        # Backward-compatible fields
        assert data["config_exists"] is False
        assert data["animas_count"] == 0
        assert data["initialized"] is False
        assert data["api_keys"]["anthropic"] is False
        assert data["api_keys"]["openai"] is False
        assert data["api_keys"]["google"] is False
        # New checks array
        assert "checks" in data
        checks = data["checks"]
        assert isinstance(checks, list)
        assert len(checks) >= 1
        labels = {c["label"] for c in checks}
        assert "設定ファイル" in labels
        assert "Anima登録" in labels
        assert "初期化完了" in labels
        # All checks should be not-ok when nothing is initialized
        config_check = next(c for c in checks if c["label"] == "設定ファイル")
        assert config_check["ok"] is False
        init_check = next(c for c in checks if c["label"] == "初期化完了")
        assert init_check["ok"] is False

    async def test_with_config_and_animas(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        base_dir = tmp_path / ".animaworks"
        base_dir.mkdir()
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        # Create an anima with identity.md
        animas_dir = base_dir / "animas"
        animas_dir.mkdir()
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("# Alice", encoding="utf-8")

        # Also create shared dir
        shared_dir = base_dir / "shared"
        shared_dir.mkdir()

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        # Backward-compatible fields
        assert data["config_exists"] is True
        assert data["animas_count"] == 1
        assert data["initialized"] is True
        assert data["shared_dir_exists"] is True
        # New checks array
        checks = data["checks"]
        config_check = next(c for c in checks if c["label"] == "設定ファイル")
        assert config_check["ok"] is True
        anima_check = next(c for c in checks if c["label"] == "Anima登録")
        assert anima_check["ok"] is True
        assert anima_check["detail"] == "1名"
        shared_check = next(c for c in checks if c["label"] == "共有ディレクトリ")
        assert shared_check["ok"] is True
        init_check = next(c for c in checks if c["label"] == "初期化完了")
        assert init_check["ok"] is True

    async def test_api_key_detection(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        # Backward-compatible fields
        assert data["api_keys"]["anthropic"] is True
        assert data["api_keys"]["openai"] is True
        assert data["api_keys"]["google"] is False
        # New checks array - API key checks
        checks = data["checks"]
        anthropic_check = next(c for c in checks if c["label"] == "Anthropic APIキー")
        assert anthropic_check["ok"] is True
        openai_check = next(c for c in checks if c["label"] == "OpenAI APIキー")
        assert openai_check["ok"] is True
        google_check = next(c for c in checks if c["label"] == "Google APIキー")
        assert google_check["ok"] is False

    async def test_animas_without_identity_not_counted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        base_dir = tmp_path / ".animaworks"
        base_dir.mkdir()
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        animas_dir = base_dir / "animas"
        animas_dir.mkdir()
        # Directory without identity.md should not be counted
        (animas_dir / "incomplete").mkdir()

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        assert data["animas_count"] == 0
        assert data["initialized"] is False
        # checks array should reflect zero animas
        checks = data["checks"]
        anima_check = next(c for c in checks if c["label"] == "Anima登録")
        assert anima_check["ok"] is False
        assert anima_check["detail"] == "0名"

    async def test_checks_array_has_all_expected_labels(self, tmp_path, monkeypatch):
        """Verify that the checks array contains all expected labels."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        checks = data["checks"]
        expected_labels = {
            "設定ファイル",
            "Anima登録",
            "共有ディレクトリ",
            "Anthropic APIキー",
            "OpenAI APIキー",
            "Google APIキー",
            "初期化完了",
        }
        actual_labels = {c["label"] for c in checks}
        assert expected_labels == actual_labels

    async def test_checks_items_have_ok_field(self, tmp_path, monkeypatch):
        """Every check item must have at least 'label' and 'ok' fields."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        for check in data["checks"]:
            assert "label" in check, f"Missing 'label' in check: {check}"
            assert "ok" in check, f"Missing 'ok' in check: {check}"
            assert isinstance(check["ok"], bool), f"'ok' should be bool: {check}"

    async def test_animas_detail_with_multiple(self, tmp_path, monkeypatch):
        """Anima check detail should show correct count with multiple animas."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        base_dir = tmp_path / ".animaworks"
        base_dir.mkdir()
        (base_dir / "config.json").write_text("{}", encoding="utf-8")

        animas_dir = base_dir / "animas"
        animas_dir.mkdir()
        for name in ("alice", "bob", "charlie"):
            d = animas_dir / name
            d.mkdir()
            (d / "identity.md").write_text(f"# {name}", encoding="utf-8")

        app = _make_test_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/system/init-status")
        data = resp.json()
        assert data["animas_count"] == 3
        checks = data["checks"]
        anima_check = next(c for c in checks if c["label"] == "Anima登録")
        assert anima_check["detail"] == "3名"
        assert anima_check["ok"] is True
