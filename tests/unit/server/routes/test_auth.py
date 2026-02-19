"""Unit tests for server/routes/auth.py — Auth API endpoints."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.auth.manager import hash_password, save_auth
from core.auth.models import AuthConfig, AuthUser


def _create_test_app(data_dir: Path):
    """Create a minimal test app with auth routes."""
    import json
    from unittest.mock import MagicMock
    from server.app import create_app

    animas_dir = data_dir / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Ensure config.json has setup_complete so setup_guard doesn't block API
    config_path = data_dir / "config.json"
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config_data = {}
    config_data["setup_complete"] = True
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    from core.config import invalidate_cache
    invalidate_cache()

    with patch("core.paths.get_data_dir", return_value=data_dir), \
         patch("server.app.ProcessSupervisor"), \
         patch("server.app.WebSocketManager"):
        app = create_app(animas_dir, shared_dir)
    return app


class TestLogin:
    def test_login_success(self, data_dir):
        pw_hash = hash_password("secret123")
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=pw_hash, role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["role"] == "owner"
        assert "session_token" in resp.cookies

    def test_login_wrong_password(self, data_dir):
        pw_hash = hash_password("secret123")
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=pw_hash, role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_unknown_user(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "nobody", "password": "pw"})
        assert resp.status_code == 401

    def test_login_disabled_in_local_trust(self, data_dir):
        config = AuthConfig(auth_mode="local_trust")
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "pw"})
        assert resp.status_code == 400


class TestLogout:
    def test_logout_clears_session(self, data_dir):
        pw_hash = hash_password("secret123")
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=pw_hash, role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login first
        login_resp = client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
        assert login_resp.status_code == 200

        # Logout
        logout_resp = client.post("/api/auth/logout")
        assert logout_resp.status_code == 200

        # /auth/me should now fail
        me_resp = client.get("/api/auth/me")
        assert me_resp.status_code == 401


class TestMe:
    def test_me_authenticated(self, data_dir):
        pw_hash = hash_password("secret123")
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", display_name="Admin", password_hash=pw_hash, role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)

        client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["display_name"] == "Admin"
        assert data["role"] == "owner"

    def test_me_unauthenticated(self, data_dir):
        config = AuthConfig(auth_mode="password")
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_me_local_trust_returns_owner(self, data_dir):
        config = AuthConfig(
            auth_mode="local_trust",
            owner=AuthUser(username="admin", role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"

    def test_me_includes_auth_mode(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        client.post("/api/auth/login", json={"username": "admin", "password": "pw"})
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["auth_mode"] == "password"

    def test_me_local_trust_includes_auth_mode(self, data_dir):
        config = AuthConfig(
            auth_mode="local_trust",
            owner=AuthUser(username="admin", role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["auth_mode"] == "local_trust"


class TestLoginPasswordMaxLength:
    def test_login_rejects_password_over_128(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        long_pw = "a" * 129
        resp = client.post("/api/auth/login", json={"username": "admin", "password": long_pw})
        assert resp.status_code == 401

    def test_login_accepts_password_at_128(self, data_dir):
        pw = "a" * 128
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password(pw), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "admin", "password": pw})
        assert resp.status_code == 200
