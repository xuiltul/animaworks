"""Unit tests for server/routes/users.py — User management API."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


def _login_as_owner(client, password="secret123"):
    resp = client.post("/api/auth/login", json={"username": "admin", "password": password})
    assert resp.status_code == 200
    return resp


class TestListUsers:
    def test_list_users(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
            users=[AuthUser(username="alice", password_hash=hash_password("pw"), role="user")],
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.get("/api/users")
        assert resp.status_code == 200
        users = resp.json()
        assert len(users) == 2
        usernames = [u["username"] for u in users]
        assert "admin" in usernames
        assert "alice" in usernames
        # No password_hash in response
        for u in users:
            assert "password_hash" not in u


class TestAddUser:
    def test_add_user_as_owner(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.post("/api/users", json={
            "username": "newuser",
            "display_name": "New User",
            "password": "pass1234",
        })
        assert resp.status_code == 200
        assert resp.json()["username"] == "newuser"

    def test_add_user_as_non_owner_rejected(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
            users=[AuthUser(username="user1", password_hash=hash_password("userpw"), role="user")],
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Login as non-owner
        client.post("/api/auth/login", json={"username": "user1", "password": "userpw"})

        resp = client.post("/api/users", json={
            "username": "hacker",
            "password": "pass1234",
        })
        assert resp.status_code == 403

    def test_add_duplicate_user_rejected(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.post("/api/users", json={"username": "admin", "password": "pw1234"})
        assert resp.status_code == 409


class TestDeleteUser:
    def test_delete_user_as_owner(self, data_dir):
        config = AuthConfig(
            auth_mode="multi_user",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
            users=[AuthUser(username="alice", password_hash=hash_password("pw"), role="user")],
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.delete("/api/users/alice")
        assert resp.status_code == 200

    def test_delete_self_rejected(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.delete("/api/users/admin")
        assert resp.status_code == 400


class TestChangePassword:
    def test_change_password(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("oldpw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        client.post("/api/auth/login", json={"username": "admin", "password": "oldpw"})

        resp = client.put("/api/users/me/password", json={
            "current_password": "oldpw",
            "new_password": "newpw123",
        })
        assert resp.status_code == 200

        # Login with new password
        client.post("/api/auth/logout")
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "newpw123"})
        assert resp.status_code == 200

    def test_change_password_wrong_current(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("correctpw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        client.post("/api/auth/login", json={"username": "admin", "password": "correctpw"})

        resp = client.put("/api/users/me/password", json={
            "current_password": "wrongpw",
            "new_password": "newpw",
        })
        assert resp.status_code == 401


class TestPasswordMaxLength:
    def test_add_user_password_too_long(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("secret123"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        _login_as_owner(client)

        resp = client.post("/api/users", json={
            "username": "newuser",
            "password": "a" * 129,
        })
        assert resp.status_code == 400
        assert "128" in resp.json()["error"]

    def test_change_password_too_long(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("oldpw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        client.post("/api/auth/login", json={"username": "admin", "password": "oldpw"})

        resp = client.put("/api/users/me/password", json={
            "current_password": "oldpw",
            "new_password": "a" * 129,
        })
        assert resp.status_code == 400
        assert "128" in resp.json()["error"]
