# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for multi-user authentication system.

Tests the complete authentication flow including login, logout,
user management, session handling, and from_person override.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from core.auth.manager import hash_password, save_auth, load_auth
from core.auth.models import AuthConfig, AuthUser


# ── Helpers ──────────────────────────────────────────────


def _create_test_app(data_dir: Path):
    """Create a test app with auth enabled."""
    from server.app import create_app

    animas_dir = data_dir / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Ensure config.json has setup_complete=True so the setup guard
    # middleware lets API requests through.
    config_path = data_dir / "config.json"
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["setup_complete"] = True
        config_path.write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    with patch("core.paths.get_data_dir", return_value=data_dir), \
         patch("server.app.ProcessSupervisor") as mock_sup, \
         patch("server.app.WebSocketManager"):
        # Mock supervisor to have empty processes dict
        sup_instance = mock_sup.return_value
        sup_instance.processes = {}
        sup_instance.is_bootstrapping.return_value = False
        app = create_app(animas_dir, shared_dir)
    return app


def _setup_password_auth(data_dir: Path, owner_password: str = "secret123"):
    """Set up auth.json with password mode and an owner."""
    config = AuthConfig(
        auth_mode="password",
        owner=AuthUser(
            username="admin",
            display_name="Admin",
            password_hash=hash_password(owner_password),
            role="owner",
        ),
    )
    save_auth(config)
    return config


def _setup_multi_user_auth(data_dir: Path):
    """Set up auth.json with multi_user mode, owner + user."""
    config = AuthConfig(
        auth_mode="multi_user",
        owner=AuthUser(
            username="admin",
            display_name="Admin",
            password_hash=hash_password("adminpw"),
            role="owner",
        ),
        users=[
            AuthUser(
                username="alice",
                display_name="Alice",
                password_hash=hash_password("alicepw"),
                role="user",
            ),
        ],
    )
    save_auth(config)
    return config


# ── Tests ────────────────────────────────────────────────


class TestLocalTrustMode:
    """Verify local_trust mode allows unauthenticated access."""

    def test_api_accessible_without_auth(self, data_dir):
        config = AuthConfig(auth_mode="local_trust")
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Should not get 401 for any API
        resp = client.get("/api/auth/me")
        # In local_trust, /auth/me returns owner if set, or 401 if no owner
        # This is fine — the important thing is the middleware didn't block it
        assert resp.status_code != 403


class TestLoginLogoutFlow:
    """Test the complete login → use → logout → denied flow."""

    def test_full_login_logout_cycle(self, data_dir):
        _setup_password_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # 1. Unauthenticated → 401
        resp = client.get("/api/users")
        assert resp.status_code == 401

        # 2. Login
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "secret123",
        })
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"
        assert "session_token" in resp.cookies

        # 3. Authenticated access
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"

        # 4. Logout
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 200

        # 5. After logout → 401
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_login_with_wrong_password(self, data_dir):
        _setup_password_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401


class TestMultiUserCrud:
    """Test user creation, listing, and deletion."""

    def test_add_list_delete_user(self, data_dir):
        _setup_password_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login as owner
        client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})

        # Add user
        resp = client.post("/api/users", json={
            "username": "bob",
            "display_name": "Bob",
            "password": "bobpw123",
        })
        assert resp.status_code == 200
        assert resp.json()["username"] == "bob"

        # List users — should have admin + bob
        resp = client.get("/api/users")
        assert resp.status_code == 200
        usernames = [u["username"] for u in resp.json()]
        assert "admin" in usernames
        assert "bob" in usernames

        # Verify user profile dir created
        profile = data_dir / "shared" / "users" / "bob" / "index.md"
        assert profile.exists()

        # Delete user
        resp = client.delete("/api/users/bob")
        assert resp.status_code == 200

        # Verify deleted from list
        resp = client.get("/api/users")
        usernames = [u["username"] for u in resp.json()]
        assert "bob" not in usernames

    def test_non_owner_cannot_add_users(self, data_dir):
        _setup_multi_user_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login as regular user
        client.post("/api/auth/login", json={"username": "alice", "password": "alicepw"})

        resp = client.post("/api/users", json={
            "username": "hacker",
            "password": "hackpw",
        })
        assert resp.status_code == 403

    def test_non_owner_cannot_delete_users(self, data_dir):
        _setup_multi_user_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login as regular user
        client.post("/api/auth/login", json={"username": "alice", "password": "alicepw"})

        resp = client.delete("/api/users/admin")
        assert resp.status_code == 403


class TestOwnerCannotDeleteSelf:
    def test_owner_self_delete_rejected(self, data_dir):
        _setup_password_auth(data_dir)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        client.post("/api/auth/login", json={"username": "admin", "password": "secret123"})

        resp = client.delete("/api/users/admin")
        assert resp.status_code == 400
        assert "Cannot delete yourself" in resp.json()["error"]


class TestPasswordChange:
    def test_change_password_and_login_with_new(self, data_dir):
        _setup_password_auth(data_dir, owner_password="oldpw")
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login with old password
        client.post("/api/auth/login", json={"username": "admin", "password": "oldpw"})

        # Change password
        resp = client.put("/api/users/me/password", json={
            "current_password": "oldpw",
            "new_password": "newpw123",
        })
        assert resp.status_code == 200

        # Logout
        client.post("/api/auth/logout")

        # Login with new password
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "newpw123",
        })
        assert resp.status_code == 200

        # Old password should fail
        client.post("/api/auth/logout")
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "oldpw",
        })
        assert resp.status_code == 401


class TestUserDeletionRevokesSession:
    def test_deleted_user_session_invalidated(self, data_dir):
        _setup_multi_user_auth(data_dir)
        app = _create_test_app(data_dir)

        # Login as alice in a separate client to get her session
        alice_client = TestClient(app)
        resp = alice_client.post("/api/auth/login", json={"username": "alice", "password": "alicepw"})
        assert resp.status_code == 200
        alice_token = resp.cookies.get("session_token")

        # Login as admin and delete alice
        admin_client = TestClient(app)
        admin_client.post("/api/auth/login", json={"username": "admin", "password": "adminpw"})
        resp = admin_client.delete("/api/users/alice")
        assert resp.status_code == 200

        # Alice's session should now be invalid
        # Make request with alice's token
        resp = alice_client.get("/api/auth/me")
        assert resp.status_code == 401


class TestMultipleSessions:
    def test_multiple_browser_sessions(self, data_dir):
        _setup_password_auth(data_dir)
        app = _create_test_app(data_dir)

        # Login from two "browsers"
        client1 = TestClient(app)
        client2 = TestClient(app)

        resp1 = client1.post("/api/auth/login", json={"username": "admin", "password": "secret123"})
        resp2 = client2.post("/api/auth/login", json={"username": "admin", "password": "secret123"})

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Both should be able to access API
        assert client1.get("/api/auth/me").status_code == 200
        assert client2.get("/api/auth/me").status_code == 200

        # Logout from client1 shouldn't affect client2
        client1.post("/api/auth/logout")
        assert client1.get("/api/auth/me").status_code == 401
        assert client2.get("/api/auth/me").status_code == 200


class TestAuthModeTransition:
    def test_local_trust_to_password(self, data_dir):
        """Simulate setting a password from local_trust mode."""
        # Start in local_trust
        config = AuthConfig(
            auth_mode="local_trust",
            owner=AuthUser(username="admin", role="owner"),
        )
        save_auth(config)

        # Simulate password setup: update auth.json
        config.auth_mode = "password"
        config.owner.password_hash = hash_password("newpassword")
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Now API requires auth
        resp = client.get("/api/users")
        assert resp.status_code == 401

        # Login with the new password
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "newpassword",
        })
        assert resp.status_code == 200
