# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Hybrid Localhost Trust authentication feature.

Verifies the full auth flow where localhost connections can bypass password
authentication when ``trust_localhost`` is enabled in auth.json, including
CSRF/Host header validation and backward compatibility.

Note: Starlette's ``TestClient`` sets ``request.client.host`` to
``"testclient"`` (not a real IP), so ``_is_localhost()`` naturally returns
``False``.  To simulate genuine localhost connections we patch only that
narrow function; all CSRF and Host-header checks in
``_is_safe_localhost_request`` remain unpatched and execute against real
header values.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from core.auth.manager import hash_password, save_auth, load_auth
from core.auth.models import AuthConfig, AuthUser


# ── Helpers ──────────────────────────────────────────────


def _create_test_app(data_dir: Path):
    """Create a test app with auth enabled.

    Mirrors the pattern from ``test_multi_user_auth_e2e.py``.
    """
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
        sup_instance = mock_sup.return_value
        sup_instance.processes = {}
        sup_instance.is_bootstrapping.return_value = False
        app = create_app(animas_dir, shared_dir)
    return app


def _setup_password_auth(
    data_dir: Path,
    *,
    trust_localhost: bool = True,
    owner_password: str = "secret123",
) -> AuthConfig:
    """Set up auth.json in password mode with the given trust_localhost flag."""
    config = AuthConfig(
        auth_mode="password",
        trust_localhost=trust_localhost,
        owner=AuthUser(
            username="admin",
            display_name="Admin",
            password_hash=hash_password(owner_password),
            role="owner",
        ),
    )
    save_auth(config)
    return config


# ── Tests ────────────────────────────────────────────────


class TestLocalhostTrustEnabled:
    """Password mode + trust_localhost=True: localhost requests bypass auth."""

    def test_animas_list_accessible_from_localhost(self, data_dir):
        """GET /api/animas should succeed without auth cookie from localhost."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "localhost:8000"},
            )

        # The middleware lets the request through; the endpoint returns
        # the anima list (empty in this test, so 200 with []).
        assert resp.status_code == 200

    def test_csrf_origin_mismatch_blocks_even_with_trust(self, data_dir):
        """Localhost trust must NOT bypass when Origin header is non-local."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={
                    "host": "localhost:8000",
                    "origin": "http://evil.com",
                },
            )

        # CSRF check fails -> middleware blocks -> 401
        assert resp.status_code == 401

    def test_dns_rebinding_blocks_even_with_trust(self, data_dir):
        """Localhost trust must NOT bypass when Host header is non-local."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "evil.com:8000"},
            )

        # Host check fails -> middleware blocks -> 401
        assert resp.status_code == 401


class TestLocalhostTrustDisabled:
    """Password mode + trust_localhost=False: no localhost bypass."""

    def test_unauthenticated_request_returns_401(self, data_dir):
        """Without localhost trust, unauthenticated API calls must get 401."""
        _setup_password_auth(data_dir, trust_localhost=False)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Even if the client were from localhost, trust_localhost=False
        # means the bypass is disabled entirely.
        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "localhost:8000"},
            )

        assert resp.status_code == 401

    def test_remote_ip_without_trust_returns_401(self, data_dir):
        """Remote IP + trust disabled → 401 (natural TestClient behavior)."""
        _setup_password_auth(data_dir, trust_localhost=False)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # No patching at all: TestClient's "testclient" host is not loopback.
        resp = client.get("/api/animas")
        assert resp.status_code == 401


class TestLocalhostTrustWithLogin:
    """Login flow still works even when localhost trust is enabled."""

    def test_login_endpoint_works_with_trust_enabled(self, data_dir):
        """Login should succeed via password even when localhost trust is on."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login is on the auth whitelist, so it works regardless.
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "secret123",
        })
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"
        assert "session_token" in resp.cookies

    def test_session_cookie_valid_alongside_localhost_trust(self, data_dir):
        """A valid session cookie should grant access even from non-localhost."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login to get a session cookie
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "secret123",
        })
        assert resp.status_code == 200

        # Access API from non-localhost (natural TestClient behavior,
        # _is_localhost returns False for "testclient" host).
        # The session cookie should still grant access.
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"

    def test_full_login_logout_with_trust_enabled(self, data_dir):
        """Full login -> use -> logout -> denied cycle, trust_localhost on."""
        _setup_password_auth(data_dir, trust_localhost=True)
        app = _create_test_app(data_dir)
        client = TestClient(app)

        # 1. Without localhost bypass and no cookie -> 401
        resp = client.get("/api/animas")
        assert resp.status_code == 401

        # 2. Login
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "secret123",
        })
        assert resp.status_code == 200

        # 3. Authenticated request
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200

        # 4. Logout
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 200

        # 5. After logout -> 401 (no cookie, no localhost bypass)
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


class TestBackwardCompatibility:
    """auth.json without ``trust_localhost`` key defaults to True."""

    def test_missing_trust_localhost_key_defaults_to_true(self, data_dir):
        """An auth.json written before the feature should default trust on."""
        # Write auth.json without the trust_localhost field, simulating
        # a pre-feature auth.json file.
        auth_path = data_dir / "auth.json"
        auth_data = {
            "auth_mode": "password",
            "owner": {
                "username": "admin",
                "display_name": "Admin",
                "password_hash": hash_password("secret123"),
                "role": "owner",
            },
            "users": [],
            "token_version": 1,
            "sessions": {},
            "secret_key": "",
        }
        auth_path.write_text(
            json.dumps(auth_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Verify the loaded config defaults trust_localhost to True
        with patch("core.paths.get_data_dir", return_value=data_dir):
            loaded = load_auth()
        assert loaded.trust_localhost is True

        # Localhost bypass should work (trust defaults to True)
        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "localhost:8000"},
            )
        assert resp.status_code == 200

    def test_explicit_false_in_auth_json_disables_trust(self, data_dir):
        """Explicitly setting trust_localhost=false disables the bypass."""
        auth_path = data_dir / "auth.json"
        auth_data = {
            "auth_mode": "password",
            "trust_localhost": False,
            "owner": {
                "username": "admin",
                "display_name": "Admin",
                "password_hash": hash_password("secret123"),
                "role": "owner",
            },
            "users": [],
            "token_version": 1,
            "sessions": {},
            "secret_key": "",
        }
        auth_path.write_text(
            json.dumps(auth_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        app = _create_test_app(data_dir)
        client = TestClient(app)

        with patch("core.paths.get_data_dir", return_value=data_dir):
            loaded = load_auth()
        assert loaded.trust_localhost is False

        # Even with localhost IP, trust is disabled -> 401
        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "localhost:8000"},
            )
        assert resp.status_code == 401


class TestLocalTrustModeUnaffected:
    """local_trust auth_mode skips ALL auth, regardless of trust_localhost."""

    def test_local_trust_mode_ignores_trust_localhost_flag(self, data_dir):
        """In local_trust mode, trust_localhost has no effect (auth is off)."""
        config = AuthConfig(
            auth_mode="local_trust",
            trust_localhost=False,  # would block in password mode
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)

        # local_trust mode skips auth entirely, even though trust_localhost=False
        resp = client.get("/api/animas")
        # Should not be 401
        assert resp.status_code != 401
