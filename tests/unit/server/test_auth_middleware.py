"""Unit tests for auth_guard middleware in server/app.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.auth.manager import hash_password, create_session, save_auth
from core.auth.models import AuthConfig, AuthUser


def _create_test_app(data_dir: Path):
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


class TestAuthGuardMiddleware:
    def test_local_trust_skips_auth(self, data_dir):
        config = AuthConfig(auth_mode="local_trust")
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Should access API without auth
        resp = client.get("/api/system/health")
        # Should not get 401
        assert resp.status_code != 401

    def test_password_mode_blocks_without_cookie(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/api/animas")
        assert resp.status_code == 401

    def test_password_mode_allows_login_without_cookie(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "pw"})
        assert resp.status_code == 200

    def test_password_mode_allows_with_valid_session(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)

        # Login to get session
        login_resp = client.post("/api/auth/login", json={"username": "admin", "password": "pw"})
        assert login_resp.status_code == 200

        # Now API access should work (cookie is auto-sent by TestClient)
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200

    def test_static_files_skip_auth(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Static file paths should not be blocked by auth
        resp = client.get("/")
        assert resp.status_code != 401

    def test_health_endpoint_skips_auth(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code != 401

    # ── Localhost trust tests ─────────────────────────────

    def test_password_mode_allows_localhost_with_trust(self, data_dir):
        """password mode + trust_localhost=True + localhost request -> not 401."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=True,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        with patch("server.app._is_safe_localhost_request", return_value=True):
            resp = client.get("/api/animas")
        assert resp.status_code != 401

    def test_password_mode_blocks_localhost_without_trust(self, data_dir):
        """password mode + trust_localhost=False + localhost request -> 401."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=False,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Even though _is_safe_localhost_request would return True,
        # trust_localhost=False means the check is never reached.
        with patch("server.app._is_safe_localhost_request", return_value=True):
            resp = client.get("/api/animas")
        assert resp.status_code == 401

    def test_password_mode_blocks_remote_with_trust(self, data_dir):
        """password mode + trust_localhost=True + remote IP -> 401."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=True,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Remote IP: _is_safe_localhost_request returns False
        with patch("server.app._is_safe_localhost_request", return_value=False):
            resp = client.get("/api/animas")
        assert resp.status_code == 401

    def test_password_mode_blocks_localhost_csrf_origin(self, data_dir):
        """localhost + Origin=http://evil.com -> 401 (CSRF protection)."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=True,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        # Let the real _is_safe_localhost_request run, but make _is_localhost
        # return True to simulate a localhost client. The evil Origin header
        # should cause _is_safe_localhost_request to return False.
        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={
                    "origin": "http://evil.com",
                    "host": "localhost:8000",
                },
            )
        assert resp.status_code == 401

    def test_password_mode_blocks_localhost_dns_rebinding(self, data_dir):
        """localhost + Host=evil.com -> 401 (DNS rebinding protection)."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=True,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        with patch("server.localhost._is_localhost", return_value=True):
            resp = client.get(
                "/api/animas",
                headers={"host": "evil.com:8000"},
            )
        assert resp.status_code == 401

    def test_local_trust_mode_ignores_trust_localhost_false(self, data_dir):
        """local_trust mode still works even when trust_localhost=False."""
        config = AuthConfig(
            auth_mode="local_trust",
            trust_localhost=False,
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        resp = client.get("/api/system/health")
        assert resp.status_code != 401

    def test_password_mode_allows_localhost_no_origin(self, data_dir):
        """localhost + no Origin + Host=localhost -> not 401."""
        config = AuthConfig(
            auth_mode="password",
            trust_localhost=True,
            owner=AuthUser(username="admin", password_hash=hash_password("pw"), role="owner"),
        )
        save_auth(config)

        app = _create_test_app(data_dir)
        client = TestClient(app)
        with patch("server.app._is_safe_localhost_request", return_value=True):
            resp = client.get("/api/animas")
        assert resp.status_code != 401

    def test_trust_localhost_default_true(self):
        """AuthConfig() has trust_localhost=True by default."""
        config = AuthConfig()
        assert config.trust_localhost is True
