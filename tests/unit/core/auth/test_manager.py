"""Unit tests for core/auth/manager.py — Authentication configuration manager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from core.auth.manager import (
    create_session,
    get_auth_path,
    find_user,
    get_all_users,
    hash_password,
    load_auth,
    revoke_all_sessions,
    revoke_session,
    save_auth,
    validate_session,
    verify_password,
)
from core.auth.models import AuthConfig, AuthUser, Session


# ── get_auth_path ────────────────────────────────────────


class TestGetAuthPath:
    def test_returns_auth_json_under_data_dir(self, data_dir: Path):
        result = get_auth_path()
        assert result == data_dir / "auth.json"

    def test_filename_is_auth_json(self, data_dir: Path):
        result = get_auth_path()
        assert result.name == "auth.json"


# ── load_auth ────────────────────────────────────────────


class TestLoadAuth:
    def test_returns_default_when_file_missing(self, data_dir: Path):
        config = load_auth()
        assert config.auth_mode == "local_trust"
        assert config.owner is None
        assert config.users == []
        assert config.token_version == 1

    def test_loads_from_file(self, data_dir: Path):
        auth_path = data_dir / "auth.json"
        auth_data = {
            "auth_mode": "password",
            "owner": {
                "username": "taro",
                "display_name": "Taro Yamada",
                "bio": "the owner",
            },
            "users": [],
            "token_version": 2,
        }
        auth_path.write_text(json.dumps(auth_data), encoding="utf-8")

        config = load_auth()
        assert config.auth_mode == "password"
        assert config.owner is not None
        assert config.owner.username == "taro"
        assert config.owner.display_name == "Taro Yamada"
        assert config.token_version == 2


# ── save_auth ────────────────────────────────────────────


class TestSaveAuth:
    def test_writes_auth_json(self, data_dir: Path):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="alice", display_name="Alice"),
        )
        save_auth(config)

        auth_path = data_dir / "auth.json"
        assert auth_path.exists()

        loaded = json.loads(auth_path.read_text(encoding="utf-8"))
        assert loaded["auth_mode"] == "password"
        assert loaded["owner"]["username"] == "alice"

    def test_sets_restrictive_permissions(self, data_dir: Path):
        config = AuthConfig()
        save_auth(config)

        auth_path = data_dir / "auth.json"
        mode = auth_path.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_creates_parent_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(nested_dir))

        # Invalidate caches so new env var is picked up
        from core.config import invalidate_cache
        from core.paths import _prompt_cache
        invalidate_cache()
        _prompt_cache.clear()

        config = AuthConfig()
        save_auth(config)

        auth_path = nested_dir / "auth.json"
        assert auth_path.exists()

        # Cleanup
        invalidate_cache()
        _prompt_cache.clear()

    def test_atomic_write(self, data_dir: Path):
        """Verify the file is written atomically (no .tmp leftover)."""
        config = AuthConfig(
            owner=AuthUser(username="bob"),
        )
        save_auth(config)

        auth_path = data_dir / "auth.json"
        tmp_path = auth_path.with_suffix(".tmp")
        assert auth_path.exists()
        assert not tmp_path.exists()

    def test_roundtrip(self, data_dir: Path):
        """Save and load should produce equivalent config."""
        original = AuthConfig(
            auth_mode="multi_user",
            owner=AuthUser(username="owner", display_name="Owner"),
            users=[
                AuthUser(username="user1"),
                AuthUser(username="user2", display_name="User Two"),
            ],
            token_version=3,
        )
        save_auth(original)
        loaded = load_auth()

        assert loaded.auth_mode == original.auth_mode
        assert loaded.owner is not None
        assert loaded.owner.username == original.owner.username
        assert len(loaded.users) == 2
        assert loaded.token_version == 3


# ── Password hashing ────────────────────────────────────


class TestPasswordHashing:
    def test_hash_produces_argon2id(self):
        h = hash_password("test123")
        assert h.startswith("$argon2id$")

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_different_passwords_different_hashes(self):
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2

    def test_same_password_different_hashes(self):
        """Argon2 uses random salt, so same password gives different hashes."""
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2  # different salts


# ── Session management ──────────────────────────────────


class TestSessionManagement:
    def test_create_session(self, data_dir):
        config = AuthConfig(
            auth_mode="password",
            owner=AuthUser(username="admin"),
        )
        save_auth(config)

        token = create_session(config, "admin")
        save_auth(config)

        assert token is not None
        assert len(token) > 20
        assert token in config.sessions
        assert config.sessions[token].username == "admin"

    def test_create_session_generates_secret_key(self, data_dir):
        config = AuthConfig()
        assert config.secret_key == ""
        create_session(config, "user1")
        assert config.secret_key != ""

    def test_validate_session_valid(self, data_dir):
        config = AuthConfig(owner=AuthUser(username="admin"))
        token = create_session(config, "admin")
        save_auth(config)

        session = validate_session(token)
        assert session is not None
        assert session.username == "admin"

    def test_validate_session_invalid(self, data_dir):
        config = AuthConfig()
        save_auth(config)
        assert validate_session("nonexistent_token") is None

    def test_validate_session_none(self, data_dir):
        assert validate_session(None) is None

    def test_validate_session_empty(self, data_dir):
        assert validate_session("") is None

    def test_revoke_session(self, data_dir):
        config = AuthConfig(owner=AuthUser(username="admin"))
        token = create_session(config, "admin")
        save_auth(config)

        revoke_session(token)
        assert validate_session(token) is None

    def test_revoke_session_nonexistent(self, data_dir):
        config = AuthConfig()
        save_auth(config)
        # Should not raise
        revoke_session("nonexistent_token")

    def test_revoke_all_sessions(self, data_dir):
        config = AuthConfig(owner=AuthUser(username="admin"))
        t1 = create_session(config, "admin")
        t2 = create_session(config, "admin")
        save_auth(config)

        revoke_all_sessions()
        assert validate_session(t1) is None
        assert validate_session(t2) is None

    def test_revoke_all_sessions_by_username(self, data_dir):
        config = AuthConfig(
            owner=AuthUser(username="admin"),
            users=[AuthUser(username="user1")],
        )
        admin_token = create_session(config, "admin")
        user_token = create_session(config, "user1")
        save_auth(config)

        revoke_all_sessions(username="user1")

        assert validate_session(admin_token) is not None
        assert validate_session(user_token) is None

    def test_multiple_sessions_per_user(self, data_dir):
        config = AuthConfig(owner=AuthUser(username="admin"))
        t1 = create_session(config, "admin")
        t2 = create_session(config, "admin")
        save_auth(config)

        assert validate_session(t1) is not None
        assert validate_session(t2) is not None
        assert t1 != t2

    def test_session_limit_evicts_oldest(self, data_dir):
        """When a user exceeds MAX_SESSIONS_PER_USER, oldest sessions are evicted."""
        from core.auth.manager import MAX_SESSIONS_PER_USER

        config = AuthConfig(owner=AuthUser(username="admin"))
        tokens = []
        for _ in range(MAX_SESSIONS_PER_USER):
            tokens.append(create_session(config, "admin"))
        save_auth(config)

        # All 10 should be valid
        assert len(config.sessions) == MAX_SESSIONS_PER_USER
        for t in tokens:
            assert validate_session(t) is not None

        # Create 11th — should evict the 1st
        new_token = create_session(config, "admin")
        save_auth(config)

        assert len(config.sessions) == MAX_SESSIONS_PER_USER
        assert validate_session(tokens[0]) is None  # oldest evicted
        assert validate_session(new_token) is not None

    def test_session_limit_does_not_evict_other_users(self, data_dir):
        """Session limit applies per-user, not globally."""
        from core.auth.manager import MAX_SESSIONS_PER_USER

        config = AuthConfig(
            owner=AuthUser(username="admin"),
            users=[AuthUser(username="alice")],
        )
        alice_token = create_session(config, "alice")
        for _ in range(MAX_SESSIONS_PER_USER):
            create_session(config, "admin")
        save_auth(config)

        # Alice's session should still be valid
        assert validate_session(alice_token) is not None

        # Admin creating one more should only evict admin's oldest
        create_session(config, "admin")
        save_auth(config)
        assert validate_session(alice_token) is not None


# ── User lookup ─────────────────────────────────────────


class TestUserLookup:
    def test_find_owner(self):
        config = AuthConfig(owner=AuthUser(username="admin"))
        user = find_user(config, "admin")
        assert user is not None
        assert user.username == "admin"

    def test_find_user_in_list(self):
        config = AuthConfig(
            owner=AuthUser(username="admin"),
            users=[AuthUser(username="alice"), AuthUser(username="bob")],
        )
        user = find_user(config, "bob")
        assert user is not None
        assert user.username == "bob"

    def test_find_user_not_found(self):
        config = AuthConfig(owner=AuthUser(username="admin"))
        assert find_user(config, "nobody") is None

    def test_find_user_no_owner(self):
        config = AuthConfig()
        assert find_user(config, "anyone") is None

    def test_get_all_users_owner_only(self):
        config = AuthConfig(owner=AuthUser(username="admin"))
        users = get_all_users(config)
        assert len(users) == 1
        assert users[0].username == "admin"

    def test_get_all_users_with_list(self):
        config = AuthConfig(
            owner=AuthUser(username="admin"),
            users=[AuthUser(username="alice"), AuthUser(username="bob")],
        )
        users = get_all_users(config)
        assert len(users) == 3
        assert [u.username for u in users] == ["admin", "alice", "bob"]

    def test_get_all_users_no_owner(self):
        config = AuthConfig(users=[AuthUser(username="alice")])
        users = get_all_users(config)
        assert len(users) == 1
        assert users[0].username == "alice"
