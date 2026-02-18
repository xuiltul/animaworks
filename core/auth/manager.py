from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Authentication configuration manager."""

import json
import logging
import os
import secrets
from pathlib import Path

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from core.auth.models import AuthConfig, AuthUser, Session
from core.paths import get_data_dir

logger = logging.getLogger("animaworks.auth")

_AUTH_FILENAME = "auth.json"

_hasher = PasswordHash((Argon2Hasher(),))


# ── Path helpers ──────────────────────────────────────────────

def get_auth_path() -> Path:
    """Return the path to the auth.json configuration file."""
    return get_data_dir() / _AUTH_FILENAME


# ── Load / Save ──────────────────────────────────────────────

def load_auth() -> AuthConfig:
    """Load auth configuration from disk, returning defaults if not found."""
    path = get_auth_path()
    if not path.exists():
        return AuthConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return AuthConfig.model_validate(raw)


def save_auth(config: AuthConfig) -> None:
    """Atomically save auth configuration to disk with restrictive permissions."""
    path = get_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    logger.info("Saved auth config to %s", path)


# ── Password hashing ────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a plaintext password with Argon2id."""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against an Argon2id hash."""
    return _hasher.verify(password, password_hash)


# ── Session management ──────────────────────────────────────

MAX_SESSIONS_PER_USER = 10

def _ensure_secret_key(config: AuthConfig) -> None:
    """Generate secret_key if not yet set (reserved for future HMAC signing)."""
    if not config.secret_key:
        config.secret_key = secrets.token_urlsafe(32)


def create_session(config: AuthConfig, username: str) -> str:
    """Create a new session for *username* and return the token.

    If the user already has ``MAX_SESSIONS_PER_USER`` active sessions, the
    oldest one is evicted before creating the new session.

    The caller is responsible for calling ``save_auth(config)`` afterwards.
    """
    _ensure_secret_key(config)

    # Evict oldest sessions if user exceeds limit
    user_sessions = [
        (tok, sess) for tok, sess in config.sessions.items()
        if sess.username == username
    ]
    if len(user_sessions) >= MAX_SESSIONS_PER_USER:
        user_sessions.sort(key=lambda x: x[1].created_at)
        excess = len(user_sessions) - MAX_SESSIONS_PER_USER + 1
        for tok, _ in user_sessions[:excess]:
            del config.sessions[tok]
        logger.info(
            "Evicted %d oldest session(s) for user '%s'", excess, username,
        )

    token = secrets.token_urlsafe(48)
    config.sessions[token] = Session(username=username)
    return token


def validate_session(token: str | None) -> Session | None:
    """Return the ``Session`` for *token*, or ``None`` if invalid."""
    if not token:
        return None
    config = load_auth()
    return config.sessions.get(token)


def revoke_session(token: str) -> None:
    """Remove a single session by token."""
    config = load_auth()
    if token in config.sessions:
        del config.sessions[token]
        save_auth(config)
        logger.info("Revoked session")


def revoke_all_sessions(username: str | None = None) -> None:
    """Revoke all sessions, or only those belonging to *username*."""
    config = load_auth()
    if username:
        config.sessions = {
            t: s for t, s in config.sessions.items() if s.username != username
        }
    else:
        config.sessions = {}
    save_auth(config)
    logger.info("Revoked sessions for %s", username or "all users")


# ── User lookup ─────────────────────────────────────────────

def find_user(config: AuthConfig, username: str) -> AuthUser | None:
    """Find a user by username (checks owner first, then users list)."""
    if config.owner and config.owner.username == username:
        return config.owner
    for u in config.users:
        if u.username == username:
            return u
    return None


def get_all_users(config: AuthConfig) -> list[AuthUser]:
    """Return all users (owner + users list)."""
    result: list[AuthUser] = []
    if config.owner:
        result.append(config.owner)
    result.extend(config.users)
    return result
