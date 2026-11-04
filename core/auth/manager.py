from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Authentication configuration manager.

How auth state stays fresh without blocking the event loop:

- ``load_auth`` checks ``auth.json`` with one ``stat`` per call and re-parses it
  only when the file changed. It is blocking, so the server calls it off the
  event loop through :func:`run_auth_call`.
- ``save_auth`` publishes the new state in memory before it starts writing, and
  ``load_auth`` returns that state while the write is in flight. A revocation
  therefore takes effect at once, even if the disk stalls the write for many
  seconds. If the write fails, the published state is withdrawn.
- Changes made by other processes are seen by the next ``stat``. Nothing is
  cached by time, so no auth decision uses state older than that request's own
  check. An unreadable or corrupt file raises instead of falling back to the
  last good parse.
- Read-modify-write cycles hold :func:`auth_write_lock`, so that concurrent
  requests on worker threads cannot lose each other's updates and bring a
  revoked session back.
"""

import asyncio
import contextvars
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from pathlib import Path
from typing import Any, TypeVar

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from core.auth.models import AuthConfig, AuthUser, Session
from core.paths import get_data_dir
from core.platform.atomic_io import atomic_write_text

logger = logging.getLogger("animaworks.auth")

_AUTH_FILENAME = "auth.json"

_hasher = PasswordHash((Argon2Hasher(),))

_T = TypeVar("_T")

# Serializes read-modify-write cycles. May be held across a stalled disk write,
# so never acquire it on the event loop.
_write_lock = threading.RLock()

# Guards the two cache fields below. Never held across disk I/O.
_state_lock = threading.Lock()
_parsed: tuple[Path, tuple[int, int, int, int], AuthConfig] | None = None
_pending_writes: dict[Path, AuthConfig] = {}

# A file modified this recently may change again without its stat signature
# changing on filesystems with coarse timestamps, so it is not cached yet.
_RACY_MTIME_WINDOW_NS = 2_000_000_000

# Auth checks use their own threads so that disk work queued on the shared
# executors cannot delay them. Only the rare session-expiry path can wait on a
# disk write here, so a few threads are enough.
_AUTH_CHECK_THREADS = 4
AUTH_CHECK_TIMEOUT_S = 10.0
_auth_executor = ThreadPoolExecutor(max_workers=_AUTH_CHECK_THREADS, thread_name_prefix="auth-check")


# ── Path helpers ──────────────────────────────────────────────


def get_auth_path() -> Path:
    """Return the path to the auth.json configuration file."""
    return get_data_dir() / _AUTH_FILENAME


# ── Load / Save ──────────────────────────────────────────────


def load_auth() -> AuthConfig:
    """Return the current auth configuration as a private, mutable copy.

    Returns defaults if the file does not exist. Blocking: on the event loop,
    call this through :func:`run_auth_call`.
    """
    global _parsed

    path = get_auth_path()
    with _state_lock:
        pending = _pending_writes.get(path)
    if pending is not None:
        return pending.model_copy(deep=True)

    try:
        # Stat before reading: if the file is replaced in between, the newer
        # content is stored under the older signature and the next call simply
        # reloads. The other order could keep old content forever.
        st = os.stat(path)
    except FileNotFoundError:
        return AuthConfig()
    signature = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    with _state_lock:
        cached = _parsed
    if cached is not None and cached[0] == path and cached[1] == signature:
        return cached[2].model_copy(deep=True)

    config = AuthConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if time.time_ns() - st.st_mtime_ns > _RACY_MTIME_WINDOW_NS:
        with _state_lock:
            _parsed = (path, signature, config.model_copy(deep=True))
    return config


def save_auth(config: AuthConfig) -> None:
    """Atomically save auth configuration to disk with restrictive permissions.

    The new state is visible to :func:`load_auth` before the write starts, and
    is withdrawn if the write fails. Blocking: never call this on the event loop.
    """
    path = get_auth_path()
    payload = config.model_dump_json(indent=2)
    published = AuthConfig.model_validate_json(payload)
    with _write_lock:
        with _state_lock:
            _pending_writes[path] = published
        try:
            atomic_write_text(path, payload)
        finally:
            with _state_lock:
                del _pending_writes[path]
    logger.info("Saved auth config to %s", path)


def auth_write_lock() -> threading.RLock:
    """Return the lock to hold across a ``load_auth`` ... ``save_auth`` cycle.

    Without it, a concurrent update between the load and the save is lost, which
    could restore a session that was just revoked. Never acquire it on the event
    loop, because it can be held for as long as a disk write stalls.
    """
    return _write_lock


async def run_auth_call(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a blocking auth function off the event loop.

    Raises ``TimeoutError`` after ``AUTH_CHECK_TIMEOUT_S``. Callers must treat
    that as "could not authenticate", never as success.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await asyncio.wait_for(
        loop.run_in_executor(_auth_executor, context.run, fn, *args),
        AUTH_CHECK_TIMEOUT_S,
    )


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
    user_sessions = [(tok, sess) for tok, sess in config.sessions.items() if sess.username == username]
    if len(user_sessions) >= MAX_SESSIONS_PER_USER:
        user_sessions.sort(key=lambda x: x[1].created_at)
        excess = len(user_sessions) - MAX_SESSIONS_PER_USER + 1
        for tok, _ in user_sessions[:excess]:
            del config.sessions[tok]
        logger.info(
            "Evicted %d oldest session(s) for user '%s'",
            excess,
            username,
        )

    token = secrets.token_urlsafe(48)
    config.sessions[token] = Session(username=username)
    return token


def validate_session(token: str | None) -> Session | None:
    """Return the ``Session`` for *token*, or ``None`` if invalid/expired."""
    if not token:
        return None
    config = load_auth()
    session = config.sessions.get(token)
    if session is None:
        return None

    from datetime import datetime, timedelta

    from core.config.models import load_config

    ttl_days = load_config().server.session_ttl_days
    if ttl_days is not None:
        created = session.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        expiry = created + timedelta(days=ttl_days)
        if datetime.now(UTC) > expiry:
            revoke_session(token)
            logger.info("Session expired (TTL=%d days)", ttl_days)
            return None
    return session


def get_session_cookie_max_age() -> int:
    """Return the ``max-age`` for session cookies in seconds.

    Derived from ``server.session_ttl_days`` in config.  When the TTL is
    unlimited (``None``), returns 400 days which is the practical upper
    bound most browsers enforce.
    """
    from core.config.models import load_config

    ttl_days = load_config().server.session_ttl_days
    if ttl_days is None:
        return 400 * 86400
    return ttl_days * 86400


def revoke_session(token: str) -> None:
    """Remove a single session by token."""
    with auth_write_lock():
        config = load_auth()
        if token in config.sessions:
            del config.sessions[token]
            save_auth(config)
            logger.info("Revoked session")


def revoke_all_sessions(username: str | None = None) -> None:
    """Revoke all sessions, or only those belonging to *username*."""
    with auth_write_lock():
        config = load_auth()
        if username:
            config.sessions = {t: s for t, s in config.sessions.items() if s.username != username}
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
