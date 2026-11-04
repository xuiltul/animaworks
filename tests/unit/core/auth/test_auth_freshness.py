"""Auth state must be fresh without doing disk I/O on the event loop.

These cover the security properties of the ``auth.json`` cache: a revocation
takes effect before the disk write finishes, changes by other processes are seen
by the next check, and a failure never falls back to older state.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import core.auth.manager as auth_manager
from core.auth.manager import (
    auth_write_lock,
    create_session,
    hash_password,
    load_auth,
    revoke_all_sessions,
    revoke_session,
    run_auth_call,
    save_auth,
    validate_session,
)
from core.auth.models import AuthConfig, AuthUser
from core.infra import startup_progress


def _owner(password: str = "old-password") -> AuthUser:
    return AuthUser(username="owner", role="owner", password_hash=hash_password(password))


def _password_config(password: str = "old-password") -> AuthConfig:
    return AuthConfig(auth_mode="password", trust_localhost=False, owner=_owner(password))


@contextmanager
def _stalled_disk_write() -> Iterator[threading.Event]:
    """Hold every ``auth.json`` write until the yielded event is set."""
    release = threading.Event()
    real_write = auth_manager.atomic_write_text

    def stalled(path: Path, text: str, **kwargs: object) -> None:
        release.wait(timeout=30.0)
        real_write(path, text, **kwargs)

    with patch.object(auth_manager, "atomic_write_text", stalled):
        try:
            yield release
        finally:
            release.set()


def _on_disk(data_dir: Path) -> dict:
    return json.loads((data_dir / "auth.json").read_text(encoding="utf-8"))


def _age_file(path: Path) -> None:
    """Move a file's mtime past the window in which it is not cached."""
    old = time.time() - 60
    os.utime(path, (old, old))


# ── Writes by this process take effect before the disk finishes ──


def test_revocation_applies_while_the_write_is_stalled(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    assert validate_session(token) is not None

    with _stalled_disk_write() as release:
        writer = threading.Thread(target=revoke_session, args=(token,))
        writer.start()
        deadline = time.monotonic() + 5.0
        while validate_session(token) is not None:
            assert time.monotonic() < deadline, "revocation did not apply during the stalled write"
            time.sleep(0.005)
        assert token in _on_disk(data_dir)["sessions"]  # the disk has not caught up yet
        release.set()
        writer.join(timeout=5.0)

    assert token not in _on_disk(data_dir)["sessions"]
    assert validate_session(token) is None


def test_failed_write_withdraws_the_published_state(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)

    with patch.object(auth_manager, "atomic_write_text", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            revoke_session(token)

    # The caller saw the error, and memory matches what the disk still says.
    assert validate_session(token) is not None
    assert token in _on_disk(data_dir)["sessions"]


def test_concurrent_updates_do_not_bring_back_a_revoked_session(data_dir: Path) -> None:
    config = _password_config()
    victim = create_session(config, "owner")
    save_auth(config)
    new_token: list[str] = []

    def login() -> None:
        with auth_write_lock():
            fresh = load_auth()
            new_token.append(create_session(fresh, "owner"))
            save_auth(fresh)

    with _stalled_disk_write() as release:
        revoker = threading.Thread(target=revoke_session, args=(victim,))
        revoker.start()
        while validate_session(victim) is not None:
            time.sleep(0.005)
        # The revocation's write is in flight. This login must build on top of it.
        other = threading.Thread(target=login)
        other.start()
        time.sleep(0.1)
        release.set()
        revoker.join(timeout=5.0)
        other.join(timeout=5.0)

    sessions = _on_disk(data_dir)["sessions"]
    assert victim not in sessions
    assert new_token[0] in sessions


def test_revoke_all_sessions_for_one_user_keeps_the_others(data_dir: Path) -> None:
    config = _password_config()
    config.users.append(AuthUser(username="bob"))
    owner_token = create_session(config, "owner")
    bob_token = create_session(config, "bob")
    save_auth(config)

    revoke_all_sessions("owner")

    assert validate_session(owner_token) is None
    assert validate_session(bob_token) is not None


# ── Changes by other processes are seen by the next check ──


def test_external_atomic_replace_is_seen_by_the_next_load(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    path = data_dir / "auth.json"
    _age_file(path)
    assert load_auth().sessions  # now cached

    replacement = path.with_suffix(".new")
    config.sessions = {}
    replacement.write_text(config.model_dump_json(), encoding="utf-8")
    os.replace(replacement, path)

    assert load_auth().sessions == {}
    assert validate_session(token) is None


def test_external_in_place_edit_of_the_same_size_is_seen(data_dir: Path) -> None:
    """A recently modified file must not be cached, whatever its timestamp granularity."""
    path = data_dir / "auth.json"
    path.write_text(json.dumps({"auth_mode": "password"}), encoding="utf-8")
    assert load_auth().auth_mode == "password"
    path.write_text(json.dumps({"auth_mode": "password", "token_version": 2}), encoding="utf-8")
    assert load_auth().token_version == 2
    same_size = json.dumps({"auth_mode": "password", "token_version": 3})
    path.write_text(same_size, encoding="utf-8")
    assert load_auth().token_version == 3


def test_token_version_and_auth_mode_changes_are_seen(data_dir: Path) -> None:
    save_auth(AuthConfig(auth_mode="password", token_version=1))
    path = data_dir / "auth.json"
    _age_file(path)
    assert load_auth().token_version == 1

    updated = json.loads(path.read_text(encoding="utf-8"))
    updated.update(auth_mode="multi_user", token_version=2)
    path.write_text(json.dumps(updated), encoding="utf-8")

    fresh = load_auth()
    assert fresh.token_version == 2
    assert fresh.auth_mode == "multi_user"


def test_corrupt_file_raises_instead_of_using_the_last_good_state(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    path = data_dir / "auth.json"
    _age_file(path)
    assert validate_session(token) is not None

    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_auth()
    with pytest.raises(json.JSONDecodeError):
        validate_session(token)


def test_deleted_file_is_seen_by_the_next_load(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    path = data_dir / "auth.json"
    _age_file(path)
    assert validate_session(token) is not None

    path.unlink()

    assert load_auth() == AuthConfig()
    assert validate_session(token) is None


def test_cache_follows_a_change_of_data_dir(data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_auth(AuthConfig(auth_mode="password"))
    _age_file(data_dir / "auth.json")
    assert load_auth().auth_mode == "password"

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(other))
    assert load_auth() == AuthConfig()


# ── The cache itself ──


def test_unchanged_file_is_parsed_once(data_dir: Path) -> None:
    save_auth(_password_config())
    _age_file(data_dir / "auth.json")
    real_read_text = Path.read_text
    reads: list[Path] = []

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "auth.json":
            reads.append(self)
        return real_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", counting_read_text):
        for _ in range(5):
            load_auth()

    assert len(reads) == 1


def test_returned_config_is_a_private_copy(data_dir: Path) -> None:
    save_auth(_password_config())
    _age_file(data_dir / "auth.json")
    load_auth()

    mine = load_auth()
    mine.owner.password_hash = "tampered"
    mine.sessions["forged"] = None  # type: ignore[assignment]

    fresh = load_auth()
    assert fresh.owner.password_hash != "tampered"
    assert "forged" not in fresh.sessions


# ── Through the app ──


@pytest.fixture
def _ready_server() -> Iterator[None]:
    startup_progress._reset_for_testing()
    startup_progress.set_phase("ready")
    yield
    startup_progress._reset_for_testing()


def _make_app(data_dir: Path):
    from core.config.models import AnimaWorksConfig, invalidate_cache, save_config
    from server.app import create_app

    save_config(AnimaWorksConfig(setup_complete=True), data_dir / "config.json")
    invalidate_cache()
    return create_app(data_dir / "animas", data_dir / "shared")


def _client(app, token: str | None = None) -> AsyncClient:
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    if token:
        client.cookies.set("session_token", token)
    return client


@pytest.mark.asyncio
@pytest.mark.usefixtures("_ready_server")
async def test_password_change_locks_out_old_sessions_during_a_stalled_write(data_dir: Path) -> None:
    config = _password_config()
    old_session = create_session(config, "owner")
    save_auth(config)
    app = _make_app(data_dir)

    async with _client(app, old_session) as client:
        assert (await client.get("/api/auth/me")).status_code == 200

        with _stalled_disk_write() as release:
            change = asyncio.create_task(
                client.put(
                    "/api/users/me/password",
                    json={"current_password": "old-password", "new_password": "new-password"},
                )
            )
            deadline = time.monotonic() + 10.0
            while True:
                # Answering at all proves the loop is alive during the stall.
                response = await client.get("/api/auth/me")
                if response.status_code == 401:
                    break
                assert time.monotonic() < deadline, "old session stayed valid during the stalled write"
                await asyncio.sleep(0.01)

            assert not change.done()
            release.set()
            result = await asyncio.wait_for(change, timeout=10.0)

        assert result.status_code == 200
        assert (await client.get("/api/auth/me")).status_code == 401

    saved = _on_disk(data_dir)
    assert saved["sessions"] == {}
    assert auth_manager.verify_password("new-password", saved["owner"]["password_hash"])


@pytest.mark.asyncio
@pytest.mark.usefixtures("_ready_server")
async def test_logout_applies_during_a_stalled_write(data_dir: Path) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    app = _make_app(data_dir)

    async with _client(app, token) as client:
        with _stalled_disk_write() as release:
            logout = asyncio.create_task(client.post("/api/auth/logout"))
            deadline = time.monotonic() + 10.0
            while (await client.get("/api/auth/me")).status_code != 401:
                assert time.monotonic() < deadline, "session stayed valid during the stalled write"
                await asyncio.sleep(0.01)
            release.set()
            assert (await asyncio.wait_for(logout, timeout=10.0)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.usefixtures("_ready_server")
async def test_login_fails_if_the_password_changed_during_verification(data_dir: Path) -> None:
    save_auth(_password_config("first"))
    app = _make_app(data_dir)
    real_verify = auth_manager.verify_password

    def verify_then_change(password: str, password_hash: str) -> bool:
        result = real_verify(password, password_hash)
        changed = load_auth()
        changed.owner.password_hash = hash_password("second")
        save_auth(changed)
        return result

    with patch("server.routes.auth.verify_password", verify_then_change):
        async with _client(app) as client:
            response = await client.post("/api/auth/login", json={"username": "owner", "password": "first"})

    assert response.status_code == 401
    assert load_auth().sessions == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_ready_server")
async def test_auth_check_timeout_fails_closed_and_health_still_answers(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _password_config()
    token = create_session(config, "owner")
    save_auth(config)
    app = _make_app(data_dir)
    monkeypatch.setattr(auth_manager, "AUTH_CHECK_TIMEOUT_S", 0.1)
    auth_path = str(data_dir / "auth.json")
    real_stat = os.stat
    release = threading.Event()

    def hung_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if str(path) == auth_path:
            release.wait(timeout=30.0)
        return real_stat(path, *args, **kwargs)

    with patch("os.stat", hung_stat):
        try:
            async with _client(app, token) as client:
                protected = await client.get("/api/auth/me")
                health = await client.get("/api/system/health")
        finally:
            release.set()

    assert protected.status_code == 503
    assert protected.headers["retry-after"] == "5"
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_run_auth_call_runs_off_the_loop_and_keeps_context() -> None:
    import contextvars

    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="unset")
    var.set("request-1")

    def probe() -> tuple[int, str]:
        return threading.get_ident(), var.get()

    thread_id, value = await run_auth_call(probe)
    assert thread_id != threading.get_ident()
    assert value == "request-1"
