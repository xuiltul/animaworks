from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from unittest.mock import patch

from fastapi import Request
from fastapi.testclient import TestClient

from core.auth.manager import hash_password, save_auth
from core.auth.models import AuthConfig, AuthUser
from core.config import invalidate_cache
from core.config.vault import VaultManager
from core.config.vault_sources import OnePasswordSettings, VaultSources, add_source, save_sources
from core.hub.members import HubMember, load_members, member_path, save_member, token_digest
from tests.unit.vault_fake_op import FakeOp


def _app(
    data_dir: Path,
    *,
    auth_mode: str = "password",
    trust_localhost: bool = False,
    role: str = "owner",
    setup_complete: bool = True,
):
    from server.app import create_app

    config_path = data_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "setup_complete": setup_complete,
                "hub": {"enabled": True},
                "credentials": {"svc": {"api_key": "config-plaintext-secret"}},
            }
        ),
        encoding="utf-8",
    )
    invalidate_cache()
    owner = AuthUser(username="admin", password_hash=hash_password("correct-password"), role="owner")
    users = []
    if role != "owner":
        owner = AuthUser(username="root", password_hash=hash_password("root-password"), role="owner")
        users = [AuthUser(username="admin", password_hash=hash_password("correct-password"), role=role)]
    save_auth(
        AuthConfig(
            auth_mode=auth_mode,
            trust_localhost=trust_localhost,
            owner=owner,
            users=users,
        )
    )
    animas_dir = data_dir / "animas"
    animas_dir.mkdir(exist_ok=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir(exist_ok=True)
    with (
        patch("server.app.ProcessSupervisor"),
        patch("server.app.WebSocketManager"),
    ):
        app = create_app(animas_dir, shared_dir)

    def auth_via_probe(request: Request):
        return {"auth_via": request.state.auth_via}

    app.add_api_route("/api/vault/auth-via-probe", auth_via_probe, methods=["GET"])
    app.add_api_route("/api/internal/auth-via-probe", auth_via_probe, methods=["GET"])
    for path in ("/api/vault/auth-via-probe", "/api/internal/auth-via-probe"):
        route = next(route for route in reversed(app.router.routes) if getattr(route, "path", None) == path)
        app.router.routes.remove(route)
        app.router.routes.insert(0, route)
    return app


def _setup_vault(data_dir: Path, fake_op: FakeOp) -> VaultManager:
    vault = VaultManager(data_dir)
    vault.generate_key()
    vault.store("shared", "MANAGED", "old-managed-secret")
    save_sources(data_dir, VaultSources(onepassword=OnePasswordSettings(command=fake_op.command, timeout_s=5.0)))
    add_source(data_dir, "MANAGED", "op://V/Managed/credential")
    fake_op.set_value("op://V/Managed/credential", "remote-managed-secret")
    return vault


def _login(client: TestClient, username: str = "admin", password: str = "correct-password") -> None:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200


def test_vault_writer_requires_owner_session_and_reentered_password(data_dir: Path) -> None:
    _setup_vault(data_dir, FakeOp(data_dir))
    app = _app(data_dir, trust_localhost=True)
    client = TestClient(app)

    unauthorized = client.get("/api/vault/overview")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["Cache-Control"] == "no-store"

    # A trusted localhost request carries an owner identity but is not a session.
    with patch("server.app._is_safe_localhost_request", return_value=True):
        response = client.put(
            "/api/vault/shared/LOCAL",
            json={"value": "local-secret", "password": "correct-password"},
        )
    assert response.status_code == 403
    assert response.headers["Cache-Control"] == "no-store"

    _login(client)
    wrong = client.put("/api/vault/shared/LOCAL", json={"value": "local-secret", "password": "wrong"})
    assert wrong.status_code == 403
    correct = client.put(
        "/api/vault/shared/LOCAL",
        json={"value": "local-secret", "password": "correct-password"},
    )
    assert correct.status_code == 200


def test_vault_writer_rejects_non_owner_and_local_trust_allows_write(data_dir: Path) -> None:
    _setup_vault(data_dir, FakeOp(data_dir))
    app = _app(data_dir, role="user")
    client = TestClient(app)
    _login(client)
    denied = client.put("/api/vault/shared/LOCAL", json={"value": "secret", "password": "correct-password"})
    assert denied.status_code == 403

    local_app = _app(data_dir, auth_mode="local_trust")
    local_client = TestClient(local_app)
    allowed = local_client.put("/api/vault/shared/LOCAL_TRUST", json={"value": "local-trust-secret"})
    assert allowed.status_code == 200
    capability = local_client.get("/api/vault/overview").json()["write"]
    assert capability == {"allowed": True, "requires_password": False, "reason": None}


def test_vault_writer_is_forbidden_during_setup(data_dir: Path) -> None:
    _setup_vault(data_dir, FakeOp(data_dir))
    client = TestClient(_app(data_dir, auth_mode="local_trust", setup_complete=False))

    response = client.put("/api/vault/shared/LOCAL", json={"value": "setup-secret"})

    assert response.status_code == 403
    assert response.headers["Cache-Control"] == "no-store"


def test_vault_onepassword_read_requires_session_owner(data_dir: Path, monkeypatch) -> None:
    fake_op = FakeOp(data_dir)
    _setup_vault(data_dir, fake_op)
    monkeypatch.setenv("FAKE_OP_STATE", fake_op.env["FAKE_OP_STATE"])
    monkeypatch.setenv("FAKE_OP_ARGV", fake_op.env["FAKE_OP_ARGV"])
    app = _app(data_dir, trust_localhost=True)
    client = TestClient(app)

    with patch("server.app._is_safe_localhost_request", return_value=True):
        denied = client.post("/api/vault/check", json={})
    assert denied.status_code == 403

    _login(client)
    allowed = client.post("/api/vault/check", json={})
    assert allowed.status_code == 200
    assert allowed.json()["results"]["MANAGED"] == "differs"


def test_vault_api_is_value_free_and_audited(data_dir: Path, monkeypatch) -> None:
    fake_op = FakeOp(data_dir)
    vault = _setup_vault(data_dir, fake_op)
    monkeypatch.setenv("FAKE_OP_STATE", fake_op.env["FAKE_OP_STATE"])
    monkeypatch.setenv("FAKE_OP_ARGV", fake_op.env["FAKE_OP_ARGV"])
    save_member(
        data_dir / "shared",
        HubMember(
            id="external-bot",
            display_name="External Bot",
            company="Example",
            token_sha256="0" * 64,
            message_route={"type": "webhook", "url_ref": "BOT_URL", "key_ref": "BOT_KEY"},
        ),
    )
    app = _app(data_dir)
    client = TestClient(app)
    _login(client)

    responses = []
    responses.append(
        client.put("/api/vault/shared/LOCAL", json={"value": "first-local-secret", "password": "correct-password"})
    )
    responses.append(
        client.put("/api/vault/shared/LOCAL", json={"value": "second-local-secret", "password": "correct-password"})
    )
    overview = client.get("/api/vault/overview")
    assert overview.json()["write"] == {"allowed": True, "requires_password": True, "reason": None}
    responses.append(overview)
    responses.append(client.post("/api/vault/check", json={}))
    responses.append(client.post("/api/vault/sync/plan", json={}))
    responses.append(
        client.post(
            "/api/vault/sync/apply",
            json={"keys": ["MANAGED"], "password": "correct-password", "items": [{"status": "unchanged"}]},
        )
    )
    responses.append(
        client.put(
            "/api/vault/shared/MANAGED",
            json={"value": "blocked-secret", "password": "correct-password"},
        )
    )
    responses.append(client.post("/api/vault/shared/MANAGED/delete", json={"password": "correct-password"}))
    responses.append(client.post("/api/vault/shared/LOCAL/rollback", json={"password": "correct-password"}))
    responses.append(client.post("/api/vault/shared/LOCAL/delete", json={"password": "correct-password"}))

    token_response = client.post(
        "/api/vault/hub-members/external-bot/reissue-token",
        json={"password": "correct-password"},
    )
    assert token_response.status_code == 200
    token = token_response.json()["token"]
    saved_member = load_members(data_dir / "shared")["external-bot"]
    assert saved_member.token_sha256 == token_digest(token)
    assert token not in member_path(data_dir / "shared", "external-bot").read_text(encoding="utf-8")

    migration = client.post("/api/vault/migrate-config", json={"password": "correct-password"})
    assert migration.status_code == 200
    config_raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert isinstance(config_raw["credentials"]["svc"]["api_key"], dict)
    assert "$vault" in config_raw["credentials"]["svc"]["api_key"]

    secrets_to_check = (
        "first-local-secret",
        "second-local-secret",
        "old-managed-secret",
        "remote-managed-secret",
        "blocked-secret",
        "config-plaintext-secret",
    )
    for response in responses + [migration]:
        assert response.headers["Cache-Control"] == "no-store"
        body = response.content.decode("utf-8")
        for secret in secrets_to_check:
            assert secret not in body
            assert hashlib.sha256(secret.encode()).hexdigest() not in body

    audit_path = data_dir / "vault_audit.jsonl"
    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert audit_lines
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o600
    for line in audit_lines:
        for secret in secrets_to_check:
            assert secret not in line
            assert hashlib.sha256(secret.encode()).hexdigest() not in line
    assert vault.get("shared", "MANAGED") == "remote-managed-secret"


def test_auth_via_marks_local_trust_localhost_session_and_internal(data_dir: Path) -> None:
    local_app = _app(data_dir, auth_mode="local_trust")
    local = TestClient(local_app).get("/api/vault/auth-via-probe")
    assert local.json() == {"auth_via": "local_trust"}

    localhost_app = _app(data_dir, trust_localhost=True)
    with patch("server.app._is_safe_localhost_request", return_value=True):
        localhost = TestClient(localhost_app).get("/api/vault/auth-via-probe")
    assert localhost.json() == {"auth_via": "localhost"}

    session_app = _app(data_dir)
    session_client = TestClient(session_app)
    _login(session_client)
    session = session_client.get("/api/vault/auth-via-probe")
    assert session.json() == {"auth_via": "session"}

    internal_app = _app(data_dir)
    token = internal_app.state.internal_auth.operator_token()
    internal = TestClient(internal_app).get(
        "/api/internal/auth-via-probe",
        headers={"X-AnimaWorks-Internal-Auth": token},
    )
    assert internal.json() == {"auth_via": "internal"}


def test_sync_apply_uses_fresh_server_plan(data_dir: Path, monkeypatch) -> None:
    fake_op = FakeOp(data_dir)
    vault = _setup_vault(data_dir, fake_op)
    fake_op.set_value("op://V/Managed/credential", "fresh-remote-value")
    monkeypatch.setenv("FAKE_OP_STATE", fake_op.env["FAKE_OP_STATE"])
    monkeypatch.setenv("FAKE_OP_ARGV", fake_op.env["FAKE_OP_ARGV"])
    client = TestClient(_app(data_dir))
    _login(client)

    response = client.post(
        "/api/vault/sync/apply",
        json={
            "keys": ["MANAGED"],
            "password": "correct-password",
            "items": [{"key": "MANAGED", "status": "unchanged"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["applied"] == ["MANAGED"]
    assert vault.get("shared", "MANAGED") == "fresh-remote-value"


def test_localhost_trust_with_signed_in_session_can_write(data_dir: Path) -> None:
    """Behind a local reverse proxy every request is "localhost"; a real sign-in still counts."""
    _setup_vault(data_dir, FakeOp(data_dir))
    app = _app(data_dir, trust_localhost=True)
    client = TestClient(app)
    _login(client)

    with patch("server.app._is_safe_localhost_request", return_value=True):
        overview = client.get("/api/vault/overview")
        wrong = client.put("/api/vault/shared/LOCAL", json={"value": "local-secret", "password": "wrong"})
        correct = client.put(
            "/api/vault/shared/LOCAL",
            json={"value": "local-secret", "password": "correct-password"},
        )
        client.cookies.clear()
        no_cookie = client.put(
            "/api/vault/shared/LOCAL",
            json={"value": "local-secret", "password": "correct-password"},
        )

    assert overview.status_code == 200
    assert overview.json()["write"]["allowed"] is True
    assert wrong.status_code == 403
    assert correct.status_code == 200
    assert no_cookie.status_code == 403
