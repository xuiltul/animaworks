# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for enclave-mode startup guards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config.schemas import AnimaWorksConfig
from core.enclave import EnclaveViolationError, collect_enclave_violations, enforce_enclave_runtime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    return {
        "version": 1,
        "system": {"mode": "server"},
        "locale": "ja",
        "credentials": {"anthropic": {"api_key": "", "base_url": None}},
        "anima_defaults": {"model": "claude-sonnet-4-5", "credential": "anthropic"},
        "animas": {},
        "external_messaging": {
            "slack": {"enabled": False},
            "chatwork": {"enabled": False},
            "discord": {"enabled": False},
            "zoom": {"enabled": False},
        },
        "github_webhook": {"enabled": False},
        "phone": {"enabled": False},
        "human_notification": {"enabled": False},
        "external_tasks": {"enabled": False},
        "event_export": {"url": None},
    }


def _make_config(**overrides: object) -> AnimaWorksConfig:
    data = _base_config()
    data.update(overrides)
    return AnimaWorksConfig.model_validate(data)


def _enclave(**overrides: object) -> dict:
    data: dict = {
        "enabled": True,
        "name": "primary",
        "socket_path": "/run/animaworks/enclave.sock",
        "entry_anima": "main",
        "allowed_peer_uids": [1000, 2000],
        "allowed_llm_credentials": ["anthropic"],
        "egress": {"stages": [{"type": "regex_denylist", "patterns": [{"regex": "x", "action": "redact"}]}]},
    }
    data.update(overrides)
    return data


def _prepare_data_dir(data_dir: Path, *, anima: str = "main", mode: int = 0o700) -> Path:
    """Set up an enclave-friendly data directory under *data_dir*.

    Writes the anima directory, a full-access-free ``permissions.json``,
    a password-auth ``auth.json``, and hardens directory permissions.
    """
    anima_dir = data_dir / "animas" / anima
    anima_dir.mkdir(parents=True, exist_ok=True)

    perms_path = anima_dir / "permissions.json"
    perms_path.write_text(json.dumps({"version": 1, "file_roots": ["/workspaces/app"]}), encoding="utf-8")

    auth_path = data_dir / "auth.json"
    auth_path.write_text(
        json.dumps({"auth_mode": "password", "trust_localhost": False, "users": [], "sessions": {}}),
        encoding="utf-8",
    )

    for entry in _guard_dir_targets(data_dir):
        entry.chmod(mode)
    return anima_dir


def _guard_dir_targets(data_dir: Path) -> list[Path]:
    targets = [data_dir]
    animas_dir = data_dir / "animas"
    if animas_dir.is_dir():
        targets.append(animas_dir)
        for child in sorted(animas_dir.iterdir()):
            if child.is_dir():
                targets.append(child)
    return targets


def _violations(config: AnimaWorksConfig, data_dir: Path, *, host: str | None = None) -> list[str]:
    return collect_enclave_violations(config, data_dir, host=host)


def _has(violations: list[str], fragment: str) -> bool:
    return any(fragment in v for v in violations)


# ---------------------------------------------------------------------------
# enabled=False short-circuits
# ---------------------------------------------------------------------------


def test_disabled_does_no_checks(data_dir: Path) -> None:
    # Even a deliberately broken layout yields no violations when disabled.
    config = _make_config(
        enclave={
            "enabled": False,
            "name": "",
            "socket_path": "",
            "entry_anima": "missing",
            "allowed_llm_credentials": [],
        },
        event_export={"url": "https://example.com"},
        external_messaging={"slack": {"enabled": True}},
    )
    assert _violations(config, data_dir) == []


# ---------------------------------------------------------------------------
# Guard 1: required fields + entry anima presence
# ---------------------------------------------------------------------------


def test_guard1_required_fields(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(name="", socket_path="", entry_anima=""))
    violations = _violations(config, data_dir)
    assert _has(violations, "name")
    assert _has(violations, "socket_path")
    assert _has(violations, "entry_anima")


def test_guard1_entry_anima_missing(data_dir: Path) -> None:
    _prepare_data_dir(data_dir, anima="other")
    config = _make_config(enclave=_enclave(entry_anima="main"))
    violations = _violations(config, data_dir)
    assert _has(violations, "main")


def test_guard1_empty_peer_uids(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(allowed_peer_uids=[]))
    assert _has(_violations(config, data_dir), "allowed_peer_uids")


def test_guard1_invalid_egress(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(egress={}))
    assert _has(_violations(config, data_dir), "egress")


def test_guard1_missing_socket_group(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(socket_group="__no_such_group__"))
    assert _has(_violations(config, data_dir), "socket_group")


# ---------------------------------------------------------------------------
# Guard 2: event export URL
# ---------------------------------------------------------------------------


def test_guard2_event_export_url(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(
        enclave=_enclave(),
        event_export={"url": "https://events.example.com/ingest"},
    )
    assert _has(_violations(config, data_dir), "event_export")


# ---------------------------------------------------------------------------
# Guard 3: external integrations disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel_key",
    ["slack", "chatwork", "discord", "zoom"],
)
def test_guard3_external_messaging(data_dir: Path, channel_key: str) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(
        enclave=_enclave(),
        external_messaging={channel_key: {"enabled": True}},
    )
    assert _has(_violations(config, data_dir), channel_key)


@pytest.mark.parametrize(
    "field",
    ["github_webhook", "phone", "human_notification"],
)
def test_guard3_other_integrations(data_dir: Path, field: str) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(), **{field: {"enabled": True}})
    assert _has(_violations(config, data_dir), field)


# ---------------------------------------------------------------------------
# Guard 4: ownership and directory mode
# ---------------------------------------------------------------------------


def test_guard4_owner_violation(data_dir: Path) -> None:
    target = _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave())
    # Simulate a foreign-owned directory via a non-matching uid patch.
    original_uid = os.getuid()

    def _fake_getuid() -> int:
        return original_uid + 1

    import core.enclave.guards as guards

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(guards.os, "getuid", _fake_getuid)
    try:
        violations = _violations(config, data_dir)
    finally:
        monkeypatch.undo()
    assert _has(violations, str(target))


def test_guard4_mode_violation(data_dir: Path) -> None:
    target = _prepare_data_dir(data_dir, mode=0o744)
    config = _make_config(enclave=_enclave())
    violations = _violations(config, data_dir)
    assert _has(violations, str(target))


# ---------------------------------------------------------------------------
# Guard 5: LLM credential allow-list
# ---------------------------------------------------------------------------


def test_guard5_empty_allowlist(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(allowed_llm_credentials=[]))
    assert _has(_violations(config, data_dir), "allowed_llm_credentials")


def test_guard3_external_tasks_enabled(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(), external_tasks={"enabled": True})
    assert _has(_violations(config, data_dir), "external_tasks")


def test_guard5_unallowed_credential(data_dir: Path) -> None:
    anima_dir = _prepare_data_dir(data_dir)
    (anima_dir / "status.json").write_text(
        json.dumps({"model": "x", "credential": "bedrock-account-a"}),
        encoding="utf-8",
    )
    config = _make_config(enclave=_enclave())
    assert _has(_violations(config, data_dir), "bedrock-account-a")


def test_guard5_allowed_credential_passes(data_dir: Path) -> None:
    anima_dir = _prepare_data_dir(data_dir)
    (anima_dir / "status.json").write_text(
        json.dumps({"model": "x", "credential": "bedrock-account-a"}),
        encoding="utf-8",
    )
    config = _make_config(enclave=_enclave(allowed_llm_credentials=["anthropic", "bedrock-account-a"]))
    assert not _has(_violations(config, data_dir), "bedrock-account-a")


def test_guard5_unreadable_status_is_violation(data_dir: Path) -> None:
    anima_dir = _prepare_data_dir(data_dir)
    (anima_dir / "status.json").write_text("{broken", encoding="utf-8")
    config = _make_config(enclave=_enclave())
    assert _has(_violations(config, data_dir), "status.json")


# ---------------------------------------------------------------------------
# Guard 6: no "/" file root
# ---------------------------------------------------------------------------


def test_guard6_full_access_root(data_dir: Path) -> None:
    anima_dir = _prepare_data_dir(data_dir)
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots": ["/"]}),
        encoding="utf-8",
    )
    config = _make_config(enclave=_enclave())
    assert _has(_violations(config, data_dir), anima_dir.name)


# ---------------------------------------------------------------------------
# Guard 7: auth mode + trust localhost
# ---------------------------------------------------------------------------


def test_guard7_bad_auth(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    (data_dir / "auth.json").write_text(
        json.dumps({"auth_mode": "local_trust", "trust_localhost": True, "users": [], "sessions": {}}),
        encoding="utf-8",
    )
    config = _make_config(enclave=_enclave())
    violations = _violations(config, data_dir)
    assert _has(violations, "local_trust")
    assert _has(violations, "trust_localhost")


# ---------------------------------------------------------------------------
# Guard 8: bind host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.5", "example.com"],
)
def test_guard8_non_loopback_host(data_dir: Path, host: str) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave())
    assert _has(_violations(config, data_dir, host=host), host)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_guard8_loopback_host_passes(data_dir: Path, host: str) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave())
    assert not _has(_violations(config, data_dir, host=host), host)


# ---------------------------------------------------------------------------
# Guard 9: egress probe
# ---------------------------------------------------------------------------


def test_guard9_reachable_probe_target_is_violation(data_dir: Path) -> None:
    import socket

    _prepare_data_dir(data_dir)
    listener = socket.create_server(("127.0.0.1", 0))
    try:
        target = f"127.0.0.1:{listener.getsockname()[1]}"
        config = _make_config(enclave=_enclave(egress_probe=[target]))
        assert _has(_violations(config, data_dir), target)
    finally:
        listener.close()


def test_guard9_blocked_probe_target_passes(data_dir: Path) -> None:
    import socket

    _prepare_data_dir(data_dir)
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()  # nothing listens -> connection refused, as a firewall reject would
    config = _make_config(enclave=_enclave(egress_probe=[f"127.0.0.1:{port}"]))
    assert _violations(config, data_dir) == []


@pytest.mark.parametrize("target", ["no-port", "host:abc", ":443"])
def test_guard9_malformed_probe_target_is_violation(data_dir: Path, target: str) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave(egress_probe=[target]))
    assert _has(_violations(config, data_dir), target)


# ---------------------------------------------------------------------------
# enforce_enclave_runtime
# ---------------------------------------------------------------------------


def test_enforce_raises_on_violation(data_dir: Path) -> None:
    config = _make_config(enclave=_enclave(name="", socket_path="", entry_anima=""))
    with pytest.raises(EnclaveViolationError):
        enforce_enclave_runtime(config, data_dir, host=None)


def test_enforce_passes_clean_config(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave())
    # No exception expected.
    enforce_enclave_runtime(config, data_dir, host="127.0.0.1")


# ---------------------------------------------------------------------------
# Full clean config produces no violations
# ---------------------------------------------------------------------------


def test_clean_enclave_no_violations(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    config = _make_config(enclave=_enclave())
    assert _violations(config, data_dir, host="127.0.0.1") == []


# ---------------------------------------------------------------------------
# cli _start_foreground refuses to boot on violation
# ---------------------------------------------------------------------------


def test_start_foreground_exits_without_uvicorn(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cli.commands import server as server_cmd

    _prepare_data_dir(data_dir)
    # Force a violation: empty required fields.
    (data_dir / "config.json").write_text(
        json.dumps(
            {
                **_base_config(),
                "enclave": _enclave(name="", socket_path="", entry_anima=""),
            }
        ),
        encoding="utf-8",
    )
    from core.config import invalidate_cache

    invalidate_cache()

    called = {"uvicorn": False}

    def _fake_uvicorn_run(*args: object, **kwargs: object) -> None:
        called["uvicorn"] = True

    monkeypatch.setattr("uvicorn.run", _fake_uvicorn_run)
    monkeypatch.setattr(server_cmd, "read_server_pid", lambda: None)
    monkeypatch.setattr(server_cmd, "_find_server_pid_by_process", lambda **_: None)
    monkeypatch.setattr(server_cmd, "_kill_orphan_runners", lambda **_: 0)
    monkeypatch.setattr(server_cmd, "read_server_pid", lambda: None)

    args = SimpleNamespace(host="127.0.0.1", port=18500)
    with pytest.raises(SystemExit) as exc_info:
        server_cmd._start_foreground(args)
    assert exc_info.value.code == 2
    assert called["uvicorn"] is False
    # Refused before any side effects (PID file, watchdog).
    assert not (data_dir / "server.pid").exists()


def test_guard5_model_profile_credential_must_be_allowed(data_dir: Path) -> None:
    _prepare_data_dir(data_dir)
    profiles = {"luna-max": "a:azure/responses/gpt-6-luna", "opus-high": ""}
    config = _make_config(enclave=_enclave(model_profiles=profiles))
    assert _has(_violations(config, data_dir), "azure")
    config = _make_config(enclave=_enclave(allowed_llm_credentials=["anthropic", "azure"], model_profiles=profiles))
    assert not _has(_violations(config, data_dir), "azure")
