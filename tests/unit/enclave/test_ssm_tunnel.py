# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for enclave SSM port-forward tunnel management."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from core.config import invalidate_cache
from core.enclave import ssm_tunnel
from core.enclave.config import EnclaveSsmTunnelConfig
from tests.helpers.filesystem import DEFAULT_TEST_CONFIG

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, args: list[str], *, alive: bool = True) -> None:
        self.args = args
        self.alive = alive
        self.terminated = False
        self.killed = False
        self.pid = 4242

    def poll(self) -> int | None:
        return None if self.alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return 0


class _FakeSSM:
    def __init__(self, instances: list | None = None, *, multiple: bool = False) -> None:
        self.started: list[dict] = []
        self.describes: list[dict] = []
        self._instances = instances if instances is not None else [{"InstanceId": "i-bastion"}]
        self._multiple = multiple

    def describe_instances(self, **kwargs: Any) -> dict:
        self.describes.append(kwargs)
        instances = self._instances
        if self._multiple:
            instances = instances + [{"InstanceId": "i-other"}]
        return {"Reservations": [{"Instances": instances}]}

    def terminate_session(self, **kwargs: Any) -> dict:
        self.terminated = kwargs
        return {}

    def start_session(self, **kwargs: Any) -> dict:
        self.started.append(kwargs)
        return {"SessionId": "session-id", "TokenValue": "session-token", "StreamUrl": "wss://example"}


class _FakeBotoSession:
    def __init__(self, ssm: _FakeSSM) -> None:
        self._ssm = ssm
        self.kwargs: dict = {}

    def client(self, service: str, region_name: str | None = None) -> _FakeSSM:
        # One fake serves both clients: start_session (ssm) and describe_instances (ec2).
        assert service in ("ssm", "ec2")
        assert region_name == "ap-northeast-1"
        return self._ssm


class _FakeBoto3:
    def __init__(self, session: _FakeBotoSession) -> None:
        self.session = session

    def Session(self, **kwargs: Any) -> _FakeBotoSession:  # noqa: N802
        self.session.kwargs = kwargs
        return self.session


def _tunnel_config(**overrides: Any) -> EnclaveSsmTunnelConfig:
    base: dict[str, Any] = {
        "region": "ap-northeast-1",
        "target_instance_id": "i-bastion",
        "aws_secret": "aws-creds",
        "plugin_path": "/fake/session-manager-plugin",
        "idle_shutdown_s": 600,
    }
    base.update(overrides)
    return EnclaveSsmTunnelConfig(**base)


def _write_config(data_dir: Path, *, secrets_dir: str) -> None:
    config: dict = dict(DEFAULT_TEST_CONFIG)
    config["enclave"] = {"enabled": True, "secrets_dir": secrets_dir}
    (data_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    invalidate_cache()


@pytest.fixture(autouse=True)
def _reset_tunnels() -> None:
    ssm_tunnel._TUNNELS.clear()
    ssm_tunnel._LOCKS.clear()
    ssm_tunnel._REAPER_STARTED = False
    yield
    ssm_tunnel.stop_all_tunnels()
    ssm_tunnel._TUNNELS.clear()
    ssm_tunnel._LOCKS.clear()
    ssm_tunnel._REAPER_STARTED = False


def _setup(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    ssm: _FakeSSM | None = None,
) -> tuple[_FakeBotoSession, list[str]]:
    secrets_dir = data_dir / "secretstore"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    (secrets_dir / "aws-creds").write_text(
        json.dumps({"aws_access_key_id": "access", "aws_secret_access_key": "secret"}),
        encoding="utf-8",
    )
    _write_config(data_dir, secrets_dir=str(secrets_dir))

    boto_session = _FakeBotoSession(ssm or _FakeSSM())
    boto3_mod = _FakeBoto3(boto_session)
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)

    spawned: list[str] = []

    def _fake_start_plugin(
        config: EnclaveSsmTunnelConfig,
        session: dict,
        request: dict,
        credentials: dict,  # noqa: ARG001
        port_lock_fd: int | None,  # noqa: ARG001
    ) -> _FakeProc:
        proc = _FakeProc(ssm_tunnel._plugin_command(config, session, request))
        spawned.append(json.dumps(proc.args))
        return proc

    monkeypatch.setattr(ssm_tunnel, "_start_plugin", _fake_start_plugin)
    monkeypatch.setattr(ssm_tunnel, "_free_port", lambda: 65432)
    monkeypatch.setattr(ssm_tunnel, "_port_open", lambda port: True)
    return boto_session, spawned


def _age_state(source_name: str, seconds: float) -> None:
    path = ssm_tunnel._state_path(source_name)
    then = time.time() - seconds
    os.utime(path, (then, then))


def _new_process() -> None:
    """Forget this process's tunnels without stopping them, as a later tool process would start."""
    ssm_tunnel._TUNNELS.clear()
    ssm_tunnel._LOCKS.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_session_uses_configured_target_and_document(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)

    port = ssm_tunnel.ensure_tunnel("main", _tunnel_config(), "rds.example", 3306)

    assert port == 65432
    assert len(boto_session._ssm.started) == 1
    started = boto_session._ssm.started[0]
    assert started["Target"] == "i-bastion"
    assert started["DocumentName"] == "AWS-StartPortForwardingSessionToRemoteHost"
    assert started["Parameters"] == {"host": ["rds.example"], "portNumber": ["3306"], "localPortNumber": ["65432"]}
    assert boto_session.kwargs["region_name"] == "ap-northeast-1"
    assert boto_session.kwargs["aws_access_key_id"] == "access"
    assert boto_session.kwargs["aws_secret_access_key"] == "secret"


def test_plugin_command_gets_session_output_not_aws_keys(data_dir: Path) -> None:  # noqa: ARG001
    config = _tunnel_config()
    args = ssm_tunnel._plugin_command(
        config,
        {"SessionId": "session-id", "TokenValue": "session-token", "StreamUrl": "wss://example"},
        {"Target": "i-bastion", "DocumentName": "AWS-StartPortForwardingSessionToRemoteHost", "Parameters": {}},
    )
    assert args[0] == "/fake/session-manager-plugin"
    response = json.loads(args[1])
    assert response["SessionId"] == "session-id"
    assert json.loads(args[5])["Target"] == "i-bastion"
    assert response["TokenValue"] == "session-token"
    assert response["StreamUrl"] == "wss://example"
    assert args[2] == "ap-northeast-1"
    assert args[3] == "StartSession"
    assert args[6] == "https://ssm.ap-northeast-1.amazonaws.com"
    # AWS keys must not appear anywhere on the plugin command line.
    assert "access" not in json.dumps(args)
    assert "secret" not in json.dumps(args)


def test_resolves_target_tag_to_instance(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    config = _tunnel_config(target_instance_id=None, target_tag_name="example-bastion")

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert boto_session._ssm.describes
    filters = boto_session._ssm.describes[0]["Filters"]
    assert {"Name": "tag:Name", "Values": ["example-bastion"]} in filters
    assert boto_session._ssm.started[0]["Target"] == "i-bastion"


def test_target_tag_zero_matches_raises(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch, ssm=_FakeSSM(instances=[]))
    config = _tunnel_config(target_instance_id=None, target_tag_name="nope")
    with pytest.raises(ssm_tunnel.TunnelError):
        ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)


def test_target_tag_multiple_matches_raises(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch, ssm=_FakeSSM(multiple=True))
    config = _tunnel_config(target_instance_id=None, target_tag_name="dup")
    with pytest.raises(ssm_tunnel.TunnelError):
        ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)


def test_tunnel_is_reused_within_idle_window(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, spawned = _setup(data_dir, monkeypatch)
    config = _tunnel_config()

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert len(boto_session._ssm.started) == 1
    assert len(spawned) == 1


def test_configured_local_port_is_used(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch)
    config = _tunnel_config(local_port=33061)

    assert ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306) == 33061


def test_concurrent_processes_reserve_distinct_ports(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch)
    config = _tunnel_config(local_port=33061, local_port_count=2)

    first, first_fd = ssm_tunnel._reserve_port(config)
    second, second_fd = ssm_tunnel._reserve_port(config)
    try:
        assert {first, second} == {33061, 33062}
        with pytest.raises(ssm_tunnel.TunnelError):
            ssm_tunnel._reserve_port(config)
    finally:
        ssm_tunnel._release_port(first_fd)
        ssm_tunnel._release_port(second_fd)

    port, fd = ssm_tunnel._reserve_port(config)
    ssm_tunnel._release_port(fd)
    assert port == 33061


def test_stopped_tunnel_releases_its_port(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch)
    config = _tunnel_config(local_port=33061)

    assert ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306) == 33061
    with pytest.raises(ssm_tunnel.TunnelError):
        ssm_tunnel._reserve_port(config)
    ssm_tunnel.stop_tunnel("main")
    port, fd = ssm_tunnel._reserve_port(config)
    ssm_tunnel._release_port(fd)
    assert port == 33061


def test_dead_process_is_reestablished(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    config = _tunnel_config()
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    # Simulate the plugin process dying.
    tunnel = ssm_tunnel._TUNNELS["main"]
    tunnel.process.alive = False

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert len(boto_session._ssm.started) == 2


def test_idle_tunnel_is_stopped(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    config = _tunnel_config(idle_shutdown_s=1)
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    tunnel = ssm_tunnel._TUNNELS["main"]
    tunnel.last_used -= 10
    _age_state("main", 10)

    ssm_tunnel.stop_idle_tunnels()

    assert "main" not in ssm_tunnel._TUNNELS
    assert tunnel.process.terminated is True
    # idle tunnel is re-established on next use
    port = ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    assert port == 65432
    assert len(boto_session._ssm.started) == 2


def test_stop_tunnel_tears_down(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch)
    config = _tunnel_config()
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    ssm_tunnel.stop_tunnel("main")

    assert "main" not in ssm_tunnel._TUNNELS


def test_plugin_signs_with_keys_from_its_environment(
    data_dir: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_popen(args: list[str], **kwargs: Any) -> _FakeProc:
        captured["args"] = args
        captured.update(kwargs)
        return _FakeProc(args)

    monkeypatch.setenv("AWS_PROFILE", "someone-else")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-token")
    config = _tunnel_config()
    # subprocess is shared with the fixtures' cleanup, so patch Popen only around the call.
    with monkeypatch.context() as patch:
        patch.setattr(ssm_tunnel.subprocess, "Popen", _fake_popen)
        ssm_tunnel._start_plugin(
            config,
            {"SessionId": "session-id", "TokenValue": "session-token", "StreamUrl": "wss://example"},
            {"Target": "i-bastion"},
            {"aws_access_key_id": "access", "aws_secret_access_key": "secret"},
            7,
        )

    env = captured["env"]
    assert env["AWS_ACCESS_KEY_ID"] == "access"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["AWS_REGION"] == "ap-northeast-1"
    assert "AWS_PROFILE" not in env
    assert "AWS_SESSION_TOKEN" not in env
    assert "secret" not in json.dumps(captured["args"])
    assert captured["start_new_session"] is True
    assert captured["pass_fds"] == (7,)


def test_later_process_reuses_the_recorded_tunnel(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, spawned = _setup(data_dir, monkeypatch)
    monkeypatch.setattr(ssm_tunnel, "_pid_runs_session", lambda pid, sid: pid == 4242 and sid == "session-id")
    config = _tunnel_config()

    first = ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    ssm_tunnel._leave_shared_tunnels_at_exit()
    _age_state("main", 300)
    _new_process()
    second = ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert first == second == 65432
    assert len(boto_session._ssm.started) == 1
    assert len(spawned) == 1
    # Reuse counts as use: the idle clock restarts.
    assert time.time() - ssm_tunnel._state_path("main").stat().st_mtime < 5


def test_exit_leaves_the_recorded_tunnel_running(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(data_dir, monkeypatch)
    ssm_tunnel.ensure_tunnel("main", _tunnel_config(), "rds.example", 3306)
    tunnel = ssm_tunnel._TUNNELS["main"]

    ssm_tunnel._leave_shared_tunnels_at_exit()

    assert tunnel.process.terminated is False
    assert ssm_tunnel._state_path("main").exists()


def test_idle_recorded_tunnel_is_replaced_and_its_session_ended(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    monkeypatch.setattr(ssm_tunnel, "_pid_runs_session", lambda pid, sid: pid == 4242 and sid == "session-id")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(ssm_tunnel.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    config = _tunnel_config(idle_shutdown_s=60)

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    ssm_tunnel._leave_shared_tunnels_at_exit()
    _age_state("main", 120)
    _new_process()
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert len(boto_session._ssm.started) == 2
    assert boto_session._ssm.terminated == {"SessionId": "session-id"}
    assert killed == [(4242, ssm_tunnel.signal.SIGTERM)]


def test_dead_recorded_tunnel_is_replaced_without_signalling(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    monkeypatch.setattr(ssm_tunnel, "_pid_runs_session", lambda pid, sid: False)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(ssm_tunnel.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    config = _tunnel_config()

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    ssm_tunnel._leave_shared_tunnels_at_exit()
    _new_process()
    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)

    assert len(boto_session._ssm.started) == 2
    assert killed == []


def test_recorded_tunnel_to_another_target_is_not_reused(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto_session, _ = _setup(data_dir, monkeypatch)
    monkeypatch.setattr(ssm_tunnel, "_pid_runs_session", lambda pid, sid: True)
    monkeypatch.setattr(ssm_tunnel.os, "kill", lambda pid, sig: None)
    config = _tunnel_config()

    ssm_tunnel.ensure_tunnel("main", config, "rds.example", 3306)
    ssm_tunnel._leave_shared_tunnels_at_exit()
    _new_process()
    ssm_tunnel.ensure_tunnel("main", config, "other-rds.example", 3306)

    assert len(boto_session._ssm.started) == 2
    assert boto_session._ssm.started[1]["Parameters"]["host"] == ["other-rds.example"]


def test_pid_check_rejects_a_reused_pid() -> None:
    # This test process is alive but is not the plugin of that session.
    assert ssm_tunnel._pid_runs_session(os.getpid(), "session-that-is-not-here") is False
    assert ssm_tunnel._pid_runs_session(0, "session-id") is False
