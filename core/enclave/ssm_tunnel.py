# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""SSM Session Manager port-forward tunnels for enclave SQL sources.

Each configured source owns a single tunnel (a ``session-manager-plugin``
process bound to a loopback port).  The plugin runs in its own session and is
recorded in a state file, so the short-lived ``animaworks-tool`` processes
that run one query each reuse the same tunnel instead of opening an SSM
session per query.  A tunnel is re-established when its plugin has died or its
target changed, and shut down after ``idle_shutdown_s`` of inactivity: by the
reaper of a long-lived process, by the next caller, or at the latest by the
Session Manager idle timeout.

AWS credentials are obtained from the secrets store.  They reach the plugin
through its environment, so it can sign ``OpenDataChannel`` with SigV4 (AWS
requires this from 2027-05-28), and never appear on its command line.
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from core.enclave.config import EnclaveSsmTunnelConfig
from core.enclave.secrets import read_enclave_secret

logger = logging.getLogger(__name__)

_PLUGIN_WAIT_TIMEOUT_S = 30.0
_DOCUMENT_NAME = "AWS-StartPortForwardingSessionToRemoteHost"


class TunnelError(RuntimeError):
    """Raised when an SSM tunnel cannot be established or kept alive."""


class _Tunnel:
    __slots__ = (
        "name",
        "config",
        "fingerprint",
        "local_port",
        "process",
        "session",
        "last_used",
        "ssm",
        "port_lock_fd",
    )

    def __init__(
        self,
        name: str,
        config: EnclaveSsmTunnelConfig,
        fingerprint: str,
        local_port: int,
        process: subprocess.Popen,
        session: dict[str, Any],
    ) -> None:
        self.name = name
        self.config = config
        self.fingerprint = fingerprint
        self.local_port = local_port
        self.process = process
        self.session = session
        self.last_used = time.time()
        self.ssm: Any = None
        self.port_lock_fd: int | None = None


_LOCKS: dict[str, threading.Lock] = {}
_TUNNELS: dict[str, _Tunnel] = {}
_REAPER_STARTED = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_dir(name: str) -> Path:
    from core.paths import get_data_dir

    path = get_data_dir() / "run" / name
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _port_lock_dir() -> Path:
    return _run_dir("ssm-ports")


def _reserve_port(config: EnclaveSsmTunnelConfig) -> tuple[int, int | None]:
    """Pick the loopback port for a new tunnel.

    Without ``local_port`` a random free port is used.  With it, the first of
    ``local_port_count`` consecutive ports that no other process has reserved
    (an exclusive ``flock`` held for the tunnel's lifetime) and that is free to
    bind is returned together with the lock descriptor.
    """
    if config.local_port is None:
        return _free_port(), None
    lock_dir = _port_lock_dir()
    for port in range(config.local_port, config.local_port + config.local_port_count):
        fd = os.open(lock_dir / f"{port}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        if not _port_bindable(port):
            # Held by something outside this reservation scheme; leave it alone.
            _release_port(fd)
            continue
        return port, fd
    raise TunnelError("no free loopback port for the SSM tunnel")


def _port_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _release_port(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Shared state (one file per source, so other processes can reuse a tunnel)
# ---------------------------------------------------------------------------


def _state_path(source_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", source_name)
    return _run_dir("ssm-tunnels") / f"{safe}.json"


@contextlib.contextmanager
def _state_lock(source_name: str) -> Iterator[None]:
    """Serialize find-or-create of a source's tunnel across processes."""
    fd = os.open(_state_path(source_name).with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fingerprint(config: EnclaveSsmTunnelConfig, remote_host: str, remote_port: int) -> str:
    payload = json.dumps(
        {"config": config.model_dump(mode="json"), "host": remote_host, "port": remote_port},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _read_state(source_name: str) -> tuple[dict[str, Any], float] | None:
    path = _state_path(source_name)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        mtime = path.stat().st_mtime
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    if not isinstance(state.get("pid"), int) or not isinstance(state.get("local_port"), int):
        return None
    if not isinstance(state.get("session_id"), str) or not isinstance(state.get("fingerprint"), str):
        return None
    return state, mtime


def _write_state(tunnel: _Tunnel) -> None:
    path = _state_path(tunnel.name)
    data = {
        "pid": tunnel.process.pid,
        "local_port": tunnel.local_port,
        "session_id": str(tunnel.session.get("SessionId") or ""),
        "fingerprint": tunnel.fingerprint,
    }
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _owns_state(tunnel: _Tunnel) -> float | None:
    """Return the state file's mtime (last use by any process) if it records *tunnel*."""
    found = _read_state(tunnel.name)
    if found is None:
        return None
    state, mtime = found
    if state["session_id"] != tunnel.session.get("SessionId"):
        return None
    return mtime


def _touch(tunnel: _Tunnel) -> None:
    tunnel.last_used = time.time()
    with contextlib.suppress(OSError):
        os.utime(_state_path(tunnel.name))


def _pid_runs_session(pid: int, session_id: str) -> bool:
    """True when *pid* is still the plugin of *session_id* (guards against pid reuse)."""
    if pid <= 0 or not session_id:
        return False
    try:
        out = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return session_id in out


def _teardown_shared(config: EnclaveSsmTunnelConfig, source_name: str, state: dict[str, Any]) -> None:
    """Stop a tunnel recorded by another process and forget it."""
    if _pid_runs_session(state["pid"], state["session_id"]):
        try:
            credentials = _read_aws_credentials(config)
            _ssm_client(config, credentials).terminate_session(SessionId=state["session_id"])
        except Exception:  # noqa: BLE001 - best-effort; the session also times out server-side
            logger.debug("Could not terminate shared SSM session", exc_info=True)
        with contextlib.suppress(OSError):
            os.kill(state["pid"], signal.SIGTERM)
    with contextlib.suppress(OSError):
        _state_path(source_name).unlink()


def _adopt_shared(source_name: str, config: EnclaveSsmTunnelConfig, fingerprint: str) -> int | None:
    """Return the port of a live, matching, non-idle tunnel another process started."""
    found = _read_state(source_name)
    if found is None:
        return None
    state, mtime = found
    reusable = (
        state["fingerprint"] == fingerprint
        and time.time() - mtime <= config.idle_shutdown_s
        and _pid_runs_session(state["pid"], state["session_id"])
        and _port_open(state["local_port"])
    )
    if not reusable:
        _teardown_shared(config, source_name, state)
        return None
    with contextlib.suppress(OSError):
        os.utime(_state_path(source_name))
    return state["local_port"]


# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------


def _resolve_target(ec2: Any, config: EnclaveSsmTunnelConfig) -> str:
    if config.target_instance_id:
        return config.target_instance_id
    assert config.target_tag_name is not None  # enforced by the config model
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [config.target_tag_name]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    instances = [
        instance for reservation in response.get("Reservations", []) for instance in reservation.get("Instances", [])
    ]
    if not instances:
        raise TunnelError("no running instance matched the configured target tag")
    if len(instances) > 1:
        raise TunnelError("more than one running instance matched the configured target tag")
    return instances[0]["InstanceId"]


def _read_aws_credentials(config: EnclaveSsmTunnelConfig) -> dict[str, str]:
    raw = read_enclave_secret(config.aws_secret)
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise TunnelError("the configured AWS secret is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TunnelError("the configured AWS secret must be a JSON object")
    access_key = payload.get("aws_access_key_id")
    secret_key = payload.get("aws_secret_access_key")
    if not isinstance(access_key, str) or not access_key or not isinstance(secret_key, str) or not secret_key:
        raise TunnelError("the configured AWS secret must contain aws_access_key_id and aws_secret_access_key")
    return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key}


def _boto_session(config: EnclaveSsmTunnelConfig, credentials: dict[str, str]) -> Any:
    import boto3

    return boto3.Session(region_name=config.region, **credentials)


def _ssm_client(config: EnclaveSsmTunnelConfig, credentials: dict[str, str]) -> Any:
    return _boto_session(config, credentials).client("ssm", region_name=config.region)


def _start_session(
    config: EnclaveSsmTunnelConfig,
    credentials: dict[str, str],
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Start an SSM session; return ``(session, request, ssm_client)``."""
    boto_session = _boto_session(config, credentials)
    ssm = boto_session.client("ssm", region_name=config.region)
    target = _resolve_target(boto_session.client("ec2", region_name=config.region), config)
    request = {
        "Target": target,
        "DocumentName": _DOCUMENT_NAME,
        "Parameters": {
            "host": [remote_host],
            "portNumber": [str(remote_port)],
            "localPortNumber": [str(local_port)],
        },
    }
    response = ssm.start_session(**request)
    if not isinstance(response, dict):
        raise TunnelError("SSM start_session returned an invalid response")
    return response, request, ssm


def _plugin_command(config: EnclaveSsmTunnelConfig, session: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """Build the session-manager-plugin command line (no AWS keys on it).

    Mirrors what the AWS CLI passes: the start_session response, the region,
    the operation name, an empty profile, the original request and the endpoint.
    """
    response = {key: session.get(key) for key in ("SessionId", "TokenValue", "StreamUrl")}
    return [
        config.plugin_path,
        json.dumps(response),
        config.region,
        "StartSession",
        "",
        json.dumps(request),
        f"https://ssm.{config.region}.amazonaws.com",
    ]


def _plugin_env(config: EnclaveSsmTunnelConfig, credentials: dict[str, str]) -> dict[str, str]:
    """The plugin's environment: the caller's, minus any AWS settings, plus the tunnel's keys.

    With an empty profile the plugin signs ``OpenDataChannel`` with the
    credentials it finds in the environment; without them the request goes
    out unsigned, which Session Manager rejects from 2027-05-28.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(
        {
            "AWS_ACCESS_KEY_ID": credentials["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": credentials["aws_secret_access_key"],
            "AWS_REGION": config.region,
            "AWS_DEFAULT_REGION": config.region,
        }
    )
    return env


def _start_plugin(
    config: EnclaveSsmTunnelConfig,
    session: dict[str, Any],
    request: dict[str, Any],
    credentials: dict[str, str],
    port_lock_fd: int | None,
) -> subprocess.Popen:
    # Its own session, so the plugin outlives the short-lived tool process that started it.
    # It inherits the port reservation lock, which then lasts as long as the plugin does.
    return subprocess.Popen(
        _plugin_command(config, session, request),
        env=_plugin_env(config, credentials),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=() if port_lock_fd is None else (port_lock_fd,),
    )


def _wait_for_port(port: int, timeout: float = _PLUGIN_WAIT_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.1)
    raise TunnelError("the port-forward tunnel did not become reachable within the timeout")


# ---------------------------------------------------------------------------
# Tunnels owned by this process
# ---------------------------------------------------------------------------


def _lock_for(name: str) -> threading.Lock:
    lock = _LOCKS.get(name)
    if lock is None:
        lock = threading.Lock()
        _LOCKS[name] = lock
    return lock


def _is_idle(tunnel: _Tunnel) -> bool:
    last_used = max(tunnel.last_used, _owns_state(tunnel) or 0.0)
    return time.time() - last_used > tunnel.config.idle_shutdown_s


def _is_alive(tunnel: _Tunnel) -> bool:
    return tunnel.process.poll() is None and _port_open(tunnel.local_port)


def _establish(
    source_name: str,
    config: EnclaveSsmTunnelConfig,
    fingerprint: str,
    remote_host: str,
    remote_port: int,
) -> int:
    credentials = _read_aws_credentials(config)
    local_port, port_lock_fd = _reserve_port(config)
    try:
        session, request, ssm = _start_session(config, credentials, local_port, remote_host, remote_port)
        process = _start_plugin(config, session, request, credentials, port_lock_fd)
    except BaseException:
        _release_port(port_lock_fd)
        raise
    tunnel = _Tunnel(source_name, config, fingerprint, local_port, process, session)
    tunnel.ssm = ssm
    tunnel.port_lock_fd = port_lock_fd
    _TUNNELS[source_name] = tunnel
    try:
        _wait_for_port(local_port)
    except TunnelError:
        _stop_tunnel(tunnel)
        _TUNNELS.pop(source_name, None)
        raise
    _write_state(tunnel)
    _touch(tunnel)
    logger.info(
        "Established SSM tunnel for enclave source %s (port %d, pid %s)",
        source_name,
        local_port,
        tunnel.process.pid,
    )
    return local_port


def ensure_tunnel(
    source_name: str,
    config: EnclaveSsmTunnelConfig,
    remote_host: str,
    remote_port: int,
) -> int:
    """Return the loopback port for *source_name*, creating or reusing its tunnel.

    A tunnel started by this or another process is reused while its plugin is
    alive, its target is unchanged and it has not been idle too long; otherwise
    it is torn down and re-established.  Safe to call from multiple threads and
    processes.
    """
    fingerprint = _fingerprint(config, remote_host, remote_port)
    with _lock_for(source_name):
        _ensure_reaper()
        current = _TUNNELS.get(source_name)
        if current is not None and current.fingerprint == fingerprint and _is_alive(current) and not _is_idle(current):
            _touch(current)
            return current.local_port

        with _state_lock(source_name):
            if current is not None:
                _TUNNELS.pop(source_name, None)
                _stop_tunnel(current)
            port = _adopt_shared(source_name, config, fingerprint)
            if port is not None:
                return port
            return _establish(source_name, config, fingerprint, remote_host, remote_port)


def _stop_tunnel(tunnel: _Tunnel) -> None:
    try:
        if _owns_state(tunnel) is not None:
            with contextlib.suppress(OSError):
                _state_path(tunnel.name).unlink()
        session_id = tunnel.session.get("SessionId")
        if tunnel.ssm is not None and session_id:
            try:
                tunnel.ssm.terminate_session(SessionId=session_id)
            except Exception:  # noqa: BLE001 - best-effort; the session also times out server-side
                logger.debug("Could not terminate SSM session", exc_info=True)
        try:
            tunnel.process.terminate()
        except OSError:
            pass
        try:
            tunnel.process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            try:
                tunnel.process.kill()
            except OSError:
                pass
    finally:
        _release_port(tunnel.port_lock_fd)
        tunnel.port_lock_fd = None


def stop_tunnel(source_name: str) -> None:
    """Stop and forget the tunnel for *source_name* (if any)."""
    with _lock_for(source_name), _state_lock(source_name):
        tunnel = _TUNNELS.pop(source_name, None)
        if tunnel is not None:
            _stop_tunnel(tunnel)
            logger.info("Stopped SSM tunnel for enclave source %s", source_name)


def stop_idle_tunnels() -> None:
    """Stop any tunnel that has been idle for longer than its configured window."""
    for source_name, tunnel in list(_TUNNELS.items()):
        with _lock_for(source_name), _state_lock(source_name):
            current = _TUNNELS.get(source_name)
            if current is not tunnel:
                continue
            if current is not None and _is_idle(current):
                _TUNNELS.pop(source_name, None)
                _stop_tunnel(current)
                logger.info("Stopped idle SSM tunnel for enclave source %s", source_name)


def stop_all_tunnels() -> None:
    """Stop every tunnel this process started."""
    for source_name in list(_TUNNELS):
        stop_tunnel(source_name)


def _leave_shared_tunnels_at_exit() -> None:
    """At exit, keep recorded tunnels running for the next process; stop the rest."""
    for source_name, tunnel in list(_TUNNELS.items()):
        if _owns_state(tunnel) is None:
            stop_tunnel(source_name)
    _TUNNELS.clear()


def _ensure_reaper() -> None:
    global _REAPER_STARTED
    if _REAPER_STARTED:
        return
    _REAPER_STARTED = True

    def _loop() -> None:
        while True:
            time.sleep(5)
            try:
                stop_idle_tunnels()
            except Exception:  # noqa: BLE001 - a reaper must never crash
                logger.debug("Idle tunnel reaper caught an error", exc_info=True)

    threading.Thread(target=_loop, name="enclave-ssm-reaper", daemon=True).start()


atexit.register(_leave_shared_tunnels_at_exit)


__all__ = [
    "TunnelError",
    "ensure_tunnel",
    "stop_all_tunnels",
    "stop_idle_tunnels",
    "stop_tunnel",
]
