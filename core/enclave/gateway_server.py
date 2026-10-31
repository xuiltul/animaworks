# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Lifecycle and Unix-socket wiring for the enclave gateway.

* :func:`prepare_socket` creates and binds the Unix socket with 0660
  permissions (plus an optional group ownership).
* :class:`PeerCredHttpProtocol` is a uvicorn HTTP protocol that records the
  connecting process's identity (pid/uid/gid) from ``SO_PEERCRED`` (Linux) or
  ``LOCAL_PEERCRED`` (macOS) into the
  ASGI scope so the gateway can authorise by uid.
* :func:`start_gateway` / :func:`stop_gateway` run/stop a uvicorn server
  over that socket inside the app's event loop.
"""

from __future__ import annotations

import asyncio
import grp
import logging
import os
import socket
import struct
from pathlib import Path
from typing import Any

import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from core.enclave.config import EnclaveConfig
from core.enclave.gateway import UsageReader, create_gateway_app

logger = logging.getLogger(__name__)

_SO_PEERCRED_SIZE = struct.calcsize("3i")

# macOS/BSD: getsockopt(SOL_LOCAL, LOCAL_PEERCRED) returns ``struct xucred``
# (u_int cr_version; uid_t cr_uid; short cr_ngroups; gid_t cr_groups[16]) and
# LOCAL_PEERPID the peer pid.  SOL_LOCAL is 0 and is not exported by Python.
_SOL_LOCAL = 0
_LOCAL_PEERPID = 2
_XUCRED_FORMAT = "IIh16I"
_XUCRED_VERSION = 0


def _read_peercred_linux(sock: socket.socket) -> dict[str, int]:
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _SO_PEERCRED_SIZE)
    pid, uid, gid = struct.unpack("3i", raw)
    return {"pid": pid, "uid": uid, "gid": gid}


def _read_peercred_local(sock: socket.socket) -> dict[str, int]:
    raw = sock.getsockopt(_SOL_LOCAL, socket.LOCAL_PEERCRED, struct.calcsize(_XUCRED_FORMAT))
    version, uid, ngroups, *groups = struct.unpack(_XUCRED_FORMAT, raw)
    if version != _XUCRED_VERSION or ngroups < 1:
        raise OSError("unexpected xucred layout")
    pid_raw = sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID, struct.calcsize("i"))
    (pid,) = struct.unpack("i", pid_raw)
    return {"pid": pid, "uid": uid, "gid": groups[0]}


def read_peercred(sock: socket.socket | None) -> dict[str, int] | None:
    """Read (pid, uid, gid) of the connected peer.

    Uses ``SO_PEERCRED`` on Linux and ``LOCAL_PEERCRED``/``LOCAL_PEERPID`` on
    macOS.  Returns ``None`` when the socket is unavailable, the platform
    offers neither option, or the option cannot be read, which callers must
    treat as an untrusted peer.
    """
    if sock is None:
        return None
    try:
        if hasattr(socket, "SO_PEERCRED"):
            return _read_peercred_linux(sock)
        if hasattr(socket, "LOCAL_PEERCRED"):
            return _read_peercred_local(sock)
    except (OSError, TypeError, ValueError, struct.error):
        return None
    return None


class PeerCredHttpProtocol(H11Protocol):
    """h11 HTTP protocol that exposes the peer credentials in the ASGI scope.

    Choose this over the ``httptools`` variant because it is pure-Python and
    needs no extra C extension; it is registered via ``Config(http=...)``.
    """

    def __init__(self, config: Any, server_state: Any, app_state: Any, _loop: Any = None) -> None:
        super().__init__(config, server_state, app_state, _loop)
        self._peercred: dict[str, int] | None = None

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        try:
            raw_sock = transport.get_extra_info("socket")
        except Exception:
            raw_sock = None
        self._peercred = read_peercred(raw_sock)

    def handle_events(self) -> None:
        super().handle_events()
        if self._peercred and self.scope is not None:
            extensions = self.scope.setdefault("extensions", {})
            extensions["peercred"] = self._peercred


def prepare_socket(config: EnclaveConfig) -> socket.socket:
    """Create, bind, and tighten a Unix socket per *config*.

    Creates the parent directory (0750 if new), removes any stale socket
    file, binds ``socket_path``, applies 0660 permissions, and — when a
    ``socket_group`` is set — changes ownership to that group.  Raises on
    failure so the server can fail closed.
    """
    path = Path(config.socket_path)
    parent = path.parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(parent, 0o750)
        except OSError as exc:
            logger.warning("enclave gateway: could not chmod socket dir %s: %s", parent, exc)
    if path.exists():
        path.unlink()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        os.chmod(path, 0o660)
        if config.socket_group:
            gid = grp.getgrnam(config.socket_group).gr_gid
            os.chown(path, -1, gid)
        sock.listen(128)
    except Exception:
        sock.close()
        raise
    return sock


def _build_server(app: Any, config: EnclaveConfig) -> uvicorn.Server:
    server_config = uvicorn.Config(
        app,
        lifespan="off",
        http=PeerCredHttpProtocol,
        ws=None,
        log_level="warning",
        timeout_keep_alive=max(5, min(config.request_timeout_s, 30)),
    )
    return uvicorn.Server(server_config)


async def start_gateway(app: Any, *, usage_reader: UsageReader | None = None) -> None:
    """Start the enclave gateway in the current event loop, if enabled.

    Reads the global config, prepares the Unix socket, and runs a uvicorn
    server over it as a background task.  Any socket-preparation failure
    propagates so the outer lifespan raises (fail-closed).
    """
    from core.config import load_config
    from core.paths import get_data_dir

    # Seed the state slots so stop_gateway always finds an explicit None (even
    # on mock app.state objects that auto-create attributes).
    app.state.enclave_gateway = None
    app.state.enclave_gateway_task = None

    config = load_config()
    if not config.enclave.enabled:
        return
    data_dir = get_data_dir()
    supervisor = app.state.supervisor
    gateway_app = create_gateway_app(
        get_supervisor=lambda: supervisor,
        config=config.enclave,
        data_dir=data_dir,
        usage_reader=usage_reader,
    )
    sock = prepare_socket(config.enclave)
    server = _build_server(gateway_app, config.enclave)
    task = asyncio.get_running_loop().create_task(server.serve(sockets=[sock]))
    app.state.enclave_gateway = server
    app.state.enclave_gateway_task = task
    logger.info("enclave gateway started on %s", config.enclave.socket_path)


async def stop_gateway(app: Any) -> None:
    """Stop the enclave gateway, if it was started."""
    server = getattr(app.state, "enclave_gateway", None)
    task = getattr(app.state, "enclave_gateway_task", None)
    if server is not None:
        server.should_exit = True
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                logger.debug("enclave gateway task ended after stop request", exc_info=True)
    app.state.enclave_gateway = None
    app.state.enclave_gateway_task = None


__all__ = [
    "PeerCredHttpProtocol",
    "prepare_socket",
    "read_peercred",
    "start_gateway",
    "stop_gateway",
]
