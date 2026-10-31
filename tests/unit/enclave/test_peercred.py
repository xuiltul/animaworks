# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Peer credential reading for the enclave gateway (Linux and macOS)."""

from __future__ import annotations

import os
import socket
import struct

import pytest

from core.enclave import gateway_server
from core.enclave.gateway_server import read_peercred


def test_read_peercred_real_socketpair_returns_own_uid() -> None:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        cred = read_peercred(a)
    finally:
        a.close()
        b.close()
    assert cred is not None
    assert cred["uid"] == os.getuid()
    assert cred["pid"] == os.getpid()


class _FakeLocalSock:
    def __init__(self, xucred: bytes, pid: int = 4242) -> None:
        self._xucred = xucred
        self._pid = pid

    def getsockopt(self, level: int, option: int, size: int) -> bytes:
        assert level == 0
        if option == gateway_server._LOCAL_PEERPID:
            return struct.pack("i", self._pid)
        assert size == struct.calcsize(gateway_server._XUCRED_FORMAT)
        return self._xucred


def _macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(socket, "SO_PEERCRED", raising=False)
    monkeypatch.setattr(socket, "LOCAL_PEERCRED", 1, raising=False)


def test_read_peercred_macos_parses_xucred(monkeypatch: pytest.MonkeyPatch) -> None:
    _macos(monkeypatch)
    groups = [20, 80] + [0] * 14
    sock = _FakeLocalSock(struct.pack(gateway_server._XUCRED_FORMAT, 0, 501, 2, *groups))
    assert read_peercred(sock) == {"pid": 4242, "uid": 501, "gid": 20}  # type: ignore[arg-type]


def test_read_peercred_macos_rejects_unknown_xucred_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _macos(monkeypatch)
    sock = _FakeLocalSock(struct.pack(gateway_server._XUCRED_FORMAT, 7, 501, 1, *([20] + [0] * 15)))
    assert read_peercred(sock) is None  # type: ignore[arg-type]


def test_read_peercred_without_platform_support_is_untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(socket, "SO_PEERCRED", raising=False)
    monkeypatch.delattr(socket, "LOCAL_PEERCRED", raising=False)
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert read_peercred(a) is None
    finally:
        a.close()
        b.close()
