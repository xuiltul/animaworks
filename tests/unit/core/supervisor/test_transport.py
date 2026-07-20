"""Tests for supervisor IPC transport selection."""

from __future__ import annotations

import logging
import os

import pytest

from core.supervisor.transport import cleanup_ipc_endpoint, resolve_client_endpoint, start_ipc_server

pytestmark = pytest.mark.skipif(os.name == "nt", reason="AF_UNIX path limits only apply on POSIX")


async def test_long_unix_socket_path_falls_back_to_tcp(tmp_path, monkeypatch, caplog):
    """A path that cannot fit in sun_path uses TCP metadata for clients."""
    monkeypatch.delenv("ANIMAWORKS_IPC_TRANSPORT", raising=False)
    socket_path = tmp_path / ("s" * 100 + ".sock")
    assert len(os.fsencode(str(socket_path))) >= 108

    async def handler(_reader, writer):
        writer.close()
        await writer.wait_closed()

    with caplog.at_level(logging.WARNING, logger="core.supervisor.transport"):
        server, endpoint = await start_ipc_server(socket_path, handler, limit=64 * 1024)

    try:
        assert endpoint.transport == "tcp"
        assert endpoint.host == "127.0.0.1"
        assert endpoint.port is not None

        client_endpoint = resolve_client_endpoint(socket_path)
        assert client_endpoint == endpoint
        assert "falling back to loopback TCP" in caplog.text
    finally:
        server.close()
        await server.wait_closed()
        cleanup_ipc_endpoint(socket_path)
