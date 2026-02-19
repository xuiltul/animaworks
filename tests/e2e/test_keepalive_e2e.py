# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for IPC keep-alive and per-chunk timeout.

Validates:
  - IPCClient._resolve_ipc_timeout reads from config with correct defaults
  - ServerConfig.keepalive_interval default and configurability
  - End-to-end keep-alive streaming over real Unix sockets:
    keep-alive chunks prevent timeout, and missing keep-alive triggers timeout
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import ServerConfig
from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse, IPCServer

pytestmark = pytest.mark.e2e


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def socket_path(tmp_path: Path) -> Path:
    """Return a Unix socket path inside a temporary directory."""
    return tmp_path / "test.sock"


# ── 1. IPC timeout configuration (mirror of existing tests) ──


class TestIPCTimeoutConfigurable:
    """Test IPCClient._resolve_ipc_timeout reads from config."""

    def test_resolve_timeout_returns_default_60(self) -> None:
        """_resolve_ipc_timeout returns 60.0 when ServerConfig uses defaults."""
        with patch(
            "core.config.load_config",
            return_value=MagicMock(server=ServerConfig()),
        ):
            timeout = IPCClient._resolve_ipc_timeout()
        assert timeout == 60.0

    def test_resolve_timeout_returns_configured_value(self) -> None:
        """_resolve_ipc_timeout returns the configured ipc_stream_timeout."""
        with patch(
            "core.config.load_config",
            return_value=MagicMock(
                server=ServerConfig(ipc_stream_timeout=120),
            ),
        ):
            timeout = IPCClient._resolve_ipc_timeout()
        assert timeout == 120.0


# ── 2. Keep-alive config on ServerConfig ─────────────────────


class TestKeepaliveConfig:
    """Test ServerConfig.keepalive_interval defaults and configurability."""

    def test_keepalive_interval_default(self) -> None:
        """Default keepalive_interval is 30 seconds."""
        config = ServerConfig()
        assert config.keepalive_interval == 30

    def test_keepalive_interval_configurable(self) -> None:
        """keepalive_interval can be set to a custom value."""
        config = ServerConfig(keepalive_interval=15)
        assert config.keepalive_interval == 15


# ── 3. End-to-end keep-alive streaming ───────────────────────


class TestEndToEndKeepaliveStream:
    """Integration tests using real IPCServer + IPCClient over Unix sockets."""

    async def test_keepalive_prevents_timeout(
        self, socket_path: Path,
    ) -> None:
        """Keep-alive chunks reset the per-chunk timer so the stream survives.

        The handler takes >1s total but sends keep-alive chunks every 0.5s.
        With a per-chunk timeout of 0.8s the stream should complete without
        a TimeoutError.
        """

        async def _stream_keepalive(
            request: IPCRequest,
        ) -> AsyncIterator[IPCResponse]:
            """Async generator that emits keep-alive chunks with delays."""
            # First keep-alive chunk
            yield IPCResponse(
                id=request.id,
                stream=True,
                chunk="[keep-alive]",
            )
            await asyncio.sleep(0.5)
            # Second keep-alive chunk
            yield IPCResponse(
                id=request.id,
                stream=True,
                chunk="[keep-alive]",
            )
            await asyncio.sleep(0.5)
            # Final response
            yield IPCResponse(
                id=request.id,
                stream=True,
                done=True,
                result={"status": "ok"},
            )

        async def handler(request: IPCRequest) -> AsyncIterator[IPCResponse]:
            return _stream_keepalive(request)

        server = IPCServer(socket_path, handler)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()

            collected: list[IPCResponse] = []
            req = IPCRequest(id="ka-1", method="test")
            async for resp in client.send_request_stream(req, timeout=0.8):
                collected.append(resp)

            # Total time >1s but no timeout because chunks arrive every 0.5s
            assert len(collected) == 3

            # Verify keep-alive chunks are present
            keepalive_chunks = [
                r for r in collected
                if r.chunk == "[keep-alive]"
            ]
            assert len(keepalive_chunks) == 2

            # Verify final done response
            final = collected[-1]
            assert final.done is True
            assert final.result == {"status": "ok"}

        finally:
            await client.close()
            await server.stop()

    async def test_timeout_fires_without_keepalive(
        self, socket_path: Path,
    ) -> None:
        """Without keep-alive chunks the per-chunk timeout fires.

        The handler sends one initial chunk and then sleeps forever.
        The client should raise TimeoutError after 0.5s of silence.
        """
        async def _stream_stuck(
            request: IPCRequest,
        ) -> AsyncIterator[IPCResponse]:
            """Async generator that sends one chunk then hangs forever."""
            # Send one chunk, then go silent
            yield IPCResponse(
                id=request.id,
                stream=True,
                chunk="first",
            )
            # Simulate a stuck handler — no more chunks
            await asyncio.sleep(9999)

        async def handler(request: IPCRequest) -> AsyncIterator[IPCResponse]:
            return _stream_stuck(request)

        server = IPCServer(socket_path, handler)
        await server.start()

        client = IPCClient(socket_path)
        try:
            await client.connect()

            req = IPCRequest(id="to-1", method="test")
            with pytest.raises(TimeoutError):
                async for _ in client.send_request_stream(req, timeout=0.5):
                    pass

        finally:
            await client.close()
            # Force-close the server socket to unblock the stuck handler.
            # server.stop() alone would hang because wait_closed() waits
            # for the connection handler (which is stuck in sleep(9999)).
            if server.server:
                server.server.close()
            # Clean up the socket file
            if socket_path.exists():
                socket_path.unlink()
