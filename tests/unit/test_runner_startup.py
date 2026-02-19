"""Unit tests for AnimaRunner startup order and ping readiness."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── AnimaRunner ping readiness ───────────────────────────


class TestAnimaRunnerPingReadiness:
    """Verify that ping returns 'initializing' before DigitalAnima is ready."""

    def _make_runner(self, tmp_path: Path):
        from core.supervisor.runner import AnimaRunner

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        anima_dir = animas_dir / "test-anima"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("test")
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        socket_path = tmp_path / "test.sock"

        return AnimaRunner(
            anima_name="test-anima",
            socket_path=socket_path,
            animas_dir=animas_dir,
            shared_dir=shared_dir,
        )

    @pytest.mark.asyncio
    async def test_ping_returns_initializing_before_ready(self, tmp_path):
        """Before _ready_event is set, ping should return status=initializing."""
        runner = self._make_runner(tmp_path)

        # _ready_event is not set, so ping should report initializing
        result = await runner._handle_ping({})
        assert result["status"] == "initializing"
        assert result["anima"] == "test-anima"

    @pytest.mark.asyncio
    async def test_ping_returns_ok_after_ready(self, tmp_path):
        """After _ready_event is set, ping should return status=ok."""
        runner = self._make_runner(tmp_path)
        runner._ready_event.set()

        result = await runner._handle_ping({})
        assert result["status"] == "ok"
        assert result["anima"] == "test-anima"
        assert "uptime_sec" in result

    @pytest.mark.asyncio
    async def test_run_starts_ipc_before_anima_init(self, tmp_path):
        """IPC server should start before DigitalAnima is constructed."""
        runner = self._make_runner(tmp_path)
        call_order: list[str] = []

        mock_ipc_server = AsyncMock()

        async def mock_ipc_start():
            call_order.append("ipc_start")

        mock_ipc_server.start = mock_ipc_start
        mock_ipc_server.stop = AsyncMock()

        mock_anima = MagicMock()

        def mock_anima_init(*args, **kwargs):
            call_order.append("anima_init")
            return mock_anima

        with (
            patch(
                "core.supervisor.runner.IPCServer",
                return_value=mock_ipc_server,
            ),
            patch(
                "core.supervisor.runner.DigitalAnima",
                side_effect=mock_anima_init,
            ),
        ):
            # Start run() in background, then trigger shutdown
            async def trigger_shutdown():
                # Wait for IPC + anima init to complete
                for _ in range(50):
                    if runner._ready_event.is_set():
                        break
                    await asyncio.sleep(0.05)
                runner.shutdown_event.set()

            task = asyncio.create_task(runner.run())
            shutdown_task = asyncio.create_task(trigger_shutdown())

            await asyncio.wait_for(
                asyncio.gather(task, shutdown_task), timeout=5.0
            )

        # IPC should start BEFORE anima initialization
        assert call_order == ["ipc_start", "anima_init"]
