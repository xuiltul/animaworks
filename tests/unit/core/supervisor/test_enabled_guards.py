"""Unit tests for enabled guards on start_anima / health respawn / wake paths."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor
from core.supervisor.process_handle import ProcessState


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor with test paths."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


def _write_status(animas_dir: Path, name: str, *, enabled: bool) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": enabled}),
        encoding="utf-8",
    )
    return anima_dir


class TestStartAnimaEnabledGuard:
    """start_anima refuses to spawn disabled animas."""

    @pytest.mark.asyncio
    async def test_start_anima_refuses_disabled(
        self, supervisor: ProcessSupervisor,
    ) -> None:
        """Disabled anima: no ProcessHandle spawn, processes stays empty."""
        _write_status(supervisor.animas_dir, "test-anima", enabled=False)

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

            MockHandle.assert_not_called()
            mock_handle.start.assert_not_awaited()
            assert "test-anima" not in supervisor.processes

    @pytest.mark.asyncio
    async def test_start_anima_starts_when_enabled(
        self, supervisor: ProcessSupervisor,
    ) -> None:
        """Enabled anima: process is spawned as before."""
        _write_status(supervisor.animas_dir, "test-anima", enabled=True)

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.get_pid = MagicMock(return_value=12345)
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False}),
            )
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

            MockHandle.assert_called_once()
            mock_handle.start.assert_awaited_once()
            assert "test-anima" in supervisor.processes

    @pytest.mark.asyncio
    async def test_start_anima_starts_when_no_status_file(
        self, supervisor: ProcessSupervisor,
    ) -> None:
        """Missing status.json is treated as enabled (backward compatible)."""
        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.get_pid = MagicMock(return_value=12345)
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False}),
            )
            MockHandle.return_value = mock_handle

            await supervisor.start_anima("test-anima")

            mock_handle.start.assert_awaited_once()
            assert "test-anima" in supervisor.processes


class TestHealthRespawnEnabledGuard:
    """Health failure / respawn must not re-spawn disabled animas."""

    @pytest.mark.asyncio
    async def test_respawn_transaction_does_not_spawn_disabled(
        self, supervisor: ProcessSupervisor,
    ) -> None:
        """_respawn_anima_transaction does not install a process when disabled."""
        _write_status(supervisor.animas_dir, "test-anima", enabled=False)
        supervisor.restart_policy.max_retries = 1

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            MockHandle.return_value = mock_handle

            result = await supervisor._respawn_anima_transaction("test-anima")

            assert result is None
            MockHandle.assert_not_called()
            assert "test-anima" not in supervisor.processes

    @pytest.mark.asyncio
    async def test_handle_process_failure_does_not_spawn_disabled(
        self, supervisor: ProcessSupervisor,
    ) -> None:
        """_handle_process_failure does not re-spawn a disabled anima."""
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 1

        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED
        supervisor.processes[name] = old_handle

        async def mock_stop(anima_name: str) -> None:
            supervisor.processes.pop(anima_name, None)

        supervisor.stop_anima = AsyncMock(side_effect=mock_stop)

        with (
            patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock),
            patch("core.supervisor.manager.ProcessHandle") as MockHandle,
        ):
            mock_handle = AsyncMock()
            MockHandle.return_value = mock_handle

            await supervisor._handle_process_failure(name, old_handle)

            MockHandle.assert_not_called()
            assert name not in supervisor.processes
            assert name not in supervisor._restarting
