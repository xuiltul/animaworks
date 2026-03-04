"""Unit tests for ProcessSupervisor._inbox_wake_dispatcher()."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestInboxWakeDispatcher:
    """Test inbox wake file detection and IPC dispatch."""

    def _make_supervisor(self, tmp_path: Path):
        """Create a minimal ProcessSupervisor for testing."""
        from core.supervisor.manager import ProcessSupervisor

        animas_dir = tmp_path / "animas"
        shared_dir = tmp_path / "shared"
        run_dir = tmp_path / "run"
        animas_dir.mkdir()
        shared_dir.mkdir()
        run_dir.mkdir()

        sup = ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
        )
        return sup

    @pytest.mark.asyncio
    async def test_wake_file_triggers_process_inbox(self, tmp_path: Path) -> None:
        """A wake file should trigger a process_inbox IPC request."""
        sup = self._make_supervisor(tmp_path)

        mock_handle = MagicMock()
        mock_handle.send_request = AsyncMock(
            return_value=MagicMock(error=None, result={"action": "processed"}),
        )
        sup.processes["alice"] = mock_handle

        wake_dir = tmp_path / "run" / "inbox_wake"
        wake_dir.mkdir(parents=True)
        (wake_dir / "alice").write_text("alice", encoding="utf-8")

        # Run one iteration by starting the dispatcher and cancelling after brief delay
        sup._shutdown = False

        async def _run_briefly():
            task = asyncio.create_task(sup._inbox_wake_dispatcher())
            await asyncio.sleep(0.8)
            sup._shutdown = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_briefly()

        mock_handle.send_request.assert_called_once()
        call_args = mock_handle.send_request.call_args
        assert call_args[0][0] == "process_inbox"
        assert not (wake_dir / "alice").exists()

    @pytest.mark.asyncio
    async def test_wake_file_for_unknown_anima_is_cleaned(self, tmp_path: Path) -> None:
        """Wake file for unknown anima should be removed without error."""
        sup = self._make_supervisor(tmp_path)

        wake_dir = tmp_path / "run" / "inbox_wake"
        wake_dir.mkdir(parents=True)
        (wake_dir / "unknown").write_text("unknown", encoding="utf-8")

        sup._shutdown = False

        async def _run_briefly():
            task = asyncio.create_task(sup._inbox_wake_dispatcher())
            await asyncio.sleep(0.8)
            sup._shutdown = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_briefly()
        assert not (wake_dir / "unknown").exists()

    @pytest.mark.asyncio
    async def test_dotfiles_ignored(self, tmp_path: Path) -> None:
        """Files starting with '.' should be ignored."""
        sup = self._make_supervisor(tmp_path)

        wake_dir = tmp_path / "run" / "inbox_wake"
        wake_dir.mkdir(parents=True)
        (wake_dir / ".gitkeep").write_text("", encoding="utf-8")

        sup._shutdown = False

        async def _run_briefly():
            task = asyncio.create_task(sup._inbox_wake_dispatcher())
            await asyncio.sleep(0.8)
            sup._shutdown = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_briefly()
        assert (wake_dir / ".gitkeep").exists()
