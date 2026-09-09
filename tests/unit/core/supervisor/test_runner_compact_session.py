"""Unit tests for AnimaRunner._handle_compact_session."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.runner import AnimaRunner


def _make_runner(tmp_path: Path) -> AnimaRunner:
    runner = AnimaRunner(
        anima_name="sakura",
        socket_path=tmp_path / "sakura.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    runner.anima = MagicMock()
    runner.anima.agent.execution_mode = "s"
    return runner


@pytest.mark.asyncio
async def test_compact_ok(tmp_path: Path):
    runner = _make_runner(tmp_path)
    with patch(
        "core.session_compactor.run_idle_compaction",
        AsyncMock(return_value=True),
    ) as mock_compact:
        result = await runner._handle_compact_session({"thread_id": "alice-chat"})
    assert result == {"status": "ok", "thread_id": "alice-chat", "mode": "s"}
    mock_compact.assert_awaited_once_with(runner.anima, "alice-chat")


@pytest.mark.asyncio
async def test_compact_skipped_when_lock_fails(tmp_path: Path):
    runner = _make_runner(tmp_path)
    with patch(
        "core.session_compactor.run_idle_compaction",
        AsyncMock(return_value=False),
    ) as mock_compact:
        result = await runner._handle_compact_session({})
    assert result["status"] == "skipped"
    assert result["thread_id"] == "default"
    assert result["mode"] == "s"
    mock_compact.assert_awaited_once_with(runner.anima, "default")


@pytest.mark.asyncio
async def test_compact_invalid_thread_id(tmp_path: Path):
    runner = _make_runner(tmp_path)
    with patch("core.session_compactor.run_idle_compaction", AsyncMock()) as mock_compact, pytest.raises(ValueError):
        await runner._handle_compact_session({"thread_id": "bad thread id"})
    mock_compact.assert_not_called()


@pytest.mark.asyncio
async def test_compact_anima_not_initialized(tmp_path: Path):
    from core.exceptions import AnimaNotRunningError

    runner = AnimaRunner(
        anima_name="sakura",
        socket_path=tmp_path / "sakura.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    with pytest.raises(AnimaNotRunningError):
        await runner._handle_compact_session({})
