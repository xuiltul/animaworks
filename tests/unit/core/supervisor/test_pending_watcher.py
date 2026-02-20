# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for pending task watcher in core/supervisor/pending_executor.py.

Validates ``watcher_loop()`` and ``execute_pending_task()``:
- Watcher picks up pending JSON files and deletes them
- ``execute_pending_task`` calls BackgroundTaskManager.submit()
- Graceful handling when anima is not initialized
- Graceful handling when BackgroundTaskManager is not available
- Corrupt JSON files are removed with a warning
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.pending_executor import PendingTaskExecutor


# ── Helpers ──────────────────────────────────────────────────


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    """Create a PendingTaskExecutor pointing at a temporary directory."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    return PendingTaskExecutor(
        anima=None,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _make_executor_with_anima(tmp_path: Path) -> PendingTaskExecutor:
    """Create a PendingTaskExecutor with a mocked anima and BackgroundTaskManager."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)

    mock_anima = MagicMock()
    mock_anima._lock = asyncio.Lock()

    # Mock the background manager chain: anima.agent.background_manager
    bg_mgr = MagicMock()
    bg_mgr.submit = MagicMock(return_value="mock-task-id")
    mock_anima.agent.background_manager = bg_mgr

    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _write_pending_task(
    anima_dir: Path,
    task_id: str = "abc123def456",
    tool_name: str = "image_gen",
    subcommand: str = "3d",
    raw_args: list[str] | None = None,
) -> Path:
    """Write a pending task JSON file and return its path."""
    pending_dir = anima_dir / "state" / "background_tasks" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    task_desc = {
        "task_id": task_id,
        "tool_name": tool_name,
        "subcommand": subcommand,
        "raw_args": raw_args or [subcommand, "assets/avatar.png"],
        "anima_name": "test-anima",
        "anima_dir": str(anima_dir),
        "submitted_at": 1739800000.0,
        "status": "pending",
    }
    path = pending_dir / f"{task_id}.json"
    path.write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ── TestPendingTaskWatcherLoop ───────────────────────────────


class TestPendingTaskWatcherLoop:
    """Tests for watcher_loop()."""

    async def test_picks_up_pending_and_deletes_file(self, tmp_path: Path) -> None:
        """Watcher finds a pending JSON, processes it, and deletes the file."""
        executor = _make_executor_with_anima(tmp_path)
        task_path = _write_pending_task(executor._anima_dir)

        assert task_path.exists()

        # Let the watcher run for one iteration, then shut down
        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        # File should be deleted after being picked up
        assert not task_path.exists()

    async def test_calls_execute_pending_task(self, tmp_path: Path) -> None:
        """Watcher calls execute_pending_task with the parsed task descriptor."""
        executor = _make_executor_with_anima(tmp_path)
        _write_pending_task(executor._anima_dir, tool_name="local_llm", subcommand="generate")

        executed_tasks: list[dict] = []
        original_execute = executor.execute_pending_task

        async def capture_execute(task_desc: dict) -> None:
            executed_tasks.append(task_desc)

        executor.execute_pending_task = capture_execute  # type: ignore[assignment]

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        assert len(executed_tasks) == 1
        assert executed_tasks[0]["tool_name"] == "local_llm"
        assert executed_tasks[0]["subcommand"] == "generate"

    async def test_handles_corrupt_json_gracefully(self, tmp_path: Path) -> None:
        """Corrupt JSON files are deleted with a warning, not crashing the watcher."""
        executor = _make_executor_with_anima(tmp_path)

        # Write a corrupt JSON file
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = pending_dir / "corrupt.json"
        corrupt_path.write_text("{invalid json content", encoding="utf-8")

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        # Corrupt file should be cleaned up
        assert not corrupt_path.exists()

    async def test_processes_multiple_pending_files(self, tmp_path: Path) -> None:
        """Watcher processes all pending files in a single scan iteration."""
        executor = _make_executor_with_anima(tmp_path)

        # Write multiple pending tasks
        paths = [
            _write_pending_task(executor._anima_dir, task_id="task_aaa001", tool_name="image_gen"),
            _write_pending_task(executor._anima_dir, task_id="task_bbb002", tool_name="local_llm"),
            _write_pending_task(executor._anima_dir, task_id="task_ccc003", tool_name="transcribe"),
        ]

        executed_tasks: list[dict] = []

        async def capture_execute(task_desc: dict) -> None:
            executed_tasks.append(task_desc)

        executor.execute_pending_task = capture_execute  # type: ignore[assignment]

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        # All files should be deleted
        for p in paths:
            assert not p.exists()

        assert len(executed_tasks) == 3

    async def test_creates_pending_dir_if_missing(self, tmp_path: Path) -> None:
        """Watcher creates the pending directory if it does not exist."""
        executor = _make_executor_with_anima(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"

        assert not pending_dir.exists()

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        assert pending_dir.is_dir()

    async def test_stops_on_cancellation(self, tmp_path: Path) -> None:
        """Watcher exits cleanly on asyncio.CancelledError."""
        executor = _make_executor_with_anima(tmp_path)

        async def cancel_sleep(duration: float) -> None:
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=cancel_sleep):
            # Should not raise; just exit cleanly
            await executor.watcher_loop()


# ── TestExecutePendingTask ───────────────────────────────────


class TestExecutePendingTask:
    """Tests for execute_pending_task()."""

    async def test_calls_background_manager_submit(self, tmp_path: Path) -> None:
        """execute_pending_task submits the task to BackgroundTaskManager."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "assets/avatar.png"],
            "anima_name": "test-anima",
            "anima_dir": str(executor._anima_dir),
            "submitted_at": 1739800000.0,
            "status": "pending",
        }

        await executor.execute_pending_task(task_desc)

        bg_mgr.submit.assert_called_once()
        call_args = bg_mgr.submit.call_args
        # First positional arg should be composite name "image_gen:3d"
        assert call_args[0][0] == "image_gen:3d"
        # Second positional arg should be the tool_args dict
        tool_args = call_args[0][1]
        assert tool_args["subcommand"] == "3d"
        assert tool_args["raw_args"] == ["3d", "assets/avatar.png"]

    async def test_composite_name_without_subcommand(self, tmp_path: Path) -> None:
        """When subcommand is empty, composite name is just the tool_name."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        task_desc = {
            "task_id": "xyz789abc012",
            "tool_name": "transcribe",
            "subcommand": "",
            "raw_args": ["/path/to/audio.wav"],
            "anima_name": "test-anima",
            "anima_dir": str(executor._anima_dir),
        }

        await executor.execute_pending_task(task_desc)

        bg_mgr.submit.assert_called_once()
        composite_name = bg_mgr.submit.call_args[0][0]
        assert composite_name == "transcribe"

    async def test_handles_missing_anima_gracefully(self, tmp_path: Path) -> None:
        """When anima is None, execute_pending_task logs warning and returns."""
        executor = _make_executor(tmp_path)
        assert executor._anima is None

        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "test.png"],
        }

        # Should not raise
        await executor.execute_pending_task(task_desc)

    async def test_handles_missing_background_manager_gracefully(
        self, tmp_path: Path,
    ) -> None:
        """When BackgroundTaskManager is None, logs warning and returns."""
        executor = _make_executor_with_anima(tmp_path)
        executor._anima.agent.background_manager = None

        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "test.png"],
        }

        # Should not raise
        await executor.execute_pending_task(task_desc)

    async def test_passes_anima_dir_to_tool_args(self, tmp_path: Path) -> None:
        """anima_dir from the task descriptor is passed through to tool_args."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        custom_dir = "/home/user/.animaworks/animas/custom-anima"
        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "image_gen",
            "subcommand": "fullbody",
            "raw_args": ["fullbody", "--prompt", "test"],
            "anima_dir": custom_dir,
        }

        await executor.execute_pending_task(task_desc)

        tool_args = bg_mgr.submit.call_args[0][1]
        assert tool_args["anima_dir"] == custom_dir

    async def test_defaults_anima_dir_when_not_in_task(self, tmp_path: Path) -> None:
        """When anima_dir is not in task_desc, falls back to executor's _anima_dir."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "local_llm",
            "subcommand": "generate",
            "raw_args": ["generate", "hello"],
            # No "anima_dir" key
        }

        await executor.execute_pending_task(task_desc)

        tool_args = bg_mgr.submit.call_args[0][1]
        assert tool_args["anima_dir"] == str(executor._anima_dir)

    async def test_dispatch_fn_is_callable(self, tmp_path: Path) -> None:
        """The third argument to bg_mgr.submit() is a callable dispatch function."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        task_desc = {
            "task_id": "abc123def456",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "test.png"],
            "anima_dir": str(executor._anima_dir),
        }

        await executor.execute_pending_task(task_desc)

        dispatch_fn = bg_mgr.submit.call_args[0][2]
        assert callable(dispatch_fn)


# ── TestDispatchFn ────────────────────────────────────────────


class TestDispatchFn:
    """Tests for the _dispatch_fn closure built inside execute_pending_task."""

    async def _capture_dispatch_fn(
        self, tmp_path: Path, task_desc: dict,
    ):
        """Run execute_pending_task and return the captured dispatch function."""
        executor = _make_executor_with_anima(tmp_path)
        bg_mgr = executor._anima.agent.background_manager

        await executor.execute_pending_task(task_desc)

        bg_mgr.submit.assert_called_once()
        dispatch_fn = bg_mgr.submit.call_args[0][2]
        return dispatch_fn, executor

    async def test_dispatch_fn_builds_correct_command(self, tmp_path: Path) -> None:
        """_dispatch_fn builds: animaworks-tool <tool> <subcmd> ...raw_args -j."""
        task_desc = {
            "task_id": "cmd_test_001",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "--model", "test"],
            "anima_dir": str(tmp_path),
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "result output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            dispatch_fn("image_gen", {
                "subcommand": "3d",
                "raw_args": ["3d", "--model", "test"],
                "anima_dir": str(tmp_path),
            })

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd == ["animaworks-tool", "image_gen", "3d", "--model", "test", "-j"]

    async def test_dispatch_fn_deduplicates_subcommand(self, tmp_path: Path) -> None:
        """When raw_args[0] == subcommand, subcommand does not appear twice."""
        task_desc = {
            "task_id": "dedup_test",
            "tool_name": "local_llm",
            "subcommand": "generate",
            "raw_args": ["generate", "hello"],
            "anima_dir": str(tmp_path),
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            dispatch_fn("local_llm", {
                "subcommand": "generate",
                "raw_args": ["generate", "hello"],
                "anima_dir": str(tmp_path),
            })

            cmd = mock_run.call_args[0][0]
            # "generate" should appear only once, not twice
            assert cmd.count("generate") == 1
            assert cmd == ["animaworks-tool", "local_llm", "generate", "hello", "-j"]

    async def test_dispatch_fn_no_dedup_when_different(self, tmp_path: Path) -> None:
        """When raw_args[0] != subcommand, subcommand is prepended."""
        task_desc = {
            "task_id": "nodedup_test",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["--prompt", "1girl"],
            "anima_dir": str(tmp_path),
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            dispatch_fn("image_gen", {
                "subcommand": "3d",
                "raw_args": ["--prompt", "1girl"],
                "anima_dir": str(tmp_path),
            })

            cmd = mock_run.call_args[0][0]
            assert cmd == [
                "animaworks-tool", "image_gen", "3d",
                "--prompt", "1girl", "-j",
            ]

    async def test_dispatch_fn_sets_anima_dir_env(self, tmp_path: Path) -> None:
        """_dispatch_fn sets ANIMAWORKS_ANIMA_DIR environment variable."""
        anima_dir = str(tmp_path / "animas" / "sakura")
        task_desc = {
            "task_id": "env_test_001",
            "tool_name": "transcribe",
            "subcommand": "",
            "raw_args": ["/path/to/audio.wav"],
            "anima_dir": anima_dir,
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "transcribed"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            dispatch_fn("transcribe", {
                "subcommand": "",
                "raw_args": ["/path/to/audio.wav"],
                "anima_dir": anima_dir,
            })

            call_kwargs = mock_run.call_args[1]
            env = call_kwargs.get("env") or mock_run.call_args[1].get("env", {})
            assert env["ANIMAWORKS_ANIMA_DIR"] == anima_dir

    async def test_dispatch_fn_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        """Non-zero exit code from subprocess raises RuntimeError."""
        task_desc = {
            "task_id": "err_test_001",
            "tool_name": "image_gen",
            "subcommand": "3d",
            "raw_args": ["3d", "test.png"],
            "anima_dir": str(tmp_path),
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Something went wrong"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Tool image_gen failed"):
                dispatch_fn("image_gen", {
                    "subcommand": "3d",
                    "raw_args": ["3d", "test.png"],
                    "anima_dir": str(tmp_path),
                })

    async def test_dispatch_fn_returns_stdout(self, tmp_path: Path) -> None:
        """_dispatch_fn returns stripped stdout on success."""
        task_desc = {
            "task_id": "ret_test_001",
            "tool_name": "web_search",
            "subcommand": "search",
            "raw_args": ["search", "python asyncio"],
            "anima_dir": str(tmp_path),
        }

        dispatch_fn, _ = await self._capture_dispatch_fn(tmp_path, task_desc)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  search results here  \n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = dispatch_fn("web_search", {
                "subcommand": "search",
                "raw_args": ["search", "python asyncio"],
                "anima_dir": str(tmp_path),
            })
            assert result == "search results here"
