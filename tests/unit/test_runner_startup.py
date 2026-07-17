"""Unit tests for AnimaRunner startup order and ping readiness."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.ipc import IPCClient, IPCRequest


@pytest.fixture(autouse=True)
def _isolate_process_global_tool_executors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep complete runner lifecycles from leaking shutdown across tests.

    A real worker exits after ``AnimaRunner._cleanup`` retires the global Mode
    A pools. This module runs multiple worker lifecycles inside one pytest
    process, while the shutdown contract itself has a dedicated unit test.
    """

    monkeypatch.setattr(
        "core.execution._litellm_tools.shutdown_tool_executors",
        MagicMock(),
    )

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

    def test_runner_does_not_require_startup_ack_without_parent_env(self, tmp_path, monkeypatch):
        """Existing parent processes remain compatible until the server restarts."""
        monkeypatch.delenv("ANIMAWORKS_EXPECT_STARTUP_ACK", raising=False)

        runner = self._make_runner(tmp_path)

        assert runner._expects_startup_ack is False

    def test_runner_requires_startup_ack_with_parent_env(self, tmp_path, monkeypatch):
        """New parent processes opt into the startup ack gate explicitly."""
        monkeypatch.setenv("ANIMAWORKS_EXPECT_STARTUP_ACK", "1")

        runner = self._make_runner(tmp_path)

        assert runner._expects_startup_ack is True

    @pytest.mark.asyncio
    async def test_startup_ack_handler_allows_autonomous_start(self, tmp_path):
        """startup_ack should release the runner startup gate."""
        runner = self._make_runner(tmp_path)

        wait_task = asyncio.create_task(runner._wait_for_startup_ack())
        await asyncio.sleep(0)
        assert not wait_task.done()

        result = await runner._handle_startup_ack({})

        await asyncio.wait_for(wait_task, timeout=1.0)
        assert result == {"status": "acknowledged"}

    @pytest.mark.asyncio
    async def test_run_starts_ipc_before_anima_init(self, tmp_path):
        """IPC server should start before DigitalAnima is constructed."""
        runner = self._make_runner(tmp_path)
        runner._expects_startup_ack = True
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

            await asyncio.wait_for(asyncio.gather(task, shutdown_task), timeout=5.0)

        # IPC should start BEFORE anima initialization
        assert call_order == ["ipc_start", "anima_init"]

    @pytest.mark.asyncio
    async def test_run_gates_autonomous_services_until_startup_ack(self, tmp_path):
        """Scheduler and watcher tasks must not start until parent ack arrives."""
        runner = self._make_runner(tmp_path)
        runner._expects_startup_ack = True

        mock_ipc_server = AsyncMock()
        mock_ipc_server.start = AsyncMock()
        mock_ipc_server.stop = AsyncMock()

        mock_anima = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.setup = MagicMock()
        mock_scheduler.shutdown = MagicMock()

        async def long_running_loop():
            await asyncio.Event().wait()

        mock_inbox_limiter = MagicMock()
        mock_inbox_limiter.inbox_watcher_loop = long_running_loop
        mock_inbox_limiter.cancel_deferred_timer = MagicMock()

        mock_pending_executor = MagicMock()
        mock_pending_executor.wake = MagicMock()
        mock_pending_executor.watcher_loop = long_running_loop

        with (
            patch("core.supervisor.runner.IPCServer", return_value=mock_ipc_server),
            patch("core.supervisor.runner.DigitalAnima", return_value=mock_anima),
            patch("core.supervisor.runner.SchedulerManager", return_value=mock_scheduler),
            patch("core.supervisor.runner.InboxRateLimiter", return_value=mock_inbox_limiter),
            patch("core.supervisor.runner.PendingTaskExecutor", return_value=mock_pending_executor),
            patch.object(runner, "_recover_streaming_journal"),
            patch.object(runner, "_startup_idle_compress", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(runner.run())

            for _ in range(50):
                if runner._ready_event.is_set():
                    break
                await asyncio.sleep(0.01)

            assert runner._ready_event.is_set()
            await asyncio.sleep(0.05)
            mock_scheduler.setup.assert_not_called()
            assert runner.inbox_watcher_task is None
            assert runner.pending_task_watcher_task is None

            await runner._handle_startup_ack({})
            for _ in range(50):
                if mock_scheduler.setup.called:
                    break
                await asyncio.sleep(0.01)

            mock_scheduler.setup.assert_called_once()
            assert runner.inbox_watcher_task is not None
            assert runner.pending_task_watcher_task is not None

            runner.shutdown_event.set()
            await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_ipc_ping_stays_responsive_during_slow_startup_inbox(self, tmp_path):
        """Immediate inbox memory/RAG work must not block IPC after startup ack."""
        from core._anima_inbox import _append_episode_off_loop

        runner = self._make_runner(tmp_path)
        runner._expects_startup_ack = True
        inbox_started = asyncio.Event()
        inbox_finished = asyncio.Event()

        class SlowMemory:
            def append_episode(self, episode: str, *, origin: str) -> None:
                time.sleep(0.3)

        async def slow_process_inbox_message():
            inbox_started.set()
            await _append_episode_off_loop(SlowMemory(), "startup inbox", origin="anima")
            inbox_finished.set()

        mock_anima = MagicMock()
        mock_anima._conversation_locks = {}
        mock_anima._background_lock = asyncio.Lock()
        mock_anima._inbox_lock = asyncio.Lock()
        mock_anima._last_progress_at = None
        mock_anima._busy_since = None
        mock_anima._active_parallel_tasks = {}
        mock_anima._set_pending_executor_wake = MagicMock()
        mock_anima._set_active_parallel_tasks_getter = MagicMock()
        mock_anima.set_on_lock_released = MagicMock()
        mock_anima.set_on_message_sent = MagicMock()
        mock_anima.process_inbox_message = slow_process_inbox_message

        mock_scheduler = MagicMock()
        mock_scheduler.setup = MagicMock()
        mock_scheduler.shutdown = MagicMock()

        async def inbox_watcher_loop():
            await mock_anima.process_inbox_message()
            await asyncio.Event().wait()

        mock_inbox_limiter = MagicMock()
        mock_inbox_limiter.inbox_watcher_loop = inbox_watcher_loop
        mock_inbox_limiter.cancel_deferred_timer = MagicMock()

        async def pending_watcher_loop():
            await asyncio.Event().wait()

        mock_pending_executor = MagicMock()
        mock_pending_executor.wake = MagicMock()
        mock_pending_executor.watcher_loop = pending_watcher_loop

        with (
            patch("core.supervisor.runner.DigitalAnima", return_value=mock_anima),
            patch("core.supervisor.runner.SchedulerManager", return_value=mock_scheduler),
            patch("core.supervisor.runner.InboxRateLimiter", return_value=mock_inbox_limiter),
            patch("core.supervisor.runner.PendingTaskExecutor", return_value=mock_pending_executor),
            patch.object(runner, "_recover_streaming_journal"),
            patch.object(runner, "_startup_idle_compress", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(runner.run())
            client = IPCClient(runner.socket_path)

            try:
                for _ in range(50):
                    if runner.socket_path.exists():
                        break
                    await asyncio.sleep(0.01)
                assert runner.socket_path.exists()

                await asyncio.wait_for(client.connect(timeout=1.0), timeout=2.0)

                ready = await client.send_request(
                    IPCRequest(id="ready_ping", method="ping", params={}),
                    timeout=1.0,
                )
                assert ready.result and ready.result["status"] == "ok"
                assert not inbox_started.is_set()

                ack = await client.send_request(
                    IPCRequest(id="startup_ack", method="startup_ack", params={}),
                    timeout=1.0,
                )
                assert ack.result == {"status": "acknowledged"}

                await asyncio.wait_for(inbox_started.wait(), timeout=1.0)
                assert not inbox_finished.is_set()

                ping_during_inbox = await client.send_request(
                    IPCRequest(id="busy_ping", method="ping", params={}),
                    timeout=0.2,
                )

                assert ping_during_inbox.result and ping_during_inbox.result["status"] == "ok"
                await asyncio.wait_for(inbox_finished.wait(), timeout=1.0)
            finally:
                await client.close()
                runner.shutdown_event.set()
                await asyncio.wait_for(task, timeout=5.0)


# ── _conversation_contains_recovery (F9) ─────────────────────


class TestConversationContainsRecovery:
    """Crash-recovery dedup must be marker-independent (F9).

    A response that streamed to completion and was saved cleanly (no
    interruption marker) just before the journal was deleted must be detected
    as already-present so recovery does not append a duplicate turn.
    """

    @staticmethod
    def _conv(turns):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        conv = MagicMock()
        conv.load.return_value = SimpleNamespace(
            turns=[SimpleNamespace(role=r, content=c) for r, c in turns]
        )
        return conv

    def test_clean_saved_turn_without_marker_is_deduped(self):
        from core.i18n import t
        from core.supervisor.runner import AnimaRunner

        recovered = "完全に生成された応答"
        saved_text = recovered + "\n" + t("anima.response_interrupted")
        # Clean save: the assistant turn holds the full text, no marker.
        conv = self._conv([("human", "質問"), ("assistant", recovered)])

        assert AnimaRunner._conversation_contains_recovery(conv, recovered, saved_text) is True

    def test_clean_saved_longer_final_is_deduped(self):
        from core.i18n import t
        from core.supervisor.runner import AnimaRunner

        recovered = "部分的な応答"
        final = recovered + "、そして続きの完了テキスト"
        saved_text = recovered + "\n" + t("anima.response_interrupted")
        conv = self._conv([("assistant", final)])

        assert AnimaRunner._conversation_contains_recovery(conv, recovered, saved_text) is True

    def test_marked_turn_is_deduped(self):
        from core.i18n import t
        from core.supervisor.runner import AnimaRunner

        recovered = "中断された応答"
        saved_text = recovered + "\n" + t("anima.response_interrupted")
        conv = self._conv([("assistant", saved_text)])

        assert AnimaRunner._conversation_contains_recovery(conv, recovered, saved_text) is True

    def test_unrelated_turns_not_deduped(self):
        from core.i18n import t
        from core.supervisor.runner import AnimaRunner

        recovered = "回復すべき固有の応答テキスト"
        saved_text = recovered + "\n" + t("anima.response_interrupted")
        conv = self._conv([("human", "別の話題"), ("assistant", "無関係な過去の応答")])

        assert AnimaRunner._conversation_contains_recovery(conv, recovered, saved_text) is False
