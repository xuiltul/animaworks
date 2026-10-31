# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the inbox lane (task-runner based inbox processing).

Covers the transition of inbox LLM execution from the root process to an
isolated task runner child (lane ``inbox``): the root only decides whether
to launch and inspects the isolated result.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.runtime.inbox_rate_limiter import InboxRateLimiter
from core.runtime.scheduler_manager import SchedulerManager
from core.schemas import Message


def _default_config():
    from core.config.models import AnimaWorksConfig

    return AnimaWorksConfig()


def _make_message(
    *,
    intent: str = "",
    source: str = "anima",
    from_person: str = "bob",
    content: str = "hello",
) -> Message:
    return Message(
        from_person=from_person,
        to_person="alice",
        content=content,
        intent=intent,
        source=source,
    )


def _make_limiter(messages: list[Message], *, anima_name: str = "alice") -> InboxRateLimiter:
    """Create an InboxRateLimiter whose inbox runs in an isolated task runner."""
    mock_anima = MagicMock()
    # Nonexistent status.json → _read_anima_enabled defaults to True.
    mock_anima.anima_dir = Path("/nonexistent") / anima_name
    mock_anima.messenger = MagicMock()
    mock_anima.messenger.receive.return_value = messages

    supervisor = MagicMock()
    supervisor.run_inbox = AsyncMock(
        return_value={
            "task_type": "inbox",
            "result": {"action": "responded", "reason": "", "summary": "ok"},
            "success": True,
        }
    )

    mock_scheduler_mgr = MagicMock(spec=SchedulerManager)
    mock_scheduler_mgr.heartbeat_running = False
    mock_scheduler_mgr._task_runner_supervisor = supervisor

    limiter = InboxRateLimiter(
        anima=mock_anima,
        anima_name=anima_name,
        shutdown_event=asyncio.Event(),
        scheduler_mgr=mock_scheduler_mgr,
    )
    limiter._pending_trigger = True  # mimic the real trigger path
    return limiter


class TestRunInboxCall:
    """message_triggered_inbox must launch the isolated run_inbox, not call the anima."""

    @pytest.mark.asyncio
    async def test_calls_supervisor_run_inbox_once(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()

        limiter._scheduler_mgr._task_runner_supervisor.run_inbox.assert_awaited_once()
        limiter._anima.process_inbox_message.assert_not_called()
        assert limiter._pending_trigger is False
        assert limiter._failure_retry_until == 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_retry_until(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        # An expired retry window (in the past on any host's monotonic clock).
        limiter._failure_retry_until = time.monotonic() - 1.0
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until == 0


class TestRecordProcessingFailure:
    """Failure signals must be recorded via _record_processing_failure."""

    @pytest.mark.asyncio
    async def test_success_false_records_failure(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(
            return_value={"task_type": "inbox", "result": {"action": "interrupted", "reason": "x"}, "success": False}
        )
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until != 0

    @pytest.mark.asyncio
    async def test_action_error_records_failure(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(
            return_value={"task_type": "inbox", "result": {"action": "error", "reason": "network"}, "success": False}
        )
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until != 0

    @pytest.mark.asyncio
    async def test_result_reason_records_failure(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(
            return_value={"task_type": "inbox", "result": {"action": "responded", "reason": "budget"}, "success": False}
        )
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until != 0

    @pytest.mark.asyncio
    async def test_exception_records_failure(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until != 0


class TestNotRunConditions:
    """run_inbox must wait while a scheduled heartbeat is active."""

    @pytest.mark.asyncio
    async def test_heartbeat_running_defers(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr.heartbeat_running = True
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox.assert_not_called()
        assert limiter._pending_trigger is False
        assert limiter._deferred_timer is not None
        limiter.cancel_deferred_timer()


class TestHeartbeatRunningFlag:
    """heartbeat_running must be True during execution, False after (even on error)."""

    @pytest.mark.asyncio
    async def test_flag_true_during_false_after(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        observed: list[bool] = []

        async def side_effect() -> dict:
            observed.append(limiter._scheduler_mgr.heartbeat_running)
            return {"task_type": "inbox", "result": {"action": "responded", "reason": ""}, "success": True}

        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(side_effect=side_effect)
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert observed == [True]
        assert limiter._scheduler_mgr.heartbeat_running is False

    @pytest.mark.asyncio
    async def test_flag_reset_on_exception(self) -> None:
        limiter = _make_limiter([_make_message(intent="question")])
        limiter._scheduler_mgr._task_runner_supervisor.run_inbox = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("core.runtime.inbox_rate_limiter.load_config", return_value=_default_config()):
            await limiter.message_triggered_inbox()
        assert limiter._scheduler_mgr.heartbeat_running is False


class TestPrepareExecutionInbox:
    """task_runner._prepare_execution must route lane=inbox to execute_inbox_contract."""

    @staticmethod
    def _identity():
        from core.runtime.ipc_v2 import IPCV2Identity

        return IPCV2Identity(
            job_id="job-inbox",
            root_epoch=str(uuid.uuid4()),
            attempt=1,
            lane="inbox",
            display_lane="inbox",
        )

    @pytest.mark.asyncio
    async def test_returns_inbox_contract_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.runtime import task_runner

        identity = self._identity()
        args = argparse.Namespace(anima="sakura", lane="inbox", job="job-inbox")
        monkeypatch.setattr(task_runner, "DigitalAnima", lambda **kwargs: object())
        mock_contract = AsyncMock(return_value={"task_type": "inbox", "result": {}, "success": True})
        monkeypatch.setattr(task_runner, "execute_inbox_contract", mock_contract)

        execution = await task_runner._prepare_execution(args, identity, {}, control={})
        result = await execution

        mock_contract.assert_awaited_once()
        assert result["task_type"] == "inbox"


class TestIPCAndRecovery:
    """The inbox lane must pass IPC validation and be recovered on exit."""

    def test_ipc_identity_accepts_inbox(self) -> None:
        from core.runtime.ipc_v2 import IPCV2Identity

        identity = IPCV2Identity(
            job_id="job-inbox",
            root_epoch=str(uuid.uuid4()),
            attempt=1,
            lane="inbox",
            display_lane="inbox",
        )
        identity.validate()  # must not raise

    def test_recover_task_journals_defaults_include_inbox(self) -> None:
        from core.runtime.task_runner_supervisor import TaskRunnerSupervisor

        sig = inspect.signature(TaskRunnerSupervisor._recover_task_journals)
        default = sig.parameters["session_types"].default
        assert "inbox" in default
