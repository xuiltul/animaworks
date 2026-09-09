# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for inbox watcher enabled guard (runner-local path)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.inbox_rate_limiter import InboxRateLimiter


def _make_limiter(anima_dir: Path, *, name: str = "alice") -> InboxRateLimiter:
    anima = MagicMock()
    anima.anima_dir = anima_dir
    anima.messenger = MagicMock()
    anima._inbox_lock = MagicMock()
    anima._inbox_lock.locked.return_value = False
    anima._background_lock = MagicMock()
    anima._background_lock.locked.return_value = False
    anima.process_inbox_message = AsyncMock()

    scheduler_mgr = MagicMock()
    scheduler_mgr.heartbeat_running = False
    shutdown = asyncio.Event()

    with patch("core.supervisor.inbox_rate_limiter.load_config") as mock_cfg:
        cfg = MagicMock()
        cfg.heartbeat.msg_heartbeat_cooldown_s = 0.0
        cfg.heartbeat.cascade_window_s = 60.0
        cfg.heartbeat.cascade_threshold = 5
        mock_cfg.return_value = cfg
        limiter = InboxRateLimiter(
            anima=anima,
            anima_name=name,
            shutdown_event=shutdown,
            scheduler_mgr=scheduler_mgr,
            cooldown_sec=0.0,
        )
    return limiter


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_external_inbox_keeps_unread_and_cannot_bypass_retry_delay(tmp_path, raises):
    from core.schemas import CycleResult, Message

    limiter = _make_limiter(tmp_path)
    limiter._anima.messenger.receive.return_value = [
        Message(from_person="human", to_person="alice", content="request", source="slack", intent="question")
    ]
    limiter._anima.messenger.has_unread.return_value = True
    if raises:
        limiter._anima.process_inbox_message.side_effect = ConnectionError("Connection refused")
    else:
        limiter._anima.process_inbox_message.return_value = CycleResult(
            trigger="inbox", action="error", reason="network", summary="API Error: ConnectionRefused"
        )
    with patch("core.supervisor.inbox_rate_limiter.time.monotonic", return_value=100.0):
        await limiter.message_triggered_inbox()
        assert limiter._failure_retry_until >= 130.0
        await limiter.message_triggered_inbox()
        assert limiter._anima.process_inbox_message.await_count == 1
        assert limiter._deferred_timer is not None
        limiter._deferred_timer.cancel()
        limiter._deferred_timer = None
        await limiter.try_deferred_trigger()
        assert limiter._anima.process_inbox_message.await_count == 1
        limiter._deferred_timer.cancel()
        limiter._deferred_timer = None
    limiter._anima.messenger.archive_paths.assert_not_called()
    limiter._anima.process_inbox_message.side_effect = None
    limiter._anima.process_inbox_message.return_value = CycleResult(trigger="inbox", action="responded", summary="ok")
    with patch("core.supervisor.inbox_rate_limiter.time.monotonic", return_value=131.0):
        await limiter.message_triggered_inbox()
        assert limiter._anima.process_inbox_message.await_count == 2
        assert limiter._failure_retry_until == 0
        # A healthy external inbox regains immediate handling after recovery.
        await limiter.message_triggered_inbox()
        assert limiter._anima.process_inbox_message.await_count == 3


def test_inbox_failure_waits_for_all_provider_guards_to_expire(tmp_path):
    from core.schemas import ModelConfig

    limiter = _make_limiter(tmp_path)
    config = ModelConfig(model="claude-sonnet-4-6", fallback_models=["c:codex/gpt-5.6-luna"])
    limiter._anima.agent.model_config = config
    with (
        patch("core.config.model_config.resolve_effective_model_config", return_value=config),
        patch("core.config.model_config._guard_key_for_model_config", return_value="test:blocked"),
        patch("core.execution.rate_guard.get_rate_guard") as guard,
        patch("core.supervisor.inbox_rate_limiter.time.monotonic", return_value=100.0),
    ):
        guard.return_value.blocked_remaining.return_value = 1800.0
        limiter._record_processing_failure()
    assert limiter._failure_retry_until == 1900.0


class TestInboxWatcherEnabledGuard:
    @pytest.mark.asyncio
    async def test_disabled_skips_processing_and_keeps_inbox(
        self,
        tmp_path: Path,
    ) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": False}),
            encoding="utf-8",
        )

        limiter = _make_limiter(anima_dir)
        limiter._anima.messenger.has_unread.return_value = True

        async def _run_briefly():
            task = asyncio.create_task(limiter.inbox_watcher_loop())
            await asyncio.sleep(0.3)
            limiter._shutdown_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_briefly()

        limiter._anima.process_inbox_message.assert_not_awaited()
        # has_unread was consulted (unread path) but processing never started
        assert limiter._pending_trigger is False

    @pytest.mark.asyncio
    async def test_enabled_triggers_processing(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": True}),
            encoding="utf-8",
        )

        limiter = _make_limiter(anima_dir)
        limiter._anima.messenger.has_unread.return_value = True

        # message_triggered_inbox is scheduled as a task; stub it to observe
        triggered = asyncio.Event()

        async def _fake_triggered():
            triggered.set()
            limiter._pending_trigger = False

        with patch.object(
            limiter,
            "message_triggered_inbox",
            side_effect=_fake_triggered,
        ):
            task = asyncio.create_task(limiter.inbox_watcher_loop())
            await asyncio.wait_for(triggered.wait(), timeout=2.0)
            limiter._shutdown_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert triggered.is_set()

    @pytest.mark.asyncio
    async def test_disabled_then_enabled_starts_processing(
        self,
        tmp_path: Path,
    ) -> None:
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        status_path = anima_dir / "status.json"
        status_path.write_text(json.dumps({"enabled": False}), encoding="utf-8")

        limiter = _make_limiter(anima_dir)
        limiter._anima.messenger.has_unread.return_value = True

        triggered = asyncio.Event()

        async def _fake_triggered():
            triggered.set()
            limiter._pending_trigger = False

        with patch.object(
            limiter,
            "message_triggered_inbox",
            side_effect=_fake_triggered,
        ):
            task = asyncio.create_task(limiter.inbox_watcher_loop())
            # While disabled, processing must not fire
            await asyncio.sleep(0.25)
            assert not triggered.is_set()

            # Re-enable → next poll should process
            status_path.write_text(
                json.dumps({"enabled": True}),
                encoding="utf-8",
            )
            await asyncio.wait_for(triggered.wait(), timeout=3.0)
            limiter._shutdown_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert triggered.is_set()


class TestDeferredTriggerEnabledGuard:
    """deferred timer → try_deferred_trigger → message_triggered_inbox must respect enabled."""

    @pytest.mark.asyncio
    async def test_deferred_path_skips_when_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        """message_triggered_inbox entry (deferred destination) bails when disabled."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": False}),
            encoding="utf-8",
        )

        limiter = _make_limiter(anima_dir)
        limiter._anima.messenger.has_unread.return_value = True
        msg = MagicMock()
        msg.source = "human"
        msg.intent = "request"
        msg.from_person = "bob"
        limiter._anima.messenger.receive.return_value = [msg]
        limiter._pending_trigger = True

        with patch("core.supervisor.inbox_rate_limiter.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.heartbeat.actionable_intents = ["request"]
            mock_cfg.return_value = cfg
            await limiter.message_triggered_inbox()

        limiter._anima.process_inbox_message.assert_not_awaited()
        assert limiter._pending_trigger is False

    @pytest.mark.asyncio
    async def test_try_deferred_trigger_disabled_does_not_process(
        self,
        tmp_path: Path,
    ) -> None:
        """Full deferred chain: try_deferred_trigger → message_triggered_inbox skips disabled."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": False}),
            encoding="utf-8",
        )

        limiter = _make_limiter(anima_dir)
        limiter._anima.messenger.has_unread.return_value = True
        msg = MagicMock()
        msg.source = "human"
        msg.intent = "request"
        msg.from_person = "bob"
        limiter._anima.messenger.receive.return_value = [msg]

        # Bypass cooldown so try_deferred_trigger proceeds to create the task
        limiter._last_msg_heartbeat_end = 0.0
        limiter._cooldown_sec = 0.0

        with patch("core.supervisor.inbox_rate_limiter.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.heartbeat.actionable_intents = ["request"]
            mock_cfg.return_value = cfg
            await limiter.try_deferred_trigger()
            # message_triggered_inbox is scheduled as a task
            await asyncio.sleep(0.1)

        limiter._anima.process_inbox_message.assert_not_awaited()
        assert limiter._pending_trigger is False


class TestReadAnimaEnabledMalformed:
    """Non-object status.json must default to enabled without raising."""

    def test_list_status_json_defaults_true(self, tmp_path: Path) -> None:
        from core.supervisor.inbox_rate_limiter import _read_anima_enabled

        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text("[]", encoding="utf-8")
        assert _read_anima_enabled(anima_dir) is True

    def test_null_status_json_defaults_true(self, tmp_path: Path) -> None:
        from core.supervisor.inbox_rate_limiter import _read_anima_enabled

        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text("null", encoding="utf-8")
        assert _read_anima_enabled(anima_dir) is True

    def test_manager_read_anima_enabled_non_dict_defaults_true(self, tmp_path: Path) -> None:
        from core.supervisor.manager import ProcessSupervisor

        anima_dir = tmp_path / "alice"
        anima_dir.mkdir()
        (anima_dir / "status.json").write_text("[1, 2]", encoding="utf-8")
        assert ProcessSupervisor.read_anima_enabled(anima_dir) is True
