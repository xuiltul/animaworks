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
