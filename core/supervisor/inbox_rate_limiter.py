# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Inbox rate limiting, cascade detection, and deferred trigger management.

Monitors the Anima inbox for new messages and triggers heartbeats with
rate limiting to prevent cascade loops between Animas.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from core.config.models import load_config

if TYPE_CHECKING:
    from core.anima import DigitalAnima
    from core.supervisor.scheduler_manager import SchedulerManager

logger = logging.getLogger(__name__)

class InboxRateLimiter:
    """Inbox rate limiting, cascade detection, and deferred trigger management."""

    def __init__(
        self,
        anima: DigitalAnima,
        anima_name: str,
        shutdown_event: asyncio.Event,
        scheduler_mgr: SchedulerManager,
        cooldown_sec: float | None = None,
    ) -> None:
        cfg = load_config()
        self._anima = anima
        self._anima_name = anima_name
        self._shutdown_event = shutdown_event
        self._scheduler_mgr = scheduler_mgr
        self._cooldown_sec = cooldown_sec if cooldown_sec is not None else cfg.heartbeat.msg_heartbeat_cooldown_s
        self._cascade_window_s = cfg.heartbeat.cascade_window_s
        self._cascade_threshold = cfg.heartbeat.cascade_threshold

        self._pending_trigger: bool = False
        self._deferred_timer: asyncio.Handle | None = None
        self._last_msg_heartbeat_end: float = 0.0
        self._pair_heartbeat_times: dict[tuple[str, str], list[float]] = {}

    # ── Cooldown ─────────────────────────────────────────────────

    def is_in_cooldown(self) -> bool:
        """Return True if a message-triggered heartbeat finished too recently."""
        return (time.monotonic() - self._last_msg_heartbeat_end) < self._cooldown_sec

    # ── Cascade Detection ────────────────────────────────────────

    def check_cascade(self, senders: set[str]) -> bool:
        """Return True if any (anima, sender) pair exceeds cascade threshold."""
        now = time.monotonic()
        for sender in senders:
            keys = [(self._anima_name, sender), (sender, self._anima_name)]
            total = 0
            for k in keys:
                times = self._pair_heartbeat_times.get(k, [])
                # Evict expired entries
                times = [t for t in times if now - t < self._cascade_window_s]
                self._pair_heartbeat_times[k] = times
                if not times and k in self._pair_heartbeat_times:
                    del self._pair_heartbeat_times[k]
                total += len(times)
            if total >= self._cascade_threshold:
                logger.warning(
                    "CASCADE DETECTED: %s <-> %s (%d round-trips in %ds window). "
                    "Suppressing message-triggered heartbeat.",
                    self._anima_name, sender, total, self._cascade_window_s,
                )
                return True
        return False

    def record_pair_heartbeat(self, senders: set[str]) -> None:
        """Record a heartbeat exchange for cascade tracking."""
        now = time.monotonic()
        for sender in senders:
            key = (self._anima_name, sender)
            self._pair_heartbeat_times.setdefault(key, []).append(now)

    # ── Lock-Released Callback ───────────────────────────────────

    async def on_anima_lock_released(self) -> None:
        """Check deferred inbox after the anima's lock is released.

        If unread messages exist, schedule a deferred trigger to ensure
        they are processed even when cooldown is still active.
        """
        if not self._anima:
            return
        if not self._anima.messenger.has_unread():
            return
        if self._pending_trigger:
            return
        # Instead of giving up when in cooldown, schedule deferred trigger
        self.schedule_deferred_trigger()

    # ── Deferred Trigger ─────────────────────────────────────────

    def schedule_deferred_trigger(self) -> None:
        """Schedule a deferred heartbeat trigger after cooldown expires.

        Only one timer is maintained.  If a timer is already pending,
        the call is a no-op (the existing timer will fire and re-check
        the inbox).
        """
        if self._deferred_timer is not None:
            return  # already scheduled
        remaining = self._cooldown_sec - (
            time.monotonic() - self._last_msg_heartbeat_end
        )
        # If not in cooldown (e.g. lock-only), use a short retry delay
        delay = max(remaining, 2.0)
        loop = asyncio.get_running_loop()
        self._deferred_timer = loop.call_later(
            delay,
            lambda: asyncio.create_task(self.try_deferred_trigger()),
        )
        logger.debug(
            "Deferred trigger scheduled for %s in %.1fs",
            self._anima_name, delay,
        )

    async def try_deferred_trigger(self) -> None:
        """Attempt to trigger a deferred heartbeat.

        Re-schedules itself if the anima is still blocked by cooldown
        or lock, ensuring messages are never forgotten.
        """
        self._deferred_timer = None
        if not self._anima:
            return
        if not self._anima.messenger.has_unread():
            return
        if self._pending_trigger:
            return
        if self.is_in_cooldown():
            self.schedule_deferred_trigger()
            return
        if self._anima._lock.locked():
            self.schedule_deferred_trigger()
            return
        self._pending_trigger = True
        asyncio.create_task(self.message_triggered_heartbeat())

    # ── Message-Triggered Heartbeat ──────────────────────────────

    async def message_triggered_heartbeat(self) -> None:
        """Execute a heartbeat triggered by incoming messages."""
        if not self._anima:
            self._pending_trigger = False
            return

        if self._scheduler_mgr.heartbeat_running:
            logger.info("Message-triggered heartbeat SKIPPED (already running): %s", self._anima_name)
            self._pending_trigger = False
            return

        # Peek at inbox senders for cascade detection and intent filtering
        inbox_messages = self._anima.messenger.receive()
        senders = {m.from_person for m in inbox_messages}

        # ── Intent-based trigger filtering ──
        # Only trigger immediate heartbeat for actionable messages or human messages.
        # Non-actionable messages (ack, thanks, FYI) wait for the scheduled heartbeat.
        cfg = load_config()
        has_human = any(m.source == "human" for m in inbox_messages)
        has_actionable = any(
            m.intent in cfg.heartbeat.actionable_intents for m in inbox_messages
        )
        if not has_human and not has_actionable:
            logger.info(
                "Intent filter: %s — no actionable messages, deferring to scheduled heartbeat",
                self._anima_name,
            )
            self._pending_trigger = False
            return

        if senders and self.check_cascade(senders):
            self._pending_trigger = False
            return

        self._scheduler_mgr.heartbeat_running = True
        try:
            logger.info("Message-triggered heartbeat: %s", self._anima_name)
            await self._anima.run_heartbeat()
        except Exception:
            logger.exception(
                "Message-triggered heartbeat failed: %s", self._anima_name,
            )
        finally:
            self._scheduler_mgr.heartbeat_running = False
            self._pending_trigger = False
            self._last_msg_heartbeat_end = time.monotonic()
            if senders:
                self.record_pair_heartbeat(senders)

    # ── Inbox Watcher Loop ───────────────────────────────────────

    async def inbox_watcher_loop(self) -> None:
        """Poll inbox every 2s; trigger heartbeat on new messages.

        Applies rate limiting to prevent cascade loops between animas
        and cooldown to avoid excessive heartbeat triggers.
        """
        if not self._anima:
            return

        logger.info("Inbox watcher started for %s", self._anima_name)

        while not self._shutdown_event.is_set():
            try:
                if self._pending_trigger:
                    await asyncio.sleep(2.0)
                    continue
                if not self._anima.messenger.has_unread():
                    await asyncio.sleep(2.0)
                    continue
                if self.is_in_cooldown():
                    self.schedule_deferred_trigger()
                    await asyncio.sleep(2.0)
                    continue
                if self._anima._lock.locked():
                    self.schedule_deferred_trigger()
                    await asyncio.sleep(2.0)
                    continue

                self._pending_trigger = True
                asyncio.create_task(self.message_triggered_heartbeat())
                await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in inbox watcher for %s: %s", self._anima_name, e,
                )
                await asyncio.sleep(2.0)

        logger.info("Inbox watcher stopped for %s", self._anima_name)

    # ── Cleanup ──────────────────────────────────────────────────

    def cancel_deferred_timer(self) -> None:
        """Cancel deferred trigger timer if active."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None
