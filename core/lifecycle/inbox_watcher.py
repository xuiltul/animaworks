from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import logging
import time

from core.schemas import EXTERNAL_PLATFORM_SOURCES

logger = logging.getLogger("animaworks.lifecycle")


class InboxWatcherMixin:
    """Mixin providing inbox polling and message-triggered heartbeat logic."""

    async def _inbox_watcher_loop(self) -> None:
        """Poll inbox dirs every 2s; trigger heartbeat on new messages."""
        logger.info("Inbox watcher started (poll interval: 2s)")
        while True:
            await asyncio.sleep(2)
            for name, anima in self.animas.items():
                if name in self._pending_triggers:
                    continue
                if not anima.messenger.has_unread():
                    continue
                if self._is_in_cooldown(name):
                    self._schedule_deferred_trigger(name)
                    continue
                if anima._inbox_lock.locked():
                    self._schedule_deferred_trigger(name)
                    continue
                if anima._background_lock.locked():
                    self._schedule_deferred_trigger(name)
                    continue
                self._pending_triggers.add(name)
                asyncio.create_task(self._message_triggered_heartbeat(name))

    async def _on_anima_lock_released(self, name: str) -> None:
        """Check deferred inbox after an anima's lock is released.

        If unread messages exist, schedule a deferred trigger to ensure
        they are processed even when cooldown is still active.
        """
        anima = self.animas.get(name)
        if not anima:
            return
        if not anima.messenger.has_unread():
            return
        if name in self._pending_triggers:
            return
        # Instead of giving up when in cooldown, schedule deferred trigger
        self._schedule_deferred_trigger(name)

    def _schedule_deferred_trigger(self, name: str) -> None:
        """Schedule a deferred heartbeat trigger after cooldown expires.

        Only one timer per anima is maintained.  If a timer is already
        pending, the call is a no-op (the existing timer will fire and
        re-check the inbox).
        """
        if name in self._deferred_timers:
            return  # already scheduled
        last = self._last_msg_heartbeat_end.get(name, 0.0)
        remaining = self._cooldown_s - (time.monotonic() - last)
        # If not in cooldown (e.g. lock-only), use a short retry delay
        delay = max(remaining, 2.0)
        loop = asyncio.get_running_loop()
        self._deferred_timers[name] = loop.call_later(
            delay,
            lambda n=name: asyncio.create_task(self._try_deferred_trigger(n)),
        )
        logger.debug(
            "Deferred trigger scheduled for %s in %.1fs",
            name,
            delay,
        )

    async def _try_deferred_trigger(self, name: str) -> None:
        """Attempt to trigger a deferred heartbeat.

        Re-schedules itself if the anima is still blocked by cooldown
        or lock, ensuring messages are never forgotten.
        """
        self._deferred_timers.pop(name, None)
        anima = self.animas.get(name)
        if not anima:
            return
        if not anima.messenger.has_unread():
            return
        if name in self._pending_triggers:
            return
        if self._is_in_cooldown(name):
            self._schedule_deferred_trigger(name)
            return
        if anima._inbox_lock.locked():
            self._schedule_deferred_trigger(name)
            return
        if anima._background_lock.locked():
            self._schedule_deferred_trigger(name)
            return
        self._pending_triggers.add(name)
        asyncio.create_task(self._message_triggered_heartbeat(name))

    async def _message_triggered_heartbeat(self, name: str) -> None:
        anima = self.animas.get(name)
        if not anima:
            self._pending_triggers.discard(name)
            return

        # Peek at inbox senders for cascade detection and rate limiting
        inbox_messages = anima.messenger.receive()
        senders = {m.from_person for m in inbox_messages}

        # ── Intent-based trigger filtering ──
        # Only trigger immediate heartbeat for actionable messages or human messages.
        # Non-actionable messages (ack, thanks, FYI) wait for the scheduled heartbeat.
        has_human = any(m.source == "human" for m in inbox_messages)
        has_external_directed = any(m.source in EXTERNAL_PLATFORM_SOURCES and m.intent for m in inbox_messages)
        has_actionable = any(m.intent in self._actionable_intents for m in inbox_messages)
        if not has_human and not has_external_directed and not has_actionable:
            logger.info(
                "Intent filter: %s — no actionable messages, deferring to scheduled heartbeat",
                name,
            )
            self._pending_triggers.discard(name)
            return

        if senders and self._check_cascade(name, senders):
            self._pending_triggers.discard(name)
            return

        # ── Per-sender rate limiting (Phase 4) ──
        # If a single sender has 5+ pending messages, skip message-triggered
        # heartbeat and let the next scheduled heartbeat handle them with dedup.
        try:
            sender_counts: dict[str, int] = {}
            for m in inbox_messages:
                sender_counts[m.from_person] = sender_counts.get(m.from_person, 0) + 1
            for sender, count in sender_counts.items():
                if count >= 5:
                    logger.info(
                        "Per-sender rate limit: %s has %d messages for %s, deferring to scheduled heartbeat",
                        sender,
                        count,
                        name,
                    )
                    self._pending_triggers.discard(name)
                    return
        except Exception:
            logger.debug("Per-sender rate limit check failed for %s", name, exc_info=True)

        try:
            logger.info("Message-triggered inbox: %s", name)
            result = await anima.process_inbox_message()
            if self._ws_broadcast:
                await self._ws_broadcast(
                    {
                        "type": "anima.message_heartbeat",
                        "data": {"name": name, "result": result.model_dump()},
                    }
                )
        except Exception:
            logger.exception("Message-triggered heartbeat failed: %s", name)
        finally:
            self._pending_triggers.discard(name)
            self._last_msg_heartbeat_end[name] = time.monotonic()
            if senders:
                self._record_pair_heartbeat(name, senders)
