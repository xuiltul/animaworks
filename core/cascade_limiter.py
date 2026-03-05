from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Conversation depth limiter for preventing bilateral message cascades.

Tracks message exchange depth between Anima pairs by reading the
unified activity log (file-based).  Survives process restarts because
the state is derived from persistent JSONL files, not in-memory
dictionaries.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.config.models import load_config
from core.time_utils import ensure_aware, now_jst

logger = logging.getLogger("animaworks.cascade_limiter")

# Kept for backward compatibility; authoritative values now live in HeartbeatConfig.
_DEPTH_WINDOW_S = 600      # 10 minutes
_MAX_DEPTH_DEFAULT = 6     # 6 turns = 3 round-trips


class ConversationDepthLimiter:
    """Track message exchange depth between Anima pairs via activity log.

    File-based: reads ``activity_log/{date}.jsonl`` to count recent
    message_sent and message_received events involving the target pair.
    Alias resolution in ActivityLogger ensures legacy ``dm_sent`` /
    ``dm_received`` entries are also matched.
    """

    def __init__(
        self,
        window_s: float | None = None,
        max_depth: int | None = None,
        max_per_hour: int | None = None,
        max_per_day: int | None = None,
    ) -> None:
        cfg = load_config()
        self._window_s = window_s if window_s is not None else cfg.heartbeat.depth_window_s
        self._max_depth = max_depth if max_depth is not None else cfg.heartbeat.max_depth
        self._max_per_hour = max_per_hour if max_per_hour is not None else cfg.heartbeat.max_messages_per_hour
        self._max_per_day = max_per_day if max_per_day is not None else cfg.heartbeat.max_messages_per_day

    def check_global_outbound(
        self,
        sender: str,
        sender_anima_dir: Path,
    ) -> bool | str:
        """Check if sender has exceeded global outbound message limit.

        Counts all dm_sent / message_sent events from sender
        in the last hour and last 24 hours.

        Returns True if allowed, or a descriptive error string if blocked.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(sender_anima_dir)
            entries = activity.recent(
                days=2,
                limit=500,
                types=["dm_sent", "message_sent"],
            )
        except Exception:
            logger.warning(
                "Failed to read activity log for global outbound check: %s",
                sender_anima_dir,
                exc_info=True,
            )
            return "GlobalOutboundLimitExceeded: アクティビティログ読み取り失敗のため送信をブロックしました"

        now = now_jst()
        hourly_cutoff = now - timedelta(hours=1)
        daily_cutoff = now - timedelta(hours=24)
        hourly_count = 0
        daily_count = 0
        earliest_hourly_ts: datetime | None = None

        for e in entries:
            try:
                ts = ensure_aware(datetime.fromisoformat(e.ts))
                if ts >= daily_cutoff:
                    daily_count += 1
                    if ts >= hourly_cutoff:
                        hourly_count += 1
                        if earliest_hourly_ts is None or ts < earliest_hourly_ts:
                            earliest_hourly_ts = ts
            except (ValueError, TypeError):
                continue

        if hourly_count >= self._max_per_hour:
            logger.warning(
                "GLOBAL HOURLY LIMIT: %s blocked (%d msgs in last hour)",
                sender,
                hourly_count,
            )
            reset_at = ""
            if earliest_hourly_ts:
                reset_time = earliest_hourly_ts + timedelta(hours=1)
                reset_at = f" 次の送信可能時刻（目安）: {reset_time.strftime('%H:%M')}"
            return (
                f"GlobalOutboundLimitExceeded: 1時間あたりの送信上限"
                f"（{self._max_per_hour}通）に到達しています"
                f"（現在{hourly_count}通/1h, {daily_count}通/24h）。"
                f"{reset_at}"
                f" このターンではsend_messageを使わず、送信内容を"
                f"current_task.mdに記録して次のセッションで送信してください。"
            )
        if daily_count >= self._max_per_day:
            logger.warning(
                "GLOBAL DAILY LIMIT: %s blocked (%d msgs in last 24h)",
                sender,
                daily_count,
            )
            return (
                f"GlobalOutboundLimitExceeded: 24時間あたりの送信上限"
                f"（{self._max_per_day}通）に到達しています"
                f"（現在{daily_count}通/24h）。"
                f" このターンではsend_messageを使わず、送信内容を"
                f"current_task.mdに記録して次のセッションで送信してください。"
            )
        return True

    def check_depth(
        self,
        sender: str,
        receiver: str,
        sender_anima_dir: Path,
    ) -> bool:
        """Check if the exchange is allowed by scanning activity_log.

        Counts message_sent and message_received entries involving the
        (sender, receiver) pair within ``depth_window_s``.

        Args:
            sender: The Anima name that wants to send.
            receiver: The target Anima name.
            sender_anima_dir: Path to the sender's anima directory
                (e.g. ``~/.animaworks/animas/rin``).

        Returns:
            True if allowed, False if depth exceeded.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(sender_anima_dir)
            entries = activity.recent(
                days=1,
                limit=200,
                types=["dm_sent", "dm_received"],
                involving=receiver,
            )
        except Exception:
            logger.warning(
                "Failed to read activity log for depth check: %s — blocking send (fail-closed)",
                sender_anima_dir,
                exc_info=True,
            )
            return False  # fail-closed: block if we can't verify

        cutoff = now_jst() - timedelta(seconds=self._window_s)
        count = 0
        for e in entries:
            try:
                ts = ensure_aware(datetime.fromisoformat(e.ts))
                if ts >= cutoff:
                    count += 1
            except (ValueError, TypeError):
                continue

        if count >= self._max_depth:
            logger.warning(
                "DEPTH LIMIT: %s -> %s blocked (%d exchanges in %ds window)",
                sender,
                receiver,
                count,
                self._window_s,
            )
            return False
        return True

    def current_depth(
        self,
        a: str,
        b: str,
        anima_dir: Path,
    ) -> int:
        """Return current exchange count for a pair within the active window.

        Args:
            a: First Anima name.
            b: Second Anima name (the peer).
            anima_dir: Path to Anima ``a``'s directory.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(anima_dir)
            entries = activity.recent(
                days=1,
                limit=200,
                types=["dm_sent", "dm_received"],
                involving=b,
            )
        except Exception:
            return 0

        cutoff = now_jst() - timedelta(seconds=self._window_s)
        count = 0
        for e in entries:
            try:
                ts = ensure_aware(datetime.fromisoformat(e.ts))
                if ts >= cutoff:
                    count += 1
            except (ValueError, TypeError):
                continue
        return count

    # ── Backward-compatible aliases ────────────────────────────
    def check_and_record(self, sender: str, receiver: str) -> bool:
        """Legacy API -- always returns True (no-op).

        .. deprecated::
            Use :meth:`check_depth` with ``sender_anima_dir`` instead.
        """
        import warnings

        warnings.warn(
            "check_and_record is deprecated; use check_depth with sender_anima_dir",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.debug(
            "Deprecated check_and_record() called for %s -> %s; "
            "use check_depth() with sender_anima_dir instead",
            sender,
            receiver,
        )
        return True


def get_depth_limiter() -> ConversationDepthLimiter:
    """Return a ConversationDepthLimiter with current config.

    Config is reloaded on each call so changes to heartbeat.max_depth
    and max_messages_per_hour take effect without process restart.
    """
    return ConversationDepthLimiter()


# Backward-compatible alias (deprecated)
depth_limiter = get_depth_limiter()
