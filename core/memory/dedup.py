from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Message deduplication and rate limiting for heartbeat processing.

Provides three mechanisms:
1. Resolved topic detection — suppress messages about already-resolved issues
2. Same-sender consolidation — merge 3+ messages from same sender into summary
3. Rate limiting — defer 5+ messages from same sender to next heartbeat
"""

import json
import logging
import os
from collections import Counter
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.dedup")

# Threshold for message consolidation (same sender)
_CONSOLIDATION_THRESHOLD = 3
# Threshold for rate limiting (same sender)
_RATE_LIMIT_THRESHOLD = 5


class MessageDeduplicator:
    """Deduplicates and rate-limits heartbeat messages."""

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self._suppressed_path = anima_dir / "state" / "suppressed_messages.jsonl"
        self._deferred_path = anima_dir / "state" / "deferred_messages.jsonl"

    def is_resolved_topic(self, message_content: str, resolutions: list[dict[str, str]]) -> bool:
        """Check if a message relates to an already-resolved issue.

        Args:
            message_content: The message text to check.
            resolutions: List of resolution dicts with 'issue' key.

        Returns:
            True if the message topic appears to be already resolved.
        """
        if not resolutions or not message_content:
            return False

        content_lower = message_content.lower()
        for resolution in resolutions:
            issue = resolution.get("issue", "").lower()
            if not issue:
                continue
            # Extract keywords from resolution (words >= 3 chars)
            keywords = [w for w in issue.split() if len(w) >= 3]
            if not keywords:
                continue
            # If 2+ keywords match, consider it resolved
            match_count = sum(1 for kw in keywords if kw in content_lower)
            if match_count >= 2:
                logger.debug(
                    "Resolved topic detected: %d keywords matched from '%s'",
                    match_count, issue[:50],
                )
                return True
        return False

    def consolidate_messages(
        self, messages: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Consolidate messages from same sender when 3+ messages exist.

        Args:
            messages: List of message objects with from_person and content attributes.

        Returns:
            Tuple of (consolidated_messages, suppressed_messages).
            Consolidated messages replace N originals with 1 summary message.
        """
        if len(messages) < _CONSOLIDATION_THRESHOLD:
            return messages, []

        # Count messages per sender
        sender_counts: Counter[str] = Counter()
        for m in messages:
            sender_counts[m.from_person] += 1

        consolidated: list[Any] = []
        suppressed: list[Any] = []

        # Group by sender
        by_sender: dict[str, list[Any]] = {}
        for m in messages:
            by_sender.setdefault(m.from_person, []).append(m)

        for sender, sender_msgs in by_sender.items():
            if len(sender_msgs) >= _CONSOLIDATION_THRESHOLD:
                # Keep a copy of the first message with summary of all
                first = copy(sender_msgs[0])
                summaries = [
                    m.content[:100] for m in sender_msgs
                ]
                summary_text = (
                    f"[{len(sender_msgs)}件のメッセージを統合] "
                    + " / ".join(summaries)
                )
                first.content = summary_text
                consolidated.append(first)
                suppressed.extend(sender_msgs[1:])
            else:
                consolidated.extend(sender_msgs)

        return consolidated, suppressed

    def apply_rate_limit(
        self, messages: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Rate limit: defer messages when same sender has 5+ messages.

        Args:
            messages: List of message objects.

        Returns:
            Tuple of (accepted_messages, deferred_messages).
            Deferred messages are saved to state/deferred_messages.jsonl.
        """
        if len(messages) < _RATE_LIMIT_THRESHOLD:
            return messages, []

        sender_counts: Counter[str] = Counter()
        for m in messages:
            sender_counts[m.from_person] += 1

        accepted: list[Any] = []
        deferred: list[Any] = []

        # Per-sender: accept first 3, defer the rest if 5+ total
        seen_per_sender: Counter[str] = Counter()
        for m in messages:
            seen_per_sender[m.from_person] += 1
            if sender_counts[m.from_person] >= _RATE_LIMIT_THRESHOLD:
                if seen_per_sender[m.from_person] <= _CONSOLIDATION_THRESHOLD:
                    accepted.append(m)
                else:
                    deferred.append(m)
            else:
                accepted.append(m)

        # Save deferred messages
        if deferred:
            self._save_deferred(deferred)
            logger.info(
                "Rate limited: %d messages deferred to next heartbeat",
                len(deferred),
            )

        return accepted, deferred

    def load_deferred(self) -> list[dict[str, Any]]:
        """Load previously deferred messages.

        Reads and deletes the deferred messages file.
        Returns list of raw message dicts.
        """
        if not self._deferred_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            for line in self._deferred_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            # Delete after reading
            self._deferred_path.unlink(missing_ok=True)
            logger.info("Loaded %d deferred messages", len(entries))
        except Exception:
            logger.exception("Failed to load deferred messages")
        return entries

    def archive_suppressed(self, messages: list[Any]) -> None:
        """Archive suppressed messages to state/suppressed_messages.jsonl."""
        if not messages:
            return
        try:
            self._suppressed_path.parent.mkdir(parents=True, exist_ok=True)
            with self._suppressed_path.open("a", encoding="utf-8") as f:
                for m in messages:
                    entry = {
                        "ts": datetime.now().isoformat(),
                        "from": getattr(m, "from_person", str(m)),
                        "content": getattr(m, "content", str(m))[:500],
                        "reason": "dedup_suppressed",
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            logger.debug("Archived %d suppressed messages", len(messages))
        except Exception:
            logger.exception("Failed to archive suppressed messages")

    def _save_deferred(self, messages: list[Any]) -> None:
        """Save deferred messages for next heartbeat."""
        try:
            self._deferred_path.parent.mkdir(parents=True, exist_ok=True)
            with self._deferred_path.open("a", encoding="utf-8") as f:
                for m in messages:
                    entry = {
                        "ts": datetime.now().isoformat(),
                        "from": getattr(m, "from_person", str(m)),
                        "content": getattr(m, "content", str(m)),
                        "type": getattr(m, "type", "message"),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.exception("Failed to save deferred messages")
