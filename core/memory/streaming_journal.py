from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Streaming journal — crash-resilient write-ahead log for streaming output.

Incrementally persists streaming text chunks to disk so that a hard crash
(SIGKILL, OOM) loses at most ~1 second of output instead of everything.

File layout::

    {anima_dir}/shortterm/streaming_journal.jsonl

Lifecycle:
    open()  → write_text() / write_tool_*()  → finalize()  (normal)
    open()  → write_text() / write_tool_*()  → <crash>     (abnormal)
    recover() on next startup reads the orphan and returns JournalRecovery.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from core.time_utils import now_jst

logger = logging.getLogger("animaworks.streaming_journal")

_JOURNAL_FILENAME = "streaming_journal.jsonl"

# ── Buffering configuration ───────────────────────────────────────
_FLUSH_INTERVAL_SEC = 1.0   # Flush at most every 1 second
_FLUSH_SIZE_CHARS = 500     # Flush when buffer reaches 500 chars


# ── Data models ───────────────────────────────────────────────────

@dataclass
class JournalRecovery:
    """Data recovered from an orphaned streaming journal."""

    trigger: str = ""
    from_person: str = ""
    session_id: str = ""
    started_at: str = ""
    recovered_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_event_at: str = ""
    is_complete: bool = False


# ── StreamingJournal ──────────────────────────────────────────────

class StreamingJournal:
    """Write-ahead journal for streaming output.

    One journal file per streaming cycle.  On normal completion the file
    is deleted.  If the process crashes, the file survives and can be
    recovered on the next startup.
    """

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._shortterm_dir = anima_dir / "shortterm"
        self._journal_path = self._shortterm_dir / _JOURNAL_FILENAME
        self._fd: TextIOWrapper | None = None
        self._buffer: str = ""
        self._last_flush: float = 0.0
        self._finalized: bool = False

    # ── Lifecycle ─────────────────────────────────────────────

    def open(
        self,
        trigger: str,
        from_person: str = "",
        session_id: str = "",
    ) -> None:
        """Open the journal and write the start event.

        If an orphaned journal already exists (e.g. from a concurrent
        heartbeat + chat), it is recovered first to prevent data loss.
        """
        self._shortterm_dir.mkdir(parents=True, exist_ok=True)
        # Recover orphaned journal before overwriting — persist to episode log
        if self._journal_path.exists():
            recovery = StreamingJournal.recover(self._anima_dir)
            if recovery and recovery.recovered_text:
                logger.warning(
                    "Orphaned journal recovered on open: %d chars, trigger=%s",
                    len(recovery.recovered_text),
                    recovery.trigger,
                )
                self._persist_recovery(recovery)
            else:
                logger.warning(
                    "Orphaned journal found on open; no content recovered",
                )
        self._fd = open(self._journal_path, "w", encoding="utf-8")
        self._buffer = ""
        self._last_flush = time.monotonic()
        self._finalized = False
        self._write_event({
            "ev": "start",
            "trigger": trigger,
            "from": from_person,
            "session_id": session_id,
        })
        logger.debug("Streaming journal opened: trigger=%s", trigger)

    def write_text(self, text: str) -> None:
        """Append a text fragment.  Buffered to reduce I/O."""
        if not self._fd or self._finalized:
            return
        self._buffer += text
        now = time.monotonic()
        if (
            len(self._buffer) >= _FLUSH_SIZE_CHARS
            or now - self._last_flush >= _FLUSH_INTERVAL_SEC
        ):
            self._flush_buffer()

    def write_tool_start(
        self, tool: str, args_summary: str = "", *, tool_id: str = "",
    ) -> None:
        """Record tool execution start.

        Args:
            tool: Tool name.
            args_summary: Truncated summary of tool arguments.
            tool_id: Provider-assigned tool call ID for correlation.
        """
        if not self._fd or self._finalized:
            return
        # Flush pending text before tool event
        self._flush_buffer()
        event: dict[str, Any] = {
            "ev": "tool_start",
            "tool": tool,
            "args_summary": args_summary[:200],
        }
        if tool_id:
            event["tool_id"] = tool_id
        self._write_event(event)

    def write_tool_end(
        self, tool: str, result_summary: str = "", *, tool_id: str = "",
    ) -> None:
        """Record tool execution end.

        Args:
            tool: Tool name.
            result_summary: Truncated summary of tool result.
            tool_id: Provider-assigned tool call ID for correlation.
        """
        if not self._fd or self._finalized:
            return
        self._flush_buffer()
        event: dict[str, Any] = {
            "ev": "tool_end",
            "tool": tool,
            "result_summary": result_summary[:200],
        }
        if tool_id:
            event["tool_id"] = tool_id
        self._write_event(event)

    def finalize(self, summary: str = "") -> None:
        """Write done event, close file handle, and delete journal."""
        if not self._fd or self._finalized:
            return
        self._flush_buffer()
        self._write_event({"ev": "done", "summary": summary[:500]})
        self._finalized = True
        self._close_fd()
        # Delete the journal file on successful completion
        try:
            self._journal_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete journal file: %s", self._journal_path)
        logger.debug("Streaming journal finalized and deleted")

    def close(self) -> None:
        """Close file handle without finalizing.

        Called in finally blocks.  If finalize() was already called this
        is a no-op.  Otherwise the orphaned journal file remains on disk
        for crash recovery.
        """
        if self._finalized:
            return
        # Flush remaining buffer before closing
        if self._fd and self._buffer:
            try:
                self._flush_buffer()
            except Exception:
                logger.debug("Failed to flush buffer on close", exc_info=True)
        self._close_fd()

    # ── Recovery (class methods) ──────────────────────────────

    @classmethod
    def has_orphan(cls, anima_dir: Path) -> bool:
        """Check whether an orphaned journal exists."""
        path = anima_dir / "shortterm" / _JOURNAL_FILENAME
        return path.exists()

    @classmethod
    def recover(cls, anima_dir: Path) -> JournalRecovery | None:
        """Read and delete an orphaned journal.

        Returns ``None`` if no journal exists.  Corrupted JSONL lines
        are silently skipped.
        """
        path = anima_dir / "shortterm" / _JOURNAL_FILENAME
        if not path.exists():
            return None

        trigger = ""
        from_person = ""
        session_id = ""
        started_at = ""
        last_event_at = ""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        is_complete = False

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Failed to read orphan journal: %s", path)
            return None

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # Corrupted line (partial write during crash) — skip
                logger.debug("Skipping corrupted journal line: %.60s", line)
                continue

            ev = entry.get("ev", "")
            ts = entry.get("ts", "")
            if ts:
                last_event_at = ts

            if ev == "start":
                trigger = entry.get("trigger", "")
                from_person = entry.get("from", "")
                session_id = entry.get("session_id", "")
                started_at = ts
            elif ev == "text":
                text_parts.append(entry.get("t", ""))
            elif ev == "tool_start":
                tc_entry: dict[str, Any] = {
                    "tool": entry.get("tool", ""),
                    "args_summary": entry.get("args_summary", ""),
                    "status": "started",
                }
                if entry.get("tool_id"):
                    tc_entry["tool_id"] = entry["tool_id"]
                tool_calls.append(tc_entry)
            elif ev == "tool_end":
                tool_name = entry.get("tool", "")
                end_tool_id = entry.get("tool_id", "")
                # Update matching tool_start entry:
                # prefer tool_id match, fall back to tool_name (backward compat)
                matched = False
                if end_tool_id:
                    for tc in reversed(tool_calls):
                        if (
                            tc.get("tool_id") == end_tool_id
                            and tc["status"] == "started"
                        ):
                            tc["status"] = "completed"
                            tc["result_summary"] = entry.get("result_summary", "")
                            matched = True
                            break
                if not matched:
                    for tc in reversed(tool_calls):
                        if tc["tool"] == tool_name and tc["status"] == "started":
                            tc["status"] = "completed"
                            tc["result_summary"] = entry.get("result_summary", "")
                            matched = True
                            break
                if not matched:
                    tc_entry = {
                        "tool": tool_name,
                        "result_summary": entry.get("result_summary", ""),
                        "status": "completed",
                    }
                    if end_tool_id:
                        tc_entry["tool_id"] = end_tool_id
                    tool_calls.append(tc_entry)
            elif ev == "done":
                is_complete = True

        recovery = JournalRecovery(
            trigger=trigger,
            from_person=from_person,
            session_id=session_id,
            started_at=started_at,
            recovered_text="".join(text_parts),
            tool_calls=tool_calls,
            last_event_at=last_event_at,
            is_complete=is_complete,
        )

        logger.info(
            "Recovered streaming journal: %d chars, %d tool calls, complete=%s",
            len(recovery.recovered_text),
            len(recovery.tool_calls),
            recovery.is_complete,
        )
        return recovery

    @classmethod
    def confirm_recovery(cls, anima_dir: Path) -> None:
        """Delete journal after recovery data has been safely persisted."""
        path = anima_dir / "shortterm" / _JOURNAL_FILENAME
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete recovered journal: %s", path)

    # ── Private helpers ───────────────────────────────────────

    def _persist_recovery(self, recovery: JournalRecovery) -> None:
        """Persist recovered orphan content to an episode file.

        Saves the recovered text to ``episodes/recovered_{timestamp}.md``
        so the data is not permanently lost.
        """
        episodes_dir = self._anima_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        ts = now_jst().strftime("%Y-%m-%d_%H%M%S")
        recovery_file = episodes_dir / f"recovered_{ts}.md"
        content_lines = [
            f"# Recovered Streaming Journal ({recovery.trigger})",
            f"- from: {recovery.from_person}",
            f"- session_id: {recovery.session_id}",
            f"- started_at: {recovery.started_at}",
            f"- last_event_at: {recovery.last_event_at}",
            f"- complete: {recovery.is_complete}",
            "",
            "## Recovered Text",
            "",
            recovery.recovered_text,
        ]
        try:
            recovery_file.write_text("\n".join(content_lines), encoding="utf-8")
            logger.info("Persisted orphan recovery to %s", recovery_file)
        except OSError:
            logger.warning("Failed to persist orphan recovery to %s", recovery_file)

    def _write_event(self, event: dict[str, Any]) -> None:
        """Write a single JSONL event line with timestamp."""
        if not self._fd:
            return
        event.setdefault("ts", now_jst().isoformat(timespec="seconds"))
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self._fd.write(line + "\n")
        self._fd.flush()
        try:
            os.fsync(self._fd.fileno())
        except OSError:
            logger.debug("fsync failed for journal", exc_info=True)

    def _flush_buffer(self) -> None:
        """Write buffered text as a single text event."""
        if not self._buffer or not self._fd:
            return
        self._write_event({"ev": "text", "t": self._buffer})
        self._buffer = ""
        self._last_flush = time.monotonic()

    def _close_fd(self) -> None:
        """Close the file descriptor safely."""
        if self._fd:
            try:
                self._fd.close()
            except OSError:
                logger.debug("Failed to close journal fd", exc_info=True)
            self._fd = None
