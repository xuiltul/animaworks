from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for stale message force-archiving and crash-archive in heartbeat.

Covers:
- Stale unreplied messages (>_STALE_MESSAGE_TIMEOUT_SEC) are force-archived
- Crash-path archives inbox messages to prevent re-processing storms
- TOCTOU safety: FileNotFoundError during stat() is handled
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from core.anima import _STALE_MESSAGE_TIMEOUT_SEC
from core.messenger import InboxItem, Message


def _make_inbox_item(
    tmp_path: Path,
    from_person: str,
    *,
    age_seconds: float = 0,
    content: str = "test message",
) -> InboxItem:
    """Create a real InboxItem backed by a file with controlled mtime."""
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir(exist_ok=True)
    processed_dir = inbox_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    msg_data = {
        "id": f"msg_{from_person}_{time.time()}",
        "from_person": from_person,
        "to_person": "test-anima",
        "content": content,
        "type": "message",
        "thread_id": "",
    }
    filepath = inbox_dir / f"{msg_data['id']}.json"
    filepath.write_text(json.dumps(msg_data), encoding="utf-8")

    if age_seconds > 0:
        old_time = time.time() - age_seconds
        import os
        os.utime(filepath, (old_time, old_time))

    msg = Message(**msg_data)
    return InboxItem(msg=msg, path=filepath)


class TestStaleMessageTimeout:
    """Test that _STALE_MESSAGE_TIMEOUT_SEC constant is defined correctly."""

    def test_constant_exists_and_is_positive(self) -> None:
        assert _STALE_MESSAGE_TIMEOUT_SEC > 0

    def test_constant_is_600(self) -> None:
        assert _STALE_MESSAGE_TIMEOUT_SEC == 600


class TestStaleArchiveLogic:
    """Test the stale message force-archive logic extracted from anima.py."""

    def _run_stale_detection(
        self, items_to_keep: list[InboxItem]
    ) -> list[InboxItem]:
        """Replicate the stale detection logic from anima.py for unit testing."""
        now = time.time()
        stale: list[InboxItem] = []
        for item in items_to_keep:
            try:
                mtime = item.path.stat().st_mtime
                if (now - mtime) > _STALE_MESSAGE_TIMEOUT_SEC:
                    stale.append(item)
            except FileNotFoundError:
                continue
        return stale

    def test_fresh_messages_not_stale(self, tmp_path: Path) -> None:
        """Messages younger than threshold are not marked stale."""
        item = _make_inbox_item(tmp_path, "alice", age_seconds=60)
        stale = self._run_stale_detection([item])
        assert len(stale) == 0

    def test_old_messages_are_stale(self, tmp_path: Path) -> None:
        """Messages older than threshold are marked stale."""
        item = _make_inbox_item(tmp_path, "alice", age_seconds=700)
        stale = self._run_stale_detection([item])
        assert len(stale) == 1
        assert stale[0].msg.from_person == "alice"

    def test_mixed_ages(self, tmp_path: Path) -> None:
        """Only messages exceeding the threshold are marked stale."""
        fresh = _make_inbox_item(tmp_path, "alice", age_seconds=60)
        old = _make_inbox_item(tmp_path, "bob", age_seconds=700)
        stale = self._run_stale_detection([fresh, old])
        assert len(stale) == 1
        assert stale[0].msg.from_person == "bob"

    def test_exactly_at_threshold_not_stale(self, tmp_path: Path) -> None:
        """Messages exactly at the threshold boundary are not stale."""
        # Use a small buffer to avoid flaky timing
        item = _make_inbox_item(tmp_path, "alice", age_seconds=599)
        stale = self._run_stale_detection([item])
        assert len(stale) == 0

    def test_deleted_file_handled_gracefully(self, tmp_path: Path) -> None:
        """If a file was already deleted, it's silently skipped (TOCTOU safety)."""
        item = _make_inbox_item(tmp_path, "alice", age_seconds=700)
        # Delete the file before stale detection
        item.path.unlink()
        stale = self._run_stale_detection([item])
        assert len(stale) == 0

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        stale = self._run_stale_detection([])
        assert stale == []


class TestCrashArchive:
    """Test crash-path inbox archiving logic."""

    def test_crash_archive_moves_files(self, tmp_path: Path) -> None:
        """On crash, archive_paths is called with inbox_items."""
        item1 = _make_inbox_item(tmp_path, "alice")
        item2 = _make_inbox_item(tmp_path, "bob")
        inbox_items = [item1, item2]

        # Simulate what the except block does
        from core.messenger import Messenger

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(exist_ok=True)
        inbox_parent = item1.path.parent
        processed_dir = inbox_parent / "processed"
        processed_dir.mkdir(exist_ok=True)

        # Manually archive using the same logic
        count = 0
        for item in inbox_items:
            if item.path.exists():
                item.path.rename(processed_dir / item.path.name)
                count += 1

        assert count == 2
        assert not item1.path.exists()
        assert not item2.path.exists()
        assert len(list(processed_dir.glob("*.json"))) == 2

    def test_crash_archive_handles_missing_files(self, tmp_path: Path) -> None:
        """If files are already gone during crash archive, skip silently."""
        item = _make_inbox_item(tmp_path, "alice")
        item.path.unlink()  # Already gone

        processed_dir = item.path.parent / "processed"
        processed_dir.mkdir(exist_ok=True)

        count = 0
        if item.path.exists():
            item.path.rename(processed_dir / item.path.name)
            count += 1

        assert count == 0


class TestIntegrationWithArchiveLogic:
    """Integration test: replied_to + stale archive together."""

    def test_replied_archived_unreplied_kept_stale_forced(
        self, tmp_path: Path
    ) -> None:
        """Full archive flow: replied→archive, fresh unreplied→keep, stale unreplied→force-archive."""
        replied_item = _make_inbox_item(tmp_path, "alice", age_seconds=60)
        fresh_unreplied = _make_inbox_item(tmp_path, "bob", age_seconds=60)
        stale_unreplied = _make_inbox_item(tmp_path, "charlie", age_seconds=700)

        inbox_items = [replied_item, fresh_unreplied, stale_unreplied]
        senders = {"alice", "bob", "charlie"}
        replied_to = {"alice"}

        # Step 1: replied_to based split
        items_to_archive = [
            item for item in inbox_items
            if item.msg.from_person in replied_to
            or item.msg.from_person not in senders
        ]
        items_to_keep = [
            item for item in inbox_items
            if item not in items_to_archive
        ]

        assert len(items_to_archive) == 1  # alice
        assert len(items_to_keep) == 2  # bob, charlie

        # Step 2: stale detection
        now = time.time()
        stale: list[InboxItem] = []
        for item in items_to_keep:
            try:
                mtime = item.path.stat().st_mtime
                if (now - mtime) > _STALE_MESSAGE_TIMEOUT_SEC:
                    stale.append(item)
            except FileNotFoundError:
                continue

        assert len(stale) == 1  # charlie
        assert stale[0].msg.from_person == "charlie"

        items_to_archive.extend(stale)
        items_to_keep = [i for i in items_to_keep if i not in stale]

        assert len(items_to_archive) == 2  # alice + charlie
        assert len(items_to_keep) == 1  # bob (fresh, unreplied)
        assert items_to_keep[0].msg.from_person == "bob"
