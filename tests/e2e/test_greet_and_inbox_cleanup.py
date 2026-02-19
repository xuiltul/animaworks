"""E2E tests for inbox cleanup of non-JSON files."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from core.messenger import Messenger


class TestArchiveAllCleansNonJsonFiles:
    """Verify that archive_all moves non-JSON files to processed/."""

    def test_archive_all_cleans_non_json_files_e2e(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()

        alice = Messenger(shared_dir=shared, anima_name="alice")
        bob = Messenger(shared_dir=shared, anima_name="bob")

        # bob sends a regular JSON message to alice
        msg = bob.send(to="alice", content="hello alice")
        json_file = shared / "inbox" / "alice" / f"{msg.id}.json"
        assert json_file.exists(), "JSON message file should exist in alice's inbox"

        # Manually create a stray .txt file in alice's inbox
        txt_file = shared / "inbox" / "alice" / "stray_note.txt"
        txt_file.write_text("not a real message", encoding="utf-8")
        assert txt_file.exists()

        # archive_all should move both files
        alice.archive_all()

        processed = shared / "inbox" / "alice" / "processed"
        assert (processed / f"{msg.id}.json").exists(), (
            ".json file should be in processed/"
        )
        assert (processed / "stray_note.txt").exists(), (
            ".txt file should be in processed/"
        )

        # inbox should be empty (except the processed/ subdirectory)
        remaining = [
            p for p in (shared / "inbox" / "alice").iterdir() if p.is_file()
        ]
        assert remaining == [], (
            f"No files should remain in inbox, but found: {remaining}"
        )


class TestNonJsonFileDoesNotAffectHasUnread:
    """Verify that .txt files do not make has_unread() return True."""

    def test_non_json_file_does_not_affect_has_unread(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()

        messenger = Messenger(shared_dir=shared, anima_name="carol")

        # Place only a .txt file in the inbox
        txt_file = shared / "inbox" / "carol" / "spurious.txt"
        txt_file.write_text("agent sdk leftover", encoding="utf-8")
        assert txt_file.exists()

        # has_unread only checks *.json, so it should be False
        assert messenger.has_unread() is False, (
            "has_unread() should be False when only non-JSON files are present"
        )

        # archive_all should still clean up the .txt file
        messenger.archive_all()
        processed = shared / "inbox" / "carol" / "processed"
        assert (processed / "spurious.txt").exists(), (
            ".txt file should be moved to processed/ by archive_all"
        )
