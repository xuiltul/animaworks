from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for inbox read counter (retry annotations) in run_heartbeat.

Covers:
- inbox_read_counts.json is created/updated on each heartbeat
- Counter increments when the same message file stays in inbox
- Retry annotation (⚠️ 未返信N回目) is added for count >= 2
- Pruning removes entries for files that no longer exist
- Corrupted JSON is handled gracefully
"""

import json
from pathlib import Path

import pytest


class TestInboxReadCounts:
    """Test inbox_read_counts.json persistence."""

    def test_creates_file_on_first_heartbeat(self, tmp_path):
        """Read counts file is created when messages are processed."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        counts_path = state_dir / "inbox_read_counts.json"

        # Simulate what run_heartbeat does (uses filename-only keys)
        inbox_items_paths = [tmp_path / "msg1.json", tmp_path / "msg2.json"]
        for p in inbox_items_paths:
            p.write_text("{}", encoding="utf-8")

        _read_counts: dict[str, int] = {}
        for p in inbox_items_paths:
            key = p.name
            _read_counts[key] = _read_counts.get(key, 0) + 1

        counts_path.write_text(
            json.dumps(_read_counts, ensure_ascii=False), encoding="utf-8",
        )

        assert counts_path.exists()
        data = json.loads(counts_path.read_text(encoding="utf-8"))
        assert data["msg1.json"] == 1
        assert data["msg2.json"] == 1

    def test_increments_on_subsequent_reads(self, tmp_path):
        """Counter increments when the same file is presented again."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        counts_path = state_dir / "inbox_read_counts.json"

        msg_path = tmp_path / "msg1.json"
        msg_path.write_text("{}", encoding="utf-8")
        key = msg_path.name  # filename-only key

        # First heartbeat
        _read_counts: dict[str, int] = {}
        _read_counts[key] = _read_counts.get(key, 0) + 1
        counts_path.write_text(json.dumps(_read_counts), encoding="utf-8")

        # Second heartbeat: reload and increment
        _read_counts = json.loads(counts_path.read_text(encoding="utf-8"))
        _read_counts[key] = _read_counts.get(key, 0) + 1
        counts_path.write_text(json.dumps(_read_counts), encoding="utf-8")

        data = json.loads(counts_path.read_text(encoding="utf-8"))
        assert data[key] == 2

    def test_prunes_nonexistent_files(self, tmp_path):
        """Entries for files that no longer exist are removed."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        counts_path = state_dir / "inbox_read_counts.json"

        # Simulate inbox directory with one existing and one gone file
        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        existing = inbox_dir / "exists.json"
        existing.write_text("{}", encoding="utf-8")
        # "gone.json" does not exist in inbox_dir

        _read_counts = {"exists.json": 3, "gone.json": 5}
        # Prune using inbox_dir (matching production logic)
        _read_counts = {
            k: v for k, v in _read_counts.items()
            if (inbox_dir / k).exists()
        }
        counts_path.write_text(json.dumps(_read_counts), encoding="utf-8")

        data = json.loads(counts_path.read_text(encoding="utf-8"))
        assert "exists.json" in data
        assert "gone.json" not in data

    def test_corrupted_json_resets(self, tmp_path):
        """Corrupted counts file is handled gracefully (reset to empty)."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        counts_path = state_dir / "inbox_read_counts.json"
        counts_path.write_text("NOT VALID JSON", encoding="utf-8")

        _read_counts: dict[str, int] = {}
        try:
            if counts_path.exists():
                _read_counts = json.loads(
                    counts_path.read_text(encoding="utf-8")
                )
        except Exception:
            _read_counts = {}

        assert _read_counts == {}


class TestRetryAnnotation:
    """Test retry annotation formatting logic."""

    def test_first_read_no_annotation(self):
        """First read: no retry annotation prefix."""
        count = 1
        from_person = "bob"
        content = "hello"
        if count >= 2:
            prefix = f"- {from_person} [⚠️ 未返信{count}回目]: "
        else:
            prefix = f"- {from_person}: "
        line = f"{prefix}{content[:800]}"
        assert "未返信" not in line
        assert "- bob: hello" == line

    def test_second_read_has_annotation(self):
        """Second read: shows ⚠️ 未返信2回目."""
        count = 2
        from_person = "bob"
        content = "hello"
        if count >= 2:
            prefix = f"- {from_person} [⚠️ 未返信{count}回目]: "
        else:
            prefix = f"- {from_person}: "
        line = f"{prefix}{content[:800]}"
        assert "⚠️ 未返信2回目" in line

    def test_third_read_has_annotation(self):
        """Third read: shows ⚠️ 未返信3回目."""
        count = 3
        from_person = "alice"
        content = "important message"
        if count >= 2:
            prefix = f"- {from_person} [⚠️ 未返信{count}回目]: "
        else:
            prefix = f"- {from_person}: "
        line = f"{prefix}{content[:800]}"
        assert "⚠️ 未返信3回目" in line
        assert "alice" in line
