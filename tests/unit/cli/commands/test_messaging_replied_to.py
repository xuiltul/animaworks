from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _persist_replied_to_for_a1 in CLI messaging module.

Covers the A1-mode replied_to tracking bridge: when ANIMAWORKS_ANIMA_DIR is
set, cmd_send writes to {anima_dir}/run/replied_to.jsonl so the Agent SDK
executor can detect which senders were replied to.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPersistRepliedToForA1:
    """Test _persist_replied_to_for_a1() writes to replied_to.jsonl."""

    def _call(self, to: str) -> None:
        from cli.commands.messaging import _persist_replied_to_for_a1

        _persist_replied_to_for_a1(to)

    def test_writes_entry_when_env_set(self, tmp_path: Path) -> None:
        """When ANIMAWORKS_ANIMA_DIR is set, the function writes a JSONL entry."""
        with patch.dict(os.environ, {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            self._call("mio")

        path = tmp_path / "run" / "replied_to.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry == {"to": "mio", "success": True}

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple calls append to the same file."""
        with patch.dict(os.environ, {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            self._call("alice")
            self._call("bob")
            self._call("charlie")

        path = tmp_path / "run" / "replied_to.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        names = [json.loads(line)["to"] for line in lines]
        assert names == ["alice", "bob", "charlie"]

    def test_noop_when_env_not_set(self, tmp_path: Path) -> None:
        """When ANIMAWORKS_ANIMA_DIR is not set, nothing is written."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANIMAWORKS_ANIMA_DIR", None)
            self._call("mio")

        # No run/ directory should be created anywhere
        assert not (tmp_path / "run" / "replied_to.jsonl").exists()

    def test_creates_run_directory(self, tmp_path: Path) -> None:
        """The run/ directory is created if it doesn't exist."""
        assert not (tmp_path / "run").exists()
        with patch.dict(os.environ, {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            self._call("yuki")
        assert (tmp_path / "run" / "replied_to.jsonl").exists()

    def test_handles_write_error_gracefully(self, tmp_path: Path) -> None:
        """If writing fails, the function logs but does not raise."""
        # Point to a non-writable path
        bad_path = tmp_path / "readonly"
        bad_path.mkdir()
        readonly_file = bad_path / "run" / "replied_to.jsonl"
        readonly_file.parent.mkdir(parents=True)
        readonly_file.touch()
        readonly_file.chmod(0o000)
        try:
            with patch.dict(os.environ, {"ANIMAWORKS_ANIMA_DIR": str(bad_path)}):
                # Should not raise
                self._call("mio")
        finally:
            readonly_file.chmod(0o644)

    def test_format_matches_toolhandler(self, tmp_path: Path) -> None:
        """Output format must match ToolHandler._persist_replied_to()."""
        with patch.dict(os.environ, {"ANIMAWORKS_ANIMA_DIR": str(tmp_path)}):
            self._call("test-anima")

        path = tmp_path / "run" / "replied_to.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        # Must have exactly these keys (same as ToolHandler format)
        assert set(entry.keys()) == {"to", "success"}
        assert isinstance(entry["to"], str)
        assert isinstance(entry["success"], bool)
