from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for core.memory._io — crash-safe I/O utilities.

Tests cover:
- Atomic write normal operation (write, read-back, overwrite)
- Parent directory auto-creation
- Error handling (original preserved, temp cleaned up)
- cleanup_tmp_files removal, non-existent dir, non-tmp file safety
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory._io import atomic_write_text, cleanup_tmp_files


# ── TestAtomicWriteText ────────────────────────────────────────────


class TestAtomicWriteText:
    """Tests for atomic_write_text()."""

    def test_normal_write(self, tmp_path: Path):
        """Write content and verify file content is correct."""
        target = tmp_path / "test.md"
        atomic_write_text(target, "hello world")

        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Write to a path with non-existent parent directories."""
        target = tmp_path / "a" / "b" / "c" / "deep.txt"
        assert not target.parent.exists()

        atomic_write_text(target, "nested content")

        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested content"

    def test_overwrites_existing(self, tmp_path: Path):
        """Write, then overwrite, verify new content."""
        target = tmp_path / "overwrite.txt"
        atomic_write_text(target, "original")
        assert target.read_text(encoding="utf-8") == "original"

        atomic_write_text(target, "replaced")
        assert target.read_text(encoding="utf-8") == "replaced"

    def test_preserves_original_on_error(self, tmp_path: Path):
        """If the write raises mid-way, the original file is unchanged."""
        target = tmp_path / "preserve.txt"
        atomic_write_text(target, "original content")

        # Patch os.fdopen so the write inside atomic_write_text raises
        with patch("core.memory._io.os.fdopen", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_write_text(target, "should not appear")

        assert target.read_text(encoding="utf-8") == "original content"

    def test_cleans_up_temp_on_error(self, tmp_path: Path):
        """After a failed write, no .tmp files remain in the directory."""
        target = tmp_path / "cleanup.txt"

        with patch("core.memory._io.os.fdopen", side_effect=OSError("fail")):
            with pytest.raises(OSError):
                atomic_write_text(target, "boom")

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Stale temp files found: {tmp_files}"

    def test_no_temp_files_left_on_success(self, tmp_path: Path):
        """After successful write, no .tmp files remain in the directory."""
        target = tmp_path / "clean.txt"
        atomic_write_text(target, "success")

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Temp files found after success: {tmp_files}"


# ── TestCleanupTmpFiles ───────────────────────────────────────────


class TestCleanupTmpFiles:
    """Tests for cleanup_tmp_files()."""

    def test_removes_stale_tmps(self, tmp_path: Path):
        """Create .tmp files, call cleanup, verify removed and count."""
        (tmp_path / ".foo.tmp").write_text("stale1")
        (tmp_path / ".bar.tmp").write_text("stale2")
        (tmp_path / ".baz.tmp").write_text("stale3")

        removed = cleanup_tmp_files(tmp_path)

        assert removed == 3
        remaining = list(tmp_path.glob(".*.tmp"))
        assert remaining == []

    def test_returns_zero_for_nonexistent_dir(self, tmp_path: Path):
        """Call on non-existent directory, verify returns 0."""
        missing = tmp_path / "does_not_exist"
        assert not missing.exists()

        removed = cleanup_tmp_files(missing)
        assert removed == 0

    def test_ignores_non_tmp_files(self, tmp_path: Path):
        """Non-.tmp files must not be removed."""
        (tmp_path / ".keep.md").write_text("important")
        (tmp_path / ".data.json").write_text("{}")
        (tmp_path / "regular.txt").write_text("normal file")
        # Add one .tmp to verify only it is removed
        (tmp_path / ".stale.tmp").write_text("remove me")

        removed = cleanup_tmp_files(tmp_path)

        assert removed == 1
        assert (tmp_path / ".keep.md").exists()
        assert (tmp_path / ".data.json").exists()
        assert (tmp_path / "regular.txt").exists()
        assert not (tmp_path / ".stale.tmp").exists()
