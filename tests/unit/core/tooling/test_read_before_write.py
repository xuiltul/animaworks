"""Unit tests for read-before-write guard and filename token hint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.tooling.handler import ToolHandler


@pytest.fixture
def anima_dir(tmp_path):
    """Create a minimal anima directory structure."""
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "procedures").mkdir()
    (tmp_path / "episodes").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "shortterm").mkdir()
    (tmp_path / "identity.md").write_text("identity")
    (tmp_path / "injection.md").write_text("injection")
    return tmp_path


@pytest.fixture
def handler(anima_dir):
    """Create a ToolHandler with mocked dependencies."""
    memory = MagicMock()
    memory.anima_dir = anima_dir
    h = ToolHandler(anima_dir=anima_dir, memory=memory)
    return h


# ── Read-before-write guard ─────────────────────────────


class TestReadBeforeWriteGuard:
    def test_block_overwrite_without_read(self, handler, anima_dir):
        """Overwriting an existing file without read_memory_file should be blocked."""
        (anima_dir / "knowledge" / "topic.md").write_text("original content")

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic.md", "content": "new content", "mode": "overwrite"},
        )

        assert "ReadBeforeWrite" in result or "read_memory_file" in result
        assert "original content" in result

    def test_allow_overwrite_after_read(self, handler, anima_dir):
        """Overwriting after read_memory_file should succeed."""
        (anima_dir / "knowledge" / "topic.md").write_text("original content")

        handler.handle("read_memory_file", {"path": "knowledge/topic.md"})
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic.md", "content": "updated content", "mode": "overwrite"},
        )

        assert "Written" in result
        assert (anima_dir / "knowledge" / "topic.md").read_text().endswith("updated content")

    def test_allow_new_file_without_read(self, handler, anima_dir):
        """Creating a new file should not require prior read."""
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/brand-new.md", "content": "new knowledge"},
        )

        assert "Written" in result

    def test_allow_append_without_read(self, handler, anima_dir):
        """Append mode should not require prior read."""
        (anima_dir / "knowledge" / "topic.md").write_text("original")

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic.md", "content": "\nmore", "mode": "append"},
        )

        assert "Written" in result

    def test_skip_episodes(self, handler, anima_dir):
        """Episodes directory should skip read-before-write check."""
        (anima_dir / "episodes" / "2026-03-15.md").write_text("old log")

        result = handler.handle(
            "write_memory_file",
            {"path": "episodes/2026-03-15.md", "content": "updated log"},
        )

        assert "Written" in result

    def test_skip_state(self, handler, anima_dir):
        """State directory should skip read-before-write check."""
        (anima_dir / "state" / "current_task.md").write_text("old task")

        result = handler.handle(
            "write_memory_file",
            {"path": "state/current_task.md", "content": "new task"},
        )

        assert "Written" in result

    def test_skip_shortterm(self, handler, anima_dir):
        """Shortterm directory should skip read-before-write check."""
        (anima_dir / "shortterm" / "session.md").write_text("old session")

        result = handler.handle(
            "write_memory_file",
            {"path": "shortterm/session.md", "content": "new session"},
        )

        assert "Written" in result

    def test_existing_content_preview_in_error(self, handler, anima_dir):
        """Error response should include preview of existing content."""
        long_content = "A" * 3000
        (anima_dir / "knowledge" / "big.md").write_text(long_content)

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/big.md", "content": "replacement"},
        )

        assert "ReadBeforeWrite" in result or "read_memory_file" in result
        assert "A" * 100 in result
        assert len(result) < 3500  # truncated

    def test_reset_session_clears_read_paths(self, handler, anima_dir):
        """reset_session_id should clear read tracking."""
        (anima_dir / "knowledge" / "topic.md").write_text("content")
        handler.handle("read_memory_file", {"path": "knowledge/topic.md"})

        handler.reset_session_id()

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic.md", "content": "overwrite"},
        )
        assert "ReadBeforeWrite" in result or "read_memory_file" in result

    def test_procedures_also_guarded(self, handler, anima_dir):
        """Procedures should also require read-before-write."""
        (anima_dir / "procedures" / "deploy.md").write_text("---\ndescription: Deploy\n---\nsteps")

        result = handler.handle(
            "write_memory_file",
            {"path": "procedures/deploy.md", "content": "---\ndescription: Deploy\n---\nnew steps"},
        )

        assert "ReadBeforeWrite" in result or "read_memory_file" in result


# ── Filename token hint ──────────────────────────────────


class TestFilenameTokenHint:
    def test_show_similar_files(self, handler, anima_dir):
        """New knowledge file should show similar existing files."""
        (anima_dir / "knowledge" / "japan_tax_treaty.md").write_text("---\nconfidence: 0.5\n---\ntax info")
        (anima_dir / "knowledge" / "japan_visa_guide.md").write_text("---\nconfidence: 0.5\n---\nvisa info")

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/japan-tax-law.md", "content": "new tax law"},
        )

        assert "Written" in result
        assert "japan_tax_treaty.md" in result

    def test_no_hint_when_no_similar(self, handler, anima_dir):
        """No hint when no similar files exist."""
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/unique-topic-xyz.md", "content": "unique content"},
        )

        assert "Written" in result
        assert "Similar" not in result and "類似" not in result

    def test_no_hint_for_existing_file_overwrite(self, handler, anima_dir):
        """Hint should not appear for overwriting existing files (only new)."""
        (anima_dir / "knowledge" / "topic.md").write_text("original")
        (anima_dir / "knowledge" / "topic_related.md").write_text("related")

        handler.handle("read_memory_file", {"path": "knowledge/topic.md"})
        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic.md", "content": "updated"},
        )

        assert "Written" in result
        assert "topic_related.md" not in result

    def test_hint_matches_hyphen_underscore_variants(self, handler, anima_dir):
        """Hint should match across hyphen/underscore naming."""
        (anima_dir / "knowledge" / "oss_license_risk.md").write_text("---\nconfidence: 0.5\n---\noss risk")

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/oss-license-law.md", "content": "oss law"},
        )

        assert "Written" in result
        assert "oss_license_risk.md" in result

    def test_hint_limited_to_10(self, handler, anima_dir):
        """Hint should show at most 10 similar files."""
        for i in range(15):
            (anima_dir / "knowledge" / f"topic_sub_{i:02d}.md").write_text(f"content {i}")

        result = handler.handle(
            "write_memory_file",
            {"path": "knowledge/topic-sub-new.md", "content": "new"},
        )

        assert "Written" in result
        lines = [l for l in result.split("\n") if l.strip().startswith("- topic_sub_")]
        assert len(lines) <= 10

    def test_no_hint_for_non_knowledge(self, handler, anima_dir):
        """Hint should not appear for non-knowledge directories."""
        (anima_dir / "state" / "task_queue.jsonl").write_text("")

        result = handler.handle(
            "write_memory_file",
            {"path": "state/task_new.md", "content": "task"},
        )

        assert "Written" in result
        assert "Similar" not in result and "類似" not in result


# ── Read path tracking ───────────────────────────────────


class TestReadPathTracking:
    def test_read_adds_to_read_paths(self, handler, anima_dir):
        """Successful read should track the path."""
        (anima_dir / "knowledge" / "file.md").write_text("content")

        handler.handle("read_memory_file", {"path": "knowledge/file.md"})

        assert "knowledge/file.md" in handler._read_paths

    def test_failed_read_does_not_track(self, handler):
        """Read of nonexistent file should not track."""
        handler.handle("read_memory_file", {"path": "knowledge/nonexistent.md"})

        assert "knowledge/nonexistent.md" not in handler._read_paths

    def test_reset_read_paths(self, handler, anima_dir):
        """reset_read_paths should clear all tracked paths."""
        (anima_dir / "knowledge" / "file.md").write_text("content")
        handler.handle("read_memory_file", {"path": "knowledge/file.md"})

        handler.reset_read_paths()

        assert len(handler._read_paths) == 0
