from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import PermissionsConfig
from core.tooling.handler import ToolHandler


@pytest.fixture
def anima_dir(tmp_path):
    """Create a minimal anima directory structure."""
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "procedures").mkdir()
    (tmp_path / "identity.md").write_text("identity")
    (tmp_path / "injection.md").write_text("injection")
    return tmp_path


@pytest.fixture
def handler(anima_dir):
    """Create a ToolHandler with mocked dependencies."""
    memory = MagicMock()
    memory.anima_dir = anima_dir
    h = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
    )
    return h


class TestArchiveMemoryFile:
    def test_explicit_deny_blocks_archive_source(self, handler, anima_dir):
        denied = anima_dir / "knowledge" / "private"
        denied.mkdir()
        target = denied / "secret.md"
        target.write_text("secret")
        config = PermissionsConfig(file_roots_denied=[str(denied)])

        with patch("core.tooling.handler_perms.load_permissions", return_value=config):
            result = handler.handle(
                "archive_memory_file",
                {"path": "knowledge/private/secret.md", "reason": "test"},
            )

        assert json.loads(result)["error_type"] == "PermissionDenied"
        assert target.exists()

    def test_explicit_deny_blocks_archive_destination(self, handler, anima_dir):
        target = anima_dir / "knowledge" / "old-info.md"
        target.write_text("secret")
        denied = anima_dir / "archive"
        config = PermissionsConfig(file_roots_denied=[str(denied)])

        with patch("core.tooling.handler_perms.load_permissions", return_value=config):
            result = handler.handle(
                "archive_memory_file",
                {"path": "knowledge/old-info.md", "reason": "test"},
            )

        assert json.loads(result)["error_type"] == "PermissionDenied"
        assert target.exists()
        assert not denied.exists()

    def test_archive_knowledge_file(self, handler, anima_dir):
        """Successfully archive a knowledge file."""
        target = anima_dir / "knowledge" / "old-info.md"
        target.write_text("old knowledge content")

        result = handler.handle(
            "archive_memory_file",
            {
                "path": "knowledge/old-info.md",
                "reason": "superseded by new-info.md",
            },
        )

        assert "Archived" in result
        assert not target.exists()
        archive_dir = anima_dir / "archive" / "superseded"
        assert archive_dir.exists()
        archived = list(archive_dir.glob("old-info.md*"))
        assert len(archived) == 1

    def test_archive_procedure_file(self, handler, anima_dir):
        """Successfully archive a procedure file."""
        target = anima_dir / "procedures" / "old-proc.md"
        target.write_text("old procedure")

        result = handler.handle(
            "archive_memory_file",
            {
                "path": "procedures/old-proc.md",
                "reason": "outdated",
            },
        )

        assert "Archived" in result
        assert not target.exists()

    def test_reject_protected_file(self, handler, anima_dir):
        """Reject archiving protected files like identity.md."""
        result = handler.handle(
            "archive_memory_file",
            {
                "path": "identity.md",
                "reason": "test",
            },
        )

        assert "error" in result.lower() or "Error" in result or "denied" in result.lower() or "only" in result.lower()
        assert (anima_dir / "identity.md").exists()

    def test_reject_non_memory_directory(self, handler, anima_dir):
        """Reject archiving from non-knowledge/procedures directories."""
        (anima_dir / "state").mkdir(exist_ok=True)
        (anima_dir / "state" / "current_state.md").write_text("state")

        result = handler.handle(
            "archive_memory_file",
            {
                "path": "state/current_state.md",
                "reason": "test",
            },
        )

        assert "knowledge" in result.lower() or "procedures" in result.lower()
        assert (anima_dir / "state" / "current_state.md").exists()

    def test_nonexistent_file(self, handler, anima_dir):
        """Return error for nonexistent file."""
        result = handler.handle(
            "archive_memory_file",
            {
                "path": "knowledge/does-not-exist.md",
                "reason": "test",
            },
        )

        assert "not found" in result.lower() or "error" in result.lower() or "Error" in result

    def test_missing_required_args(self, handler):
        """Return error when required args are missing."""
        result = handler.handle(
            "archive_memory_file",
            {
                "path": "knowledge/test.md",
            },
        )
        assert "reason" in result.lower() or "error" in result.lower() or "Error" in result

    def test_archive_filename_collision(self, handler, anima_dir):
        """Handle filename collision in archive directory."""
        target = anima_dir / "knowledge" / "info.md"
        target.write_text("content v2")

        # Pre-create archive with same name
        archive_dir = anima_dir / "archive" / "superseded"
        archive_dir.mkdir(parents=True)
        (archive_dir / "info.md").write_text("content v1")

        result = handler.handle(
            "archive_memory_file",
            {
                "path": "knowledge/info.md",
                "reason": "updated",
            },
        )

        assert "Archived" in result
        assert not target.exists()
        # Should have 2 files in archive now
        archived = list(archive_dir.glob("info*"))
        assert len(archived) == 2
