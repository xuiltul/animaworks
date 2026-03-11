from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for path traversal prevention in memory file handlers and create_anima.

Covers:
- read_memory_file: common_knowledge/ path traversal blocked
- write_memory_file: common_knowledge/ path traversal blocked
- read_memory_file: valid common_knowledge/ paths still work
- write_memory_file: valid common_knowledge/ paths still work
- read_memory_file: reference/ path traversal blocked, valid paths work
- write_memory_file: reference/ prefix rejected (read-only)
- create_anima: character_sheet_path traversal blocked
- create_anima: valid character_sheet_path still works
"""

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Helpers ──────────────────────────────────────────────


def _make_handler(tmp_path: Path, anima_name: str = "test_anima"):
    """Create a ToolHandler with minimal mocked dependencies."""
    from core.tooling.handler import ToolHandler

    with patch("core.tooling.handler.ToolHandler.__init__", lambda self, **kw: None):
        handler = ToolHandler.__new__(ToolHandler)

    handler._anima_dir = tmp_path / "animas" / anima_name
    handler._anima_dir.mkdir(parents=True, exist_ok=True)
    handler._anima_name = anima_name
    handler._memory = MagicMock()
    handler._messenger = None
    handler._on_message_sent = None
    handler._on_schedule_changed = None
    handler._human_notifier = None
    handler._background_manager = None
    handler._pending_notifications = []
    handler._replied_to = {"chat": set(), "background": set()}
    handler._superuser = False
    handler._subordinate_activity_dirs = []
    handler._subordinate_management_files = []
    handler._subordinate_root_dirs = []
    handler._descendant_activity_dirs = []
    handler._descendant_state_files = []
    handler._descendant_state_dirs = []
    handler._state_file_lock = None
    handler._session_id = uuid.uuid4().hex[:12]
    handler._process_supervisor = None

    from core.memory.activity import ActivityLogger

    handler._activity = MagicMock(spec=ActivityLogger)

    from core.tooling.dispatch import ExternalToolDispatcher

    handler._external = MagicMock(spec=ExternalToolDispatcher)

    return handler


# ── common_knowledge read traversal ──────────────────────


class TestCommonKnowledgeReadTraversal:
    """Path traversal prevention in _handle_read_memory_file for common_knowledge/."""

    def test_rejects_traversal_with_dotdot(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_read_memory_file(
                {"path": "common_knowledge/../../etc/passwd"},
            )

        assert "PermissionDenied" in result
        assert "traversal" in result.lower()

    def test_rejects_deep_traversal(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_read_memory_file(
                {"path": "common_knowledge/../../../tmp/malicious"},
            )

        assert "PermissionDenied" in result

    def test_allows_valid_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)
        (ck_dir / "valid_file.md").write_text("hello", encoding="utf-8")

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_read_memory_file(
                {"path": "common_knowledge/valid_file.md"},
            )

        assert result == "hello"

    def test_allows_nested_valid_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        sub = ck_dir / "subdir"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "nested.md").write_text("nested content", encoding="utf-8")

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_read_memory_file(
                {"path": "common_knowledge/subdir/nested.md"},
            )

        assert result == "nested content"


# ── common_knowledge write traversal ─────────────────────


class TestCommonKnowledgeWriteTraversal:
    """Path traversal prevention in _handle_write_memory_file for common_knowledge/."""

    def test_rejects_traversal_with_dotdot(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_write_memory_file(
                {"path": "common_knowledge/../../../tmp/malicious.sh", "content": "bad"},
            )

        assert "PermissionDenied" in result
        assert "traversal" in result.lower()

    def test_rejects_single_level_escape(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_write_memory_file(
                {"path": "common_knowledge/../secret.txt", "content": "x"},
            )

        assert "PermissionDenied" in result

    def test_allows_valid_write(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ck_dir = tmp_path / "common_knowledge"
        ck_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_common_knowledge_dir", return_value=ck_dir):
            result = handler._handle_write_memory_file(
                {"path": "common_knowledge/valid_file.md", "content": "content"},
            )

        assert "Written to" in result
        assert (ck_dir / "valid_file.md").read_text(encoding="utf-8") == "content"


# ── reference/ read traversal and valid paths ─────────────


class TestReferenceReadTraversal:
    """Path traversal prevention and valid reads for reference/ (read-only)."""

    def test_rejects_traversal_with_dotdot(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)

        with patch("core.paths.get_reference_dir", return_value=ref_dir):
            result = handler._handle_read_memory_file(
                {"path": "reference/../../etc/passwd"},
            )

        assert "PermissionDenied" in result
        assert "traversal" in result.lower()

    def test_allows_valid_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "tech_spec.md").write_text("reference content", encoding="utf-8")

        with patch("core.paths.get_reference_dir", return_value=ref_dir):
            result = handler._handle_read_memory_file(
                {"path": "reference/tech_spec.md"},
            )

        assert result == "reference content"

    def test_allows_nested_valid_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ref_dir = tmp_path / "reference"
        sub = ref_dir / "api"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "endpoints.md").write_text("API docs", encoding="utf-8")

        with patch("core.paths.get_reference_dir", return_value=ref_dir):
            result = handler._handle_read_memory_file(
                {"path": "reference/api/endpoints.md"},
            )

        assert result == "API docs"


# ── reference/ write rejected ─────────────────────────────


class TestReferenceWriteRejected:
    """write_memory_file rejects reference/ prefix (read-only)."""

    def test_rejects_reference_prefix(self, tmp_path: Path):
        handler = _make_handler(tmp_path)
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)

        result = handler._handle_write_memory_file(
            {"path": "reference/tech_spec.md", "content": "attempted write"},
        )

        assert "PermissionDenied" in result
        assert "read-only" in result.lower()
        assert not (ref_dir / "tech_spec.md").exists()


# ── create_anima path traversal ──────────────────────────


class TestCreateAnimaPathTraversal:
    """Path traversal prevention in _handle_create_anima for character_sheet_path."""

    def test_rejects_traversal_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)

        result = handler._handle_create_anima(
            {"character_sheet_path": "../../other_anima/identity.md"},
        )

        assert "PermissionDenied" in result
        assert "character_sheet_path" in result

    def test_rejects_deep_traversal(self, tmp_path: Path):
        handler = _make_handler(tmp_path)

        result = handler._handle_create_anima(
            {"character_sheet_path": "../../../etc/passwd"},
        )

        assert "PermissionDenied" in result

    def test_allows_valid_path(self, tmp_path: Path):
        handler = _make_handler(tmp_path)

        sheet = handler._anima_dir / "character_sheet.md"
        sheet.write_text(
            "# Character: TestChild\n\n## 基本情報\n\n"
            "| 項目 | 設定 |\n|------|------|\n| 英名 | testchild |\n\n"
            "## 人格\n\nテスト人格\n\n## 役割・行動方針\n\nテスト役割\n",
            encoding="utf-8",
        )

        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(exist_ok=True)
        data_dir = tmp_path

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.paths.get_data_dir", return_value=data_dir),
            patch("core.anima_factory.create_from_md") as mock_create,
            patch("cli.commands.init_cmd._register_anima_in_config"),
        ):
            mock_create.return_value = animas_dir / "testchild"
            (animas_dir / "testchild").mkdir(parents=True, exist_ok=True)
            status = animas_dir / "testchild" / "status.json"
            status.write_text(json.dumps({"enabled": True}), encoding="utf-8")

            result = handler._handle_create_anima(
                {"character_sheet_path": "character_sheet.md"},
            )

        assert "created successfully" in result
