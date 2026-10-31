# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the create_anima flow.

Tests the full anima creation pipeline from character sheet to filesystem,
including create_from_md, ToolHandler integration, rollback on failure,
and duplicate name handling.  External APIs are mocked but internal
components (anima_factory, MemoryManager, ToolHandler) are exercised
as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.anima.factory import create_from_md
from core.memory import MemoryManager
from core.tooling.handler import ToolHandler

# ── Sample character sheets ──────────────────────────────────

FULL_CHARACTER_SHEET = """\
# Character: testanima

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testanima |
| 日本語名 | テスト太郎 |
| 役職/専門 | テスト担当 |
| 上司 | sakura |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-5-5 |
| credential | anthropic |

## 人格

テスト用の人格設定です。明るく元気な性格。

## 役割・行動方針

テスト業務を担当します。品質管理に注力します。
"""

MINIMAL_CHARACTER_SHEET = """\
# Character: minimal

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | minimal |
| 日本語名 | ミニマル |
| 役職/専門 | 汎用担当 |
| 上司 | (なし) |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-5-5 |
| credential | anthropic |

## 人格

シンプルな性格です。

## 役割・行動方針

汎用的な業務を担当します。
"""


# ── Helpers ──────────────────────────────────────────────────


def _write_character_sheet(directory: Path, content: str, filename: str = "sheet.md") -> Path:
    """Write a character sheet file and return its path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


def _root_api_response(anima_dir: Path, *, status_code: int = 200, detail: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"anima_dir": str(anima_dir)} if status_code < 400 else {"detail": detail}
    response.text = detail
    return response


# ── Tests ────────────────────────────────────────────────────


class TestCreateFromMdFullFlow:
    """Test the full create_from_md flow with a complete character sheet."""

    def test_anima_directory_created_with_all_files(self, data_dir: Path, tmp_path: Path):
        """create_from_md should create anima dir with all expected files."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        assert anima_dir.exists()
        assert anima_dir.name == "testanima"

        # Core files must exist
        expected_files = [
            "IDENTITY.md",
            "AGENTS.md",
            "permissions.json",
            "status.json",
            "character_sheet.md",
            "BOOTSTRAP.md",
        ]
        for fname in expected_files:
            assert (anima_dir / fname).exists(), f"Missing file: {fname}"

    def test_status_json_parsed_correctly(self, data_dir: Path, tmp_path: Path):
        """status.json should contain values parsed from the character sheet."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == "sakura"
        assert status["role"] == "general"
        # "autonomous" is the sheet's prose label, not a mode; pinning it would
        # force Mode A (API key) on Claude subscription hires.
        assert "execution_mode" not in status
        assert status["model"] == "claude-sonnet-5-5"
        assert status["credential"] == "anthropic"

    def test_soul_md_populated_from_personality_section(self, data_dir: Path, tmp_path: Path):
        """SOUL.md should contain the content from the character sheet's personality section."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        soul = (anima_dir / "SOUL.md").read_text(encoding="utf-8")
        assert "テスト用の人格設定です" in soul
        assert "明るく元気な性格" in soul

    def test_injection_md_populated_from_role_section(self, data_dir: Path, tmp_path: Path):
        """AGENTS.md should contain the content from the role/behavior section."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        injection = (anima_dir / "AGENTS.md").read_text(encoding="utf-8")
        assert "テスト業務を担当します" in injection
        assert "品質管理に注力します" in injection

    def test_character_sheet_preserved(self, data_dir: Path, tmp_path: Path):
        """The original character sheet should be saved as character_sheet.md."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        saved_sheet = (anima_dir / "character_sheet.md").read_text(encoding="utf-8")
        assert saved_sheet == FULL_CHARACTER_SHEET

    def test_runtime_subdirectories_created(self, data_dir: Path, tmp_path: Path):
        """All runtime subdirectories should be created."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        expected_subdirs = [
            "episodes",
            "knowledge",
            "procedures",
            "skills",
            "state",
            "shortterm",
            "shortterm/archive",
        ]
        for subdir in expected_subdirs:
            assert (anima_dir / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_state_files_initialized(self, data_dir: Path, tmp_path: Path):
        """State files should be initialized with default content."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        current_state = anima_dir / "state" / "current_state.md"
        assert current_state.exists()
        assert current_state.read_text(encoding="utf-8") == "status: idle\n"

        pending = anima_dir / "state" / "pending.md"
        assert pending.exists()
        assert pending.read_text(encoding="utf-8") == ""

    def test_name_extracted_from_eimei_field(self, data_dir: Path, tmp_path: Path):
        """Anima name should be extracted from the table's English name field."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        assert anima_dir.name == "testanima"

    def test_explicit_name_overrides_extraction(self, data_dir: Path, tmp_path: Path):
        """An explicit name parameter should override the name extracted from the sheet."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path, name="custom-name")

        assert anima_dir.name == "custom-name"

    def test_no_supervisor_sets_empty_string(self, data_dir: Path, tmp_path: Path):
        """When supervisor is '(なし)', status.json should have an empty supervisor."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["supervisor"] == ""


class TestCreateFromMdOmittedSections:
    """Test create_from_md with only required sections (optional sections omitted)."""

    def test_default_template_files_preserved(self, data_dir: Path, tmp_path: Path):
        """When optional sections are omitted, default template files should be kept."""
        animas_dir = data_dir / "animas"
        # The MINIMAL_CHARACTER_SHEET has no permissions or heartbeat/cron sections
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        # HEARTBEAT.md and cron.md should exist with template defaults
        heartbeat = anima_dir / "HEARTBEAT.md"
        assert heartbeat.exists()
        heartbeat_content = heartbeat.read_text(encoding="utf-8")
        # Template defaults have {name} replaced with the anima name
        assert "Heartbeat" in heartbeat_content or "チェックリスト" in heartbeat_content

        cron = anima_dir / "cron.md"
        assert cron.exists()
        cron_content = cron.read_text(encoding="utf-8")
        assert "Cron" in cron_content or "毎朝" in cron_content

    def test_permissions_json_keeps_default_content(self, data_dir: Path, tmp_path: Path):
        """permissions.json should retain template defaults when not overridden."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        anima_dir = create_from_md(animas_dir, sheet_path)

        permissions = anima_dir / "permissions.json"
        assert permissions.exists()
        data = json.loads(permissions.read_text(encoding="utf-8"))
        assert data.get("version") == 1
        assert data.get("file_roots") == ["/"]


class TestDuplicateAnimaNameError:
    """Test that creating an anima with an existing name raises FileExistsError."""

    def test_duplicate_name_raises_file_exists_error(self, data_dir: Path, tmp_path: Path):
        """Creating two animas with the same name should raise FileExistsError."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # Create the first anima
        create_from_md(animas_dir, sheet_path)

        # Re-write the sheet (same content) and try again
        sheet_path2 = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET, filename="sheet2.md")

        with pytest.raises(FileExistsError, match="testanima"):
            create_from_md(animas_dir, sheet_path2)

    def test_duplicate_name_with_explicit_name(self, data_dir: Path, tmp_path: Path):
        """Duplicate explicit name should also raise FileExistsError."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        create_from_md(animas_dir, sheet_path, name="dupname")

        sheet_path2 = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET, filename="sheet2.md")

        with pytest.raises(FileExistsError, match="dupname"):
            create_from_md(animas_dir, sheet_path2, name="dupname")


class TestRollbackOnFailure:
    """Test that partial directories are cleaned up when creation fails."""

    def test_rollback_removes_directory_on_status_json_failure(self, data_dir: Path, tmp_path: Path):
        """If _create_status_json fails, the anima directory should be removed."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        with (
            patch(
                "core.anima.factory._create_status_json",
                side_effect=RuntimeError("Simulated status.json failure"),
            ),
            pytest.raises(RuntimeError, match="Simulated status.json failure"),
        ):
            create_from_md(animas_dir, sheet_path)

        # Anima directory should have been rolled back
        assert not (animas_dir / "testanima").exists()

    def test_rollback_removes_directory_on_apply_defaults_failure(self, data_dir: Path, tmp_path: Path):
        """If _apply_defaults_from_sheet fails, the directory should be removed."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        with (
            patch(
                "core.anima.factory._apply_defaults_from_sheet",
                side_effect=OSError("Simulated write failure"),
            ),
            pytest.raises(OSError, match="Simulated write failure"),
        ):
            create_from_md(animas_dir, sheet_path)

        assert not (animas_dir / "testanima").exists()

    def test_can_recreate_after_rollback(self, data_dir: Path, tmp_path: Path):
        """After a rollback, the same name can be used again successfully."""
        animas_dir = data_dir / "animas"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # First attempt fails
        with (
            patch(
                "core.anima.factory._create_status_json",
                side_effect=RuntimeError("fail"),
            ),
            pytest.raises(RuntimeError),
        ):
            create_from_md(animas_dir, sheet_path)

        # Second attempt should succeed (no leftover directory)
        anima_dir = create_from_md(animas_dir, sheet_path)
        assert anima_dir.exists()
        assert anima_dir.name == "testanima"


class TestCreateAnimaToolHandler:
    """Test the ToolHandler's root-API create_anima contract."""

    # Anima creation became root-owned with S2b (72e47cf4).

    def test_create_anima_via_tool_handler(self, data_dir: Path, tmp_path: Path):
        """ToolHandler.handle('create_anima', ...) should create an anima."""
        animas_dir = data_dir / "animas"

        # Set up a caller anima directory (the anima invoking create_anima)
        caller_dir = animas_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "IDENTITY.md").write_text("# Caller\nI am the caller.", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Permissions\nAll tools allowed.", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(anima_dir=caller_dir, memory=memory)

        # Write the character sheet file
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        expected_anima_dir = animas_dir / "testanima"
        with patch(
            "core.host_api.host_api.post",
            return_value=_root_api_response(expected_anima_dir),
        ) as mock_post:
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet_path)},
            )

        assert "testanima" in result
        assert "created successfully" in result
        mock_post.assert_called_once_with(
            "/api/internal/anima/create",
            json={
                "calling_anima": "caller",
                "character_sheet_content": FULL_CHARACTER_SHEET,
            },
            timeout=60.0,
        )
        # The worker sends creation to the root; it must not write the anima locally.
        assert not expected_anima_dir.exists()

    def test_create_anima_tool_missing_sheet(self, data_dir: Path, tmp_path: Path):
        """ToolHandler should return error when character sheet path doesn't exist."""
        animas_dir = data_dir / "animas"

        caller_dir = animas_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "IDENTITY.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(anima_dir=caller_dir, memory=memory)

        result = handler.handle(
            "create_anima",
            {"character_sheet_path": str(tmp_path / "nonexistent.md")},
        )

        assert "FileNotFound" in result or "not found" in result.lower()

    def test_create_anima_tool_duplicate_returns_error(self, data_dir: Path, tmp_path: Path):
        """ToolHandler should return error (not raise) for duplicate anima name."""
        animas_dir = data_dir / "animas"

        caller_dir = animas_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "IDENTITY.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(anima_dir=caller_dir, memory=memory)

        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        expected_anima_dir = animas_dir / "testanima"
        responses = [
            _root_api_response(expected_anima_dir),
            _root_api_response(expected_anima_dir, status_code=409, detail="Anima already exists"),
        ]
        with patch("core.host_api.host_api.post", side_effect=responses) as mock_post:
            result1 = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet_path)},
            )
            assert "created successfully" in result1

            # A root-side duplicate is translated to an AnimaExists tool error.
            sheet_path2 = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET, filename="sheet2.md")
            result2 = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet_path2)},
            )

        assert "AnimaExists" in result2 or "already exists" in result2.lower()
        assert mock_post.call_count == 2
        assert not expected_anima_dir.exists()

    def test_create_anima_tool_with_explicit_name(self, data_dir: Path, tmp_path: Path):
        """ToolHandler should support the optional 'name' parameter."""
        animas_dir = data_dir / "animas"

        caller_dir = animas_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "IDENTITY.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(anima_dir=caller_dir, memory=memory)

        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        expected_anima_dir = animas_dir / "custom-worker"
        with patch(
            "core.host_api.host_api.post",
            return_value=_root_api_response(expected_anima_dir),
        ) as mock_post:
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet_path), "name": "custom-worker"},
            )

        assert "custom-worker" in result
        assert mock_post.call_args.kwargs["json"]["name"] == "custom-worker"
        assert not expected_anima_dir.exists()

    def test_create_anima_tool_relative_path_resolved(self, data_dir: Path, tmp_path: Path):
        """Relative character_sheet_path should be resolved relative to anima_dir."""
        animas_dir = data_dir / "animas"

        caller_dir = animas_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "IDENTITY.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        # Write the character sheet inside the caller's directory
        sheet_path = caller_dir / "new_hire.md"
        sheet_path.write_text(MINIMAL_CHARACTER_SHEET, encoding="utf-8")

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(anima_dir=caller_dir, memory=memory)

        expected_anima_dir = animas_dir / "minimal"
        with patch(
            "core.host_api.host_api.post",
            return_value=_root_api_response(expected_anima_dir),
        ) as mock_post:
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": "new_hire.md"},
            )

        assert "minimal" in result
        assert "created successfully" in result
        payload = mock_post.call_args.kwargs["json"]
        assert payload["character_sheet_content"] == MINIMAL_CHARACTER_SHEET
        assert payload["calling_anima"] == "caller"
        assert "character_sheet_path" not in payload
        assert not expected_anima_dir.exists()


class TestCharacterSheetValidation:
    """Test that invalid character sheets are properly rejected."""

    def test_missing_basic_info_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without basic info section should raise ValueError."""
        animas_dir = data_dir / "animas"
        bad_sheet = """\
# キャラクターシート: テスト

## 人格 (→ IDENTITY.md)

テスト用の人格設定です。

## 役割・行動方針 (→ AGENTS.md)

テスト業務を担当します。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="基本情報"):
            create_from_md(animas_dir, sheet_path, name="badanima")

    def test_missing_personality_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without personality section should raise ValueError."""
        animas_dir = data_dir / "animas"
        bad_sheet = """\
# キャラクターシート: テスト

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | badanima |

## 役割・行動方針 (→ AGENTS.md)

テスト業務を担当します。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="人格"):
            create_from_md(animas_dir, sheet_path)

    def test_missing_injection_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without injection section should raise ValueError."""
        animas_dir = data_dir / "animas"
        bad_sheet = """\
# キャラクターシート: テスト

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | badanima |

## 人格 (→ IDENTITY.md)

テスト用の人格設定です。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="役割・行動方針"):
            create_from_md(animas_dir, sheet_path)

    def test_validation_prevents_directory_creation(self, data_dir: Path, tmp_path: Path):
        """An invalid sheet should not create any directory at all."""
        animas_dir = data_dir / "animas"
        bad_sheet = "# Just a heading, nothing valid"
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError):
            create_from_md(animas_dir, sheet_path, name="should-not-exist")

        assert not (animas_dir / "should-not-exist").exists()
