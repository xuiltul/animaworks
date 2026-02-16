"""E2E tests for the create_person flow.

Tests the full person creation pipeline from character sheet to filesystem,
including create_from_md, ToolHandler integration, rollback on failure,
and duplicate name handling.  External APIs are mocked but internal
components (person_factory, MemoryManager, ToolHandler) are exercised
as-is.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory import MemoryManager
from core.person_factory import create_from_md
from core.tooling.handler import ToolHandler

# ── Sample character sheets ──────────────────────────────────

FULL_CHARACTER_SHEET = """\
# Character: testperson

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | testperson |
| 日本語名 | テスト太郎 |
| 役職/専門 | テスト担当 |
| 上司 | sakura |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
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
| モデル | claude-sonnet-4-20250514 |
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


# ── Tests ────────────────────────────────────────────────────


class TestCreateFromMdFullFlow:
    """Test the full create_from_md flow with a complete character sheet."""

    def test_person_directory_created_with_all_files(self, data_dir: Path, tmp_path: Path):
        """create_from_md should create person dir with all expected files."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        assert person_dir.exists()
        assert person_dir.name == "testperson"

        # Core files must exist
        expected_files = [
            "identity.md",
            "injection.md",
            "permissions.md",
            "status.json",
            "character_sheet.md",
            "bootstrap.md",
        ]
        for fname in expected_files:
            assert (person_dir / fname).exists(), f"Missing file: {fname}"

    def test_status_json_parsed_correctly(self, data_dir: Path, tmp_path: Path):
        """status.json should contain values parsed from the character sheet."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        status = json.loads(
            (person_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == "sakura"
        assert status["role"] == "worker"
        assert status["execution_mode"] == "autonomous"
        assert status["model"] == "claude-sonnet-4-20250514"
        assert status["credential"] == "anthropic"

    def test_identity_md_populated_from_personality_section(
        self, data_dir: Path, tmp_path: Path
    ):
        """identity.md should contain the content from the character sheet's personality section."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        identity = (person_dir / "identity.md").read_text(encoding="utf-8")
        assert "テスト用の人格設定です" in identity
        assert "明るく元気な性格" in identity

    def test_injection_md_populated_from_role_section(
        self, data_dir: Path, tmp_path: Path
    ):
        """injection.md should contain the content from the role/behavior section."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        injection = (person_dir / "injection.md").read_text(encoding="utf-8")
        assert "テスト業務を担当します" in injection
        assert "品質管理に注力します" in injection

    def test_character_sheet_preserved(self, data_dir: Path, tmp_path: Path):
        """The original character sheet should be saved as character_sheet.md."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        saved_sheet = (person_dir / "character_sheet.md").read_text(encoding="utf-8")
        assert saved_sheet == FULL_CHARACTER_SHEET

    def test_runtime_subdirectories_created(self, data_dir: Path, tmp_path: Path):
        """All runtime subdirectories should be created."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

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
            assert (person_dir / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_state_files_initialized(self, data_dir: Path, tmp_path: Path):
        """State files should be initialized with default content."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        current_task = person_dir / "state" / "current_task.md"
        assert current_task.exists()
        assert current_task.read_text(encoding="utf-8") == "status: idle\n"

        pending = person_dir / "state" / "pending.md"
        assert pending.exists()
        assert pending.read_text(encoding="utf-8") == ""

    def test_name_extracted_from_eimei_field(self, data_dir: Path, tmp_path: Path):
        """Person name should be extracted from the table's English name field."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        assert person_dir.name == "testperson"

    def test_explicit_name_overrides_extraction(self, data_dir: Path, tmp_path: Path):
        """An explicit name parameter should override the name extracted from the sheet."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path, name="custom-name")

        assert person_dir.name == "custom-name"

    def test_no_supervisor_sets_empty_string(self, data_dir: Path, tmp_path: Path):
        """When supervisor is '(なし)', status.json should have an empty supervisor."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        status = json.loads(
            (person_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == ""


class TestCreateFromMdOmittedSections:
    """Test create_from_md with only required sections (optional sections omitted)."""

    def test_default_template_files_preserved(self, data_dir: Path, tmp_path: Path):
        """When optional sections are omitted, default template files should be kept."""
        persons_dir = data_dir / "persons"
        # The MINIMAL_CHARACTER_SHEET has no permissions or heartbeat/cron sections
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        # heartbeat.md and cron.md should exist with template defaults
        heartbeat = person_dir / "heartbeat.md"
        assert heartbeat.exists()
        heartbeat_content = heartbeat.read_text(encoding="utf-8")
        # Template defaults have {name} replaced with the person name
        assert "Heartbeat" in heartbeat_content or "チェックリスト" in heartbeat_content

        cron = person_dir / "cron.md"
        assert cron.exists()
        cron_content = cron.read_text(encoding="utf-8")
        assert "Cron" in cron_content or "毎朝" in cron_content

    def test_permissions_md_keeps_default_content(self, data_dir: Path, tmp_path: Path):
        """permissions.md should retain template defaults when not overridden."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, MINIMAL_CHARACTER_SHEET)

        person_dir = create_from_md(persons_dir, sheet_path)

        permissions = person_dir / "permissions.md"
        assert permissions.exists()
        perm_content = permissions.read_text(encoding="utf-8")
        # Should contain the default template permissions (e.g., tool list)
        assert "Permissions" in perm_content or "ツール" in perm_content


class TestDuplicatePersonNameError:
    """Test that creating a person with an existing name raises FileExistsError."""

    def test_duplicate_name_raises_file_exists_error(
        self, data_dir: Path, tmp_path: Path
    ):
        """Creating two persons with the same name should raise FileExistsError."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # Create the first person
        create_from_md(persons_dir, sheet_path)

        # Re-write the sheet (same content) and try again
        sheet_path2 = _write_character_sheet(
            tmp_path, FULL_CHARACTER_SHEET, filename="sheet2.md"
        )

        with pytest.raises(FileExistsError, match="testperson"):
            create_from_md(persons_dir, sheet_path2)

    def test_duplicate_name_with_explicit_name(
        self, data_dir: Path, tmp_path: Path
    ):
        """Duplicate explicit name should also raise FileExistsError."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        create_from_md(persons_dir, sheet_path, name="dupname")

        sheet_path2 = _write_character_sheet(
            tmp_path, MINIMAL_CHARACTER_SHEET, filename="sheet2.md"
        )

        with pytest.raises(FileExistsError, match="dupname"):
            create_from_md(persons_dir, sheet_path2, name="dupname")


class TestRollbackOnFailure:
    """Test that partial directories are cleaned up when creation fails."""

    def test_rollback_removes_directory_on_status_json_failure(
        self, data_dir: Path, tmp_path: Path
    ):
        """If _create_status_json fails, the person directory should be removed."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        with patch(
            "core.person_factory._create_status_json",
            side_effect=RuntimeError("Simulated status.json failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated status.json failure"):
                create_from_md(persons_dir, sheet_path)

        # Person directory should have been rolled back
        assert not (persons_dir / "testperson").exists()

    def test_rollback_removes_directory_on_apply_defaults_failure(
        self, data_dir: Path, tmp_path: Path
    ):
        """If _apply_defaults_from_sheet fails, the directory should be removed."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        with patch(
            "core.person_factory._apply_defaults_from_sheet",
            side_effect=OSError("Simulated write failure"),
        ):
            with pytest.raises(OSError, match="Simulated write failure"):
                create_from_md(persons_dir, sheet_path)

        assert not (persons_dir / "testperson").exists()

    def test_can_recreate_after_rollback(self, data_dir: Path, tmp_path: Path):
        """After a rollback, the same name can be used again successfully."""
        persons_dir = data_dir / "persons"
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # First attempt fails
        with patch(
            "core.person_factory._create_status_json",
            side_effect=RuntimeError("fail"),
        ):
            with pytest.raises(RuntimeError):
                create_from_md(persons_dir, sheet_path)

        # Second attempt should succeed (no leftover directory)
        person_dir = create_from_md(persons_dir, sheet_path)
        assert person_dir.exists()
        assert person_dir.name == "testperson"


class TestCreatePersonToolHandler:
    """Test create_person through the ToolHandler dispatch (tool integration)."""

    def test_create_person_via_tool_handler(self, data_dir: Path, tmp_path: Path):
        """ToolHandler.handle('create_person', ...) should create a person."""
        persons_dir = data_dir / "persons"

        # Set up a caller person directory (the person invoking create_person)
        caller_dir = persons_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "identity.md").write_text(
            "# Caller\nI am the caller.", encoding="utf-8"
        )
        (caller_dir / "permissions.md").write_text(
            "# Permissions\nAll tools allowed.", encoding="utf-8"
        )
        for sub in ["episodes", "knowledge", "procedures", "skills", "state",
                     "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(person_dir=caller_dir, memory=memory)

        # Write the character sheet file
        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # Mock the config registration (imported inline in the handler method)
        with patch("cli.commands.init_cmd._register_person_in_config"):
            result = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet_path)},
            )

        assert "testperson" in result
        assert "created successfully" in result

        # Verify the person was actually created
        new_person_dir = persons_dir / "testperson"
        assert new_person_dir.exists()
        assert (new_person_dir / "character_sheet.md").exists()
        assert (new_person_dir / "status.json").exists()
        assert (new_person_dir / "identity.md").exists()

    def test_create_person_tool_missing_sheet(self, data_dir: Path, tmp_path: Path):
        """ToolHandler should return error when character sheet path doesn't exist."""
        persons_dir = data_dir / "persons"

        caller_dir = persons_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "identity.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state",
                     "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(person_dir=caller_dir, memory=memory)

        result = handler.handle(
            "create_person",
            {"character_sheet_path": str(tmp_path / "nonexistent.md")},
        )

        assert "FileNotFound" in result or "not found" in result.lower()

    def test_create_person_tool_duplicate_returns_error(
        self, data_dir: Path, tmp_path: Path
    ):
        """ToolHandler should return error (not raise) for duplicate person name."""
        persons_dir = data_dir / "persons"

        caller_dir = persons_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "identity.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state",
                     "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(person_dir=caller_dir, memory=memory)

        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        # First creation succeeds
        with patch("cli.commands.init_cmd._register_person_in_config"):
            result1 = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet_path)},
            )
        assert "created successfully" in result1

        # Second creation should return an error string, not raise
        sheet_path2 = _write_character_sheet(
            tmp_path, FULL_CHARACTER_SHEET, filename="sheet2.md"
        )
        with patch("cli.commands.init_cmd._register_person_in_config"):
            result2 = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet_path2)},
            )
        assert "PersonExists" in result2 or "already exists" in result2.lower()

    def test_create_person_tool_with_explicit_name(
        self, data_dir: Path, tmp_path: Path
    ):
        """ToolHandler should support the optional 'name' parameter."""
        persons_dir = data_dir / "persons"

        caller_dir = persons_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "identity.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state",
                     "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(person_dir=caller_dir, memory=memory)

        sheet_path = _write_character_sheet(tmp_path, FULL_CHARACTER_SHEET)

        with patch("cli.commands.init_cmd._register_person_in_config"):
            result = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet_path), "name": "custom-worker"},
            )

        assert "custom-worker" in result
        assert (persons_dir / "custom-worker").exists()

    def test_create_person_tool_relative_path_resolved(
        self, data_dir: Path, tmp_path: Path
    ):
        """Relative character_sheet_path should be resolved relative to person_dir."""
        persons_dir = data_dir / "persons"

        caller_dir = persons_dir / "caller"
        caller_dir.mkdir(parents=True)
        (caller_dir / "identity.md").write_text("# Caller", encoding="utf-8")
        (caller_dir / "permissions.md").write_text("# Perms", encoding="utf-8")
        for sub in ["episodes", "knowledge", "procedures", "skills", "state",
                     "shortterm", "shortterm/archive"]:
            (caller_dir / sub).mkdir(parents=True, exist_ok=True)

        # Write the character sheet inside the caller's directory
        sheet_path = caller_dir / "new_hire.md"
        sheet_path.write_text(MINIMAL_CHARACTER_SHEET, encoding="utf-8")

        memory = MemoryManager(caller_dir)
        handler = ToolHandler(person_dir=caller_dir, memory=memory)

        with patch("cli.commands.init_cmd._register_person_in_config"):
            result = handler.handle(
                "create_person",
                {"character_sheet_path": "new_hire.md"},
            )

        assert "minimal" in result
        assert "created successfully" in result
        assert (persons_dir / "minimal").exists()


class TestCharacterSheetValidation:
    """Test that invalid character sheets are properly rejected."""

    def test_missing_basic_info_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without basic info section should raise ValueError."""
        persons_dir = data_dir / "persons"
        bad_sheet = """\
# キャラクターシート: テスト

## 人格 (→ identity.md)

テスト用の人格設定です。

## 役割・行動方針 (→ injection.md)

テスト業務を担当します。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="基本情報"):
            create_from_md(persons_dir, sheet_path, name="badperson")

    def test_missing_personality_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without personality section should raise ValueError."""
        persons_dir = data_dir / "persons"
        bad_sheet = """\
# キャラクターシート: テスト

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | badperson |

## 役割・行動方針 (→ injection.md)

テスト業務を担当します。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="人格"):
            create_from_md(persons_dir, sheet_path)

    def test_missing_injection_section(self, data_dir: Path, tmp_path: Path):
        """Character sheet without injection section should raise ValueError."""
        persons_dir = data_dir / "persons"
        bad_sheet = """\
# キャラクターシート: テスト

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | badperson |

## 人格 (→ identity.md)

テスト用の人格設定です。
"""
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError, match="役割・行動方針"):
            create_from_md(persons_dir, sheet_path)

    def test_validation_prevents_directory_creation(
        self, data_dir: Path, tmp_path: Path
    ):
        """An invalid sheet should not create any directory at all."""
        persons_dir = data_dir / "persons"
        bad_sheet = "# Just a heading, nothing valid"
        sheet_path = _write_character_sheet(tmp_path, bad_sheet)

        with pytest.raises(ValueError):
            create_from_md(persons_dir, sheet_path, name="should-not-exist")

        assert not (persons_dir / "should-not-exist").exists()
