"""Unit tests for the create_person tool in core/tooling/handler.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path) -> ToolHandler:
    """Build a minimal ToolHandler with mocked dependencies."""
    person_dir = tmp_path / "persons" / "boss"
    person_dir.mkdir(parents=True)
    (person_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    messenger = MagicMock()

    return ToolHandler(
        person_dir=person_dir,
        memory=memory,
        messenger=messenger,
    )


class TestHandleCreatePerson:
    def test_successful_creation(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        fake_person_dir = tmp_path / "persons" / "hinata"
        fake_person_dir.mkdir(parents=True)

        with patch("core.person_factory.create_from_md", return_value=fake_person_dir) as mock_create, \
             patch("core.paths.get_persons_dir", return_value=tmp_path / "persons"), \
             patch("core.paths.get_data_dir", return_value=tmp_path), \
             patch("cli.commands.init_cmd._register_person_in_config"):
            result = handler.handle("create_person", {"character_sheet_path": str(sheet)})

        assert "hinata" in result
        assert "created successfully" in result
        mock_create.assert_called_once()

    def test_file_not_found(self, tmp_path):
        handler = _make_handler(tmp_path)
        result = handler.handle(
            "create_person",
            {"character_sheet_path": str(tmp_path / "nonexistent.md")},
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "FileNotFound"

    def test_duplicate_person(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        with patch(
            "core.person_factory.create_from_md",
            side_effect=FileExistsError("Person 'hinata' already exists"),
        ), \
             patch("core.paths.get_persons_dir", return_value=tmp_path / "persons"), \
             patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet), "name": "hinata"},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "PersonExists"

    def test_invalid_character_sheet(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        with patch(
            "core.person_factory.create_from_md",
            side_effect=ValueError("Missing required sections: 基本情報"),
        ), \
             patch("core.paths.get_persons_dir", return_value=tmp_path / "persons"), \
             patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle(
                "create_person",
                {"character_sheet_path": str(sheet)},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidCharacterSheet"
