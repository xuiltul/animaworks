"""Unit tests for the create_anima tool in core/tooling/handler.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path) -> ToolHandler:
    """Build a minimal ToolHandler with mocked dependencies."""
    anima_dir = tmp_path / "animas" / "boss"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    messenger = MagicMock()

    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )


class TestHandleCreateAnima:
    def test_successful_creation(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        fake_anima_dir = tmp_path / "animas" / "hinata"
        fake_anima_dir.mkdir(parents=True)

        with patch("core.anima_factory.create_from_md", return_value=fake_anima_dir) as mock_create, \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"), \
             patch("core.paths.get_data_dir", return_value=tmp_path), \
             patch("cli.commands.init_cmd._register_anima_in_config"):
            result = handler.handle("create_anima", {"character_sheet_path": str(sheet)})

        assert "hinata" in result
        assert "created successfully" in result
        mock_create.assert_called_once()

    def test_file_not_found(self, tmp_path):
        handler = _make_handler(tmp_path)
        result = handler.handle(
            "create_anima",
            {"character_sheet_path": str(tmp_path / "nonexistent.md")},
        )
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "FileNotFound"

    def test_duplicate_anima(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        with patch(
            "core.anima_factory.create_from_md",
            side_effect=FileExistsError("Anima 'hinata' already exists"),
        ), \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"), \
             patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet), "name": "hinata"},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "AnimaExists"

    def test_invalid_character_sheet(self, tmp_path):
        handler = _make_handler(tmp_path)
        sheet = tmp_path / "sheet.md"
        sheet.write_text("# Character: Hinata", encoding="utf-8")

        with patch(
            "core.anima_factory.create_from_md",
            side_effect=ValueError("Missing required sections: 基本情報"),
        ), \
             patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"), \
             patch("core.paths.get_data_dir", return_value=tmp_path):
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet)},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidCharacterSheet"
