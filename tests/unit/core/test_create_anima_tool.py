"""Unit tests for the create_anima tool in core/tooling/handler.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from core.tooling.handler import ToolHandler

_SHEET_CONTENT = """\
# Character: hinata

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | hinata |

## 人格

test

## 役割・行動方針

test
"""


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

        with (
            patch("core.anima_factory.create_from_md", return_value=fake_anima_dir) as mock_create,
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch("cli.commands.init_cmd._register_anima_in_config"),
        ):
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

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=FileExistsError("Anima 'hinata' already exists"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
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

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=ValueError("Missing required sections: 基本情報"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
        ):
            result = handler.handle(
                "create_anima",
                {"character_sheet_path": str(sheet)},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidCharacterSheet"


class TestCreateAnimaErofsFallback:
    """EROFS / sandbox write failure falls back to internal API."""

    def test_erofs_falls_back_to_server_api(self, tmp_path, monkeypatch):
        handler = _make_handler(tmp_path)
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://server.test:18500")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "anima_dir": str(tmp_path / "animas" / "hinata"),
        }

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=OSError(30, "Read-only file system"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch("httpx.post", return_value=mock_resp) as mock_post,
        ):
            result = handler.handle(
                "create_anima",
                {
                    "character_sheet_content": _SHEET_CONTENT,
                    "name": "hinata",
                    "supervisor": "boss",
                },
            )

        assert "hinata" in result
        assert "created successfully" in result
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://server.test:18500/api/internal/anima/create"
        payload = call_kwargs[1]["json"]
        assert payload["character_sheet_content"] == _SHEET_CONTENT
        assert payload["name"] == "hinata"
        assert payload["supervisor"] == "boss"
        assert payload["calling_anima"] == "boss"
        assert "character_sheet_path" not in payload

    def test_erofs_server_409_returns_anima_exists(self, tmp_path, monkeypatch):
        handler = _make_handler(tmp_path)
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://server.test:18500")

        mock_resp = MagicMock()
        mock_resp.status_code = 409
        mock_resp.json.return_value = {"detail": "Anima 'hinata' already exists"}
        mock_resp.text = "Anima 'hinata' already exists"

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=OSError(30, "Read-only file system"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("httpx.post", return_value=mock_resp),
        ):
            result = handler.handle(
                "create_anima",
                {"character_sheet_content": _SHEET_CONTENT, "name": "hinata"},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "AnimaExists"

    def test_erofs_server_422_returns_invalid_sheet(self, tmp_path, monkeypatch):
        handler = _make_handler(tmp_path)
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://server.test:18500")

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"detail": "Missing required sections"}
        mock_resp.text = "Missing required sections"

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=OSError(30, "Read-only file system"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("httpx.post", return_value=mock_resp),
        ):
            result = handler.handle(
                "create_anima",
                {"character_sheet_content": "bad"},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "InvalidCharacterSheet"

    def test_erofs_connect_error_includes_original_oserror(self, tmp_path, monkeypatch):
        handler = _make_handler(tmp_path)
        monkeypatch.setenv("ANIMAWORKS_SERVER_URL", "http://server.test:18500")

        with (
            patch(
                "core.anima_factory.create_from_md",
                side_effect=OSError(30, "Read-only file system"),
            ),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch(
                "httpx.post",
                side_effect=httpx.ConnectError("connection refused"),
            ),
        ):
            result = handler.handle(
                "create_anima",
                {"character_sheet_content": _SHEET_CONTENT, "name": "hinata"},
            )

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Read-only file system" in parsed["message"]
        assert "connection refused" in parsed["message"].lower() or "unreachable" in parsed["message"].lower()
