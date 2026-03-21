"""Unit tests for core.platform.cursor."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.platform import cursor


class TestCursorAgentAvailability:
    def test_returns_true_when_agent_on_path(self):
        def fake_which(name: str) -> str | None:
            return "/usr/bin/agent" if name == "agent" else None

        with patch("core.platform.cursor.shutil.which", side_effect=fake_which):
            assert cursor.is_cursor_agent_available() is True

    def test_returns_true_when_cursor_agent_on_path(self):
        def fake_which(name: str) -> str | None:
            return "/usr/bin/cursor-agent" if name == "cursor-agent" else None

        with patch("core.platform.cursor.shutil.which", side_effect=fake_which):
            assert cursor.is_cursor_agent_available() is True

    def test_returns_false_when_no_binary(self):
        with patch("core.platform.cursor.shutil.which", return_value=None):
            assert cursor.is_cursor_agent_available() is False


class TestCursorAgentAuthentication:
    def test_returns_false_when_dir_missing(self, tmp_path: Path):
        auth_dir = tmp_path / ".cursor-agent"
        with patch("core.platform.cursor.cursor_agent_auth_dir", return_value=auth_dir):
            assert cursor.is_cursor_agent_authenticated() is False

    def test_returns_true_for_valid_auth_file(self, tmp_path: Path):
        auth_dir = tmp_path / ".cursor-agent"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text('{"access_token":"abc"}', encoding="utf-8")

        with patch("core.platform.cursor.cursor_agent_auth_dir", return_value=auth_dir):
            assert cursor.is_cursor_agent_authenticated() is True

    def test_returns_false_for_empty_json(self, tmp_path: Path):
        auth_dir = tmp_path / ".cursor-agent"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text("{}", encoding="utf-8")

        with patch("core.platform.cursor.cursor_agent_auth_dir", return_value=auth_dir):
            assert cursor.is_cursor_agent_authenticated() is False

    def test_returns_false_for_invalid_json(self, tmp_path: Path):
        auth_dir = tmp_path / ".cursor-agent"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text("{not-json", encoding="utf-8")

        with patch("core.platform.cursor.cursor_agent_auth_dir", return_value=auth_dir):
            assert cursor.is_cursor_agent_authenticated() is False

    def test_returns_false_when_no_auth_file(self, tmp_path: Path):
        auth_dir = tmp_path / ".cursor-agent"
        auth_dir.mkdir(parents=True)

        with patch("core.platform.cursor.cursor_agent_auth_dir", return_value=auth_dir):
            assert cursor.is_cursor_agent_authenticated() is False
