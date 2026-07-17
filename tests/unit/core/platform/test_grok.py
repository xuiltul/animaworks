"""Unit tests for core.platform.grok."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.platform import grok


class TestGrokCliAvailability:
    def test_returns_executable_when_grok_on_path(self):
        with patch("core.platform.grok.shutil.which", return_value="/usr/bin/grok"):
            assert grok.get_grok_executable() == "/usr/bin/grok"
            assert grok.is_grok_cli_available() is True

    def test_returns_none_when_not_on_path(self):
        with patch("core.platform.grok.shutil.which", return_value=None):
            assert grok.get_grok_executable() is None
            assert grok.is_grok_cli_available() is False


class TestGrokAuthentication:
    def test_returns_true_with_auth(self, tmp_path: Path):
        grok_dir = tmp_path / ".grok"
        grok_dir.mkdir(parents=True)
        (grok_dir / "auth.json").write_text('{"access_token":"abc"}', encoding="utf-8")

        with patch("core.platform.grok.grok_config_dir", return_value=grok_dir):
            assert grok.is_grok_authenticated() is True

    def test_returns_false_when_no_auth(self, tmp_path: Path):
        grok_dir = tmp_path / ".grok"

        with patch("core.platform.grok.grok_config_dir", return_value=grok_dir):
            assert grok.is_grok_authenticated() is False

    def test_returns_false_for_empty_auth(self, tmp_path: Path):
        grok_dir = tmp_path / ".grok"
        grok_dir.mkdir(parents=True)
        (grok_dir / "auth.json").write_text("{}", encoding="utf-8")

        with patch("core.platform.grok.grok_config_dir", return_value=grok_dir):
            assert grok.is_grok_authenticated() is False

    def test_returns_false_for_invalid_json(self, tmp_path: Path):
        grok_dir = tmp_path / ".grok"
        grok_dir.mkdir(parents=True)
        (grok_dir / "auth.json").write_text("{not-json", encoding="utf-8")

        with patch("core.platform.grok.grok_config_dir", return_value=grok_dir):
            assert grok.is_grok_authenticated() is False
