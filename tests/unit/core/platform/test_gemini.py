"""Unit tests for core.platform.gemini."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.platform import gemini


class TestGeminiCliAvailability:
    def test_returns_true_when_gemini_on_path(self):
        with patch("core.platform.gemini.shutil.which", return_value="/usr/bin/gemini"):
            assert gemini.is_gemini_cli_available() is True

    def test_returns_false_when_not_on_path(self):
        with patch("core.platform.gemini.shutil.which", return_value=None):
            assert gemini.is_gemini_cli_available() is False


class TestGeminiAuthentication:
    def test_returns_true_with_api_key_env(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "sk-test"}, clear=True):
            assert gemini.is_gemini_authenticated() is True

    def test_returns_true_with_oauth_creds(self, tmp_path: Path):
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir(parents=True)
        (gemini_dir / "oauth_creds.json").write_text('{"refresh_token":"abc"}', encoding="utf-8")

        with (
            patch("core.platform.gemini.gemini_config_dir", return_value=gemini_dir),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert gemini.is_gemini_authenticated() is True

    def test_returns_false_when_no_auth(self, tmp_path: Path):
        gemini_dir = tmp_path / ".gemini"

        with (
            patch("core.platform.gemini.gemini_config_dir", return_value=gemini_dir),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert gemini.is_gemini_authenticated() is False

    def test_returns_false_for_empty_oauth(self, tmp_path: Path):
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir(parents=True)
        (gemini_dir / "oauth_creds.json").write_text("{}", encoding="utf-8")

        with (
            patch("core.platform.gemini.gemini_config_dir", return_value=gemini_dir),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert gemini.is_gemini_authenticated() is False

    def test_returns_false_for_invalid_json(self, tmp_path: Path):
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir(parents=True)
        (gemini_dir / "oauth_creds.json").write_text("{not-json", encoding="utf-8")

        with (
            patch("core.platform.gemini.gemini_config_dir", return_value=gemini_dir),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert gemini.is_gemini_authenticated() is False
