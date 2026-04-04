"""Unit tests for core.platform.codex."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.platform import codex


class TestDefaultHomeDir:
    def test_prefers_home_env(self):
        with patch.dict("os.environ", {"HOME": "/tmp/home", "USERPROFILE": "C:/Users/test"}, clear=True):
            assert codex.default_home_dir() == "/tmp/home"

    def test_falls_back_to_userprofile(self):
        with patch.dict("os.environ", {"USERPROFILE": "C:/Users/test"}, clear=True):
            assert codex.default_home_dir() == "C:/Users/test"


class TestCodexLoginAvailability:
    def test_returns_false_when_auth_missing(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        with (
            patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)),
            patch("core.platform.codex._run_codex_command", return_value=None),
        ):
            assert codex.is_codex_login_available() is False

    def test_returns_true_for_valid_auth_file(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        auth_path = tmp_path / ".codex" / "auth.json"
        auth_path.parent.mkdir(parents=True)
        auth_path.write_text('{"access_token":"abc"}', encoding="utf-8")

        with patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)):
            assert codex.codex_auth_path() == auth_path
            assert codex.is_codex_login_available() is True

    def test_returns_false_for_invalid_json(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        auth_path = tmp_path / ".codex" / "auth.json"
        auth_path.parent.mkdir(parents=True)
        auth_path.write_text("{not-json", encoding="utf-8")

        with (
            patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)),
            patch("core.platform.codex._run_codex_command", return_value=None),
        ):
            assert codex.is_codex_login_available() is False

    def test_finds_embedded_codex_executable(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        ext_dir = tmp_path / ".antigravity" / "extensions" / "openai.chatgpt-1" / "bin" / "windows-x86_64"
        ext_dir.mkdir(parents=True)
        exe = ext_dir / "codex.exe"
        exe.write_text("", encoding="utf-8")

        with (
            patch("shutil.which", return_value=None),
            patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)),
            patch("core.platform.codex._is_usable_codex_executable", return_value=True),
        ):
            assert codex.get_codex_executable() == str(exe)

    def test_skips_unusable_windowsapps_alias(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        ext_dir = tmp_path / ".antigravity" / "extensions" / "openai.chatgpt-1" / "bin" / "windows-x86_64"
        ext_dir.mkdir(parents=True)
        exe = ext_dir / "codex.exe"
        exe.write_text("", encoding="utf-8")

        def _usable(candidate: str) -> bool:
            return candidate == str(exe)

        with (
            patch("shutil.which", return_value=r"C:\Program Files\WindowsApps\OpenAI.Codex\codex.exe"),
            patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)),
            patch("core.platform.codex._is_usable_codex_executable", side_effect=_usable),
        ):
            assert codex.get_codex_executable() == str(exe)

    def test_login_available_via_cli_status(self, tmp_path: Path):
        codex.get_codex_executable.cache_clear()
        completed = MagicMock(returncode=0, stdout="Logged in using ChatGPT\n", stderr="")
        with (
            patch("core.platform.codex.default_home_dir", return_value=str(tmp_path)),
            patch("core.platform.codex._run_codex_command", return_value=completed),
        ):
            assert codex.is_codex_login_available() is True

    def test_get_codex_device_login_returns_url_and_code(self):
        codex.get_codex_executable.cache_clear()
        status = MagicMock(returncode=1, stdout="", stderr="")
        output = """
Follow these steps to sign in with ChatGPT using device code authorization:
https://auth.openai.com/codex/device
73U2-6NBJ3
"""
        fake_proc = MagicMock()
        fake_proc.communicate.side_effect = [subprocess.TimeoutExpired(cmd="codex", timeout=8.0), (output, "")]
        with (
            patch("core.platform.codex.get_codex_executable", return_value="codex.exe"),
            patch("core.platform.codex._run_codex_command", return_value=status),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            result = codex.get_codex_device_login()

        assert result["ok"] is True
        assert result["login_url"] == "https://auth.openai.com/codex/device"
        assert result["device_code"] == "73U2-6NBJ3"
