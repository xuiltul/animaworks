"""Unit tests for cli/commands/init_cmd.py — Init command."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestCmdInit:
    """Tests for cmd_init function."""

    @patch("cli.commands.init_cmd._interactive_user_setup")
    @patch("cli.commands.init_cmd._interactive_anima_setup")
    @patch("core.init.ensure_runtime_dir")
    @patch("core.paths.get_data_dir")
    def test_first_time_init(
        self, mock_get_dir, mock_ensure, mock_anima_setup, mock_user_setup, tmp_path
    ):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        # No config.json => first time init
        mock_get_dir.return_value = data_dir

        args = argparse.Namespace(
            force=False, reset=False, template=None,
            from_md=None, blank=None, skip_anima=False, name=None,
        )
        cmd_init(args)

        mock_ensure.assert_called_once_with(skip_animas=True)
        mock_anima_setup.assert_called_once_with(data_dir)
        mock_user_setup.assert_called_once_with(data_dir)

    @patch("core.paths.get_data_dir")
    def test_already_initialized(self, mock_get_dir, tmp_path, capsys):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "config.json").write_text("{}", encoding="utf-8")
        mock_get_dir.return_value = data_dir

        args = argparse.Namespace(
            force=False, reset=False, template=None,
            from_md=None, blank=None, skip_anima=False, name=None,
        )
        cmd_init(args)

        captured = capsys.readouterr()
        assert "already exists" in captured.out

    @patch("core.init.merge_templates")
    @patch("core.paths.get_data_dir")
    def test_force_merge(self, mock_get_dir, mock_merge, tmp_path, capsys):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "config.json").write_text("{}", encoding="utf-8")
        mock_get_dir.return_value = data_dir
        mock_merge.return_value = ["prompts/new.md"]

        args = argparse.Namespace(
            force=True, reset=False, template=None,
            from_md=None, blank=None, skip_anima=False, name=None,
        )
        cmd_init(args)

        captured = capsys.readouterr()
        assert "Merged 1 new file" in captured.out

    @patch("core.init.merge_templates")
    @patch("core.paths.get_data_dir")
    def test_force_no_changes(self, mock_get_dir, mock_merge, tmp_path, capsys):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "config.json").write_text("{}", encoding="utf-8")
        mock_get_dir.return_value = data_dir
        mock_merge.return_value = []

        args = argparse.Namespace(
            force=True, reset=False, template=None,
            from_md=None, blank=None, skip_anima=False, name=None,
        )
        cmd_init(args)

        captured = capsys.readouterr()
        assert "Already up to date" in captured.out

    @patch("core.init.ensure_runtime_dir")
    @patch("core.paths.get_data_dir")
    def test_skip_anima(self, mock_get_dir, mock_ensure, tmp_path, capsys):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        mock_get_dir.return_value = data_dir

        args = argparse.Namespace(
            force=False, reset=False, template=None,
            from_md=None, blank=None, skip_anima=True, name=None,
        )
        cmd_init(args)

        captured = capsys.readouterr()
        assert "no animas" in captured.out

    @patch("cli.commands.init_cmd._register_anima_in_config")
    @patch("core.anima_factory.create_from_template")
    @patch("core.init.ensure_runtime_dir")
    @patch("core.paths.get_data_dir")
    def test_template_mode(
        self, mock_get_dir, mock_ensure, mock_create, mock_register, tmp_path, capsys
    ):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "animas").mkdir()
        mock_get_dir.return_value = data_dir

        anima_dir = data_dir / "animas" / "alice"
        anima_dir.mkdir()
        mock_create.return_value = anima_dir

        args = argparse.Namespace(
            force=False, reset=False, template="basic",
            from_md=None, blank=None, skip_anima=False, name=None,
        )
        cmd_init(args)

        mock_create.assert_called_once()
        captured = capsys.readouterr()
        assert "alice" in captured.out

    @patch("cli.commands.init_cmd._register_anima_in_config")
    @patch("core.anima_factory.validate_anima_name")
    @patch("core.anima_factory.create_blank")
    @patch("core.init.ensure_runtime_dir")
    @patch("core.paths.get_data_dir")
    def test_blank_mode(
        self, mock_get_dir, mock_ensure, mock_create, mock_validate, mock_register,
        tmp_path, capsys
    ):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "animas").mkdir()
        mock_get_dir.return_value = data_dir
        mock_validate.return_value = None

        anima_dir = data_dir / "animas" / "bob"
        anima_dir.mkdir()
        mock_create.return_value = anima_dir

        args = argparse.Namespace(
            force=False, reset=False, template=None,
            from_md=None, blank="bob", skip_anima=False, name=None,
        )
        cmd_init(args)

        mock_create.assert_called_once()
        captured = capsys.readouterr()
        assert "bob" in captured.out

    @patch("core.anima_factory.validate_anima_name")
    @patch("core.init.ensure_runtime_dir")
    @patch("core.paths.get_data_dir")
    def test_blank_mode_invalid_name(
        self, mock_get_dir, mock_ensure, mock_validate, tmp_path, capsys
    ):
        from cli.commands.init_cmd import cmd_init

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "animas").mkdir()
        mock_get_dir.return_value = data_dir
        mock_validate.return_value = "Name must be lowercase"

        args = argparse.Namespace(
            force=False, reset=False, template=None,
            from_md=None, blank="INVALID", skip_anima=False, name=None,
        )
        with pytest.raises(SystemExit):
            cmd_init(args)


class TestRegisterAnimaInConfig:
    """Tests for _register_anima_in_config."""

    def test_register_new_anima(self, tmp_path):
        import json
        from cli.commands.init_cmd import _register_anima_in_config

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"version": 1, "animas": {}}),
            encoding="utf-8",
        )

        with (
            patch("core.config.models.load_config") as mock_load,
            patch("core.config.models.save_config") as mock_save,
        ):
            mock_config = MagicMock()
            mock_config.animas = {}
            mock_load.return_value = mock_config

            _register_anima_in_config(tmp_path, "alice")

            mock_save.assert_called_once()

    def test_register_no_config_file(self, tmp_path):
        from cli.commands.init_cmd import _register_anima_in_config

        # No config.json exists — should return silently
        _register_anima_in_config(tmp_path, "alice")


class TestInteractiveUserSetup:
    """Tests for _interactive_user_setup."""

    @patch("builtins.input", side_effect=["n"])
    def test_decline(self, mock_input, tmp_path):
        from cli.commands.init_cmd import _interactive_user_setup

        _interactive_user_setup(tmp_path)
        # No user dir should be created
        assert not (tmp_path / "shared" / "users").exists()

    @patch("builtins.input", side_effect=["y", "TestUser", "UTC", "some notes"])
    def test_accept_with_notes(self, mock_input, tmp_path):
        from cli.commands.init_cmd import _interactive_user_setup

        _interactive_user_setup(tmp_path)
        user_dir = tmp_path / "shared" / "users" / "TestUser"
        assert user_dir.exists()
        index = (user_dir / "index.md").read_text(encoding="utf-8")
        assert "TestUser" in index
        assert "UTC" in index
        assert "some notes" in index

    @patch("builtins.input", side_effect=["y", ""])
    def test_empty_name(self, mock_input, tmp_path):
        from cli.commands.init_cmd import _interactive_user_setup

        _interactive_user_setup(tmp_path)
        # Should not create anything
        assert not (tmp_path / "shared" / "users").exists()
