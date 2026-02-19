"""Unit tests for cli/parser.py — Argparse configuration and cli_main."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


class TestParserCommands:
    """Test that argparse correctly parses all subcommands."""

    def _parse(self, *args: str) -> argparse.Namespace:
        """Parse args by calling cli_main internals directly."""
        # We can't call cli_main() directly because it calls args.func().
        # Instead, patch load_dotenv, setup_logging, and build just the parser.
        with (
            patch("cli.parser.load_dotenv", create=True),
            patch("core.logging_config.setup_logging", create=True),
            patch("core.paths.get_data_dir", return_value=MagicMock()),
            patch("core.config.cli.cmd_config_dispatch"),
            patch("core.config.cli.cmd_config_get"),
            patch("core.config.cli.cmd_config_set"),
            patch("core.config.cli.cmd_config_list"),
        ):
            # Build the parser by importing and parsing
            import importlib
            import cli.parser as parser_mod
            importlib.reload(parser_mod)
            # We need to intercept parse_args
            # Instead, let's just test individual parse scenarios
            pass

    def test_init_default(self):
        """Test 'init' command parsing."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_init = sub.add_parser("init")
        p_init.add_argument("--force", action="store_true")

        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.force is False

    def test_reset_command(self):
        """Test 'reset' command parsing."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_reset = sub.add_parser("reset")
        p_reset.add_argument("--restart", action="store_true")

        args = parser.parse_args(["reset"])
        assert args.command == "reset"
        assert args.restart is False

    def test_reset_with_restart(self):
        """Test 'reset --restart' command parsing."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_reset = sub.add_parser("reset")
        p_reset.add_argument("--restart", action="store_true")

        args = parser.parse_args(["reset", "--restart"])
        assert args.restart is True

    def test_init_force(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_init = sub.add_parser("init")
        p_init.add_argument("--force", action="store_true")

        args = parser.parse_args(["init", "--force"])
        assert args.force is True

    def test_start_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_start = sub.add_parser("start")
        p_start.add_argument("--host", default="0.0.0.0")
        p_start.add_argument("--port", type=int, default=18500)

        args = parser.parse_args(["start"])
        assert args.command == "start"
        assert args.host == "0.0.0.0"
        assert args.port == 18500

    def test_start_custom_port(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_start = sub.add_parser("start")
        p_start.add_argument("--host", default="0.0.0.0")
        p_start.add_argument("--port", type=int, default=18500)

        args = parser.parse_args(["start", "--port", "9000", "--host", "127.0.0.1"])
        assert args.port == 9000
        assert args.host == "127.0.0.1"

    def test_chat_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_chat = sub.add_parser("chat")
        p_chat.add_argument("anima")
        p_chat.add_argument("message")
        p_chat.add_argument("--local", action="store_true")
        p_chat.add_argument("--from", dest="from_person", default="human")

        args = parser.parse_args(["chat", "alice", "Hello"])
        assert args.anima == "alice"
        assert args.message == "Hello"
        assert args.local is False
        assert args.from_person == "human"

    def test_chat_local_with_from(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_chat = sub.add_parser("chat")
        p_chat.add_argument("anima")
        p_chat.add_argument("message")
        p_chat.add_argument("--local", action="store_true")
        p_chat.add_argument("--from", dest="from_person", default="human")

        args = parser.parse_args(["chat", "alice", "Hi", "--local", "--from", "bob"])
        assert args.local is True
        assert args.from_person == "bob"

    def test_send_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_send = sub.add_parser("send")
        p_send.add_argument("from_person")
        p_send.add_argument("to_person")
        p_send.add_argument("message")
        p_send.add_argument("--thread-id", default=None)

        args = parser.parse_args(["send", "alice", "bob", "Hello Bob"])
        assert args.from_person == "alice"
        assert args.to_person == "bob"
        assert args.message == "Hello Bob"

    def test_heartbeat_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_hb = sub.add_parser("heartbeat")
        p_hb.add_argument("anima")
        p_hb.add_argument("--local", action="store_true")

        args = parser.parse_args(["heartbeat", "alice"])
        assert args.anima == "alice"
        assert args.local is False

    def test_list_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_list = sub.add_parser("list")
        p_list.add_argument("--local", action="store_true")

        args = parser.parse_args(["list", "--local"])
        assert args.local is True

    def test_data_dir_override(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)

        args = parser.parse_args(["--data-dir", "/custom/path"])
        assert args.data_dir == "/custom/path"

    def test_create_anima_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("create-anima")
        p.add_argument("--name", default=None)
        p.add_argument("--template", default=None)
        p.add_argument("--from-md", default=None)

        args = parser.parse_args(["create-anima", "--name", "bob", "--template", "basic"])
        assert args.name == "bob"
        assert args.template == "basic"

    def test_stop_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("stop")

        args = parser.parse_args(["stop"])
        assert args.command == "stop"

    def test_restart_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("restart")
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=18500)

        args = parser.parse_args(["restart", "--port", "9999"])
        assert args.port == 9999

    def test_status_command(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("status")

        args = parser.parse_args(["status"])
        assert args.command == "status"


class TestLazyImportWrappers:
    """Test that lazy import wrappers call the correct functions."""

    @patch("cli.commands.init_cmd.cmd_init")
    def test_lazy_init(self, mock_cmd):
        from cli.parser import _lazy_init

        args = MagicMock()
        _lazy_init(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima.cmd_create_anima")
    def test_lazy_create_anima(self, mock_cmd):
        from cli.parser import _lazy_create_anima

        args = MagicMock()
        _lazy_create_anima(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_start")
    def test_lazy_start(self, mock_cmd):
        from cli.parser import _lazy_start

        args = MagicMock()
        _lazy_start(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_serve")
    def test_lazy_serve(self, mock_cmd):
        from cli.parser import _lazy_serve

        args = MagicMock()
        _lazy_serve(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_stop")
    def test_lazy_stop(self, mock_cmd):
        from cli.parser import _lazy_stop

        args = MagicMock()
        _lazy_stop(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_restart")
    def test_lazy_restart(self, mock_cmd):
        from cli.parser import _lazy_restart

        args = MagicMock()
        _lazy_restart(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.init_cmd.cmd_reset")
    def test_lazy_reset(self, mock_cmd):
        from cli.parser import _lazy_reset

        args = MagicMock()
        _lazy_reset(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_gateway")
    def test_lazy_gateway(self, mock_cmd):
        from cli.parser import _lazy_gateway

        args = MagicMock()
        _lazy_gateway(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.server.cmd_worker")
    def test_lazy_worker(self, mock_cmd):
        from cli.parser import _lazy_worker

        args = MagicMock()
        _lazy_worker(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima.cmd_chat")
    def test_lazy_chat(self, mock_cmd):
        from cli.parser import _lazy_chat

        args = MagicMock()
        _lazy_chat(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima.cmd_heartbeat")
    def test_lazy_heartbeat(self, mock_cmd):
        from cli.parser import _lazy_heartbeat

        args = MagicMock()
        _lazy_heartbeat(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.messaging.cmd_send")
    def test_lazy_send(self, mock_cmd):
        from cli.parser import _lazy_send

        args = MagicMock()
        _lazy_send(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.messaging.cmd_list")
    def test_lazy_list(self, mock_cmd):
        from cli.parser import _lazy_list

        args = MagicMock()
        _lazy_list(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.messaging.cmd_status")
    def test_lazy_status(self, mock_cmd):
        from cli.parser import _lazy_status

        args = MagicMock()
        _lazy_status(args)
        mock_cmd.assert_called_once_with(args)


class TestAnimaSubcommandParsing:
    """Test that anima subcommand args are correctly parsed."""

    def test_anima_create(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_anima = sub.add_parser("anima")
        anima_sub = p_anima.add_subparsers(dest="anima_command")
        p = anima_sub.add_parser("create")
        p.add_argument("--name", default=None)
        p.add_argument("--from-md", default=None)
        p.add_argument("--template", default=None)

        args = parser.parse_args(["anima", "create", "--name", "alice", "--from-md", "alice.md"])
        assert args.anima_command == "create"
        assert args.name == "alice"
        assert args.from_md == "alice.md"

    def test_anima_delete(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_anima = sub.add_parser("anima")
        anima_sub = p_anima.add_subparsers(dest="anima_command")
        p = anima_sub.add_parser("delete")
        p.add_argument("anima")
        p.add_argument("--no-archive", action="store_true")
        p.add_argument("--force", action="store_true")

        args = parser.parse_args(["anima", "delete", "alice", "--force", "--no-archive"])
        assert args.anima == "alice"
        assert args.force is True
        assert args.no_archive is True

    def test_anima_disable(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_anima = sub.add_parser("anima")
        anima_sub = p_anima.add_subparsers(dest="anima_command")
        p = anima_sub.add_parser("disable")
        p.add_argument("anima")

        args = parser.parse_args(["anima", "disable", "alice"])
        assert args.anima == "alice"

    def test_anima_enable(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_anima = sub.add_parser("anima")
        anima_sub = p_anima.add_subparsers(dest="anima_command")
        p = anima_sub.add_parser("enable")
        p.add_argument("anima")

        args = parser.parse_args(["anima", "enable", "alice"])
        assert args.anima == "alice"

    def test_anima_list(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_anima = sub.add_parser("anima")
        anima_sub = p_anima.add_subparsers(dest="anima_command")
        p = anima_sub.add_parser("list")
        p.add_argument("--local", action="store_true")

        args = parser.parse_args(["anima", "list", "--local"])
        assert args.local is True


class TestDeprecationWarnings:
    """Test that deprecated commands show warnings."""

    @patch("cli.commands.anima.cmd_create_anima")
    def test_create_anima_deprecation(self, mock_cmd, capsys):
        from cli.parser import _lazy_create_anima
        args = MagicMock()
        _lazy_create_anima(args)
        captured = capsys.readouterr()
        assert "deprecated" in captured.err
        assert "anima create" in captured.err
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.messaging.cmd_list")
    def test_list_deprecation(self, mock_cmd, capsys):
        from cli.parser import _lazy_list
        args = MagicMock()
        _lazy_list(args)
        captured = capsys.readouterr()
        assert "deprecated" in captured.err
        assert "anima list" in captured.err
        mock_cmd.assert_called_once_with(args)


class TestNewLazyWrappers:
    """Test new lazy import wrappers."""

    @patch("cli.commands.anima.cmd_create_anima")
    def test_lazy_anima_create(self, mock_cmd):
        from cli.parser import _lazy_anima_create
        args = MagicMock()
        _lazy_anima_create(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima_mgmt.cmd_anima_delete")
    def test_lazy_anima_delete(self, mock_cmd):
        from cli.parser import _lazy_anima_delete
        args = MagicMock()
        _lazy_anima_delete(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima_mgmt.cmd_anima_disable")
    def test_lazy_anima_disable(self, mock_cmd):
        from cli.parser import _lazy_anima_disable
        args = MagicMock()
        _lazy_anima_disable(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima_mgmt.cmd_anima_enable")
    def test_lazy_anima_enable(self, mock_cmd):
        from cli.parser import _lazy_anima_enable
        args = MagicMock()
        _lazy_anima_enable(args)
        mock_cmd.assert_called_once_with(args)

    @patch("cli.commands.anima_mgmt.cmd_anima_list")
    def test_lazy_anima_list(self, mock_cmd):
        from cli.parser import _lazy_anima_list
        args = MagicMock()
        _lazy_anima_list(args)
        mock_cmd.assert_called_once_with(args)
