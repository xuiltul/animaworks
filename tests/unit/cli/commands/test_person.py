"""Unit tests for cli/commands/person.py — Person management CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── cmd_create_person ────────────────────────────────────


class TestCmdCreatePerson:
    @patch("cli.commands.init_cmd._register_person_in_config")
    @patch("core.person_factory.create_from_md")
    @patch("core.paths.get_persons_dir")
    @patch("core.paths.get_data_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_create_from_md(
        self, mock_ensure, mock_data_dir, mock_persons_dir,
        mock_create_md, mock_register, tmp_path, capsys,
    ):
        from cli.commands.person import cmd_create_person

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()

        mock_data_dir.return_value = data_dir
        mock_persons_dir.return_value = persons_dir

        person_dir = persons_dir / "alice"
        person_dir.mkdir()
        mock_create_md.return_value = person_dir

        md_file = tmp_path / "alice.md"
        md_file.write_text("# Alice", encoding="utf-8")

        args = argparse.Namespace(
            from_md=str(md_file), template=None, name="alice"
        )
        cmd_create_person(args)

        captured = capsys.readouterr()
        assert "alice" in captured.out

    @patch("cli.commands.init_cmd._register_person_in_config")
    @patch("core.person_factory.create_from_template")
    @patch("core.paths.get_persons_dir")
    @patch("core.paths.get_data_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_create_from_template(
        self, mock_ensure, mock_data_dir, mock_persons_dir,
        mock_create_tpl, mock_register, tmp_path, capsys,
    ):
        from cli.commands.person import cmd_create_person

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()

        mock_data_dir.return_value = data_dir
        mock_persons_dir.return_value = persons_dir

        person_dir = persons_dir / "alice"
        person_dir.mkdir()
        mock_create_tpl.return_value = person_dir

        args = argparse.Namespace(
            from_md=None, template="basic", name="alice"
        )
        cmd_create_person(args)

        captured = capsys.readouterr()
        assert "alice" in captured.out

    @patch("cli.commands.init_cmd._register_person_in_config")
    @patch("core.person_factory.validate_person_name", return_value=None)
    @patch("core.person_factory.create_blank")
    @patch("core.paths.get_persons_dir")
    @patch("core.paths.get_data_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_create_blank(
        self, mock_ensure, mock_data_dir, mock_persons_dir,
        mock_create, mock_validate, mock_register, tmp_path, capsys,
    ):
        from cli.commands.person import cmd_create_person

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()

        mock_data_dir.return_value = data_dir
        mock_persons_dir.return_value = persons_dir

        person_dir = persons_dir / "bob"
        person_dir.mkdir()
        mock_create.return_value = person_dir

        args = argparse.Namespace(
            from_md=None, template=None, name="bob"
        )
        cmd_create_person(args)

        captured = capsys.readouterr()
        assert "bob" in captured.out

    @patch("core.paths.get_persons_dir")
    @patch("core.paths.get_data_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_create_blank_no_name(
        self, mock_ensure, mock_data_dir, mock_persons_dir, tmp_path
    ):
        from cli.commands.person import cmd_create_person

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()
        mock_data_dir.return_value = data_dir
        mock_persons_dir.return_value = persons_dir

        args = argparse.Namespace(
            from_md=None, template=None, name=None
        )
        with pytest.raises(SystemExit):
            cmd_create_person(args)

    @patch("core.person_factory.validate_person_name", return_value="Invalid name")
    @patch("core.paths.get_persons_dir")
    @patch("core.paths.get_data_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_create_blank_invalid_name(
        self, mock_ensure, mock_data_dir, mock_persons_dir, mock_validate, tmp_path
    ):
        from cli.commands.person import cmd_create_person

        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        persons_dir = data_dir / "persons"
        persons_dir.mkdir()
        mock_data_dir.return_value = data_dir
        mock_persons_dir.return_value = persons_dir

        args = argparse.Namespace(
            from_md=None, template=None, name="INVALID"
        )
        with pytest.raises(SystemExit):
            cmd_create_person(args)


# ── cmd_chat ─────────────────────────────────────────────


class TestCmdChat:
    @patch("core.person.DigitalPerson")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_persons_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_chat_local(
        self, mock_ensure, mock_persons_dir, mock_shared, mock_dp_cls,
        tmp_path, capsys,
    ):
        from cli.commands.person import cmd_chat

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        person_dir = persons_dir / "alice"
        person_dir.mkdir()
        mock_persons_dir.return_value = persons_dir

        mock_person = MagicMock()
        mock_person.process_message = AsyncMock(return_value="Hello!")
        mock_dp_cls.return_value = mock_person

        args = argparse.Namespace(
            local=True, person="alice", message="Hi",
            from_person="human", gateway_url=None,
        )
        cmd_chat(args)

        captured = capsys.readouterr()
        assert "Hello!" in captured.out

    @patch("core.paths.get_persons_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_chat_local_person_not_found(
        self, mock_ensure, mock_persons_dir, tmp_path
    ):
        from cli.commands.person import cmd_chat

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        mock_persons_dir.return_value = persons_dir

        args = argparse.Namespace(
            local=True, person="nobody", message="Hi",
            from_person="human", gateway_url=None,
        )
        with pytest.raises(SystemExit):
            cmd_chat(args)

    @patch("httpx.request")
    def test_chat_remote(self, mock_request, capsys):
        from cli.commands.person import cmd_chat

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Remote reply"}
        mock_request.return_value = mock_resp

        args = argparse.Namespace(
            local=False, person="alice", message="Hi",
            from_person="human", gateway_url="http://localhost:18500",
        )
        cmd_chat(args)

        captured = capsys.readouterr()
        assert "Remote reply" in captured.out

    @patch("httpx.post", side_effect=__import__("httpx").ConnectError("fail"))
    def test_chat_remote_connection_error(self, mock_post):
        from cli.commands.person import cmd_chat

        args = argparse.Namespace(
            local=False, person="alice", message="Hi",
            from_person="human", gateway_url="http://localhost:18500",
        )
        with pytest.raises(SystemExit):
            cmd_chat(args)


# ── cmd_heartbeat ────────────────────────────────────────


class TestCmdHeartbeat:
    @patch("core.person.DigitalPerson")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.paths.get_persons_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_heartbeat_local(
        self, mock_ensure, mock_persons_dir, mock_shared, mock_dp_cls,
        tmp_path, capsys,
    ):
        from cli.commands.person import cmd_heartbeat

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        person_dir = persons_dir / "alice"
        person_dir.mkdir()
        mock_persons_dir.return_value = persons_dir

        mock_result = MagicMock()
        mock_result.action = "skip"
        mock_result.summary = "No pending tasks"
        mock_person = MagicMock()
        mock_person.run_heartbeat = AsyncMock(return_value=mock_result)
        mock_dp_cls.return_value = mock_person

        args = argparse.Namespace(
            local=True, person="alice", gateway_url=None,
        )
        cmd_heartbeat(args)

        captured = capsys.readouterr()
        assert "skip" in captured.out

    @patch("core.paths.get_persons_dir")
    @patch("core.init.ensure_runtime_dir")
    def test_heartbeat_local_not_found(self, mock_ensure, mock_persons_dir, tmp_path):
        from cli.commands.person import cmd_heartbeat

        persons_dir = tmp_path / "persons"
        persons_dir.mkdir()
        mock_persons_dir.return_value = persons_dir

        args = argparse.Namespace(
            local=True, person="nobody", gateway_url=None,
        )
        with pytest.raises(SystemExit):
            cmd_heartbeat(args)

    @patch("httpx.request")
    def test_heartbeat_remote(self, mock_request, capsys):
        from cli.commands.person import cmd_heartbeat

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"action": "skip"}
        mock_request.return_value = mock_resp

        args = argparse.Namespace(
            local=False, person="alice", gateway_url="http://localhost:18500",
        )
        cmd_heartbeat(args)

        captured = capsys.readouterr()
        assert "skip" in captured.out

    @patch("httpx.post", side_effect=__import__("httpx").ConnectError("fail"))
    def test_heartbeat_remote_error(self, mock_post):
        from cli.commands.person import cmd_heartbeat

        args = argparse.Namespace(
            local=False, person="alice", gateway_url="http://localhost:18500",
        )
        with pytest.raises(SystemExit):
            cmd_heartbeat(args)
