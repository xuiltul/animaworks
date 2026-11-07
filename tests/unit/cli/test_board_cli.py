"""Unit tests for CLI Board adapters and operator paths."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _tool_result_payload(output: str) -> str:
    return output.split("\n", 1)[1].rsplit("\n</tool_result>", 1)[0]


def _make_handler(anima_dir: Path, shared_dir: Path):
    from core.tooling.handler import ToolHandler

    messenger = MagicMock()
    messenger.anima_name = anima_dir.name
    messenger.shared_dir = shared_dir
    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=MagicMock(),
        messenger=messenger,
        tool_registry=[],
    )
    return handler, messenger


# ── Board read ────────────────────────────────────────────


class TestCmdBoardRead:
    def test_anima_read_routes_through_read_channel(self, tmp_path: Path, capsys) -> None:
        from cli.commands.board import cmd_board_read

        args = argparse.Namespace(channel="general", limit=7, human_only=True, latest=True)
        with (
            patch("cli.commands.board.current_anima_dir", return_value=tmp_path / "alice"),
            patch("cli._anima_tool.run_anima_tool", return_value='[{"text":"hi"}]') as run_tool,
        ):
            cmd_board_read(args)

        run_tool.assert_called_once_with(
            "read_channel",
            {"channel": "general", "limit": 7, "human_only": True, "latest": True},
        )
        output = capsys.readouterr().out
        assert '<tool_result tool="animaworks-tool board read" trust="untrusted">' in output
        assert json.loads(_tool_result_payload(output)) == [{"text": "hi"}]

    def test_operator_read_keeps_messenger_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        from cli.commands.board import cmd_board_read

        shared_dir = tmp_path / "shared"
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        (channels_dir / "general.jsonl").write_text('{"from":"alice","text":"hello"}\n', encoding="utf-8")
        mock_messenger = MagicMock()
        mock_messenger.read_channel.return_value = [{"from": "alice", "text": "hello"}]
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared_dir)
        monkeypatch.setattr("core.messaging.messenger.Messenger", lambda *_args: mock_messenger)
        cmd_board_read(argparse.Namespace(channel="general", limit=10, human_only=False))

        mock_messenger.read_channel.assert_called_once_with("general", limit=10, human_only=False, source="human")
        output = capsys.readouterr().out
        assert 'trust="untrusted"' in output
        assert json.loads(_tool_result_payload(output))[0]["from"] == "alice"

    def test_operator_read_distinguishes_missing_and_empty_channels(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        from cli.commands.board import cmd_board_read

        shared_dir = tmp_path / "shared"
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        (channels_dir / "general.jsonl").touch()
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared_dir)
        monkeypatch.setattr(
            "core.messaging.messenger.Messenger", lambda *_args: MagicMock(read_channel=lambda *_a, **_k: [])
        )

        cmd_board_read(argparse.Namespace(channel="general", limit=10, human_only=False))
        empty_output = capsys.readouterr().out
        assert "no posts" in empty_output.lower() or "投稿がありません" in empty_output
        with pytest.raises(SystemExit) as exc_info:
            cmd_board_read(argparse.Namespace(channel="missing", limit=10, human_only=False))

        assert exc_info.value.code == 1
        missing_output = capsys.readouterr().out
        assert "not found" in missing_output.lower() or "見つかりません" in missing_output
        assert empty_output != missing_output

    def test_operator_read_rejects_unreadable_channel_metadata(self, tmp_path: Path, monkeypatch, capsys) -> None:
        from cli.commands.board import cmd_board_read

        shared_dir = tmp_path / "shared"
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        (channels_dir / "general.jsonl").touch()
        (channels_dir / "general.meta.json").write_text("{", encoding="utf-8")
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared_dir)
        monkeypatch.setattr("core.messaging.messenger.Messenger", lambda *_args: MagicMock())

        with pytest.raises(SystemExit) as exc_info:
            cmd_board_read(argparse.Namespace(channel="general", limit=10, human_only=False))

        assert exc_info.value.code == 1
        assert 'trust="untrusted"' in capsys.readouterr().out


# ── Board post ────────────────────────────────────────────


class TestCmdBoardPost:
    @pytest.fixture
    def operator_board(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from core.messaging.board import operator_principal, post_to_board

        shared_dir = tmp_path / "shared"
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        (channels_dir / "general.jsonl").touch()

        def post_direct(
            channel: str,
            text: str,
            *,
            from_anima: str,
            source: str = "anima",
            from_name: str | None = None,
            request_id: str | None = None,
            suppress_mentions: bool = False,
        ) -> dict:
            result = post_to_board(
                channel,
                text,
                operator_principal(from_name or from_anima),
                request_id=request_id,
                origin="internal",
                notify=not suppress_mentions,
                shared_dir=shared_dir,
            )
            return {"post": result.to_dict()}

        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.messaging.board_client.post_via_server", post_direct)
        return shared_dir

    def test_operator_post_uses_internal_board_client(
        self,
        operator_board: Path,
        capsys,
    ) -> None:
        from cli.commands.board import cmd_board_post

        args = argparse.Namespace(from_anima="alice", channel="general", text="Hello @bob")

        cmd_board_post(args)

        row = json.loads((operator_board / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert row["from"] == "alice"
        assert row["source"] == "human"
        assert row["kind"] == "operator"
        from core.i18n import t

        assert capsys.readouterr().out.strip() == t("cli.board_post_success", channel="general")

    def test_anima_post_uses_handler_fanout_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cli.commands.board import cmd_board_post

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        args = argparse.Namespace(post_args=["general", "Hello @bob"], as_name=None)
        run_tool = MagicMock()
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: anima_dir)
        monkeypatch.setattr("cli.commands.board.run_and_print", run_tool)

        cmd_board_post(args)

        run_tool.assert_called_once_with("post_channel", {"channel": "general", "text": "Hello @bob"})

    @pytest.mark.parametrize("post_args", [["general"], ["general", "-"]])
    def test_anima_post_stdin_content_is_forwarded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        post_args: list[str],
    ) -> None:
        from io import StringIO

        from cli.commands.board import cmd_board_post

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        run_tool = MagicMock()
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: anima_dir)
        monkeypatch.setattr("cli.commands.board.run_and_print", run_tool)
        monkeypatch.setattr("cli.commands.board.sys.stdin", StringIO("Hello from stdin\n\n"))

        cmd_board_post(argparse.Namespace(post_args=post_args, as_name=None))

        run_tool.assert_called_once_with("post_channel", {"channel": "general", "text": "Hello from stdin\n"})

    def test_operator_post_new_form_defaults_to_operator(self, operator_board: Path, capsys) -> None:
        from cli.commands.board import cmd_board_post

        cmd_board_post(argparse.Namespace(post_args=["general", "Hello"], as_name=None))

        row = json.loads((operator_board / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert row["from"] == "operator"
        assert row["kind"] == "operator"
        assert "general" in capsys.readouterr().out

    def test_operator_post_as_overrides_legacy_from(self, operator_board: Path) -> None:
        from cli.commands.board import cmd_board_post

        cmd_board_post(argparse.Namespace(post_args=["alice", "general", "Legacy body"], as_name="bob"))

        row = json.loads((operator_board / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert row["from"] == "bob"
        assert row["kind"] == "operator"

    def test_operator_post_reads_channel_only_form_from_stdin(
        self,
        operator_board: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from io import StringIO

        from cli.commands.board import cmd_board_post

        monkeypatch.setattr("cli.commands.board.sys.stdin", StringIO("stdin body\n"))
        cmd_board_post(argparse.Namespace(post_args=["general"], as_name=None))

        row = json.loads((operator_board / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert row["text"] == "stdin body"

    @pytest.mark.parametrize(
        ("post_args", "expected_from"),
        [(["general", "-"], "operator"), (["alice", "general", "-"], "alice")],
    )
    def test_operator_post_dash_reads_stdin_for_new_and_legacy_forms(
        self,
        operator_board: Path,
        monkeypatch: pytest.MonkeyPatch,
        post_args: list[str],
        expected_from: str,
    ) -> None:
        from io import StringIO

        from cli.commands.board import cmd_board_post

        monkeypatch.setattr("cli.commands.board.sys.stdin", StringIO("stdin body\r\n"))
        cmd_board_post(argparse.Namespace(post_args=post_args, as_name=None))

        row = json.loads((operator_board / "channels" / "general.jsonl").read_text(encoding="utf-8"))
        assert row["text"] == "stdin body"
        assert row["from"] == expected_from

    def test_anima_post_refuses_impersonation(self, tmp_path: Path, capsys) -> None:
        from cli.commands.board import cmd_board_post

        args = argparse.Namespace(post_args=["bob", "general", "hello"], as_name=None)
        with (
            patch("cli.commands.board.current_anima_dir", return_value=tmp_path / "alice"),
            patch("cli._anima_tool.run_anima_tool") as run_tool,
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_board_post(args)

        assert exc_info.value.code == 1
        assert "alice" in capsys.readouterr().err
        run_tool.assert_not_called()

    def test_anima_post_rejects_as_impersonation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        from cli.commands.board import cmd_board_post

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        run_tool = MagicMock()
        monkeypatch.setattr("cli.commands.board.current_anima_dir", lambda: anima_dir)
        monkeypatch.setattr("cli.commands.board.run_and_print", run_tool)

        with pytest.raises(SystemExit) as exc_info:
            cmd_board_post(argparse.Namespace(post_args=["general", "hello"], as_name="bob"))

        assert exc_info.value.code == 1
        assert "alice" in capsys.readouterr().err
        run_tool.assert_not_called()

    @pytest.mark.parametrize("post_args", [["general"], ["general", "-"]])
    def test_empty_stdin_is_an_error(self, monkeypatch: pytest.MonkeyPatch, post_args: list[str], capsys) -> None:
        from io import StringIO

        from cli.commands.board import cmd_board_post

        monkeypatch.setattr("cli.commands.board.sys.stdin", StringIO(""))
        with pytest.raises(SystemExit) as exc_info:
            cmd_board_post(argparse.Namespace(post_args=post_args, as_name=None))

        assert exc_info.value.code == 1
        error = capsys.readouterr().err
        assert "empty" in error.lower() or "空" in error

    def test_too_many_post_arguments_are_rejected(self, capsys) -> None:
        from cli.commands.board import cmd_board_post

        with pytest.raises(SystemExit) as exc_info:
            cmd_board_post(argparse.Namespace(post_args=["alice", "general", "text", "extra"], as_name=None))

        assert exc_info.value.code == 1
        error = capsys.readouterr().err
        assert "arguments" in error.lower() or "引数" in error

    def test_operator_post_missing_channel_exits_with_error(
        self,
        operator_board: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        from cli.commands.board import cmd_board_post
        from core.messaging.board import operator_principal, post_to_board

        def post_missing(
            channel: str,
            text: str,
            *,
            from_anima: str,
            source: str = "anima",
            from_name: str | None = None,
            request_id: str | None = None,
            suppress_mentions: bool = False,
        ) -> dict:
            result = post_to_board(
                channel,
                text,
                operator_principal(from_name or from_anima),
                request_id=request_id,
                origin="internal",
                notify=not suppress_mentions,
                shared_dir=operator_board,
            )
            return {"post": result.to_dict()}

        monkeypatch.setattr("core.messaging.board_client.post_via_server", post_missing)
        args = argparse.Namespace(from_anima="alice", channel="missing", text="hello")

        with pytest.raises(SystemExit) as exc_info:
            cmd_board_post(args)

        assert exc_info.value.code == 1
        from core.i18n import t

        assert capsys.readouterr().err.strip() == t("cli.board_post_failed", detail="not_found")


# ── DM history ───────────────────────────────────────────


class TestCmdBoardDmHistory:
    def test_anima_dm_history_maps_all_tool_arguments(self, tmp_path: Path, capsys) -> None:
        from cli.commands.board import cmd_board_dm_history

        args = argparse.Namespace(
            from_anima="alice",
            peer="bob",
            limit=40,
            direction="sent",
            hours=24,
            keyword="deploy",
        )
        with (
            patch("cli.commands.board.current_anima_dir", return_value=tmp_path / "alice"),
            patch("cli._anima_tool.run_anima_tool", return_value="[]") as run_tool,
        ):
            cmd_board_dm_history(args)

        run_tool.assert_called_once_with(
            "read_dm_history",
            {"peer": "bob", "limit": 40, "direction": "sent", "hours": 24, "keyword": "deploy"},
        )
        output = capsys.readouterr().out
        assert '<tool_result tool="animaworks-tool board dm-history" trust="untrusted">' in output
        assert _tool_result_payload(output) == "[]"

    @patch("core.messaging.messenger.Messenger")
    @patch("core.paths.get_shared_dir", return_value=Path("/tmp/shared"))
    @patch("core.infra.runtime_init.ensure_runtime_dir")
    @patch("cli.commands.board.current_anima_dir", return_value=None)
    def test_operator_dm_history_forwards_filter_flags(
        self,
        mock_context,
        mock_ensure,
        mock_shared,
        mock_messenger_cls,
        capsys,
    ) -> None:
        from cli.commands.board import cmd_board_dm_history

        mock_messenger = MagicMock()
        mock_messenger.read_dm_history.return_value = [{"from": "alice", "text": "deploy"}]
        mock_messenger_cls.return_value = mock_messenger
        args = argparse.Namespace(
            from_anima="alice",
            peer="bob",
            limit=40,
            direction="received",
            hours=12,
            keyword="deploy",
        )

        cmd_board_dm_history(args)

        mock_messenger.read_dm_history.assert_called_once_with(
            "bob",
            limit=40,
            direction="received",
            hours=12,
            keyword="deploy",
        )
        output = capsys.readouterr().out
        assert 'tool="animaworks-tool board dm-history" trust="untrusted"' in output
        assert json.loads(_tool_result_payload(output))[0]["text"] == "deploy"

    def test_anima_dm_history_refuses_impersonation(self, tmp_path: Path, capsys) -> None:
        from cli.commands.board import cmd_board_dm_history

        args = argparse.Namespace(from_anima="bob", peer="carol", limit=20)
        with (
            patch("cli.commands.board.current_anima_dir", return_value=tmp_path / "alice"),
            patch("cli._anima_tool.run_anima_tool") as run_tool,
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_board_dm_history(args)

        assert exc_info.value.code == 1
        assert "alice" in capsys.readouterr().err
        run_tool.assert_not_called()


def test_board_dm_history_parser_exposes_schema_filters() -> None:
    from cli.parser import build_parser

    args = build_parser().parse_args(
        ["board", "dm-history", "alice", "bob", "--direction", "sent", "--hours", "24", "--keyword", "deploy"]
    )
    assert args.direction == "sent"
    assert args.hours == 24
    assert args.keyword == "deploy"


@pytest.mark.parametrize(
    ("post_args", "as_name"),
    [
        (["general"], None),
        (["general", "hello"], None),
        (["general", "-"], None),
        (["alice", "general", "legacy"], None),
        (["general", "hello"], "bob"),
    ],
)
def test_board_post_parser_accepts_new_and_legacy_forms(post_args: list[str], as_name: str | None) -> None:
    from cli.parser import build_parser

    argv = ["board", "post", *post_args]
    if as_name:
        argv.extend(["--as", as_name])
    args = build_parser().parse_args(argv)

    assert args.post_args == post_args
    assert args.as_name == as_name


def test_board_post_help_describes_stdin_and_legacy_form(capsys) -> None:
    from cli.parser import build_parser

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["board", "post", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "read it from stdin" in output or "標準入力" in output
    assert "legacy" in output.lower() or "旧形式" in output
