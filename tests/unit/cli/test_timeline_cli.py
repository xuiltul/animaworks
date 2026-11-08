# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestTimelineParser:
    def test_add_parses_done(self) -> None:
        from cli.parser import build_parser

        args = build_parser().parse_args(
            ["timeline", "add", "--kind", "done", "--text", "Finished", "--about", "tl_0123456789abcdef"]
        )

        assert args.command == "timeline"
        assert args.timeline_command == "add"
        assert args.kind == "done"
        assert args.text == "Finished"
        assert args.about == "tl_0123456789abcdef"

    def test_add_rejects_notified(self) -> None:
        from cli.parser import build_parser

        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["timeline", "add", "--kind", "notified", "--text", "x"])

        assert exc.value.code == 2

    def test_read_default_limit(self) -> None:
        from cli.parser import build_parser

        args = build_parser().parse_args(["timeline", "read"])

        assert args.timeline_command == "read"
        assert args.limit == 20


class TestTimelineCommands:
    def test_operator_add_posts_internal(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        from cli.commands.timeline import cmd_timeline_add

        response = MagicMock(status_code=200)
        host_api = MagicMock()
        host_api.post.return_value = response
        monkeypatch.setattr("cli.commands.timeline.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.host_api.host_api", host_api)

        cmd_timeline_add(argparse.Namespace(kind="done", text="Finished", about=None, ref=None, from_name="alice"))

        payload = host_api.post.call_args.kwargs["json"]
        assert payload["from_anima"] == "alice"
        assert payload["kind"] == "done"
        assert payload["text"] == "Finished"
        from core.i18n import t

        assert t("cli.timeline_add_success", kind="done") in capsys.readouterr().out

    def test_anima_add_rejects_impersonation(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
        from cli.commands.timeline import cmd_timeline_add

        host_api = MagicMock()
        monkeypatch.setattr("cli.commands.timeline.current_anima_dir", lambda: tmp_path / "alice")
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.host_api.host_api", host_api)

        with pytest.raises(SystemExit) as exc:
            cmd_timeline_add(argparse.Namespace(kind="done", text="Finished", about=None, ref=None, from_name="bob"))

        assert exc.value.code == 1
        host_api.post.assert_not_called()
        from core.i18n import t

        assert t("cli.board_impersonation_refused", anima="alice", requested="bob") in capsys.readouterr().err

    def test_read_prints_internal_response(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        from cli.commands.timeline import cmd_timeline_read

        response = MagicMock(status_code=200)
        response.json.return_value = {"enabled": True, "company": "a", "items": [], "unread": 0}
        host_api = MagicMock()
        host_api.get.return_value = response
        monkeypatch.setattr("cli.commands.timeline.current_anima_dir", lambda: None)
        monkeypatch.setattr("core.infra.runtime_init.ensure_runtime_dir", lambda: None)
        monkeypatch.setattr("core.host_api.host_api", host_api)

        cmd_timeline_read(argparse.Namespace(limit=5))

        params = host_api.get.call_args.kwargs["params"]
        assert params["limit"] == 5
        assert "since" in params
        assert json.loads(capsys.readouterr().out)["company"] == "a"
