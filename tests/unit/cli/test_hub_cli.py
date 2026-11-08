from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest

from cli.parser import build_parser
from core.hub.members import find_member_by_token, load_members


@pytest.fixture
def shared_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr("cli.commands.hub.get_shared_dir", lambda: shared)
    return shared


def _run(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


def test_hub_member_add_list_set_token_and_toggle(shared_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll", "--display-name", "Grok"])
    assert "grok" in capsys.readouterr().out

    stored = load_members(shared_dir)["grok"]
    assert stored.display_name == "Grok"
    assert stored.message_route == {"type": "poll"}
    assert stored.token_sha256 == ""

    monkeypatch.setenv("HUB_TOKEN", "super-secret")
    _run(["hub", "member", "set-token", "grok", "--from-env", "HUB_TOKEN"])
    token_file = (shared_dir / "hub" / "members" / "grok.json").read_text(encoding="utf-8")
    assert hashlib.sha256(b"super-secret").hexdigest() in token_file
    assert "super-secret" not in token_file
    assert find_member_by_token(shared_dir, "super-secret") is not None

    _run(["hub", "member", "list"])
    listing = capsys.readouterr().out
    assert "grok" in listing
    assert "super-secret" not in listing

    _run(["hub", "member", "disable", "grok"])
    assert find_member_by_token(shared_dir, "super-secret") is None
    _run(["hub", "member", "enable", "grok"])
    assert find_member_by_token(shared_dir, "super-secret") is not None


def test_hub_member_webhook_route_and_duplicate(shared_dir: Path, capsys) -> None:
    _run(
        [
            "hub",
            "member",
            "add",
            "grok",
            "--company",
            "a",
            "--route",
            "webhook",
            "--url-ref",
            "hub.url",
            "--key-ref",
            "hub.key",
            "--expect-every",
            "3h",
        ]
    )
    capsys.readouterr()
    member = load_members(shared_dir)["grok"]
    assert member.message_route == {"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"}
    assert member.expect_every == "3h"

    with pytest.raises(SystemExit) as exc:
        _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll"])
    assert exc.value.code == 1


def test_hub_member_errors(shared_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    with pytest.raises(SystemExit):
        _run(["hub", "member", "add", "grok", "--company", "a", "--route", "webhook"])

    with pytest.raises(SystemExit):
        _run(["hub", "member", "add", "Grok!", "--company", "a", "--route", "poll"])

    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        _run(["hub", "member", "set-token", "grok", "--from-env", "MISSING_TOKEN"])

    with pytest.raises(SystemExit):
        _run(["hub", "member", "disable", "nobody"])
