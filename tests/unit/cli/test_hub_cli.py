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


def test_hub_member_set_avatar_and_clear(shared_dir: Path, tmp_path: Path, capsys) -> None:
    from io import BytesIO

    from PIL import Image

    _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll"])
    capsys.readouterr()
    image_path = tmp_path / "avatar.png"
    output = BytesIO()
    Image.new("RGB", (300, 200), (10, 20, 30)).save(output, format="PNG")
    image_path.write_bytes(output.getvalue())

    _run(["hub", "member", "set-avatar", "grok", str(image_path)])
    assert "アバターを登録しました: grok" in capsys.readouterr().out
    avatar_path = shared_dir / "hub" / "avatars" / "grok.png"
    assert avatar_path.is_file()
    with Image.open(avatar_path) as image:
        assert image.size == (256, 256)

    _run(["hub", "member", "set-avatar", "grok", "--clear"])
    assert "アバターを削除しました: grok" in capsys.readouterr().out
    assert not avatar_path.exists()


def test_hub_member_set_avatar_errors(shared_dir: Path, tmp_path: Path, capsys) -> None:
    _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll"])
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _run(["hub", "member", "set-avatar", "grok"])
    assert "--clear" in capsys.readouterr().err

    bad = tmp_path / "bad.txt"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(["hub", "member", "set-avatar", "grok", str(bad)])
    assert "アバター画像が不正です" in capsys.readouterr().err


def test_hub_setup_prompt_prints_instructions(shared_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(
        locale="ja",
        hub=SimpleNamespace(
            enabled=True,
            company="a",
            board="hub",
            public_hosts=["hub.example.com"],
            max_posts_per_hour=20,
            max_round_trips=4,
        ),
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)
    monkeypatch.setattr("core.hub.setup_prompt.load_config", lambda: config)
    _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll", "--display-name", "Grok"])
    capsys.readouterr()

    _run(["hub", "setup-prompt", "grok"])
    output = capsys.readouterr().out
    assert "https://hub.example.com/hub/mcp" in output
    assert "Grok" in output
    assert "まず @human" in output

    _run(["hub", "setup-prompt", "grok", "--url", "https://custom.example.com/hub/mcp"])
    assert "https://custom.example.com/hub/mcp" in capsys.readouterr().out


def test_hub_setup_prompt_errors(shared_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(
        locale="ja",
        hub=SimpleNamespace(
            enabled=True,
            company="a",
            board="hub",
            public_hosts=[],
            max_posts_per_hour=20,
            max_round_trips=4,
        ),
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)
    monkeypatch.setattr("core.hub.setup_prompt.load_config", lambda: config)
    _run(["hub", "member", "add", "grok", "--company", "a", "--route", "poll"])
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _run(["hub", "setup-prompt", "grok"])
    assert "--url" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _run(["hub", "setup-prompt", "nobody", "--url", "https://hub.example.com/hub/mcp"])
    assert "nobody" in capsys.readouterr().err
