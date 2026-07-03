"""Tests for human sender resolution in the CLI send command.

A sender name that does not resolve to a registered anima must be delivered
with source="human" so the receiving anima does not discard the message as
an unknown-anima sender (Messenger inbox validation ignores source=="anima"
messages whose from_person is not a known anima).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from core.messenger import Messenger


# ── Messenger.send source passthrough ─────────────────────────


def _read_single_inbox_message(shared_dir: Path, to: str) -> dict:
    files = list((shared_dir / "inbox" / to).glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_messenger_send_default_source_is_anima(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    messenger = Messenger(shared_dir, "alice")
    messenger.send(to="bob", content="hello")
    data = _read_single_inbox_message(shared_dir, "bob")
    assert data["source"] == "anima"
    assert data["from_person"] == "alice"


def test_messenger_send_human_source(tmp_path: Path) -> None:
    shared_dir = tmp_path / "shared"
    messenger = Messenger(shared_dir, "someuser")
    messenger.send(to="bob", content="hello", source="human")
    data = _read_single_inbox_message(shared_dir, "bob")
    assert data["source"] == "human"
    assert data["from_person"] == "someuser"


def test_messenger_human_source_skips_cascade_limiter(tmp_path: Path) -> None:
    """Human-sourced sends must not consult the anima cascade limiter."""
    shared_dir = tmp_path / "shared"
    # Make the recipient look like an internal anima so the depth-check
    # branch would be reached for anima-sourced messages.
    (tmp_path / "animas" / "bob").mkdir(parents=True)
    messenger = Messenger(shared_dir, "someuser")
    with patch("core.cascade_limiter.get_depth_limiter") as get_limiter:
        msg = messenger.send(to="bob", content="hello", source="human")
    get_limiter.assert_not_called()
    assert msg.type == "message"


# ── CLI sender resolution ─────────────────────────────────────


def test_resolve_sender_source_known_anima_dir(tmp_path: Path) -> None:
    from cli.commands.messaging import _resolve_sender_source

    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# alice\n", encoding="utf-8")
    with (
        patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        patch("core.config.models.load_config", side_effect=Exception("no config")),
    ):
        assert _resolve_sender_source("alice") == "anima"


def test_resolve_sender_source_unknown_name_is_human(tmp_path: Path) -> None:
    from cli.commands.messaging import _resolve_sender_source

    (tmp_path / "animas").mkdir(parents=True)
    with (
        patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        patch("core.config.models.load_config", side_effect=Exception("no config")),
    ):
        assert _resolve_sender_source("someuser") == "human"


def test_resolve_sender_source_config_registered_anima(tmp_path: Path) -> None:
    from cli.commands.messaging import _resolve_sender_source

    class _Cfg:
        animas = {"carol": object()}

    (tmp_path / "animas").mkdir(parents=True)
    with (
        patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
        patch("core.config.models.load_config", return_value=_Cfg()),
    ):
        assert _resolve_sender_source("carol") == "anima"
