from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.messaging.board import clear_post_hooks


@pytest.fixture
def shared_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shared = tmp_path / "shared"
    channels = shared / "channels"
    channels.mkdir(parents=True)
    (channels / "general.jsonl").touch()
    (channels / "ops.jsonl").touch()
    config = SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(board_mapping={"C-SLACK": "general"}),
            discord=SimpleNamespace(board_mapping={"C-DISCORD": "ops"}),
        )
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)
    monkeypatch.setattr("server.gateways.slack_socket.load_config", lambda: config)
    monkeypatch.setattr("server.gateways.discord_gateway.load_config", lambda: config)
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: shared)
    clear_post_hooks()
    yield shared
    clear_post_hooks()


def test_slack_gateway_posts_with_stable_request_id(shared_dir: Path) -> None:
    from server.gateways.slack_socket import _route_to_board

    _route_to_board("C-SLACK", "hello", "external-user", ts="1710000000.123")
    _route_to_board("C-SLACK", "duplicate", "external-user", ts="1710000000.123")

    rows = [json.loads(line) for line in (shared_dir / "channels" / "general.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["from"] == "external-user"
    assert rows[0]["source"] == "slack"
    assert rows[0]["kind"] == "external_user"
    assert rows[0]["request_id"] == "slack:1710000000.123"
    assert set(rows[0]) >= {"ts", "from", "text", "source", "id", "kind"}


def test_discord_gateway_records_bot_kind_and_message_id(shared_dir: Path) -> None:
    from server.gateways.discord_gateway import _route_to_board

    _route_to_board(
        "C-DISCORD",
        "hello",
        "service-bot",
        message_id="123456789",
        board_mapping={"C-DISCORD": "ops"},
        external_kind="bot",
    )

    row = json.loads((shared_dir / "channels" / "ops.jsonl").read_text(encoding="utf-8").strip())
    assert row["from"] == "service-bot"
    assert row["source"] == "discord"
    assert row["kind"] == "external_bot"
    assert row["request_id"] == "discord:123456789"
