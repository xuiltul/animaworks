from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.i18n import t
from core.tooling.handler import ToolHandler


@pytest.fixture
def post_handler(
    tmp_path: Path,
    data_dir_at_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ToolHandler, MagicMock]:
    from core.messaging.board import anima_principal, post_to_board

    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text("{}", encoding="utf-8")
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    messenger = MagicMock()
    messenger.anima_name = "alice"
    messenger.shared_dir = tmp_path / "shared"
    (messenger.shared_dir / "channels").mkdir(parents=True)
    (messenger.shared_dir / "channels" / "general.jsonl").touch()

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
            anima_principal(from_anima, animas_dir=tmp_path / "animas"),
            request_id=request_id,
            origin="internal",
            notify=not suppress_mentions,
            shared_dir=messenger.shared_dir,
        )
        return {"post": result.to_dict()}

    monkeypatch.setattr("core.messaging.board_client.post_via_server", post_direct)
    handler = ToolHandler(anima_dir, memory, messenger, tool_registry=[])
    handler._activity.log = MagicMock()
    return handler, messenger


def test_rejected_post_does_not_create_channel_post_activity(
    post_handler: tuple[ToolHandler, MagicMock],
) -> None:
    handler, messenger = post_handler
    from core.messaging.messenger import ChannelMeta, save_channel_meta

    save_channel_meta(messenger.shared_dir, "general", ChannelMeta(members=["bob"]))
    result = handler.handle("post_channel", {"channel": "general", "text": "hello"})

    assert result.startswith("Error")
    assert not (messenger.shared_dir / "channels" / "general.jsonl").read_text(encoding="utf-8")
    assert not (messenger.shared_dir.parent / "animas" / "alice" / "activity_log").exists()


def test_successful_post_creates_channel_post_activity(post_handler: tuple[ToolHandler, MagicMock]) -> None:
    handler, _messenger = post_handler
    result = handler.handle("post_channel", {"channel": "general", "text": "hello"})

    assert result == t("handler.channel_post_success", channel="general")
    activity_types = [call.args[0] for call in handler._activity.log.call_args_list]
    assert "channel_post" not in activity_types
    log_files = list((handler._anima_dir / "activity_log").glob("*.jsonl"))
    entries = [json.loads(line) for path in log_files for line in path.read_text(encoding="utf-8").splitlines()]
    assert [entry["type"] for entry in entries].count("channel_post") == 1
