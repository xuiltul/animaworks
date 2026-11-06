from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler


@pytest.fixture
def post_handler(tmp_path: Path, data_dir_at_tmp_path: Path) -> tuple[ToolHandler, MagicMock]:
    anima_dir = tmp_path / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    messenger = MagicMock()
    messenger.anima_name = "alice"
    messenger.shared_dir = tmp_path / "shared"
    (messenger.shared_dir / "channels").mkdir(parents=True)
    handler = ToolHandler(anima_dir, memory, messenger, tool_registry=[])
    handler._activity.log = MagicMock()
    handler._fanout_board_mentions = MagicMock()
    handler._fire_board_slack_sync = MagicMock()
    return handler, messenger


@pytest.mark.parametrize("failure", ["acl", "company", "poster"])
def test_rejected_post_does_not_create_channel_post_activity(
    post_handler: tuple[ToolHandler, MagicMock],
    failure: str,
) -> None:
    handler, messenger = post_handler
    handler._channel_company_boundary_error = MagicMock(return_value="Error: company boundary")
    if failure != "company":
        handler._channel_company_boundary_error.return_value = None
    if failure == "poster":
        messenger.post_channel.side_effect = ValueError("invalid poster")

    with patch("core.messaging.messenger.is_channel_member", return_value=failure != "acl"):
        result = handler.handle("post_channel", {"channel": "general", "text": "hello"})

    if failure == "poster":
        assert json.loads(result)["status"] == "error"
    else:
        assert result.startswith("Error")
    activity_types = [call.args[0] for call in handler._activity.log.call_args_list]
    assert "channel_post" not in activity_types


def test_successful_post_creates_channel_post_activity(post_handler: tuple[ToolHandler, MagicMock]) -> None:
    handler, _messenger = post_handler
    handler._channel_company_boundary_error = MagicMock(return_value=None)

    with patch("core.messaging.messenger.is_channel_member", return_value=True):
        result = handler.handle("post_channel", {"channel": "general", "text": "hello"})

    assert result == "Posted to #general"
    activity_types = [call.args[0] for call in handler._activity.log.call_args_list]
    assert "channel_post" in activity_types
