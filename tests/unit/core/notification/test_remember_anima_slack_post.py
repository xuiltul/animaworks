"""Human replies in a thread an Anima posted to must route back to that Anima."""

from __future__ import annotations

from typing import Any

import pytest

from core.notification import reply_routing
from core.notification.reply_routing import (
    lookup_notification_mapping,
    remember_anima_slack_post,
    save_notification_mapping,
)


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(reply_routing, "_map_path", lambda: tmp_path / "run" / "notification_map.json")
    monkeypatch.setattr(reply_routing, "post_notification_mapping_via_api", lambda *a, **k: False)


def test_top_level_post_is_keyed_by_its_own_ts() -> None:
    remember_anima_slack_post("mei", "C1", {"ts": "100.1", "channel": "C1"}, text="hello")

    mapping = lookup_notification_mapping("100.1")
    assert mapping is not None
    assert mapping["anima"] == "mei"
    assert mapping["channel"] == "C1"
    assert mapping["notification_text"] == "hello"


def test_reply_in_human_thread_is_keyed_by_thread_ts() -> None:
    remember_anima_slack_post("mei", "C1", {"ts": "200.2", "channel": "C1"}, thread_ts="100.1")

    assert lookup_notification_mapping("100.1")["anima"] == "mei"
    assert lookup_notification_mapping("200.2") is None


def test_existing_owner_is_kept() -> None:
    save_notification_mapping("100.1", "C1", "librarian", notification_text="call_human")

    remember_anima_slack_post("mei", "C1", {"ts": "200.2"}, thread_ts="100.1")

    assert lookup_notification_mapping("100.1")["anima"] == "librarian"


def test_slack_sdk_response_is_accepted() -> None:
    from slack_sdk.web.slack_response import SlackResponse

    response = SlackResponse(
        client=None,
        http_verb="POST",
        api_url="https://slack.com/api/chat.postMessage",
        req_args={},
        data={"ok": True, "ts": "400.4", "channel": "C1"},
        headers={},
        status_code=200,
    )

    remember_anima_slack_post("mei", "C1", response)

    assert lookup_notification_mapping("400.4")["anima"] == "mei"


@pytest.mark.parametrize("response", [None, {}, 7])
def test_missing_ts_is_ignored(response: Any) -> None:
    remember_anima_slack_post("mei", "C1", response)

    assert not (reply_routing._map_path()).exists()


def test_dispatch_slack_send_remembers_post(tmp_path, monkeypatch) -> None:
    from core.integrations import slack

    class _Client:
        def __init__(self, token: str) -> None:
            pass

        def resolve_channel(self, channel: str) -> str:
            return "C1"

        def post_message(self, *args: Any, **kwargs: Any) -> dict[str, str]:
            return {"ts": "300.3", "channel": "C1"}

    monkeypatch.setattr(slack, "SlackClient", _Client)
    monkeypatch.setattr(slack, "_resolve_slack_token", lambda args: "xoxb-test")
    monkeypatch.setattr(slack, "_resolve_slack_identity", lambda args: ("mei", ""))
    monkeypatch.setattr(slack, "_record_external_reply", lambda *a, **k: None)

    slack._dispatch_slack_send({"anima_dir": str(tmp_path / "animas" / "mei"), "channel": "#x", "message": "hi"})

    assert lookup_notification_mapping("300.3")["anima"] == "mei"
