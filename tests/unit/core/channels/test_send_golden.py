"""Golden request contracts for external channel send paths.

These tests capture the wire-level details before the send clients are
centralized under ``core.channels``. They deliberately assert authentication
headers as well as URL, method, and payload so later refactors cannot
accidentally change which credential or endpoint a path uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from core.schemas import Message


@pytest.fixture
def http_capture(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Route httpx clients through a MockTransport and record requests."""
    requests: list[dict] = []
    real_async_client = httpx.AsyncClient
    real_sync_client = httpx.Client

    def capture(request: httpx.Request) -> httpx.Response:
        content_type = request.headers.get("content-type", "")
        body_json = None
        body_form = None
        if request.content and "application/json" in content_type:
            body_json = json.loads(request.content)
        elif request.content and "application/x-www-form-urlencoded" in content_type:
            body_form = {
                key: values[0] if len(values) == 1 else values
                for key, values in parse_qs(request.content.decode(), keep_blank_values=True).items()
            }

        requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": {
                    "Authorization": request.headers.get("Authorization"),
                    "X-ChatWorkToken": request.headers.get("X-ChatWorkToken"),
                },
                "json": body_json,
                "data": body_form,
                "params": dict(request.url.params),
            }
        )

        path = request.url.path
        if request.url.host == "slack.com" and path == "/api/chat.postMessage":
            response_data = {"ok": True, "ts": "1710000000.000001"}
            if isinstance(body_json, dict) and body_json.get("channel"):
                response_data["channel"] = body_json["channel"]
        elif path == "/api/v10/users/@me/channels":
            response_data = {"id": "dm-channel"}
        elif path == "/api/v10/channels/dm-channel/messages":
            response_data = {"id": "discord-message"}
        elif path == "/api/v10/channels/C-WEBHOOK/webhooks" and request.method == "GET":
            response_data = []
        elif path == "/api/v10/channels/C-WEBHOOK/webhooks":
            response_data = {"id": "created-webhook", "token": "created-secret"}
        elif "/webhooks/" in path:
            response_data = {"id": "webhook-message"}
        elif request.url.host == "api.chatwork.com":
            response_data = {"message_id": "chatwork-message"}
        else:
            response_data = {}

        return httpx.Response(200, json=response_data, request=request)

    def async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(capture)
        return real_async_client(*args, **kwargs)

    def sync_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(capture)
        return real_sync_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(httpx, "Client", sync_client)
    return requests


def _assert_http_request(
    request: dict,
    *,
    method: str,
    url: str,
    authorization: str | None = None,
    chatwork_token: str | None = None,
    json_body: dict | None = None,
    form_body: dict | None = None,
    params: dict | None = None,
) -> None:
    assert request == {
        "method": method,
        "url": url,
        "headers": {
            "Authorization": authorization,
            "X-ChatWorkToken": chatwork_token,
        },
        "json": json_body,
        "data": form_body,
        "params": params or {},
    }


def _install_fake_slack_sdk(monkeypatch: pytest.MonkeyPatch, calls: list[dict], token: str) -> None:
    """Record Slack SDK calls at its Web API boundary (chat.postMessage)."""
    from core.integrations import _slack_client

    class FakeWebClient:
        def __init__(self, *, token: str) -> None:
            self.token = token

        def chat_postMessage(self, **kwargs):
            calls.append(
                {
                    "method": "POST",
                    "url": "https://slack.com/api/chat.postMessage",
                    "headers": {
                        "Authorization": f"Bearer {self.token}",
                        "X-ChatWorkToken": None,
                    },
                    "json": kwargs,
                    "data": None,
                    "params": {},
                }
            )
            return {"ok": True, "ts": "1710000000.000002", "channel": kwargs["channel"]}

    monkeypatch.setattr(_slack_client, "WebClient", FakeWebClient)
    monkeypatch.setattr(_slack_client, "_require_slack_sdk", lambda: FakeWebClient)
    monkeypatch.setattr(_slack_client, "get_credential", lambda *_args, **_kwargs: token)


def test_outbound_slack_request_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.messaging.outbound import ResolvedRecipient, send_external

    sdk_calls: list[dict] = []
    _install_fake_slack_sdk(monkeypatch, sdk_calls, "slack-outbound-token")
    result = send_external(
        ResolvedRecipient(is_internal=False, name="owner", channel="slack", slack_user_id="U123456789"),
        "hello",
        sender_name="sakura",
    )

    assert json.loads(result)["status"] == "sent"
    assert len(sdk_calls) == 1
    _assert_http_request(
        sdk_calls[0],
        method="POST",
        url="https://slack.com/api/chat.postMessage",
        authorization="Bearer slack-outbound-token",
        json_body={"channel": "U123456789", "text": "[sakura] hello", "username": "sakura"},
    )


def test_outbound_discord_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
) -> None:
    from core.messaging.outbound import ResolvedRecipient, send_external

    monkeypatch.setattr("core.credentials.get_credential", lambda *_args, **_kwargs: "discord-outbound-token")
    result = send_external(
        ResolvedRecipient(is_internal=False, name="owner", channel="discord", discord_user_id="123456789012345678"),
        "hello",
        sender_name="sakura",
    )

    assert json.loads(result)["status"] == "sent"
    assert len(http_capture) == 2
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/v10/users/@me/channels",
        authorization="Bot discord-outbound-token",
        json_body={"recipient_id": "123456789012345678"},
    )
    _assert_http_request(
        http_capture[1],
        method="POST",
        url="https://discord.com/api/v10/channels/dm-channel/messages",
        authorization="Bot discord-outbound-token",
        json_body={"content": "[sakura] hello"},
    )


def test_outbound_chatwork_request_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.integrations import _chatwork_client
    from core.messaging.outbound import ResolvedRecipient, send_external

    requests: list[dict] = []

    class FakeResponse:
        status_code = 200
        headers: dict[str, str] = {}
        text = '{"message_id":"chatwork-message"}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"message_id": "chatwork-message"}

    class FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def request(self, method: str, url: str, **kwargs) -> FakeResponse:
            requests.append(
                {
                    "method": method,
                    "url": url,
                    "headers": {
                        "Authorization": self.headers.get("Authorization"),
                        "X-ChatWorkToken": self.headers.get("X-ChatWorkToken"),
                    },
                    "json": kwargs.get("json"),
                    "data": kwargs.get("data"),
                    "params": kwargs.get("params") or {},
                }
            )
            return FakeResponse()

    monkeypatch.setattr(_chatwork_client, "requests", SimpleNamespace(Session=FakeSession))
    monkeypatch.setenv("CHATWORK_API_TOKEN__owner", "chatwork-outbound-token")
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

    result = send_external(
        ResolvedRecipient(is_internal=False, name="owner", channel="chatwork", chatwork_room_id="42"),
        "hello",
        sender_name="sakura",
    )

    assert json.loads(result)["status"] == "sent"
    assert requests == [
        {
            "method": "POST",
            "url": "https://api.chatwork.com/v2/rooms/42/messages",
            "headers": {"Authorization": None, "X-ChatWorkToken": "chatwork-outbound-token"},
            "json": None,
            "data": {"body": "[sakura] hello"},
            "params": {},
        }
    ]


@pytest.mark.asyncio
async def test_notification_slack_bot_request_golden(http_capture: list[dict]) -> None:
    from core.notification.channels.slack import SlackChannel

    result = await SlackChannel({"bot_token": "slack-notification-token", "channel": "C123"}).send(
        "Alert", "Body", "high"
    )

    assert result == "slack: OK"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://slack.com/api/chat.postMessage",
        authorization="Bearer slack-notification-token",
        json_body={"channel": "C123", "text": "[HIGH] *Alert*\nBody"},
    )


@pytest.mark.asyncio
async def test_notification_slack_webhook_request_golden(http_capture: list[dict]) -> None:
    from core.notification.channels.slack import SlackChannel

    result = await SlackChannel({"webhook_url": "https://hooks.slack.com/services/T/B/X"}).send("Alert", "Body")

    assert result == "slack: OK"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://hooks.slack.com/services/T/B/X",
        json_body={"text": "*Alert*\nBody"},
    )


@pytest.mark.asyncio
async def test_notification_discord_dm_request_golden(http_capture: list[dict]) -> None:
    from core.notification.channels.discord import DiscordChannel

    result = await DiscordChannel({"bot_token": "discord-notification-token", "user_id": "123456789012345678"}).send(
        "Alert", "Body", "urgent"
    )

    assert result.startswith("discord: DM sent to 123456789012345678")
    assert len(http_capture) == 2
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/v10/users/@me/channels",
        authorization="Bot discord-notification-token",
        json_body={"recipient_id": "123456789012345678"},
    )
    _assert_http_request(
        http_capture[1],
        method="POST",
        url="https://discord.com/api/v10/channels/dm-channel/messages",
        authorization="Bot discord-notification-token",
        json_body={"content": "**[URGENT]** **Alert**\nBody"},
    )


@pytest.mark.asyncio
async def test_notification_discord_channel_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
    tmp_path: Path,
) -> None:
    from core.messaging import discord_webhooks
    from core.notification.channels.discord import DiscordChannel

    monkeypatch.setattr(discord_webhooks, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(discord_webhooks, "get_credential", lambda *_args, **_kwargs: "discord-manager-token")
    monkeypatch.setattr(discord_webhooks, "resolve_anima_icon_url", lambda _name: "")
    manager = discord_webhooks.DiscordWebhookManager()
    manager._webhooks["C-WEBHOOK"] = {"id": "webhook-id", "token": "webhook-secret"}
    monkeypatch.setattr(discord_webhooks, "get_webhook_manager", lambda: manager)

    result = await DiscordChannel({"bot_token": "discord-channel-config-token", "channel_id": "C-WEBHOOK"}).send(
        "Alert", "Body"
    )

    assert result == "discord: channel C-WEBHOOK (msg_id=webhook-message)"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/v10/webhooks/webhook-id/webhook-secret?wait=true",
        authorization="Bot discord-manager-token",
        json_body={"content": "**Alert**\nBody", "username": "AnimaWorks"},
        params={"wait": "true"},
    )


@pytest.mark.asyncio
async def test_notification_discord_webhook_request_golden(http_capture: list[dict]) -> None:
    from core.notification.channels.discord import DiscordChannel

    result = await DiscordChannel({"webhook_url": "https://discord.com/api/webhooks/55/secret"}).send("Alert", "Body")

    assert result == "discord: webhook sent"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/webhooks/55/secret?wait=true",
        json_body={"content": "**Alert**\nBody"},
        params={"wait": "true"},
    )


@pytest.mark.asyncio
async def test_notification_chatwork_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
) -> None:
    from core.notification.channels.chatwork import ChatworkChannel

    monkeypatch.setenv("CW_NOTIFICATION_TOKEN", "chatwork-notification-token")
    result = await ChatworkChannel({"api_token_env": "CW_NOTIFICATION_TOKEN", "room_id": "42"}).send("Alert", "Body")

    assert result == "chatwork: OK"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://api.chatwork.com/v2/rooms/42/messages",
        chatwork_token="chatwork-notification-token",
        form_body={"body": "[info][title]Alert[/title]Body[/info]"},
    )


@pytest.mark.asyncio
async def test_call_human_uses_notification_slack_channel(http_capture: list[dict]) -> None:
    from core.notification.channels.slack import SlackChannel

    result = await SlackChannel({"bot_token": "call-human-token", "channel": "C123"}).send("Alert", "Body")

    assert result == "slack: OK"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://slack.com/api/chat.postMessage",
        authorization="Bearer call-human-token",
        json_body={"channel": "C123", "text": "*Alert*\nBody"},
    )


@pytest.mark.asyncio
async def test_slack_auto_responder_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
    tmp_path: Path,
) -> None:
    from core.messaging.messenger import InboxItem
    from core.messaging.outbound_auto import SlackAutoResponder

    monkeypatch.setattr("core.messaging.outbound_auto._resolve_bot_token", lambda _name: "slack-auto-token")
    monkeypatch.setattr("core.messaging.outbound_auto._resolve_avatar_url", lambda _name: "")
    message = Message(
        from_person="human",
        to_person="sakura",
        content="question",
        source="slack",
        source_message_id="1710000000.000010",
        external_user_id="U123",
        external_channel_id="C123",
        external_thread_ts="1710000000.000009",
    )

    result = await SlackAutoResponder().on_inbox_response(
        "sakura", "Hello", [InboxItem(msg=message, path=tmp_path / "message.json")]
    )

    assert result == ["1710000000.000001"]
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://slack.com/api/chat.postMessage",
        authorization="Bearer slack-auto-token",
        json_body={
            "channel": "C123",
            "text": "<@U123> Hello",
            "thread_ts": "1710000000.000009",
            "username": "sakura",
        },
    )


@pytest.mark.asyncio
async def test_board_slack_sync_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
) -> None:
    from core.messaging.outbound_auto import BoardSlackSync

    slack_config = SimpleNamespace(
        enabled=True,
        board_outbound_sync=[],
        board_outbound_sync_all=True,
        board_mapping={"C456": "general"},
    )
    monkeypatch.setattr(
        "core.config.models.load_config",
        lambda: SimpleNamespace(external_messaging=SimpleNamespace(slack=slack_config)),
    )
    monkeypatch.setattr("core.messaging.outbound_auto._resolve_bot_token", lambda _name: "slack-board-token")
    monkeypatch.setattr("core.messaging.outbound_auto._resolve_avatar_url", lambda _name: "")

    result = await BoardSlackSync().sync_board_post("general", "Board update", "sakura")

    assert result == "sent"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://slack.com/api/chat.postMessage",
        authorization="Bearer slack-board-token",
        json_body={"channel": "C456", "text": "Board update", "username": "sakura"},
    )


@pytest.mark.asyncio
async def test_discord_auto_responder_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
    tmp_path: Path,
) -> None:
    from core.messaging import discord_webhooks
    from core.messaging.messenger import InboxItem
    from core.messaging.outbound_auto import DiscordAutoResponder

    monkeypatch.setattr(discord_webhooks, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(discord_webhooks, "get_credential", lambda *_args, **_kwargs: "discord-auto-token")
    monkeypatch.setattr(discord_webhooks, "resolve_anima_icon_url", lambda _name: "")
    manager = discord_webhooks.DiscordWebhookManager()
    manager._webhooks["C-WEBHOOK"] = {"id": "webhook-id", "token": "webhook-secret"}
    monkeypatch.setattr(discord_webhooks, "get_webhook_manager", lambda: manager)
    message = Message(
        from_person="human",
        to_person="sakura",
        content="question",
        source="discord",
        source_message_id="discord-source-message",
        external_user_id="123456789012345678",
        external_channel_id="C-WEBHOOK",
    )

    result = await DiscordAutoResponder().on_inbox_response(
        "sakura", "Hello", [InboxItem(msg=message, path=tmp_path / "message.json")]
    )

    assert result == ["webhook-message"]
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/v10/webhooks/webhook-id/webhook-secret?wait=true",
        authorization="Bot discord-auto-token",
        json_body={
            "content": "<@123456789012345678> Hello",
            "username": "sakura",
        },
        params={"wait": "true"},
    )


def test_discord_webhook_manager_request_golden(
    monkeypatch: pytest.MonkeyPatch,
    http_capture: list[dict],
    tmp_path: Path,
) -> None:
    from core.messaging import discord_webhooks

    monkeypatch.setattr(discord_webhooks, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(discord_webhooks, "get_credential", lambda *_args, **_kwargs: "discord-webhook-bot-token")
    monkeypatch.setattr(discord_webhooks, "resolve_anima_icon_url", lambda _name: "")
    manager = discord_webhooks.DiscordWebhookManager()
    manager._webhooks["C-WEBHOOK"] = {"id": "webhook-id", "token": "webhook-secret"}

    message_id = manager.send_as_anima(
        "C-WEBHOOK",
        "sakura",
        "Hello from webhook",
        components=[{"type": 1, "components": []}],
    )

    assert message_id == "webhook-message"
    assert len(http_capture) == 1
    _assert_http_request(
        http_capture[0],
        method="POST",
        url="https://discord.com/api/v10/webhooks/webhook-id/webhook-secret?wait=true",
        authorization="Bot discord-webhook-bot-token",
        json_body={
            "content": "Hello from webhook",
            "username": "sakura",
            "components": [{"type": 1, "components": []}],
        },
        params={"wait": "true"},
    )
