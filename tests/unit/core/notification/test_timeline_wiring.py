# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def routing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("core.notification.reply_routing.get_data_dir", lambda: tmp_path)
    return tmp_path


def test_notification_mapping_round_trips_key(routing_dir: Path) -> None:
    from core.notification.reply_routing import lookup_notification_mapping, save_notification_mapping

    save_notification_mapping(
        "123.456",
        "C1",
        "sakura",
        notification_text="subject\nbody",
        callback_id="cb-1",
        notification_key="key-1",
    )
    mapping = lookup_notification_mapping("123.456")
    assert mapping is not None
    assert mapping["notification_key"] == "key-1"
    assert mapping["callback_id"] == "cb-1"


@pytest.mark.asyncio
async def test_deliver_notification_reply_records_human_and_answer_link(routing_dir: Path) -> None:
    from core.notification.reply_routing import deliver_notification_reply

    with (
        patch("core.messaging.messenger.Messenger") as messenger_cls,
        patch("core.hub.timeline_human.resolve_person", return_value="alice"),
        patch("core.hub.timeline_human.find_notified", return_value="tl_1234567890abcdef") as find_notified,
        patch("core.hub.timeline_human.record_human") as record_human,
    ):
        messenger_cls.return_value.receive_external.return_value = SimpleNamespace(id="msg-1")
        assert await deliver_notification_reply(
            "sakura",
            "承認します",
            routing_dir,
            actor="alice",
            source="slack",
            external_user_id="U_ALICE",
            notification_key="key-1",
            callback_id="cb-1",
        )

    find_notified.assert_called_once_with(key="key-1", callback_id="cb-1")
    record_human.assert_called_once_with(
        person="alice",
        anima="sakura",
        text="承認します",
        via="slack_thread",
        about="tl_1234567890abcdef",
        ref="msg-1",
    )


@pytest.mark.asyncio
async def test_deliver_notification_reply_without_original_records_about_none(routing_dir: Path) -> None:
    from core.notification.reply_routing import deliver_notification_reply

    with (
        patch("core.messaging.messenger.Messenger") as messenger_cls,
        patch("core.hub.timeline_human.resolve_person", return_value="alice"),
        patch("core.hub.timeline_human.find_notified", return_value=None),
        patch("core.hub.timeline_human.record_human") as record_human,
    ):
        messenger_cls.return_value.receive_external.return_value = SimpleNamespace(id="msg-2")
        assert await deliver_notification_reply(
            "sakura",
            "返事",
            routing_dir,
            actor="alice",
            source="web",
        )

    assert record_human.call_args.kwargs["about"] is None
    assert record_human.call_args.kwargs["via"] == "notification_reply"


@pytest.mark.asyncio
async def test_deliver_notification_reply_skips_when_not_delivered(routing_dir: Path) -> None:
    from core.notification.reply_routing import deliver_notification_reply

    with (
        patch("core.messaging.messenger.Messenger") as messenger_cls,
        patch("core.hub.timeline_human.record_human") as record_human,
    ):
        messenger_cls.return_value.receive_external.return_value = None
        assert await deliver_notification_reply("sakura", "返事", routing_dir, source="web")

    record_human.assert_not_called()


@pytest.mark.asyncio
async def test_route_thread_reply_passes_notification_key(routing_dir: Path) -> None:
    from core.notification.reply_routing import (
        deliver_notification_reply,
        route_thread_reply,
        save_notification_mapping,
    )

    save_notification_mapping(
        "123.456",
        "C1",
        "sakura",
        notification_text="subject\nbody",
        callback_id="",
        notification_key="key-9",
    )
    with patch(
        "core.notification.reply_routing.deliver_notification_reply",
        new=AsyncMock(return_value=True),
    ) as deliver:
        routed = await route_thread_reply(
            {
                "thread_ts": "123.456",
                "ts": "123.999",
                "channel": "C1",
                "user": "U_ALICE",
                "text": "thread reply",
            },
            routing_dir / "shared",
        )

    assert routed is True
    assert deliver.await_args.kwargs["notification_key"] == "key-9"
    assert deliver.await_args.kwargs["callback_id"] == ""
    assert deliver_notification_reply is not None
