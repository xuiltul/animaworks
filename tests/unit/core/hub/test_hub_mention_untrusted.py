from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.anima.inbox import _SOURCE_TO_ORIGIN
from core.hub.members import HubMember, token_digest
from core.messaging.board import BoardPost, clear_post_hooks, hub_principal
from core.messaging.board_fanout import fanout_board_mentions
from core.messaging.messenger import ChannelMeta, save_channel_meta
from core.trust import ORIGIN_UNKNOWN, wrap_inbox_message


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_post_hooks()
    yield
    clear_post_hooks()


def test_hub_mention_arrives_as_untrusted_board_mention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_dir = tmp_path / "shared"
    (shared_dir / "channels").mkdir(parents=True)
    (shared_dir / "channels" / "hub.jsonl").touch()
    save_channel_meta(shared_dir, "hub", ChannelMeta(members=[], company="a"))
    alice_dir = tmp_path / "animas" / "alice"
    alice_dir.mkdir(parents=True)
    (alice_dir / "status.json").write_text(json.dumps({"enabled": True, "company": "a"}), encoding="utf-8")

    config = SimpleNamespace(
        hub=SimpleNamespace(
            enabled=True,
            timeline_enabled=True,
            company="a",
            board="hub",
            public_hosts=[],
            max_posts_per_hour=20,
            max_round_trips=4,
        )
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)

    member = HubMember(
        id="grok",
        display_name="Grok",
        company="a",
        token_sha256=token_digest("token"),
        message_route={"type": "poll"},
    )
    post = BoardPost(
        channel="hub",
        ts="2026-01-01T00:00:00+00:00",
        from_person="grok",
        text="@alice please check",
        source="hub",
        id="post1",
        kind="external_agent",
        origin="hub",
        principal=hub_principal(member),
    )

    notified, not_notified = fanout_board_mentions(shared_dir, post)

    assert notified == ["alice"]
    assert not_notified == {}
    inbox_path = shared_dir / "inbox" / "alice"
    messages = [json.loads(path.read_text(encoding="utf-8")) for path in inbox_path.glob("*.json")]
    assert len(messages) == 1
    message = messages[0]
    assert message["type"] == "board_mention"
    assert message["source"] == "hub"
    assert message["meta"] == {"board_channel": "hub", "board_post_id": "post1"}

    origin = _SOURCE_TO_ORIGIN.get("hub", ORIGIN_UNKNOWN)
    wrapped = wrap_inbox_message(message["content"], source="hub", origin=origin)
    assert wrapped.startswith('<external_message source="hub" trust="untrusted"')
    assert 'origin="unknown"' in wrapped
