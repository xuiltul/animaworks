"""Tests for hub wakes, the round-trip limit, and the silent-member alert."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from core.config.vault import VaultManager
from core.hub import wake as wake_module
from core.hub.members import HubMember, last_seen, member_path, save_member, state_path, token_digest, touch_last_seen
from core.hub.silence import check_silent_members
from core.messaging.board import (
    BoardPost,
    anima_principal,
    clear_post_hooks,
    hub_principal,
    post_to_board_in_shared_dir,
)
from core.messaging.board_deliveries import append_delivery, load_deliveries
from core.messaging.board_fanout import fanout_board_mentions
from core.messaging.messenger import ChannelMeta, save_channel_meta


@pytest.fixture(autouse=True)
def _clear_hooks():
    from core.activity.logger import ActivityLogger

    # The live-event throttle is class-level and keyed by anima name.  Reset it
    # so channel_post activity in this module cannot starve later tests.
    ActivityLogger._live_rate_limiter.reset()
    clear_post_hooks()
    yield
    clear_post_hooks()
    ActivityLogger._live_rate_limiter.reset()


def _hub_config(monkeypatch: pytest.MonkeyPatch, *, board: str = "hub", max_round_trips: int = 4) -> None:
    config = SimpleNamespace(
        hub=SimpleNamespace(
            enabled=True,
            timeline_enabled=False,
            company="a",
            board=board,
            public_hosts=[],
            max_posts_per_hour=20,
            max_round_trips=max_round_trips,
        ),
        human_notification=SimpleNamespace(channels=[], web_ui=False),
    )
    monkeypatch.setattr("core.config.models.load_config", lambda: config)


def _shared(tmp_path: Path) -> Path:
    shared = tmp_path / "shared"
    (shared / "channels").mkdir(parents=True)
    (shared / "channels" / "hub.jsonl").touch()
    save_channel_meta(shared, "hub", ChannelMeta(members=[], company="a"))
    return shared


def _anima(shared: Path, name: str, *, company: str = "a") -> None:
    anima_dir = shared.parent / "animas" / name
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True, "company": company}), encoding="utf-8")


def _member(member_id: str = "grok", **overrides: object) -> HubMember:
    data: dict[str, object] = {
        "id": member_id,
        "display_name": member_id.title(),
        "kind": "vendor_bot",
        "company": "a",
        "enabled": True,
        "token_sha256": token_digest(f"{member_id}-token"),
        "message_route": {"type": "poll"},
    }
    data.update(overrides)
    return HubMember(**data)  # type: ignore[arg-type]


def _post(
    *,
    channel: str = "hub",
    text: str,
    post_id: str = "p1",
    from_person: str = "grok",
    principal=None,
    reply_to: str | None = None,
    kind: str = "external_agent",
) -> BoardPost:
    if principal is None:
        principal = hub_principal(_member(from_person))
    return BoardPost(
        channel=channel,
        ts="2026-01-01T00:00:00+00:00",
        from_person=from_person,
        text=text,
        source="hub",
        id=post_id,
        kind=kind,
        origin="hub",
        principal=principal,
        reply_to=reply_to,
    )


def _write_post(shared: Path, record: dict[str, object]) -> None:
    path = shared / "channels" / "hub.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def test_webhook_wake_sends_bearer_payload_and_records_notified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    save_member(shared, _member("grok"))
    member = _member(
        "muse",
        message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"},
    )
    save_member(shared, member)
    vault = VaultManager(tmp_path)
    vault.store("shared", "hub.url", "https://hooks.example.com/wake")
    vault.store("shared", "hub.key", "s3cret")

    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_send(url: str, key: str, payload: dict[str, object]) -> tuple[bool, str]:
        calls.append((url, key, payload))
        return True, ""

    monkeypatch.setattr(wake_module, "send_webhook", fake_send)

    result = post_to_board_in_shared_dir(
        shared,
        "hub",
        "@muse check this " + "x" * 600,
        hub_principal(_member("grok")),
        origin="hub",
    )
    assert result.committed
    wake_module.wait_for_pending_wakes(timeout=5.0)

    assert len(calls) == 1
    url, key, payload = calls[0]
    assert url == "https://hooks.example.com/wake"
    assert key == "s3cret"
    assert payload["board"] == "hub"
    assert payload["post_id"] == result.id
    assert payload["from"] == "grok"
    assert payload["request_id"] == result.id
    assert payload["text"] == ("@muse check this " + "x" * 600)[:500]
    assert payload["task"].startswith("AnimaWorks のハブで @muse 宛ての投稿があります。")

    delivery = load_deliveries(shared, "hub")[result.id]
    assert delivery["notified"] == ["muse"]
    assert delivery["not_notified"] == {}


def test_send_webhook_uses_bearer_and_no_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 204

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(wake_module.httpx, "Client", FakeClient)
    ok, reason = wake_module.send_webhook("https://hooks.example.com/x", "key", {"a": 1})

    assert (ok, reason) == (True, "")
    assert captured["url"] == "https://hooks.example.com/x"
    assert captured["headers"] == {"Authorization": "Bearer key", "Content-Type": "application/json"}
    kwargs = captured["client_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["follow_redirects"] is False
    assert kwargs["timeout"] == wake_module.WEBHOOK_TIMEOUT_S


@pytest.mark.parametrize(
    ("url", "secret_present", "sender_result", "expected"),
    [
        ("https://hooks.example.com/x", True, (False, "http_404"), "http_404"),
        ("https://hooks.example.com/x", True, "timeout", "timeout"),
        ("https://hooks.example.com/x", False, (True, ""), "no_secret"),
        ("http://hooks.example.com/x", True, (True, ""), "bad_route"),
    ],
)
def test_webhook_failures_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    secret_present: bool,
    sender_result: object,
    expected: str,
) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    member = _member(message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"})
    save_member(shared, member)
    if secret_present:
        vault = VaultManager(tmp_path)
        vault.store("shared", "hub.url", url)
        vault.store("shared", "hub.key", "s3cret")

    def fake_send(_url: str, _key: str, _payload: dict[str, object]) -> tuple[bool, str]:
        if sender_result == "timeout":
            raise httpx.TimeoutException("slow")
        return sender_result  # type: ignore[return-value]

    monkeypatch.setattr(wake_module, "send_webhook", fake_send)

    post = _post(text="@grok hello", post_id="p-fail")
    notified, not_notified = wake_module.deliver_wake(shared, "hub", post, member)

    assert notified == []
    assert not_notified == {"grok": expected}
    assert load_deliveries(shared, "hub")["p-fail"]["not_notified"] == {"grok": expected}


def test_poll_member_is_recorded_as_polls_without_sending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    save_member(shared, _member("grok"))
    member = _member("muse", message_route={"type": "poll"})
    save_member(shared, member)
    sent: list[object] = []
    monkeypatch.setattr(wake_module, "send_webhook", lambda *args, **kwargs: sent.append(args) or (True, ""))

    post = _post(text="@muse read this", post_id="p-poll", from_person="grok")
    result = post_to_board_in_shared_dir(
        shared,
        "hub",
        post.text,
        hub_principal(_member("grok")),
        origin="hub",
    )

    assert result.notified == []
    assert result.not_notified == {"muse": "polls"}
    assert sent == []
    assert load_deliveries(shared, "hub")[result.id]["not_notified"] == {"muse": "polls"}


def test_member_is_not_a_target_on_other_boards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch, board="hub")
    shared = _shared(tmp_path)
    (shared / "channels" / "general.jsonl").touch()
    save_channel_meta(shared, "general", ChannelMeta(members=[], company="a"))
    member = _member(message_route={"type": "poll"})
    save_member(shared, member)

    post = _post(channel="general", text="@grok hello", post_id="p-other")
    notified, not_notified = fanout_board_mentions(shared, post)

    assert notified == []
    assert not_notified == {}


def test_external_all_wakes_nobody_and_anima_all_skips_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    _anima(shared, "bob")
    member = _member("muse", message_route={"type": "poll"})
    save_member(shared, member)

    external_all = _post(text="@all hello", post_id="p-all-hub", from_person="grok")
    assert fanout_board_mentions(shared, external_all) == ([], {})

    anima_all = _post(
        text="@all hello",
        post_id="p-all-anima",
        from_person="alice",
        kind="anima",
        principal=anima_principal("alice", animas_dir=shared.parent / "animas"),
    )
    notified, not_notified = fanout_board_mentions(shared, anima_all)

    assert notified == ["bob"]
    assert "muse" not in notified
    assert "muse" not in not_notified


def test_hub_member_to_anima_cross_company_is_not_notified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice", company="b")
    member = _member("grok", company="a", message_route={"type": "poll"})
    save_member(shared, member)

    post = _post(text="@alice hello", post_id="p-cross")
    notified, not_notified = fanout_board_mentions(shared, post)

    assert notified == []
    assert not_notified == {"alice": "cross_company"}


def _write_round_trip_chain(shared: Path) -> None:
    _write_post(shared, {"id": "p1", "kind": "anima", "from": "alice", "reply_to": None})
    _write_post(shared, {"id": "p2", "kind": "external_agent", "from": "grok", "reply_to": "p1"})
    _write_post(shared, {"id": "p3", "kind": "anima", "from": "alice", "reply_to": "p2"})


def test_loop_limit_wakes_nobody_and_alerts_once_per_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch, max_round_trips=1)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    member = _member("muse", message_route={"type": "poll"})
    save_member(shared, member)
    _write_round_trip_chain(shared)

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        wake_module,
        "notify_human",
        lambda subject, body, priority="normal", **kwargs: calls.append((subject, body, priority)),
    )

    current = _post(text="@alice @muse ping", post_id="p4", reply_to="p3")
    transitions, top_id = wake_module.count_round_trips(shared, "hub", current)
    assert transitions == 3
    assert top_id == "p1"

    notified, not_notified = fanout_board_mentions(shared, current)
    assert notified == []
    assert not_notified == {"alice": "loop_limit", "muse": "loop_limit"}
    assert len(calls) == 1
    assert calls[0][0] == "ハブの往復が上限に達しました"
    assert calls[0][2] == "high"
    assert "p4" in calls[0][1]
    assert json.loads(state_path(shared, "unused").with_name("loop_alerts.json").read_text())["p1"]

    _write_post(shared, {"id": "p4", "kind": "external_agent", "from": "grok", "reply_to": "p3"})
    follow_up = _post(text="@alice @muse again", post_id="p5", reply_to="p4")
    fanout_board_mentions(shared, follow_up)
    assert len(calls) == 1


def test_load_deliveries_merges_records_for_the_same_post(tmp_path: Path) -> None:
    shared = _shared(tmp_path)
    append_delivery(shared, "hub", post_id="p1", ts="t1", notified=["alice"], not_notified={"muse": "polls"})
    append_delivery(shared, "hub", post_id="p1", ts="t2", notified=["grok"], not_notified={"muse": "polls"})

    record = load_deliveries(shared, "hub")["p1"]
    assert record["notified"] == ["alice", "grok"]
    assert record["not_notified"] == {"muse": "polls"}


class _FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def notify(self, subject: str, body: str, priority: str = "normal", **kwargs: object) -> list[str]:
        self.calls.append((subject, body, priority))
        return ["ok"]


def test_silence_alert_is_once_per_last_seen_and_repeats_after_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _shared(tmp_path)
    member = _member("grok", expect_every="1h")
    save_member(shared, member)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    old_seen = (now - timedelta(hours=2)).isoformat()
    state_path(shared, "grok").parent.mkdir(parents=True, exist_ok=True)
    state_path(shared, "grok").write_text(json.dumps({"last_seen": old_seen}), encoding="utf-8")
    notifier = _FakeNotifier()

    assert check_silent_members(shared, now=now, notifier=notifier) == ["grok"]
    assert len(notifier.calls) == 1
    assert notifier.calls[0][0] == "Grok がハブに来ていません"
    assert notifier.calls[0][2] == "normal"
    assert old_seen in notifier.calls[0][1]
    assert json.loads(state_path(shared, "grok").read_text())["silence_alerted_for"] == old_seen

    assert check_silent_members(shared, now=now, notifier=notifier) == []
    assert len(notifier.calls) == 1

    newer_seen = (now - timedelta(hours=1, minutes=30)).isoformat()
    state_path(shared, "grok").write_text(
        json.dumps({"last_seen": newer_seen, "silence_alerted_for": old_seen}), encoding="utf-8"
    )
    assert check_silent_members(shared, now=now, notifier=notifier) == ["grok"]
    assert len(notifier.calls) == 2


def test_silence_uses_member_file_creation_time_and_skips_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _shared(tmp_path)
    save_member(shared, _member("grok", expect_every="1h"))
    save_member(shared, _member("muse", expect_every="1h", enabled=False))
    old_stamp = (datetime(2026, 1, 1, 10, 0, tzinfo=UTC)).timestamp()
    os.utime(member_path(shared, "grok"), (old_stamp, old_stamp))
    os.utime(member_path(shared, "muse"), (old_stamp, old_stamp))
    notifier = _FakeNotifier()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    assert check_silent_members(shared, now=now, notifier=notifier) == ["grok"]
    assert len(notifier.calls) == 1
    assert "muse" not in notifier.calls[0][0].lower()


def test_touch_last_seen_preserves_silence_marker(tmp_path: Path) -> None:
    shared = _shared(tmp_path)
    save_member(shared, _member("grok"))
    state_path(shared, "grok").parent.mkdir(parents=True, exist_ok=True)
    state_path(shared, "grok").write_text(json.dumps({"silence_alerted_for": "old"}), encoding="utf-8")

    touch_last_seen(shared, "grok")

    state = json.loads(state_path(shared, "grok").read_text())
    assert state["silence_alerted_for"] == "old"
    assert last_seen(shared, "grok") is not None


def test_suppressed_anima_reply_wakes_only_external_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    save_member(
        shared,
        _member("muse", message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"}),
    )
    vault = VaultManager(tmp_path)
    vault.store("shared", "hub.url", "https://hooks.example.com/wake")
    vault.store("shared", "hub.key", "s3cret")
    calls: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        wake_module,
        "send_webhook",
        lambda url, key, payload: calls.append((url, key, payload)) or (True, ""),
    )

    result = post_to_board_in_shared_dir(
        shared,
        "hub",
        "@muse @alice quick reply",
        anima_principal("alice", animas_dir=shared.parent / "animas"),
        origin="internal",
        notify=False,
    )
    assert result.committed
    wake_module.wait_for_pending_wakes(timeout=5.0)

    assert len(calls) == 1
    assert calls[0][2]["from"] == "alice"
    delivery = load_deliveries(shared, "hub")[result.id]
    assert delivery["notified"] == ["muse"]
    assert delivery["not_notified"] == {}
    assert "alice" not in delivery["notified"]


def test_suppressed_anima_reply_records_poll_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    save_member(shared, _member("muse", message_route={"type": "poll"}))

    result = post_to_board_in_shared_dir(
        shared,
        "hub",
        "@muse check",
        anima_principal("alice", animas_dir=shared.parent / "animas"),
        origin="internal",
        notify=False,
    )

    assert result.notified == []
    assert result.not_notified == {"muse": "polls"}
    delivery = load_deliveries(shared, "hub")[result.id]
    assert delivery["notified"] == []
    assert delivery["not_notified"] == {"muse": "polls"}


def test_suppressed_anima_reply_does_not_wake_on_other_board_or_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hub_config(monkeypatch)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    (shared / "channels" / "other.jsonl").touch()
    save_channel_meta(shared, "other", ChannelMeta(members=[], company="a"))
    save_member(shared, _member("muse"))

    other = post_to_board_in_shared_dir(
        shared,
        "other",
        "@muse check",
        anima_principal("alice", animas_dir=shared.parent / "animas"),
        origin="internal",
        notify=False,
    )
    mirror = post_to_board_in_shared_dir(
        shared,
        "hub",
        "@muse check",
        anima_principal("alice", animas_dir=shared.parent / "animas"),
        origin="mirror",
        notify=False,
    )

    assert other.notified == [] and other.not_notified == {}
    assert mirror.notified == [] and mirror.not_notified == {}
    assert load_deliveries(shared, "hub") == {}


def test_suppressed_anima_reply_records_loop_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _hub_config(monkeypatch, max_round_trips=0)
    shared = _shared(tmp_path)
    _anima(shared, "alice")
    save_member(shared, _member("muse", message_route={"type": "webhook", "url_ref": "hub.url", "key_ref": "hub.key"}))
    _write_round_trip_chain(shared)
    monkeypatch.setattr(wake_module, "notify_human", lambda *args, **kwargs: [])

    result = post_to_board_in_shared_dir(
        shared,
        "hub",
        "@muse reply",
        anima_principal("alice", animas_dir=shared.parent / "animas"),
        origin="internal",
        notify=False,
        reply_to="p3",
    )

    assert result.not_notified == {"muse": "loop_limit"}
    delivery = load_deliveries(shared, "hub")[result.id]
    assert delivery["not_notified"] == {"muse": "loop_limit"}
