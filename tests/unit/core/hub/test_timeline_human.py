# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import invalidate_cache
from core.hub.timeline import append_item, find_latest, list_items
from core.hub.timeline_human import (
    find_human_item,
    find_notified,
    mentioned_animas,
    record_answered,
    record_human,
    resolve_person,
)


def _configure(data_dir: Path, make_anima, *, company: str = "acme") -> None:
    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hub"] = {"enabled": True, "timeline_enabled": True, "company": company}
    config.setdefault("external_messaging", {})["user_aliases"] = {
        "alice": {"slack_user_id": "U_ALICE", "discord_user_id": "D_ALICE"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    for name in ("sakura", "muse"):
        anima_dir = make_anima(name)
        status_path = anima_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["company"] = company
        status_path.write_text(json.dumps(status), encoding="utf-8")
    invalidate_cache()


@pytest.fixture
def hub(data_dir: Path, make_anima):
    _configure(data_dir, make_anima)
    return data_dir


def _timeline(data_dir: Path):
    return list_items(data_dir / "shared", company="acme", limit=100)


def test_resolve_person_for_web_tui_slack_and_actors(hub: Path) -> None:
    assert resolve_person(via="web", auth_via="session", username="alice") == "alice"
    assert resolve_person(via="tui", auth_via="localhost", client_header="tui", username="alice") == "alice"
    assert resolve_person(via="web", auth_via="localhost", client_header="", username="alice") is None
    assert resolve_person(via="slack", external_user_id="U_ALICE") == "alice"
    assert resolve_person(via="discord", external_user_id="D_ALICE") == "alice"
    assert resolve_person(via="slack", external_user_id="U_UNKNOWN") is None


def test_resolve_person_rejects_animas_and_hub_members(hub: Path) -> None:
    assert resolve_person(via="approval", actor="tester", anima="sakura") == "tester"
    assert resolve_person(via="approval", actor="sakura", anima="sakura") is None
    assert resolve_person(via="notification_reply", actor="system", anima="sakura") is None
    assert resolve_person(via="notification_reply", actor="", anima="sakura") is None


def test_record_human_and_answered_round_trip(hub: Path) -> None:
    human_id = record_human(
        person="alice",
        anima="sakura",
        text="  本文です  ",
        via="web",
        ref="msg-1",
    )
    assert human_id

    item = find_latest(
        hub / "shared",
        company="acme",
        kind="human",
        ref="msg-1",
    )
    assert item is not None
    assert item["id"] == human_id
    assert item["from"] == "alice"
    assert item["text"] == "本文です"
    assert item["source"] == "human"
    assert item["to"] == "sakura"
    assert item["via"] == "web"

    answered_id = record_answered(
        anima="sakura",
        person="alice",
        text="返事です",
        via="web",
        about=human_id,
    )
    assert answered_id
    answered = [entry for entry in _timeline(hub) if entry["id"] == answered_id][0]
    assert answered["kind"] == "answered"
    assert answered["about"] == human_id
    assert answered["to"] == "alice"
    assert answered["source"] == "anima"


def test_record_human_truncates_and_skips_empty(hub: Path) -> None:
    human_id = record_human(person="alice", anima="sakura", text="x" * 5000, via="web")
    assert human_id
    item = find_latest(hub / "shared", company="acme", kind="human", ref=None)
    assert item is not None
    assert len(item["text"]) == 4000
    assert record_human(person="alice", anima="sakura", text="   ", via="web") is None


def test_company_gate_blocks_unknown_company(hub: Path, make_anima) -> None:
    other = make_anima("ritsu")
    status_path = other / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["company"] = "other"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    invalidate_cache()

    assert record_human(person="alice", anima="ritsu", text="hello", via="web") is None


def test_record_human_without_anima_uses_company_for_board(hub: Path) -> None:
    human_id = record_human(
        person="alice",
        text="board post",
        via="board",
        label="general",
        ref="post-1",
        to="sakura",
    )
    assert human_id
    item = find_human_item("post-1")
    assert item is not None
    assert item["label"] == "general"
    assert item["to"] == "sakura"


def test_find_notified_by_key_and_callback(hub: Path) -> None:
    notified = append_item(
        hub / "shared",
        from_name="system",
        company="acme",
        kind="notified",
        text="通知",
        source="system",
        key="notification-key",
        callback_id="cb-1",
    )
    assert find_notified(key="notification-key") == notified["id"]
    assert find_notified(callback_id="cb-1") == notified["id"]
    assert find_notified(key="missing", callback_id="missing") is None


def test_mentioned_animas_filters_unknown_and_all(hub: Path) -> None:
    assert mentioned_animas("@sakura @muse @all @unknown") == "muse, sakura"
    assert mentioned_animas("no mentions") is None


def test_record_inbox_answers_links_each_human_message(hub: Path) -> None:
    from core.anima.inbox import _record_inbox_answers

    first = record_human(person="alice", anima="sakura", text="one", via="slack", ref="m1")
    second = record_human(person="alice", anima="sakura", text="two", via="web", ref="m2")
    board = record_human(person="alice", text="board", via="board", ref="board-1")
    assert first and second and board

    anima_msg = record_human(person="alice", anima="sakura", text="looks human", via="slack", ref="m4")
    assert anima_msg

    items = [
        SimpleNamespace(msg=SimpleNamespace(id="m1", source="slack", meta={})),
        SimpleNamespace(msg=SimpleNamespace(id="m2", source="web", meta={})),
        SimpleNamespace(msg=SimpleNamespace(id="m3", source="anima", meta={"board_post_id": "board-1"})),
        SimpleNamespace(msg=SimpleNamespace(id="unknown", source="slack", meta={})),
        # Anima-to-anima traffic is skipped without scanning the timeline,
        # even if its id happens to collide with a recorded ref.
        SimpleNamespace(msg=SimpleNamespace(id="m4", source="anima", meta={})),
    ]
    _record_inbox_answers(items, "sakura", "まとめた返事")

    answered = [entry for entry in _timeline(hub) if entry["kind"] == "answered"]
    assert {entry["about"] for entry in answered} == {first, second, board}
    assert all(entry["to"] == "alice" for entry in answered)
    assert {entry["via"] for entry in answered} == {"slack", "web", "board"}
