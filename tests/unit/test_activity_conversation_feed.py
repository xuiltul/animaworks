"""Tests for the compact user-centered conversation feed."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.activity.conversation import clear_conversation_feed_cache
from core.activity.logger import ActivityLogger
from core.time_utils import now_local
from server.routes.sessions import create_sessions_router


def _write_entries(anima_dir: Path, entries: list[dict]) -> Path:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    day = entries[0]["ts"][:10]
    path = log_dir / f"{day}.jsonl"
    with path.open("a", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _event(ts: str, event_type: str, content: str = "", **fields: object) -> dict:
    return {"ts": ts, "type": event_type, "content": content, **fields}


def _test_activity(tmp_path: Path, name: str = "mei") -> tuple[Path, ActivityLogger]:
    anima_dir = tmp_path / "animas" / name
    (anima_dir / "activity_log").mkdir(parents=True)
    return anima_dir, ActivityLogger(anima_dir)


@pytest.fixture(autouse=True)
def clear_feed_cache() -> None:
    clear_conversation_feed_cache()
    yield
    clear_conversation_feed_cache()


def test_feed_finds_minimum_user_items_across_many_background_entries(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    entries: list[dict] = []
    for turn in range(14):
        start = now - timedelta(minutes=(14 - turn) * 3)
        for work in range(8):
            ts = (start + timedelta(seconds=work * 10)).isoformat()
            entries.append(
                _event(ts, "cron_executed", f"private cron body {turn}-{work}", meta={"task_name": "nightly"})
            )
        user_ts = (start + timedelta(seconds=90)).isoformat()
        entries.append(
            _event(
                user_ts,
                "message_received",
                f"question {turn}",
                from_person="human",
                channel="chat",
                meta={"from_type": "human", "thread_id": "default"},
            )
        )
    _write_entries(anima_dir, entries)

    feed = activity.get_conversation_feed(min_user_items=10, _schedule_prefetch=False)

    user_messages = [item["message"] for item in feed["items"] if item["kind"] == "message"]
    assert len(user_messages) >= 10
    assert all(message["role"] == "human" for message in user_messages)
    groups = [item for item in feed["items"] if item["kind"] == "bg_group"]
    assert groups
    assert all("content" not in group and "tool_calls" not in group for group in groups)
    assert feed["next_before"]


def test_file_metadata_is_exposed_by_feed_and_conversation_history(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    files = [
        {
            "name": "estimate.pdf",
            "path": "attachments/20261029_101500_0_estimate.pdf",
            "size": 42,
            "media_type": "application/pdf",
        }
    ]
    _write_entries(
        anima_dir,
        [
            _event(
                now.isoformat(),
                "message_received",
                "Please review",
                from_person="human",
                channel="chat",
                meta={"from_type": "human", "thread_id": "default", "files": files},
            )
        ],
    )

    history = activity.get_conversation_view(thread_id="default")
    history_messages = [message for session in history["sessions"] for message in session["messages"]]
    feed = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
    feed_messages = [item["message"] for item in feed["items"] if item["kind"] == "message"]

    assert history_messages[0]["files"] == files
    assert feed_messages[0]["files"] == files


def test_assistant_file_metadata_is_exposed_by_feed_and_conversation_history(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    files = [
        {
            "name": "summary.md",
            "path": "attachments/20261029_101500_a1_summary.md",
            "size": 128,
            "media_type": "text/markdown",
        }
    ]
    _write_entries(
        anima_dir,
        [
            _event(
                now.isoformat(),
                "response_sent",
                "Here is the summary",
                to_person="human",
                channel="chat",
                meta={"thread_id": "default", "files": files},
            )
        ],
    )

    history = activity.get_conversation_view(thread_id="default")
    history_messages = [message for session in history["sessions"] for message in session["messages"]]
    feed = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
    feed_messages = [item["message"] for item in feed["items"] if item["kind"] == "message"]

    assert history_messages[-1]["role"] == "assistant"
    assert history_messages[-1]["files"] == files
    assert feed_messages[-1]["files"] == files


def test_feed_coalesces_cron_dm_and_self_notice_without_leaking_bodies(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    (anima_dir.parent / "rin").mkdir()
    base = now_local().replace(microsecond=0) - timedelta(minutes=8)
    entries = [
        _event((base + timedelta(seconds=1)).isoformat(), "cron_executed", "cron secret"),
        _event((base + timedelta(seconds=2)).isoformat(), "cron_executed", "cron secret 2"),
        _event(
            (base + timedelta(seconds=3)).isoformat(),
            "message_sent",
            "dm secret",
            from_person="mei",
            to_person="rin",
            channel="chat",
        ),
        _event(
            (base + timedelta(seconds=4)).isoformat(),
            "message_received",
            "dm secret 2",
            from_person="rin",
            to_person="mei",
            channel="chat",
            meta={"from_type": "anima"},
        ),
        _event(
            (base + timedelta(seconds=5)).isoformat(),
            "message_sent",
            "self notice secret",
            from_person="mei",
            to_person="mei",
            channel="chat",
        ),
        _event(
            (base + timedelta(seconds=6)).isoformat(),
            "message_received",
            "visible human message",
            from_person="human",
            to_person="mei",
            channel="chat",
            meta={"from_type": "human"},
        ),
    ]
    _write_entries(anima_dir, entries)

    feed = activity.get_conversation_feed(min_user_items=10, _schedule_prefetch=False)

    assert len(feed["items"]) == 2
    first, last = feed["items"]
    assert first["kind"] == "bg_group"
    assert first["count"] == 5
    assert first["kinds"] == {"cron": 2, "dm": 2, "notice": 1}
    assert first["participants"] == ["rin"]
    assert "cron secret" not in json.dumps(first)
    assert "dm secret" not in json.dumps(first)
    assert last["kind"] == "message"


def test_user_item_rules_cover_external_humans_notifications_calls_and_direct_replies(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    (anima_dir.parent / "rin").mkdir()
    base = now_local().replace(microsecond=0) - timedelta(minutes=7)
    entries = [
        _event(
            (base + timedelta(seconds=1)).isoformat(),
            "message_received",
            "Slack from a person",
            from_person="slack-user",
            channel="slack",
            meta={"from_type": "slack"},
        ),
        _event(
            (base + timedelta(seconds=2)).isoformat(),
            "response_sent",
            "A direct reply",
            to_person="human",
            channel="chat",
        ),
        _event(
            (base + timedelta(seconds=3)).isoformat(),
            "response_sent",
            "Anima DM",
            to_person="rin",
            channel="chat",
        ),
        _event((base + timedelta(seconds=4)).isoformat(), "human_notify", "Approval needed"),
        _event(
            (base + timedelta(seconds=5)).isoformat(),
            "human_notify",
            "Private self notice",
            from_person="mei",
            to_person="mei",
        ),
        _event((base + timedelta(seconds=6)).isoformat(), "human_reply", "Approved", from_person="human"),
        _event(
            (base + timedelta(seconds=7)).isoformat(),
            "voice_call",
            "Call summary",
            summary="Call summary",
        ),
        _event(
            (base + timedelta(seconds=8)).isoformat(),
            "message_sent",
            "Self notice",
            from_person="mei",
            to_person="mei",
            channel="chat",
        ),
    ]
    _write_entries(anima_dir, entries)

    feed = activity.get_conversation_feed(min_user_items=10, _schedule_prefetch=False)
    user_texts = [item["message"].get("content") for item in feed["items"] if item["kind"] == "message"]

    assert user_texts == ["Slack from a person", "A direct reply", "Approval needed", "Approved", "Call summary"]
    bg_groups = [item for item in feed["items"] if item["kind"] == "bg_group"]
    assert bg_groups[0]["kinds"]["dm"] == 1
    assert bg_groups[0]["participants"] == ["rin"]
    assert bg_groups[-1]["kinds"]["notice"] == 1


def test_background_endpoint_returns_only_in_range_background_messages(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    base = now_local().replace(microsecond=0) - timedelta(minutes=5)
    cron_ts = (base + timedelta(seconds=2)).isoformat()
    user_ts = (base + timedelta(seconds=3)).isoformat()
    entries = [
        _event((base + timedelta(seconds=1)).isoformat(), "cron_executed", "before range"),
        _event(cron_ts, "cron_executed", "in range background"),
        _event(
            user_ts,
            "message_received",
            "in range human",
            from_person="human",
            channel="chat",
            meta={"from_type": "human"},
        ),
        _event((base + timedelta(seconds=4)).isoformat(), "cron_executed", "after range"),
    ]
    _write_entries(anima_dir, entries)

    result = activity.get_conversation_background(
        from_ts=cron_ts,
        to_ts=cron_ts,
    )

    messages = [message for session in result["sessions"] for message in session["messages"]]
    assert [message["content"] for message in messages] == ["in range background"]
    assert result["truncated"] is False


def test_feed_uses_scan_bound_as_a_cursor_when_recent_range_has_no_messages(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    old = now_local().replace(microsecond=0) - timedelta(days=15)
    _write_entries(
        anima_dir,
        [
            _event(
                old.isoformat(),
                "message_received",
                "older page",
                from_person="human",
                channel="chat",
                meta={"from_type": "human", "thread_id": "default"},
            )
        ],
    )

    recent = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
    older = activity.get_conversation_feed(
        min_user_items=1,
        before=recent["next_before"],
        _schedule_prefetch=False,
    )

    assert recent["items"] == []
    assert recent["has_more"] is True
    assert recent["next_before"]
    assert older["items"][0]["message"]["content"] == "older page"


def test_background_endpoint_limits_results_to_newest_200_items(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    start = now_local().replace(microsecond=0) - timedelta(minutes=10)
    entries = [
        _event((start + timedelta(seconds=index)).isoformat(), "cron_executed", f"cron {index}") for index in range(205)
    ]
    _write_entries(anima_dir, entries)

    result = activity.get_conversation_background(
        from_ts=entries[0]["ts"],
        to_ts=entries[-1]["ts"],
    )

    messages = [message for session in result["sessions"] for message in session["messages"]]
    assert len(messages) == 200
    assert messages[0]["content"] == "cron 5"
    assert messages[-1]["content"] == "cron 204"
    assert result["truncated"] is True


def test_background_range_keeps_preceding_tool_events_for_lazy_summary(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    (anima_dir.parent / "rin").mkdir()
    start = now_local().replace(microsecond=0) - timedelta(minutes=10)
    tool_ts = (start + timedelta(seconds=1)).isoformat()
    result_ts = (start + timedelta(seconds=2)).isoformat()
    response_ts = (start + timedelta(seconds=3)).isoformat()
    _write_entries(
        anima_dir,
        [
            _event(
                tool_ts,
                "tool_use",
                "",
                tool="Read",
                channel="chat",
                meta={"args": {"file_path": "notes.md"}, "tool_use_id": "tool-1"},
            ),
            _event(
                result_ts,
                "tool_result",
                "read result",
                channel="chat",
                meta={"tool_use_id": "tool-1"},
            ),
            _event(
                response_ts,
                "response_sent",
                "DM answer",
                to_person="rin",
                channel="chat",
            ),
        ],
    )

    result = activity.get_conversation_background(from_ts=response_ts, to_ts=response_ts)

    message = result["sessions"][0]["messages"][0]
    assert message["content"] == "DM answer"
    assert message["tool_calls"] == [
        {"tool_name": "Read", "tool_use_id": "tool-1", "is_error": False, "target": "notes.md"}
    ]


def test_feed_cache_skips_reads_and_invalidates_when_latest_log_changes(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    path = _write_entries(
        anima_dir,
        [
            _event(
                now.isoformat(),
                "message_received",
                "first",
                from_person="human",
                channel="chat",
                meta={"from_type": "human", "thread_id": "default"},
            )
        ],
    )
    with patch.object(activity, "_load_feed_entries", wraps=activity._load_feed_entries) as load:
        first = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
        second = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
        assert load.call_count == 1
        assert first == second

        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    _event(
                        (now + timedelta(seconds=1)).isoformat(),
                        "message_received",
                        "newest",
                        from_person="human",
                        channel="chat",
                        meta={"from_type": "human", "thread_id": "default"},
                    )
                )
                + "\n"
            )
        refreshed = activity.get_conversation_feed(min_user_items=1, _schedule_prefetch=False)
        assert load.call_count == 2
        assert refreshed["items"][-1]["message"]["content"] == "newest"


async def test_feed_and_background_routes_keep_history_api_shape_untouched(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    now = now_local().replace(microsecond=0)
    bg_ts = (now - timedelta(seconds=1)).isoformat()
    user_ts = now.isoformat()
    _write_entries(
        anima_dir,
        [
            _event(bg_ts, "cron_executed", "cron"),
            _event(
                user_ts,
                "message_received",
                "hello",
                from_person="human",
                channel="chat",
                meta={"from_type": "human", "thread_id": "default"},
            ),
        ],
    )
    app = FastAPI()
    app.state.animas_dir = tmp_path
    app.state.anima_names = ["alice"]
    app.include_router(create_sessions_router(), prefix="/api")
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        feed = await client.get("/api/animas/alice/conversation/feed?min_user_items=1")
        bg = await client.get(
            "/api/animas/alice/conversation/bg",
            params={"from_ts": bg_ts, "to_ts": bg_ts},
        )
        history = await client.get("/api/animas/alice/conversation/history?limit=10")

    assert feed.status_code == 200
    assert feed.json()["items"][-1]["message"]["content"] == "hello"
    assert bg.status_code == 200
    assert bg.json()["sessions"][0]["messages"][0]["content"] == "cron"
    assert set(history.json()) == {"sessions", "has_more", "next_before"}


# ── Named threads: conversation-file fallback ──────────


def _write_thread_file(anima_dir: Path, thread_id: str, turns: list[dict], **extra: object) -> None:
    conv_dir = anima_dir / "state" / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / f"{thread_id}.json").write_text(json.dumps({"turns": turns, **extra}), encoding="utf-8")


def _turns(start: datetime, count: int) -> list[dict]:
    return [
        {
            "role": "human" if index % 2 == 0 else "assistant",
            "content": f"turn {index}",
            "timestamp": (start + timedelta(minutes=index)).isoformat(),
        }
        for index in range(count)
    ]


def _activity_rows_for(turns: list[dict], thread_id: str, *, skew: float = 0.05) -> list[dict]:
    rows = []
    for turn in turns:
        ts = (datetime.fromisoformat(turn["timestamp"]) + timedelta(seconds=skew)).isoformat()
        if turn["role"] == "human":
            rows.append(
                _event(
                    ts,
                    "message_received",
                    turn["content"],
                    from_person="taka",
                    channel="chat",
                    meta={"from_type": "human", "thread_id": thread_id},
                )
            )
        else:
            rows.append(
                _event(
                    ts,
                    "response_sent",
                    turn["content"],
                    to_person="taka",
                    channel="chat",
                    meta={"thread_id": thread_id},
                )
            )
    return rows


async def _read_all_pages(tmp_path: Path, thread_id: str, *, min_user_items: int = 10) -> list[dict]:
    app = FastAPI()
    app.state.animas_dir = tmp_path
    app.include_router(create_sessions_router(), prefix="/api")
    pages: list[dict] = []
    before: str | None = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(40):
            params: dict[str, object] = {"thread_id": thread_id, "strict_thread": 1, "min_user_items": min_user_items}
            if before:
                params["before"] = before
            response = await client.get("/api/animas/alice/conversation/feed", params=params)
            assert response.status_code == 200
            page = response.json()
            pages.append(page)
            if not page["has_more"]:
                return pages
            assert page["next_before"] != before, "the cursor must advance"
            before = page["next_before"]
    raise AssertionError("paging did not terminate")


def _page_messages(pages: list[dict]) -> list[dict]:
    """Messages in display order: older pages are prepended to newer ones."""
    return [item["message"] for page in reversed(pages) for item in page["items"] if item["kind"] == "message"]


async def test_feed_shows_thread_file_turns_when_activity_log_has_no_rows(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    turns = _turns(now_local().replace(microsecond=0) - timedelta(days=200), 20)
    _write_thread_file(anima_dir, "e2628517", turns)

    pages = await _read_all_pages(tmp_path, "e2628517")

    assert len(pages) == 2
    assert [len(page["items"]) for page in pages] == [10, 10]
    assert [message["content"] for message in _page_messages(pages)] == [turn["content"] for turn in turns]


@pytest.mark.parametrize("min_user_items", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("skew", [0.05, -0.05, 30.0])
async def test_feed_shows_each_turn_once_when_activity_covers_part_of_the_thread(
    tmp_path: Path, min_user_items: int, skew: float
) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    turns = _turns(now_local().replace(microsecond=0) - timedelta(hours=2), 12)
    _write_thread_file(anima_dir, "research", turns)
    # Retention dropped the first half of the thread from the activity log.
    _write_entries(anima_dir, _activity_rows_for(turns[6:], "research", skew=skew))

    pages = await _read_all_pages(tmp_path, "research", min_user_items=min_user_items)

    assert [message["content"] for message in _page_messages(pages)] == [turn["content"] for turn in turns]


async def test_feed_prefers_the_activity_row_of_a_turn_recorded_twice(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    turns = _turns(now_local().replace(microsecond=0) - timedelta(hours=1), 2)
    _write_thread_file(anima_dir, "research", turns)
    rows = _activity_rows_for(turns, "research")
    rows[1]["meta"]["files"] = [{"name": "a.txt", "path": "attachments/a.txt", "size": 1, "media_type": "text/plain"}]
    _write_entries(anima_dir, rows)

    messages = _page_messages(await _read_all_pages(tmp_path, "research"))

    assert [message["ts"] for message in messages] == [row["ts"] for row in rows]
    assert messages[1]["files"][0]["name"] == "a.txt"


async def test_feed_matches_truncated_and_remark_trimmed_turns(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    start = now_local().replace(microsecond=0) - timedelta(hours=1)
    long_text = "x" * 6000
    turns = [
        {"role": "human", "content": long_text[:5000] + "\n[truncated]", "timestamp": start.isoformat()},
        {
            "role": "assistant",
            "content": "Checking...\nThe answer",
            "timestamp": (start + timedelta(minutes=1)).isoformat(),
        },
    ]
    _write_thread_file(anima_dir, "research", turns)
    rows = _activity_rows_for(turns, "research")
    rows[0]["content"] = long_text
    rows[1]["meta"]["interim_texts"] = ["Checking..."]
    _write_entries(anima_dir, rows)

    messages = _page_messages(await _read_all_pages(tmp_path, "research"))

    assert len(messages) == 2
    assert messages[1]["content"] == "The answer"


async def test_feed_keeps_file_turns_whose_row_was_lost_and_rows_with_no_file_turn(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    start = now_local().replace(microsecond=0) - timedelta(hours=1)
    turns = _turns(start, 4)
    _write_thread_file(anima_dir, "research", turns)
    rows = _activity_rows_for([turns[0], turns[1], turns[3]], "research")
    rows.append(
        _event(
            (start + timedelta(minutes=30)).isoformat(),
            "voice_call",
            "Call summary",
            summary="Call summary",
            channel="chat",
            meta={"thread_id": "research", "duration_seconds": 5},
        )
    )
    _write_entries(anima_dir, rows)

    messages = _page_messages(await _read_all_pages(tmp_path, "research"))

    assert [message["content"] for message in messages] == ["turn 0", "turn 1", "turn 2", "turn 3", "Call summary"]
    assert messages[-1]["type"] == "voice_call"


async def test_feed_file_turns_render_like_activity_messages(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    start = now_local().replace(microsecond=0) - timedelta(days=30)
    _write_thread_file(
        anima_dir,
        "research",
        [
            {"role": "human", "content": "hello", "timestamp": start.isoformat()},
            {
                "role": "assistant",
                "content": 'hi there\n<!-- emotion: {"emotion": "smile"} -->',
                "timestamp": (start + timedelta(seconds=5)).isoformat(),
                "tool_records": [
                    {"tool_name": "Read", "tool_id": "t1", "input_summary": "a.md", "result_summary": "ok"},
                ],
            },
            {"role": "system", "content": "internal marker", "timestamp": (start + timedelta(seconds=6)).isoformat()},
        ],
    )

    human, assistant = _page_messages(await _read_all_pages(tmp_path, "research"))

    assert human["from_person"] == "human", "an empty speaker renders as the Anima itself"
    assert assistant["content"] == "hi there"
    assert assistant["emotion"] == "smile"
    assert assistant["tool_calls_detail"] == "lazy"
    assert assistant["tool_calls"][0]["tool_name"] == "Read"


async def test_feed_shows_the_compressed_summary_once_at_the_oldest_end(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    turns = _turns(now_local().replace(microsecond=0) - timedelta(days=30), 6)
    _write_thread_file(anima_dir, "research", turns, compressed_summary="Earlier they agreed.", compressed_turn_count=8)

    pages = await _read_all_pages(tmp_path, "research", min_user_items=4)
    messages = _page_messages(pages)

    summaries = [message for message in messages if message.get("type") == "thread_summary"]
    assert len(pages) == 2
    assert len(summaries) == 1
    assert messages[0] is summaries[0]
    assert summaries[0]["content"] == "Earlier they agreed."
    assert summaries[0]["compressed_turn_count"] == 8
    assert summaries[0]["role"] == "system"


async def test_feed_steps_over_empty_activity_windows_to_reach_file_turns(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    now = now_local().replace(microsecond=0)
    _write_entries(anima_dir, [_event((now - timedelta(days=40)).isoformat(), "cron_executed", "old cron")])
    turns = _turns(now - timedelta(days=100), 2)
    _write_thread_file(anima_dir, "research", turns)

    pages = await _read_all_pages(tmp_path, "research")

    assert [item["message"]["content"] for item in pages[0]["items"]] == ["turn 0", "turn 1"]
    assert pages[0]["has_more"] is False


async def test_feed_reports_more_when_empty_windows_outlast_the_hop_limit(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    now = now_local().replace(microsecond=0)
    _write_entries(anima_dir, [_event((now - timedelta(days=200)).isoformat(), "cron_executed", "old cron")])
    turns = _turns(now - timedelta(days=300), 2)
    _write_thread_file(anima_dir, "research", turns)

    pages = await _read_all_pages(tmp_path, "research")

    assert pages[0]["items"] == []
    assert pages[0]["has_more"] is True
    assert [message["content"] for message in _page_messages(pages)] == ["turn 0", "turn 1"]


async def test_feed_default_thread_does_not_read_conversation_files(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    _write_thread_file(anima_dir, "default", _turns(now_local().replace(microsecond=0) - timedelta(days=1), 2))
    app = FastAPI()
    app.state.animas_dir = tmp_path
    app.include_router(create_sessions_router(), prefix="/api")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/animas/alice/conversation/feed")

    assert response.json()["items"] == []


async def test_feed_thread_without_a_file_is_the_plain_activity_feed(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    (anima_dir / "activity_log").mkdir(parents=True)
    turns = _turns(now_local().replace(microsecond=0) - timedelta(hours=1), 2)
    _write_entries(anima_dir, _activity_rows_for(turns, "research"))

    messages = _page_messages(await _read_all_pages(tmp_path, "research"))

    assert [message["content"] for message in messages] == ["turn 0", "turn 1"]


def test_named_thread_scan_is_not_cut_off_by_unrelated_log_lines(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    thread_rows = _activity_rows_for(_turns(now - timedelta(hours=3), 2), "research")
    noise = [_event((now - timedelta(minutes=30 - index)).isoformat(), "cron_executed", "noise") for index in range(30)]
    _write_entries(anima_dir, thread_rows + noise)

    with patch("core.activity.conversation._FEED_MAX_RAW_ENTRIES", 5):
        feed = activity.get_conversation_feed(thread_id="research", strict_thread=True, _schedule_prefetch=False)

    assert [item["message"]["content"] for item in feed["items"]] == ["turn 0", "turn 1"]
    assert feed["has_more"] is False


def test_named_thread_scan_only_matches_its_own_rows(tmp_path: Path) -> None:
    anima_dir, activity = _test_activity(tmp_path)
    now = now_local().replace(microsecond=0)
    turns = _turns(now - timedelta(hours=1), 2)
    rows = _activity_rows_for(turns[:1], "research") + _activity_rows_for(turns[1:], "research-2")
    # The id appearing only in a message body must not make a row part of the thread.
    rows.append(
        _event(
            (now - timedelta(minutes=1)).isoformat(),
            "message_received",
            '"research"',
            from_person="taka",
            channel="chat",
            meta={"from_type": "human", "thread_id": "other"},
        )
    )
    _write_entries(anima_dir, rows)

    feed = activity.get_conversation_feed(thread_id="research", strict_thread=True, _schedule_prefetch=False)

    assert [item["message"]["content"] for item in feed["items"]] == ["turn 0"]
