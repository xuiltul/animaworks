# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.text import Text

from cli.tui.state import (
    AppState,
    PaletteItem,
    apply_ws_event,
    filter_palette,
    is_valid_thread_id,
    new_thread_id,
)


def _event(event_type, data):
    return {"type": event_type, "data": data}


def _state_with(animas=("sora", "rin", "mei"), current="sora", busy=None):
    state = AppState()
    state.set_animas([{"name": n, "status": "running", "busy": (busy or {}).get(n)} for n in animas])
    state.current = current
    return state


def test_status_changes_badge():
    state = _state_with()
    apply_ws_event(state, _event("anima.status", {"name": "rin", "status": "thinking"}))
    assert state.animas["rin"].status == "thinking"
    assert state.animas["sora"].status == "idle"


def test_tool_activity_updates_tool_and_feed():
    state = _state_with()
    eff = apply_ws_event(
        state, _event("anima.tool_activity", {"name": "rin", "event": "tool_start", "tool_name": "Bash"})
    )
    assert state.animas["rin"].active_tool == "Bash"
    assert eff.feed and eff.feed[0].text == "Bash"


def test_tool_detail_does_not_emit_feed():
    state = _state_with()
    eff = apply_ws_event(
        state,
        _event("anima.tool_activity", {"name": "rin", "event": "tool_detail", "tool_name": "Bash", "detail": "x"}),
    )
    assert state.animas["rin"].active_tool == "Bash"
    assert eff.feed == []


def test_tool_end_clears_tool():
    state = _state_with()
    apply_ws_event(state, _event("anima.tool_activity", {"name": "rin", "event": "tool_start", "tool_name": "Bash"}))
    apply_ws_event(state, _event("anima.tool_activity", {"name": "rin", "event": "tool_end", "tool_name": "Bash"}))
    assert state.animas["rin"].active_tool is None


# ── activity-log shapes (no `event` key) ─────────────────


def test_tool_activity_log_without_event_kind_tool_use():
    state = _state_with()
    eff = apply_ws_event(
        state,
        _event(
            "anima.tool_activity",
            {"name": "rin", "type": "task_processing_start", "kind": "tool_use", "tool": "Bash", "summary": "x"},
        ),
    )
    assert state.animas["rin"].active_tool == "Bash"
    assert eff.feed and eff.feed[0].text == "Bash"


def test_tool_activity_log_without_event_kind_tool_result():
    state = _state_with()
    apply_ws_event(
        state,
        _event("anima.tool_activity", {"name": "rin", "type": "t", "kind": "tool_use", "tool": "Bash"}),
    )
    state.animas["rin"].active_tool = "Bash"
    eff = apply_ws_event(
        state,
        _event("anima.tool_activity", {"name": "rin", "type": "t", "kind": "tool_result", "tool": "Bash"}),
    )
    assert state.animas["rin"].active_tool is None
    assert eff.feed == []


def test_tool_activity_log_without_event_kind_tool_result_error():
    state = _state_with()
    eff = apply_ws_event(
        state,
        _event(
            "anima.tool_activity",
            {"name": "rin", "type": "t", "kind": "tool_result", "tool": "Bash", "is_error": True},
        ),
    )
    assert state.animas["rin"].active_tool is None
    assert eff.feed and "✗" in eff.feed[0].text


def test_tool_activity_log_without_event_other_type_uses_summary():
    state = _state_with()
    long = "x" * 200
    eff = apply_ws_event(
        state,
        _event(
            "anima.tool_activity",
            {"name": "rin", "type": "inbox_processing_end", "kind": "", "summary": long},
        ),
    )
    assert eff.feed and eff.feed[0].text.startswith("inbox_processing_end ")
    assert len(eff.feed[0].text) == len("inbox_processing_end ") + 60


def test_activity_log_sets_busy_on_start_idle_on_end():
    state = _state_with()
    apply_ws_event(
        state,
        _event("anima.tool_activity", {"name": "rin", "type": "inbox_processing_start", "kind": ""}),
    )
    assert state.animas["rin"].status == "busy"
    assert state.animas["rin"].busy is True
    apply_ws_event(
        state,
        _event("anima.tool_activity", {"name": "rin", "type": "inbox_processing_end", "kind": ""}),
    )
    assert state.animas["rin"].status == "idle"
    assert state.animas["rin"].busy is False


def test_unknown_anima_ignored():
    state = _state_with()
    eff = apply_ws_event(state, _event("anima.status", {"name": "nobody", "status": "thinking"}))
    assert eff.feed == []
    assert "nobody" not in state.animas


def test_proactive_unread_increments_for_other_anima():
    state = _state_with(current="sora")
    apply_ws_event(
        state,
        _event("anima.proactive_message", {"anima": "rin", "subject": "s", "body": "b"}),
    )
    assert state.animas["rin"].unread == 1
    assert state.animas["sora"].unread == 0


def test_proactive_no_unread_for_current_anima():
    state = _state_with(current="sora")
    apply_ws_event(
        state,
        _event("anima.proactive_message", {"anima": "sora", "subject": "s", "body": "b"}),
    )
    assert state.animas["sora"].unread == 0


def test_notification_feed_and_toast():
    state = _state_with()
    eff = apply_ws_event(state, _event("anima.notification", {"anima": "rin", "subject": "subj", "body": "body"}))
    assert any(e.kind == "notification" for e in eff.feed)
    assert eff.toasts == [("subj", "body")]


def test_notification_with_callback_produces_card():
    state = _state_with()
    eff = apply_ws_event(
        state,
        _event(
            "anima.notification",
            {"anima": "rin", "subject": "subj", "body": "body", "callback_id": "cb1", "options": ["yes", "no"]},
        ),
    )
    assert eff.cards and eff.cards[0]["callback_id"] == "cb1"


def test_heartbeat_feed_summary_trimmed():
    state = _state_with()
    long = "x" * 200
    eff = apply_ws_event(state, _event("anima.heartbeat", {"name": "rin", "result": {"summary": long}}))
    assert eff.feed and len(eff.feed[0].text) == 80


def test_cron_and_bootstrap_feeds():
    state = _state_with()
    eff = apply_ws_event(state, _event("anima.cron", {"name": "rin", "task": "t"}))
    assert any(e.kind == "cron" for e in eff.feed)
    eff = apply_ws_event(state, _event("anima.bootstrap", {"name": "rin", "status": "started"}))
    assert any(e.kind == "bootstrap" for e in eff.feed)


def test_interaction_and_board_feeds():
    state = _state_with()
    eff = apply_ws_event(
        state, _event("anima.interaction", {"from_person": "sora", "to_person": "rin", "summary": "hi"})
    )
    assert eff.feed and "sora → rin" in eff.feed[0].text
    eff = apply_ws_event(state, _event("board.post", {"channel": "dev", "from": "mei", "text": "PR ready"}))
    assert eff.feed and "#dev" in eff.feed[0].text


def test_activity_is_ring_buffered():
    state = _state_with()
    for i in range(300):
        apply_ws_event(state, _event("anima.heartbeat", {"name": "rin", "result": {"summary": str(i)}}))
    assert len(state.activity) <= 200
    assert state.activity[0].text != "0"


# ── Palette filtering ───────────────────────────────────


def _item(value, search):
    return PaletteItem(value=value, label=Text(value), takes_args=True, search=search)


def test_filter_empty_query_returns_all():
    items = [_item("/help", "/help"), _item("/skill foo", "/skill foo /foo foo")]
    assert len(filter_palette(items, "/")) == 2


def test_filter_prefix_before_partial():
    items = [
        _item("/skill foobar", "/skill foobar /foobar foobar"),  # partial for "/bar"
        _item("/bar", "/bar"),  # prefix for "/bar"
    ]
    out = filter_palette(items, "/bar")
    assert [i.value for i in out] == ["/bar", "/skill foobar"]


def test_filter_matches_plain_skill_name():
    items = [_item("/skill pr-review", "/skill pr-review /pr-review pr-review")]
    out = filter_palette(items, "/pr-review")
    assert [i.value for i in out] == ["/skill pr-review"]


def test_filter_case_insensitive():
    items = [_item("/skill PRReview", "/skill prreview /prreview prreview")]
    out = filter_palette(items, "/prr")
    assert [i.value for i in out] == ["/skill PRReview"]


# ── Thread id rules ─────────────────────────────────────


def test_is_valid_thread_id():
    for ok in ("default", "1a2b3c4d", "a" * 36):
        assert is_valid_thread_id(ok), f"expected {ok!r} valid"
    for bad in ("", "a" * 37, "bad id", "x/y"):
        assert not is_valid_thread_id(bad), f"expected {bad!r} invalid"


def test_new_thread_id_is_8_hex():
    tid = new_thread_id()
    assert len(tid) == 8
    assert all(c in "0123456789abcdef" for c in tid)
