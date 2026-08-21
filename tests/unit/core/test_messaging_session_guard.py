from __future__ import annotations

import pytest


def test_chat_cycle_guard_accepts_matching_chat_result():
    from core._anima_messaging import _chat_cycle_isolated

    ok, meta = _chat_cycle_isolated(
        {
            "trigger": "message:mio",
            "session_type": "chat",
            "thread_id": "thread-a",
            "request_id": "req-1",
        },
        expected_trigger="message:mio",
        thread_id="thread-a",
    )

    assert ok is True
    assert meta["request_id"] == "req-1"


@pytest.mark.parametrize("from_person", ["fugaku", "shokotan", "operator"])
def test_chat_cycle_guard_accepts_line_trigger_for_multiple_bots(from_person: str):
    """LINE-backed bots keep the source-qualified trigger through the guard."""
    from core._anima_messaging import _chat_cycle_isolated

    trigger = f"message:line:{from_person}"
    ok, meta = _chat_cycle_isolated(
        {
            "trigger": trigger,
            "session_type": "chat",
            "thread_id": f"line-{from_person}",
        },
        expected_trigger=trigger,
        thread_id=f"line-{from_person}",
    )

    assert ok is True
    assert meta["actual_trigger"] == trigger


def test_chat_cycle_guard_rejects_heartbeat_result():
    from core._anima_messaging import _chat_cycle_isolated

    ok, meta = _chat_cycle_isolated(
        {
            "trigger": "heartbeat",
            "session_type": "heartbeat",
            "thread_id": "default",
            "request_id": "req-hb",
        },
        expected_trigger="message:mio",
        thread_id="thread-a",
    )

    assert ok is False
    assert meta["actual_session_type"] == "heartbeat"
    assert meta["actual_trigger"] == "heartbeat"


def test_chat_cycle_guard_rejects_other_chat_thread():
    from core._anima_messaging import _chat_cycle_isolated

    ok, meta = _chat_cycle_isolated(
        {
            "trigger": "message:mio",
            "session_type": "chat",
            "thread_id": "thread-b",
        },
        expected_trigger="message:mio",
        thread_id="thread-a",
    )

    assert ok is False
    assert meta["actual_thread_id"] == "thread-b"
