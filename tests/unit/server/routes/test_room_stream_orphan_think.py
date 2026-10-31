"""Regression: orphan ` response` (no ` thinking`) must preserve the answer body.

Matches the fix in server/routes/room.py where the final-confirmed path now uses
resolve_streamed_leaked_thinking instead of strip_thinking_tags, so a stray close
tag emitted after real content no longer deletes the already-streamed answer
(incident f93221ae).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import server.routes.room as room_mod
from server.room_manager import MeetingRoom, RoomManager
from server.routes.room import _meeting_stream

CLOSE_TAG = chr(60) + "/" + "think" + chr(62)  # </think>



class _StubRoomManager(RoomManager):
    def __init__(self) -> None:
        self.room = MeetingRoom(
            room_id="a1b2c3d4e5f6",
            participants=["sakura"],
            chair="sakura",
            created_by="human",
            created_at=datetime.now(),
            conversation=[],
        )
        self.appended: list[tuple[str, str, str, str]] = []

    def get_room(self, room_id: str) -> MeetingRoom | None:
        return self.room

    def extract_mentions(self, text: str, participants: list[str]) -> list[str]:
        return []

    def append_message(self, room_id: str, speaker: str, role: str, text: str, *, meta=None) -> None:
        self.appended.append((room_id, speaker, role, text))

    async def get_summarized_context(self, room_id: str) -> str:
        return ""

    def build_chair_prompt(self, room: MeetingRoom) -> str:
        return "chair prompt"


def _stub_config():
    return SimpleNamespace(server=SimpleNamespace(ipc_stream_timeout=10.0))


CLOSE_TAG_G = CLOSE_TAG


def _make_supervisor(full_response: str):
    """Supervisor whose stream yields a single done response with the given body."""
    supervisor = SimpleNamespace()
    supervisor.processes = {"sakura": object()}

    async def send_request_stream(**kwargs):
        yield SimpleNamespace(done=True, chunk=None, result={"response": full_response})

    supervisor.send_request_stream = send_request_stream
    return supervisor


@pytest.mark.asyncio
async def test_meeting_stream_preserves_body_on_orphan_close_tag():
    """Orphan ` response` with no ` thinking` must keep the answer (not delete it)."""
    rm = _StubRoomManager()
    supervisor = _make_supervisor(f"reasoning content {CLOSE_TAG}\n\nactual response")

    with patch.object(room_mod, "load_config", side_effect=_stub_config):
        events = [ev async for ev in _meeting_stream(rm.room.room_id, "hi", "human", rm, supervisor)]

    # The conversation is appended with the preserved body, orphan tag removed.
    answers = [t for (_id, _sp, _role, t) in rm.appended if "actual response" in t]
    assert answers, "expected an appended answer message"
    assert CLOSE_TAG not in answers[0]
    assert "reasoning content" in answers[0]
    assert "actual response" in answers[0]

    # And a done event is eventually emitted (stream completed normally).
    assert any("Meeting round complete" in ev for ev in events)
