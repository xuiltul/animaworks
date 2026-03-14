"""Unit tests for Mode S graceful interrupt session preservation.

Tests that client.interrupt() is called on session interruption and that
session_id is saved either from ResultMessage (primary) or from a previously
captured StreamEvent session_id (fallback).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Lightweight mock SDK classes ──────────────────────────────


class _ResultMessage:
    def __init__(self, *, session_id: str = "test-session", **kwargs: Any) -> None:
        self.session_id = session_id
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    return tmp_path


@pytest.fixture()
def _mock_sdk():
    """Ensure claude_agent_sdk resolves our _ResultMessage for isinstance checks."""
    mock_module = MagicMock()
    mock_module.ResultMessage = _ResultMessage
    mock_types = MagicMock()
    mock_types.StreamEvent = MagicMock
    mock_module.types = mock_types
    saved = {
        "claude_agent_sdk": sys.modules.get("claude_agent_sdk"),
        "claude_agent_sdk.types": sys.modules.get("claude_agent_sdk.types"),
    }
    sys.modules["claude_agent_sdk"] = mock_module
    sys.modules["claude_agent_sdk.types"] = mock_types
    yield mock_module
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


def _make_mock_client(
    *,
    interrupt_side_effect: BaseException | None = None,
    messages_after_interrupt: list[Any] | None = None,
    response_after_interrupt: list[Any] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    if interrupt_side_effect:
        client.interrupt.side_effect = interrupt_side_effect
    else:
        client.interrupt.return_value = None

    async def _receive_messages():
        for msg in messages_after_interrupt or []:
            yield msg

    async def _receive_response():
        for msg in response_after_interrupt or []:
            yield msg

    client.receive_messages = _receive_messages
    client.receive_response = _receive_response
    return client


# ── Test: Streaming interrupt — primary path (ResultMessage) ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_stream_interrupt_saves_session_id_from_result_message(anima_dir: Path) -> None:
    """interrupt() succeeds → ResultMessage session_id is saved."""
    from core.execution.agent_sdk import _graceful_interrupt_stream

    result = _ResultMessage(session_id="result-session-123")
    client = _make_mock_client(messages_after_interrupt=[result])

    await _graceful_interrupt_stream(
        client,
        anima_dir,
        "chat",
        captured_session_id="fallback-id",
        thread_id="default",
    )

    client.interrupt.assert_awaited_once()
    session_file = anima_dir / "state" / "current_session_chat.json"
    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert data["session_id"] == "result-session-123"


# ── Test: Streaming interrupt — fallback (StreamEvent session_id) ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_stream_interrupt_fallback_to_stream_event_session_id(anima_dir: Path) -> None:
    """interrupt() times out → StreamEvent captured session_id is saved."""
    from core.execution.agent_sdk import _graceful_interrupt_stream

    client = _make_mock_client(interrupt_side_effect=TimeoutError())

    await _graceful_interrupt_stream(
        client,
        anima_dir,
        "chat",
        captured_session_id="stream-fallback-456",
        thread_id="default",
    )

    session_file = anima_dir / "state" / "current_session_chat.json"
    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert data["session_id"] == "stream-fallback-456"


# ── Test: Streaming interrupt — no StreamEvent, no save ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_stream_interrupt_no_stream_event_no_save(anima_dir: Path) -> None:
    """interrupt() fails and no StreamEvent captured → no session_id saved."""
    from core.execution.agent_sdk import _graceful_interrupt_stream

    client = _make_mock_client(interrupt_side_effect=TimeoutError())

    await _graceful_interrupt_stream(
        client,
        anima_dir,
        "chat",
        captured_session_id=None,
        thread_id="default",
    )

    session_file = anima_dir / "state" / "current_session_chat.json"
    assert not session_file.exists()


# ── Test: Blocking interrupt — primary path ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_blocking_interrupt_saves_session_id(anima_dir: Path) -> None:
    """Blocking interrupt() succeeds → ResultMessage session_id is saved."""
    from core.execution.agent_sdk import _graceful_interrupt_blocking

    result = _ResultMessage(session_id="blocking-session-789")
    client = _make_mock_client(response_after_interrupt=[result])

    await _graceful_interrupt_blocking(
        client,
        anima_dir,
        "chat",
        thread_id="default",
    )

    client.interrupt.assert_awaited_once()
    session_file = anima_dir / "state" / "current_session_chat.json"
    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert data["session_id"] == "blocking-session-789"


# ── Test: Blocking interrupt — timeout, no save ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_blocking_interrupt_timeout_no_save(anima_dir: Path) -> None:
    """Blocking interrupt() times out → no session_id saved."""
    from core.execution.agent_sdk import _graceful_interrupt_blocking

    client = _make_mock_client(interrupt_side_effect=TimeoutError())

    await _graceful_interrupt_blocking(
        client,
        anima_dir,
        "chat",
        thread_id="default",
    )

    session_file = anima_dir / "state" / "current_session_chat.json"
    assert not session_file.exists()


# ── Test: Non-resumable session types are not saved ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_stream_interrupt_non_resumable_session_not_saved(anima_dir: Path) -> None:
    """Heartbeat/cron/task sessions should not be saved even on interrupt."""
    from core.execution.agent_sdk import _graceful_interrupt_stream

    result = _ResultMessage(session_id="hb-session-nope")
    client = _make_mock_client(messages_after_interrupt=[result])

    await _graceful_interrupt_stream(
        client,
        anima_dir,
        "heartbeat",
        captured_session_id="fallback",
        thread_id="default",
    )

    session_file = anima_dir / "state" / "current_session_heartbeat.json"
    assert not session_file.exists()


# ── Test: Thread-specific session file ──


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_mock_sdk")
async def test_stream_interrupt_thread_specific_session(anima_dir: Path) -> None:
    """Session file is thread-specific when thread_id != 'default'."""
    from core.execution.agent_sdk import _graceful_interrupt_stream

    result = _ResultMessage(session_id="thread-session-42")
    client = _make_mock_client(messages_after_interrupt=[result])

    await _graceful_interrupt_stream(
        client,
        anima_dir,
        "chat",
        captured_session_id=None,
        thread_id="custom-thread",
    )

    session_file = anima_dir / "state" / "current_session_chat_custom-thread.json"
    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert data["session_id"] == "thread-session-42"


# ── Test: Constant value ──


def test_interrupt_timeout_constant() -> None:
    """INTERRUPT_TIMEOUT_SEC is defined and reasonable."""
    from core.execution._sdk_session import INTERRUPT_TIMEOUT_SEC

    assert INTERRUPT_TIMEOUT_SEC == 5.0
