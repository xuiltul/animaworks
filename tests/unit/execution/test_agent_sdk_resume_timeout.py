"""Tests for agent_sdk.py resume timeout guard and session type constants.

Covers:
  - RESUME_TIMEOUT_SEC constant is defined
  - asyncio.wait_for is used when resuming (session_id_to_resume is set)
  - TimeoutError causes _clear_session_id to be called and fallback to fresh session
  - _RESUMABLE_SESSION_TYPES only contains chat
  - _resolve_session_type() maps triggers correctly
  - _clear_session_id() file deletion logic
  - _load_session_id() / _save_session_id() persistence (immortal sessions — no TTL)
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import ModelConfig

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-6",
        api_key="sk-test",
        context_threshold=0.50,
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test"
    d.mkdir(parents=True)
    (d / "state").mkdir(parents=True)
    return d


# ── Session resume timeout ────────────────────────────────────


class TestSessionImmortal:
    """_load_session_id() returns session ID regardless of age (no TTL)."""

    def test_recent_session_returns_id(self, anima_dir: Path) -> None:
        from core.execution._sdk_session import _load_session_id, _save_session_id

        _save_session_id(anima_dir, "sess-recent", "chat")
        assert _load_session_id(anima_dir, "chat") == "sess-recent"

    def test_old_session_still_returns_id(self, anima_dir: Path) -> None:
        """Sessions are immortal — old sessions are always returned."""
        from core.execution._sdk_session import _load_session_id

        path = anima_dir / "state" / "current_session_chat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        path.write_text(
            json.dumps({"session_id": "sess-old", "timestamp": old_ts}),
            encoding="utf-8",
        )
        assert _load_session_id(anima_dir, "chat") == "sess-old"

    def test_very_old_session_still_returns_id(self, anima_dir: Path) -> None:
        """Even week-old sessions are returned (TTL was removed)."""
        from core.execution._sdk_session import _load_session_id

        path = anima_dir / "state" / "current_session_chat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        path.write_text(
            json.dumps({"session_id": "sess-week-old", "timestamp": old_ts}),
            encoding="utf-8",
        )
        assert _load_session_id(anima_dir, "chat") == "sess-week-old"

    def test_naive_timestamp_handled_gracefully(self, anima_dir: Path) -> None:
        """Legacy files without timezone info should still return session ID."""
        from core.execution._sdk_session import _load_session_id

        path = anima_dir / "state" / "current_session_chat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timedelta

        old_ts = datetime.now(UTC) - timedelta(hours=2)
        path.write_text(
            json.dumps(
                {
                    "session_id": "sess-naive",
                    "timestamp": old_ts.replace(tzinfo=None).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        assert _load_session_id(anima_dir, "chat") == "sess-naive"

    def test_missing_timestamp_still_returns_id(self, anima_dir: Path) -> None:
        """Files without timestamp field (edge case) resume unconditionally."""
        from core.execution._sdk_session import _load_session_id

        path = anima_dir / "state" / "current_session_chat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"session_id": "sess-no-ts"}),
            encoding="utf-8",
        )
        assert _load_session_id(anima_dir, "chat") == "sess-no-ts"

    def test_heartbeat_old_session_returns_id(self, anima_dir: Path) -> None:
        """Heartbeat sessions are also immortal."""
        from core.execution._sdk_session import _load_session_id

        path = anima_dir / "state" / "current_session_heartbeat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
        path.write_text(
            json.dumps({"session_id": "hb-old", "timestamp": old_ts}),
            encoding="utf-8",
        )
        assert _load_session_id(anima_dir, "heartbeat") == "hb-old"


# ── Session persistence helpers ───────────────────────────────


class TestClearSessionId:
    """_clear_session_id() removes the session file."""

    def test_removes_existing_file(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id, _save_session_id

        _save_session_id(anima_dir, "sess-001", "chat")
        path = anima_dir / "state" / "current_session_chat.json"
        assert path.exists()

        _clear_session_id(anima_dir, "chat")
        assert not path.exists()

    def test_noop_when_no_file(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id

        # Should not raise
        _clear_session_id(anima_dir, "chat")
        _clear_session_id(anima_dir, "heartbeat")

    def test_clears_heartbeat_session(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _clear_session_id, _save_session_id

        _save_session_id(anima_dir, "sess-hb", "heartbeat")
        path = anima_dir / "state" / "current_session_heartbeat.json"
        assert path.exists()

        _clear_session_id(anima_dir, "heartbeat")
        assert not path.exists()


class TestSessionTypeConstants:
    """_RESUMABLE_SESSION_TYPES and _resolve_session_type() correctness."""

    def test_only_chat_is_resumable(self) -> None:
        from core.execution._sdk_session import (
            _RESUMABLE_SESSION_TYPES,
            SESSION_TYPE_CHAT,
            SESSION_TYPE_CRON,
            SESSION_TYPE_HEARTBEAT,
            SESSION_TYPE_INBOX,
            SESSION_TYPE_TASK,
        )

        assert SESSION_TYPE_CHAT in _RESUMABLE_SESSION_TYPES
        assert SESSION_TYPE_HEARTBEAT not in _RESUMABLE_SESSION_TYPES
        assert SESSION_TYPE_CRON not in _RESUMABLE_SESSION_TYPES
        assert SESSION_TYPE_TASK not in _RESUMABLE_SESSION_TYPES
        assert SESSION_TYPE_INBOX not in _RESUMABLE_SESSION_TYPES

    def test_resolve_session_type(self) -> None:
        from core.execution._sdk_session import (
            SESSION_TYPE_CHAT,
            SESSION_TYPE_CRON,
            SESSION_TYPE_HEARTBEAT,
            SESSION_TYPE_INBOX,
            SESSION_TYPE_TASK,
            _resolve_session_type,
        )

        assert _resolve_session_type("heartbeat") == SESSION_TYPE_HEARTBEAT
        assert _resolve_session_type("cron:daily") == SESSION_TYPE_CRON
        assert _resolve_session_type("task:abc123") == SESSION_TYPE_TASK
        assert _resolve_session_type("inbox:alice") == SESSION_TYPE_INBOX
        assert _resolve_session_type("chat") == SESSION_TYPE_CHAT
        assert _resolve_session_type("") == SESSION_TYPE_CHAT
        assert _resolve_session_type("unknown") == SESSION_TYPE_TASK

    def test_clear_session_id_for_chat(self, anima_dir: Path) -> None:
        from core.execution._sdk_session import (
            _clear_session_id,
            _save_session_id,
        )

        _save_session_id(anima_dir, "sess-chat", "chat")
        chat_path = anima_dir / "state" / "current_session_chat.json"
        assert chat_path.exists()

        _clear_session_id(anima_dir, "chat")
        assert not chat_path.exists()

    def test_clear_session_id_for_type_clears_non_chat_thread(self, anima_dir: Path) -> None:
        from core.execution._sdk_session import (
            _load_session_id,
            _save_session_id,
            clear_session_id_for_type,
        )

        _save_session_id(anima_dir, "sess-inbox", "inbox", thread_id="inbox")
        assert _load_session_id(anima_dir, "inbox", thread_id="inbox") == "sess-inbox"

        clear_session_id_for_type(anima_dir, "inbox", thread_id="inbox")
        assert _load_session_id(anima_dir, "inbox", thread_id="inbox") is None


class TestResumeTimeoutConstant:
    """RESUME_TIMEOUT_SEC is defined with a reasonable value."""

    def test_resume_timeout_defined(self) -> None:
        """Fix 4a: default raised from 15s to 60s (15s was routinely exceeded
        on a loaded host, discarding valid sessions)."""
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        assert RESUME_TIMEOUT_SEC == 60.0

    def test_resume_timeout_is_positive(self) -> None:
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        assert RESUME_TIMEOUT_SEC > 0

    def test_resume_timeout_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fix 4a: the timeout is configurable via env var."""
        from core.execution._sdk_session import _resume_timeout_from_env

        monkeypatch.setenv("ANIMAWORKS_SDK_RESUME_TIMEOUT_SEC", "90")
        assert _resume_timeout_from_env() == 90.0

    def test_resume_timeout_env_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.execution._sdk_session import _resume_timeout_from_env

        monkeypatch.setenv("ANIMAWORKS_SDK_RESUME_TIMEOUT_SEC", "not-a-number")
        assert _resume_timeout_from_env() == 60.0
        monkeypatch.setenv("ANIMAWORKS_SDK_RESUME_TIMEOUT_SEC", "-5")
        assert _resume_timeout_from_env() == 60.0

    def test_codex_resume_timeout_matches(self) -> None:
        """Fix 4a: codex_sdk shares the same default and env knob."""
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC as CLAUDE_TIMEOUT
        from core.execution.codex_sdk import RESUME_TIMEOUT_SEC as CODEX_TIMEOUT

        assert CODEX_TIMEOUT == CLAUDE_TIMEOUT == 60.0

    def test_resume_max_attempts_defined(self) -> None:
        """Fix 4c: one retry before the session id is discarded."""
        from core.execution._sdk_session import RESUME_MAX_ATTEMPTS

        assert RESUME_MAX_ATTEMPTS == 2


class TestNonChatSessionCleanup:
    @pytest.mark.asyncio
    async def test_blocking_inbox_clears_stale_session_and_does_not_save(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        from core.execution._sdk_session import _load_session_id, _save_session_id
        from core.execution.agent_sdk import AgentSDKExecutor
        from core.prompt.context import ContextTracker
        from tests.helpers.mocks import patch_agent_sdk

        _save_session_id(anima_dir, "stale-inbox-session", "inbox", thread_id="inbox")
        _save_session_id(anima_dir, "chat-session", "chat")

        with patch_agent_sdk(response_text="inbox done"):
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")
            result = await executor.execute(
                prompt="check inbox",
                system_prompt="sys",
                tracker=tracker,
                trigger="inbox:sakura",
                thread_id="inbox",
            )

        assert result.text == "inbox done"
        assert _load_session_id(anima_dir, "inbox", thread_id="inbox") is None
        assert _load_session_id(anima_dir, "chat") == "chat-session"

    @pytest.mark.asyncio
    async def test_streaming_inbox_clears_stale_session_and_does_not_save(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        from core.execution._sdk_session import _load_session_id, _save_session_id
        from core.execution.agent_sdk import AgentSDKExecutor
        from core.prompt.context import ContextTracker
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        _save_session_id(anima_dir, "stale-inbox-session", "inbox", thread_id="inbox")
        _save_session_id(anima_dir, "chat-session", "chat")

        messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "inbox stream"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("inbox stream")]),
            MockResultMessage(session_id="new-inbox-session"),
        ]

        with _patch_sdk_for_streaming(messages):
            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")
            events = []
            async for event in executor.execute_streaming(
                system_prompt="sys",
                prompt="check inbox",
                tracker=tracker,
                trigger="inbox:sakura",
                thread_id="inbox",
            ):
                events.append(event)

        assert any(e["type"] == "done" for e in events)
        assert _load_session_id(anima_dir, "inbox", thread_id="inbox") is None
        assert _load_session_id(anima_dir, "chat") == "chat-session"


# ── Resume timeout guard in execute_streaming ─────────────────


@contextmanager
def _patch_sdk_for_streaming(messages: list[Any]):
    """Patch claude_agent_sdk for streaming tests with custom message sequence."""
    from tests.helpers.mocks import (
        MockAssistantMessage,
        MockClaudeSDKClient,
        MockResultMessage,
        MockStreamEvent,
        MockSystemMessage,
        MockTextBlock,
        MockToolResultBlock,
        MockUserMessage,
    )

    def _client_factory(**kwargs: Any) -> MockClaudeSDKClient:
        return MockClaudeSDKClient(messages=messages, **kwargs)

    mock_module = MagicMock()
    mock_module.ClaudeSDKClient = _client_factory
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MagicMock
    mock_module.ToolResultBlock = MockToolResultBlock
    mock_module.UserMessage = MockUserMessage
    mock_module.SystemMessage = MockSystemMessage
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock
    mock_module.ClaudeSDKError = Exception
    mock_module.ProcessError = Exception

    mock_types = MagicMock()
    mock_types.StreamEvent = MockStreamEvent
    mock_types.HookContext = MagicMock
    mock_types.HookInput = MagicMock
    mock_types.PreToolUseHookSpecificOutput = MagicMock
    mock_types.SyncHookJSONOutput = MagicMock
    mock_module.types = mock_types

    saved: dict[str, Any] = {}
    for key in ["claude_agent_sdk", "claude_agent_sdk.types"]:
        saved[key] = sys.modules.get(key)
        sys.modules[key] = mock_types if key == "claude_agent_sdk.types" else mock_module

    try:
        yield mock_module
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


class TestResumeTimeoutGuard:
    """asyncio.wait_for is applied to first-event receive during session resume."""

    @pytest.mark.asyncio
    async def test_wait_for_called_with_resume_timeout_when_resuming(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """When a session_id is present, asyncio.wait_for wraps first-event receive."""
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        # Persist a session ID so execute_streaming takes the resume path
        _save_session_id(anima_dir, "stale-session-001", "chat")

        messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hello"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("hello")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        wait_for_calls: list[dict] = []
        original_wait_for = asyncio.wait_for

        async def _spy_wait_for(coro, timeout=None, **kwargs):
            wait_for_calls.append({"timeout": timeout})
            return await original_wait_for(coro, timeout=timeout, **kwargs)

        with _patch_sdk_for_streaming(messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with patch("asyncio.wait_for", side_effect=_spy_wait_for):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        # asyncio.wait_for should have been called with RESUME_TIMEOUT_SEC
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        timeout_values = [c["timeout"] for c in wait_for_calls]
        assert RESUME_TIMEOUT_SEC in timeout_values, (
            f"Expected wait_for to be called with {RESUME_TIMEOUT_SEC}s timeout, but got timeouts: {timeout_values}"
        )

    @pytest.mark.asyncio
    async def test_no_wait_for_when_no_session_to_resume(self, model_config: ModelConfig, anima_dir: Path) -> None:
        """When no session ID exists, asyncio.wait_for is NOT called for resume guard."""
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        # No session file → fresh session path (no resume)
        messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "fresh"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("fresh")]),
            MockResultMessage(usage={"input_tokens": 50, "output_tokens": 20}),
        ]

        wait_for_calls: list[dict] = []
        original_wait_for = asyncio.wait_for

        async def _spy_wait_for(coro, timeout=None, **kwargs):
            wait_for_calls.append({"timeout": timeout})
            return await original_wait_for(coro, timeout=timeout, **kwargs)

        with _patch_sdk_for_streaming(messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with patch("asyncio.wait_for", side_effect=_spy_wait_for):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        # No wait_for should have been called for resume timeout
        from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

        resume_timeout_calls = [c for c in wait_for_calls if c["timeout"] == RESUME_TIMEOUT_SEC]
        assert resume_timeout_calls == [], (
            f"Expected no wait_for resume guard on fresh session, but got: {resume_timeout_calls}"
        )

    @pytest.mark.asyncio
    async def test_clear_session_id_called_when_all_resume_attempts_time_out(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """Fix 4c: when every resume attempt times out, _clear_session_id is
        called once (after RESUME_MAX_ATTEMPTS) and falls back to fresh session."""
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        _save_session_id(anima_dir, "stale-session-for-timeout", "chat")

        # After timeout, fresh session produces events normally
        fresh_messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "recovered"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("recovered")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        clear_calls: list[str] = []
        original_clear = None

        def _spy_clear(anima_dir_arg, session_type, **kwargs):
            clear_calls.append(session_type)
            if original_clear:
                original_clear(anima_dir_arg, session_type, **kwargs)

        guard_timeouts = [0]

        async def _always_timeout_resume_guard(coro, timeout=None, **kwargs):
            """Raise TimeoutError on every resume-guard call, succeed otherwise."""
            from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

            if timeout == RESUME_TIMEOUT_SEC:
                guard_timeouts[0] += 1
                coro.close()
                raise TimeoutError("resume timed out")
            return await coro

        with _patch_sdk_for_streaming(fresh_messages):
            from core.execution.agent_sdk import AgentSDKExecutor, _clear_session_id
            from core.prompt.context import ContextTracker

            original_clear = _clear_session_id

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with (
                patch("asyncio.wait_for", side_effect=_always_timeout_resume_guard),
                patch("core.execution._sdk_session._clear_session_id", side_effect=_spy_clear),
            ):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        from core.execution._sdk_session import RESUME_MAX_ATTEMPTS

        # All attempts were made before discarding the session id
        assert guard_timeouts[0] == RESUME_MAX_ATTEMPTS, (
            f"Expected {RESUME_MAX_ATTEMPTS} resume attempts, got {guard_timeouts[0]}"
        )
        assert len(clear_calls) >= 1, (
            f"Expected _clear_session_id to be called on resume timeout, but got: {clear_calls}"
        )
        assert "chat" in clear_calls, f"Expected 'chat' session to be cleared, got: {clear_calls}"

    @pytest.mark.asyncio
    async def test_transient_timeout_retried_without_discarding_session(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        """Fix 4c: a single transient timeout is retried; the session id is
        NOT cleared and the resumed session streams normally."""
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        _save_session_id(anima_dir, "transient-timeout-session", "chat")

        messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "resumed fine"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("resumed fine")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        clear_calls: list[str] = []

        def _spy_clear(anima_dir_arg, session_type, **kwargs):
            clear_calls.append(session_type)

        guard_calls = [0]

        async def _timeout_on_first_guard_only(coro, timeout=None, **kwargs):
            from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

            if timeout == RESUME_TIMEOUT_SEC:
                guard_calls[0] += 1
                if guard_calls[0] == 1:
                    coro.close()
                    raise TimeoutError("transient load spike")
            return await coro

        with _patch_sdk_for_streaming(messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with (
                patch("asyncio.wait_for", side_effect=_timeout_on_first_guard_only),
                patch("core.execution._sdk_session._clear_session_id", side_effect=_spy_clear),
            ):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="test",
                    tracker=tracker,
                ):
                    events.append(event)

        assert guard_calls[0] == 2, f"Expected a retry after the transient timeout, got {guard_calls[0]} calls"
        assert clear_calls == [], f"Session id must NOT be cleared on a retried transient timeout: {clear_calls}"
        text = "".join(e.get("text", "") for e in events if e.get("type") == "text_delta")
        assert "resumed fine" in text


# ── Resume fallback context recovery (Fix 4b) ─────────────────


def _write_chat_activity_log(anima_dir: Path, thread_id: str = "default") -> None:
    """Write a minimal chat exchange into today's activity_log file."""
    from core.time_utils import now_local

    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "ts": "2026-09-19T21:50:00+09:00",
            "type": "message_received",
            "content": "monthly usage table please",
            "meta": {"from_type": "human", "thread_id": thread_id},
        },
        {
            "ts": "2026-09-19T21:50:30+09:00",
            "type": "response_sent",
            "content": "Here is the monthly usage table with the average column.",
            "meta": {"thread_id": thread_id},
        },
    ]
    log_file = log_dir / f"{now_local().date().isoformat()}.jsonl"
    log_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


class TestResumeFallbackHandoff:
    """_build_resume_fallback_handoff() — Fix 4b graceful degradation."""

    def test_digest_returned_and_shortterm_saved(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _build_resume_fallback_handoff
        from core.memory.shortterm import ShortTermMemory

        _write_chat_activity_log(anima_dir)

        digest = _build_resume_fallback_handoff(anima_dir, "chat", "default", "sess-lost")

        assert "monthly usage table" in digest
        state = ShortTermMemory(anima_dir, session_type="chat", thread_id="default").load()
        assert state is not None
        assert state.trigger == "resume_fallback"
        assert "sess-lost" in state.notes
        assert "monthly usage table" in state.accumulated_response

    def test_pending_high_fidelity_handoff_not_overwritten(self, anima_dir: Path) -> None:
        """A pending context-threshold handoff must survive (same protection
        as Fix 1); the digest is still returned for prompt injection."""
        from core.execution.agent_sdk import _build_resume_fallback_handoff
        from core.memory.shortterm import SessionState, ShortTermMemory

        _write_chat_activity_log(anima_dir)
        shortterm = ShortTermMemory(anima_dir, session_type="chat", thread_id="default")
        shortterm.save(
            SessionState(
                accumulated_response="high fidelity full transcript",
                trigger="chat",
                timestamp="2026-09-19T21:00:00+09:00",
            )
        )

        digest = _build_resume_fallback_handoff(anima_dir, "chat", "default", "sess-lost")

        assert digest  # still injected into the prompt
        state = shortterm.load()
        assert state is not None
        assert state.trigger == "chat"
        assert state.accumulated_response == "high fidelity full transcript"

    def test_stale_resume_fallback_state_is_replaced(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _build_resume_fallback_handoff
        from core.memory.shortterm import SessionState, ShortTermMemory

        _write_chat_activity_log(anima_dir)
        shortterm = ShortTermMemory(anima_dir, session_type="chat", thread_id="default")
        shortterm.save(
            SessionState(
                accumulated_response="old fallback digest",
                trigger="resume_fallback",
                timestamp="2026-09-18T10:00:00+09:00",
            )
        )

        digest = _build_resume_fallback_handoff(anima_dir, "chat", "default", "sess-lost-2")

        assert digest
        state = shortterm.load()
        assert state is not None
        assert state.trigger == "resume_fallback"
        assert "monthly usage table" in state.accumulated_response

    def test_non_chat_session_returns_empty(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _build_resume_fallback_handoff

        _write_chat_activity_log(anima_dir)
        assert _build_resume_fallback_handoff(anima_dir, "heartbeat", "default", "s") == ""

    def test_empty_activity_log_returns_empty_and_saves_nothing(self, anima_dir: Path) -> None:
        from core.execution.agent_sdk import _build_resume_fallback_handoff
        from core.memory.shortterm import ShortTermMemory

        assert _build_resume_fallback_handoff(anima_dir, "chat", "default", "s") == ""
        assert ShortTermMemory(anima_dir, session_type="chat", thread_id="default").load() is None

    def test_inject_recovered_context_wraps_prompt(self) -> None:
        from core.execution.agent_sdk import _inject_recovered_context

        out = _inject_recovered_context("original prompt", "THE DIGEST")
        assert out.startswith("<recovered_context>")
        assert "THE DIGEST" in out
        assert out.rstrip().endswith("original prompt")


class TestResumeFallbackInjectionE2E:
    """Exhausted resume attempts → fresh session prompt carries the digest."""

    @pytest.mark.asyncio
    async def test_fresh_session_prompt_contains_recovered_context(
        self, model_config: ModelConfig, anima_dir: Path
    ) -> None:
        from core.execution import _sdk_stream
        from core.execution.agent_sdk import _save_session_id
        from tests.helpers.mocks import MockAssistantMessage, MockResultMessage, MockStreamEvent, MockTextBlock

        _save_session_id(anima_dir, "doomed-session", "chat")
        _write_chat_activity_log(anima_dir)

        fresh_messages = [
            MockStreamEvent(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "degraded but alive"},
                    "index": 0,
                }
            ),
            MockAssistantMessage([MockTextBlock("degraded but alive")]),
            MockResultMessage(usage={"input_tokens": 100, "output_tokens": 50}),
        ]

        async def _always_timeout_resume_guard(coro, timeout=None, **kwargs):
            from core.execution.agent_sdk import RESUME_TIMEOUT_SEC

            if timeout == RESUME_TIMEOUT_SEC:
                coro.close()
                raise TimeoutError("resume timed out")
            return await coro

        prompts: list[str] = []
        orig_psm = _sdk_stream.process_stream_messages

        def _spy_psm(client, ctx, state):
            prompts.append(ctx.prompt)
            return orig_psm(client, ctx, state)

        with _patch_sdk_for_streaming(fresh_messages):
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.prompt.context import ContextTracker

            executor = AgentSDKExecutor(model_config=model_config, anima_dir=anima_dir)
            tracker = ContextTracker(model="claude-sonnet-4-6")

            with (
                patch("asyncio.wait_for", side_effect=_always_timeout_resume_guard),
                patch("core.execution.agent_sdk.process_stream_messages", side_effect=_spy_psm),
            ):
                events = []
                async for event in executor.execute_streaming(
                    system_prompt="sys",
                    prompt="continue please",
                    tracker=tracker,
                ):
                    events.append(event)

        assert prompts, "process_stream_messages was never called"
        fresh_prompt = prompts[-1]
        assert "<recovered_context>" in fresh_prompt
        assert "monthly usage table" in fresh_prompt
        assert fresh_prompt.rstrip().endswith("continue please")

        # The digest was also persisted as a shortterm handoff
        from core.memory.shortterm import ShortTermMemory

        state = ShortTermMemory(anima_dir, session_type="chat", thread_id="default").load()
        assert state is not None
        assert state.trigger == "resume_fallback"
