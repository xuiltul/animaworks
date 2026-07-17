from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mode X Grok ACP executor.

Every subprocess and binary lookup is mocked; these tests never invoke Grok.
"""

import asyncio
import json
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.base import ExecutionResult, ToolCallRecord
from core.execution.grok_cli import (
    _MAX_RESUME_TURNS,
    GrokCLIExecutor,
    _find_grok_binary,
    _load_session_id,
    _resolve_grok_model,
    _resolve_session_type,
    _save_session_id,
    _session_id_path,
    is_grok_cli_available,
)
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "test-grok"
    path.mkdir(parents=True)
    (path / "identity.md").write_text("# Test Grok Anima", encoding="utf-8")
    return path


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="grok/grok-4.5",
        max_tokens=4096,
        max_turns=30,
        credential="grok",
        context_threshold=0.5,
        max_chains=2,
    )


@pytest.fixture
def executor(model_config: ModelConfig, anima_dir: Path) -> GrokCLIExecutor:
    return GrokCLIExecutor(model_config, anima_dir, tool_registry=["web_search"])


def _make_ndjson_lines(events: list[dict | str | bytes]) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        if isinstance(event, bytes):
            line = event
        elif isinstance(event, str):
            line = event.encode()
        else:
            line = json.dumps(event).encode()
        lines.append(line if line.endswith(b"\n") else line + b"\n")
    lines.append(b"")
    return lines


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None


class _FakeStream:
    def __init__(self, lines: list[bytes] | None = None, data: bytes = b"") -> None:
        self.lines = list(lines or [b""])
        self.data = data

    async def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    async def read(self) -> bytes:
        return self.data


class _FakeProc:
    def __init__(
        self,
        events: list[dict | str | bytes],
        *,
        returncode: int | None = 0,
        stderr: bytes = b"",
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(_make_ndjson_lines(events))
        self.stderr = _FakeStream(data=stderr)
        self.returncode = returncode
        self.pid = 1234
        self.sent_signals: list[signal.Signals] = []
        self.kill_calls = 0

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, sig: signal.Signals) -> None:
        self.sent_signals.append(sig)

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _success_events(
    *,
    session_id: str = "session-new",
    updates: list[dict] | None = None,
    usage: dict | None = None,
    load: bool = False,
) -> list[dict]:
    events: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "result": {"agentCapabilities": {"loadSession": True}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": session_id} if not load else {}},
    ]
    events.extend(updates or [])
    events.append(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "stopReason": "end_turn",
                "_meta": {
                    "sessionId": session_id,
                    "usage": usage or {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
                },
            },
        }
    )
    return events


def _requests(proc: _FakeProc) -> list[dict]:
    return [json.loads(line) for line in proc.stdin.written.splitlines()]


async def _stream(
    executor: GrokCLIExecutor,
    proc: _FakeProc,
    *,
    trigger: str = "heartbeat",
    thread_id: str = "default",
    tracker: ContextTracker | MagicMock | None = None,
) -> list[dict]:
    tracker = tracker or ContextTracker(model="grok/grok-4.5")
    with (
        patch("core.execution.grok_cli._find_grok_binary", return_value="/usr/bin/grok"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        return [
            event
            async for event in executor.execute_streaming(
                "You are Grok.",
                "hello",
                tracker,
                trigger=trigger,
                thread_id=thread_id,
            )
        ]


class TestDiscoveryAndHelpers:
    def test_binary_discovery_and_availability(self):
        with patch("core.execution.grok_cli.shutil.which", return_value="/opt/grok"):
            assert _find_grok_binary() == "/opt/grok"
        with patch("core.execution.grok_cli._find_grok_binary", return_value="/opt/grok"):
            assert is_grok_cli_available() is True
        with patch("core.execution.grok_cli._find_grok_binary", return_value=None):
            assert is_grok_cli_available() is False

    @pytest.mark.parametrize(
        ("model", "expected"),
        [("grok/grok-4.5", "grok-4.5"), ("grok-4.5", "grok-4.5")],
    )
    def test_model_prefix(self, model: str, expected: str):
        assert _resolve_grok_model(model) == expected

    def test_session_paths_and_trigger_normalization(self, anima_dir: Path):
        assert _resolve_session_type("message:user") == "chat"
        assert _resolve_session_type("chat:web") == "chat"
        assert _resolve_session_type("heartbeat:timer") == "heartbeat"
        expected = anima_dir / "shortterm/chat/thread-a/grok_session_id.txt"
        assert _session_id_path(anima_dir, "chat", "thread-a") == expected
        _save_session_id(anima_dir, "sid-a", "chat", "thread-a", 4)
        assert _load_session_id(anima_dir, "chat", "thread-a") == ("sid-a", 4)
        assert _load_session_id(anima_dir, "chat", "thread-b") == (None, 0)


class TestACPProtocol:
    @pytest.mark.asyncio
    async def test_handshake_command_and_stdin(self, executor: GrokCLIExecutor):
        proc = _FakeProc(_success_events())
        events = await _stream(executor, proc)

        requests = _requests(proc)
        assert [request.get("method") for request in requests] == [
            "initialize",
            "session/new",
            "session/prompt",
        ]
        initialize = requests[0]["params"]
        assert initialize["protocolVersion"] == 1
        assert isinstance(initialize["protocolVersion"], int)
        new = requests[1]["params"]
        assert Path(new["cwd"]).is_absolute()
        assert new["_meta"]["systemPromptOverride"] == "You are Grok."
        assert isinstance(new["mcpServers"], list)
        aw = new["mcpServers"][0]
        assert aw["name"] == "aw"
        assert aw["args"] == ["-m", "core.mcp.server"]
        # ACP requires EnvVariable objects ([{name, value}]), not a mapping
        assert {entry["name"] for entry in aw["env"]} == {
            "ANIMAWORKS_ANIMA_DIR",
            "ANIMAWORKS_PROJECT_DIR",
            "PYTHONPATH",
            "PATH",
        }
        assert all(isinstance(entry["value"], str) for entry in aw["env"])
        assert requests[2]["params"] == {
            "sessionId": "session-new",
            "prompt": [{"type": "text", "text": "hello"}],
        }
        done = [event for event in events if event["type"] == "done"]
        assert len(done) == 1

    def test_build_command_order(self, executor: GrokCLIExecutor):
        with patch("core.execution.grok_cli._find_grok_binary", return_value="/usr/bin/grok"):
            assert executor._build_command() == [
                "/usr/bin/grok",
                "agent",
                "--always-approve",
                "-m",
                "grok-4.5",
                "stdio",
            ]

    @pytest.mark.asyncio
    async def test_permission_request_is_approved(self, executor: GrokCLIExecutor):
        events = _success_events()
        events.insert(
            0,
            {
                "jsonrpc": "2.0",
                "id": 77,
                "method": "session/request_permission",
                "params": {
                    "options": [
                        {"optionId": "deny", "name": "Deny"},
                        {"optionId": "allow", "name": "Allow once"},
                    ]
                },
            },
        )
        proc = _FakeProc(events)
        await _stream(executor, proc)
        response = next(item for item in _requests(proc) if item.get("id") == 77)
        assert response["result"]["outcome"]["optionId"] == "allow"

    @pytest.mark.asyncio
    async def test_non_json_empty_and_unknown_updates_are_ignored(self, executor: GrokCLIExecutor):
        updates = [
            {"method": "session/update", "params": {"update": {"sessionUpdate": "user_message_chunk"}}},
            {"method": "session/update", "params": {"update": {"sessionUpdate": "future_update"}}},
        ]
        raw_events: list[dict | str | bytes] = ["not-json", b"\n"]
        raw_events.extend(_success_events(updates=updates))
        proc = _FakeProc(raw_events)
        events = await _stream(executor, proc)
        assert [event["type"] for event in events] == ["done"]


class TestEventConversion:
    @pytest.mark.asyncio
    async def test_text_thinking_tools_usage_and_done_contract(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "think"}}
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hello "}}
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tool-1",
                        "title": "fallback-name",
                        "rawInput": {"query": "hello"},
                        "_meta": {"x.ai/tool": {"name": "mcp__aw__search_memory"}},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "tool-1",
                        "status": "completed",
                        "rawOutput": {"answer": "found"},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "world"}}
                },
            },
        ]
        tracker = MagicMock()
        proc = _FakeProc(
            _success_events(
                updates=updates,
                usage={"inputTokens": 101, "outputTokens": 9, "cachedReadTokens": 80},
            )
        )
        events = await _stream(executor, proc, tracker=tracker)

        assert [event["type"] for event in events] == [
            "thinking_delta",
            "text_delta",
            "tool_start",
            "tool_end",
            "text_delta",
            "done",
        ]
        done = events[-1]
        assert done["full_text"] == "Hello world"
        assert isinstance(done["result_message"].num_turns, int)
        assert done["result_message"].session_id == "session-new"
        assert done["tool_call_records"][0]["tool_name"] == "search_memory"
        assert isinstance(done["tool_call_records"][0], dict)
        assert done["usage"] == {
            "input_tokens": 101,
            "output_tokens": 9,
            "cache_read_tokens": 80,
            "cache_write_tokens": 0,
        }
        tracker.update_from_usage.assert_called_once_with(done["usage"])

    @pytest.mark.asyncio
    async def test_failed_tool_update(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "t2",
                        "title": "mcp_aw_bash",
                        "rawInput": "x" * 700,
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "t2",
                        "status": "failed",
                        "rawOutput": "bad",
                    }
                },
            },
        ]
        events = await _stream(executor, _FakeProc(_success_events(updates=updates)))
        end = next(event for event in events if event["type"] == "tool_end")
        assert end["tool_name"] == "bash"
        assert end["is_error"] is True
        record = events[-1]["tool_call_records"][0]
        assert record["is_error"] is True
        assert len(record["input_summary"]) <= 503

    @pytest.mark.asyncio
    async def test_blocking_result_uses_dataclasses(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "answer"}}},
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "t",
                        "title": "read_file",
                        "rawInput": {"path": "x"},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "t",
                        "status": "completed",
                        "rawOutput": "ok",
                    }
                },
            },
        ]
        proc = _FakeProc(_success_events(updates=updates))
        with (
            patch("core.execution.grok_cli._find_grok_binary", return_value="/usr/bin/grok"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute("hello", "system", trigger="heartbeat")
        assert isinstance(result, ExecutionResult)
        assert result.text == "answer"
        assert isinstance(result.tool_call_records[0], ToolCallRecord)
        assert result.usage is not None and result.usage.input_tokens == 12


class TestSessions:
    @pytest.mark.asyncio
    async def test_chat_persists_then_resumes_without_system_override(self, executor: GrokCLIExecutor, anima_dir: Path):
        first = _FakeProc(_success_events(session_id="saved"))
        await _stream(executor, first, trigger="message:user", thread_id="thread-a")
        assert _load_session_id(anima_dir, "chat", "thread-a") == ("saved", 1)

        second = _FakeProc(_success_events(session_id="saved", load=True))
        await _stream(executor, second, trigger="chat", thread_id="thread-a")
        requests = _requests(second)
        assert requests[1]["method"] == "session/load"
        assert requests[1]["params"]["sessionId"] == "saved"
        assert "_meta" not in requests[1]["params"]
        assert _load_session_id(anima_dir, "chat", "thread-a") == ("saved", 2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trigger", ["heartbeat", "cron:daily", "task:run"])
    async def test_non_chat_triggers_are_fresh(self, executor: GrokCLIExecutor, trigger: str):
        proc = _FakeProc(_success_events())
        await _stream(executor, proc, trigger=trigger)
        assert _requests(proc)[1]["method"] == "session/new"

    @pytest.mark.asyncio
    async def test_resume_failure_retries_once_fresh(self, executor: GrokCLIExecutor, anima_dir: Path):
        _save_session_id(anima_dir, "stale", "chat", "default", 3)
        failed = _FakeProc(
            [
                {"id": 1, "result": {}},
                {"id": 2, "error": {"code": -32000, "message": "session not found"}},
            ]
        )
        fresh = _FakeProc(_success_events(session_id="recovered"))
        with (
            patch("core.execution.grok_cli._find_grok_binary", return_value="/usr/bin/grok"),
            patch("asyncio.create_subprocess_exec", side_effect=[failed, fresh]) as create,
        ):
            events = [
                event
                async for event in executor.execute_streaming(
                    "system", "hello", ContextTracker(model="grok/grok-4.5"), trigger="chat"
                )
            ]
        assert create.call_count == 2
        assert _requests(failed)[1]["method"] == "session/load"
        assert _requests(fresh)[1]["method"] == "session/new"
        assert _load_session_id(anima_dir, "chat") == ("recovered", 1)
        assert events[-1]["session_rotated"] is True

    @pytest.mark.asyncio
    async def test_turn_rotation_and_pending(self, executor: GrokCLIExecutor, anima_dir: Path):
        _save_session_id(anima_dir, "old", "chat", "default", _MAX_RESUME_TURNS)
        rotated = _FakeProc(_success_events(session_id="fresh"))
        rotated_events = await _stream(executor, rotated, trigger="chat")
        assert _requests(rotated)[1]["method"] == "session/new"
        assert rotated_events[-1]["session_rotated"] is True
        assert rotated_events[-1]["session_rotation_pending"] is False

        _save_session_id(anima_dir, "almost", "chat", "default", _MAX_RESUME_TURNS - 1)
        pending = _FakeProc(_success_events(session_id="almost", load=True))
        pending_events = await _stream(executor, pending, trigger="chat")
        assert pending_events[-1]["session_rotated"] is False
        assert pending_events[-1]["session_rotation_pending"] is True
        assert _load_session_id(anima_dir, "chat") == ("almost", _MAX_RESUME_TURNS)


class TestTerminalPaths:
    @pytest.mark.asyncio
    async def test_not_installed_done_once(self, executor: GrokCLIExecutor):
        with patch("core.execution.grok_cli._find_grok_binary", return_value=None):
            events = [
                event
                async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
            ]
        assert len([event for event in events if event["type"] == "done"]) == 1
        done = events[-1]
        assert isinstance(done["full_text"], str)
        assert done["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_preexisting_interrupt_done_once(self, executor: GrokCLIExecutor):
        executor._interrupt_event = asyncio.Event()
        executor._interrupt_event.set()
        events = [
            event
            async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
        ]
        assert [event["type"] for event in events] == ["done"]
        assert "interrupted" in events[0]["full_text"].lower()

    @pytest.mark.asyncio
    async def test_file_not_found_done_once(self, executor: GrokCLIExecutor):
        with (
            patch("core.execution.grok_cli._find_grok_binary", return_value="/missing/grok"),
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
        ):
            events = [
                event
                async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
            ]
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "grok" in events[-1]["full_text"].lower()

    @pytest.mark.asyncio
    async def test_auth_error_from_stderr(self, executor: GrokCLIExecutor):
        proc = _FakeProc([], returncode=1, stderr=b"Unauthenticated: run grok login")
        events = await _stream(executor, proc)
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "login" in events[-1]["full_text"].lower() or "認証" in events[-1]["full_text"]

    @pytest.mark.asyncio
    async def test_idle_timeout_kills_stalled_stream(self, executor: GrokCLIExecutor):
        # A child that emits the handshake but then goes silent must be killed
        # once the idle gap between events exceeds the progress timeout, and
        # surface a single terminal done event with the timeout message.
        proc = _FakeProc(_success_events(), returncode=None)
        served = 0

        async def stall_after_session() -> bytes:
            # Serve only the two handshake responses (initialize + session/new),
            # then hang so the event-loop readline trips the idle timeout while
            # awaiting the prompt result — not EOF.
            nonlocal served
            if served < 2:
                served += 1
                return proc.stdout.lines.pop(0)
            await asyncio.sleep(3600)
            return b""

        proc.stdout.readline = stall_after_session  # type: ignore[method-assign]
        with patch("core.execution.grok_cli._IDLE_TIMEOUT_SECONDS", 0.01):
            events = await _stream(executor, proc)
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "grok" in events[-1]["full_text"].lower()
        assert signal.SIGTERM in proc.sent_signals

    @pytest.mark.asyncio
    async def test_inflight_interrupt_sends_cancel(self, executor: GrokCLIExecutor):
        executor._interrupt_event = asyncio.Event()
        proc = _FakeProc(_success_events(), returncode=None)
        original_readline = proc.stdout.readline
        reads = 0

        async def interrupt_after_session() -> bytes:
            nonlocal reads
            line = await original_readline()
            reads += 1
            if reads == 2:
                executor._interrupt_event.set()
            return line

        proc.stdout.readline = interrupt_after_session  # type: ignore[method-assign]
        events = await _stream(executor, proc)
        methods = [request.get("method") for request in _requests(proc)]
        assert "session/cancel" in methods
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "interrupted" in events[-1]["full_text"].lower()

    @pytest.mark.asyncio
    async def test_kill_escalates_after_grace_period(self, executor: GrokCLIExecutor):
        proc = MagicMock()
        proc.returncode = None
        proc.wait = AsyncMock(side_effect=[TimeoutError, -9])
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        await executor._kill_process(proc, timeout=0)
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.kill.assert_called_once()
