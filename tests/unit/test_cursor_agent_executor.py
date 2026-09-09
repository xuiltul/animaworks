from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CursorAgentExecutor (Mode D).

All tests use mocks — no cursor-agent CLI binary required.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.execution.base import ExecutionResult
from core.execution.cursor_agent import (
    _MAX_RESUME_TURNS,
    _RESUMABLE_TRIGGERS,
    _cursor_error_metadata,
    CursorAgentExecutor,
    _chat_id_path,
    _clear_chat_id,
    _find_cursor_agent_binary,
    _load_chat_id,
    _resolve_session_type,
    _save_chat_id,
    is_cursor_agent_available,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-cursor"
    d.mkdir(parents=True)
    (d / "shortterm" / "chat").mkdir(parents=True)
    (d / "shortterm" / "heartbeat").mkdir(parents=True)
    (d / "identity.md").write_text("# Test Cursor Anima", encoding="utf-8")
    (d / "state").mkdir()
    (d / "state" / "current_state.md").write_text("status: idle\n", encoding="utf-8")
    return d


@pytest.fixture
def model_config():
    from core.schemas import ModelConfig

    return ModelConfig(
        model="cursor/claude-4-sonnet",
        max_tokens=4096,
        credential="cursor",
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def executor(model_config, anima_dir):
    return CursorAgentExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


# ── Helper tests ─────────────────────────────────────────────


class TestBinaryDiscovery:
    def test_find_binary_returns_first_match(self):
        with patch("shutil.which", side_effect=lambda n: f"/usr/bin/{n}" if n == "agent" else None):
            assert _find_cursor_agent_binary() == "/usr/bin/agent"

    def test_find_binary_fallback_to_cursor_agent(self):
        def _which(name):
            return "/usr/local/bin/cursor-agent" if name == "cursor-agent" else None

        with patch("shutil.which", side_effect=_which):
            assert _find_cursor_agent_binary() == "/usr/local/bin/cursor-agent"

    def test_find_binary_returns_none_when_missing(self):
        with patch("shutil.which", return_value=None):
            assert _find_cursor_agent_binary() is None

    def test_is_available_true(self):
        with patch("core.execution.cursor_agent._find_cursor_agent_binary", return_value="/usr/bin/agent"):
            assert is_cursor_agent_available() is True

    def test_is_available_false(self):
        with patch("core.execution.cursor_agent._find_cursor_agent_binary", return_value=None):
            assert is_cursor_agent_available() is False


class TestModelResolution:
    def test_strip_cursor_prefix(self, executor):
        assert executor._resolve_cursor_model() == "claude-4-sonnet"

    def test_no_prefix(self, executor):
        executor._model_config.model = "gpt-4.1"
        assert executor._resolve_cursor_model() == "gpt-4.1"


class TestWorkspace:
    def test_ensure_workspace_creates_dirs(self, executor):
        executor._ensure_workspace()
        assert executor._workspace.is_dir()
        assert (executor._workspace / ".cursor").is_dir()

    def test_write_mcp_config(self, executor):
        executor._ensure_workspace()
        executor._write_mcp_config()
        mcp_path = executor._workspace / ".cursor" / "mcp.json"
        assert mcp_path.exists()
        config = json.loads(mcp_path.read_text())
        assert "mcpServers" in config
        assert "aw" in config["mcpServers"]
        aw_conf = config["mcpServers"]["aw"]
        assert "-m" in aw_conf["args"]
        assert "core.mcp.server" in aw_conf["args"]
        assert "ANIMAWORKS_ANIMA_DIR" in aw_conf["env"]

    def test_workspace_location(self, executor, anima_dir):
        assert executor._workspace == anima_dir / ".cursor-workspace"


class TestBuildCommand:
    def test_build_command_structure(self, executor):
        with patch.object(executor, "_find_binary", return_value="/usr/bin/agent"):
            cmd = executor._build_command("hello")
        assert cmd[0] == "/usr/bin/agent"
        assert "-p" in cmd
        assert "--force" in cmd
        assert "--trust" in cmd
        assert "--approve-mcps" in cmd
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"
        assert "--stream-partial-output" in cmd
        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-4-sonnet"
        assert cmd[-1] == "hello"
        assert "--resume" not in cmd

    def test_build_command_empty_when_no_binary(self, executor):
        with patch.object(executor, "_find_binary", return_value=None):
            assert executor._build_command("test") == []

    def test_build_command_with_resume(self, executor):
        with patch.object(executor, "_find_binary", return_value="/usr/bin/agent"):
            cmd = executor._build_command("hello", resume_chat_id="abc-123-def")
        assert "--resume" in cmd
        ri = cmd.index("--resume")
        assert cmd[ri + 1] == "abc-123-def"
        assert cmd[-1] == "hello"

    def test_build_command_without_resume(self, executor):
        with patch.object(executor, "_find_binary", return_value="/usr/bin/agent"):
            cmd = executor._build_command("hello", resume_chat_id=None)
        assert "--resume" not in cmd


class TestBuildEnv:
    def test_includes_path_and_home(self, executor):
        env = executor._build_env()
        assert "PATH" in env
        assert "HOME" in env

    def test_no_api_key_injected(self, executor):
        executor._model_config.api_key = "test-anthropic-key"
        env = executor._build_env()
        assert "CURSOR_API_KEY" not in env or env.get("CURSOR_API_KEY") != "test-anthropic-key"


class TestNDJSONParsing:
    def test_parse_valid_json(self, executor):
        event = executor._parse_ndjson_event('{"type": "system", "subtype": "init"}')
        assert event == {"type": "system", "subtype": "init"}

    def test_parse_empty_line(self, executor):
        assert executor._parse_ndjson_event("") is None
        assert executor._parse_ndjson_event("  ") is None

    def test_parse_invalid_json(self, executor):
        assert executor._parse_ndjson_event("not json") is None

    def test_parse_whitespace_stripped(self, executor):
        event = executor._parse_ndjson_event('  {"type": "result"}  \n')
        assert event == {"type": "result"}


class TestToolRecordExtraction:
    def test_read_tool_call(self, executor):
        tc = {"readToolCall": {"args": {"path": "/tmp/test.py"}}, "id": "call_1"}
        record = executor._extract_tool_record(tc)
        assert record is not None
        assert record.tool_name == "Read"
        assert record.tool_id == "call_1"

    def test_write_tool_call(self, executor):
        tc = {
            "writeToolCall": {"args": {"path": "/tmp/out.py", "content": "..."}, "result": {"success": {}}},
            "id": "call_2",
        }
        record = executor._extract_tool_record(tc)
        assert record is not None
        assert record.tool_name == "Write"

    def test_function_tool_call(self, executor):
        tc = {"function": {"name": "web_search", "arguments": '{"query": "test"}'}, "id": "call_3"}
        record = executor._extract_tool_record(tc)
        assert record is not None
        assert record.tool_name == "web_search"

    def test_mcp_aw_prefix_stripped(self, executor):
        tc = {"function": {"name": "mcp__aw__send_message", "arguments": "{}"}, "id": "call_4"}
        record = executor._extract_tool_record(tc)
        assert record is not None
        assert record.tool_name == "send_message"

    def test_error_flag(self, executor):
        tc = {"function": {"name": "test"}, "is_error": True, "id": "call_5"}
        record = executor._extract_tool_record(tc)
        assert record is not None
        assert record.is_error is True


# ── Execute tests ─────────────────────────────────────────────


def _make_ndjson_lines(events: list[dict]) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


class TestExecute:
    @pytest.mark.asyncio
    async def test_not_installed(self, executor):
        with patch.object(executor, "_find_binary", return_value=None):
            result = await executor.execute(prompt="hello")
        assert "cursor-agent" in result.text.lower() or "not found" in result.text.lower()

    @pytest.mark.asyncio
    async def test_interrupted(self, executor):
        executor._interrupt_event = asyncio.Event()
        executor._interrupt_event.set()
        result = await executor.execute(prompt="hello")
        assert "interrupted" in result.text.lower()

    @pytest.mark.asyncio
    async def test_successful_execution(self, executor):
        events = [
            {"type": "system", "subtype": "init", "model": "claude-4-sonnet"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello, "}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world!"}]}},
            {"type": "result", "subtype": "success", "result": "Hello, world!", "duration_ms": 1234},
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")

        lines = stdout_data.split(b"\n")
        line_iter = iter(lines)
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="hello", system_prompt="You are helpful")

        assert result.text == "Hello, world!"
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_tool_turn_text_is_kept_in_final_result(self, executor):
        events = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Context before the tool call."}]},
            },
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"readToolCall": {"args": {"path": "README.md"}}},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "README.md"},
                        "result": {"success": {"content": "contents"}},
                    }
                },
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Final answer after the tool call."}]},
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "Context before the tool call. Final answer after the tool call.",
            },
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        lines = iter(stdout_data.split(b"\n"))
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="use a tool")

        assert result.text == "Context before the tool call.\n\nFinal answer after the tool call."
        assert len(result.tool_call_records) == 1

    @pytest.mark.asyncio
    async def test_system_prompt_injected_as_prefix(self, executor):
        captured_cmd = []

        async def mock_create(*args, **kwargs):
            captured_cmd.extend(args)
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            result_line = json.dumps({"type": "result", "result": "ok"}).encode() + b"\n"
            lines = iter([result_line, b""])
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="do something", system_prompt="You are Alice")

        prompt_arg = captured_cmd[-1]
        assert "<system_context>" in prompt_arg
        assert "You are Alice" in prompt_arg
        assert "do something" in prompt_arg

    @pytest.mark.asyncio
    async def test_tool_records_extracted(self, executor):
        events = [
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "readToolCall": {"args": {"path": "/tmp/test.py"}},
                    "id": "tc_1",
                },
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "function": {"name": "mcp__aw__search_memory", "arguments": '{"query": "test"}'},
                    "id": "tc_2",
                },
            },
            {"type": "result", "result": "done"},
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        lines = iter(stdout_data.split(b"\n"))
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="test")

        assert len(result.tool_call_records) == 2
        assert result.tool_call_records[0].tool_name == "Read"
        assert result.tool_call_records[1].tool_name == "search_memory"

    @pytest.mark.asyncio
    async def test_auth_error_detected(self, executor):
        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Error: not authenticated, please run agent login")
        mock_proc.stdout.readline = AsyncMock(return_value=b"")

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="hello")

        assert "login" in result.text.lower() or "認証" in result.text

    @pytest.mark.asyncio
    async def test_nonzero_exit_without_auth(self, executor):
        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Some random error")
        mock_proc.stdout.readline = AsyncMock(return_value=b"")

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="hello")

        assert "error" in result.text.lower()


# ── Session persistence tests ─────────────────────────────────


class TestSessionTypeResolution:
    def test_empty_trigger_is_chat(self):
        assert _resolve_session_type("") == "chat"

    def test_chat_trigger(self):
        assert _resolve_session_type("chat") == "chat"

    def test_message_trigger_maps_to_chat(self):
        assert _resolve_session_type("message:owner") == "chat"

    def test_message_bare_maps_to_chat(self):
        assert _resolve_session_type("message") == "chat"

    def test_heartbeat_trigger(self):
        assert _resolve_session_type("heartbeat") == "heartbeat"

    def test_cron_trigger(self):
        assert _resolve_session_type("cron:daily_check") == "cron"

    def test_task_trigger(self):
        assert _resolve_session_type("task:abc123") == "task"

    def test_inbox_trigger(self):
        assert _resolve_session_type("inbox:alice") == "inbox"


class TestChatIdPersistence:
    def test_save_and_load(self, anima_dir):
        _save_chat_id(anima_dir, "sess-abc-123", "chat", turn_count=3)
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "sess-abc-123"
        assert tc == 3

    def test_save_default_turn_count(self, anima_dir):
        _save_chat_id(anima_dir, "sess-abc-123", "chat")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "sess-abc-123"
        assert tc == 1

    def test_load_missing_returns_none(self, anima_dir):
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid is None
        assert tc == 0

    def test_load_empty_file_returns_none(self, anima_dir):
        p = _chat_id_path(anima_dir, "chat")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid is None
        assert tc == 0

    def test_load_whitespace_only_returns_none(self, anima_dir):
        p = _chat_id_path(anima_dir, "chat")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("   \n  ", encoding="utf-8")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid is None
        assert tc == 0

    def test_backward_compatible_single_line(self, anima_dir):
        """Legacy 1-line format (chatId only) → turn_count=0."""
        p = _chat_id_path(anima_dir, "chat")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("legacy-session-id", encoding="utf-8")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "legacy-session-id"
        assert tc == 0

    def test_corrupted_turn_count(self, anima_dir):
        """Non-integer second line → turn_count=0."""
        p = _chat_id_path(anima_dir, "chat")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("some-session\nNOT_A_NUMBER", encoding="utf-8")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "some-session"
        assert tc == 0

    def test_clear(self, anima_dir):
        _save_chat_id(anima_dir, "sess-abc-123", "chat")
        _clear_chat_id(anima_dir, "chat")
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid is None
        assert tc == 0

    def test_clear_missing_no_error(self, anima_dir):
        _clear_chat_id(anima_dir, "chat")

    def test_thread_id_isolation(self, anima_dir):
        _save_chat_id(anima_dir, "sess-default", "chat", "default", turn_count=2)
        _save_chat_id(anima_dir, "sess-thread-a", "chat", "thread-a", turn_count=5)
        cid_d, tc_d = _load_chat_id(anima_dir, "chat", "default")
        cid_a, tc_a = _load_chat_id(anima_dir, "chat", "thread-a")
        assert cid_d == "sess-default"
        assert tc_d == 2
        assert cid_a == "sess-thread-a"
        assert tc_a == 5

    def test_path_default_thread(self, anima_dir):
        p = _chat_id_path(anima_dir, "chat", "default")
        assert p == anima_dir / "shortterm" / "chat" / "cursor_chat_id.txt"

    def test_path_custom_thread(self, anima_dir):
        p = _chat_id_path(anima_dir, "chat", "my-thread")
        assert p == anima_dir / "shortterm" / "chat" / "my-thread" / "cursor_chat_id.txt"

    def test_resumable_triggers_contains_chat(self):
        assert "chat" in _RESUMABLE_TRIGGERS
        assert "heartbeat" not in _RESUMABLE_TRIGGERS
        assert "cron" not in _RESUMABLE_TRIGGERS


class TestSessionResume:
    """Tests for session_id capture, chatId save, and resume retry."""

    @pytest.mark.asyncio
    async def test_session_id_captured_and_saved(self, executor, anima_dir):
        """First chat → session_id extracted from NDJSON and saved."""
        events = [
            {"type": "system", "subtype": "init", "model": "auto", "session_id": "sid-aaa-bbb-ccc"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi!"}]}},
            {"type": "result", "subtype": "success", "result": "Hi!", "duration_ms": 100},
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        lines = iter(stdout_data.split(b"\n"))
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await executor.execute(prompt="hello", trigger="chat")

        assert result.text == "Hi!"
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "sid-aaa-bbb-ccc"
        assert tc == 1

    @pytest.mark.asyncio
    async def test_resume_uses_saved_chat_id(self, executor, anima_dir):
        """Second chat → --resume flag with saved chatId."""
        _save_chat_id(anima_dir, "prev-session-id", "chat", turn_count=2)
        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "new-session-id"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="follow up", trigger="chat")

        flat_cmd = captured_cmds[0]
        assert "--resume" in flat_cmd
        ri = flat_cmd.index("--resume")
        assert flat_cmd[ri + 1] == "prev-session-id"
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "new-session-id"
        assert tc == 3

    @pytest.mark.asyncio
    async def test_resume_failure_retries_fresh(self, executor, anima_dir):
        """Resume fails → clear chatId → retry without --resume."""
        _save_chat_id(anima_dir, "stale-session", "chat", turn_count=3)
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.wait = AsyncMock()

            if call_count == 1:
                mock_proc.returncode = 1
                mock_proc.stderr.read = AsyncMock(return_value=b"Session not found")
                mock_proc.stdout.readline = AsyncMock(return_value=b"")
            else:
                mock_proc.returncode = 0
                mock_proc.stderr.read = AsyncMock(return_value=b"")
                events = [
                    {"type": "system", "subtype": "init", "session_id": "fresh-session"},
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "Recovered!"}]}},
                    {"type": "result", "result": "Recovered!"},
                ]
                line_data = _make_ndjson_lines(events)
                line_iter = iter(line_data.split(b"\n"))
                mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            result = await executor.execute(prompt="retry test", trigger="chat")

        assert call_count == 2
        assert result.text == "Recovered!"
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "fresh-session"
        assert tc == 1

    @pytest.mark.asyncio
    async def test_non_resumable_trigger_skips_resume(self, executor, anima_dir):
        """heartbeat trigger → no resume even if chatId file exists."""
        _save_chat_id(anima_dir, "should-not-be-used", "heartbeat")

        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [{"type": "result", "result": "done"}]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="heartbeat check", trigger="heartbeat")

        flat_cmd = captured_cmds[0]
        assert "--resume" not in flat_cmd

    @pytest.mark.asyncio
    async def test_no_session_id_in_output(self, executor, anima_dir):
        """When NDJSON has no session_id → no chatId saved."""
        events = [
            {"type": "system", "subtype": "init", "model": "auto"},
            {"type": "result", "result": "ok"},
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        lines = iter(stdout_data.split(b"\n"))
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            await executor.execute(prompt="hello", trigger="chat")

        cid, _ = _load_chat_id(anima_dir, "chat")
        assert cid is None

    @pytest.mark.asyncio
    async def test_auth_error_does_not_retry(self, executor, anima_dir):
        """Auth errors should not trigger resume retry."""
        _save_chat_id(anima_dir, "some-session", "chat", turn_count=2)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Error: not authenticated, please run agent login")
        mock_proc.stdout.readline = AsyncMock(return_value=b"")

        create_call_count = 0

        async def counting_create(*args, **kwargs):
            nonlocal create_call_count
            create_call_count += 1
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=counting_create),
        ):
            result = await executor.execute(prompt="hello", trigger="chat")

        assert create_call_count == 1
        assert "login" in result.text.lower() or "認証" in result.text

    @pytest.mark.asyncio
    async def test_thread_id_passed_to_session(self, executor, anima_dir):
        """Different thread_id → different chatId file."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "thread-b-session"},
            {"type": "result", "result": "ok"},
        ]
        stdout_data = _make_ndjson_lines(events)

        mock_proc = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        lines = iter(stdout_data.split(b"\n"))
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            await executor.execute(prompt="hello", trigger="chat", thread_id="thread-b")

        cid_b, _ = _load_chat_id(anima_dir, "chat", "thread-b")
        assert cid_b == "thread-b-session"
        cid_d, _ = _load_chat_id(anima_dir, "chat", "default")
        assert cid_d is None


# ── Turn rotation tests ──────────────────────────────────────


class TestTurnRotation:
    """Tests for A+C hybrid session rotation."""

    @pytest.mark.asyncio
    async def test_rotation_at_session_limit(self, executor, anima_dir):
        """When turn_count >= MAX → chatId cleared, fresh session."""
        _save_chat_id(anima_dir, "old-session", "chat", turn_count=_MAX_RESUME_TURNS)
        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "new-rotated-session"},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "Fresh!"}]}},
                {"type": "result", "result": "Fresh!"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            result = await executor.execute(prompt="hello", system_prompt="You are helpful", trigger="chat")

        assert result.text == "Fresh!"
        assert result.session_rotated is True
        assert result.session_rotation_pending is False
        flat_cmd = captured_cmds[0]
        assert "--resume" not in flat_cmd
        assert "<system_context>" in flat_cmd[-1]
        cid, tc = _load_chat_id(anima_dir, "chat")
        assert cid == "new-rotated-session"
        assert tc == 1

    @pytest.mark.asyncio
    async def test_rotation_pending_on_last_turn(self, executor, anima_dir):
        """When new turn_count reaches MAX → rotation_pending=True."""
        _save_chat_id(anima_dir, "session-abc", "chat", turn_count=_MAX_RESUME_TURNS - 1)

        async def mock_create(*args, **kwargs):
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "session-abc"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            result = await executor.execute(prompt="last turn", trigger="chat")

        assert result.session_rotated is False
        assert result.session_rotation_pending is True
        _, tc = _load_chat_id(anima_dir, "chat")
        assert tc == _MAX_RESUME_TURNS

    @pytest.mark.asyncio
    async def test_no_rotation_before_max(self, executor, anima_dir):
        """Mid-session: neither rotated nor pending."""
        _save_chat_id(anima_dir, "session-abc", "chat", turn_count=3)

        async def mock_create(*args, **kwargs):
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "session-abc"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            result = await executor.execute(prompt="mid session", trigger="chat")

        assert result.session_rotated is False
        assert result.session_rotation_pending is False
        _, tc = _load_chat_id(anima_dir, "chat")
        assert tc == 4


class TestResumePromptContent:
    """Tests for prompt content on resume vs fresh sessions."""

    @pytest.mark.asyncio
    async def test_resume_skips_system_prompt(self, executor, anima_dir):
        """Resume turns inject only time, not <system_context>."""
        _save_chat_id(anima_dir, "prev-session", "chat", turn_count=2)
        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "prev-session"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="follow up", system_prompt="You are Alice", trigger="chat")

        prompt_arg = captured_cmds[0][-1]
        assert "<system_context>" not in prompt_arg
        assert "You are Alice" not in prompt_arg
        assert "follow up" in prompt_arg
        assert "[" in prompt_arg  # time prefix

    @pytest.mark.asyncio
    async def test_fresh_session_includes_system_prompt(self, executor, anima_dir):
        """First turn: full system_prompt in <system_context>."""
        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "new-sess"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="first message", system_prompt="You are Bob", trigger="chat")

        prompt_arg = captured_cmds[0][-1]
        assert "<system_context>" in prompt_arg
        assert "You are Bob" in prompt_arg
        assert "first message" in prompt_arg

    @pytest.mark.asyncio
    async def test_time_prefix_always_present(self, executor, anima_dir):
        """Both resume and fresh sessions include time prefix."""
        captured_cmds: list[list[str]] = []

        async def mock_create(*args, **kwargs):
            captured_cmds.append(list(args))
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stderr = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            events = [
                {"type": "system", "subtype": "init", "session_id": "s1"},
                {"type": "result", "result": "ok"},
            ]
            line_data = _make_ndjson_lines(events)
            line_iter = iter(line_data.split(b"\n"))
            mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter, b""))
            return mock_proc

        with (
            patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
            patch.object(executor, "_ensure_workspace"),
            patch.object(executor, "_write_mcp_config"),
            patch.object(executor, "_write_cursor_rules"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="test", trigger="chat")

        prompt_arg = captured_cmds[0][-1]
        assert "[" in prompt_arg and "]" in prompt_arg


class TestCursorRules:
    """Tests for .cursor/rules static content writing."""

    def test_write_cursor_rules_creates_files(self, executor, anima_dir):
        executor._ensure_workspace()
        (anima_dir / "identity.md").write_text("# Test Identity", encoding="utf-8")
        (anima_dir / "permissions.md").write_text("# Permissions", encoding="utf-8")

        executor._write_cursor_rules()

        rules_dir = executor._workspace / ".cursor" / "rules"
        assert (rules_dir / "identity.md").exists()
        assert (rules_dir / "identity.md").read_text(encoding="utf-8") == "# Test Identity"
        assert (rules_dir / "permissions.md").exists()
        assert (rules_dir / "permissions.md").read_text(encoding="utf-8") == "# Permissions"

    def test_write_cursor_rules_skips_empty(self, executor, anima_dir):
        executor._ensure_workspace()
        (anima_dir / "identity.md").write_text("", encoding="utf-8")

        executor._write_cursor_rules()

        rules_dir = executor._workspace / ".cursor" / "rules"
        assert not (rules_dir / "identity.md").exists()

    def test_write_cursor_rules_skips_unchanged(self, executor, anima_dir):
        executor._ensure_workspace()
        (anima_dir / "identity.md").write_text("# Same Content", encoding="utf-8")

        rules_dir = executor._workspace / ".cursor" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        dest = rules_dir / "identity.md"
        dest.write_text("# Same Content", encoding="utf-8")
        mtime_before = dest.stat().st_mtime

        import time

        time.sleep(0.01)
        executor._write_cursor_rules()

        assert dest.stat().st_mtime == mtime_before

    def test_write_cursor_rules_missing_files_no_error(self, executor, anima_dir):
        executor._ensure_workspace()
        (anima_dir / "identity.md").unlink(missing_ok=True)
        (anima_dir / "permissions.md").unlink(missing_ok=True)

        executor._write_cursor_rules()

        rules_dir = executor._workspace / ".cursor" / "rules"
        assert rules_dir.is_dir()


# ── Mode D resolution test ────────────────────────────────────


class TestModeDResolution:
    def test_cursor_pattern_resolves_to_d(self):
        from core.config.model_mode import DEFAULT_MODEL_MODE_PATTERNS

        assert "cursor/*" in DEFAULT_MODEL_MODE_PATTERNS
        assert DEFAULT_MODEL_MODE_PATTERNS["cursor/*"] == "D"

    def test_resolve_execution_mode_cursor(self):
        from core.config.model_mode import resolve_execution_mode
        from core.config.models import load_config

        try:
            config = load_config()
        except Exception:
            pytest.skip("config not available in test environment")
        mode = resolve_execution_mode(config, "cursor/claude-4-sonnet")
        assert mode.upper() == "D"


# ── Rate-guard wiring ────────────────────────────────────────


class TestCursorErrorMetadata:
    def test_quota_error_records_real_guard_block(self, tmp_path: Path) -> None:
        from core.config.schemas import LlmRateGuardConfig
        from core.execution.rate_guard import LlmRateGuard

        guard_path = tmp_path / "llm_rate_guard.json"
        guard = LlmRateGuard(config=LlmRateGuardConfig(quota_block_seconds=1800), path=guard_path)

        with patch("core.execution.cursor_agent.get_rate_guard", return_value=guard):
            metadata = _cursor_error_metadata("quota exceeded", "cursor/claude-4-sonnet")

        state = json.loads(guard_path.read_text(encoding="utf-8"))
        assert metadata == {"terminal": True, "reason": "quota_exhausted"}
        assert state["anthropic:cursor"]["reason"] == "quota_exhausted"
        assert guard.blocked_remaining("anthropic:cursor") > 1700

    def test_unknown_error_does_not_register_block(self, tmp_path: Path) -> None:
        from core.config.schemas import LlmRateGuardConfig
        from core.execution.rate_guard import LlmRateGuard

        guard_path = tmp_path / "llm_rate_guard.json"
        guard = LlmRateGuard(config=LlmRateGuardConfig(), path=guard_path)

        with patch("core.execution.cursor_agent.get_rate_guard", return_value=guard):
            metadata = _cursor_error_metadata("some random internal thing", "cursor/claude-4-sonnet")

        assert metadata == {"terminal": True, "reason": "unknown"}
        assert not guard_path.exists() or json.loads(guard_path.read_text(encoding="utf-8")) == {}
