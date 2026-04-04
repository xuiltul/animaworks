from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CodexSDKExecutor (Mode C).

All tests use mocks — no Codex CLI binary or API key required.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.base import ExecutionResult
from core.execution.codex_sdk import (
    CodexSDKExecutor,
    _clear_thread_id,
    _codex_item_tool_name,
    _default_home_dir,
    _default_path_env,
    _event_idle_timeout_seconds,
    _extract_item_text,
    _extract_tool_records,
    _get_thread_id,
    _is_desktop_extension_codex,
    _item_to_tool_record,
    _load_thread_id,
    _patch_codex_exec_stream_limit,
    _resolve_codex_model,
    _save_thread_id,
    _should_cli_exec_fallback,
    _should_prefer_cli_exec,
    _stderr_contains_fatal_signal,
    _usage_to_dict,
    clear_codex_thread_ids,
)
from core.prompt.context import ContextTracker

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-codex"
    d.mkdir(parents=True)
    (d / "shortterm" / "chat").mkdir(parents=True)
    (d / "shortterm" / "heartbeat").mkdir(parents=True)
    (d / "identity.md").write_text("# Test Codex Anima", encoding="utf-8")
    (d / "state").mkdir()
    (d / "state" / "current_state.md").write_text("status: idle\n", encoding="utf-8")
    return d


@pytest.fixture
def model_config():
    from core.schemas import ModelConfig

    return ModelConfig(
        model="codex/o4-mini",
        max_tokens=4096,
        max_turns=30,
        credential="openai",
        api_key="test-key-123",
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def executor(model_config, anima_dir):
    return CodexSDKExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


# ── Helper function tests ────────────────────────────────────


class TestHelpers:
    def test_resolve_codex_model_strips_prefix(self):
        assert _resolve_codex_model("codex/o4-mini") == "o4-mini"
        assert _resolve_codex_model("codex/gpt-4.1") == "gpt-4.1"
        assert _resolve_codex_model("openai-codex/gpt-5.3-codex") == "gpt-5.3-codex"

    def test_resolve_codex_model_no_prefix(self):
        assert _resolve_codex_model("o4-mini") == "o4-mini"

    def test_get_thread_id_from_id_attr(self):
        obj = MagicMock()
        obj.id = "thread-abc-123"
        assert _get_thread_id(obj) == "thread-abc-123"

    def test_get_thread_id_from_thread_id_attr(self):
        obj = MagicMock(spec=[])
        obj.thread_id = "tid-xyz"
        assert _get_thread_id(obj) == "tid-xyz"

    def test_get_thread_id_none(self):
        obj = MagicMock(spec=[])
        assert _get_thread_id(obj) is None

    def test_extract_item_text_string_content(self):
        item = MagicMock()
        item.content = "Hello world"
        assert _extract_item_text(item) == "Hello world"

    def test_extract_item_text_list_content(self):
        part = MagicMock()
        part.text = "Hello"
        item = MagicMock()
        item.content = [part, " world"]
        assert _extract_item_text(item) == "Hello world"

    def test_extract_item_text_text_attr(self):
        item = MagicMock(spec=["text"])
        item.text = "Direct text"
        assert _extract_item_text(item) == "Direct text"

    def test_extract_item_text_empty(self):
        item = MagicMock(spec=[])
        assert _extract_item_text(item) == ""

    def test_item_to_tool_record_mcp(self):
        item = MagicMock(spec=["type", "id", "server", "tool", "arguments", "result", "error", "status"])
        item.type = "mcp_tool_call"
        item.id = "tool-1"
        item.server = "aw"
        item.tool = "web_search"
        item.arguments = {"query": "test"}
        item.result = MagicMock(content="search results")
        item.error = None
        item.status = "completed"
        record = _item_to_tool_record(item)
        assert record is not None
        assert record.tool_name == "aw/web_search"
        assert record.tool_id == "tool-1"

    def test_item_to_tool_record_command(self):
        item = MagicMock(spec=["type", "id", "command", "aggregated_output", "exit_code", "status"])
        item.type = "command_execution"
        item.id = "cmd-1"
        item.command = "ls -la"
        item.aggregated_output = "total 42\n..."
        item.exit_code = 0
        item.status = "completed"
        record = _item_to_tool_record(item)
        assert record is not None
        assert record.tool_name == "ls -la"
        assert not record.is_error

    def test_codex_item_tool_name_mcp(self):
        item = MagicMock(spec=["server", "tool"])
        item.server = "aw"
        item.tool = "search_memory"
        assert _codex_item_tool_name(item, "mcp_tool_call") == "aw/search_memory"

    def test_codex_item_tool_name_command(self):
        item = MagicMock(spec=["command"])
        item.command = "git status"
        assert _codex_item_tool_name(item, "command_execution") == "git status"

    def test_extract_tool_records(self):
        msg_item = MagicMock(spec=["type"])
        msg_item.type = "agent_message"
        tool_item = MagicMock(spec=["type", "id", "server", "tool", "arguments", "result", "error", "status"])
        tool_item.type = "mcp_tool_call"
        tool_item.id = "t1"
        tool_item.server = "aw"
        tool_item.tool = "read_file"
        tool_item.arguments = {}
        tool_item.result = MagicMock(content="content")
        tool_item.error = None
        tool_item.status = "completed"
        records = _extract_tool_records([msg_item, tool_item])
        assert len(records) == 1
        assert "read_file" in records[0].tool_name

    def test_usage_to_dict_from_dict(self):
        d = {"input_tokens": 100, "output_tokens": 50}
        assert _usage_to_dict(d) == d

    def test_usage_to_dict_from_object(self):
        obj = MagicMock()
        obj.input_tokens = 200
        obj.output_tokens = 80
        obj.prompt_tokens = None
        obj.completion_tokens = None
        result = _usage_to_dict(obj)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 80

    def test_event_idle_timeout_prefers_background_triggers(self):
        assert _event_idle_timeout_seconds("heartbeat") < _event_idle_timeout_seconds("chat")
        assert _event_idle_timeout_seconds("inbox:sakura") == _event_idle_timeout_seconds("heartbeat")

    def test_stderr_contains_fatal_signal_detects_stream_closed(self):
        assert _stderr_contains_fatal_signal("error: Stream closed")
        assert not _stderr_contains_fatal_signal("warning: tool unavailable")

    def test_should_cli_exec_fallback_detects_fatal_stderr(self):
        exc = RuntimeError("Codex Exec aborted after fatal stderr signal: Reading prompt from stdin...")
        assert _should_cli_exec_fallback(exc)

    def test_is_desktop_extension_codex(self):
        assert _is_desktop_extension_codex(
            r"C:\Users\cmnt\.antigravity\extensions\openai.chatgpt-26.313.41514-win32-x64\bin\windows-x86_64\codex.exe"
        )
        assert not _is_desktop_extension_codex(r"C:\tools\codex.exe")

    def test_should_prefer_cli_exec_honors_force_flag(self, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_CODEX_FORCE_CLI_EXEC", "1")
        assert _should_prefer_cli_exec("task:demo")

    def test_should_prefer_cli_exec_for_windows_background_desktop_bundle(self, monkeypatch):
        monkeypatch.delenv("ANIMAWORKS_CODEX_FORCE_CLI_EXEC", raising=False)
        monkeypatch.setattr("core.execution.codex_sdk.sys.platform", "win32")
        monkeypatch.setattr(
            "core.execution.codex_sdk.get_codex_executable",
            lambda: r"C:\Users\cmnt\.antigravity\extensions\openai.chatgpt-26.313.41514-win32-x64\bin\windows-x86_64\codex.exe",
        )
        assert _should_prefer_cli_exec("inbox")
        assert _should_prefer_cli_exec("task:demo")
        assert not _should_prefer_cli_exec("manual")

    @pytest.mark.asyncio
    async def test_patch_codex_exec_stream_limit_fails_fast_on_fatal_stderr(self):
        from openai_codex_sdk.errors import CodexExecError

        class _FakeStdin:
            def write(self, _data):
                return None

            async def drain(self):
                return None

            def close(self):
                return None

        class _SleepingStdout:
            async def readline(self):
                await asyncio.sleep(3600)
                return b""

        class _FatalStderr:
            def __init__(self):
                self._chunks = [
                    b"Error in hook callback\nerror: Stream closed\n",
                    b"",
                ]

            async def read(self, _size):
                await asyncio.sleep(0)
                return self._chunks.pop(0)

        class _FakeProc:
            def __init__(self):
                self.stdin = _FakeStdin()
                self.stdout = _SleepingStdout()
                self.stderr = _FatalStderr()
                self.returncode = None
                self._wait_event = asyncio.Event()
                self.kill_calls = 0

            def kill(self):
                self.kill_calls += 1
                self.returncode = 1
                self._wait_event.set()

            async def wait(self):
                await self._wait_event.wait()
                return self.returncode

        class _FakeExec:
            executable_path = "codex"

            def _build_command_args(self, _args):
                return []

            def _build_env(self, _args):
                return {}

        proc = _FakeProc()
        exec_ = _FakeExec()
        _patch_codex_exec_stream_limit(exec_)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with pytest.raises(CodexExecError, match="fatal stderr signal"):
                stream = exec_.run(SimpleNamespace(input="hello"))
                await stream.__anext__()

        assert proc.kill_calls == 1


# ── Session persistence tests ────────────────────────────────


class TestSessionPersistence:
    def test_save_and_load_thread_id(self, anima_dir):
        _save_thread_id(anima_dir, "thread-abc", "chat")
        assert _load_thread_id(anima_dir, "chat") == "thread-abc"

    def test_load_thread_id_missing(self, anima_dir):
        assert _load_thread_id(anima_dir, "chat") is None

    def test_clear_thread_id(self, anima_dir):
        _save_thread_id(anima_dir, "thread-xyz", "heartbeat")
        _clear_thread_id(anima_dir, "heartbeat")
        assert _load_thread_id(anima_dir, "heartbeat") is None

    def test_clear_all_thread_ids(self, anima_dir):
        _save_thread_id(anima_dir, "t1", "chat")
        _save_thread_id(anima_dir, "t2", "heartbeat")
        clear_codex_thread_ids(anima_dir)
        assert _load_thread_id(anima_dir, "chat") is None
        assert _load_thread_id(anima_dir, "heartbeat") is None


# ── Executor instantiation tests ─────────────────────────────


class TestExecutorInit:
    def test_supports_streaming(self, executor):
        assert executor.supports_streaming is True

    def test_build_env_includes_api_key(self, executor):
        with patch("core.execution.codex_sdk.PROJECT_DIR", "/fake/project", create=True):
            env = executor._build_env()
        assert env.get("OPENAI_API_KEY") == "test-key-123"
        assert "CODEX_HOME" in env

    def test_default_home_dir_prefers_userprofile_when_home_missing(self):
        with patch.dict("os.environ", {"USERPROFILE": r"C:\Users\Tester"}, clear=True):
            assert _default_home_dir() == r"C:\Users\Tester"

    def test_build_mcp_env(self, executor):
        env = executor._build_mcp_env()
        assert "ANIMAWORKS_ANIMA_DIR" in env
        assert "PYTHONPATH" in env

    def test_default_path_env_prepends_embedded_codex(self):
        if os.name == "nt":
            codex_exe = r"C:\Tools\codex.exe"
            base_path = r"C:\Windows\System32"
        else:
            codex_exe = "/opt/codex/bin/codex"
            base_path = "/usr/bin"
        with (
            patch("core.execution.codex_sdk.get_codex_executable", return_value=codex_exe),
            patch.dict("os.environ", {"PATH": base_path}, clear=True),
        ):
            value = _default_path_env()
        parts = value.split(os.pathsep)
        assert parts[0] == str(Path(codex_exe).resolve().parent)

    def test_default_path_env_includes_launcher_python_dir(self):
        if os.name == "nt":
            py_exe = r"E:\OneDriveBiz\Tools\General\animaworks\.venv\Scripts\python.exe"
            base_path = r"C:\Windows\System32"
        else:
            py_exe = "/home/user/proj/.venv/bin/python3"
            base_path = "/usr/bin"
        with (
            patch("core.execution.codex_sdk.get_codex_executable", return_value=None),
            patch("core.execution.codex_sdk.sys.executable", py_exe),
            patch.dict("os.environ", {"PATH": base_path}, clear=True),
        ):
            value = _default_path_env()
        parts = value.split(os.pathsep)
        assert str(Path(py_exe).resolve().parent) in parts

    def test_create_codex_client_passes_executable_override(self, executor):
        fake_client = MagicMock()
        fake_client._exec = MagicMock()

        with (
            patch("core.execution.codex_sdk.get_codex_executable", return_value=r"C:\Tools\codex.exe"),
            patch("openai_codex_sdk.Codex", return_value=fake_client) as mock_codex,
            patch("core.execution.codex_sdk._patch_codex_exec_stream_limit"),
        ):
            executor._create_codex_client()

        options = mock_codex.call_args.args[0]
        assert options["codexPathOverride"] == r"C:\Tools\codex.exe"


# ── Config writing tests ─────────────────────────────────────


class TestConfigWriting:
    def test_write_codex_config_creates_files(self, executor, anima_dir):
        executor._write_codex_config("Test system prompt")
        codex_home = anima_dir / ".codex_home"
        assert (codex_home / "config.toml").exists()
        assert (codex_home / "instructions.md").exists()
        instructions = (codex_home / "instructions.md").read_text(encoding="utf-8")
        assert instructions == "Test system prompt"

    def test_write_codex_config_toml_content(self, executor, anima_dir):
        executor._write_codex_config("My prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        assert "model_instructions_file" in config_toml
        assert "sandbox_mode" in config_toml
        assert "danger-full-access" in config_toml  # default file_roots=["/"]
        assert 'approval_policy = "never"' in config_toml
        assert "[mcp_servers.aw]" in config_toml

    def test_write_codex_config_restricted_sandbox(self, model_config, anima_dir):
        """Restricted file_roots produces workspace-write with writable_roots."""
        import json
        perms = {"version": 1, "file_roots": [str(anima_dir)], "commands": {"allow_all": True, "allow": [], "deny": []}, "external_tools": {"allow_all": True, "allow": [], "deny": []}, "tool_creation": {"personal": True, "shared": False}}
        (anima_dir / "permissions.json").write_text(json.dumps(perms), encoding="utf-8")
        exc = CodexSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        exc._write_codex_config("prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        assert "workspace-write" in config_toml
        assert "writable_roots" in config_toml

    def test_write_codex_config_includes_model_name(self, executor, anima_dir):
        """config.toml must include model = "o4-mini" (stripped prefix)."""
        executor._write_codex_config("prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        assert 'model = "o4-mini"' in config_toml

    def test_write_codex_config_model_name_gpt41(self, model_config, anima_dir):
        """codex/gpt-4.1 → model = "gpt-4.1" in config.toml."""
        model_config.model = "codex/gpt-4.1"
        exc = CodexSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        exc._write_codex_config("prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        assert 'model = "gpt-4.1"' in config_toml

    def test_toml_escapes_special_characters(self, model_config, anima_dir):
        """Paths with quotes/backslashes are escaped in TOML output."""
        from core.execution.codex_sdk import _escape_toml_string

        assert _escape_toml_string('path/with"quote') == 'path/with\\"quote'
        assert _escape_toml_string("path\\back") == "path\\\\back"

    def test_propagate_auth_copies_when_links_unavailable(self, executor, anima_dir):
        default_codex = anima_dir.parent.parent / "home" / ".codex"
        default_codex.mkdir(parents=True)
        source_auth = default_codex / "auth.json"
        source_auth.write_text('{"token":"abc"}', encoding="utf-8")

        with (
            patch("core.execution.codex_sdk.Path.home", return_value=default_codex.parent),
            patch("pathlib.Path.symlink_to", side_effect=OSError("symlink blocked")),
            patch("core.execution.codex_sdk.os.link", side_effect=OSError("hardlink blocked")),
        ):
            executor._codex_home.mkdir(parents=True, exist_ok=True)
            executor._propagate_auth()

        target_auth = executor._codex_home / "auth.json"
        assert target_auth.exists()
        assert not target_auth.is_symlink()
        assert target_auth.read_text(encoding="utf-8") == '{"token":"abc"}'


# ── Blocking execution tests ─────────────────────────────────


class TestBlockingExecution:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self, executor, anima_dir):
        mock_turn = MagicMock()
        mock_turn.final_response = "Hello from Codex!"
        mock_turn.items = []
        mock_turn.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "thread-001"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            result = await executor.execute(
                prompt="Hello",
                system_prompt="You are a test assistant",
            )

        assert isinstance(result, ExecutionResult)
        assert result.text == "Hello from Codex!"
        assert _load_thread_id(anima_dir, "chat") == "thread-001"

    @pytest.mark.asyncio
    async def test_execute_saves_thread_id(self, executor, anima_dir):
        mock_turn = MagicMock()
        mock_turn.final_response = "OK"
        mock_turn.items = []
        mock_turn.usage = None

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "tid-saved"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            await executor.execute(prompt="test")

        assert _load_thread_id(anima_dir, "chat") == "tid-saved"

    @pytest.mark.asyncio
    async def test_execute_heartbeat_trigger(self, executor, anima_dir):
        mock_turn = MagicMock()
        mock_turn.final_response = "Heartbeat response"
        mock_turn.items = []
        mock_turn.usage = None

        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(return_value=mock_turn)
        mock_thread.id = "tid-hb"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with (
            patch("core.execution.codex_sdk._should_prefer_cli_exec", return_value=False),
            patch.object(executor, "_create_codex_client", return_value=mock_codex),
        ):
            result = await executor.execute(
                prompt="heartbeat check",
                trigger="heartbeat",
            )

        assert result.text == "Heartbeat response"
        assert _load_thread_id(anima_dir, "heartbeat") == "tid-hb"
        assert _load_thread_id(anima_dir, "chat") is None

    @pytest.mark.asyncio
    async def test_execute_interrupted_before_run(self, model_config, anima_dir):
        interrupt = asyncio.Event()
        interrupt.set()
        exc = CodexSDKExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            interrupt_event=interrupt,
        )
        result = await exc.execute(prompt="test", system_prompt="sys")
        assert "interrupted" in result.text.lower()

    @pytest.mark.asyncio
    async def test_execute_error_returns_error_result(self, executor):
        mock_codex = MagicMock()
        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(side_effect=RuntimeError("CLI crashed"))
        mock_thread.id = None
        mock_codex.start_thread.return_value = mock_thread

        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            result = await executor.execute(prompt="test")

        assert "[Codex SDK Error:" in result.text

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_cli_exec_on_fatal_sdk_error(self, executor):
        mock_codex = MagicMock()
        mock_thread = MagicMock()
        mock_thread.run = AsyncMock(side_effect=RuntimeError("fatal stderr signal: Reading prompt from stdin..."))
        mock_thread.id = None
        mock_codex.start_thread.return_value = mock_thread
        fallback = ExecutionResult(text="fallback ok")

        with (
            patch.object(executor, "_create_codex_client", return_value=mock_codex),
            patch.object(executor, "_execute_via_cli_exec", AsyncMock(return_value=fallback)) as mock_fallback,
        ):
            result = await executor.execute(prompt="test", system_prompt="sys")

        assert result.text == "fallback ok"
        mock_fallback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_prefers_cli_exec_for_background_trigger(self, executor):
        fallback = ExecutionResult(text="cli preferred")

        with (
            patch("core.execution.codex_sdk._should_prefer_cli_exec", return_value=True),
            patch.object(executor, "_execute_via_cli_exec", AsyncMock(return_value=fallback)) as mock_fallback,
            patch.object(executor, "_create_codex_client") as mock_client,
        ):
            result = await executor.execute(prompt="test", system_prompt="sys", trigger="task:demo")

        assert result.text == "cli preferred"
        mock_fallback.assert_awaited_once()
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_retry_on_resume_failure(self, executor, anima_dir):
        _save_thread_id(anima_dir, "stale-thread", "chat")

        mock_turn = MagicMock()
        mock_turn.final_response = "After retry"
        mock_turn.items = []
        mock_turn.usage = None

        fresh_thread = MagicMock()
        fresh_thread.run = AsyncMock(return_value=mock_turn)
        fresh_thread.id = "new-thread"

        stale_thread = MagicMock()
        stale_thread.run = AsyncMock(side_effect=RuntimeError("Resume failed"))

        mock_codex = MagicMock()
        mock_codex.resume_thread.return_value = stale_thread
        mock_codex.start_thread.return_value = fresh_thread

        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            result = await executor.execute(prompt="retry test")

        assert result.text == "After retry"


# ── Streaming execution tests ────────────────────────────────


class TestStreamingExecution:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self, executor, anima_dir):
        msg_item = MagicMock(spec=["type", "id", "text"])
        msg_item.type = "agent_message"
        msg_item.id = "msg-1"
        msg_item.text = "Streamed text"

        msg_event = MagicMock()
        msg_event.type = "item.completed"
        msg_event.item = msg_item

        usage_obj = MagicMock()
        usage_obj.input_tokens = 100
        usage_obj.output_tokens = 50

        done_event = MagicMock()
        done_event.type = "turn.completed"
        done_event.usage = usage_obj

        async def fake_events():
            yield msg_event
            yield done_event

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()

        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "stream-thread"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="Hello",
                tracker=tracker,
            ):
                events.append(ev)

        types = [e["type"] for e in events]
        assert "text_delta" in types
        assert "done" in types
        done_ev = next(e for e in events if e["type"] == "done")
        assert "Streamed text" in done_ev["full_text"]
        assert tracker.usage_ratio > 0.0, "tracker must be updated from usage"

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_cli_exec_on_fatal_sdk_error(self, executor):
        async def broken_events():
            raise RuntimeError("fatal stderr signal: Reading prompt from stdin...")
            yield  # pragma: no cover

        mock_streamed = MagicMock()
        mock_streamed.events = broken_events()

        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "broken-thread"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        async def fallback_events(*_args, **_kwargs):
            yield {"type": "text_delta", "text": "fallback"}
            yield {
                "type": "done",
                "full_text": "fallback",
                "result_message": None,
                "replied_to_from_transcript": set(),
                "tool_call_records": [],
                "usage": {},
            }

        with (
            patch.object(executor, "_create_codex_client", return_value=mock_codex),
            patch.object(executor, "_execute_streaming_via_cli_exec", side_effect=fallback_events),
        ):
            events = []
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="Hello",
                tracker=tracker,
            ):
                events.append(ev)

        assert [e["type"] for e in events] == ["text_delta", "done"]

    @pytest.mark.asyncio
    async def test_stream_prefers_cli_exec_for_background_trigger(self, executor):
        async def cli_events(*_args, **_kwargs):
            yield {"type": "text_delta", "text": "cli stream"}
            yield {"type": "done", "full_text": "cli stream", "result_message": None}

        tracker = ContextTracker(model=executor._model_config.model)
        events = []

        with (
            patch("core.execution.codex_sdk._should_prefer_cli_exec", return_value=True),
            patch.object(executor, "_execute_streaming_via_cli_exec", side_effect=cli_events) as mock_fallback,
            patch.object(executor, "_create_codex_client") as mock_client,
        ):
            async for ev in executor.execute_streaming(
                system_prompt="sys",
                prompt="Hello",
                tracker=tracker,
                trigger="inbox",
            ):
                events.append(ev)

        assert [e["type"] for e in events] == ["text_delta", "done"]
        mock_fallback.assert_called_once()
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_tool_events(self, executor, anima_dir):
        tool_item = MagicMock(spec=["type", "id", "server", "tool", "arguments", "result", "error", "status"])
        tool_item.type = "mcp_tool_call"
        tool_item.id = "ws-1"
        tool_item.server = "aw"
        tool_item.tool = "web_search"
        tool_item.arguments = {"query": "test"}
        tool_item.result = MagicMock(content="results")
        tool_item.error = None
        tool_item.status = "completed"

        tool_event = MagicMock()
        tool_event.type = "item.completed"
        tool_event.item = tool_item

        done_event = MagicMock()
        done_event.type = "turn.completed"
        done_event.usage = None

        async def fake_events():
            yield tool_event
            yield done_event

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()

        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "tool-thread"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="search",
                tracker=tracker,
            ):
                events.append(ev)

        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert "web_search" in tool_start["tool_name"]

    @pytest.mark.asyncio
    async def test_stream_interrupted_mid_stream(self, model_config, anima_dir):
        interrupt = asyncio.Event()

        msg_item = MagicMock(spec=["type", "id", "text"])
        msg_item.type = "agent_message"
        msg_item.id = "msg-int"
        msg_item.text = "Before interrupt"

        msg_event = MagicMock()
        msg_event.type = "item.completed"
        msg_event.item = msg_item

        second_event = MagicMock()
        second_event.type = "item.completed"
        second_event.item = msg_item

        async def fake_events():
            yield msg_event
            interrupt.set()
            yield second_event

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()

        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "int-thread"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        exc = CodexSDKExecutor(
            model_config=model_config,
            anima_dir=anima_dir,
            interrupt_event=interrupt,
        )

        events = []
        with patch.object(exc, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in exc.execute_streaming(
                system_prompt="test",
                prompt="Hello",
                tracker=tracker,
            ):
                events.append(ev)

        texts = [e.get("text", "") for e in events if e.get("type") == "text_delta"]
        combined = " ".join(texts)
        assert "interrupted" in combined.lower()


# ── Progressive streaming tests ───────────────────────────────


class TestProgressiveStreaming:
    """Tests for item.started / item.updated progressive text deltas."""

    @pytest.mark.asyncio
    async def test_progressive_text_deltas(self, executor, anima_dir):
        """item.started + item.updated should yield incremental text deltas."""

        def _make_msg(item_id, text):
            item = MagicMock(spec=["type", "id", "text"])
            item.type = "agent_message"
            item.id = item_id
            item.text = text
            return item

        started_event = MagicMock()
        started_event.type = "item.started"
        started_event.item = _make_msg("msg-1", "He")

        updated1 = MagicMock()
        updated1.type = "item.updated"
        updated1.item = _make_msg("msg-1", "Hello ")

        updated2 = MagicMock()
        updated2.type = "item.updated"
        updated2.item = _make_msg("msg-1", "Hello world!")

        completed = MagicMock()
        completed.type = "item.completed"
        completed.item = _make_msg("msg-1", "Hello world!")

        done_event = MagicMock()
        done_event.type = "turn.completed"
        done_event.usage = MagicMock(input_tokens=50, output_tokens=20)

        async def fake_events():
            yield started_event
            yield updated1
            yield updated2
            yield completed
            yield done_event

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()
        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "prog-thread"
        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="Hello",
                tracker=tracker,
            ):
                events.append(ev)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) >= 3, f"Expected >= 3 text_delta events, got {len(text_deltas)}"

        # Reconstruct full text from deltas
        reconstructed = "".join(e["text"] for e in text_deltas)
        assert reconstructed == "Hello world!"

    @pytest.mark.asyncio
    async def test_reasoning_events_yield_thinking_delta(self, executor, anima_dir):
        """item.started/updated with reasoning type should yield thinking_delta."""

        def _make_reasoning(item_id, text):
            item = MagicMock(spec=["type", "id", "text"])
            item.type = "reasoning"
            item.id = item_id
            item.text = text
            return item

        started = MagicMock()
        started.type = "item.started"
        started.item = _make_reasoning("r-1", "Let me think")

        updated = MagicMock()
        updated.type = "item.updated"
        updated.item = _make_reasoning("r-1", "Let me think about this problem")

        completed = MagicMock()
        completed.type = "item.completed"
        completed.item = _make_reasoning("r-1", "Let me think about this problem")

        msg = MagicMock(spec=["type", "id", "text"])
        msg.type = "agent_message"
        msg.id = "msg-1"
        msg.text = "The answer is 42."
        msg_completed = MagicMock()
        msg_completed.type = "item.completed"
        msg_completed.item = msg

        done = MagicMock()
        done.type = "turn.completed"
        done.usage = None

        async def fake_events():
            yield started
            yield updated
            yield completed
            yield msg_completed
            yield done

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()
        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "think-thread"
        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="What is the answer?",
                tracker=tracker,
            ):
                events.append(ev)

        thinking = [e for e in events if e["type"] == "thinking_delta"]
        assert len(thinking) >= 2

    @pytest.mark.asyncio
    async def test_tool_started_before_completed(self, executor, anima_dir):
        """item.started for command_execution should emit tool_start early."""
        tool_item_partial = MagicMock(spec=["type", "id", "command", "aggregated_output", "exit_code", "status"])
        tool_item_partial.type = "command_execution"
        tool_item_partial.id = "cmd-1"
        tool_item_partial.command = "npm test"
        tool_item_partial.aggregated_output = ""
        tool_item_partial.exit_code = None
        tool_item_partial.status = "in_progress"

        started = MagicMock()
        started.type = "item.started"
        started.item = tool_item_partial

        tool_item_done = MagicMock(spec=["type", "id", "command", "aggregated_output", "exit_code", "status"])
        tool_item_done.type = "command_execution"
        tool_item_done.id = "cmd-1"
        tool_item_done.command = "npm test"
        tool_item_done.aggregated_output = "all passed"
        tool_item_done.exit_code = 0
        tool_item_done.status = "completed"

        completed = MagicMock()
        completed.type = "item.completed"
        completed.item = tool_item_done

        done = MagicMock()
        done.type = "turn.completed"
        done.usage = None

        async def fake_events():
            yield started
            yield completed
            yield done

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()
        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "tool-thread"
        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="run tests",
                tracker=tracker,
            ):
                events.append(ev)

        types = [e["type"] for e in events]
        # tool_start should appear BEFORE tool_end
        assert "tool_start" in types
        assert "tool_end" in types
        start_idx = types.index("tool_start")
        end_idx = types.index("tool_end")
        assert start_idx < end_idx

    @pytest.mark.asyncio
    async def test_turn_failed_yields_error(self, executor, anima_dir):
        """turn.failed event should yield an error event."""
        failed_event = MagicMock()
        failed_event.type = "turn.failed"
        err = MagicMock()
        err.message = "Rate limit exceeded"
        failed_event.error = err

        async def fake_events():
            yield failed_event

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()
        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "fail-thread"
        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="fail",
                tracker=tracker,
            ):
                events.append(ev)

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1
        assert "Rate limit" in error_events[0]["message"]

    @pytest.mark.asyncio
    async def test_no_duplicate_text_on_completed(self, executor, anima_dir):
        """item.completed should not re-emit text already sent via item.updated."""

        def _make_msg(item_id, text):
            item = MagicMock(spec=["type", "id", "text"])
            item.type = "agent_message"
            item.id = item_id
            item.text = text
            return item

        updated = MagicMock()
        updated.type = "item.updated"
        updated.item = _make_msg("msg-1", "Complete text here")

        completed = MagicMock()
        completed.type = "item.completed"
        completed.item = _make_msg("msg-1", "Complete text here")

        done = MagicMock()
        done.type = "turn.completed"
        done.usage = None

        async def fake_events():
            yield updated
            yield completed
            yield done

        mock_streamed = MagicMock()
        mock_streamed.events = fake_events()
        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "dup-thread"
        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        events = []
        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
            tracker = ContextTracker(model="codex/o4-mini")
            async for ev in executor.execute_streaming(
                system_prompt="test",
                prompt="Hello",
                tracker=tracker,
            ):
                events.append(ev)

        text_deltas = [e for e in events if e["type"] == "text_delta"]
        full_text = "".join(e["text"] for e in text_deltas)
        assert full_text == "Complete text here"

    @pytest.mark.asyncio
    async def test_stream_idle_timeout_raises_stream_disconnected(self, executor):
        class NeverEvents:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(3600)
                raise StopAsyncIteration

        mock_streamed = MagicMock()
        mock_streamed.events = NeverEvents()

        mock_thread = MagicMock()
        mock_thread.run_streamed = AsyncMock(return_value=mock_streamed)
        mock_thread.id = "idle-thread"

        mock_codex = MagicMock()
        mock_codex.start_thread.return_value = mock_thread

        with (
            patch("core.execution.codex_sdk._should_prefer_cli_exec", return_value=False),
            patch.object(executor, "_create_codex_client", return_value=mock_codex),
            patch("core.execution.codex_sdk._BACKGROUND_EVENT_IDLE_TIMEOUT_SEC", 0.01),
        ):
            tracker = ContextTracker(model="codex/o4-mini")
            with pytest.raises(Exception) as exc_info:
                async for _ in executor.execute_streaming(
                    system_prompt="test",
                    prompt="Hello",
                    tracker=tracker,
                    trigger="heartbeat",
                ):
                    pass

        assert "idle timeout" in str(exc_info.value)


# ── Mode resolution tests ────────────────────────────────────


class TestModeResolution:
    def test_resolve_execution_mode_codex_pattern(self):
        from core.config.models import resolve_execution_mode

        with patch("core.config.models.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.model_modes = {}
            mock_load.return_value = mock_config
            mode = resolve_execution_mode(mock_config, "codex/o4-mini")
        assert mode == "C"

    def test_resolve_execution_mode_codex_gpt41(self):
        from core.config.models import resolve_execution_mode

        mock_config = MagicMock()
        mock_config.model_modes = {}
        mode = resolve_execution_mode(mock_config, "codex/gpt-4.1")
        assert mode == "C"

    def test_normalise_mode_c(self):
        from core.config.models import _normalise_mode

        assert _normalise_mode("C") == "C"
        assert _normalise_mode("c") == "C"

    def test_openai_model_still_routes_to_a(self):
        from core.config.models import resolve_execution_mode

        mock_config = MagicMock()
        mock_config.model_modes = {}
        mode = resolve_execution_mode(mock_config, "openai/gpt-4.1")
        assert mode == "A"
