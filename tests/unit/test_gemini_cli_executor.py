from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GeminiCLIExecutor (Mode G).

All tests use mocks — no gemini CLI binary required.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.execution.base import ExecutionResult
from core.execution.gemini_cli import (
    GeminiCLIExecutor,
    _find_gemini_binary,
    _resolve_gemini_model,
    is_gemini_cli_available,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-gemini"
    d.mkdir(parents=True)
    (d / "shortterm" / "chat").mkdir(parents=True)
    (d / "shortterm" / "heartbeat").mkdir(parents=True)
    (d / "identity.md").write_text("# Test Gemini Anima", encoding="utf-8")
    (d / "state").mkdir()
    (d / "state" / "current_state.md").write_text("status: idle\n", encoding="utf-8")
    return d


@pytest.fixture
def model_config():
    from core.schemas import ModelConfig

    return ModelConfig(
        model="gemini/2.5-pro",
        max_tokens=4096,
        max_turns=30,
        credential="gemini",
        context_threshold=0.50,
        max_chains=2,
    )


@pytest.fixture
def executor(model_config, anima_dir):
    return GeminiCLIExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


# ── Helper ───────────────────────────────────────────────────


def _make_ndjson_lines(events: list[dict]) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def _mock_proc(stdout_data: bytes, returncode: int = 0, stderr_data: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.wait = AsyncMock()
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr_data)
    proc.stdout = AsyncMock()
    lines = iter(stdout_data.split(b"\n"))
    proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))
    return proc


# ── Binary discovery ─────────────────────────────────────────


class TestBinaryDiscovery:
    def test_find_binary_returns_match(self):
        with patch("shutil.which", side_effect=lambda n: "/usr/bin/gemini" if n == "gemini" else None):
            assert _find_gemini_binary() == "/usr/bin/gemini"

    def test_find_binary_returns_none_when_missing(self):
        with patch("shutil.which", return_value=None):
            assert _find_gemini_binary() is None

    def test_is_available_true(self):
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"):
            assert is_gemini_cli_available() is True

    def test_is_available_false(self):
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value=None):
            assert is_gemini_cli_available() is False


# ── Model resolution ─────────────────────────────────────────


class TestModelResolution:
    def test_strip_prefix_and_add_gemini_dash(self):
        assert _resolve_gemini_model("gemini/2.5-pro") == "gemini-2.5-pro"

    def test_strip_prefix_already_has_gemini_dash(self):
        assert _resolve_gemini_model("gemini/gemini-2.5-pro") == "gemini-2.5-pro"

    def test_no_prefix(self):
        assert _resolve_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"


# ── Workspace ────────────────────────────────────────────────


class TestWorkspace:
    def test_ensure_workspace_creates_dirs(self, executor):
        executor._ensure_workspace()
        assert executor._workspace.is_dir()
        assert (executor._workspace / ".gemini").is_dir()

    def test_workspace_location(self, executor, anima_dir):
        assert executor._workspace == anima_dir / ".gemini-workspace"

    def test_write_settings(self, executor):
        executor._ensure_workspace()
        executor._write_settings()
        settings_path = executor._workspace / ".gemini" / "settings.json"
        assert settings_path.exists()
        config = json.loads(settings_path.read_text())
        assert "mcpServers" in config
        assert "aw" in config["mcpServers"]
        aw_conf = config["mcpServers"]["aw"]
        assert "-m" in aw_conf["args"]
        assert "core.mcp.server" in aw_conf["args"]
        assert "ANIMAWORKS_ANIMA_DIR" in aw_conf["env"]


# ── System prompt ────────────────────────────────────────────


class TestSystemPrompt:
    def test_write_system_prompt_creates_file(self, executor):
        path = executor._write_system_prompt("You are a helpful assistant.")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "You are a helpful assistant."
        executor._cleanup_prompt_files()
        assert not path.exists()

    def test_cleanup_removes_all(self, executor):
        p1 = executor._write_system_prompt("prompt 1")
        p2 = executor._write_system_prompt("prompt 2")
        assert p1.exists() and p2.exists()
        executor._cleanup_prompt_files()
        assert not p1.exists() and not p2.exists()
        assert len(executor._prompt_files) == 0


# ── Build command ────────────────────────────────────────────


class TestBuildCommand:
    def test_command_structure(self, executor):
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"):
            cmd = executor._build_command("hello world")
        assert cmd[0] == "/usr/bin/gemini"
        assert "-p" in cmd
        assert "hello world" in cmd
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"
        assert "--approval-mode" in cmd
        am_idx = cmd.index("--approval-mode")
        assert cmd[am_idx + 1] == "yolo"
        assert "-m" in cmd
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == "gemini-2.5-pro"

    def test_empty_when_no_binary(self, executor):
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value=None):
            assert executor._build_command("test") == []


# ── Build env ────────────────────────────────────────────────


class TestBuildEnv:
    def test_includes_cli_home(self, executor):
        env = executor._build_env()
        assert "GEMINI_CLI_HOME" in env
        assert str(executor._workspace) in env["GEMINI_CLI_HOME"]

    def test_system_prompt_path_injected(self, executor):
        p = Path("/tmp/test_sys.md")
        env = executor._build_env(system_prompt_path=p)
        assert env["GEMINI_SYSTEM_MD"] == str(p)

    def test_api_key_passthrough(self, executor):
        executor._model_config.api_key = "test-gemini-key-123"
        env = executor._build_env()
        assert env.get("GEMINI_API_KEY") == "test-gemini-key-123"

    def test_no_api_key_when_empty(self, executor):
        executor._model_config.api_key = ""
        executor._model_config.api_key_env = ""
        with patch.dict("os.environ", {}, clear=False):
            executor._build_env()  # should not raise


# ── NDJSON parsing ───────────────────────────────────────────


class TestNDJSONParsing:
    def test_parse_valid(self, executor):
        event = executor._parse_ndjson_event('{"type":"message","role":"assistant","content":"hi"}')
        assert event["type"] == "message"

    def test_parse_empty(self, executor):
        assert executor._parse_ndjson_event("") is None
        assert executor._parse_ndjson_event("  ") is None

    def test_parse_invalid(self, executor):
        assert executor._parse_ndjson_event("not json at all") is None

    def test_parse_whitespace_stripped(self, executor):
        event = executor._parse_ndjson_event('  {"type":"result"}  \n')
        assert event == {"type": "result"}


# ── Tool record extraction ───────────────────────────────────


class TestToolRecordExtraction:
    def test_basic_tool_record(self, executor):
        evt = {"tool_name": "web_search", "tool_id": "t1", "parameters": {"query": "test"}}
        record = executor._extract_tool_record(evt)
        assert record.tool_name == "web_search"
        assert record.tool_id == "t1"
        assert "test" in record.input_summary

    def test_mcp_aw_prefix_stripped(self, executor):
        evt = {"tool_name": "mcp_aw_search_memory", "tool_id": "t2", "parameters": {}}
        record = executor._extract_tool_record(evt)
        assert record.tool_name == "search_memory"

    def test_with_result_event(self, executor):
        tool_evt = {"tool_name": "read_file", "tool_id": "t3", "parameters": {"path": "/tmp/x"}}
        result_evt = {"tool_id": "t3", "status": "success", "output": "file contents"}
        record = executor._extract_tool_record(tool_evt, result_evt)
        assert record.result_summary == "file contents"
        assert record.is_error is False

    def test_error_result(self, executor):
        tool_evt = {"tool_name": "bash", "tool_id": "t4", "parameters": {}}
        result_evt = {"tool_id": "t4", "status": "error", "error": {"type": "PERM", "message": "denied"}}
        record = executor._extract_tool_record(tool_evt, result_evt)
        assert record.is_error is True
        assert "denied" in record.result_summary


# ── Stats parsing ────────────────────────────────────────────


class TestStatsParsing:
    def test_parse_stats(self, executor):
        stats = {"input_tokens": 100, "output_tokens": 50, "cached": 20, "total_tokens": 170}
        usage = executor._parse_stats(stats)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_parse_none_stats(self, executor):
        assert executor._parse_stats(None) is None

    def test_parse_empty_stats(self, executor):
        usage = executor._parse_stats({})
        assert usage is not None
        assert usage.input_tokens == 0


# ── Execute tests ────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_not_installed(self, executor):
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value=None):
            result = await executor.execute(prompt="hello")
        assert "gemini" in result.text.lower() or "インストール" in result.text

    @pytest.mark.asyncio
    async def test_interrupted(self, executor):
        executor._interrupt_event = asyncio.Event()
        executor._interrupt_event.set()
        result = await executor.execute(prompt="hello")
        assert "interrupted" in result.text.lower()

    @pytest.mark.asyncio
    async def test_successful_execution(self, executor):
        events = [
            {"type": "init", "session_id": "s1", "model": "gemini-2.5-pro", "timestamp": "2026-03-20T00:00:00Z"},
            {
                "type": "message",
                "role": "assistant",
                "content": "Hello ",
                "delta": True,
                "timestamp": "2026-03-20T00:00:01Z",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "world!",
                "delta": True,
                "timestamp": "2026-03-20T00:00:02Z",
            },
            {
                "type": "result",
                "status": "success",
                "stats": {"input_tokens": 50, "output_tokens": 10, "cached": 0, "total_tokens": 60},
                "timestamp": "2026-03-20T00:00:03Z",
            },
        ]
        proc = _mock_proc(_make_ndjson_lines(events))

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute(prompt="hi", system_prompt="You are helpful")

        assert result.text == "Hello world!"
        assert isinstance(result, ExecutionResult)
        assert result.usage is not None
        assert result.usage.input_tokens == 50

    @pytest.mark.asyncio
    async def test_tool_use_and_result(self, executor):
        events = [
            {"type": "init", "session_id": "s2", "model": "gemini-2.5-pro", "timestamp": "2026-03-20T00:00:00Z"},
            {
                "type": "tool_use",
                "tool_name": "mcp_aw_search_memory",
                "tool_id": "t1",
                "parameters": {"query": "hello"},
                "timestamp": "2026-03-20T00:00:01Z",
            },
            {
                "type": "tool_result",
                "tool_id": "t1",
                "status": "success",
                "output": "found: greeting memory",
                "timestamp": "2026-03-20T00:00:02Z",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "I found it!",
                "delta": True,
                "timestamp": "2026-03-20T00:00:03Z",
            },
            {"type": "result", "status": "success", "timestamp": "2026-03-20T00:00:04Z"},
        ]
        proc = _mock_proc(_make_ndjson_lines(events))

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute(prompt="search")

        assert len(result.tool_call_records) == 1
        assert result.tool_call_records[0].tool_name == "search_memory"
        assert result.tool_call_records[0].result_summary == "found: greeting memory"
        assert result.text == "I found it!"

    @pytest.mark.asyncio
    async def test_auth_error(self, executor):
        proc = _mock_proc(b"", returncode=1, stderr_data=b"Error: unauthenticated, run gemini auth login")

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute(prompt="hello")

        assert "login" in result.text.lower() or "認証" in result.text

    @pytest.mark.asyncio
    async def test_nonzero_exit_generic_error(self, executor):
        proc = _mock_proc(b"", returncode=1, stderr_data=b"Something went wrong")

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute(prompt="hello")

        assert "error" in result.text.lower()

    @pytest.mark.asyncio
    async def test_result_error_event(self, executor):
        events = [
            {
                "type": "result",
                "status": "error",
                "error": {"type": "MaxTurns", "message": "max turns exceeded"},
                "timestamp": "2026-03-20T00:00:00Z",
            },
        ]
        proc = _mock_proc(_make_ndjson_lines(events))

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute(prompt="hello")

        assert "max turns" in result.text.lower()

    @pytest.mark.asyncio
    async def test_system_prompt_via_env(self, executor):
        """Verify GEMINI_SYSTEM_MD is set when system_prompt is provided."""
        captured_env = {}

        async def mock_create(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return _mock_proc(
                _make_ndjson_lines([{"type": "result", "status": "success", "timestamp": "2026-03-20T00:00:00Z"}])
            )

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", side_effect=mock_create),
        ):
            await executor.execute(prompt="test", system_prompt="You are a test assistant")

        assert "GEMINI_SYSTEM_MD" in captured_env


# ── Streaming tests ──────────────────────────────────────────


class TestExecuteStreaming:
    @pytest.mark.asyncio
    async def test_streaming_text_deltas(self, executor):
        from core.prompt.context import ContextTracker

        events = [
            {"type": "init", "session_id": "s1", "model": "gemini-2.5-pro", "timestamp": "2026-03-20T00:00:00Z"},
            {
                "type": "message",
                "role": "assistant",
                "content": "Hello ",
                "delta": True,
                "timestamp": "2026-03-20T00:00:01Z",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "world!",
                "delta": True,
                "timestamp": "2026-03-20T00:00:02Z",
            },
            {
                "type": "result",
                "status": "success",
                "stats": {"input_tokens": 30, "output_tokens": 5},
                "timestamp": "2026-03-20T00:00:03Z",
            },
        ]
        proc = _mock_proc(_make_ndjson_lines(events))
        tracker = ContextTracker(model="gemini-2.5-pro", threshold=0.5)

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            collected = []
            async for evt in executor.execute_streaming(system_prompt="You are helpful", prompt="hi", tracker=tracker):
                collected.append(evt)

        text_deltas = [e for e in collected if e["type"] == "text_delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0]["text"] == "Hello "
        assert text_deltas[1]["text"] == "world!"

        done_events = [e for e in collected if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["full_text"] == "Hello world!"
        assert "result_message" in done_events[0]

    @pytest.mark.asyncio
    async def test_streaming_tool_events(self, executor):
        from core.prompt.context import ContextTracker

        events = [
            {
                "type": "tool_use",
                "tool_name": "mcp_aw_create_skill",
                "tool_id": "t1",
                "parameters": {"skill_name": "test"},
                "timestamp": "2026-03-20T00:00:01Z",
            },
            {
                "type": "tool_result",
                "tool_id": "t1",
                "status": "success",
                "output": "skill content",
                "timestamp": "2026-03-20T00:00:02Z",
            },
            {"type": "result", "status": "success", "timestamp": "2026-03-20T00:00:03Z"},
        ]
        proc = _mock_proc(_make_ndjson_lines(events))
        tracker = ContextTracker(model="gemini-2.5-pro", threshold=0.5)

        with (
            patch("core.execution.gemini_cli._find_gemini_binary", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            collected = []
            async for evt in executor.execute_streaming(system_prompt="", prompt="test", tracker=tracker):
                collected.append(evt)

        tool_starts = [e for e in collected if e["type"] == "tool_start"]
        tool_ends = [e for e in collected if e["type"] == "tool_end"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "create_skill"
        assert len(tool_ends) == 1
        assert tool_ends[0]["result"] == "skill content"

    @pytest.mark.asyncio
    async def test_streaming_not_installed(self, executor):
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="gemini-2.5-pro", threshold=0.5)
        with patch("core.execution.gemini_cli._find_gemini_binary", return_value=None):
            collected = []
            async for evt in executor.execute_streaming(system_prompt="", prompt="hello", tracker=tracker):
                collected.append(evt)

        assert any(
            "gemini" in e.get("text", "").lower() or "インストール" in e.get("text", "")
            for e in collected
            if e["type"] == "text_delta"
        )
        assert any(e["type"] == "done" for e in collected)


# ── Mode G resolution ────────────────────────────────────────


class TestModeGResolution:
    def test_gemini_pattern_resolves_to_g(self):
        from core.config.model_mode import DEFAULT_MODEL_MODE_PATTERNS

        assert "gemini/*" in DEFAULT_MODEL_MODE_PATTERNS
        assert DEFAULT_MODEL_MODE_PATTERNS["gemini/*"] == "G"

    def test_resolve_execution_mode_gemini(self):
        from core.config.model_mode import resolve_execution_mode
        from core.config.models import load_config

        try:
            config = load_config()
        except Exception:
            pytest.skip("config not available in test environment")
        mode = resolve_execution_mode(config, "gemini/2.5-pro")
        assert mode.upper() == "G"

    def test_normalise_mode_g(self):
        from core.config.model_mode import _normalise_mode

        assert _normalise_mode("G") == "G"
        assert _normalise_mode("g") == "G"


# ── MCP_MODES includes g ─────────────────────────────────────


class TestMCPModes:
    def test_org_context_mcp_modes(self):
        from core.prompt.org_context import _MCP_MODES

        assert "g" in _MCP_MODES
