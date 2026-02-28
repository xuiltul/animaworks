from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CodexSDKExecutor (Mode C).

All tests use mocks — no Codex CLI binary or API key required.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution.codex_sdk import (
    CodexSDKExecutor,
    _escape_toml_string,
    _extract_item_text,
    _extract_tool_records,
    _get_thread_id,
    _item_to_tool_record,
    _resolve_codex_model,
    _save_thread_id,
    _load_thread_id,
    _clear_thread_id,
    _usage_to_dict,
    clear_codex_thread_ids,
)
from core.execution.base import ExecutionResult, ToolCallRecord
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
    (d / "state" / "current_task.md").write_text("status: idle\n", encoding="utf-8")
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

    def test_item_to_tool_record(self):
        item = MagicMock()
        item.name = "web_search"
        item.id = "tool-1"
        item.input = {"query": "test"}
        item.output = "search results"
        record = _item_to_tool_record(item)
        assert record is not None
        assert record.tool_name == "web_search"
        assert record.tool_id == "tool-1"

    def test_extract_tool_records(self):
        msg_item = MagicMock()
        msg_item.type = "message"
        tool_item = MagicMock()
        tool_item.type = "tool_use"
        tool_item.name = "read_file"
        tool_item.id = "t1"
        tool_item.input = {}
        tool_item.output = "content"
        records = _extract_tool_records([msg_item, tool_item])
        assert len(records) == 1
        assert records[0].tool_name == "read_file"

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

    def test_build_mcp_env(self, executor):
        env = executor._build_mcp_env()
        assert "ANIMAWORKS_ANIMA_DIR" in env
        assert "PYTHONPATH" in env


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
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(
            encoding="utf-8"
        )
        assert "model_instructions_file" in config_toml
        assert "sandbox_mode" in config_toml
        assert "workspace-write" in config_toml
        assert 'approval_policy = "never"' in config_toml
        assert "[mcp_servers.aw]" in config_toml

    def test_write_codex_config_includes_model_name(self, executor, anima_dir):
        """config.toml must include model = "o4-mini" (stripped prefix)."""
        executor._write_codex_config("prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(
            encoding="utf-8"
        )
        assert 'model = "o4-mini"' in config_toml

    def test_write_codex_config_model_name_gpt41(self, model_config, anima_dir):
        """codex/gpt-4.1 → model = "gpt-4.1" in config.toml."""
        model_config.model = "codex/gpt-4.1"
        exc = CodexSDKExecutor(model_config=model_config, anima_dir=anima_dir)
        exc._write_codex_config("prompt")
        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(
            encoding="utf-8"
        )
        assert 'model = "gpt-4.1"' in config_toml

    def test_toml_escapes_special_characters(self, model_config, anima_dir):
        """Paths with quotes/backslashes are escaped in TOML output."""
        from core.execution.codex_sdk import _escape_toml_string
        assert _escape_toml_string('path/with"quote') == 'path/with\\"quote'
        assert _escape_toml_string("path\\back") == "path\\\\back"


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

        with patch.object(executor, "_create_codex_client", return_value=mock_codex):
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
        msg_item = MagicMock()
        msg_item.type = "message"
        msg_item.content = "Streamed text"

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
    async def test_stream_tool_events(self, executor, anima_dir):
        tool_item = MagicMock()
        tool_item.type = "tool_use"
        tool_item.name = "web_search"
        tool_item.id = "ws-1"
        tool_item.input = {"query": "test"}
        tool_item.output = "results"

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
        assert tool_start["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_stream_interrupted_mid_stream(self, model_config, anima_dir):
        interrupt = asyncio.Event()

        msg_item = MagicMock()
        msg_item.type = "message"
        msg_item.content = "Before interrupt"

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


# ── Mode resolution tests ────────────────────────────────────

class TestModeResolution:
    def test_resolve_execution_mode_codex_pattern(self):
        from core.config.models import resolve_execution_mode, load_config
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
