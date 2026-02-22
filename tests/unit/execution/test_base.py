"""Tests for core.execution.base — BaseExecutor and ExecutionResult."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.execution.base import BaseExecutor, ExecutionResult
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig
from core.memory.shortterm import ShortTermMemory


# ── ExecutionResult ───────────────────────────────────────────


class TestExecutionResult:
    def test_default_fields(self):
        r = ExecutionResult(text="hello")
        assert r.text == "hello"
        assert r.result_message is None

    def test_with_result_message(self):
        r = ExecutionResult(text="hi", result_message={"usage": {"input_tokens": 100}})
        assert r.text == "hi"
        assert r.result_message == {"usage": {"input_tokens": 100}}

    def test_repr_excludes_result_message(self):
        r = ExecutionResult(text="hi", result_message="secret")
        # result_message has repr=False
        assert "secret" not in repr(r)


# ── BaseExecutor ──────────────────────────────────────────────


class ConcreteExecutor(BaseExecutor):
    """Minimal concrete subclass for testing the abstract base."""

    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        tracker: ContextTracker | None = None,
        shortterm: ShortTermMemory | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(text=f"executed: {prompt}")


class TestBaseExecutor:
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-test-key",
            api_key_env="ANTHROPIC_API_KEY",
        )

    @pytest.fixture
    def executor(self, model_config: ModelConfig, tmp_path: Path) -> ConcreteExecutor:
        return ConcreteExecutor(model_config=model_config, anima_dir=tmp_path)

    def test_stores_model_config(self, executor: ConcreteExecutor, model_config: ModelConfig):
        assert executor._model_config is model_config

    def test_stores_anima_dir(self, executor: ConcreteExecutor, tmp_path: Path):
        assert executor._anima_dir == tmp_path

    def test_resolve_api_key_from_config(self, executor: ConcreteExecutor):
        assert executor._resolve_api_key() == "sk-test-key"

    def test_resolve_api_key_from_env(self, tmp_path: Path):
        config = ModelConfig(model="test", api_key=None, api_key_env="MY_TEST_KEY")
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        with patch.dict(os.environ, {"MY_TEST_KEY": "env-key"}):
            assert executor._resolve_api_key() == "env-key"

    def test_resolve_api_key_none(self, tmp_path: Path):
        config = ModelConfig(model="test", api_key=None, api_key_env="NONEXISTENT_KEY_XYZ")
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_KEY_XYZ", None)
            assert executor._resolve_api_key() is None

    def test_resolve_api_key_prefers_config_over_env(self, tmp_path: Path):
        config = ModelConfig(
            model="test", api_key="config-key", api_key_env="MY_TEST_KEY",
        )
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        with patch.dict(os.environ, {"MY_TEST_KEY": "env-key"}):
            assert executor._resolve_api_key() == "config-key"

    @pytest.mark.asyncio
    async def test_execute(self, executor: ConcreteExecutor):
        result = await executor.execute("hello")
        assert result.text == "executed: hello"

    def test_cannot_instantiate_abstract(self, tmp_path: Path):
        with pytest.raises(TypeError):
            BaseExecutor(  # type: ignore[abstract]
                model_config=ModelConfig(), anima_dir=tmp_path,
            )


# ── _resolve_llm_timeout ─────────────────────────────────────


class TestResolveLlmTimeout:
    """Test LLM timeout resolution logic."""

    def test_explicit_timeout_from_config(self, tmp_path: Path):
        config = ModelConfig(model="claude-sonnet-4-20250514", llm_timeout=120)
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        assert executor._resolve_llm_timeout() == 120

    def test_ollama_model_default(self, tmp_path: Path):
        config = ModelConfig(model="ollama/gemma3:27b", llm_timeout=None)
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        assert executor._resolve_llm_timeout() == 300

    def test_api_model_default(self, tmp_path: Path):
        config = ModelConfig(model="claude-sonnet-4-20250514", llm_timeout=None)
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        assert executor._resolve_llm_timeout() == 600

    def test_explicit_overrides_ollama_default(self, tmp_path: Path):
        config = ModelConfig(model="ollama/gemma3:27b", llm_timeout=60)
        executor = ConcreteExecutor(model_config=config, anima_dir=tmp_path)
        assert executor._resolve_llm_timeout() == 60


# ── ToolCallRecord ───────────────────────────────────────────


class TestToolCallRecord:
    """Test ToolCallRecord dataclass including is_error field."""

    def test_is_error_defaults_to_false(self):
        """is_error should default to False when not specified."""
        from core.execution.base import ToolCallRecord

        record = ToolCallRecord(tool_name="web_search")
        assert record.is_error is False

    def test_is_error_can_be_set_to_true(self):
        """is_error should accept True as an explicit value."""
        from core.execution.base import ToolCallRecord

        record = ToolCallRecord(tool_name="web_search", is_error=True)
        assert record.is_error is True

    def test_is_error_included_in_dataclass_fields(self):
        """is_error should be a recognized field in the dataclass."""
        from dataclasses import fields

        from core.execution.base import ToolCallRecord

        field_names = {f.name for f in fields(ToolCallRecord)}
        assert "is_error" in field_names

    def test_all_fields_set(self):
        """All fields should be settable via constructor."""
        from core.execution.base import ToolCallRecord

        record = ToolCallRecord(
            tool_name="Bash",
            tool_id="toolu_123",
            input_summary="ls -la",
            result_summary="file listing",
            is_error=True,
        )
        assert record.tool_name == "Bash"
        assert record.tool_id == "toolu_123"
        assert record.input_summary == "ls -la"
        assert record.result_summary == "file listing"
        assert record.is_error is True

    def test_default_field_values(self):
        """Fields with defaults should use their defaults."""
        from core.execution.base import ToolCallRecord

        record = ToolCallRecord(tool_name="Read")
        assert record.tool_id == ""
        assert record.input_summary == ""
        assert record.result_summary == ""
        assert record.is_error is False
