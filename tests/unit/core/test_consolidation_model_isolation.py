from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core._agent_cycle import CycleMixin
from core.execution.base import ExecutionResult
from core.schemas import CycleResult, ModelConfig


class _FakeConsolidationAgent:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self._executor = object()
        self.calls: list[dict] = []

    async def run_cycle(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return CycleResult(trigger=kwargs.get("trigger", "manual"), action="responded", summary="done")


class _FakeEngine:
    episodes_dir = MagicMock()

    def collect_activity_chunks(self, *, hours: int, model: str):
        return []

    def _collect_recent_episodes(self, *, hours: int):
        return []

    def _extract_reflections_from_episodes(self, episodes_summary: str):
        return ""

    def _collect_resolved_events(self, *, hours: int):
        return []

    def _collect_error_entries(self, *, hours: int):
        return ""

    def _list_knowledge_files_with_meta(self):
        return []

    def _find_merge_candidates(self, *, max_pairs: int):
        return []


def _make_lifecycle(status_config: ModelConfig):
    from core._anima_lifecycle import LifecycleMixin

    class FakeAnima(LifecycleMixin):
        pass

    obj = FakeAnima.__new__(FakeAnima)
    obj.name = "ritsu"
    obj.memory = MagicMock()
    obj.memory.read_model_config.return_value = status_config
    obj.agent = _FakeConsolidationAgent(status_config)
    obj.model_config = status_config
    return obj


def _mock_config(consolidation_model: str = "openai/deepseek-v4-flash"):
    return SimpleNamespace(consolidation=SimpleNamespace(llm_model=consolidation_model))


@pytest.mark.asyncio
async def test_daily_phase_b_uses_status_model_without_mutating_agent_executor():
    status_config = ModelConfig(model="claude-opus-4-6", resolved_mode="S")
    anima = _make_lifecycle(status_config)
    original_executor = anima.agent._executor

    with (
        patch("core.config.load_config", return_value=_mock_config()),
        patch(
            "core._anima_lifecycle.load_prompt",
            return_value="daily prompt",
        ),
    ):
        result = await anima._run_daily_consolidation(_FakeEngine(), max_turns=7)

    assert result.trigger == "consolidation:daily"
    assert anima.agent._executor is original_executor
    assert anima.agent.model_config.model == "claude-opus-4-6"
    assert len(anima.agent.calls) == 1
    call = anima.agent.calls[0]
    assert call["trigger"] == "consolidation:daily"
    assert call["max_turns_override"] == 7
    assert call["model_config_override"].model == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_weekly_consolidation_uses_status_model_without_mutating_agent_executor():
    status_config = ModelConfig(model="claude-opus-4-6", resolved_mode="S")
    anima = _make_lifecycle(status_config)
    original_executor = anima.agent._executor

    with patch("core._anima_lifecycle.load_prompt", return_value="weekly prompt"):
        result = await anima._run_weekly_consolidation(_FakeEngine(), max_turns=11)

    assert result.trigger == "consolidation:weekly"
    assert anima.agent._executor is original_executor
    assert anima.agent.model_config.model == "claude-opus-4-6"
    assert len(anima.agent.calls) == 1
    call = anima.agent.calls[0]
    assert call["trigger"] == "consolidation:weekly"
    assert call["max_turns_override"] == 11
    assert call["model_config_override"].model == "claude-opus-4-6"


class _FakeExecutor:
    supports_streaming = True

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    async def execute(self, **kwargs):
        return ExecutionResult(text=f"blocking:{self.model_config.model}")

    async def execute_streaming(self, *args, **kwargs):
        yield {"type": "text_delta", "text": f"stream:{self.model_config.model}"}
        yield {
            "type": "done",
            "full_text": f"stream:{self.model_config.model}",
            "result_message": SimpleNamespace(num_turns=1),
            "tool_call_records": [],
            "usage": None,
        }


class _FakeCycle(CycleMixin):
    def __init__(self, tmp_path) -> None:
        self.anima_dir = tmp_path / "ritsu"
        self.anima_dir.mkdir()
        self.memory = MagicMock()
        self.model_config = ModelConfig(model="shared-model", resolved_mode="B")
        self._executor = _FakeExecutor(self.model_config)
        self._tool_registry = []
        self._personal_tools = {}
        self._tool_handler = _FakeToolHandler()
        self.created_executor_configs: list[ModelConfig] = []
        self._progress_callback = None

    def _get_agent_lock(self, thread_id: str = "default"):
        import asyncio

        return asyncio.Lock()

    def _create_executor(self, model_config=None):
        cfg = model_config or self.model_config
        self.created_executor_configs.append(cfg)
        return _FakeExecutor(cfg)

    def _resolve_execution_mode(self, model_config=None):
        return ((model_config or self.model_config).resolved_mode or "B").lower()

    def _load_context_window_overrides(self):
        return {}

    async def _run_priming(self, *args, **kwargs):
        return "", ""

    def _get_retriever(self):
        return None

    def _fit_prompt_to_context_window(self, system_prompt, *args, **kwargs):
        return system_prompt

    def _load_stream_retry_config(self):
        return {
            "checkpoint_enabled": False,
            "retry_max": 0,
            "retry_delay_s": 0,
        }

    async def _preflight_size_check(self, system_prompt, prompt, conv_memory, **kwargs):
        return system_prompt, prompt, False

    @staticmethod
    def _extract_sender(prompt: str, trigger: str) -> str:
        return trigger


def _simple_prompt_result():
    return SimpleNamespace(system_prompt="system")


@pytest.mark.asyncio
async def test_run_cycle_override_uses_local_executor_without_mutating_shared_state(tmp_path):
    agent = _FakeCycle(tmp_path)
    override = ModelConfig(model="claude-opus-4-6", resolved_mode="B")
    prompt_log_calls = []

    with (
        patch("core._agent_cycle.build_system_prompt", return_value=_simple_prompt_result()),
        patch(
            "core._agent_cycle._save_prompt_log",
            side_effect=lambda *args, **kwargs: prompt_log_calls.append(kwargs),
        ),
        patch("core._agent_cycle._save_prompt_log_end"),
    ):
        result = await agent.run_cycle("hello", trigger="manual", model_config_override=override)

    assert result.summary == "blocking:claude-opus-4-6"
    assert agent.model_config.model == "shared-model"
    assert agent._executor.model_config.model == "shared-model"
    assert agent.created_executor_configs == [override]
    assert prompt_log_calls[0]["model"] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_run_cycle_streaming_override_uses_local_executor_without_mutating_shared_state(tmp_path):
    agent = _FakeCycle(tmp_path)
    override = ModelConfig(model="claude-opus-4-6", resolved_mode="S")
    prompt_log_calls = []
    chunks = []

    with (
        patch("core._agent_cycle.build_system_prompt", return_value=_simple_prompt_result()),
        patch(
            "core._agent_cycle._save_prompt_log",
            side_effect=lambda *args, **kwargs: prompt_log_calls.append(kwargs),
        ),
        patch("core._agent_cycle._save_prompt_log_end"),
    ):
        async for chunk in agent.run_cycle_streaming(
            "hello",
            trigger="message:taka",
            model_config_override=override,
        ):
            chunks.append(chunk)

    assert chunks[-1]["cycle_result"]["summary"] == "stream:claude-opus-4-6"
    assert agent.model_config.model == "shared-model"
    assert agent._executor.model_config.model == "shared-model"
    assert agent.created_executor_configs == [override]
    assert prompt_log_calls[0]["model"] == "claude-opus-4-6"


class _FakeToolHandler:
    session_id = "sid"

    def bind_runtime_session(self, ctx):
        self.session_id = ctx.tool_session_id

    def set_active_session_type(self, session_type: str):
        from core.tooling.handler import active_session_type

        return active_session_type.set(session_type)

    def set_session_origin(self, *args, **kwargs):
        return None


class _FakeMessagingAgent:
    def __init__(self) -> None:
        self._tool_handler = _FakeToolHandler()
        self.resolved_mode_config: ModelConfig | None = None
        self.streaming_config: ModelConfig | None = None

    def set_interrupt_event(self, event):
        return None

    def _resolve_execution_mode(self, model_config=None):
        self.resolved_mode_config = model_config
        return ((model_config or ModelConfig()).resolved_mode or "A").lower()

    async def run_cycle_streaming(self, *args, **kwargs):
        self.streaming_config = kwargs.get("model_config_override")
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "trigger": kwargs.get("trigger", "message:taka"),
                "action": "responded",
                "summary": "ok",
                "duration_ms": 1,
                "context_usage_ratio": 0.0,
                "session_chained": False,
                "total_turns": 1,
                "tool_call_records": [],
            },
        }

    def drain_notifications(self):
        return []


class _FakeSessionCompactor:
    def cancel(self, *args, **kwargs):
        return None

    def schedule(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_process_message_stream_uses_status_config_for_human_chat(tmp_path):
    from core._anima_messaging import MessagingMixin

    class FakeAnima(MessagingMixin):
        pass

    status_config = ModelConfig(model="claude-opus-4-6", resolved_mode="S")
    poisoned_runtime_config = ModelConfig(model="openai/deepseek-v4-flash", resolved_mode="A")
    agent = _FakeMessagingAgent()

    anima = FakeAnima.__new__(FakeAnima)
    anima.name = "ritsu"
    anima.anima_dir = tmp_path / "animas" / "ritsu"
    anima.anima_dir.mkdir(parents=True)
    anima.model_config = poisoned_runtime_config
    anima.memory = MagicMock()
    anima.memory.read_model_config.return_value = status_config
    anima.agent = agent
    anima.needs_bootstrap = False
    anima._thread_locks = {"default": asyncio.Lock()}
    anima._interrupt_events = {}
    anima._session_compactor = _FakeSessionCompactor()
    anima._status_slots = {}
    anima._task_slots = {}
    anima._activity = MagicMock()
    anima._last_activity = None

    anima._validate_thread_id = lambda thread_id: None
    anima._get_thread_lock = lambda thread_id: anima._thread_locks.setdefault(thread_id, asyncio.Lock())
    anima._get_interrupt_event = lambda thread_id: anima._interrupt_events.setdefault(thread_id, asyncio.Event())
    anima._mark_busy_start = lambda: None
    anima._notify_lock_released = lambda: None
    anima._log_human_conversation = lambda *args, **kwargs: None
    anima._resolve_chat_external_recipient = lambda *args, **kwargs: None
    anima._maybe_neo4j_realtime_ingest = lambda *args, **kwargs: None

    chunks = [
        chunk
        async for chunk in anima.process_message_stream(
            "hello",
            from_person="taka",
            thread_id="default",
        )
    ]

    assert chunks[-1]["type"] == "cycle_done"
    assert agent.resolved_mode_config is status_config
    assert agent.streaming_config is status_config
    assert agent.streaming_config.model == "claude-opus-4-6"
    assert anima.model_config.model == "openai/deepseek-v4-flash"
