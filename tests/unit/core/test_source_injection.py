"""Unit tests for source parameter injection in process_message / process_message_stream.

Verifies that external platform source info is injected into the LLM prompt
so the Anima knows not to attempt send_message via other channels (Issue #38).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import EXTERNAL_PLATFORM_SOURCES, CycleResult
from core.tooling.handler import active_session_type


def _make_cycle_result(**kwargs) -> CycleResult:
    defaults = dict(trigger="test", action="responded", summary="ok", duration_ms=50)
    defaults.update(kwargs)
    return CycleResult(**defaults)


def _wire_session_type(dp) -> None:
    dp.agent._tool_handler.set_active_session_type = lambda st: active_session_type.set(st)


def _setup_anima(make_anima, data_dir):
    """Create a DigitalAnima with mocked dependencies."""
    anima_dir = make_anima("alice")
    shared_dir = data_dir / "shared"

    with patch("core.anima.AgentCore") as MockAgent, \
         patch("core.anima.MemoryManager") as MockMM, \
         patch("core.anima.Messenger"), \
         patch("core._anima_messaging.ConversationMemory") as MockConv:
        MockMM.return_value.read_model_config.return_value = MagicMock()
        MockConv.return_value.compress_if_needed = AsyncMock()
        MockConv.return_value.finalize_session = AsyncMock(return_value=False)
        MockConv.return_value.build_chat_prompt.return_value = "history + user msg"
        MockConv.return_value.append_turn = MagicMock()
        MockConv.return_value.save = MagicMock()
        MockConv.return_value.write_transcript = MagicMock()
        MockConv.return_value.needs_compression = MagicMock(return_value=False)

        from core.anima import DigitalAnima
        dp = DigitalAnima(anima_dir, shared_dir)
        _wire_session_type(dp)
        return dp


class TestProcessMessageSource:
    """process_message with source parameter."""

    async def test_external_source_injects_platform_context(self, data_dir, make_anima):
        """When source is an external platform, prompt includes platform_context."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _make_cycle_result()

        dp.agent.run_cycle = _capture_run_cycle
        await dp.process_message("Hello", from_person="human", source="googlechat")

        assert len(captured_prompts) == 1
        assert "platform_context" in captured_prompts[0]
        assert "googlechat" in captured_prompts[0]
        assert "send_message" in captured_prompts[0]

    async def test_no_source_no_injection(self, data_dir, make_anima):
        """When source is empty, prompt is unchanged."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _make_cycle_result()

        dp.agent.run_cycle = _capture_run_cycle
        await dp.process_message("Hello", from_person="human")

        assert len(captured_prompts) == 1
        assert "platform_context" not in captured_prompts[0]

    async def test_unknown_source_no_injection(self, data_dir, make_anima):
        """When source is not in EXTERNAL_PLATFORM_SOURCES, no injection."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _make_cycle_result()

        dp.agent.run_cycle = _capture_run_cycle
        await dp.process_message("Hello", from_person="human", source="webui")

        assert len(captured_prompts) == 1
        assert "platform_context" not in captured_prompts[0]

    async def test_slack_source_injects_context(self, data_dir, make_anima):
        """Slack source also triggers injection."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _make_cycle_result()

        dp.agent.run_cycle = _capture_run_cycle
        await dp.process_message("Hi", from_person="human", source="slack")

        assert len(captured_prompts) == 1
        assert "platform_context" in captured_prompts[0]
        assert "slack" in captured_prompts[0]

    async def test_original_content_preserved_in_prompt(self, data_dir, make_anima):
        """The original user message is still present in the prompt after injection."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            captured_prompts.append(prompt)
            return _make_cycle_result()

        dp.agent.run_cycle = _capture_run_cycle
        await dp.process_message("My unique message", from_person="human", source="googlechat")

        assert "My unique message" in captured_prompts[0]

    @pytest.mark.parametrize("from_person", ["fugaku", "shokotan", "operator"])
    async def test_line_trigger_survives_session_guard(self, data_dir, make_anima, from_person: str):
        """The source-qualified LINE trigger is used by both execution and its guard."""
        dp = _setup_anima(make_anima, data_dir)
        captured_triggers: list[str] = []

        async def _capture_run_cycle(prompt, **kwargs):
            del prompt
            captured_triggers.append(kwargs["trigger"])
            return _make_cycle_result(trigger=kwargs["trigger"], thread_id="default")

        dp.agent.run_cycle = _capture_run_cycle
        response = await dp.process_message("Hello", from_person=from_person, source="line")

        assert captured_triggers == [f"message:line:{from_person}"]
        assert response == "ok"


class TestProcessMessageStreamSource:
    """process_message_stream with source parameter."""

    async def test_stream_external_source_injects_context(self, data_dir, make_anima):
        """Streaming path also injects platform context for external sources."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_streaming(prompt, **kwargs):
            captured_prompts.append(prompt)
            yield {"type": "cycle_done", "cycle_result": {"summary": "ok", "tool_call_records": []}}

        dp.agent.run_cycle_streaming = _capture_streaming
        chunks = []
        async for chunk in dp.process_message_stream("Hello", from_person="human", source="chatwork"):
            chunks.append(chunk)

        assert len(captured_prompts) == 1
        assert "platform_context" in captured_prompts[0]
        assert "chatwork" in captured_prompts[0]

    async def test_stream_no_source_no_injection(self, data_dir, make_anima):
        """Streaming without source does not inject context."""
        dp = _setup_anima(make_anima, data_dir)
        captured_prompts: list[str] = []

        async def _capture_streaming(prompt, **kwargs):
            captured_prompts.append(prompt)
            yield {"type": "cycle_done", "cycle_result": {"summary": "ok", "tool_call_records": []}}

        dp.agent.run_cycle_streaming = _capture_streaming
        chunks = []
        async for chunk in dp.process_message_stream("Hello", from_person="human"):
            chunks.append(chunk)

        assert len(captured_prompts) == 1
        assert "platform_context" not in captured_prompts[0]


class TestExternalPlatformSources:
    """Verify EXTERNAL_PLATFORM_SOURCES includes googlechat."""

    def test_googlechat_in_sources(self):
        assert "googlechat" in EXTERNAL_PLATFORM_SOURCES

    def test_slack_in_sources(self):
        assert "slack" in EXTERNAL_PLATFORM_SOURCES

    def test_chatwork_in_sources(self):
        assert "chatwork" in EXTERNAL_PLATFORM_SOURCES


class TestSupervisorSourcePassthrough:
    """Verify that runner/streaming_handler pass source from params."""

    async def test_runner_passes_source(self):
        """_handle_process_message extracts source from params."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner.__new__(AnimaRunner)
        runner._scheduler_mgr = None
        mock_anima = MagicMock()
        mock_anima.process_message = AsyncMock(return_value={"summary": "ok", "images": []})
        runner.anima = mock_anima

        params = {
            "message": "Hello",
            "from_person": "human",
            "source": "googlechat",
        }
        await runner._handle_process_message(params)

        mock_anima.process_message.assert_called_once()
        call_kwargs = mock_anima.process_message.call_args
        assert call_kwargs.kwargs.get("source") == "googlechat"

    async def test_runner_default_source_empty(self):
        """When source is not in params, it defaults to empty string."""
        from core.supervisor.runner import AnimaRunner

        runner = AnimaRunner.__new__(AnimaRunner)
        runner._scheduler_mgr = None
        mock_anima = MagicMock()
        mock_anima.process_message = AsyncMock(return_value={"summary": "ok", "images": []})
        runner.anima = mock_anima

        params = {"message": "Hello", "from_person": "human"}
        await runner._handle_process_message(params)

        call_kwargs = mock_anima.process_message.call_args
        assert call_kwargs.kwargs.get("source") == ""
