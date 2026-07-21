"""Unit tests for the LLM-cycle monthly token budget gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core._agent_cycle import CycleMixin
from core._anima_inbox import InboxMixin, InboxResult
from core.memory.token_budget import calculate_token_budget_status
from core.messenger import InboxItem
from core.schemas import CycleResult, Message, ModelConfig


class _DummyCycle(CycleMixin):
    def __init__(self, anima_dir: Path, budget: int | None) -> None:
        self.anima_dir = anima_dir
        self.model_config = ModelConfig(token_budget_monthly=budget)
        self.inner_calls = 0
        self._tool_handler = MagicMock()

    def _get_agent_lock(self, thread_id: str = "default") -> asyncio.Lock:
        return asyncio.Lock()

    async def _run_cycle_inner(self, prompt: str, trigger: str, **kwargs) -> CycleResult:
        self.inner_calls += 1
        return CycleResult(trigger=trigger, action="responded", summary="executed")

    async def _run_cycle_streaming_inner(self, prompt: str, trigger: str, **kwargs):
        self.inner_calls += 1
        yield {
            "type": "cycle_done",
            "cycle_result": CycleResult(trigger=trigger, action="responded").model_dump(mode="json"),
        }


class _DummyInbox(InboxMixin):
    def __init__(self, anima_dir: Path, agent: _DummyCycle) -> None:
        self.anima_dir = anima_dir
        self.name = anima_dir.name
        self.agent = agent
        self._inbox_lock = asyncio.Lock()
        self._cron_idle = asyncio.Event()
        self._cron_idle.set()
        self._status_slots = {"inbox": "idle"}
        self._task_slots = {"inbox": ""}
        self._activity = MagicMock()
        self.messenger = MagicMock()
        self._interrupt_event = asyncio.Event()

    def _get_interrupt_event(self, _thread_id: str) -> asyncio.Event:
        return self._interrupt_event

    def _mark_busy_start(self) -> None:
        pass

    def _notify_lock_released(self) -> None:
        pass

    def _agent_for_lane(self, _lane: str) -> _DummyCycle:
        return self.agent


def test_calculate_token_budget_status() -> None:
    assert calculate_token_budget_status(None, 120).remaining is None
    assert calculate_token_budget_status(None, 120).exceeded is False
    assert calculate_token_budget_status(100, 70).remaining == 30
    assert calculate_token_budget_status(100, 100).exceeded is True
    assert calculate_token_budget_status(100, 120).remaining == 0


@pytest.mark.asyncio
async def test_run_cycle_stops_at_budget_and_records_activity_and_one_notification(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", return_value=125) as monthly_total:
        first = await agent.run_cycle("hello", trigger="inbox")

        notifications = list((tmp_path / "state" / "background_notifications").glob("*.md"))
        assert len(notifications) == 1
        assert "budget: 100" in notifications[0].read_text(encoding="utf-8")
        # Simulate the heartbeat draining the notification.  The durable month
        # marker must still prevent it from being recreated on the next cycle.
        notifications[0].unlink()

        second = await agent.run_cycle("hello again", trigger="inbox")

    assert first.action == "skipped"
    assert second.action == "skipped"
    assert agent.inner_calls == 0
    assert monthly_total.call_count == 2

    activity_paths = list((tmp_path / "activity_log").glob("*.jsonl"))
    assert len(activity_paths) == 1
    entries = [json.loads(line) for line in activity_paths[0].read_text(encoding="utf-8").splitlines()]
    exceeded = [entry for entry in entries if entry["type"] == "budget_exceeded"]
    assert len(exceeded) == 2
    assert exceeded[0]["meta"]["budget"] == 100
    assert exceeded[0]["meta"]["consumed"] == 125
    assert exceeded[0]["meta"]["trigger"] == "inbox"

    assert list((tmp_path / "state" / "background_notifications").glob("*.md")) == []


@pytest.mark.asyncio
async def test_streaming_stops_before_inner_cycle(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", return_value=100):
        events = [event async for event in agent.run_cycle_streaming("hello", trigger="chat")]

    assert agent.inner_calls == 0
    assert [event["type"] for event in events] == ["cycle_done"]
    assert events[0]["cycle_result"]["action"] == "skipped"


@pytest.mark.asyncio
async def test_model_override_cannot_bypass_anima_budget(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", return_value=100):
        result = await agent.run_cycle(
            "hello",
            trigger="cron:daily",
            model_config_override=ModelConfig(token_budget_monthly=None),
        )

    assert result.action == "skipped"
    assert agent.inner_calls == 0


@pytest.mark.asyncio
async def test_unlimited_budget_has_no_aggregation_io_in_both_paths(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=None)

    with patch("core.memory.token_usage.TokenUsageLogger", autospec=True) as usage_logger:
        result = await agent.run_cycle("hello", trigger="manual")
        events = [event async for event in agent.run_cycle_streaming("hello", trigger="chat")]

    assert result.action == "responded"
    assert events[-1]["cycle_result"]["action"] == "responded"
    assert agent.inner_calls == 2
    usage_logger.assert_not_called()
    assert not (tmp_path / "token_usage").exists()


@pytest.mark.asyncio
async def test_budget_blocked_inbox_is_not_read_or_archived(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)
    inbox = _DummyInbox(tmp_path, agent)
    inbox._process_inbox_messages = AsyncMock(
        return_value=InboxResult(
            messages=[MagicMock()],
            senders={"sender"},
            unread_count=1,
            prompt_parts=["message"],
        )
    )

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", return_value=100):
        result = await inbox.process_inbox_message()

    assert result.action == "skipped"
    inbox._process_inbox_messages.assert_awaited_once_with(None, track_retries=False)
    inbox.messenger.archive_paths.assert_not_called()
    assert not (tmp_path / "state" / "inbox_read_counts.json").exists()
    assert inbox._status_slots["inbox"] == "idle"
    assert inbox._task_slots["inbox"] == ""


@pytest.mark.asyncio
async def test_budget_check_io_failure_blocks_cycle(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", side_effect=OSError("unreadable")):
        result = await agent.run_cycle("hello", trigger="manual")

    assert result.action == "skipped"
    assert "could not be verified" in result.summary
    assert agent.inner_calls == 0
    activity_path = next((tmp_path / "activity_log").glob("*.jsonl"))
    entries = [json.loads(line) for line in activity_path.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["type"] == "budget_check_failed"
    assert entries[-1]["meta"]["error"] == "OSError"


@pytest.mark.asyncio
async def test_budget_gate_still_allows_non_llm_inbox_fast_path(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)
    inbox = _DummyInbox(tmp_path, agent)
    inbox_result = InboxResult(
        messages=[MagicMock()],
        senders={"sender"},
        unread_count=1,
        prompt_parts=["message"],
    )
    inbox._process_inbox_messages = AsyncMock(return_value=inbox_result)
    inbox._maybe_fast_reply_external_probe = MagicMock(
        return_value=CycleResult(trigger="inbox:sender", action="responded", summary="fast reply")
    )

    with patch("core.memory.token_usage.TokenUsageLogger.monthly_total", return_value=100):
        result = await inbox.process_inbox_message()

    assert result.action == "responded"
    inbox._process_inbox_messages.assert_awaited_once_with(None, track_retries=False)
    inbox._maybe_fast_reply_external_probe.assert_called_once()


@pytest.mark.asyncio
async def test_budget_blocked_inbox_preprocessing_has_no_persistent_side_effects(tmp_path: Path) -> None:
    agent = _DummyCycle(tmp_path, budget=100)
    inbox = _DummyInbox(tmp_path, agent)
    message = Message(
        from_person="sender",
        to_person=tmp_path.name,
        content="keep queued",
        source="anima",
        intent="question",
    )
    item = InboxItem(msg=message, path=tmp_path / "message.json")
    inbox.messenger.has_unread.return_value = True
    inbox.messenger.receive_with_paths.return_value = [item]

    with patch("core.memory.dedup.MessageDeduplicator.overflow_to_files") as overflow:
        result = await inbox._process_inbox_messages(track_retries=False)

    assert result.inbox_items == [item]
    assert result.unread_count == 1
    assert result.prompt_parts == []
    overflow.assert_not_called()
    inbox.messenger.archive_paths.assert_not_called()
    inbox._activity.log.assert_not_called()
    assert not (tmp_path / "state" / "inbox_read_counts.json").exists()
