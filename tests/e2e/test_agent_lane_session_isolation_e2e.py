from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.supervisor.pending_executor import PendingTaskExecutor


def _prepare_anima_dirs(tmp_path: Path) -> tuple[Path, Path]:
    anima_dir = tmp_path / "animas" / "lane-e2e"
    shared_dir = tmp_path / "shared"
    for directory in [
        anima_dir / "state",
        anima_dir / "episodes",
        anima_dir / "knowledge",
        anima_dir / "procedures",
        anima_dir / "skills",
        anima_dir / "shortterm",
        anima_dir / "activity_log",
        shared_dir / "inbox" / "lane-e2e",
        shared_dir / "channels",
        shared_dir / "users",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text("# Lane E2E", encoding="utf-8")
    (anima_dir / "injection.md").write_text("Lane isolation", encoding="utf-8")
    # Keep a single background worker so AgentCore is created once per lane
    # (chat / background / inbox) and the legacy lane mock wiring stays valid.
    (anima_dir / "status.json").write_text(
        '{"enabled": true, "role": "general", "model": "claude-sonnet-4-6", "background_worker_pool_size": 1}',
        encoding="utf-8",
    )
    return anima_dir, shared_dir


def _mock_agent(name: str) -> MagicMock:
    agent = MagicMock(name=name)
    agent.background_manager = None
    agent.execution_mode = "s"
    agent._tool_handler = MagicMock()
    return agent


@pytest.mark.e2e
async def test_taskexec_runs_on_background_lane_while_chat_session_lock_is_held(tmp_path: Path) -> None:
    """TaskExec should not wait on the chat lane session lock."""
    anima_dir, shared_dir = _prepare_anima_dirs(tmp_path)
    chat_agent = _mock_agent("chat_agent")
    background_agent = _mock_agent("background_agent")
    inbox_agent = _mock_agent("inbox_agent")

    async def _background_stream(*args, **kwargs):
        yield {"type": "text_delta", "text": "ok"}
        yield {
            "type": "cycle_done",
            "cycle_result": {"summary": "background ok", "action": "complete"},
        }

    background_agent.run_cycle_streaming = MagicMock(side_effect=lambda *a, **kw: _background_stream(*a, **kw))

    with patch("core.anima.AgentCore", side_effect=[chat_agent, background_agent, inbox_agent]):
        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)

    executor = PendingTaskExecutor(
        anima=anima,
        anima_name="lane-e2e",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )
    task_desc = {
        "task_id": "e2e-lane-task",
        "title": "E2E lane isolation",
        "description": "Ensure TaskExec uses background lane",
        "working_directory": str(tmp_path),
    }

    with (
        patch("core.paths.load_prompt", return_value="task prompt"),
        patch("core.memory.activity.ActivityLogger") as mock_activity,
    ):
        mock_activity.return_value.log = MagicMock()
        async with anima._agent_session_context("chat"):
            result = await asyncio.wait_for(executor._run_llm_task(task_desc), timeout=1.0)

    assert result == "background ok"
    background_agent.run_cycle_streaming.assert_called_once()
    chat_agent.run_cycle_streaming.assert_not_called()
    background_agent.set_task_cwd.assert_any_call(tmp_path)
    background_agent.set_task_cwd.assert_any_call(None)
    chat_agent.set_task_cwd.assert_not_called()
