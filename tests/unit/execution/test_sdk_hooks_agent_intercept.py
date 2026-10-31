"""Tests for Agent/Task tool hard-block in _sdk_hooks.py.

Verifies:
  - Both "Agent" and "Task" tool names are hard-blocked (no pending creation)
  - "TaskOutput" and "AgentOutput" are also blocked
  - Deny reason redirects to direct execution / delegate_task
  - No state/pending/ files are created
  - on_task_intercepted callback is NOT fired for Agent/Task
  - submit_tasks intercept only works in explicit background sessions
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "ayame"
    d.mkdir(parents=True)
    (d / "state").mkdir()
    return d


# ── PreToolUse hook: Agent/Task hard-block ────────────────────


class TestPreToolHookAgentHardBlock:
    """Test the PreToolUse hook hard-blocks Agent/Task without creating pending tasks."""

    def _build_hook(self, anima_dir: Path, *, has_subordinates: bool = False):
        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

        return _build_pre_tool_hook(
            anima_dir,
            has_subordinates=has_subordinates,
        )

    @pytest.mark.asyncio
    async def test_agent_tool_hard_blocked(self, anima_dir: Path):
        """'Agent' tool should be hard-blocked with redirect message."""
        hook = self._build_hook(anima_dir, has_subordinates=False)

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Agent",
            "tool_input": {
                "description": "Research task",
                "prompt": "Find information about X",
            },
        }
        result = await hook(input_data, "tu_agent_01", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]
        assert "submit_tasks" not in output["permissionDecisionReason"]

        pending_files = list((anima_dir / "state" / "pending").glob("*.json"))
        assert len(pending_files) == 0, "No pending task should be written"

    @pytest.mark.asyncio
    async def test_task_tool_hard_blocked(self, anima_dir: Path):
        """'Task' tool should also be hard-blocked."""
        hook = self._build_hook(anima_dir, has_subordinates=False)

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Task",
            "tool_input": {
                "description": "Build task",
                "prompt": "Compile the project",
            },
        }
        result = await hook(input_data, "tu_task_01", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]

        pending_files = list((anima_dir / "state" / "pending").glob("*.json"))
        assert len(pending_files) == 0

    @pytest.mark.asyncio
    async def test_agent_blocked_for_supervisor_too(self, anima_dir: Path):
        """Agent is hard-blocked even for supervisors (no delegation attempt)."""
        hook = self._build_hook(anima_dir, has_subordinates=True)

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Agent",
            "tool_input": {
                "description": "Delegate task",
                "prompt": "Do something important",
            },
        }
        result = await hook(input_data, "tu_agent_02", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]
        assert "delegate_task" in output["permissionDecisionReason"]

        pending_files = list((anima_dir / "state" / "pending").glob("*.json"))
        assert len(pending_files) == 0

    @pytest.mark.asyncio
    async def test_agent_output_blocked(self, anima_dir: Path):
        """'AgentOutput' should be blocked (Agent/Task are disabled)."""
        hook = self._build_hook(anima_dir, has_subordinates=False)

        mock_context = MagicMock()
        output_input = {
            "tool_name": "AgentOutput",
            "tool_input": {"task_id": "some_task_id"},
        }
        result = await hook(output_input, "tu_output_01", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_task_output_blocked(self, anima_dir: Path):
        """'TaskOutput' should be blocked."""
        hook = self._build_hook(anima_dir, has_subordinates=False)

        mock_context = MagicMock()
        output_input = {
            "tool_name": "TaskOutput",
            "tool_input": {"task_id": "some_task_id"},
        }
        result = await hook(output_input, "tu_output_02", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_on_task_intercepted_not_called_for_agent(self, anima_dir: Path):
        """on_task_intercepted callback should NOT fire for hard-blocked Agent."""
        callback_called = []

        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            anima_dir,
            has_subordinates=False,
            on_task_intercepted=lambda: callback_called.append(True),
        )

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Agent",
            "tool_input": {"description": "test", "prompt": "test"},
        }
        await hook(input_data, "tu_cb_01", mock_context)

        assert len(callback_called) == 0, "Callback should NOT fire for hard-blocked Agent"

    def _build_hook_with_trigger(self, anima_dir: Path, trigger: str):
        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

        return _build_pre_tool_hook(
            anima_dir,
            has_subordinates=False,
            session_stats={
                "tool_call_count": 0,
                "total_result_bytes": 0,
                "system_prompt_tokens": 100,
                "user_prompt_tokens": 50,
                "force_chain": False,
                "trigger": trigger,
                "start_time": 0.0,
                "hb_soft_warned": False,
                "hb_soft_timeout": 300,
            },
        )

    @pytest.mark.asyncio
    async def test_agent_blocked_in_chat_and_task_triggers(self, anima_dir: Path):
        """Agent tool should be blocked for chat and task triggers."""
        for trigger in ["chat", "task:abc123", "cron:daily"]:
            hook = self._build_hook_with_trigger(anima_dir, trigger)

            mock_context = MagicMock()
            input_data = {
                "tool_name": "Agent",
                "tool_input": {"description": "research", "prompt": "find Z"},
            }
            result = await hook(input_data, f"tu_{trigger}", mock_context)

            output = result.get("hookSpecificOutput")
            assert output is not None
            assert output["permissionDecision"] == "deny"
            assert "BLOCKED" in output["permissionDecisionReason"], f"Failed for trigger={trigger}"

    @pytest.mark.asyncio
    async def test_agent_blocked_in_heartbeat_after_warmup(self, anima_dir: Path):
        """Agent tool blocked in heartbeat (soft timeout fires once, then block)."""
        import time

        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

        stats = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": 100,
            "user_prompt_tokens": 50,
            "force_chain": False,
            "trigger": "heartbeat",
            "start_time": time.monotonic(),
            "hb_soft_warned": True,
            "hb_soft_timeout": 300,
        }
        hook = _build_pre_tool_hook(anima_dir, has_subordinates=False, session_stats=stats)

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Agent",
            "tool_input": {"description": "research", "prompt": "find Z"},
        }
        result = await hook(input_data, "tu_hb", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_read_tool_not_intercepted(self, anima_dir: Path):
        """Non-Agent/Task tools should pass through normally."""
        hook = self._build_hook(anima_dir, has_subordinates=False)

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Read",
            "tool_input": {"file_path": str(anima_dir / "IDENTITY.md")},
        }
        result = await hook(input_data, "tu_read_01", mock_context)

        output = result.get("hookSpecificOutput")
        if output is not None:
            assert output.get("permissionDecision") != "deny"


# ── PreToolUse hook: submit_tasks intercept deny reason ─────────────────


class TestSubmitTasksInterceptDenyReason:
    """Test submit_tasks intercept returns improved deny reason to prevent duplicate delegation."""

    def _background_stats(self) -> dict:
        return {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": 100,
            "user_prompt_tokens": 50,
            "force_chain": False,
            "trigger": "background:manual",
            "start_time": 0.0,
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
        }

    def _build_hook(self, anima_dir: Path, *, has_subordinates: bool = False, session_stats: dict | None = None):
        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

        return _build_pre_tool_hook(
            anima_dir,
            has_subordinates=has_subordinates,
            session_stats=session_stats,
        )

    @pytest.mark.asyncio
    async def test_submit_tasks_intercept_success_reason(self, anima_dir: Path):
        """Success case: deny reason is the short success message with task_ids."""
        success_result = json.dumps(
            {
                "status": "submitted",
                "batch_id": "test",
                "task_count": 2,
                "task_ids": ["t1", "t2"],
                "message": "Batch submitted",
            },
            ensure_ascii=False,
        )

        with patch(
            "core.tooling.policy.submit_tasks.submit_tasks",
            return_value=success_result,
        ):
            hook = self._build_hook(
                anima_dir,
                has_subordinates=False,
                session_stats=self._background_stats(),
            )
            mock_context = MagicMock()
            input_data = {
                "tool_name": "submit_tasks",
                "tool_input": {
                    "batch_id": "test",
                    "tasks": [
                        {"task_id": "t1", "title": "T1", "description": "D1"},
                        {"task_id": "t2", "title": "T2", "description": "D2"},
                    ],
                },
            }
            result = await hook(input_data, "tu_001", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        reason = output["permissionDecisionReason"]
        assert "t1, t2" in reason
        assert not reason.startswith("BLOCKED")
        assert "DUPLICATE" not in reason

    @pytest.mark.asyncio
    async def test_submit_tasks_intercept_error_reason(self, anima_dir: Path):
        """Error case: deny reason does NOT start with SUCCESS, contains error."""
        error_result = json.dumps(
            {
                "status": "error",
                "error_type": "InvalidArguments",
                "message": "batch_id is required",
            },
            ensure_ascii=False,
        )

        with patch(
            "core.tooling.policy.submit_tasks.submit_tasks",
            return_value=error_result,
        ):
            hook = self._build_hook(
                anima_dir,
                has_subordinates=False,
                session_stats=self._background_stats(),
            )
            mock_context = MagicMock()
            input_data = {
                "tool_name": "submit_tasks",
                "tool_input": {"batch_id": "", "tasks": []},
            }
            result = await hook(input_data, "tu_002", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        reason = output["permissionDecisionReason"]
        assert not reason.startswith("SUCCESS")
        assert "error" in reason.lower()

    @pytest.mark.asyncio
    async def test_submit_tasks_blocked_in_taskexec(self, anima_dir: Path):
        """submit_tasks should be blocked from TaskExec sessions."""
        hook = self._build_hook(
            anima_dir,
            session_stats={
                "tool_call_count": 0,
                "total_result_bytes": 0,
                "system_prompt_tokens": 100,
                "user_prompt_tokens": 50,
                "force_chain": False,
                "trigger": "task:xyz789",
                "start_time": 0.0,
                "hb_soft_warned": False,
                "hb_soft_timeout": 300,
            },
        )

        mock_context = MagicMock()
        input_data = {
            "tool_name": "submit_tasks",
            "tool_input": {
                "batch_id": "test",
                "tasks": [{"task_id": "t1", "title": "T1", "description": "D1"}],
            },
        }
        result = await hook(input_data, "tu_st_taskexec", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "BLOCKED" in output["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_submit_tasks_blocked_in_normal_chat(self, anima_dir: Path):
        """submit_tasks should not execute in normal chat sessions."""
        hook = self._build_hook(
            anima_dir,
            session_stats={
                **self._background_stats(),
                "trigger": "chat",
            },
        )

        mock_context = MagicMock()
        input_data = {
            "tool_name": "submit_tasks",
            "tool_input": {
                "batch_id": "test",
                "tasks": [{"task_id": "t1", "title": "T1", "description": "D1"}],
            },
        }
        with patch("core.tooling.policy.submit_tasks.submit_tasks") as handler:
            result = await hook(input_data, "tu_st_chat", mock_context)

        output = result.get("hookSpecificOutput")
        assert output is not None
        assert output["permissionDecision"] == "deny"
        assert "submit_tasks" in output["permissionDecisionReason"]
        assert "BLOCKED" in output["permissionDecisionReason"]
        handler.assert_not_called()
