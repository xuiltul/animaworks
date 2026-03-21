"""E2E tests for Agent/Task tool interception flow.

Verifies the full pipeline:
  1. PreToolUse hook intercepts "Agent" tool
  2. Pending JSON written to state/pending/ with reply_to
  3. _tool_summary handles "Agent" tool
  4. Channel E reads task_results for Heartbeat visibility
  5. _allowed_tools includes both "Task" and "Agent"
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "animas" / "ayame"
        for sub in [
            "episodes", "knowledge", "skills",
            "state", "state/pending", "state/task_results",
        ]:
            (d / sub).mkdir(parents=True)
        yield d


class TestAgentToolInterceptE2E:
    """Full pipeline: Agent tool → pending JSON → Channel E visibility."""

    @pytest.mark.asyncio
    async def test_full_intercept_and_channel_e_visibility(self, anima_dir: Path):
        """Agent tool call → pending file → task result → Channel E shows it."""
        # Step 1: Intercept an Agent tool call
        from core.execution._sdk_hooks import _intercept_task_to_pending

        tool_input = {
            "description": "Research AI safety standards",
            "prompt": "Find and summarize current AI safety frameworks",
        }
        task_id = _intercept_task_to_pending(anima_dir, tool_input, "tu_e2e_001")

        # Verify pending file
        pending_file = anima_dir / "state" / "pending" / f"{task_id}.json"
        assert pending_file.exists()

        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["reply_to"] == "ayame"
        assert data["task_type"] == "llm"
        assert data["submitted_by"] == "self_task_intercept"

        # Step 2: Simulate TaskExec completion by writing a result
        result_content = (
            f"# Task Result: {task_id}\n\n"
            "Found 3 major AI safety frameworks:\n"
            "1. NIST AI RMF\n"
            "2. EU AI Act\n"
            "3. ISO/IEC 42001"
        )
        (anima_dir / "state" / "task_results" / f"{task_id}.md").write_text(
            result_content, encoding="utf-8",
        )

        # Step 3: Verify Channel E shows the result
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)
        channel_e_output = await engine._channel_e_pending_tasks()

        assert task_id in channel_e_output
        assert "完了済みバックグラウンドタスク" in channel_e_output
        assert "Task Result" in channel_e_output

    def test_tool_summary_handles_agent(self):
        """_tool_summary generates detail for both Agent and Task tools."""
        from core.execution._tool_summary import make_tool_detail_chunk

        agent_chunk = make_tool_detail_chunk(
            "Agent", "tool_1", {"description": "Research task"},
        )
        assert agent_chunk is not None
        assert agent_chunk["detail"] == "Research task"
        assert agent_chunk["tool_name"] == "Agent"

        task_chunk = make_tool_detail_chunk(
            "Task", "tool_2", {"description": "Build task"},
        )
        assert task_chunk is not None
        assert task_chunk["detail"] == "Build task"

    @pytest.mark.asyncio
    async def test_hook_intercepts_agent_and_writes_reply_to(self, anima_dir: Path):
        """Full hook flow: Agent tool → intercepted → pending has reply_to."""
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            anima_dir,
            has_subordinates=False,
        )

        mock_context = MagicMock()
        input_data = {
            "tool_name": "Agent",
            "tool_input": {
                "description": "Background analysis",
                "prompt": "Analyze the codebase structure",
            },
        }

        result = await hook(input_data, "tu_e2e_002", mock_context)

        output = result.get("hookSpecificOutput")
        assert output["permissionDecision"] == "deny"
        assert "INTERCEPT_OK" in output["permissionDecisionReason"]

        reason = output["permissionDecisionReason"]
        task_id_start = reason.index("task_id: ") + len("task_id: ")
        task_id_end = reason.index(")", task_id_start)
        task_id = reason[task_id_start:task_id_end]

        # Verify the pending file has reply_to
        pending_file = anima_dir / "state" / "pending" / f"{task_id}.json"
        assert pending_file.exists()

        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["reply_to"] == "ayame"

    @pytest.mark.asyncio
    async def test_agent_output_after_agent_intercept(self, anima_dir: Path):
        """AgentOutput following an intercepted Agent call is handled."""
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            anima_dir,
            has_subordinates=False,
        )

        mock_context = MagicMock()

        # Intercept Agent call
        agent_input = {
            "tool_name": "Agent",
            "tool_input": {"description": "test", "prompt": "test"},
        }
        await hook(agent_input, "tu_e2e_003", mock_context)

        # Get task_id from pending
        pending_files = list((anima_dir / "state" / "pending").glob("*.json"))
        assert len(pending_files) == 1
        task_id = pending_files[0].stem

        # AgentOutput should be intercepted
        agent_output_input = {
            "tool_name": "AgentOutput",
            "tool_input": {"task_id": task_id},
        }
        result = await hook(agent_output_input, "tu_e2e_004", mock_context)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "INTERCEPT_OK" in result["hookSpecificOutput"]["permissionDecisionReason"]

        # TaskOutput should also be intercepted for the same task_id
        task_output_input = {
            "tool_name": "TaskOutput",
            "tool_input": {"task_id": task_id},
        }
        result = await hook(task_output_input, "tu_e2e_005", mock_context)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestBypassPermissionsConfig:
    """Verify _build_sdk_options uses bypassPermissions mode."""

    def test_bypass_permissions_mode(self):
        """The _build_sdk_options should set permission_mode to bypassPermissions."""
        from tests.helpers.mocks import patch_agent_sdk

        with patch_agent_sdk():
            from core.execution.agent_sdk import AgentSDKExecutor
            from core.schemas import ModelConfig

            config = ModelConfig(
                model="claude-sonnet-4-6",
                api_key="sk-test",
                max_turns=5,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                ad = Path(tmpdir) / "animas" / "test"
                ad.mkdir(parents=True)

                executor = AgentSDKExecutor(
                    model_config=config,
                    anima_dir=ad,
                )

                session_stats = {
                    "tool_call_count": 0,
                    "total_result_bytes": 0,
                    "system_prompt_tokens": 100,
                    "user_prompt_tokens": 50,
                    "force_chain": False,
                }

                options, pf = executor._build_sdk_options(
                    "test prompt", 5, 200000, session_stats,
                )

                assert options.permission_mode == "bypassPermissions"
                if pf:
                    pf.unlink(missing_ok=True)
