"""Unit tests for Task tool intercept → pending LLM task conversion."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── _intercept_task_to_pending ───────────────────────────────


class TestInterceptTaskToPending:
    """Verify _intercept_task_to_pending writes correct pending JSON."""

    def test_creates_pending_json(self, tmp_path: Path) -> None:
        """A JSON file is written to state/pending/ with correct structure."""
        from core.execution.agent_sdk import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        tool_input = {
            "description": "Implement feature X",
            "prompt": "Read the issue file and create a worktree...",
        }

        with patch("core.execution._sdk_hooks._log_tool_use"):
            task_id = _intercept_task_to_pending(anima_dir, tool_input, "tool-123")

        pending_dir = anima_dir / "state" / "pending"
        task_files = list(pending_dir.glob("*.json"))
        assert len(task_files) == 1

        task_desc = json.loads(task_files[0].read_text(encoding="utf-8"))
        assert task_desc["task_type"] == "llm"
        assert task_desc["task_id"] == task_id
        assert task_desc["title"] == "Implement feature X"
        assert task_desc["description"] == "Read the issue file and create a worktree..."
        assert task_desc["submitted_by"] == "self_task_intercept"
        assert "submitted_at" in task_desc

    def test_prompt_fallback_to_description(self, tmp_path: Path) -> None:
        """When prompt is absent, description is used as the task body."""
        from core.execution.agent_sdk import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        tool_input = {"description": "Quick background check"}

        with patch("core.execution._sdk_hooks._log_tool_use"):
            _intercept_task_to_pending(anima_dir, tool_input, None)

        pending_dir = anima_dir / "state" / "pending"
        task_desc = json.loads(
            next(pending_dir.glob("*.json")).read_text(encoding="utf-8"),
        )
        assert task_desc["description"] == "Quick background check"

    def test_logs_activity(self, tmp_path: Path) -> None:
        """Tool use is logged as intercept info (not blocked)."""
        from core.execution.agent_sdk import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        tool_input = {"description": "test", "prompt": "do stuff"}

        with patch("core.execution._sdk_hooks._log_tool_use") as mock_log:
            _intercept_task_to_pending(anima_dir, tool_input, "tid-1")

        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs["blocked"] is False

    def test_json_compatible_with_execute_llm_task(self, tmp_path: Path) -> None:
        """Generated JSON has all fields expected by PendingTaskExecutor._execute_llm_task."""
        from core.execution.agent_sdk import _intercept_task_to_pending

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        tool_input = {
            "description": "Build the feature",
            "prompt": "Step 1: read file\nStep 2: edit code",
        }

        with patch("core.execution._sdk_hooks._log_tool_use"):
            _intercept_task_to_pending(anima_dir, tool_input, None)

        task_desc = json.loads(
            next((anima_dir / "state" / "pending").glob("*.json")).read_text(
                encoding="utf-8",
            ),
        )

        required_keys = {
            "task_type", "task_id", "title", "description",
            "context", "acceptance_criteria", "constraints",
            "file_paths", "submitted_by", "submitted_at",
        }
        assert required_keys <= set(task_desc.keys())


# ── PreToolUse hook Task branch ──────────────────────────────


class TestPreToolHookTaskBranch:
    """Verify the PreToolUse hook intercepts the Task tool."""

    @pytest.fixture()
    def hook(self, tmp_path: Path):
        """Build a pre-tool hook for testing (supervisor mode)."""
        try:
            import claude_agent_sdk.types  # noqa: F401
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        from core.execution.agent_sdk import _build_pre_tool_hook

        anima_dir = tmp_path / "animas" / "hook_test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state" / "pending").mkdir(parents=True)

        with patch("core.execution._sdk_hooks._cache_subordinate_paths", return_value=(set(), set(), set())):
            return _build_pre_tool_hook(anima_dir, has_subordinates=True)

    @pytest.fixture()
    def hook_with_callback(self, tmp_path: Path):
        """Build a pre-tool hook with on_task_intercepted callback (supervisor mode)."""
        try:
            import claude_agent_sdk.types  # noqa: F401
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        from core.execution.agent_sdk import _build_pre_tool_hook

        anima_dir = tmp_path / "animas" / "hook_cb_test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state" / "pending").mkdir(parents=True)

        callback = MagicMock()

        with patch("core.execution._sdk_hooks._cache_subordinate_paths", return_value=(set(), set(), set())):
            hook_fn = _build_pre_tool_hook(anima_dir, on_task_intercepted=callback, has_subordinates=True)

        return hook_fn, callback, anima_dir

    async def test_task_tool_denied(self, hook) -> None:
        """Task tool should be denied by the hook."""
        input_data = {
            "tool_name": "Task",
            "tool_input": {
                "description": "Do something",
                "prompt": "Full instructions here",
            },
        }

        with patch("core.execution._sdk_hooks._log_tool_use"):
            result = await hook(input_data, "tool-id-1", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") == "deny"
        reason = output.get("permissionDecisionReason", "")
        assert "Task accepted" in reason
        assert "INTERCEPT_OK" in reason

    async def test_task_tool_creates_pending_file(self, hook, tmp_path: Path) -> None:
        """Task intercept should write a JSON file to state/pending/."""
        input_data = {
            "tool_name": "Task",
            "tool_input": {
                "description": "Background work",
                "prompt": "Do the work",
            },
        }

        with patch("core.execution._sdk_hooks._log_tool_use"):
            await hook(input_data, "tool-id-2", {})

        pending_dir = tmp_path / "animas" / "hook_test" / "state" / "pending"
        task_files = list(pending_dir.glob("*.json"))
        assert len(task_files) == 1

    async def test_callback_invoked(self, hook_with_callback) -> None:
        """on_task_intercepted callback should be called after interception."""
        hook_fn, callback, _ = hook_with_callback

        input_data = {
            "tool_name": "Task",
            "tool_input": {"description": "test", "prompt": "test"},
        }

        with patch("core.execution._sdk_hooks._log_tool_use"):
            await hook_fn(input_data, "tool-id-3", {})

        callback.assert_called_once()

    async def test_read_tool_unaffected(self, hook) -> None:
        """Non-Task tools should pass through normally."""
        input_data = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/test.txt"},
        }

        with patch("core.execution._sdk_hooks._log_tool_use"), \
             patch("core.execution._sdk_hooks._check_a1_file_access", return_value=None), \
             patch("core.execution._sdk_hooks._build_output_guard", return_value=None):
            result = await hook(input_data, "tool-id-4", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") != "deny"

    async def test_bash_tool_unaffected(self, hook) -> None:
        """Bash tool should go through normal security checks, not Task intercept."""
        input_data = {
            "tool_name": "Bash",
            "tool_input": {"command": "echo hello"},
        }

        with patch("core.execution._sdk_hooks._log_tool_use"), \
             patch("core.execution._sdk_hooks._check_a1_bash_command", return_value=None), \
             patch("core.execution._sdk_hooks._build_output_guard", return_value=None):
            result = await hook(input_data, "tool-id-5", {})

        output = result.get("hookSpecificOutput", {})
        assert output.get("permissionDecision") != "deny"

    async def test_task_output_for_intercepted_task(self, hook) -> None:
        """TaskOutput for intercepted task IDs returns expected intercept message."""
        task_input = {
            "tool_name": "Task",
            "tool_input": {"description": "bg", "prompt": "run in background"},
        }
        with patch("core.execution._sdk_hooks._log_tool_use"):
            task_result = await hook(task_input, "tool-id-task", {})

        reason = task_result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", "",
        )
        assert "task_id:" in reason
        task_id = reason.split("task_id:", 1)[1].split(")")[0].strip()

        out_input = {
            "tool_name": "TaskOutput",
            "tool_input": {"task_id": task_id, "block": False, "timeout": 5000},
        }
        with patch("core.execution._sdk_hooks._log_tool_use"):
            out_result = await hook(out_input, "tool-id-out", {})

        out = out_result.get("hookSpecificOutput", {})
        assert out.get("permissionDecision") == "deny"
        assert "INTERCEPT_OK" in out.get("permissionDecisionReason", "")
