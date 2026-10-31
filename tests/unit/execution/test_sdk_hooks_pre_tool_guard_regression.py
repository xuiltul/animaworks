from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Mode S PreToolUse guard decisions and tool logging."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def hook_builder(tmp_path: Path):
    pytest.importorskip("claude_agent_sdk.types")
    from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook

    anima_dir = tmp_path / "animas" / "guard-test"
    anima_dir.mkdir(parents=True)

    def build(*, session_stats: dict | None = None):
        with patch(
            "core.execution.engines.claude._sdk_hooks._cache_subordinate_paths",
            return_value=([], [], [], [], []),
        ):
            return _build_pre_tool_hook(anima_dir, session_stats=session_stats)

    return anima_dir, build


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "check_name", "check_result"),
    [
        ("Write", {"file_path": "/outside/secret.md"}, "_check_a1_file_access", "write denied"),
        ("Edit", {"file_path": "/outside/secret.md"}, "_check_a1_file_access", "edit denied"),
        ("Read", {"file_path": "/outside/secret.md"}, "_check_a1_file_access", "read denied"),
        ("Bash", {"command": "dangerous-command"}, "_check_a1_bash_command", "bash denied"),
    ],
)
@pytest.mark.asyncio
async def test_security_guard_denials_are_logged_once_with_original_reason(
    hook_builder,
    tool_name: str,
    tool_input: dict,
    check_name: str,
    check_result: str,
) -> None:
    anima_dir, build_hook = hook_builder
    hook = build_hook()
    guard_path = f"core.execution.engines.claude._sdk_hooks.{check_name}"

    with (
        patch(guard_path, return_value=check_result) as guard,
        patch("core.execution.engines.claude._sdk_hooks._log_tool_use") as log_tool_use,
    ):
        result = await hook({"tool_name": tool_name, "tool_input": tool_input}, "tool-guard-1", MagicMock())

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"] == check_result
    log_tool_use.assert_called_once_with(
        anima_dir,
        tool_name,
        tool_input,
        tool_use_id="tool-guard-1",
        blocked=True,
        block_reason=check_result,
    )
    if tool_name in ("Write", "Edit", "Read"):
        assert guard.call_args.args[0] == tool_input["file_path"]
        assert guard.call_args.kwargs["write"] is (tool_name != "Read")
    else:
        guard.assert_called_once_with(
            "dangerous-command",
            anima_dir,
            superuser=False,
            trigger="unknown",
        )


@pytest.mark.asyncio
async def test_allowed_tool_is_logged_once_without_block_metadata(hook_builder) -> None:
    anima_dir, build_hook = hook_builder
    hook = build_hook()
    tool_input = {"file_path": str(anima_dir / "IDENTITY.md")}

    with (
        patch("core.execution.engines.claude._sdk_hooks._check_a1_file_access", return_value=None),
        patch("core.execution.engines.claude._sdk_hooks._build_output_guard", return_value=None),
        patch("core.execution.engines.claude._sdk_hooks._log_tool_use") as log_tool_use,
    ):
        result = await hook({"tool_name": "Read", "tool_input": tool_input}, "tool-allow-1", MagicMock())

    assert result.get("hookSpecificOutput") is None
    log_tool_use.assert_called_once_with(
        anima_dir,
        "Read",
        tool_input,
        tool_use_id="tool-allow-1",
    )


@pytest.mark.asyncio
async def test_context_observation_and_final_decision_keep_their_log_records(hook_builder) -> None:
    anima_dir, build_hook = hook_builder
    stats = {
        "tool_call_count": 0,
        "total_result_bytes": 0,
        "system_prompt_tokens": 192_000,
        "user_prompt_tokens": 0,
        "trigger": "chat",
    }
    hook = build_hook(session_stats=stats)
    tool_input = {"file_path": str(anima_dir / "IDENTITY.md")}

    with (
        patch("core.execution.engines.claude._sdk_hooks._check_a1_file_access", return_value=None),
        patch("core.execution.engines.claude._sdk_hooks._build_output_guard", return_value=None),
        patch("core.execution.engines.claude._sdk_hooks._log_tool_use") as log_tool_use,
    ):
        await hook(
            {"tool_name": "Read", "tool_input": tool_input},
            "tool-context-1",
            MagicMock(),
        )

    assert log_tool_use.call_count == 2
    assert log_tool_use.call_args_list[0].kwargs == {
        "tool_use_id": "tool-context-1",
        "blocked": False,
        "block_reason": "context_observation: estimated 192000 tokens, remaining 8000 — SDK managing",
    }
    assert log_tool_use.call_args_list[1].kwargs == {"tool_use_id": "tool-context-1"}


@pytest.mark.asyncio
async def test_intercepted_submit_tasks_keeps_successful_tool_use_logging(hook_builder) -> None:
    anima_dir, build_hook = hook_builder
    stats = {
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
    hook = build_hook(session_stats=stats)
    tool_input = {"batch_id": "batch-1", "tasks": []}
    success = json.dumps({"status": "submitted", "task_ids": ["task-1"]})

    with (
        patch(
            "core.tooling.policy.submit_tasks.submit_tasks",
            return_value=success,
        ),
        patch("core.execution.engines.claude._sdk_hooks._log_tool_use") as log_tool_use,
    ):
        result = await hook({"tool_name": "submit_tasks", "tool_input": tool_input}, "tool-submit-1", MagicMock())

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "task-1" in output["permissionDecisionReason"]
    log_tool_use.assert_called_once_with(
        anima_dir,
        "submit_tasks",
        tool_input,
        tool_use_id="tool-submit-1",
        blocked=False,
    )
