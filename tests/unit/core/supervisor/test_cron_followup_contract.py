from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.runtime.task_runner import execute_cron_contract, execute_cron_followup_contract
from core.schemas import CronTask, CycleResult


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout,stderr,exit_code,pattern,enabled,expected",
    [
        ("OK", "", 0, r"\AOK\Z", True, False),
        ("  OK\n", "", 0, r"\AOK\Z", True, False),
        ("OK\nALERT", "", 0, r"\AOK\Z", True, True),
        ("new-format", "", 0, r"\AOK\Z", True, True),
        ("OK", "sensor failed", 1, r"\AOK\Z", True, True),
        ("", "TimeoutError", 1, r"\AOK\Z", True, True),
        ("", "", 1, r"\AOK\Z", True, True),
        ("OK", "warning", 0, r"\AOK\Z", True, True),
        ("OK", "", 0, "[invalid", True, True),
        ("", "", 0, None, True, False),
        ("ALERT", "", 0, None, False, False),
        ("", "failure", 1, None, False, False),
    ],
)
async def test_execute_cron_contract_followup(
    stdout: str,
    stderr: str,
    exit_code: int,
    pattern: str | None,
    enabled: bool,
    expected: bool,
) -> None:
    anima = MagicMock()
    result = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
    anima.run_cron_command = AsyncMock(return_value=result)
    anima.run_cron_task = AsyncMock(return_value=CycleResult(trigger="cron", action="responded", summary="reviewed"))
    task = CronTask(
        name="sensor",
        schedule="*/10 * * * *",
        type="command",
        command="sensor",
        skip_pattern=pattern,
        trigger_heartbeat=enabled,
    )

    outcome = await execute_cron_contract(anima, task)

    assert outcome["success"] == (exit_code == 0)
    assert outcome["result"] == result
    assert anima.run_cron_command.await_count == 1
    assert anima.run_cron_task.await_count == int(expected)
    if expected and stderr:
        assert stderr in anima.run_cron_task.call_args.kwargs["command_output"]


@pytest.mark.asyncio
async def test_execute_cron_followup_contract_does_not_run_the_command() -> None:
    anima = MagicMock()
    anima.run_cron_command = AsyncMock()
    result = CycleResult(trigger="cron:sensor", action="completed", summary="reviewed")
    anima.run_cron_task = AsyncMock(return_value=result)
    task = CronTask(
        name="sensor",
        schedule="*/10 * * * *",
        type="command",
        command="sensor",
        description="Review sensor output",
        skills=["monitoring"],
    )

    outcome = await execute_cron_followup_contract(anima, task, '{"exit_code": 1}')

    anima.run_cron_command.assert_not_awaited()
    anima.run_cron_task.assert_awaited_once_with(
        "sensor",
        "Review sensor output",
        command_output='{"exit_code": 1}',
        skills=["monitoring"],
    )
    assert outcome == {
        "task_type": "command_followup",
        "result": result.model_dump(mode="json"),
        "success": True,
        "usage": result.usage,
    }
