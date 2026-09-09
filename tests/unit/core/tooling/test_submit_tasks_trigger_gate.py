"""submit_tasks availability per trigger.

Heartbeat must expose submit_tasks: the harness never re-executes a ledger
``pending`` task by itself, so the anima re-submits its own pending work
during heartbeat (2026-09-03). TaskExec / chat / cron stay closed.
"""

from core.tooling.schemas.builder import build_unified_tool_list, submit_tasks_enabled_for_trigger


def test_heartbeat_enables_submit_tasks() -> None:
    assert submit_tasks_enabled_for_trigger("heartbeat")
    assert submit_tasks_enabled_for_trigger("background")
    assert submit_tasks_enabled_for_trigger("background:x")


def test_non_authoring_triggers_stay_closed() -> None:
    for trig in ("", "chat", "inbox:rin", "task:abc", "cron:daily", "consolidation:daily"):
        assert not submit_tasks_enabled_for_trigger(trig), trig


def test_heartbeat_tool_list_contains_submit_tasks() -> None:
    names = {t["name"] for t in build_unified_tool_list(trigger="heartbeat")}
    assert "submit_tasks" in names
    assert "update_task" in names


def test_task_session_tool_list_excludes_submit_tasks() -> None:
    names = {t["name"] for t in build_unified_tool_list(trigger="task:abc")}
    assert "submit_tasks" not in names
