from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from core.goals import GoalJudgment, GoalManager
from core.goals.judge import GoalJudge, parse_goal_judgment
from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility
from core.taskboard.store import TaskBoardStore
from core.tooling.handler import ToolHandler
from core.tooling.schemas import _goal_tools, build_tool_list


def _anima_dir(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    anima_dir = data_dir / "animas" / "alice"
    for dirname in ("state", "episodes", "knowledge", "procedures", "skills"):
        (anima_dir / dirname).mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    return anima_dir


def test_goal_manager_replays_transitions_and_max_iteration_block(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)

    state = manager.set_goal(
        title="Ship report",
        objective="Create a weekly report",
        success_criteria=["Report exists", "Summary is sent"],
        max_iterations=1,
    )
    manager.pause(state.goal_id, reason="wait")
    manager.resume(state.goal_id, reason="continue")
    updated = manager.record_judgment(
        state.goal_id,
        GoalJudgment(
            goal_id=state.goal_id,
            task_id="t1",
            verdict="continue",
            reason="not enough evidence",
            iteration=1,
        ),
        result_summary="draft only",
    )

    assert updated is not None
    assert updated.status == "blocked"
    assert updated.iteration_count == 1
    assert "max_iterations_reached:1/1" in updated.blocked_reason

    replayed = GoalManager(anima_dir).get_goal(state.goal_id)
    assert replayed is not None
    assert replayed.status == "blocked"
    assert replayed.last_task_id == "t1"


def test_enqueue_continuation_creates_normal_task_and_avoids_duplicates(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    state = manager.set_goal(
        objective="Finish the deployment",
        success_criteria=["Deployment verified"],
        max_iterations=3,
    )
    manager.record_judgment(
        state.goal_id,
        GoalJudgment(
            goal_id=state.goal_id,
            task_id="seed",
            verdict="continue",
            reason="needs verification",
            continuation_prompt="Run verification and fix failures.",
            iteration=1,
        ),
        result_summary="build prepared",
    )
    judgment = manager.get_goal(state.goal_id).last_judgment  # type: ignore[union-attr]

    entry = manager.enqueue_continuation(
        state.goal_id,
        judgment,
        source_task_desc={"working_directory": "/tmp/work", "reply_to": "alice"},
        result_summary="build prepared",
    )
    duplicate = manager.enqueue_continuation(
        state.goal_id,
        judgment,
        source_task_desc={},
        result_summary="build prepared",
    )

    assert entry is not None
    assert duplicate is not None
    assert duplicate.task_id == entry.task_id
    assert entry.meta["goal_id"] == state.goal_id
    assert entry.meta["executor"] == "taskexec"
    pending_json = anima_dir / "state" / "pending" / f"{entry.task_id}.json"
    assert pending_json.exists()
    assert json.loads(pending_json.read_text(encoding="utf-8"))["task_type"] == "llm"


def test_human_task_defers_continuation(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    state = manager.set_goal(objective="Do background work", max_iterations=3)
    TaskQueueManager(anima_dir).add_task(
        source="human",
        original_instruction="urgent user task",
        assignee="alice",
        summary="urgent",
        deadline="1h",
    )

    result = manager.enqueue_continuation(
        state.goal_id,
        GoalJudgment(goal_id=state.goal_id, verdict="continue", iteration=1),
        source_task_desc={},
        result_summary="needs more",
    )

    assert result is None


def test_get_active_goal_task_ignores_archived_taskboard_rows(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="anima",
        original_instruction="continue",
        assignee="alice",
        summary="continue",
        status="in_progress",
        meta={"goal_id": "goal-1"},
    )
    assert queue.get_active_goal_task("goal-1").task_id == entry.task_id

    TaskBoardStore().upsert_metadata(
        anima_name="alice",
        task_id=entry.task_id,
        actor="test",
        visibility=AttentionVisibility.ARCHIVED,
    )

    assert queue.get_active_goal_task("goal-1") is None


def test_goal_tool_schema_and_handler(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    tools = build_tool_list(include_goal_tools=True)
    assert _goal_tools()[0]["name"] == "goal"
    assert "goal" in {tool["name"] for tool in tools}

    handler = ToolHandler(anima_dir, memory=object())  # type: ignore[arg-type]
    result = json.loads(
        handler.handle(
            "goal",
            {
                "action": "set",
                "objective": "Prepare a release",
                "success_criteria": ["tag exists"],
                "max_iterations": 2,
            },
        )
    )

    assert result["status"] == "active"
    assert result["objective"] == "Prepare a release"
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction="release",
        assignee="alice",
        summary="Release task",
        task_id="release-task",
        status="pending",
        meta={"executor": "taskexec", "goal_id": result["goal_id"], "goal_iteration": 1},
    )
    all_status = json.loads(handler.handle("goal", {"action": "status"}))
    assert [item["goal_id"] for item in all_status] == [result["goal_id"]]
    assert all_status[0]["related_tasks"][0]["task_id"] == "release-task"
    assert all_status[0]["related_tasks"][0]["status"] == "pending"

    status = json.loads(handler.handle("goal", {"action": "status", "goal_id": result["goal_id"]}))
    assert status["goal_id"] == result["goal_id"]
    assert status["related_tasks"][0]["summary"] == "Release task"

    paused = json.loads(handler.handle("goal", {"action": "pause", "goal_id": result["goal_id"], "reason": "wait"}))
    assert paused["status"] == "paused"

    resumed = json.loads(handler.handle("goal", {"action": "resume", "goal_id": result["goal_id"]}))
    assert resumed["status"] == "active"

    judged = json.loads(
        handler.handle(
            "goal",
            {
                "action": "judge",
                "goal_id": result["goal_id"],
                "task_id": "task-1",
                "result_summary": "tag exists",
                "verdict": "done",
                "reason": "criteria satisfied",
            },
        )
    )
    assert judged["status"] == "done"
    assert judged["last_judgment"]["verdict"] == "done"

    cleared = json.loads(handler.handle("goal", {"action": "clear", "goal_id": result["goal_id"]}))
    assert cleared["status"] == "cleared"

    invalid = json.loads(handler.handle("goal", {"action": "unknown"}))
    assert invalid["status"] == "error"

    missing = json.loads(handler.handle("goal", {"action": "status", "goal_id": "missing"}))
    assert missing["status"] == "error"

    no_current = json.loads(handler.handle("goal", {"action": "pause"}))
    assert no_current["status"] == "error"

    bad_set = json.loads(handler.handle("goal", {"action": "set", "objective": "x", "max_iterations": "bad"}))
    assert bad_set["status"] == "error"


def test_goal_tool_auto_judge_path_uses_sync_runner(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    handler = ToolHandler(anima_dir, memory=object())  # type: ignore[arg-type]
    goal = json.loads(
        handler.handle(
            "goal",
            {
                "action": "set",
                "objective": "Ship docs",
                "skills": "missing-skill, other-missing",
            },
        )
    )
    assert goal["skill_refs"] == []
    assert {item["reason"] for item in goal["skill_rejections"]} == {"not_found"}

    class FakeJudge:
        def __init__(self, _anima_dir):
            pass

        async def judge(self, state, *, task_id, result_summaries, verification_output=""):
            assert state.goal_id == goal["goal_id"]
            assert result_summaries == ["docs shipped"]
            return GoalJudgment(goal_id=state.goal_id, task_id=task_id, verdict="done", iteration=1)

    monkeypatch.setattr("core.tooling.handler_goals.GoalJudge", FakeJudge)
    judged = json.loads(
        handler.handle(
            "goal",
            {
                "action": "judge",
                "goal_id": goal["goal_id"],
                "task_id": "t1",
                "result_summary": "docs shipped",
            },
        )
    )

    assert judged["status"] == "done"


def test_goal_judgment_parser_fail_open_and_valid_json() -> None:
    parsed = parse_goal_judgment(
        '{"verdict":"blocked","reason":"missing credentials"}',
        goal_id="g1",
        task_id="t1",
        iteration=2,
    )
    assert parsed.verdict == "blocked"
    assert parsed.reason == "missing credentials"

    failed_open = parse_goal_judgment("not-json", goal_id="g1", task_id="t1", iteration=2)
    assert failed_open.verdict == "continue"
    assert failed_open.failed_open is True
    assert failed_open.reason == "judge_parse_failed"


def test_fail_open_judgment_records_activity(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    state = manager.set_goal(objective="Recover flaky judge")

    manager.record_judgment(
        state.goal_id,
        GoalJudgment(
            goal_id=state.goal_id,
            task_id="t1",
            verdict="continue",
            reason="judge_error:TimeoutError",
            failed_open=True,
            iteration=1,
        ),
        result_summary="judge timed out",
    )

    activity_text = "\n".join(path.read_text(encoding="utf-8") for path in (anima_dir / "activity_log").glob("*.jsonl"))
    assert "goal_judge_failed_open" in activity_text
    assert "judge_error:TimeoutError" in activity_text


async def test_goal_judge_uses_injected_callable(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    state = GoalManager(anima_dir).set_goal(objective="Verify release", max_iterations=2)

    async def fake_judge(prompt, payload):
        assert "Verify release" in prompt
        assert payload["objective"] == "Verify release"
        return {"verdict": "continue", "reason": "needs more proof"}

    judgment = await GoalJudge(anima_dir, judge_fn=fake_judge).judge(
        state,
        task_id="t1",
        result_summaries=["draft"],
        verification_output="tests pending",
    )

    assert judgment.verdict == "continue"
    assert judgment.reason == "needs more proof"
    assert judgment.verification_output == "tests pending"


async def test_goal_judge_background_model_call_is_narrow_and_parseable(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    state = GoalManager(anima_dir).set_goal(
        objective="Verify release",
        success_criteria=["tests pass"],
        judge_model="nanogpt/custom-judge",
    )
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"verdict":"done","reason":"tests pass"}')
                )
            ]
        )

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=fake_acompletion))
    monkeypatch.setattr(
        "core.config.model_config.load_model_config",
        lambda _anima_dir: SimpleNamespace(
            background_model=None,
            model="openai/fallback",
            api_key="",
            api_key_env="MISSING_KEY",
            api_base_url="http://llm.local",
            llm_timeout=12,
        ),
    )

    judgment = await GoalJudge(anima_dir).judge(
        state,
        task_id="t1",
        result_summaries=["all tests passed"],
        verification_output="pytest green",
    )

    assert judgment.verdict == "done"
    assert captured["model"] == "openai/custom-judge"
    assert captured["api_base"] == "http://llm.local"
    user_payload = captured["messages"][1]["content"]
    assert "Verify release" in user_payload
    assert "all tests passed" in user_payload
    assert "pytest green" in user_payload
