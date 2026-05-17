from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Goal completion judge.

The judge receives a deliberately narrow prompt: objective, success criteria,
latest task result summaries, verification output, and iteration counters.
It never receives hidden tool instructions or a replacement system prompt for
the next TaskExec task.
"""

import inspect
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from core.goals.models import GoalJudgment, GoalState

logger = logging.getLogger("animaworks.goals.judge")

JudgeFn = Callable[
    [str, dict[str, Any]], GoalJudgment | dict[str, Any] | str | Awaitable[GoalJudgment | dict[str, Any] | str]
]

_VALID_VERDICTS: set[str] = {"done", "continue", "blocked"}
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_goal_judgment(
    raw: str | dict[str, Any] | GoalJudgment,
    *,
    goal_id: str,
    task_id: str,
    iteration: int,
    verification_output: str = "",
) -> GoalJudgment:
    """Parse a judge response, failing open to ``continue`` on malformed output."""
    if isinstance(raw, GoalJudgment):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        text = str(raw or "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(text)
            if not match:
                return _fail_open(goal_id, task_id, iteration, "judge_parse_failed", text, verification_output)
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return _fail_open(goal_id, task_id, iteration, "judge_parse_failed", text, verification_output)

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        return _fail_open(
            goal_id,
            task_id,
            iteration,
            f"judge_invalid_verdict:{verdict or 'missing'}",
            json.dumps(data, ensure_ascii=False),
            verification_output,
        )

    return GoalJudgment(
        goal_id=str(data.get("goal_id") or goal_id),
        task_id=str(data.get("task_id") or task_id),
        verdict=verdict,  # type: ignore[arg-type]
        reason=str(data.get("reason") or ""),
        continuation_prompt=str(data.get("continuation_prompt") or ""),
        verification_output=str(data.get("verification_output") or verification_output),
        raw_response=str(raw if not isinstance(raw, dict) else json.dumps(raw, ensure_ascii=False)),
        failed_open=bool(data.get("failed_open", False)),
        iteration=int(data.get("iteration") or iteration),
    )


class GoalJudge:
    """Judge whether a persistent goal is done, should continue, or is blocked."""

    def __init__(
        self,
        anima_dir: Path,
        *,
        judge_fn: JudgeFn | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.anima_dir = anima_dir
        self._judge_fn = judge_fn
        self._timeout_s = timeout_s

    async def judge(
        self,
        state: GoalState,
        *,
        task_id: str,
        result_summaries: list[str],
        verification_output: str = "",
    ) -> GoalJudgment:
        """Return a judge verdict, failing open to ``continue`` on errors."""
        iteration = state.iteration_count + 1
        payload = _judge_payload(
            state,
            result_summaries=result_summaries,
            verification_output=verification_output,
            iteration=iteration,
        )
        prompt = _render_prompt(payload)

        try:
            if self._judge_fn is not None:
                raw = self._judge_fn(prompt, payload)
                if inspect.isawaitable(raw):
                    raw = await raw
            else:
                raw = await self._call_background_model(state, prompt)
            return parse_goal_judgment(
                raw,
                goal_id=state.goal_id,
                task_id=task_id,
                iteration=iteration,
                verification_output=verification_output,
            )
        except Exception as exc:
            logger.warning("Goal judge failed open: goal=%s task=%s", state.goal_id, task_id, exc_info=True)
            return _fail_open(
                state.goal_id,
                task_id,
                iteration,
                f"judge_error:{type(exc).__name__}:{str(exc)[:160]}",
                "",
                verification_output,
            )

    async def _call_background_model(self, state: GoalState, prompt: str) -> str:
        import litellm

        from core.config.model_config import load_model_config

        config = load_model_config(self.anima_dir)
        model = state.judge_model or config.background_model or config.model
        kwargs: dict[str, Any] = {
            "model": _litellm_model_name(model),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You judge whether a persistent operational goal is complete. "
                        "Return only JSON with verdict, reason, and optional continuation_prompt."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "timeout": config.llm_timeout or self._timeout_s,
            "num_retries": 0,
        }
        api_key = config.api_key or os.environ.get(config.api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key
        if config.api_base_url:
            kwargs["api_base"] = config.api_base_url

        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return content or ""


def _judge_payload(
    state: GoalState,
    *,
    result_summaries: list[str],
    verification_output: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "goal_id": state.goal_id,
        "objective": state.objective,
        "success_criteria": state.success_criteria,
        "latest_result_summaries": [str(item)[:2000] for item in result_summaries[-5:]],
        "verification_output": verification_output[:2000],
        "iteration": iteration,
        "max_iterations": state.max_iterations,
    }


def _render_prompt(payload: dict[str, Any]) -> str:
    return (
        "Judge this goal using only the supplied evidence.\n"
        "Allowed verdicts: done, continue, blocked.\n"
        'JSON schema: {"verdict":"done|continue|blocked","reason":"...",'
        '"continuation_prompt":"only when verdict is continue"}\n\n'
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _fail_open(
    goal_id: str,
    task_id: str,
    iteration: int,
    reason: str,
    raw_response: str,
    verification_output: str,
) -> GoalJudgment:
    return GoalJudgment(
        goal_id=goal_id,
        task_id=task_id,
        verdict="continue",
        reason=reason,
        raw_response=raw_response,
        failed_open=True,
        iteration=iteration,
        verification_output=verification_output,
    )


def _litellm_model_name(model: str) -> str:
    if model.startswith("nanogpt/"):
        return "openai/" + model[len("nanogpt/") :]
    return model
