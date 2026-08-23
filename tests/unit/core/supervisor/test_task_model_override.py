"""Unit tests for per-task model override in PendingTaskExecutor.

Covers override construction (explicit CLI mode / S-mode credential
resolution / missing credential / invalid value) and that
``model_config_override`` is passed to ``run_cycle_streaming`` when the task
sets ``model`` (and stays None when unset).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.schemas import AnimaWorksConfig, CredentialConfig
from core.schemas import ModelConfig
from core.supervisor.pending_executor import PendingTaskExecutor


@pytest.fixture(autouse=True)
def _legacy_completion_semantics():
    with patch(
        "core.supervisor.pending_executor._completion_declaration_required",
        return_value=False,
    ):
        yield


@pytest.fixture(autouse=True)
def _silence_activity():
    with patch("core.memory.activity.ActivityLogger") as mock_activity:
        mock_activity.return_value.log = MagicMock()
        yield


class _LaneAnima:
    """Minimal anima stub with a real model_config and lane helpers."""

    def __init__(self) -> None:
        self.agent = MagicMock(name="chat_agent")
        self.background_agent = MagicMock(name="background_agent")
        self.messenger = MagicMock()
        self._events: dict[str, asyncio.Event] = {}
        self._background_lock = asyncio.Lock()
        self.model_config = ModelConfig(
            model="claude-sonnet-4-6",
            fallback_models=["b:ollama/qwen3:14b"],
            credential="main-cred",
        )

    def _agent_for_lane(self, lane: str):
        return self.background_agent if lane == "background" else self.agent

    def _agent_session_context(self, lane: str):
        return self._background_lock if lane == "background" else asyncio.Lock()

    def _get_interrupt_event(self, name: str) -> asyncio.Event:
        self._events.setdefault(name, asyncio.Event())
        return self._events[name]


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    return PendingTaskExecutor(
        anima=_LaneAnima(),
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _config_with_anthropic() -> AnimaWorksConfig:
    return AnimaWorksConfig(
        credentials={
            "anthropic": CredentialConfig(
                type="api_key",
                api_key="sk-test",
                base_url="https://api.example",
                keys={"api_version": "v1"},
            )
        }
    )


class TestTaskModelOverrideConstruction:
    def test_explicit_cli_mode(self, tmp_path):
        # CLI-auth engines (C/D/G/X) clear (not preserve) base credential
        # fields — each authenticates via its own CLI store.
        executor = _make_executor(tmp_path)
        with patch("core.config.load_config", return_value=_config_with_anthropic()):
            override = executor._task_model_config_override(
                {"task_id": "t1", "model": "c:codex/gpt-5.6-sol"}
            )
        assert override is not None
        assert override.model == "codex/gpt-5.6-sol"
        assert override.resolved_mode == "C"
        assert override.execution_mode == "C"
        # base fallback_models are inherited; credential is cleared for CLI modes
        assert override.fallback_models == ["b:ollama/qwen3:14b"]
        assert override.credential is None

    def test_s_mode_replaces_credential(self, tmp_path):
        # S-mode resolves the family credential and replaces credential fields.
        executor = _make_executor(tmp_path)
        with patch("core.config.load_config", return_value=_config_with_anthropic()):
            override = executor._task_model_config_override(
                {"task_id": "t2", "model": "claude-sonnet-4-6"}
            )
        assert override is not None
        assert override.model == "claude-sonnet-4-6"
        assert override.resolved_mode == "S"
        assert override.credential == "anthropic"
        assert override.api_key == "sk-test"
        assert override.api_base_url == "https://api.example"
        assert override.extra_keys == {"api_version": "v1"}
        assert override.mode_s_auth == "api"
        assert override.fallback_models == ["b:ollama/qwen3:14b"]  # inherited

    def test_missing_credential_returns_none(self, tmp_path):
        # S-mode without a resolvable credential must fall back to None so the
        # task uses the anima default (avoid auth-error → quota misclassification).
        executor = _make_executor(tmp_path)
        cfg = AnimaWorksConfig(credentials={})
        with patch("core.config.load_config", return_value=cfg):
            override = executor._task_model_config_override(
                {"task_id": "t2", "model": "claude-sonnet-4-6"}
            )
        assert override is None

    def test_invalid_value_returns_none(self, tmp_path):
        executor = _make_executor(tmp_path)
        with patch("core.config.load_config", return_value=_config_with_anthropic()):
            override = executor._task_model_config_override(
                {"task_id": "t3", "model": "z:bad-model"}
            )
        assert override is None

    def test_unset_model_returns_none(self, tmp_path):
        executor = _make_executor(tmp_path)
        assert executor._task_model_config_override({"task_id": "t4"}) is None
        assert executor._task_model_config_override({"task_id": "t4", "model": ""}) is None
        assert executor._task_model_config_override({"task_id": "t4", "model": "  "}) is None


class TestTaskModelOverridePassedToCycle:
    async def _run(self, executor, model):
        agent = executor._anima.background_agent

        async def _stream(*args, **kwargs):
            yield {"type": "text_delta", "text": "done"}
            yield {
                "type": "cycle_done",
                "cycle_result": {"summary": "result", "action": "complete"},
            }

        agent.run_cycle_streaming = MagicMock(side_effect=lambda *a, **kw: _stream(*a, **kw))
        agent.reset_reply_tracking = MagicMock()
        agent.reset_read_paths = MagicMock()
        agent.set_interrupt_event = MagicMock()
        agent.set_task_cwd = MagicMock()
        with patch("core.paths.load_prompt", return_value="prompt"), patch(
            "core.config.load_config", return_value=_config_with_anthropic()
        ):
            await executor._run_llm_task(
                {
                    "task_id": "mt-1",
                    "title": "T",
                    "description": "desc",
                    "model": model,
                }
            )
        return agent

    async def test_override_passed_when_model_set(self, tmp_path):
        executor = _make_executor(tmp_path)
        agent = await self._run(executor, "c:codex/gpt-5.6-sol")
        assert agent.run_cycle_streaming.call_count >= 1
        kw = agent.run_cycle_streaming.call_args_list[0].kwargs
        assert kw.get("model_config_override") is not None
        assert kw["model_config_override"].model == "codex/gpt-5.6-sol"

    async def test_override_none_when_unset(self, tmp_path):
        executor = _make_executor(tmp_path)
        agent = await self._run(executor, None)
        kw = agent.run_cycle_streaming.call_args_list[0].kwargs
        assert kw.get("model_config_override") is None


class TestModelConfigHelper:
    """Direct tests of the shared build_model_override_config helper."""

    def test_shared_helper_cli_mode(self):
        from core.config.model_config import build_model_override_config

        base = ModelConfig(model="claude-sonnet-4-6", credential="main-cred")
        cfg = _config_with_anthropic()
        cand = build_model_override_config(base, "c", "codex/gpt-5.6-sol", cfg)
        assert cand is not None
        assert cand.model == "codex/gpt-5.6-sol"
        assert cand.credential is None  # CLI clears base credential
        assert cand.fallback_models == []

    def test_shared_helper_s_mode_replaces_credential(self):
        from core.config.model_config import build_model_override_config

        base = ModelConfig(model="claude-sonnet-4-6", credential="main-cred")
        cfg = _config_with_anthropic()
        cand = build_model_override_config(base, "s", "claude-sonnet-4-6", cfg)
        assert cand is not None
        assert cand.credential == "anthropic"
        assert cand.api_key == "sk-test"
        assert cand.mode_s_auth == "api"

    def test_shared_helper_missing_credential(self):
        from core.config.model_config import build_model_override_config

        base = ModelConfig(model="claude-sonnet-4-6", credential="main-cred")
        cfg = AnimaWorksConfig(credentials={})
        cand = build_model_override_config(base, "s", "claude-sonnet-4-6", cfg)
        assert cand is None
