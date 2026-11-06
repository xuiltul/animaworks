from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.helper_models import ResolvedHelperModel
from core.config.schemas import AnimaWorksConfig, HelperModelFallback


def _resolved(role: str, model: str = "openai/helper") -> ResolvedHelperModel:
    return ResolvedHelperModel(
        model=model,
        credential="openai",
        fallbacks=[HelperModelFallback(model="openai/fallback", credential="gpu40-direct")],
        allow_agent_sdk_fallback=False,
        max_output_tokens=321,
        source=f"config.helper_models.{role}",
    )


def test_fact_extraction_compatibility_facade_resolves_registry_role(tmp_path: Path) -> None:
    from core.memory.facts.config import _resolve_extraction_config

    config = AnimaWorksConfig()
    helper = _resolved("fact_extraction")
    with (
        patch("core.config.load_config", return_value=config),
        patch("core.memory.facts.config.resolve_helper_model", return_value=helper) as resolver,
    ):
        model, _extra, _locale, _timeout, credential = _resolve_extraction_config(tmp_path / "alice")

    assert (model, credential) == (helper.model, helper.credential)
    resolver.assert_called_once_with("fact_extraction", tmp_path / "alice", config=config)


def test_fact_reconcile_compatibility_facade_resolves_registry_role(tmp_path: Path) -> None:
    from core.memory.facts.invalidation_llm import _resolve_reconcile_llm_config

    config = AnimaWorksConfig()
    helper = _resolved("fact_reconcile")
    with (
        patch("core.config.load_config", return_value=config),
        patch("core.memory.facts.invalidation_llm.resolve_helper_model", return_value=helper) as resolver,
    ):
        model, _extra, _timeout, credential = _resolve_reconcile_llm_config(tmp_path / "alice")

    assert (model, credential) == (helper.model, helper.credential)
    resolver.assert_called_once_with("fact_reconcile", tmp_path / "alice", config=config)


def test_live_extraction_options_resolve_fact_extraction_role(tmp_path: Path) -> None:
    from core.memory.facts.live import _load_options

    config = AnimaWorksConfig()
    helper = _resolved("fact_extraction")
    with (
        patch("core.config.load_config", return_value=config),
        patch("core.memory.facts.live.resolve_helper_model", return_value=helper) as resolver,
    ):
        options = _load_options(tmp_path / "mei")

    assert options.model == helper.model
    assert options.credential == helper.credential
    assert options.helper_model == helper
    resolver.assert_called_once_with("fact_extraction", tmp_path / "mei", config=config)


def test_episode_summary_candidates_resolve_role_and_ignore_caller_model(tmp_path: Path) -> None:
    from core.anima.lifecycle import _episode_summary_model_configs
    from core.schemas import ModelConfig

    config = AnimaWorksConfig()
    helper = ResolvedHelperModel(
        model="openai/episode-summary",
        credential="openai",
        fallbacks=[],
        allow_agent_sdk_fallback=False,
        max_output_tokens=None,
        source="config.helper_models.episode_summary",
    )
    with patch("core.config.helper_models.resolve_helper_model", return_value=helper) as resolver:
        configs = _episode_summary_model_configs(
            ModelConfig(model="claude-opus-5-5"),
            "claude-opus-5-5",
            config,
            anima_dir=tmp_path / "mei",
        )

    assert [(item.model, item.credential) for item in configs] == [(helper.model, helper.credential)]
    resolver.assert_called_once_with("episode_summary", tmp_path / "mei", config=config)


@pytest.mark.asyncio
async def test_helper_one_shot_uses_resolved_fallbacks_and_policy() -> None:
    from core.llm.helper_completion import one_shot_helper_completion

    helper = _resolved("distillation")
    completion = AsyncMock(side_effect=[None, "fallback response"])
    with (
        patch("core.llm.helper_completion.resolve_helper_model", return_value=helper) as resolver,
        patch("core.llm.oneshot.one_shot_completion", completion),
    ):
        result = await one_shot_helper_completion("prompt", role="distillation", max_tokens=2048)

    assert result == "fallback response"
    resolver.assert_called_once_with("distillation", None)
    assert [call.kwargs["model"] for call in completion.await_args_list] == [helper.model, "openai/fallback"]
    assert all(call.kwargs["max_tokens"] == 321 for call in completion.await_args_list)
    assert all(call.kwargs["allow_agent_sdk_fallback"] is False for call in completion.await_args_list)
    assert completion.await_args_list[1].kwargs["credential"] == "gpu40-direct"


@pytest.mark.asyncio
async def test_conversation_helpers_select_distinct_registered_roles(tmp_path: Path) -> None:
    from core.memory.conversation.compression import _call_compression_llm, _call_llm

    completion = AsyncMock(return_value="summary")
    with patch("core.llm.helper_completion.one_shot_helper_completion", completion):
        assert await _call_llm("system", "session", anima_dir=tmp_path / "mei") == "summary"
        assert await _call_compression_llm("old", "new", anima_dir=tmp_path / "mei") == "summary"

    assert completion.await_args_list[0].kwargs["role"] == "episode_summary"
    assert completion.await_args_list[1].kwargs["role"] == "conversation_compression"
    assert all(call.kwargs["anima_dir"] == tmp_path / "mei" for call in completion.await_args_list)


@pytest.mark.asyncio
async def test_distillation_and_reconsolidation_calls_pass_their_helper_roles(tmp_path: Path) -> None:
    from core.memory.maintenance.distillation import ProceduralDistiller
    from core.memory.maintenance.reconsolidation import ReconsolidationEngine

    distiller = ProceduralDistiller(tmp_path / "alice", "alice")
    distiller._load_activity_entries = lambda **_kwargs: [{"type": "tool_use"}]  # type: ignore[method-assign]
    distiller._cluster_activities = lambda *_args, **_kwargs: [[]]  # type: ignore[method-assign]
    distiller._format_clusters_for_prompt = lambda _clusters: "clusters"  # type: ignore[method-assign]
    distiller._load_existing_procedures = lambda: ""  # type: ignore[method-assign]

    helper_completion = AsyncMock(return_value="[]")
    with patch("core.llm.helper_completion.one_shot_helper_completion", helper_completion):
        result = await distiller.weekly_pattern_distill()
    assert result["patterns_detected"] == 1
    assert helper_completion.await_args.kwargs["role"] == "distillation"

    engine = ReconsolidationEngine(
        tmp_path / "alice",
        "alice",
        memory_manager=MagicMock(),
        activity_logger=MagicMock(),
    )
    engine._archive_version = MagicMock()  # type: ignore[method-assign]
    with (
        patch("core.memory.maintenance.reconsolidation.load_prompt", return_value="revise prompt"),
        patch(
            "core.llm.helper_completion.one_shot_helper_completion",
            new=AsyncMock(return_value="revised procedure"),
        ) as helper_call,
    ):
        revised = await engine._revise_procedure("procedure", {}, "")

    assert revised == "revised procedure"
    assert helper_call.await_args.kwargs["role"] == "reconsolidation"
    assert helper_call.await_args.kwargs["anima_dir"] == tmp_path / "alice"


@pytest.mark.asyncio
async def test_project_and_weekly_consolidation_resolve_distinct_roles(tmp_path: Path) -> None:
    from core.anima.lifecycle import LifecycleMixin
    from core.schemas import CycleResult, ModelConfig

    config = AnimaWorksConfig.model_validate(
        {
            "credentials": {"openai": {"api_key": "key"}},
            "consolidation": {"llm_model": "openai/legacy"},
        }
    )
    anima_dir = tmp_path / "animas" / "mei"
    owner = SimpleNamespace(
        name="mei",
        anima_dir=anima_dir,
        memory=MagicMock(),
        _agent_session_lock=asyncio.Lock(),
        _run_autonomous_skill_learning=lambda: None,
    )
    owner.memory.read_model_config.return_value = ModelConfig(model="claude-opus-5-5")
    agent = MagicMock()
    agent.run_cycle = AsyncMock(return_value=CycleResult(trigger="helper", action="completed", summary="ok"))
    owner.agent = agent
    engine = MagicMock()
    engine.project = "project-a"
    engine._collect_recent_episodes.return_value = [{"date": "2026-10-01", "time": "09:00", "content": "update"}]

    project_helper = _resolved("project_consolidation", "openai/project-helper")
    weekly_helper = _resolved("weekly_consolidation", "openai/weekly-helper")
    with (
        patch("core.config.load_config", return_value=config),
        patch(
            "core.config.helper_models.resolve_helper_model",
            side_effect=[project_helper, weekly_helper],
        ) as resolver,
        patch("core.anima.lifecycle.load_prompt", return_value="prompt"),
        patch("core.memory.maintenance.forgetting.ForgettingEngine") as forgetter,
        patch("core.memory.maintenance.hygiene.scan_memory_hygiene", return_value={}),
    ):
        forgetter.return_value.list_forgetting_candidates.return_value = []
        await LifecycleMixin._run_daily_consolidation(owner, engine)
        project_engine = MagicMock()
        project_engine.project = None
        project_engine._find_merge_candidates.return_value = []
        project_engine._find_conflicting_fact_candidates.return_value = []
        await LifecycleMixin._run_weekly_consolidation(owner, project_engine)

    assert [call.args[0] for call in resolver.call_args_list] == [
        "project_consolidation",
        "weekly_consolidation",
    ]
    assert agent.run_cycle.await_args_list[0].kwargs["model_config_override"].model == project_helper.model
    assert agent.run_cycle.await_args_list[1].kwargs["model_config_override"].model == weekly_helper.model


@pytest.mark.asyncio
async def test_asset_summary_uses_helper_registry(tmp_path: Path) -> None:
    from core.anima.asset_reconciler import _synthesize_prompt_via_llm

    helper = _resolved("asset_reconcile")
    completion = AsyncMock(return_value="asset prompt")
    with (
        patch("core.config.helper_models.resolve_helper_model", return_value=helper) as resolver,
        patch("core.llm.helper_completion.one_shot_helper_completion", completion),
        patch("core.anima.asset_reconciler.load_prompt", return_value="system prompt"),
    ):
        assert await _synthesize_prompt_via_llm(tmp_path / "mei", "identity") == "asset prompt"

    resolver.assert_called_once_with("asset_reconcile", tmp_path / "mei")
    assert completion.await_args.kwargs["role"] == "asset_reconcile"
