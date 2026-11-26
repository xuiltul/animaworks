from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Self roles run on the Anima's own model; unconfigured helper roles fall back to it."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.helper_models import (
    HELPER_MODEL_ROLES,
    SELF_HELPER_ROLES,
    SELF_SOURCE,
    SELF_UNCONFIGURED_HELPER_SOURCE,
    ResolvedHelperModel,
    resolve_helper_model,
    validate_helper_model_credentials,
)
from core.config.schemas import DEFAULT_CONSOLIDATION_MODEL, AnimaWorksConfig, HelperModelFallback
from core.schemas import CycleResult, ModelConfig

SELF_ROLES = sorted(SELF_HELPER_ROLES)
HELPER_ROLES = [role for role in HELPER_MODEL_ROLES if role not in SELF_HELPER_ROLES]

_CREDENTIALS = {
    "anthropic": {"type": "claude_code_login"},
    "openai": {"type": "codex_login"},
}


def _write_status(anima_dir: Path, payload: dict) -> Path:
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    return anima_dir


def _config(**overrides: object) -> AnimaWorksConfig:
    return AnimaWorksConfig.model_validate({"credentials": _CREDENTIALS, **overrides})


def _pairs(fallbacks: list[HelperModelFallback]) -> list[tuple[str, str | None]]:
    return [(item.model, item.credential) for item in fallbacks]


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    return _write_status(tmp_path / "animas" / "mei", {"model": "claude-opus-5-5", "credential": "anthropic"})


# ── Self roles ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", SELF_ROLES)
def test_self_roles_use_the_animas_own_model_by_default(anima_dir: Path, role: str) -> None:
    resolved = resolve_helper_model(role, anima_dir, config=_config())

    assert (resolved.model, resolved.credential) == ("claude-opus-5-5", "anthropic")
    assert resolved.source == SELF_SOURCE
    assert resolved.is_self
    assert resolved.allow_agent_sdk_fallback is True
    assert resolved.fallbacks == []


def test_legacy_keys_and_helper_default_become_fallbacks_for_self_roles(anima_dir: Path) -> None:
    config = _config(
        consolidation={
            "llm_model": "codex/gpt-6-luna",
            "llm_credential": "openai",
            "weekly_llm_model": "openai/weekly",
            "weekly_llm_credential": "openai",
            "llm_fallback_model": "ollama/fallback",
            "llm_fallback_credential": "local",
        },
        helper_models={"default": {"model": "openai/default", "credential": "openai"}},
    )

    for role in ("episode_summary", "distillation", "reconsolidation"):
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert (resolved.model, resolved.source) == ("claude-opus-5-5", SELF_SOURCE)
        assert _pairs(resolved.fallbacks) == [("codex/gpt-6-luna", "openai"), ("ollama/fallback", "local")]
        assert resolved.allow_agent_sdk_fallback is True

    for role in ("weekly_consolidation", "project_consolidation"):
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert (resolved.model, resolved.source) == ("claude-opus-5-5", SELF_SOURCE)
        assert _pairs(resolved.fallbacks) == [("openai/weekly", "openai")]


def test_helper_default_alone_is_a_self_role_fallback(anima_dir: Path) -> None:
    config = _config(helper_models={"default": {"model": "openai/default", "credential": "openai"}})

    resolved = resolve_helper_model("episode_summary", anima_dir, config=config)

    assert resolved.model == "claude-opus-5-5"
    assert _pairs(resolved.fallbacks) == [("openai/default", "openai")]


def test_explicit_role_model_opts_a_self_role_out(anima_dir: Path) -> None:
    _write_status(
        anima_dir,
        {
            "model": "claude-opus-5-5",
            "credential": "anthropic",
            "helper_models": {"episode_summary": {"model": "openai/status-summary", "credential": "openai"}},
        },
    )
    config = _config(helper_models={"distillation": {"model": "codex/config-distill", "credential": "openai"}})

    summary = resolve_helper_model("episode_summary", anima_dir, config=config)
    distill = resolve_helper_model("distillation", anima_dir, config=config)

    assert (summary.model, summary.source) == ("openai/status-summary", "status.helper_models.episode_summary")
    assert (distill.model, distill.source) == ("codex/config-distill", "config.helper_models.distillation")
    assert not summary.is_self and not distill.is_self
    assert summary.allow_agent_sdk_fallback is False
    assert distill.allow_agent_sdk_fallback is False


def test_role_policy_without_a_model_applies_to_the_self_resolution(anima_dir: Path) -> None:
    config = _config(
        helper_models={
            "reconsolidation": {
                "fallbacks": [{"model": "openai/role-fallback", "credential": "openai"}],
                "allow_agent_sdk_fallback": False,
                "max_output_tokens": 999,
            }
        }
    )

    resolved = resolve_helper_model("reconsolidation", anima_dir, config=config)

    assert (resolved.model, resolved.source) == ("claude-opus-5-5", SELF_SOURCE)
    assert _pairs(resolved.fallbacks) == [("openai/role-fallback", "openai")]
    assert resolved.allow_agent_sdk_fallback is False
    assert resolved.max_output_tokens == 999


def test_saved_helper_default_does_not_disable_the_sdk_for_self(anima_dir: Path) -> None:
    # save_config dumps every field, so helper_models.default always carries
    # allow_agent_sdk_fallback=false and llm_model at its schema default.
    saved = AnimaWorksConfig.model_validate({"credentials": _CREDENTIALS}).model_dump(mode="json")
    config = AnimaWorksConfig.model_validate(saved)
    assert "allow_agent_sdk_fallback" in config.helper_models.default.model_fields_set

    for role in HELPER_MODEL_ROLES:
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert resolved.model == "claude-opus-5-5"
        assert resolved.is_self
        assert resolved.allow_agent_sdk_fallback is True


# ── Helper roles ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", HELPER_ROLES)
def test_unconfigured_helper_roles_fall_back_to_the_animas_own_model(anima_dir: Path, role: str) -> None:
    resolved = resolve_helper_model(role, anima_dir, config=_config())

    assert (resolved.model, resolved.credential) == ("claude-opus-5-5", "anthropic")
    assert resolved.source == SELF_UNCONFIGURED_HELPER_SOURCE
    assert resolved.is_self
    assert resolved.allow_agent_sdk_fallback is True


@pytest.mark.parametrize("role", HELPER_ROLES)
def test_configured_helper_default_keeps_helper_roles_on_the_helper(anima_dir: Path, role: str) -> None:
    config = _config(helper_models={"default": {"model": "openai/helper", "credential": "openai"}})

    resolved = resolve_helper_model(role, anima_dir, config=config)

    assert (resolved.model, resolved.credential, resolved.source) == (
        "openai/helper",
        "openai",
        "helper_models.default",
    )
    assert not resolved.is_self
    assert resolved.allow_agent_sdk_fallback is False


def test_legacy_keys_keep_helper_roles_on_the_helper(anima_dir: Path) -> None:
    config = _config(
        consolidation={
            "llm_model": "codex/gpt-6-luna",
            "llm_credential": "openai",
            "fact_reconcile_model": "openai/deepseek-v4-flash",
            "fact_reconcile_credential": "gpu40-direct",
        }
    )

    for role in ("fact_extraction", "fact_reconcile"):
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert (resolved.model, resolved.credential) == ("openai/deepseek-v4-flash", "gpu40-direct")
    for role in ("conversation_compression", "asset_reconcile"):
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert (resolved.model, resolved.credential) == ("codex/gpt-6-luna", "openai")
        assert resolved.source == "config.consolidation.llm_model"


def test_status_extraction_key_keeps_fact_roles_on_the_helper(anima_dir: Path) -> None:
    _write_status(
        anima_dir,
        {
            "model": "claude-opus-5-5",
            "credential": "anthropic",
            "extraction_model": "openai/status-fact",
            "extraction_credential": "gpu-status",
        },
    )

    resolved = resolve_helper_model("fact_extraction", anima_dir, config=_config())

    assert (resolved.model, resolved.source) == ("openai/status-fact", "status.extraction_model")


def test_schema_default_llm_model_counts_as_a_choice_only_with_a_credential(anima_dir: Path) -> None:
    unset = _config(consolidation={"llm_model": DEFAULT_CONSOLIDATION_MODEL})
    chosen = _config(consolidation={"llm_model": DEFAULT_CONSOLIDATION_MODEL, "llm_credential": "anthropic"})

    assert resolve_helper_model("asset_reconcile", anima_dir, config=unset).is_self
    resolved = resolve_helper_model("asset_reconcile", anima_dir, config=chosen)
    assert (resolved.model, resolved.source) == (DEFAULT_CONSOLIDATION_MODEL, "config.consolidation.llm_model")


# ── When self is not possible ────────────────────────────────────────────────


def test_without_anima_dir_roles_keep_the_helper_chain() -> None:
    legacy = _config(consolidation={"llm_model": "codex/gpt-6-luna", "llm_credential": "openai"})

    for role in SELF_ROLES:
        resolved = resolve_helper_model(role, config=legacy)
        assert (resolved.model, resolved.source) == ("codex/gpt-6-luna", "config.consolidation.llm_model")
        assert resolved.allow_agent_sdk_fallback is False
    for role in HELPER_ROLES:
        resolved = resolve_helper_model(role, config=_config())
        assert (resolved.model, resolved.source) == (DEFAULT_CONSOLIDATION_MODEL, "code.DEFAULT_CONSOLIDATION_MODEL")
        assert resolved.allow_agent_sdk_fallback is False


@pytest.mark.parametrize(
    "status",
    [
        {"model": "claude-opus-5-5", "credential": "anthropic", "execution_mode": "D"},
        {"model": "gemini/gemini-3-pro", "credential": "anthropic"},
        {"model": "grok/grok-5", "credential": "anthropic"},
    ],
)
def test_modes_without_a_one_shot_backend_keep_the_helper_chain(tmp_path: Path, status: dict) -> None:
    anima_dir = _write_status(tmp_path / "animas" / "kai", status)
    legacy = _config(consolidation={"llm_model": "codex/gpt-6-luna", "llm_credential": "openai"})

    for role in SELF_ROLES:
        resolved = resolve_helper_model(role, anima_dir, config=legacy)
        assert (resolved.model, resolved.is_self) == ("codex/gpt-6-luna", False)
    for role in HELPER_ROLES:
        resolved = resolve_helper_model(role, anima_dir, config=_config())
        assert (resolved.model, resolved.source) == (DEFAULT_CONSOLIDATION_MODEL, "code.DEFAULT_CONSOLIDATION_MODEL")


def test_unconfigured_anima_credential_means_no_own_model(tmp_path: Path) -> None:
    anima_dir = _write_status(tmp_path / "animas" / "kai", {"model": "claude-opus-5-5", "credential": "missing"})

    resolved = resolve_helper_model("episode_summary", anima_dir, config=_config())

    assert (resolved.model, resolved.source) == (DEFAULT_CONSOLIDATION_MODEL, "code.DEFAULT_CONSOLIDATION_MODEL")


@pytest.mark.parametrize(
    ("model", "credential"),
    [
        ("claude-opus-5-5", "openai"),  # Mode S with a codex_login credential name
        ("codex/gpt-6-luna", "openai"),  # Mode C
        ("openai/gpt-test", "openai"),  # Mode A
    ],
)
def test_one_shot_modes_self_resolve_with_the_animas_credential(tmp_path: Path, model: str, credential: str) -> None:
    anima_dir = _write_status(tmp_path / "animas" / "kai", {"model": model, "credential": credential})

    resolved = resolve_helper_model("episode_summary", anima_dir, config=_config())

    assert (resolved.model, resolved.credential, resolved.is_self) == (model, credential, True)


def test_validation_does_not_flag_self_resolutions(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    _write_status(animas_dir / "rin", {"model": "claude-opus-5-5", "credential": "openai"})

    diagnostics = validate_helper_model_credentials(animas_dir, config=_config())

    assert not any("anima=rin" in item for item in diagnostics)


# ── Call sites ───────────────────────────────────────────────────────────────


def _self_resolution(**overrides: object) -> ResolvedHelperModel:
    values: dict = {
        "model": "claude-opus-5-5",
        "credential": "openai",
        "fallbacks": [HelperModelFallback(model="openai/helper", credential="openai")],
        "allow_agent_sdk_fallback": True,
        "max_output_tokens": None,
        "source": SELF_SOURCE,
    }
    values.update(overrides)
    return ResolvedHelperModel(**values)


def _anima_model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-opus-5-5",
        credential="openai",
        credential_type="codex_login",
        execution_mode="S",
        resolved_mode="S",
        mode_s_auth="max",
        max_tokens=16000,
        fallback_model="openai/main-fallback",
        fallback_models=["openai/main-fallback-2"],
        background_model="openai/background",
        background_credential="openai",
    )


def test_self_consolidation_turn_runs_with_the_animas_own_model_config() -> None:
    from core.anima.lifecycle import _consolidation_model_configs

    base = _anima_model_config()

    configs = _consolidation_model_configs(base, _self_resolution(), _config())

    own = configs[0]
    stripped = {"fallback_model", "fallback_models", "background_model", "background_credential"}
    assert own.model_dump(exclude=stripped) == base.model_dump(exclude=stripped)
    assert own.model_dump(include=stripped) == {
        "fallback_model": None,
        "fallback_models": [],
        "background_model": None,
        "background_credential": None,
    }
    # The helper fallback is still synthesised from the helper model.
    assert (configs[1].model, configs[1].credential, configs[1].execution_mode) == ("openai/helper", "openai", "A")
    assert configs[1].mode_s_auth is None


def test_self_consolidation_turn_honours_role_max_output_tokens() -> None:
    from core.anima.lifecycle import _consolidation_model_configs

    configs = _consolidation_model_configs(_anima_model_config(), _self_resolution(max_output_tokens=4000), _config())

    assert configs[0].max_tokens == 4000


@pytest.mark.asyncio
async def test_weekly_consolidation_self_turn_uses_the_anima_model_config(anima_dir: Path) -> None:
    from core.anima.lifecycle import LifecycleMixin

    base = ModelConfig(
        model="claude-opus-5-5",
        credential="anthropic",
        credential_type="claude_code_login",
        resolved_mode="S",
        mode_s_auth="max",
        background_model="openai/background",
        fallback_models=["openai/main-fallback"],
    )
    owner = SimpleNamespace(
        name="mei",
        anima_dir=anima_dir,
        memory=MagicMock(),
        _agent_session_lock=asyncio.Lock(),
        _run_autonomous_skill_learning=lambda: None,
    )
    owner.memory.read_model_config.return_value = base
    agent = MagicMock()
    agent.run_cycle = AsyncMock(return_value=CycleResult(trigger="weekly", action="completed", summary="ok"))
    owner.agent = agent
    engine = MagicMock()
    engine.project = None
    engine._find_merge_candidates.return_value = []
    engine._find_conflicting_fact_candidates.return_value = []

    with (
        patch("core.config.load_config", return_value=_config()),
        patch("core.anima.lifecycle.load_prompt", return_value="prompt"),
        patch("core.memory.maintenance.forgetting.ForgettingEngine") as forgetter,
        patch("core.memory.maintenance.hygiene.scan_memory_hygiene", return_value={}),
    ):
        forgetter.return_value.list_forgetting_candidates.return_value = []
        await LifecycleMixin._run_weekly_consolidation(owner, engine)

    override = agent.run_cycle.await_args.kwargs["model_config_override"]
    assert (override.model, override.credential, override.resolved_mode, override.mode_s_auth) == (
        "claude-opus-5-5",
        "anthropic",
        "S",
        "max",
    )
    assert override.background_model is None and override.fallback_models == []
    assert agent.run_cycle.await_count == 1


@pytest.mark.asyncio
async def test_background_review_runs_on_own_model_with_sdk_fallback(anima_dir: Path) -> None:
    from core.memory.maintenance.background_review import _run_model

    memory = MagicMock()
    memory.read_model_config.return_value = ModelConfig(model="claude-opus-5-5", credential="anthropic")
    completion = AsyncMock(return_value="[]")
    with patch("core.llm.oneshot.one_shot_completion", completion):
        raw, model = await _run_model("prompt", anima_dir, memory, _config())

    assert (raw, model) == ("[]", "claude-opus-5-5")
    assert completion.await_args.kwargs["model"] == "claude-opus-5-5"
    assert completion.await_args.kwargs["credential"] == "anthropic"
    assert completion.await_args.kwargs["allow_agent_sdk_fallback"] is True


@pytest.mark.asyncio
async def test_helper_one_shot_tries_own_model_then_configured_helper(anima_dir: Path) -> None:
    from core.llm.helper_completion import one_shot_helper_completion

    config = _config(consolidation={"llm_model": "codex/gpt-6-luna", "llm_credential": "openai"})
    resolved = resolve_helper_model("distillation", anima_dir, config=config)
    completion = AsyncMock(side_effect=[None, "helper response"])
    with patch("core.llm.oneshot.one_shot_completion", completion):
        result = await one_shot_helper_completion("prompt", role="distillation", resolved=resolved)

    assert result == "helper response"
    assert [(call.kwargs["model"], call.kwargs["credential"]) for call in completion.await_args_list] == [
        ("claude-opus-5-5", "anthropic"),
        ("codex/gpt-6-luna", "openai"),
    ]
    assert all(call.kwargs["allow_agent_sdk_fallback"] is True for call in completion.await_args_list)
