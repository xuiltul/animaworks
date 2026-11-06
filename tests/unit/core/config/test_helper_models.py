from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.config.helper_models import (
    HELPER_MODEL_ROLES,
    resolve_helper_model,
    validate_helper_model_credentials,
)
from core.config.schemas import DEFAULT_CONSOLIDATION_MODEL, AnimaWorksConfig
from core.schemas import ModelConfig


def _write_status(anima_dir: Path, payload: dict) -> None:
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_status_role_override_has_highest_priority_and_resolves_policy(tmp_path: Path) -> None:
    anima_dir = tmp_path / "mei"
    _write_status(
        anima_dir,
        {
            "extraction_model": "status/legacy",
            "helper_models": {
                "episode_summary": {
                    "model": "status/role",
                    "credential": "status-credential",
                    "fallbacks": [{"model": "status/fallback", "credential": "fallback-credential"}],
                    "allow_agent_sdk_fallback": True,
                    "max_output_tokens": 1234,
                }
            },
        },
    )
    config = AnimaWorksConfig.model_validate(
        {
            "helper_models": {
                "episode_summary": {"model": "config/role"},
                "default": {"model": "config/default"},
            },
            "consolidation": {"llm_model": "legacy/config"},
        }
    )

    resolved = resolve_helper_model("episode_summary", anima_dir, config=config)

    assert resolved.model == "status/role"
    assert resolved.credential == "status-credential"
    assert [(item.model, item.credential) for item in resolved.fallbacks] == [
        ("status/fallback", "fallback-credential")
    ]
    assert resolved.allow_agent_sdk_fallback is True
    assert resolved.max_output_tokens == 1234
    assert resolved.source == "status.helper_models.episode_summary"


def test_status_extraction_legacy_key_precedes_config_role(tmp_path: Path) -> None:
    anima_dir = tmp_path / "mio"
    _write_status(anima_dir, {"extraction_model": "openai/status-fact", "extraction_credential": "status-gpu"})
    config = AnimaWorksConfig.model_validate(
        {
            "helper_models": {"fact_extraction": {"model": "openai/new-role"}},
            "consolidation": {"llm_model": "openai/legacy", "llm_credential": "openai"},
        }
    )

    resolved = resolve_helper_model("fact_extraction", anima_dir, config=config)

    assert resolved.model == "openai/status-fact"
    assert resolved.credential == "status-gpu"
    assert resolved.source == "status.extraction_model"


def test_config_role_precedes_legacy_keys_and_uses_default_credential() -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "helper_models": {
                "episode_summary": {"model": "codex/role-summary"},
                "default": {"credential": "shared-helper-credential"},
            },
            "consolidation": {"llm_model": "openai/legacy-summary", "llm_credential": "openai"},
        }
    )

    resolved = resolve_helper_model("episode_summary", config=config)

    assert resolved.model == "codex/role-summary"
    assert resolved.credential == "shared-helper-credential"
    assert resolved.source == "config.helper_models.episode_summary"


def test_legacy_config_precedes_helper_default() -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "consolidation": {"llm_model": "codex/legacy", "llm_credential": "openai"},
            "helper_models": {"default": {"model": "openai/default"}},
        }
    )

    resolved = resolve_helper_model("distillation", config=config)

    assert resolved.model == "codex/legacy"
    assert resolved.credential == "openai"
    assert resolved.source == "config.consolidation.llm_model"


def test_helper_default_precedes_code_default_for_asset_reconcile() -> None:
    config = AnimaWorksConfig.model_validate(
        {"helper_models": {"default": {"model": "ollama/helper-default", "credential": "local-gateway"}}}
    )

    resolved = resolve_helper_model("asset_reconcile", config=config)

    assert resolved.model == "ollama/helper-default"
    assert resolved.credential == "local-gateway"
    assert resolved.source == "helper_models.default"


def test_code_default_is_used_when_no_explicit_helper_or_legacy_model_exists() -> None:
    resolved = resolve_helper_model("episode_summary", config=AnimaWorksConfig())

    assert resolved.model == DEFAULT_CONSOLIDATION_MODEL
    assert resolved.credential == "anthropic"
    assert resolved.source == "code.DEFAULT_CONSOLIDATION_MODEL"


def test_old_production_config_resolves_existing_helper_roles() -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "consolidation": {
                "llm_model": "codex/gpt-6-luna",
                "llm_credential": "openai",
                "live_fact_model": None,
                "fact_reconcile_model": "openai/deepseek-v4-flash",
                "fact_reconcile_credential": "gpu40-direct",
            }
        }
    )

    for role in (
        "episode_summary",
        "weekly_consolidation",
        "project_consolidation",
        "conversation_compression",
        "distillation",
        "reconsolidation",
        "asset_reconcile",
    ):
        resolved = resolve_helper_model(role, config=config)
        assert (resolved.model, resolved.credential) == ("codex/gpt-6-luna", "openai")

    for role in ("fact_extraction", "fact_reconcile"):
        resolved = resolve_helper_model(role, config=config)
        assert (resolved.model, resolved.credential) == ("openai/deepseek-v4-flash", "gpu40-direct")


def test_fact_roles_accept_legacy_status_extraction_keys(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_status(anima_dir, {"extraction_model": "openai/status-fact", "extraction_credential": "gpu-status"})
    config = AnimaWorksConfig.model_validate({"consolidation": {"llm_model": "openai/legacy"}})

    for role in ("fact_extraction", "fact_reconcile"):
        resolved = resolve_helper_model(role, anima_dir, config=config)
        assert (resolved.model, resolved.credential) == ("openai/status-fact", "gpu-status")
        assert resolved.source == "status.extraction_model"


def test_status_background_and_anima_fallbacks_never_select_helper_models(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    _write_status(
        anima_dir,
        {
            "model": "claude-opus-5-5",
            "credential": "anthropic",
            "background_model": "claude-opus-5-5",
            "background_credential": "anthropic",
            "fallback_model": "anthropic/claude-opus-5-5",
            "fallback_models": ["openai/deepseek-v4-flash-0731"],
        },
    )
    config = AnimaWorksConfig.model_validate(
        {
            "anima_defaults": {
                "model": "claude-opus-5-5",
                "background_model": "claude-opus-5-5",
                "fallback_models": ["openai/deepseek-v4-flash-0731"],
            }
        }
    )

    resolved = [resolve_helper_model(role, anima_dir, config=config) for role in HELPER_MODEL_ROLES]

    assert all(item.model != "claude-opus-5-5" for item in resolved)
    assert all("claude-opus-5-5" not in item.model for item in resolved)
    assert all(not any("deepseek-v4-flash-0731" in fallback.model for fallback in item.fallbacks) for item in resolved)


def test_daily_summary_candidates_use_only_registry_fallbacks(tmp_path: Path) -> None:
    from core.anima.lifecycle import _episode_summary_model_configs

    anima_dir = tmp_path / "alice"
    _write_status(
        anima_dir,
        {
            "background_model": "claude-opus-5-5",
            "fallback_models": ["openai/deepseek-v4-flash-0731"],
        },
    )
    config = AnimaWorksConfig.model_validate(
        {
            "consolidation": {
                "llm_model": "codex/gpt-6-luna",
                "llm_credential": "openai",
                "llm_fallback_model": "openai/deepseek-v4-flash",
                "llm_fallback_credential": "gpu40-direct",
            }
        }
    )
    base = ModelConfig(
        model="claude-opus-5-5",
        background_model="claude-opus-5-5",
        fallback_models=["openai/deepseek-v4-flash-0731"],
    )
    helper = resolve_helper_model("episode_summary", anima_dir, config=config)

    candidates = _episode_summary_model_configs(
        base,
        "claude-opus-5-5",
        config,
        anima_dir=anima_dir,
        helper_model=helper,
    )

    assert [(item.model, item.credential) for item in candidates] == [
        ("codex/gpt-6-luna", "openai"),
        ("openai/deepseek-v4-flash", "gpu40-direct"),
    ]
    assert all("claude-opus" not in item.model for item in candidates)
    assert all("deepseek-v4-flash-0731" not in item.model for item in candidates)


def test_legacy_llm_fallback_maps_only_to_compatible_helper_roles() -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "consolidation": {
                "llm_model": "openai/primary",
                "llm_fallback_model": "ollama/fallback",
                "llm_fallback_credential": "local",
            }
        }
    )

    for role in ("episode_summary", "distillation", "reconsolidation", "conversation_compression", "asset_reconcile"):
        resolved = resolve_helper_model(role, config=config)
        assert [(item.model, item.credential) for item in resolved.fallbacks] == [("ollama/fallback", "local")]

    assert resolve_helper_model("fact_extraction", config=config).fallbacks == []
    assert resolve_helper_model("weekly_consolidation", config=config).fallbacks == []


def test_resolver_rejects_unknown_roles() -> None:
    with pytest.raises(ValueError, match="Unknown helper-model role"):
        resolve_helper_model("background_model", config=AnimaWorksConfig())


def test_startup_validation_warns_for_missing_and_incompatible_credentials(caplog: pytest.LogCaptureFixture) -> None:
    config = AnimaWorksConfig.model_validate(
        {
            "credentials": {"openai": {"api_key": "do-not-log-this-secret"}},
            "helper_models": {
                "episode_summary": {"model": "openai/gpt-test", "credential": "missing-credential"},
                "fact_reconcile": {"model": "claude-sonnet-4-6", "credential": "openai"},
            },
        }
    )

    diagnostics = validate_helper_model_credentials(config=config)

    assert any("role=episode_summary" in item and "credential_missing" in item for item in diagnostics)
    assert any("role=fact_reconcile" in item and "provider_mismatch" in item for item in diagnostics)
    assert "do-not-log-this-secret" not in caplog.text
    assert "Helper model credential validation" in caplog.text


@pytest.mark.asyncio
async def test_episode_summary_one_shot_receives_default_sdk_fallback_policy() -> None:
    from core.anima.lifecycle import _complete_episode_prompt

    completion = AsyncMock(return_value="summary")
    with patch("core.llm.oneshot.one_shot_completion", completion):
        result, reason = await _complete_episode_prompt("prompt", [ModelConfig(model="openai/helper")])

    assert result == "summary"
    assert reason == ""
    assert completion.await_args.kwargs["allow_agent_sdk_fallback"] is False
    assert completion.await_args.kwargs["max_tokens"] == 8192
