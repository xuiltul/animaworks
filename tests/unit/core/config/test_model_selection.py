"""Routing contracts shared by request, background and task entry points."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from core.config.model_config import resolve_model_selection, resolve_unavailable_model_config
from core.config.schemas import AnimaWorksConfig, CredentialConfig
from core.schemas import ModelConfig


@pytest.fixture
def config() -> AnimaWorksConfig:
    return AnimaWorksConfig(
        credentials={
            "anthropic": CredentialConfig(type="claude_code_login"),
            "openai": CredentialConfig(api_key="openai-secret"),
            "ollama": CredentialConfig(base_url="http://localhost:11434"),
            "gateway": CredentialConfig(api_key="gateway-secret", base_url="https://gateway.invalid/v1"),
        }
    )


@pytest.mark.parametrize(
    ("requested", "mode", "credential"),
    [
        ("s:opus", "S", "anthropic"),
        ("c:codex/model", "C", None),
        ("d:cursor/model", "D", None),
        ("g:gemini/model", "G", None),
        ("x:grok/model", "X", None),
        ("a:openai/model", "A", "openai"),
        ("b:ollama/model", "B", "ollama"),
    ],
)
def test_explicit_routes_preserve_all_adapters(config, requested, mode, credential):
    base = ModelConfig(model="claude-original", api_key="original-secret", mode_s_auth="max")
    selection = resolve_model_selection(
        base, lane="task", requested_model=requested, config=config, apply_fallback=False
    )
    assert selection.mode == mode
    assert selection.credential == credential
    assert selection.reason == "explicit_override"
    assert base.api_key == "original-secret"
    assert base.model == "claude-original"
    if credential is None:
        assert selection.effective.api_key is None
        assert selection.effective.api_key_env == ""
        assert selection.effective.api_base_url is None
        assert selection.effective.extra_keys == {}
        assert selection.effective.mode_s_auth is None
    if mode == "S":
        assert selection.guard_key == "anthropic:max"
    assert not any("session_id" in key for key in asdict(selection))


def test_background_credential_and_effort_then_explicit_override(config):
    base = ModelConfig(
        model="claude-original",
        resolved_mode="S",
        credential="anthropic",
        mode_s_auth="max",
        background_model="openai/background",
        background_credential="gateway",
        background_thinking_effort="low",
        thinking_effort="high",
    )
    background = resolve_model_selection(base, lane="heartbeat", config=config, apply_fallback=False)
    assert background.effective.model == "openai/background"
    assert background.credential == "gateway"
    assert background.effective.api_key == "gateway-secret"
    assert background.effective.api_base_url == "https://gateway.invalid/v1"
    assert background.effective.mode_s_auth is None
    assert background.effective.thinking_effort == "low"
    explicit = resolve_model_selection(
        base,
        lane="heartbeat",
        requested_model="g:gemini/override",
        config=config,
        apply_fallback=False,
    )
    assert explicit.mode == "G"
    assert explicit.effective.thinking_effort == "low"
    assert explicit.effective.api_key is None
    assert base.thinking_effort == "high"


def test_same_model_background_credential_switch_is_not_lost(config):
    base = ModelConfig(model="openai/model", resolved_mode="A", credential="openai", background_credential="gateway")
    selected = resolve_model_selection(base, lane="cron", config=config, apply_fallback=False)
    assert selected.credential == "gateway"
    assert selected.effective.api_key == "gateway-secret"


@pytest.mark.parametrize("lane", ["background", "heartbeat", "cron"])
@pytest.mark.parametrize("background_model", [None, "claude-original", "claude-other"])
@pytest.mark.parametrize("foreign_type", ["openai", "codex_login"])
def test_claude_background_does_not_inherit_openai_gateway(config, lane, background_model, foreign_type):
    config.credentials["foreign"] = CredentialConfig(
        type=foreign_type, api_key="foreign-secret", base_url="http://localhost:4000/v1"
    )
    base = ModelConfig(
        model="claude-original",
        resolved_mode="S",
        credential="anthropic",
        credential_type="claude_code_login",
        mode_s_auth="max",
        background_model=background_model,
        background_credential="foreign",
        background_thinking_effort="low",
    )
    selected = resolve_model_selection(base, lane=lane, config=config, apply_fallback=False)
    assert selected.effective.model == (background_model or base.model)
    assert selected.mode == "S"
    assert selected.credential == "anthropic"
    assert selected.guard_key == "anthropic:max"
    assert selected.effective.api_base_url is None
    assert selected.effective.api_key is None
    assert selected.effective.thinking_effort == "low"
    assert base.background_credential == "foreign"


def test_claude_explicit_api_background_gateway_is_preserved(config):
    base = ModelConfig(
        model="claude-original",
        resolved_mode="S",
        credential="anthropic",
        mode_s_auth="max",
        background_credential="gateway",
    )
    selected = resolve_model_selection(base, lane="background", config=config, apply_fallback=False)
    assert selected.effective.mode_s_auth == "api"
    assert selected.effective.api_base_url == "https://gateway.invalid/v1"
    assert selected.effective.api_key == "gateway-secret"


def test_foreign_credential_cannot_be_selected_for_explicit_claude_override(config):
    config.credentials["foreign"] = CredentialConfig(type="openai", base_url="http://localhost:4000/v1")
    with (
        patch("core.config.model_config._match_models_json", return_value={"credential": "foreign"}),
        pytest.raises(ValueError, match="No credential configured"),
    ):
        resolve_model_selection(
            ModelConfig(model="codex/model", resolved_mode="C"),
            requested_model="s:claude-original",
            config=config,
            apply_fallback=False,
        )


def test_same_route_request_keeps_custom_auth(config):
    base = ModelConfig(model="openai/model", resolved_mode="A", credential="gateway", api_key="custom-secret")
    selected = resolve_model_selection(base, requested_model="a:openai/model", config=config, apply_fallback=False)
    assert selected.effective is base
    assert selected.effective.api_key == "custom-secret"


def test_background_switch_to_cli_clears_same_provider_api_credentials(config):
    base = ModelConfig(
        model="openai/model",
        resolved_mode="A",
        credential="gateway",
        api_key="custom-secret",
        api_base_url="https://gateway.invalid",
        background_model="c:openai/other",
    )
    selected = resolve_model_selection(base, lane="background", config=config, apply_fallback=False)
    assert selected.mode == "C"
    assert selected.credential is None
    assert selected.effective.api_key is None
    assert selected.effective.api_base_url is None


def test_precedence_primary_recovers_without_rewriting_base(config):
    base = ModelConfig(model="codex/model", resolved_mode="C", fallback_models=["a:openai/backup"])
    guard = MagicMock()
    blocked = {"openai:codex": 100.0}
    guard.blocked_remaining.side_effect = lambda key: blocked.get(key, 0.0)
    guard.blocked_until.side_effect = lambda key: blocked.get(key, 0.0)
    with patch("core.execution.rate_guard.get_rate_guard", return_value=guard):
        first = resolve_model_selection(base, config=config)
        blocked.clear()
        recovered = resolve_model_selection(base, config=config)
    assert first.effective.model == "openai/backup"
    assert first.effective.api_key == "openai-secret"
    assert recovered.effective is base
    assert base.model == "codex/model"
    assert base.fallback_models == ["a:openai/backup"]


def test_unavailable_engine_honors_explicit_legacy_fallback(config):
    base = ModelConfig(model="codex/model", resolved_mode="C", fallback_model="a:openai/backup", api_key="wrong-secret")
    guard = MagicMock()
    guard.blocked_remaining.side_effect = lambda key: 100.0 if key == "openai:codex" else 0.0
    guard.blocked_until.return_value = 100.0
    with (
        patch("core.config.io.load_config", return_value=config),
        patch("core.execution.rate_guard.get_rate_guard", return_value=guard),
    ):
        selected = resolve_unavailable_model_config(base, unavailable_modes=frozenset({"C"}))
    assert selected.model == "openai/backup"
    assert selected.api_key == "openai-secret"
    assert base.api_key == "wrong-secret"


def test_unusable_explicit_override_is_rejected(config):
    with pytest.raises(ValueError, match="No credential configured"):
        resolve_model_selection(ModelConfig(), requested_model="a:missing/model", config=config)


def test_models_json_credential_is_used_without_provider_guess(config):
    with patch("core.config.model_config._match_models_json", return_value={"mode": "A", "credential": "gateway"}):
        selected = resolve_model_selection(
            ModelConfig(),
            requested_model="a:custom/model",
            config=config,
            apply_fallback=False,
        )
    assert selected.credential == "gateway"
    assert selected.effective.api_key == "gateway-secret"
