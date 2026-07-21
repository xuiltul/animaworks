"""Tests for per-Anima monthly token budget configuration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from core.config.model_config import load_model_config
from core.config.models import (
    AnimaDefaults,
    AnimaModelConfig,
    AnimaWorksConfig,
    CredentialConfig,
    resolve_anima_config,
)
from core.memory.config_reader import ConfigReader
from core.schemas import ModelConfig


def test_token_budget_monthly_defaults_are_unlimited() -> None:
    assert AnimaDefaults().token_budget_monthly is None
    assert AnimaModelConfig().token_budget_monthly is None
    assert ModelConfig().token_budget_monthly is None


def test_token_budget_monthly_per_anima_override_wins() -> None:
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(token_budget_monthly=10_000),
        animas={"alice": AnimaModelConfig(token_budget_monthly=20_000)},
    )

    resolved, _ = resolve_anima_config(config, "alice")

    assert resolved.token_budget_monthly == 20_000


def test_token_budget_monthly_unspecified_falls_back_to_defaults() -> None:
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(token_budget_monthly=10_000),
        animas={"alice": AnimaModelConfig()},
    )

    resolved, _ = resolve_anima_config(config, "alice")

    assert resolved.token_budget_monthly == 10_000


def test_token_budget_monthly_status_json_override_wins(tmp_path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps({"token_budget_monthly": 30_000}),
        encoding="utf-8",
    )
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(token_budget_monthly=10_000),
        animas={"alice": AnimaModelConfig(token_budget_monthly=20_000)},
    )

    resolved, _ = resolve_anima_config(config, "alice", anima_dir=tmp_path)

    assert resolved.token_budget_monthly == 30_000


def test_token_budget_monthly_status_json_null_disables_inherited_budget(tmp_path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps({"token_budget_monthly": None}),
        encoding="utf-8",
    )
    config = AnimaWorksConfig(
        anima_defaults=AnimaDefaults(token_budget_monthly=10_000),
        animas={"alice": AnimaModelConfig(token_budget_monthly=20_000)},
    )

    resolved, _ = resolve_anima_config(config, "alice", anima_dir=tmp_path)

    assert resolved.token_budget_monthly is None


def test_token_budget_monthly_propagates_through_config_reader(tmp_path) -> None:
    resolved = AnimaDefaults(token_budget_monthly=40_000)
    credential = CredentialConfig()
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("core.config.get_config_path", return_value=config_path),
        patch("core.config.load_config", return_value=AnimaWorksConfig()),
        patch("core.config.resolve_anima_config", return_value=(resolved, credential)),
        patch("core.config.resolve_execution_mode", return_value="A"),
    ):
        model_config = ConfigReader(tmp_path).read_model_config()

    assert model_config.token_budget_monthly == 40_000


def test_token_budget_monthly_propagates_through_standalone_loader(tmp_path) -> None:
    resolved = AnimaDefaults(token_budget_monthly=50_000)
    credential = CredentialConfig()
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("core.config.models.get_config_path", return_value=config_path),
        patch("core.config.models.load_config", return_value=AnimaWorksConfig()),
        patch("core.config.models.resolve_anima_config", return_value=(resolved, credential)),
        patch("core.config.models.resolve_execution_mode", return_value="A"),
    ):
        model_config = load_model_config(tmp_path)

    assert model_config.token_budget_monthly == 50_000
