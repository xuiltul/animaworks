"""Tests for the core model catalog after the config_routes move."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.config.model_catalog import (
    _build_static_model_catalog,
    available_model_id_set,
    validate_chat_model,
    validate_model_override,
)
from core.config.models import AnimaWorksConfig, CredentialConfig


class TestModelCatalogMove:
    @pytest.fixture
    def provider_config(self) -> AnimaWorksConfig:
        return AnimaWorksConfig(
            credentials={
                "anthropic": CredentialConfig(type="claude_code_login", api_key=""),
                "openai": CredentialConfig(type="api_key", api_key="sk-openai"),
                "google": CredentialConfig(type="api_key", api_key="sk-google"),
            }
        )

    def test_validate_chat_model_is_alias(self):
        assert validate_chat_model is validate_model_override

    def test_catalog_ids_are_canonical(self, provider_config):
        with (
            patch("core.config.model_catalog.is_codex_login_available", return_value=False),
            patch("core.config.model_catalog.is_grok_authenticated", return_value=False),
        ):
            models = _build_static_model_catalog(provider_config)
        ids = {m["id"] for m in models}
        assert "openai/gpt-4.1" in ids
        assert "gpt-4.1" not in ids
        assert "claude-sonnet-4-6" in ids
        assert "gemini/gemini-2.5-flash" in ids

    def test_available_model_id_set_matches_catalog(self, provider_config):
        with (
            patch("core.config.model_catalog.is_codex_login_available", return_value=False),
            patch("core.config.model_catalog.is_grok_authenticated", return_value=False),
        ):
            ids = available_model_id_set(provider_config)
        assert "openai/gpt-4.1" in ids
        assert "claude-sonnet-4-6" in ids


class TestValidateModelOverride:
    def _patch_allowed(self, allowed_set, anima_model="claude-sonnet-4-6"):
        from contextlib import ExitStack

        from core.schemas import ModelConfig

        mc = ModelConfig(model=anima_model, fallback_models=["x:grok/grok-4.5"])
        stack = ExitStack()
        stack.enter_context(
            patch(
                "core.config.model_catalog.available_model_id_set",
                return_value=set(allowed_set),
            )
        )
        stack.enter_context(
            patch(
                "core.config.model_config.load_model_config",
                return_value=mc,
            )
        )
        return stack

    def test_empty_or_none_passes(self):
        assert validate_model_override("anima", None) is None
        assert validate_model_override("anima", "") is None
        assert validate_model_override("anima", "   ") is None

    def test_allowed_model_passes(self):
        with self._patch_allowed({"openai/gpt-4.1"}):
            assert validate_model_override("anima", "openai/gpt-4.1") is None

    def test_unknown_model_rejected(self):
        with self._patch_allowed({"openai/gpt-4.1"}):
            err = validate_model_override("anima", "not-a-real-model")
        assert err is not None
        assert "not-a-real-model" in err
