"""Tests for chat model override validation and canonical available-model IDs.

Covers H-1 (server-side model param validation) and H-2 (available-models
must return canonical IDs that resolve to a real execution mode, never B).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from unittest.mock import patch

from core.config.models import AnimaWorksConfig, CredentialConfig
from server.routes.config_routes import (
    _build_static_model_catalog,
    available_model_id_set,
    validate_chat_model,
)


class TestAvailableModelsCanonicalIds:
    @pytest.fixture
    def provider_config(self) -> AnimaWorksConfig:
        return AnimaWorksConfig(
            credentials={
                "anthropic": CredentialConfig(type="claude_code_login", api_key=""),
                "openai": CredentialConfig(type="api_key", api_key="sk-openai"),
                "google": CredentialConfig(type="api_key", api_key="sk-google"),
            }
        )

    @pytest.mark.parametrize("credential", ["openai", "google", "anthropic"])
    def test_catalog_ids_are_canonical(self, provider_config, credential):
        config = provider_config
        if credential not in ("openai", "google", "anthropic"):
            pytest.skip()
        with (
            patch("server.routes.config_routes.is_codex_login_available", return_value=False),
            patch("server.routes.config_routes.is_grok_authenticated", return_value=False),
        ):
            models = _build_static_model_catalog(config)
        if credential == "openai":
            ids = {m["id"] for m in models if m["credential"] == "openai"}
            assert "openai/gpt-4.1" in ids
            assert "gpt-4.1" not in ids  # bare name must not be returned
        elif credential == "google":
            ids = {m["id"] for m in models if m["credential"] == "google"}
            assert "gemini/gemini-2.5-flash" in ids
            assert "gemini-2.5-flash" not in ids
        elif credential == "anthropic":
            ids = {m["id"] for m in models if m["credential"] == "anthropic"}
            assert "claude-sonnet-4-6" in ids

    def test_all_available_ids_resolve_to_non_b_mode(self, provider_config):
        """H-2: every returned id must resolve to a real mode (never Mode B)."""
        from core.config.model_mode import resolve_execution_mode

        with (
            patch("server.routes.config_routes.is_codex_login_available", return_value=False),
            patch("server.routes.config_routes.is_grok_authenticated", return_value=False),
        ):
            ids = available_model_id_set(provider_config)

        assert ids, "expected a non-empty available-model set for the provider config"
        for mid in ids:
            mode = resolve_execution_mode(provider_config, mid)
            assert mode != "B", f"{mid} resolves to Mode B (bare id must be prefixed)"


class TestValidateChatModel:
    def _patch_allowed(self, allowed_set, anima_model="claude-sonnet-4-6"):
        from contextlib import ExitStack
        from core.schemas import ModelConfig

        mc = ModelConfig(model=anima_model, fallback_models=["x:grok/grok-4.5"])
        stack = ExitStack()
        stack.enter_context(
            patch(
                "server.routes.config_routes.available_model_id_set",
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
        assert validate_chat_model("anima", None) is None
        assert validate_chat_model("anima", "") is None
        assert validate_chat_model("anima", "   ") is None

    def test_allowed_model_passes(self):
        with self._patch_allowed({"openai/gpt-4.1"}):
            assert validate_chat_model("anima", "openai/gpt-4.1") is None

    def test_anima_default_model_passes(self):
        # anima's current model is always allowed even if outside the catalog.
        with self._patch_allowed({"openai/gpt-4.1"}):
            assert validate_chat_model("anima", "claude-sonnet-4-6") is None

    def test_fallback_entry_and_model_part_allowed(self):
        # full fallback entry and its bare model part are both allowed.
        with self._patch_allowed({"openai/gpt-4.1"}):
            assert validate_chat_model("anima", "grok/grok-4.5") is None

    def test_mode_model_form_checks_model_part(self):
        with self._patch_allowed({"codex/gpt-5.6-sol"}):
            assert validate_chat_model("anima", "c:codex/gpt-5.6-sol") is None

    def test_unknown_model_rejected(self):
        with self._patch_allowed({"openai/gpt-4.1"}):
            err = validate_chat_model("anima", "not-a-real-model")
        assert err is not None
        assert "not-a-real-model" in err

    def test_too_long_rejected(self):
        with self._patch_allowed({"openai/gpt-4.1"}):
            err = validate_chat_model("anima", "x" * 200)
        assert err is not None
        assert "128" in err
