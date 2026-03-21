# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core/memory/_llm_utils.py."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.memory._llm_utils as llm_utils

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_credentials_exported() -> None:
    """Reset _credentials_exported between tests so ensure_credentials_in_env runs."""
    yield
    llm_utils._credentials_exported = False


def _make_cred(api_key: str = "") -> MagicMock:
    """Create a CredentialConfig-like mock with api_key attribute."""
    cred = MagicMock()
    cred.api_key = api_key
    return cred


def _make_config(
    llm_model: str = "anthropic/claude-sonnet-4-6",
    credentials: dict[str, MagicMock] | None = None,
) -> MagicMock:
    """Create a config mock with consolidation and credentials."""
    cfg = MagicMock()
    cfg.consolidation.llm_model = llm_model
    cfg.credentials = credentials or {}
    return cfg


# ── get_consolidation_llm_kwargs ──────────────────────────────────────────────


class TestGetConsolidationLlmKwargs:
    """Tests for get_consolidation_llm_kwargs()."""

    def test_returns_model_from_config(self) -> None:
        """get_consolidation_llm_kwargs returns dict with 'model' key from config."""
        cfg = _make_config(llm_model="anthropic/claude-sonnet-4-6")
        with patch("core.config.load_config", return_value=cfg):
            result = llm_utils.get_consolidation_llm_kwargs()
        assert result["model"] == "anthropic/claude-sonnet-4-6"

    def test_includes_api_key_when_credential_exists(self) -> None:
        """get_consolidation_llm_kwargs includes api_key when credential exists."""
        cred = _make_cred(api_key="sk-test-key")
        cfg = _make_config(
            llm_model="anthropic/claude-sonnet-4-6",
            credentials={"anthropic": cred},
        )
        with patch("core.config.load_config", return_value=cfg):
            result = llm_utils.get_consolidation_llm_kwargs()
        assert result["model"] == "anthropic/claude-sonnet-4-6"
        assert result["api_key"] == "sk-test-key"

    def test_works_without_api_key_model_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_consolidation_llm_kwargs works without api_key (model only)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = _make_config(
            llm_model="anthropic/claude-sonnet-4-6",
            credentials={"anthropic": _make_cred(api_key="")},
        )
        with patch("core.config.load_config", return_value=cfg):
            result = llm_utils.get_consolidation_llm_kwargs()
        assert result["model"] == "anthropic/claude-sonnet-4-6"
        assert "api_key" not in result

    def test_explicit_ollama_model_uses_local_base_url(self) -> None:
        cfg = _make_config(llm_model="anthropic/claude-sonnet-4-6")
        cfg.credentials = {"ollama": MagicMock(api_key="", base_url="http://127.0.0.1:11434")}
        cfg.local_llm.base_url = "http://127.0.0.1:11434"

        with patch("core.config.load_config", return_value=cfg):
            result = llm_utils.get_llm_kwargs_for_model("ollama/qwen2.5-coder:14b")

        assert result["model"] == "ollama/qwen2.5-coder:14b"
        assert result["api_base"] == "http://127.0.0.1:11434"
        assert "api_key" not in result


# ── ensure_credentials_in_env ─────────────────────────────────────────────────


class TestEnsureCredentialsInEnv:
    """Tests for ensure_credentials_in_env()."""

    def test_exports_credentials_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ensure_credentials_in_env exports credentials to environment."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cred = _make_cred(api_key="sk-exported")
        cfg = _make_config(credentials={"anthropic": cred})
        with patch("core.config.load_config", return_value=cfg):
            llm_utils.ensure_credentials_in_env()
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-exported"

    def test_does_not_overwrite_existing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ensure_credentials_in_env does not overwrite existing env vars."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-key")
        cred = _make_cred(api_key="sk-from-config")
        cfg = _make_config(credentials={"anthropic": cred})
        with patch("core.config.load_config", return_value=cfg):
            llm_utils.ensure_credentials_in_env()
        assert os.environ.get("ANTHROPIC_API_KEY") == "existing-key"

    def test_runs_only_once_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ensure_credentials_in_env runs only once (idempotent)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cred = _make_cred(api_key="sk-first")
        cfg = _make_config(credentials={"anthropic": cred})
        with patch("core.config.load_config", return_value=cfg) as mock_load:
            llm_utils.ensure_credentials_in_env()
            llm_utils.ensure_credentials_in_env()
            llm_utils.ensure_credentials_in_env()
        # load_config called once in ensure_credentials_in_env (first run only)
        assert mock_load.call_count == 1
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-first"

    def test_silently_returns_on_config_load_failure(self) -> None:
        """ensure_credentials_in_env silently returns on config load failure."""
        with patch("core.config.load_config", side_effect=RuntimeError("config error")):
            llm_utils.ensure_credentials_in_env()
        # No exception raised; function returns normally


# ── one_shot_completion and helpers ──────────────────────────────────────────


class TestOneShotCompletion:
    """Tests for one_shot_completion() and its fallback behavior."""

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_consolidation_llm_kwargs")
    @patch("core.memory._llm_utils._try_agent_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_litellm_success(
        self,
        mock_try_litellm: MagicMock,
        mock_try_agent_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        """Mock litellm to succeed; verify function returns text and Agent SDK is NOT called."""
        mock_get_kwargs.return_value = {"model": "anthropic/claude-sonnet-4-6"}
        mock_try_litellm.return_value = "LLM response text"

        result = await llm_utils.one_shot_completion("Hello")

        assert result == "LLM response text"
        mock_try_litellm.assert_called_once()
        mock_try_agent_sdk.assert_not_called()

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_consolidation_llm_kwargs")
    @patch("core.memory._llm_utils._try_agent_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_litellm_fails_sdk_success(
        self,
        mock_try_litellm: MagicMock,
        mock_try_agent_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        """LiteLLM raises; Agent SDK succeeds; verify fallback returns text."""
        mock_get_kwargs.return_value = {"model": "anthropic/claude-sonnet-4-6"}
        mock_try_litellm.side_effect = RuntimeError("LiteLLM failed")
        mock_try_agent_sdk.return_value = "SDK fallback text"

        result = await llm_utils.one_shot_completion("Hello")

        assert result == "SDK fallback text"
        mock_try_litellm.assert_called_once()
        mock_try_agent_sdk.assert_called_once()

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_consolidation_llm_kwargs")
    @patch("core.memory._llm_utils._try_agent_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_both_fail_returns_none(
        self,
        mock_try_litellm: MagicMock,
        mock_try_agent_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        """Both LiteLLM and Agent SDK fail; verify function returns None."""
        mock_get_kwargs.return_value = {"model": "anthropic/claude-sonnet-4-6"}
        mock_try_litellm.side_effect = RuntimeError("LiteLLM failed")
        mock_try_agent_sdk.return_value = None

        result = await llm_utils.one_shot_completion("Hello")

        assert result is None
        mock_try_litellm.assert_called_once()
        mock_try_agent_sdk.assert_called_once()

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.litellm", create=True)
    async def test_system_prompt_passed_to_litellm(
        self,
        mock_litellm: MagicMock,
    ) -> None:
        """Verify system_prompt is included as system message in LiteLLM call."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)

        with patch("core.memory._llm_utils.litellm", mock_litellm):
            # Import litellm inside _try_litellm; patch at module level
            pass
        # Patch where litellm is used - it's imported inside _try_litellm
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_resp
            result = await llm_utils._try_litellm(
                "user prompt",
                system_prompt="You are a helpful assistant.",
                model="anthropic/claude-sonnet-4-6",
                max_tokens=1024,
                llm_kwargs={},
            )
        assert result == "ok"
        call_kwargs = mock_acompletion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages is not None
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "user prompt"}

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_llm_kwargs_for_model")
    @patch("core.memory._llm_utils._try_agent_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_default_model_from_config(
        self,
        mock_try_litellm: MagicMock,
        mock_try_agent_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        """When model='' (default), model is resolved from get_llm_kwargs_for_model()."""
        mock_get_kwargs.return_value = {"model": "anthropic/claude-sonnet-4-6"}
        mock_try_litellm.return_value = "ok"

        await llm_utils.one_shot_completion("Hi", model="")

        mock_get_kwargs.assert_called()
        mock_try_litellm.assert_called_once()
        call_kwargs = mock_try_litellm.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_consolidation_llm_kwargs")
    @patch("core.memory._llm_utils._try_agent_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_non_anthropic_model_skips_sdk(
        self,
        mock_try_litellm: MagicMock,
        mock_try_agent_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        """Non-Anthropic model (e.g. openai/gpt-4.1): LiteLLM failure returns None without SDK."""
        mock_get_kwargs.return_value = {"model": "openai/gpt-4.1"}
        mock_try_litellm.side_effect = RuntimeError("LiteLLM failed")

        result = await llm_utils.one_shot_completion("Hi", model="openai/gpt-4.1")

        assert result is None
        mock_try_litellm.assert_called_once()
        mock_try_agent_sdk.assert_not_called()

    @pytest.mark.asyncio
    @patch("core.memory._llm_utils.get_llm_kwargs_for_model")
    @patch("core.memory._llm_utils._try_codex_sdk")
    @patch("core.memory._llm_utils._try_litellm")
    async def test_codex_model_falls_back_to_codex_sdk(
        self,
        mock_try_litellm: MagicMock,
        mock_try_codex_sdk: MagicMock,
        mock_get_kwargs: MagicMock,
    ) -> None:
        mock_get_kwargs.return_value = {"model": "codex/gpt-5.4-mini"}
        mock_try_litellm.side_effect = RuntimeError("LiteLLM failed")
        mock_try_codex_sdk.return_value = "Codex fallback text"

        result = await llm_utils.one_shot_completion("Hi", model="codex/gpt-5.4-mini")

        assert result == "Codex fallback text"
        mock_try_litellm.assert_called_once()
        mock_try_codex_sdk.assert_called_once()


class TestIsAnthropicModel:
    """Tests for _is_anthropic_model() helper."""

    def test_is_anthropic_model_true(self) -> None:
        """Various Anthropic model patterns return True."""
        assert llm_utils._is_anthropic_model("anthropic/claude-sonnet-4-6") is True
        assert llm_utils._is_anthropic_model("bedrock/claude-sonnet-4-6") is True
        assert llm_utils._is_anthropic_model("vertex_ai/claude-sonnet-4-6") is True
        assert llm_utils._is_anthropic_model("claude-sonnet-4-6") is True

    def test_is_anthropic_model_false(self) -> None:
        """Non-Anthropic models return False."""
        assert llm_utils._is_anthropic_model("openai/gpt-4.1") is False
        assert llm_utils._is_anthropic_model("ollama/gemma3") is False
        assert llm_utils._is_anthropic_model("google/gemini-2.0") is False


class TestStripProviderPrefix:
    """Tests for _strip_provider_prefix() helper."""

    def test_strip_provider_prefix(self) -> None:
        """Provider prefix is stripped for Agent SDK model name."""
        assert (
            llm_utils._strip_provider_prefix("anthropic/claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )
        assert (
            llm_utils._strip_provider_prefix("bedrock/jp.anthropic.claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )
        assert (
            llm_utils._strip_provider_prefix("vertex_ai/claude-sonnet-4-6")
            == "claude-sonnet-4-6"
        )
