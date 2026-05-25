from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.locomo.llm_config import (
    default_answer_model,
    default_llm_credential,
    resolve_locomo_litellm_kwargs,
)


class TestLoCoMoLlmConfig:
    def test_default_answer_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCOMO_ANSWER_MODEL", "custom-model")
        assert default_answer_model() == "custom-model"

    def test_default_llm_credential_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCOMO_LLM_CREDENTIAL", "my-cred")
        assert default_llm_credential() == "my-cred"

    def test_resolve_via_vllm_lb_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("LOCOMO_LLM_CREDENTIAL", "vllm-lb")

        host_cfg = {
            "credentials": {
                "vllm-lb": {"api_key": "dummy", "base_url": "http://localhost:4000/v1"},
            },
            "consolidation": {"llm_credential": "vllm-lb"},
        }
        with patch("benchmarks.locomo.llm_config._load_host_config", return_value=host_cfg):
            model, kwargs = resolve_locomo_litellm_kwargs("deepseek-v4-flash")

        assert model == "openai/deepseek-v4-flash"
        assert kwargs["api_base"] == "http://localhost:4000/v1"
        assert kwargs["api_key"] == "dummy"

    def test_resolve_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_BASE", "http://proxy.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model, kwargs = resolve_locomo_litellm_kwargs("deepseek-v4-flash")

        assert model == "openai/deepseek-v4-flash"
        assert kwargs["api_base"] == "http://proxy.example/v1"
        assert kwargs["api_key"] == "test-key"

    def test_qwen_disables_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_BASE", "http://proxy.example/v1")

        _, kwargs = resolve_locomo_litellm_kwargs("openai/mlx-community/Qwen3.5-397B-A17B-4bit")

        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_resolve_ignores_temp_animaworks_data_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """LoCoMo sets ANIMAWORKS_DATA_DIR to temp dir; host config must still win."""
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        host_cfg = {
            "credentials": {
                "vllm-lb": {"api_key": "dummy", "base_url": "http://localhost:4000/v1"},
            },
            "consolidation": {"llm_credential": "vllm-lb", "llm_model": "openai/deepseek-v4-flash"},
        }
        with patch("benchmarks.locomo.llm_config._load_host_config", return_value=host_cfg):
            model, kwargs = resolve_locomo_litellm_kwargs("deepseek-v4-flash")

        assert model == "openai/deepseek-v4-flash"
        assert kwargs["api_base"] == "http://localhost:4000/v1"
