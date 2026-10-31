"""Unit tests for provider-specific LiteLLM kwargs (Azure, Vertex AI)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from core.execution.base import BaseExecutor
from core.schemas import ModelConfig


class _StubExecutor(BaseExecutor):
    """Minimal concrete executor for testing base-class helpers."""

    async def execute(self, prompt, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def execute_streaming(self, prompt, **kwargs):  # type: ignore[override]
        raise NotImplementedError
        yield  # pragma: no cover


# ── Azure ─────────────────────────────────────────────────────


class TestAzureProviderKwargs:
    def _make_executor(self, *, extra_keys: dict[str, str] | None = None) -> _StubExecutor:
        mc = ModelConfig(
            model="azure/gpt-4.1-mini",
            api_key="test-key",
            api_base_url="https://my-resource.openai.azure.com",
            extra_keys=extra_keys or {},
        )
        return _StubExecutor(mc, Path("/tmp/test-anima"))

    def test_api_version_from_extra_keys(self):
        exe = self._make_executor(extra_keys={"api_version": "2024-12-01-preview"})
        kwargs: dict = {}
        exe._apply_provider_kwargs(kwargs)
        assert kwargs["api_version"] == "2024-12-01-preview"

    def test_api_version_from_env(self):
        exe = self._make_executor()
        with patch.dict(os.environ, {"AZURE_API_VERSION": "2025-01-01"}):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert kwargs["api_version"] == "2025-01-01"

    def test_api_version_extra_keys_over_env(self):
        exe = self._make_executor(extra_keys={"api_version": "from-config"})
        with patch.dict(os.environ, {"AZURE_API_VERSION": "from-env"}):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert kwargs["api_version"] == "from-config"

    def test_no_api_version_when_absent(self):
        exe = self._make_executor()
        with patch.dict(os.environ, {}, clear=True):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert "api_version" not in kwargs

    def test_non_azure_model_ignored(self):
        mc = ModelConfig(
            model="openai/gpt-4.1-mini",
            extra_keys={"api_version": "should-not-appear"},
        )
        exe = _StubExecutor(mc, Path("/tmp/test-anima"))
        kwargs: dict = {}
        exe._apply_provider_kwargs(kwargs)
        assert "api_version" not in kwargs


# ── Vertex AI ─────────────────────────────────────────────────


class TestVertexAIProviderKwargs:
    def _make_executor(self, *, extra_keys: dict[str, str] | None = None) -> _StubExecutor:
        mc = ModelConfig(
            model="vertex_ai/gemini-2.5-flash",
            extra_keys=extra_keys or {},
        )
        return _StubExecutor(mc, Path("/tmp/test-anima"))

    def test_vertex_project_from_extra_keys(self):
        exe = self._make_executor(extra_keys={
            "vertex_project": "my-project",
            "vertex_location": "us-central1",
        })
        kwargs: dict = {}
        exe._apply_provider_kwargs(kwargs)
        assert kwargs["vertex_project"] == "my-project"
        assert kwargs["vertex_location"] == "us-central1"
        assert "vertex_credentials" not in kwargs

    def test_vertex_credentials_from_extra_keys(self):
        exe = self._make_executor(extra_keys={
            "vertex_project": "proj",
            "vertex_location": "asia-northeast1",
            "vertex_credentials": "/path/to/sa.json",
        })
        kwargs: dict = {}
        exe._apply_provider_kwargs(kwargs)
        assert kwargs["vertex_credentials"] == "/path/to/sa.json"

    def test_vertex_from_env(self):
        exe = self._make_executor()
        env = {
            "VERTEX_PROJECT": "env-project",
            "VERTEX_LOCATION": "europe-west1",
        }
        with patch.dict(os.environ, env):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert kwargs["vertex_project"] == "env-project"
            assert kwargs["vertex_location"] == "europe-west1"

    def test_extra_keys_over_env(self):
        exe = self._make_executor(extra_keys={"vertex_project": "from-config"})
        with patch.dict(os.environ, {"VERTEX_PROJECT": "from-env"}):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert kwargs["vertex_project"] == "from-config"

    def test_no_vertex_keys_when_absent(self):
        exe = self._make_executor()
        with patch.dict(os.environ, {}, clear=True):
            kwargs: dict = {}
            exe._apply_provider_kwargs(kwargs)
            assert "vertex_project" not in kwargs
            assert "vertex_location" not in kwargs
            assert "vertex_credentials" not in kwargs

    def test_non_vertex_model_ignored(self):
        mc = ModelConfig(
            model="google/gemini-2.5-flash",
            extra_keys={"vertex_project": "should-not-appear"},
        )
        exe = _StubExecutor(mc, Path("/tmp/test-anima"))
        kwargs: dict = {}
        exe._apply_provider_kwargs(kwargs)
        assert "vertex_project" not in kwargs


# ── ModelConfig extra_keys ────────────────────────────────────


class TestModelConfigExtraKeys:
    def test_default_empty(self):
        mc = ModelConfig()
        assert mc.extra_keys == {}

    def test_round_trip(self):
        mc = ModelConfig(extra_keys={"api_version": "v1", "vertex_project": "proj"})
        data = mc.model_dump()
        restored = ModelConfig.model_validate(data)
        assert restored.extra_keys == {"api_version": "v1", "vertex_project": "proj"}


# ── Mode resolution for vertex_ai/* ──────────────────────────


class TestVertexAIModeResolution:
    def test_vertex_ai_resolves_to_mode_a(self):
        from core.config.models import AnimaWorksConfig, resolve_execution_mode
        config = AnimaWorksConfig()
        mode = resolve_execution_mode(config, "vertex_ai/gemini-2.5-flash")
        assert mode == "A"
