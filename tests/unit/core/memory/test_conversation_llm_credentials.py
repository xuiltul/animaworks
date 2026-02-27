from __future__ import annotations

"""Tests for ConversationMemory._call_llm() provider credential passing."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test_anima"
    d.mkdir()
    (d / "state").mkdir()
    return d


def _make_acompletion_mock() -> AsyncMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "summary"
    return AsyncMock(return_value=resp)


class TestCallLlmProviderCredentials:
    """_call_llm must forward provider-specific kwargs to litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_bedrock_credentials_passed(self, anima_dir: Path) -> None:
        cfg = ModelConfig(
            model="bedrock/jp.anthropic.claude-sonnet-4-6",
            extra_keys={
                "aws_access_key_id": "AKIATEST",
                "aws_secret_access_key": "secret123",
                "aws_region_name": "ap-northeast-1",
            },
        )
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        mock_ac.assert_called_once()
        kw = mock_ac.call_args
        assert kw.kwargs["aws_access_key_id"] == "AKIATEST"
        assert kw.kwargs["aws_secret_access_key"] == "secret123"
        assert kw.kwargs["aws_region_name"] == "ap-northeast-1"

    @pytest.mark.asyncio
    async def test_azure_api_version_passed(self, anima_dir: Path) -> None:
        cfg = ModelConfig(
            model="azure/gpt-4.1-mini",
            extra_keys={"api_version": "2024-12-01-preview"},
        )
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        kw = mock_ac.call_args
        assert kw.kwargs["api_version"] == "2024-12-01-preview"

    @pytest.mark.asyncio
    async def test_vertex_credentials_passed(self, anima_dir: Path) -> None:
        cfg = ModelConfig(
            model="vertex_ai/gemini-2.5-flash",
            extra_keys={
                "vertex_project": "my-project",
                "vertex_location": "us-central1",
            },
        )
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        kw = mock_ac.call_args
        assert kw.kwargs["vertex_project"] == "my-project"
        assert kw.kwargs["vertex_location"] == "us-central1"

    @pytest.mark.asyncio
    async def test_no_extra_kwargs_for_generic_model(self, anima_dir: Path) -> None:
        cfg = ModelConfig(model="claude-sonnet-4-6")
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        kw = mock_ac.call_args
        for key in (
            "aws_access_key_id", "aws_secret_access_key", "aws_region_name",
            "api_version", "vertex_project", "vertex_location", "vertex_credentials",
        ):
            assert key not in kw.kwargs

    @pytest.mark.asyncio
    async def test_bedrock_env_fallback(self, anima_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

        cfg = ModelConfig(model="bedrock/anthropic.claude-3-haiku", extra_keys={})
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        kw = mock_ac.call_args
        assert kw.kwargs["aws_access_key_id"] == "AKIAENV"
        assert kw.kwargs["aws_secret_access_key"] == "envsecret"
        assert kw.kwargs["aws_region_name"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_fallback_model_prefix_used(self, anima_dir: Path) -> None:
        """When fallback_model is set, its prefix determines provider kwargs."""
        cfg = ModelConfig(
            model="claude-sonnet-4-6",
            fallback_model="bedrock/anthropic.claude-3-haiku",
            extra_keys={
                "aws_access_key_id": "AKIAFALLBACK",
                "aws_secret_access_key": "fbsecret",
                "aws_region_name": "eu-west-1",
            },
        )
        from core.memory.conversation import ConversationMemory

        conv = ConversationMemory(anima_dir, cfg)
        mock_ac = _make_acompletion_mock()

        with patch("litellm.acompletion", mock_ac):
            await conv._call_llm("sys", "user msg")

        kw = mock_ac.call_args
        assert kw.kwargs["aws_access_key_id"] == "AKIAFALLBACK"
