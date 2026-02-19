"""Unit tests for vibe_image parameter in ImageGenPipeline.generate_all()."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from core.config.models import ImageGenConfig
from core.tools.image_gen import ImageGenPipeline, PipelineResult


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    """Return a temporary anima directory with an assets subdirectory."""
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def reference_image() -> bytes:
    """Dummy reference image bytes."""
    return b"\x89PNG_FAKE_FULLBODY_DATA"


# ── Vibe Transfer override tests ──────────────────────────


class TestVibeImageOverridesConfig:
    """When vibe_image is passed directly, it takes precedence over config."""

    def test_vibe_image_overrides_config_style_reference(
        self, anima_dir: Path, tmp_path: Path,
    ):
        """Direct vibe_image bytes override config.style_reference path."""
        # Create a style reference file configured via ImageGenConfig
        config_style_path = tmp_path / "config_style.png"
        config_style_path.write_bytes(b"CONFIG_STYLE_IMAGE")

        config = ImageGenConfig(
            style_reference=str(config_style_path),
            vibe_strength=0.3,
            vibe_info_extracted=0.5,
        )
        pipeline = ImageGenPipeline(anima_dir, config=config)

        direct_vibe = b"DIRECT_VIBE_IMAGE"

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                vibe_image=direct_vibe,
            )

        # NovelAIClient.generate_fullbody should receive direct_vibe,
        # NOT the config style_reference bytes
        call_kwargs = mock_client.generate_fullbody.call_args[1]
        assert call_kwargs["vibe_image"] == direct_vibe


class TestVibeStrengthOverride:
    """When vibe_strength is passed, it overrides config.vibe_strength."""

    def test_vibe_strength_override(self, anima_dir: Path):
        """Direct vibe_strength overrides config value."""
        config = ImageGenConfig(
            vibe_strength=0.3,
            vibe_info_extracted=0.5,
        )
        pipeline = ImageGenPipeline(anima_dir, config=config)

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                vibe_image=b"SOME_VIBE",
                vibe_strength=0.9,
                vibe_info_extracted=0.7,
            )

        call_kwargs = mock_client.generate_fullbody.call_args[1]
        # Should use the directly passed values, not config defaults
        assert call_kwargs["vibe_strength"] == 0.9
        assert call_kwargs["vibe_info_extracted"] == 0.7

    def test_falls_back_to_config_when_none(self, anima_dir: Path):
        """When vibe_strength/info are None, config values are used."""
        config = ImageGenConfig(
            vibe_strength=0.4,
            vibe_info_extracted=0.6,
        )
        pipeline = ImageGenPipeline(anima_dir, config=config)

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                vibe_image=b"SOME_VIBE",
                # vibe_strength and vibe_info_extracted are None (default)
            )

        call_kwargs = mock_client.generate_fullbody.call_args[1]
        assert call_kwargs["vibe_strength"] == 0.4
        assert call_kwargs["vibe_info_extracted"] == 0.6


# ── Progress callback ─────────────────────────────────────


class TestProgressCallback:
    """Verify progress_callback is invoked at step transitions."""

    def test_progress_callback_called(self, anima_dir: Path):
        """progress_callback receives generating/completed for fullbody step."""
        pipeline = ImageGenPipeline(anima_dir)
        callback = MagicMock()

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                progress_callback=callback,
            )

        # Callback should have been called with "generating" and "completed"
        callback_calls = callback.call_args_list
        step_status_pairs = [(c[0][0], c[0][1]) for c in callback_calls]
        assert ("fullbody", "generating") in step_status_pairs
        assert ("fullbody", "completed") in step_status_pairs

    def test_progress_callback_error_on_failure(self, anima_dir: Path):
        """progress_callback receives 'error' when a step fails."""
        pipeline = ImageGenPipeline(anima_dir)
        callback = MagicMock()

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.side_effect = RuntimeError("API failure")
            mock_nai_cls.return_value = mock_client

            result = pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                progress_callback=callback,
            )

        callback_calls = callback.call_args_list
        step_status_pairs = [(c[0][0], c[0][1]) for c in callback_calls]
        assert ("fullbody", "error") in step_status_pairs
        assert len(result.errors) > 0


# ── Seed parameter ─────────────────────────────────────────


class TestSeedPassedToClient:
    """Verify seed parameter reaches the underlying client."""

    def test_seed_passed_to_client(self, anima_dir: Path):
        """Seed is forwarded to NovelAIClient.generate_fullbody."""
        pipeline = ImageGenPipeline(anima_dir)

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                seed=12345,
            )

        call_kwargs = mock_client.generate_fullbody.call_args[1]
        assert call_kwargs["seed"] == 12345

    def test_seed_none_passed_to_client(self, anima_dir: Path):
        """When seed is None, it is still forwarded (client decides default)."""
        pipeline = ImageGenPipeline(anima_dir)

        with (
            patch("core.tools.image_gen.os.environ", {"NOVELAI_TOKEN": "test-token"}),
            patch("core.tools.image_gen.NovelAIClient") as mock_nai_cls,
        ):
            mock_client = MagicMock()
            mock_client.generate_fullbody.return_value = b"GENERATED_PNG"
            mock_nai_cls.return_value = mock_client

            pipeline.generate_all(
                prompt="1girl, test",
                steps=["fullbody"],
                skip_existing=False,
                seed=None,
            )

        call_kwargs = mock_client.generate_fullbody.call_args[1]
        assert call_kwargs["seed"] is None
