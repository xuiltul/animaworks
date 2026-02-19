"""Unit tests for ImageGenPipeline.generate_bustup_expression()."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import ImageGenConfig
from core.tools.image_gen import (
    ImageGenPipeline,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
    _VALID_EXPRESSION_NAMES,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    """Return a temporary anima directory."""
    return tmp_path / "animas" / "test-anima"


@pytest.fixture()
def pipeline(anima_dir: Path) -> ImageGenPipeline:
    """Return an ImageGenPipeline with default config."""
    return ImageGenPipeline(anima_dir)


@pytest.fixture()
def reference_image() -> bytes:
    """Dummy reference image bytes."""
    return b"\x89PNG_FAKE_IMAGE_DATA"


@pytest.fixture()
def generated_bytes() -> bytes:
    """Dummy bytes returned by FluxKontextClient."""
    return b"\x89PNG_GENERATED_RESULT"


# ── Tests ──────────────────────────────────────────────


class TestUnknownExpression:
    """Unknown expression returns None without calling any client."""

    def test_returns_none(self, pipeline: ImageGenPipeline, reference_image: bytes):
        result = pipeline.generate_bustup_expression(
            reference_image=reference_image,
            expression="nonexistent_emotion",
        )
        assert result is None


class TestSkipExisting:
    """skip_existing behaviour when the output file already exists."""

    def test_skip_existing_returns_path(
        self, anima_dir: Path, pipeline: ImageGenPipeline, reference_image: bytes
    ):
        """skip_existing=True returns existing path without generation."""
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        existing = assets_dir / "avatar_bustup.png"
        existing.write_bytes(b"old_data")

        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            result = pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="neutral",
                skip_existing=True,
            )

        assert result == existing
        mock_cls.assert_not_called()

    def test_skip_existing_false_regenerates(
        self,
        anima_dir: Path,
        pipeline: ImageGenPipeline,
        reference_image: bytes,
        generated_bytes: bytes,
    ):
        """skip_existing=False regenerates even if file exists."""
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        existing = assets_dir / "avatar_bustup.png"
        existing.write_bytes(b"old_data")

        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            result = pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="neutral",
                skip_existing=False,
            )

        assert result == existing
        mock_client.generate_from_reference.assert_called_once()
        assert existing.read_bytes() == generated_bytes


class TestStylePrefixSuffix:
    """Style prefix/suffix from config are applied to the prompt."""

    def test_style_prefix_and_suffix_applied(
        self, anima_dir: Path, reference_image: bytes, generated_bytes: bytes
    ):
        config = ImageGenConfig(
            style_prefix="[PREFIX] ",
            style_suffix=" [SUFFIX]",
        )
        pipeline = ImageGenPipeline(anima_dir, config=config)

        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="smile",
            )

        call_kwargs = mock_client.generate_from_reference.call_args
        prompt_sent = call_kwargs.kwargs.get("prompt") or call_kwargs[1].get(
            "prompt", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )
        # Fallback: inspect by keyword
        if prompt_sent is None:
            prompt_sent = call_kwargs.kwargs["prompt"]

        base_prompt = _EXPRESSION_PROMPTS["smile"]
        assert prompt_sent == f"[PREFIX] {base_prompt} [SUFFIX]"


class TestOutputFilename:
    """Output filename follows naming convention."""

    @pytest.mark.parametrize(
        ("expression", "expected_filename"),
        [
            ("neutral", "avatar_bustup.png"),
            ("smile", "avatar_bustup_smile.png"),
            ("laugh", "avatar_bustup_laugh.png"),
            ("troubled", "avatar_bustup_troubled.png"),
            ("surprised", "avatar_bustup_surprised.png"),
            ("thinking", "avatar_bustup_thinking.png"),
            ("embarrassed", "avatar_bustup_embarrassed.png"),
        ],
    )
    def test_output_filename(
        self,
        anima_dir: Path,
        reference_image: bytes,
        generated_bytes: bytes,
        expression: str,
        expected_filename: str,
    ):
        pipeline = ImageGenPipeline(anima_dir)

        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            result = pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression=expression,
            )

        assert result is not None
        assert result.name == expected_filename


class TestFluxKontextClientCall:
    """FluxKontextClient is called with correct parameters."""

    def test_called_with_correct_params(
        self, pipeline: ImageGenPipeline, reference_image: bytes, generated_bytes: bytes
    ):
        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="neutral",
            )

        mock_cls.assert_called_once()
        mock_client.generate_from_reference.assert_called_once_with(
            reference_image=reference_image,
            prompt=_EXPRESSION_PROMPTS["neutral"],
            aspect_ratio="3:4",
            guidance_scale=_EXPRESSION_GUIDANCE["neutral"],
        )


class TestImageBytesWritten:
    """Generated image bytes are written to the correct path."""

    def test_bytes_written_to_output(
        self,
        anima_dir: Path,
        pipeline: ImageGenPipeline,
        reference_image: bytes,
        generated_bytes: bytes,
    ):
        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            result = pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="surprised",
            )

        assert result is not None
        assert result.exists()
        assert result.read_bytes() == generated_bytes


class TestAssetsDirectoryCreation:
    """Assets directory is created if missing (mkdir parents=True)."""

    def test_mkdir_parents_true(
        self,
        anima_dir: Path,
        pipeline: ImageGenPipeline,
        reference_image: bytes,
        generated_bytes: bytes,
    ):
        assets_dir = anima_dir / "assets"
        assert not assets_dir.exists()

        with patch("core.tools.image_gen.FluxKontextClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_from_reference.return_value = generated_bytes
            mock_cls.return_value = mock_client

            result = pipeline.generate_bustup_expression(
                reference_image=reference_image,
                expression="thinking",
            )

        assert assets_dir.is_dir()
        assert result is not None
        assert result.exists()
