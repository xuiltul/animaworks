"""Tests for Diffusers-backed local image generation integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.config.models import ImageGenConfig
from core.tools.image_gen import ImageGenPipeline, dispatch


def test_pipeline_uses_local_diffusers_for_fullbody(tmp_path: Path) -> None:
    config = ImageGenConfig(image_style="anime", backend="diffusers")
    pipeline = ImageGenPipeline(tmp_path, config=config)

    with patch("core.tools._image_pipeline.LocalDiffusersClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_fullbody.return_value = b"PNG-DATA"
        mock_cls.return_value = mock_client

        result = pipeline.generate_all(
            prompt="1girl, black hair",
            skip_existing=False,
            steps=["fullbody"],
        )

    mock_cls.assert_called_once_with(config)
    mock_client.generate_fullbody.assert_called_once()
    assert result.fullbody_path is not None
    assert result.fullbody_path.read_bytes() == b"PNG-DATA"


def test_realistic_diffusers_fullbody_enforces_single_subject(tmp_path: Path) -> None:
    config = ImageGenConfig(image_style="realistic", backend="diffusers")
    pipeline = ImageGenPipeline(tmp_path, config=config)

    with patch("core.tools._image_pipeline.LocalDiffusersClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_fullbody.return_value = b"PNG-DATA"
        mock_cls.return_value = mock_client

        pipeline.generate_all(
            prompt="young Japanese office lady, professional outfit",
            negative_prompt="lowres",
            skip_existing=False,
            steps=["fullbody"],
        )

    kwargs = mock_client.generate_fullbody.call_args.kwargs
    assert "single subject" in kwargs["prompt"]
    assert "one person only" in kwargs["prompt"]
    assert "multiple people" in kwargs["negative_prompt"]
    assert "extra person" in kwargs["negative_prompt"]
    assert "collage" in kwargs["negative_prompt"]
    assert "floating person" in kwargs["negative_prompt"]


def test_pipeline_uses_local_diffusers_for_bustup_expression(tmp_path: Path) -> None:
    config = ImageGenConfig(image_style="anime", backend="diffusers")
    pipeline = ImageGenPipeline(tmp_path, config=config)

    with patch("core.tools._image_pipeline.LocalDiffusersClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"BUSTUP-DATA"
        mock_cls.return_value = mock_client

        result = pipeline.generate_bustup_expression(
            reference_image=b"REFERENCE",
            expression="neutral",
            skip_existing=False,
        )

    mock_cls.assert_called_once_with(config)
    mock_client.generate_from_reference.assert_called_once()
    assert result is not None
    assert result.read_bytes() == b"BUSTUP-DATA"


def test_dispatch_generate_fullbody_uses_local_diffusers(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    image_config = ImageGenConfig(image_style="anime", backend="diffusers")
    fake_config = SimpleNamespace(image_gen=image_config)

    with (
        patch("core.config.models.load_config", return_value=fake_config),
        patch("core.tools.image_gen.LocalDiffusersClient") as mock_cls,
    ):
        mock_client = MagicMock()
        mock_client.generate_fullbody.return_value = b"FULLBODY"
        mock_cls.return_value = mock_client

        result = dispatch(
            "generate_fullbody",
            {
                "anima_dir": str(anima_dir),
                "prompt": "1girl",
            },
        )

    mock_cls.assert_called_once_with(image_config)
    mock_client.generate_fullbody.assert_called_once()
    assert result["size"] == len(b"FULLBODY")
    assert Path(result["path"]).read_bytes() == b"FULLBODY"


def test_dispatch_generate_bustup_uses_local_diffusers(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "avatar_fullbody.png").write_bytes(b"FULLBODY")

    image_config = ImageGenConfig(image_style="anime", backend="diffusers")
    fake_config = SimpleNamespace(image_gen=image_config)

    with (
        patch("core.config.models.load_config", return_value=fake_config),
        patch("core.tools.image_gen.LocalDiffusersClient") as mock_cls,
    ):
        mock_client = MagicMock()
        mock_client.generate_from_reference.return_value = b"BUSTUP"
        mock_cls.return_value = mock_client

        result = dispatch(
            "generate_bustup",
            {
                "anima_dir": str(anima_dir),
            },
        )

    mock_cls.assert_called_once_with(image_config)
    mock_client.generate_from_reference.assert_called_once()
    assert result["size"] == len(b"BUSTUP")
    assert Path(result["path"]).read_bytes() == b"BUSTUP"
