"""Pixel-sheet specific reconciler tests."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.anima import asset_reconciler


class TestPixelRequiredAssets:
    def test_pixel_sheet_is_required_when_codex_is_available(self, monkeypatch) -> None:
        monkeypatch.setattr(asset_reconciler, "_pixel_generation_available", lambda: True)
        assert asset_reconciler._required_assets_for_style("anime")["pixel_sheet"] == "pixel_sheet.png"
        assert asset_reconciler._required_assets_for_style("realistic")["pixel_sheet"] == "pixel_sheet.png"

    def test_codex_unavailable_removes_pixel_sheet_from_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(asset_reconciler, "_pixel_generation_available", lambda: False)
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        result = asset_reconciler.check_anima_assets(anima_dir, image_style="anime")

        assert "pixel_sheet" not in result["missing"]
        assert result["complete"] is False


class TestPixelOnlyReconcile:
    @pytest.mark.asyncio
    async def test_pixel_only_missing_anima_runs_pixel_step(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(asset_reconciler, "_pixel_generation_available", lambda: True)
        anima_dir = tmp_path / "pixel-only"
        assets = anima_dir / "assets"
        assets.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Pixel\n", encoding="utf-8")
        for key, filename in asset_reconciler.REQUIRED_ASSETS.items():
            if key == "pixel_sheet":
                continue
            if key in asset_reconciler._3D_ASSET_KEYS:
                continue
            (assets / filename).write_bytes(b"present")

        pixel_path = assets / "pixel_sheet.png"
        mock_result = MagicMock()
        mock_result.fullbody_path = None
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.pixel_sheet_path = pixel_path
        mock_result.errors = []
        mock_result.skipped = []

        with patch("core.integrations.image_gen.ImageGenPipeline") as pipeline_cls:
            pipeline = MagicMock()

            def generate_all(**kwargs):
                pixel_path.write_bytes(b"generated")
                return mock_result

            pipeline.generate_all.side_effect = generate_all
            pipeline_cls.return_value = pipeline

            result = await asset_reconciler.reconcile_anima_assets(
                anima_dir,
                prompt="a pixel character",
                enable_3d=False,
                image_style="anime",
            )

        assert result["skipped"] is False
        assert result["generated"]["generated"] == ["pixel"]
        pipeline.generate_all.assert_called_once_with(
            prompt="a pixel character",
            skip_existing=True,
            steps=["fullbody", "bustup", "chibi", "pixel"],
        )
