"""Unit tests for core.anima.asset_reconciler module."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_asset_failure_records():
    """Isolate the module-level failure registry between tests."""
    from core.anima.asset_reconciler import _asset_failure_records

    _asset_failure_records.clear()
    yield
    _asset_failure_records.clear()


@pytest.fixture(autouse=True)
def _assume_pixel_generator_available(monkeypatch):
    """Keep required-asset counts deterministic regardless of the host's codex install."""
    from core.anima import asset_reconciler

    monkeypatch.setattr(asset_reconciler, "_pixel_generation_available", lambda: True)


# ── check_anima_assets ──────────────────────────────────────────


class TestCheckAnimaAssets:
    """Tests for check_anima_assets()."""

    def test_no_assets_dir(self, tmp_path: Path) -> None:
        """Anima with no assets/ directory reports all missing."""
        from core.anima.asset_reconciler import REALISTIC_REQUIRED_ASSETS, check_anima_assets

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()

        result = check_anima_assets(anima_dir)
        assert result["complete"] is False
        assert result["has_assets_dir"] is False
        assert len(result["missing"]) == len(REALISTIC_REQUIRED_ASSETS)
        assert result["present"] == []

    def test_empty_assets_dir(self, tmp_path: Path) -> None:
        """Anima with empty assets/ directory reports all missing."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, check_anima_assets

        anima_dir = tmp_path / "anima"
        (anima_dir / "assets").mkdir(parents=True)

        result = check_anima_assets(anima_dir, image_style="anime")
        assert result["complete"] is False
        assert result["has_assets_dir"] is True
        assert len(result["missing"]) == len(REQUIRED_ASSETS)

    def test_all_assets_present(self, tmp_path: Path) -> None:
        """Anima with all required assets reports complete."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, check_anima_assets

        anima_dir = tmp_path / "anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        result = check_anima_assets(anima_dir, image_style="anime")
        assert result["complete"] is True
        assert result["missing"] == []
        assert len(result["present"]) == len(REQUIRED_ASSETS)

    def test_partial_assets(self, tmp_path: Path) -> None:
        """Anima with some assets reports the correct missing ones."""
        from core.anima.asset_reconciler import check_anima_assets

        anima_dir = tmp_path / "anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "avatar_fullbody.png").write_bytes(b"fake")
        (assets_dir / "avatar_bustup.png").write_bytes(b"fake")

        result = check_anima_assets(anima_dir, image_style="anime")
        assert result["complete"] is False
        assert "avatar_fullbody" in result["present"]
        assert "avatar_bustup" in result["present"]
        assert "avatar_chibi" in result["missing"]
        assert "model_chibi" in result["missing"]
        assert "model_rigged" in result["missing"]


# ── find_animas_with_missing_assets ─────────────────────────────


class TestFindAnimasWithMissingAssets:
    """Tests for find_animas_with_missing_assets()."""

    def test_no_animas_dir(self, tmp_path: Path) -> None:
        """Non-existent animas_dir returns empty."""
        from core.anima.asset_reconciler import find_animas_with_missing_assets

        result = find_animas_with_missing_assets(tmp_path / "nonexistent")
        assert result == []

    def test_all_complete(self, tmp_path: Path) -> None:
        """All animas with complete assets returns empty."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, find_animas_with_missing_assets

        animas_dir = tmp_path / "animas"
        for name in ("alice", "bob"):
            anima_dir = animas_dir / name
            assets_dir = anima_dir / "assets"
            assets_dir.mkdir(parents=True)
            (anima_dir / "IDENTITY.md").write_text("# Test", encoding="utf-8")
            for filename in REQUIRED_ASSETS.values():
                (assets_dir / filename).write_bytes(b"fake")

        result = find_animas_with_missing_assets(animas_dir, image_style="anime")
        assert result == []

    def test_mixed(self, tmp_path: Path) -> None:
        """Mix of complete and incomplete animas returns only incomplete."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, find_animas_with_missing_assets

        animas_dir = tmp_path / "animas"

        # Complete anima
        complete_dir = animas_dir / "alice"
        assets_dir = complete_dir / "assets"
        assets_dir.mkdir(parents=True)
        (complete_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        # Incomplete anima
        incomplete_dir = animas_dir / "bob"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "IDENTITY.md").write_text("# Bob", encoding="utf-8")

        result = find_animas_with_missing_assets(animas_dir, image_style="anime")
        assert len(result) == 1
        assert result[0][0] == "bob"

    def test_skips_dirs_without_identity(self, tmp_path: Path) -> None:
        """Directories without IDENTITY.md are ignored."""
        from core.anima.asset_reconciler import find_animas_with_missing_assets

        animas_dir = tmp_path / "animas"
        (animas_dir / "notananima").mkdir(parents=True)

        result = find_animas_with_missing_assets(animas_dir)
        assert result == []


# ── _extract_prompt ──────────────────────────────────────────────


class TestExtractPrompt:
    """Tests for _extract_prompt() helper."""

    @pytest.mark.asyncio
    async def test_extracts_image_prompt(self, tmp_path: Path) -> None:
        """Extracts prompt from image_prompt field."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Alice\nimage_prompt: 1girl, black hair, red eyes\n",
            encoding="utf-8",
        )
        result = await _extract_prompt(anima_dir)
        assert result == "1girl, black hair, red eyes"

    @pytest.mark.asyncio
    async def test_extracts_japanese_prompt(self, tmp_path: Path) -> None:
        """Extracts prompt from 外見 field."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Alice\n外見: 1girl, blue hair\n",
            encoding="utf-8",
        )
        result = await _extract_prompt(anima_dir)
        assert result == "1girl, blue hair"

    @pytest.mark.asyncio
    async def test_returns_none_no_prompt(self, tmp_path: Path) -> None:
        """Returns None when no prompt field and no appearance table found."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Alice\nJust an anima.\n",
            encoding="utf-8",
        )
        with patch("core.anima.asset_reconciler._synthesize_prompt_via_llm", new_callable=AsyncMock, return_value=None):
            result = await _extract_prompt(anima_dir)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_no_identity(self, tmp_path: Path) -> None:
        """Returns None when IDENTITY.md doesn't exist."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        result = await _extract_prompt(anima_dir)
        assert result is None

    @pytest.mark.asyncio
    async def test_reads_cached_prompt_txt(self, tmp_path: Path) -> None:
        """Reads prompt from assets/prompt.txt when no regex match."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Alice\nNo prompt field here.\n",
            encoding="utf-8",
        )
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir()
        (assets_dir / "prompt.txt").write_text(
            "1girl, cached prompt\n",
            encoding="utf-8",
        )
        result = await _extract_prompt(anima_dir)
        assert result == "1girl, cached prompt"

    @pytest.mark.asyncio
    async def test_synthesizes_via_llm(self, tmp_path: Path) -> None:
        """Calls LLM when no regex match and no cache, with appearance table."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\n\n| 項目 | 設定 |\n|------|------|\n| 髪色 | 黒 |\n| 瞳の色 | 赤 |\n",
            encoding="utf-8",
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="1girl, black hair, red eyes, full body, standing, white background"))
        ]

        mock_model_config = MagicMock()
        mock_model_config.model = "anthropic/claude-sonnet-4-6"
        mock_model_config.api_key = "test-key"
        mock_model_config.api_key_env = "ANTHROPIC_API_KEY"
        mock_model_config.api_base_url = None

        with (
            patch("core.config.models.load_model_config", return_value=mock_model_config),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
        ):
            result = await _extract_prompt(anima_dir)

        assert result == "1girl, black hair, red eyes, full body, standing, white background"

    @pytest.mark.asyncio
    async def test_saves_to_prompt_txt(self, tmp_path: Path) -> None:
        """LLM synthesis result is saved to assets/prompt.txt."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\n| 髪色 | 青 |\n",
            encoding="utf-8",
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="1girl, blue hair, full body, standing, white background"))
        ]

        mock_model_config = MagicMock()
        mock_model_config.model = "anthropic/claude-sonnet-4-6"
        mock_model_config.api_key = "test-key"
        mock_model_config.api_key_env = "ANTHROPIC_API_KEY"
        mock_model_config.api_base_url = None

        with (
            patch("core.config.models.load_model_config", return_value=mock_model_config),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
        ):
            await _extract_prompt(anima_dir)

        cache_file = anima_dir / "assets" / "prompt.txt"
        assert cache_file.exists()
        assert "1girl, blue hair" in cache_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, tmp_path: Path) -> None:
        """Returns None when LLM call fails (graceful degradation)."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\n| 髪色 | 黒 |\n",
            encoding="utf-8",
        )

        mock_model_config = MagicMock()
        mock_model_config.model = "anthropic/claude-sonnet-4-6"
        mock_model_config.api_key = "test-key"
        mock_model_config.api_key_env = "ANTHROPIC_API_KEY"
        mock_model_config.api_base_url = None

        with (
            patch("core.config.models.load_model_config", return_value=mock_model_config),
            patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("API error")),
            patch(
                "core.llm.oneshot._try_agent_sdk",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Agent SDK error"),
            ),
        ):
            result = await _extract_prompt(anima_dir)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_appearance_table_returns_none(self, tmp_path: Path) -> None:
        """Returns None when IDENTITY.md has no appearance table."""
        from core.anima.asset_reconciler import _extract_prompt

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\n\nJust personality description, no table.\n",
            encoding="utf-8",
        )
        with patch("core.anima.asset_reconciler._synthesize_prompt_via_llm", new_callable=AsyncMock, return_value=None):
            result = await _extract_prompt(anima_dir)
        assert result is None

    def test_prompt_synthesis_model_ignores_local_and_anima_models(self, tmp_path: Path) -> None:
        from core.anima.asset_reconciler import _resolve_prompt_synthesis_model
        from core.config.schemas import AnimaWorksConfig

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        config = AnimaWorksConfig.model_validate(
            {
                "anima_defaults": {
                    "model": "codex/gpt-5.4-mini",
                    "credential": "openai",
                },
                "local_llm": {"default_model": "ollama/qwen2.5-coder:14b"},
            }
        )

        with patch("core.config.io.load_config", return_value=config):
            result = _resolve_prompt_synthesis_model(anima_dir)

        assert result == ("claude-sonnet-5-5", "anthropic")

    def test_prompt_synthesis_model_uses_legacy_helper_model_and_credential(self, tmp_path: Path) -> None:
        from core.anima.asset_reconciler import _resolve_prompt_synthesis_model
        from core.config.schemas import AnimaWorksConfig

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        config = AnimaWorksConfig.model_validate(
            {"consolidation": {"llm_model": "openai/qwen3.6-35b-a3b", "llm_credential": "vllm-lb"}}
        )

        with patch("core.config.io.load_config", return_value=config):
            result = _resolve_prompt_synthesis_model(anima_dir)

        assert result == ("openai/qwen3.6-35b-a3b", "vllm-lb")

    def test_prompt_synthesis_model_falls_back_to_registry_code_default(self, tmp_path: Path) -> None:
        from core.anima.asset_reconciler import _resolve_prompt_synthesis_model
        from core.config.schemas import AnimaWorksConfig

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        config = AnimaWorksConfig.model_validate({"anima_defaults": {"model": "codex/gpt-5.4-mini"}})

        with patch("core.config.io.load_config", return_value=config):
            result = _resolve_prompt_synthesis_model(anima_dir)

        assert result == ("claude-sonnet-5-5", "anthropic")


# ── reconcile_anima_assets ──────────────────────────────────────


class TestReconcileAnimaAssets:
    """Tests for reconcile_anima_assets()."""

    @pytest.mark.asyncio
    async def test_skips_complete_anima(self, tmp_path: Path) -> None:
        """Skips generation if all assets are present."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, reconcile_anima_assets

        anima_dir = tmp_path / "anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Test", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        result = await reconcile_anima_assets(anima_dir, image_style="anime")
        assert result["skipped"] is True
        assert result["reason"] == "complete"

    @pytest.mark.asyncio
    async def test_no_prompt_is_logged_once_and_not_retried(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing prompt is informational and unchanged inputs are not retried."""
        import logging

        from core.anima.asset_reconciler import reconcile_anima_assets

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\nNo prompt here.",
            encoding="utf-8",
        )
        caplog.set_level(logging.INFO, logger="animaworks.asset_reconciler")

        with patch(
            "core.anima.asset_reconciler._synthesize_prompt_via_llm",
            new_callable=AsyncMock,
            return_value=None,
        ) as synthesize:
            first_result = await reconcile_anima_assets(anima_dir)
            second_result = await reconcile_anima_assets(anima_dir)

        assert first_result["skipped"] is True
        assert first_result["reason"] == "no_prompt"
        assert second_result["skipped"] is True
        assert second_result["reason"] == "previous_failure"
        synthesize.assert_awaited_once()
        messages = [record.message for record in caplog.records if "No prompt available" in record.message]
        assert len(messages) == 1
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    @pytest.mark.asyncio
    async def test_uses_provided_prompt(self, tmp_path: Path) -> None:
        """Uses the explicitly provided prompt instead of extracting."""
        from core.anima.asset_reconciler import reconcile_anima_assets

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Test", encoding="utf-8")

        mock_result = MagicMock()
        mock_result.fullbody_path = anima_dir / "assets" / "avatar_fullbody.png"
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.errors = []
        mock_result.skipped = []

        with patch("core.integrations.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            result = await reconcile_anima_assets(
                anima_dir,
                prompt="1girl, test prompt",
            )

        assert result["skipped"] is False
        mock_pipeline.generate_all.assert_called_once_with(
            prompt="1girl, test prompt",
            skip_existing=True,
            steps=None,
        )

    @pytest.mark.asyncio
    async def test_handles_generation_error(self, tmp_path: Path) -> None:
        """Returns error info when generation fails."""
        from core.anima.asset_reconciler import reconcile_anima_assets

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\nimage_prompt: 1girl\n",
            encoding="utf-8",
        )

        with patch(
            "core.integrations.image_gen.ImageGenPipeline",
            side_effect=RuntimeError("API down"),
        ):
            result = await reconcile_anima_assets(anima_dir)

        assert result["skipped"] is False
        assert "error" in result
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_generation_failure_is_not_retried_for_unchanged_inputs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from core.anima.asset_reconciler import reconcile_anima_assets

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Test\nimage_prompt: 1girl\n", encoding="utf-8")
        caplog.set_level(logging.WARNING, logger="animaworks.asset_reconciler")

        with patch("core.integrations.image_gen.ImageGenPipeline", side_effect=RuntimeError("API down")) as pipeline:
            first_result = await reconcile_anima_assets(anima_dir)
            second_result = await reconcile_anima_assets(anima_dir)

        assert first_result["skipped"] is False
        assert "API down" in first_result["error"]
        assert second_result["skipped"] is True
        assert second_result["reason"] == "previous_failure"
        pipeline.assert_called_once()
        failures = [record for record in caplog.records if "Asset generation failed" in record.message]
        assert len(failures) == 1

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent(self, tmp_path: Path) -> None:
        """Lock mechanism prevents concurrent generation for the same anima."""
        from core.anima.asset_reconciler import _get_lock, reconcile_anima_assets

        anima_dir = tmp_path / "concurrent-test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text(
            "# Test\nimage_prompt: 1girl\n",
            encoding="utf-8",
        )

        lock = _get_lock("concurrent-test")

        # Hold the lock to simulate ongoing generation
        async with lock:
            result = await reconcile_anima_assets(anima_dir)

        assert result["skipped"] is True
        assert result["reason"] == "locked"


# ── reconcile_all_assets ─────────────────────────────────────────


class TestReconcileAllAssets:
    """Tests for reconcile_all_assets()."""

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, tmp_path: Path) -> None:
        """No incomplete animas returns empty list."""
        from core.anima.asset_reconciler import REQUIRED_ASSETS, reconcile_all_assets

        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "alice"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        results = await reconcile_all_assets(animas_dir, image_style="anime")
        assert results == []

    @pytest.mark.asyncio
    async def test_processes_incomplete_sequentially(self, tmp_path: Path) -> None:
        """Processes each incomplete anima and returns results."""
        from core.anima.asset_reconciler import reconcile_all_assets

        animas_dir = tmp_path / "animas"
        for name in ("aoi", "rin"):
            anima_dir = animas_dir / name
            anima_dir.mkdir(parents=True)
            (anima_dir / "IDENTITY.md").write_text(
                f"# {name}\nimage_prompt: 1girl, {name}\n",
                encoding="utf-8",
            )

        mock_result = MagicMock()
        mock_result.fullbody_path = Path("/fake/fullbody.png")
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.errors = []
        mock_result.skipped = []

        with patch("core.integrations.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            results = await reconcile_all_assets(animas_dir)

        assert len(results) == 2
        assert mock_pipeline.generate_all.call_count == 2

    @pytest.mark.asyncio
    async def test_broadcasts_on_generation(self, tmp_path: Path) -> None:
        """Broadcasts WebSocket event when assets are generated."""
        from core.anima.asset_reconciler import reconcile_all_assets

        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "aoi"
        anima_dir.mkdir(parents=True)
        (anima_dir / "IDENTITY.md").write_text(
            "# Aoi\nimage_prompt: 1girl\n",
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.fullbody_path = Path("/fake/fullbody.png")
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.errors = []
        mock_result.skipped = []

        ws_manager = AsyncMock()

        with patch("core.integrations.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            await reconcile_all_assets(animas_dir, ws_manager=ws_manager)

        ws_manager.broadcast.assert_called_once_with(
            "anima.assets_updated",
            {"name": "aoi", "source": "reconciliation"},
        )


# ── _extract_appearance_field ────────────────────────────────────


class TestExtractAppearanceField:
    """Bold-marker tolerance and multi-line bullet sections (yoru regression)."""

    def test_bold_label_inline_value(self) -> None:
        from core.anima.asset_reconciler import _extract_appearance_field

        assert _extract_appearance_field("- **外見**: 銀髪ショート") == "銀髪ショート"

    def test_bold_label_multiline_bullets(self) -> None:
        from core.anima.asset_reconciler import _extract_appearance_field

        text = "- **性格**: 大人びている\n- **外見**:\n  - 髪: 漆黒のロング\n  - 瞳: ガーネット\n- **口調**: 丁寧語\n"
        result = _extract_appearance_field(text)
        assert result == "髪: 漆黒のロング、瞳: ガーネット"

    def test_multiline_stops_at_blank_line(self) -> None:
        from core.anima.asset_reconciler import _extract_appearance_field

        text = "外見:\n  - 銀髪\n\n  - これは別セクション\n"
        assert _extract_appearance_field(text) == "銀髪"

    def test_no_label_returns_none(self) -> None:
        from core.anima.asset_reconciler import _extract_appearance_field

        assert _extract_appearance_field("性格: 優しい\n") is None


async def test_first_profile_wait_does_not_record_asset_failure(tmp_path):
    from core.anima.asset_reconciler import _asset_failure_records, reconcile_anima_assets
    from core.anima.bootstrap_state import get_bootstrap_status

    anima_dir = tmp_path / "newcomer"
    anima_dir.mkdir()
    (anima_dir / "IDENTITY.md").write_text("# Identity\n未定義\n", encoding="utf-8")
    (anima_dir / "AGENTS.md").write_text("未定義\n", encoding="utf-8")
    (anima_dir / "BOOTSTRAP.md").write_text("First conversation", encoding="utf-8")
    assert get_bootstrap_status(anima_dir)["needs_user_input"]
    with patch("core.anima.asset_reconciler._extract_prompt", new_callable=AsyncMock) as extract:
        result = await reconcile_anima_assets(anima_dir)
        assert result["reason"] == "awaiting_profile"
        extract.assert_not_awaited()
    assert _asset_failure_records == {}


def test_asset_helper_ignores_codex_anima_model(tmp_path):
    from core.anima.asset_reconciler import _resolve_prompt_synthesis_model
    from core.config import AnimaWorksConfig

    config = AnimaWorksConfig()
    config.anima_defaults.model = "codex/account-model"
    config.anima_defaults.credential = "openai"
    with patch("core.config.io.load_config", return_value=config):
        assert _resolve_prompt_synthesis_model(tmp_path) == ("claude-sonnet-5-5", "anthropic")
