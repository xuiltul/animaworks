"""Unit tests for core.asset_reconciler module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── check_person_assets ──────────────────────────────────────────


class TestCheckPersonAssets:
    """Tests for check_person_assets()."""

    def test_no_assets_dir(self, tmp_path: Path) -> None:
        """Person with no assets/ directory reports all missing."""
        from core.asset_reconciler import check_person_assets

        person_dir = tmp_path / "person"
        person_dir.mkdir()

        result = check_person_assets(person_dir)
        assert result["complete"] is False
        assert result["has_assets_dir"] is False
        assert len(result["missing"]) == 5
        assert result["present"] == []

    def test_empty_assets_dir(self, tmp_path: Path) -> None:
        """Person with empty assets/ directory reports all missing."""
        from core.asset_reconciler import check_person_assets

        person_dir = tmp_path / "person"
        (person_dir / "assets").mkdir(parents=True)

        result = check_person_assets(person_dir)
        assert result["complete"] is False
        assert result["has_assets_dir"] is True
        assert len(result["missing"]) == 5

    def test_all_assets_present(self, tmp_path: Path) -> None:
        """Person with all required assets reports complete."""
        from core.asset_reconciler import REQUIRED_ASSETS, check_person_assets

        person_dir = tmp_path / "person"
        assets_dir = person_dir / "assets"
        assets_dir.mkdir(parents=True)
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        result = check_person_assets(person_dir)
        assert result["complete"] is True
        assert result["missing"] == []
        assert len(result["present"]) == 5

    def test_partial_assets(self, tmp_path: Path) -> None:
        """Person with some assets reports the correct missing ones."""
        from core.asset_reconciler import check_person_assets

        person_dir = tmp_path / "person"
        assets_dir = person_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "avatar_fullbody.png").write_bytes(b"fake")
        (assets_dir / "avatar_bustup.png").write_bytes(b"fake")

        result = check_person_assets(person_dir)
        assert result["complete"] is False
        assert "avatar_fullbody" in result["present"]
        assert "avatar_bustup" in result["present"]
        assert "avatar_chibi" in result["missing"]
        assert "model_chibi" in result["missing"]
        assert "model_rigged" in result["missing"]


# ── find_persons_with_missing_assets ─────────────────────────────


class TestFindPersonsWithMissingAssets:
    """Tests for find_persons_with_missing_assets()."""

    def test_no_persons_dir(self, tmp_path: Path) -> None:
        """Non-existent persons_dir returns empty."""
        from core.asset_reconciler import find_persons_with_missing_assets

        result = find_persons_with_missing_assets(tmp_path / "nonexistent")
        assert result == []

    def test_all_complete(self, tmp_path: Path) -> None:
        """All persons with complete assets returns empty."""
        from core.asset_reconciler import REQUIRED_ASSETS, find_persons_with_missing_assets

        persons_dir = tmp_path / "persons"
        for name in ("alice", "bob"):
            person_dir = persons_dir / name
            assets_dir = person_dir / "assets"
            assets_dir.mkdir(parents=True)
            (person_dir / "identity.md").write_text("# Test", encoding="utf-8")
            for filename in REQUIRED_ASSETS.values():
                (assets_dir / filename).write_bytes(b"fake")

        result = find_persons_with_missing_assets(persons_dir)
        assert result == []

    def test_mixed(self, tmp_path: Path) -> None:
        """Mix of complete and incomplete persons returns only incomplete."""
        from core.asset_reconciler import REQUIRED_ASSETS, find_persons_with_missing_assets

        persons_dir = tmp_path / "persons"

        # Complete person
        complete_dir = persons_dir / "alice"
        assets_dir = complete_dir / "assets"
        assets_dir.mkdir(parents=True)
        (complete_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        # Incomplete person
        incomplete_dir = persons_dir / "bob"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "identity.md").write_text("# Bob", encoding="utf-8")

        result = find_persons_with_missing_assets(persons_dir)
        assert len(result) == 1
        assert result[0][0] == "bob"

    def test_skips_dirs_without_identity(self, tmp_path: Path) -> None:
        """Directories without identity.md are ignored."""
        from core.asset_reconciler import find_persons_with_missing_assets

        persons_dir = tmp_path / "persons"
        (persons_dir / "notaperson").mkdir(parents=True)

        result = find_persons_with_missing_assets(persons_dir)
        assert result == []


# ── _extract_prompt ──────────────────────────────────────────────


class TestExtractPrompt:
    """Tests for _extract_prompt() helper."""

    def test_extracts_image_prompt(self, tmp_path: Path) -> None:
        """Extracts prompt from image_prompt field."""
        from core.asset_reconciler import _extract_prompt

        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text(
            "# Alice\nimage_prompt: 1girl, black hair, red eyes\n",
            encoding="utf-8",
        )
        result = _extract_prompt(person_dir)
        assert result == "1girl, black hair, red eyes"

    def test_extracts_japanese_prompt(self, tmp_path: Path) -> None:
        """Extracts prompt from 外見 field."""
        from core.asset_reconciler import _extract_prompt

        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text(
            "# Alice\n外見: 1girl, blue hair\n",
            encoding="utf-8",
        )
        result = _extract_prompt(person_dir)
        assert result == "1girl, blue hair"

    def test_returns_none_no_prompt(self, tmp_path: Path) -> None:
        """Returns None when no prompt field found."""
        from core.asset_reconciler import _extract_prompt

        person_dir = tmp_path / "person"
        person_dir.mkdir()
        (person_dir / "identity.md").write_text(
            "# Alice\nJust a person.\n",
            encoding="utf-8",
        )
        result = _extract_prompt(person_dir)
        assert result is None

    def test_returns_none_no_identity(self, tmp_path: Path) -> None:
        """Returns None when identity.md doesn't exist."""
        from core.asset_reconciler import _extract_prompt

        person_dir = tmp_path / "person"
        person_dir.mkdir()
        result = _extract_prompt(person_dir)
        assert result is None


# ── reconcile_person_assets ──────────────────────────────────────


class TestReconcilePersonAssets:
    """Tests for reconcile_person_assets()."""

    @pytest.mark.asyncio
    async def test_skips_complete_person(self, tmp_path: Path) -> None:
        """Skips generation if all assets are present."""
        from core.asset_reconciler import REQUIRED_ASSETS, reconcile_person_assets

        person_dir = tmp_path / "person"
        assets_dir = person_dir / "assets"
        assets_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text("# Test", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        result = await reconcile_person_assets(person_dir)
        assert result["skipped"] is True
        assert result["reason"] == "complete"

    @pytest.mark.asyncio
    async def test_skips_no_prompt(self, tmp_path: Path) -> None:
        """Skips generation if no prompt can be extracted."""
        from core.asset_reconciler import reconcile_person_assets

        person_dir = tmp_path / "person"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text(
            "# Test\nNo prompt here.", encoding="utf-8",
        )

        result = await reconcile_person_assets(person_dir)
        assert result["skipped"] is True
        assert result["reason"] == "no_prompt"

    @pytest.mark.asyncio
    async def test_uses_provided_prompt(self, tmp_path: Path) -> None:
        """Uses the explicitly provided prompt instead of extracting."""
        from core.asset_reconciler import reconcile_person_assets

        person_dir = tmp_path / "person"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text("# Test", encoding="utf-8")

        mock_result = MagicMock()
        mock_result.fullbody_path = person_dir / "assets" / "avatar_fullbody.png"
        mock_result.bustup_path = None
        mock_result.chibi_path = None
        mock_result.model_path = None
        mock_result.rigged_model_path = None
        mock_result.animation_paths = {}
        mock_result.errors = []
        mock_result.skipped = []

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            result = await reconcile_person_assets(
                person_dir, prompt="1girl, test prompt",
            )

        assert result["skipped"] is False
        mock_pipeline.generate_all.assert_called_once_with(
            prompt="1girl, test prompt",
            skip_existing=True,
        )

    @pytest.mark.asyncio
    async def test_handles_generation_error(self, tmp_path: Path) -> None:
        """Returns error info when generation fails."""
        from core.asset_reconciler import reconcile_person_assets

        person_dir = tmp_path / "person"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text(
            "# Test\nimage_prompt: 1girl\n", encoding="utf-8",
        )

        with patch(
            "core.tools.image_gen.ImageGenPipeline",
            side_effect=RuntimeError("API down"),
        ):
            result = await reconcile_person_assets(person_dir)

        assert result["skipped"] is False
        assert "error" in result
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent(self, tmp_path: Path) -> None:
        """Lock mechanism prevents concurrent generation for the same person."""
        from core.asset_reconciler import _get_lock, reconcile_person_assets

        person_dir = tmp_path / "concurrent-test"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text(
            "# Test\nimage_prompt: 1girl\n", encoding="utf-8",
        )

        lock = _get_lock("concurrent-test")

        # Hold the lock to simulate ongoing generation
        async with lock:
            result = await reconcile_person_assets(person_dir)

        assert result["skipped"] is True
        assert result["reason"] == "locked"


# ── reconcile_all_assets ─────────────────────────────────────────


class TestReconcileAllAssets:
    """Tests for reconcile_all_assets()."""

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, tmp_path: Path) -> None:
        """No incomplete persons returns empty list."""
        from core.asset_reconciler import REQUIRED_ASSETS, reconcile_all_assets

        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "alice"
        assets_dir = person_dir / "assets"
        assets_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake")

        results = await reconcile_all_assets(persons_dir)
        assert results == []

    @pytest.mark.asyncio
    async def test_processes_incomplete_sequentially(self, tmp_path: Path) -> None:
        """Processes each incomplete person and returns results."""
        from core.asset_reconciler import reconcile_all_assets

        persons_dir = tmp_path / "persons"
        for name in ("aoi", "rin"):
            person_dir = persons_dir / name
            person_dir.mkdir(parents=True)
            (person_dir / "identity.md").write_text(
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

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            results = await reconcile_all_assets(persons_dir)

        assert len(results) == 2
        assert mock_pipeline.generate_all.call_count == 2

    @pytest.mark.asyncio
    async def test_broadcasts_on_generation(self, tmp_path: Path) -> None:
        """Broadcasts WebSocket event when assets are generated."""
        from core.asset_reconciler import reconcile_all_assets

        persons_dir = tmp_path / "persons"
        person_dir = persons_dir / "aoi"
        person_dir.mkdir(parents=True)
        (person_dir / "identity.md").write_text(
            "# Aoi\nimage_prompt: 1girl\n", encoding="utf-8",
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

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            await reconcile_all_assets(persons_dir, ws_manager=ws_manager)

        ws_manager.broadcast.assert_called_once_with(
            "person.assets_updated",
            {"name": "aoi", "source": "reconciliation"},
        )
