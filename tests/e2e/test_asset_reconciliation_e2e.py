"""E2E tests for asset reconciliation pipeline.

Tests the full integration of asset checking and reconciliation as
it would run at server startup and during periodic reconciliation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────


def _create_person_with_assets(
    persons_dir: Path,
    name: str,
    *,
    complete: bool = True,
) -> Path:
    """Create a person directory, optionally with all required assets."""
    from core.asset_reconciler import REQUIRED_ASSETS

    person_dir = persons_dir / name
    person_dir.mkdir(parents=True)
    (person_dir / "identity.md").write_text(
        f"# {name}\nimage_prompt: 1girl, {name}\n", encoding="utf-8",
    )
    if complete:
        assets_dir = person_dir / "assets"
        assets_dir.mkdir()
        for filename in REQUIRED_ASSETS.values():
            (assets_dir / filename).write_bytes(b"fake-data")
    return person_dir


def _make_mock_pipeline_result() -> MagicMock:
    """Create a mock PipelineResult with typical successful output."""
    result = MagicMock()
    result.fullbody_path = Path("/fake/avatar_fullbody.png")
    result.bustup_path = Path("/fake/avatar_bustup.png")
    result.chibi_path = Path("/fake/avatar_chibi.png")
    result.model_path = Path("/fake/avatar_chibi.glb")
    result.rigged_model_path = Path("/fake/avatar_chibi_rigged.glb")
    result.animation_paths = {"idle": Path("/fake/anim_idle.glb")}
    result.errors = []
    result.skipped = []
    return result


# ── Startup reconciliation ───────────────────────────────────────


class TestStartupReconciliation:
    """Test the startup asset reconciliation flow."""

    @pytest.mark.asyncio
    async def test_startup_detects_missing_and_generates(
        self, data_dir: Path,
    ) -> None:
        """Startup reconciliation detects persons with missing assets and
        triggers generation for each one sequentially.
        """
        from core.asset_reconciler import reconcile_all_assets

        persons_dir = data_dir / "persons"

        # One complete, two incomplete
        _create_person_with_assets(persons_dir, "sakura", complete=True)
        _create_person_with_assets(persons_dir, "aoi", complete=False)
        _create_person_with_assets(persons_dir, "rin", complete=False)

        mock_result = _make_mock_pipeline_result()

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            results = await reconcile_all_assets(persons_dir)

        # Only aoi and rin should be processed (sakura is complete)
        assert len(results) == 2
        processed_names = [r["person"] for r in results]
        assert "aoi" in processed_names
        assert "rin" in processed_names
        assert "sakura" not in processed_names

        # Pipeline should be called once per incomplete person
        assert mock_pipeline.generate_all.call_count == 2

        # Verify skip_existing=True was used
        for call in mock_pipeline.generate_all.call_args_list:
            assert call.kwargs["skip_existing"] is True

    @pytest.mark.asyncio
    async def test_startup_noop_when_all_complete(
        self, data_dir: Path,
    ) -> None:
        """Startup is a no-op when all persons have complete assets."""
        from core.asset_reconciler import reconcile_all_assets

        persons_dir = data_dir / "persons"
        _create_person_with_assets(persons_dir, "sakura", complete=True)
        _create_person_with_assets(persons_dir, "kotoha", complete=True)

        results = await reconcile_all_assets(persons_dir)
        assert results == []


# ── Periodic reconciliation ──────────────────────────────────────


class TestPeriodicReconciliation:
    """Test asset reconciliation within the supervisor reconciliation loop."""

    @pytest.mark.asyncio
    async def test_reconcile_detects_newly_missing(
        self, data_dir: Path,
    ) -> None:
        """Periodic reconciliation picks up assets that become missing
        after startup (e.g., bootstrap that just ran and failed).
        """
        from core.asset_reconciler import (
            REQUIRED_ASSETS,
            find_persons_with_missing_assets,
            reconcile_person_assets,
        )

        persons_dir = data_dir / "persons"

        # Start with complete assets
        person_dir = _create_person_with_assets(
            persons_dir, "aoi", complete=True,
        )

        # Verify initially complete
        incomplete = find_persons_with_missing_assets(persons_dir)
        assert len(incomplete) == 0

        # Simulate asset deletion (e.g., corrupted file removed)
        (person_dir / "assets" / "avatar_bustup.png").unlink()

        # Now should detect as incomplete
        incomplete = find_persons_with_missing_assets(persons_dir)
        assert len(incomplete) == 1
        assert incomplete[0][0] == "aoi"
        assert "avatar_bustup" in incomplete[0][1]["missing"]

    @pytest.mark.asyncio
    async def test_reconcile_skips_on_error_and_continues(
        self, data_dir: Path,
    ) -> None:
        """If one person's generation fails, the next person still runs."""
        from core.asset_reconciler import reconcile_all_assets

        persons_dir = data_dir / "persons"
        _create_person_with_assets(persons_dir, "aoi", complete=False)
        _create_person_with_assets(persons_dir, "rin", complete=False)

        mock_result = _make_mock_pipeline_result()
        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API quota exceeded")
            return mock_result

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.side_effect = _side_effect
            mock_cls.return_value = mock_pipeline

            results = await reconcile_all_assets(persons_dir)

        # Both persons should have results
        assert len(results) == 2
        # First should have error
        assert "error" in results[0]
        # Second should succeed
        assert results[1].get("skipped") is False
        assert "error" not in results[1]


# ── Lock mechanism ───────────────────────────────────────────────


class TestLockMechanism:
    """Test the per-person lock prevents concurrent generation."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_serialized(
        self, data_dir: Path,
    ) -> None:
        """Two concurrent reconcile calls for the same person:
        one runs, the other is skipped.
        """
        from core.asset_reconciler import _person_locks, reconcile_person_assets

        persons_dir = data_dir / "persons"
        person_dir = _create_person_with_assets(
            persons_dir, "lock-test", complete=False,
        )

        # Clear any stale lock
        _person_locks.pop("lock-test", None)

        generation_started = asyncio.Event()
        generation_proceed = asyncio.Event()

        mock_result = _make_mock_pipeline_result()

        def _slow_generate(**kwargs):
            generation_started.set()
            # Block until test releases
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(generation_proceed.wait(), timeout=5.0),
            )
            return mock_result

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.side_effect = _slow_generate
            mock_cls.return_value = mock_pipeline

            # Start first reconciliation (will hold lock)
            task1 = asyncio.create_task(reconcile_person_assets(person_dir))

            # Wait for generation to start
            await asyncio.wait_for(generation_started.wait(), timeout=5.0)

            # Start second reconciliation (should be skipped)
            result2 = await reconcile_person_assets(person_dir)

            # Release the first task
            generation_proceed.set()
            result1 = await task1

        # First should have generated
        assert result1.get("skipped") is False
        # Second should be skipped due to lock
        assert result2["skipped"] is True
        assert result2["reason"] == "locked"

    @pytest.mark.asyncio
    async def test_different_persons_not_blocked(
        self, data_dir: Path,
    ) -> None:
        """Different persons use different locks and can run independently."""
        from core.asset_reconciler import _person_locks, reconcile_person_assets

        persons_dir = data_dir / "persons"
        person_a = _create_person_with_assets(
            persons_dir, "person-a", complete=False,
        )
        person_b = _create_person_with_assets(
            persons_dir, "person-b", complete=False,
        )

        # Clear stale locks
        _person_locks.pop("person-a", None)
        _person_locks.pop("person-b", None)

        mock_result = _make_mock_pipeline_result()

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            # Run both concurrently — both should succeed
            results = await asyncio.gather(
                reconcile_person_assets(person_a, prompt="1girl"),
                reconcile_person_assets(person_b, prompt="1girl"),
            )

        # Both should have run (not skipped)
        assert results[0].get("skipped") is False
        assert results[1].get("skipped") is False


# ── Differential generation ──────────────────────────────────────


class TestDifferentialGeneration:
    """Test that only missing assets are generated (skip_existing=True)."""

    @pytest.mark.asyncio
    async def test_skip_existing_passed(self, data_dir: Path) -> None:
        """Pipeline is called with skip_existing=True for differential gen."""
        from core.asset_reconciler import reconcile_person_assets

        persons_dir = data_dir / "persons"
        person_dir = _create_person_with_assets(
            persons_dir, "partial-test", complete=False,
        )
        # Add some assets
        assets_dir = person_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "avatar_fullbody.png").write_bytes(b"existing")

        mock_result = _make_mock_pipeline_result()

        with patch("core.tools.image_gen.ImageGenPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.generate_all.return_value = mock_result
            mock_cls.return_value = mock_pipeline

            await reconcile_person_assets(
                person_dir, prompt="1girl, test",
            )

        mock_pipeline.generate_all.assert_called_once()
        call_kwargs = mock_pipeline.generate_all.call_args.kwargs
        assert call_kwargs["skip_existing"] is True
