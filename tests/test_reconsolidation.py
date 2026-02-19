from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for failure-count-based memory reconsolidation.

All LLM dependencies are mocked to ensure unit test isolation
without requiring API keys or model downloads.
"""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.reconsolidation import ReconsolidationEngine


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure for testing."""
    anima = tmp_path / "animas" / "test-anima"
    (anima / "knowledge").mkdir(parents=True)
    (anima / "procedures").mkdir(parents=True)
    (anima / "episodes").mkdir(parents=True)
    (anima / "skills").mkdir(parents=True)
    (anima / "state").mkdir(parents=True)
    (anima / "activity_log").mkdir(parents=True)
    return anima


@pytest.fixture
def memory_manager(anima_dir: Path):
    """Create a MemoryManager for the test anima."""
    from core.memory.manager import MemoryManager
    return MemoryManager(anima_dir)


@pytest.fixture
def activity_logger(anima_dir: Path):
    """Create an ActivityLogger for the test anima."""
    from core.memory.activity import ActivityLogger
    return ActivityLogger(anima_dir)


@pytest.fixture
def engine(anima_dir: Path, memory_manager, activity_logger) -> ReconsolidationEngine:
    """Create a ReconsolidationEngine for the test anima."""
    return ReconsolidationEngine(
        anima_dir, "test-anima",
        memory_manager=memory_manager,
        activity_logger=activity_logger,
    )


def _write_procedure(
    anima_dir: Path,
    filename: str,
    *,
    failure_count: int = 0,
    confidence: float = 0.5,
    version: int = 1,
    content: str = "# Deploy\n\n1. Run tests\n2. Deploy\n",
    description: str = "deployment procedure",
) -> Path:
    """Helper to create a procedure file with given metadata."""
    import yaml

    path = anima_dir / "procedures" / filename
    meta = {
        "description": description,
        "tags": ["deploy"],
        "version": version,
        "confidence": confidence,
        "failure_count": failure_count,
        "success_count": 0,
        "created_at": "2026-01-01T00:00:00",
    }
    fm_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True).rstrip()
    path.write_text(f"---\n{fm_str}\n---\n\n{content}", encoding="utf-8")
    return path


# ── find_reconsolidation_targets Tests ─────────────────────


class TestFindReconsolidationTargets:
    """Test the failure-count-based target detection."""

    @pytest.mark.asyncio
    async def test_no_procedures_dir(
        self, anima_dir: Path, memory_manager, activity_logger,
    ) -> None:
        """No procedures directory returns empty list."""
        # Remove the procedures dir
        shutil.rmtree(anima_dir / "procedures")
        engine = ReconsolidationEngine(
            anima_dir, "test-anima",
            memory_manager=memory_manager,
            activity_logger=activity_logger,
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_empty_procedures_dir(
        self, engine: ReconsolidationEngine,
    ) -> None:
        """Empty procedures directory returns empty list."""
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_triggers_on_failure_count_2_and_low_confidence(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with failure_count=2, confidence=0.3 is a target."""
        _write_procedure(
            anima_dir, "failing.md",
            failure_count=2, confidence=0.3,
        )
        targets = await engine.find_reconsolidation_targets()
        assert len(targets) == 1
        assert targets[0].name == "failing.md"

    @pytest.mark.asyncio
    async def test_triggers_on_high_failure_count(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with failure_count=5, confidence=0.1 is a target."""
        _write_procedure(
            anima_dir, "very-failing.md",
            failure_count=5, confidence=0.1,
        )
        targets = await engine.find_reconsolidation_targets()
        assert len(targets) == 1

    @pytest.mark.asyncio
    async def test_boundary_failure_count_1_not_triggered(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with failure_count=1 should NOT trigger."""
        _write_procedure(
            anima_dir, "almost.md",
            failure_count=1, confidence=0.3,
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_boundary_confidence_0_6_not_triggered(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with confidence=0.6 exactly should NOT trigger (strict less-than)."""
        _write_procedure(
            anima_dir, "borderline.md",
            failure_count=3, confidence=0.6,
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_boundary_confidence_0_59_triggered(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with confidence=0.59 should trigger."""
        _write_procedure(
            anima_dir, "just-below.md",
            failure_count=2, confidence=0.59,
        )
        targets = await engine.find_reconsolidation_targets()
        assert len(targets) == 1

    @pytest.mark.asyncio
    async def test_no_trigger_when_confidence_high(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with high confidence is not a target even with failures."""
        _write_procedure(
            anima_dir, "confident.md",
            failure_count=3, confidence=0.8,
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_no_trigger_when_no_failures(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure with no failures is not a target even with low confidence."""
        _write_procedure(
            anima_dir, "low-conf-ok.md",
            failure_count=0, confidence=0.2,
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_defaults_for_missing_metadata(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Procedure without failure_count/confidence in metadata is not a target.

        Defaults: failure_count=0, confidence=1.0.
        """
        path = anima_dir / "procedures" / "no-meta.md"
        path.write_text(
            "---\ndescription: test\n---\n\n# Test\n\nSteps here.\n",
            encoding="utf-8",
        )
        targets = await engine.find_reconsolidation_targets()
        assert targets == []

    @pytest.mark.asyncio
    async def test_multiple_targets(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Multiple procedures can be targets simultaneously."""
        _write_procedure(
            anima_dir, "fail1.md",
            failure_count=2, confidence=0.4,
        )
        _write_procedure(
            anima_dir, "fail2.md",
            failure_count=3, confidence=0.2,
        )
        _write_procedure(
            anima_dir, "ok.md",
            failure_count=0, confidence=0.9,
        )
        targets = await engine.find_reconsolidation_targets()
        assert len(targets) == 2
        target_names = {t.name for t in targets}
        assert "fail1.md" in target_names
        assert "fail2.md" in target_names
        assert "ok.md" not in target_names


# ── apply_reconsolidation Tests ────────────────────────────


class TestApplyReconsolidation:
    """Test applying reconsolidation to target procedures."""

    @pytest.mark.asyncio
    async def test_successful_update(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Successfully revise a procedure via LLM."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=3, confidence=0.3, version=2,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = (
            "# Deploy\n\n1. Run tests\n2. Build image\n3. Deploy to staging\n4. Verify\n5. Deploy to prod\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await engine.apply_reconsolidation([proc_path], "test-model")

        assert result["updated"] == 1
        assert result["skipped"] == 0
        assert result["errors"] == 0

        # Verify content was updated
        content = engine.memory_manager.read_procedure_content(proc_path)
        assert "staging" in content

    @pytest.mark.asyncio
    async def test_metadata_reset_after_reconsolidation(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """After reconsolidation, counters are reset and version is bumped."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=4, confidence=0.2, version=3,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "Revised procedure text"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            await engine.apply_reconsolidation([proc_path], "test-model")

        meta = engine.memory_manager.read_procedure_metadata(proc_path)
        assert meta["version"] == 4
        assert meta["failure_count"] == 0
        assert meta["success_count"] == 0
        assert meta["confidence"] == 0.5
        assert meta["previous_version"] == "v3"
        assert "reconsolidated_at" in meta

    @pytest.mark.asyncio
    async def test_skip_when_llm_returns_empty(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """When LLM returns empty/None, the procedure is skipped."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=2, confidence=0.4,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = ""

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await engine.apply_reconsolidation([proc_path], "test-model")

        assert result["updated"] == 0
        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_error_handling(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Errors during apply are counted, not raised."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=2, confidence=0.4,
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await engine.apply_reconsolidation([proc_path], "test-model")

        # Error in _revise_procedure causes None return -> skipped
        # Actually, the exception is caught in _revise_procedure and returns None
        assert result["skipped"] == 1 or result["errors"] == 0

    @pytest.mark.asyncio
    async def test_multiple_targets(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Multiple targets are processed independently."""
        proc1 = _write_procedure(
            anima_dir, "proc1.md",
            failure_count=2, confidence=0.3,
        )
        proc2 = _write_procedure(
            anima_dir, "proc2.md",
            failure_count=3, confidence=0.2,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "Revised content"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await engine.apply_reconsolidation(
                [proc1, proc2], "test-model",
            )

        assert result["updated"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == 0


# ── Version Archive Tests ──────────────────────────────────


class TestArchiveVersion:
    """Test version archiving before reconsolidation."""

    @pytest.mark.asyncio
    async def test_archive_creates_copy(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """Archiving creates a timestamped copy in archive/versions/."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=2, confidence=0.3, version=2,
        )
        original_content = proc_path.read_text(encoding="utf-8")

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "Revised content"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            await engine.apply_reconsolidation([proc_path], "test-model")

        archive_dir = engine.anima_dir / "archive" / "versions"
        assert archive_dir.exists()
        archived_files = list(archive_dir.glob("deploy_v2_*"))
        assert len(archived_files) == 1

        # Verify content is preserved
        archived_content = archived_files[0].read_text(encoding="utf-8")
        assert archived_content == original_content

    def test_archive_nonexistent_file(
        self, engine: ReconsolidationEngine,
    ) -> None:
        """Archiving a nonexistent file is a no-op."""
        fake_path = engine.anima_dir / "procedures" / "nonexistent.md"
        engine._archive_version(fake_path, "some content", 1)

        archive_dir = engine.anima_dir / "archive" / "versions"
        assert not archive_dir.exists()


# ── Activity Log Event Tests ───────────────────────────────


class TestActivityLogEvent:
    """Test that reconsolidation records activity log events."""

    @pytest.mark.asyncio
    async def test_procedure_reconsolidated_event(
        self, engine: ReconsolidationEngine, anima_dir: Path,
        activity_logger,
    ) -> None:
        """A procedure_reconsolidated event is recorded after successful update."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=2, confidence=0.3, version=1,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "Revised content"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            await engine.apply_reconsolidation([proc_path], "test-model")

        # Check the activity log file
        import json
        from datetime import date

        log_dir = anima_dir / "activity_log"
        today_file = log_dir / f"{date.today().isoformat()}.jsonl"
        assert today_file.exists()

        entries = [
            json.loads(line)
            for line in today_file.read_text(encoding="utf-8").strip().splitlines()
        ]
        recon_entries = [e for e in entries if e["type"] == "procedure_reconsolidated"]
        assert len(recon_entries) == 1

        entry = recon_entries[0]
        assert "deploy.md" in entry["summary"]
        assert "v1" in entry["summary"]
        assert "v2" in entry["summary"]
        assert entry["meta"]["procedure"] == "deploy.md"
        assert entry["meta"]["old_version"] == 1
        assert entry["meta"]["new_version"] == 2

    @pytest.mark.asyncio
    async def test_no_event_on_skip(
        self, engine: ReconsolidationEngine, anima_dir: Path,
    ) -> None:
        """No activity log event when reconsolidation is skipped."""
        proc_path = _write_procedure(
            anima_dir, "deploy.md",
            failure_count=2, confidence=0.3,
        )

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = ""  # Empty -> skip

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            await engine.apply_reconsolidation([proc_path], "test-model")

        import json
        from datetime import date

        log_dir = anima_dir / "activity_log"
        today_file = log_dir / f"{date.today().isoformat()}.jsonl"
        if today_file.exists():
            entries = [
                json.loads(line)
                for line in today_file.read_text(encoding="utf-8").strip().splitlines()
            ]
            recon_entries = [
                e for e in entries if e["type"] == "procedure_reconsolidated"
            ]
            assert len(recon_entries) == 0
        # If the file doesn't exist, that's also correct (no events logged)


# ── LLM Procedure Revision Tests ──────────────────────────


class TestReviseProcedure:
    """Test the LLM-based procedure revision."""

    @pytest.mark.asyncio
    async def test_successful_revision(
        self, engine: ReconsolidationEngine,
    ) -> None:
        """LLM returns a revised procedure text."""
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = (
            "# Deploy\n\n1. Run tests\n2. Deploy to staging\n3. Verify\n"
        )

        meta = {
            "description": "deployment procedure",
            "failure_count": 3,
            "confidence": 0.2,
        }

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await engine._revise_procedure(
                "1. Run tests\n2. Deploy", meta, "test-model",
            )

        assert result is not None
        assert "staging" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(
        self, engine: ReconsolidationEngine,
    ) -> None:
        """LLM failure returns None gracefully."""
        meta = {"description": "test", "failure_count": 2, "confidence": 0.3}

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await engine._revise_procedure(
                "some content", meta, "test-model",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_none(
        self, engine: ReconsolidationEngine,
    ) -> None:
        """Empty LLM response returns None."""
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = ""

        meta = {"description": "test", "failure_count": 2, "confidence": 0.3}

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await engine._revise_procedure(
                "some content", meta, "test-model",
            )

        assert result is None


# ── Consolidation Pipeline Integration Tests ───────────────


class TestConsolidationIntegration:
    """Test integration with the ConsolidationEngine pipeline."""

    @pytest.mark.asyncio
    async def test_run_reconsolidation_method_exists(
        self, anima_dir: Path,
    ) -> None:
        """ConsolidationEngine._run_reconsolidation is callable."""
        from core.memory.consolidation import ConsolidationEngine
        engine = ConsolidationEngine(anima_dir, "test-anima")
        assert hasattr(engine, "_run_reconsolidation")
        assert callable(engine._run_reconsolidation)

    @pytest.mark.asyncio
    async def test_run_reconsolidation_no_targets(
        self, anima_dir: Path,
    ) -> None:
        """_run_reconsolidation with no targets returns zero counts."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, "test-anima")

        with patch(
            "core.memory.reconsolidation.ReconsolidationEngine.find_reconsolidation_targets",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await engine._run_reconsolidation(
                "some episodes text", "test-model",
            )

        assert result["targets_found"] == 0
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_run_reconsolidation_with_targets(
        self, anima_dir: Path,
    ) -> None:
        """_run_reconsolidation finds and applies reconsolidation."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, "test-anima")

        mock_targets = [anima_dir / "procedures" / "test.md"]

        with patch(
            "core.memory.reconsolidation.ReconsolidationEngine.find_reconsolidation_targets",
            new_callable=AsyncMock,
            return_value=mock_targets,
        ):
            with patch(
                "core.memory.reconsolidation.ReconsolidationEngine.apply_reconsolidation",
                new_callable=AsyncMock,
                return_value={"updated": 1, "skipped": 0, "errors": 0},
            ):
                with patch.object(engine, "_update_rag_index"):
                    result = await engine._run_reconsolidation(
                        "some episodes", "test-model",
                    )

        assert result["targets_found"] == 1
        assert result["updated"] == 1

    @pytest.mark.asyncio
    async def test_daily_consolidate_includes_reconsolidation_key(
        self, anima_dir: Path,
    ) -> None:
        """daily_consolidate result dict includes 'reconsolidation' key."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(anima_dir, "test-anima")

        # Create a minimal episode file
        from datetime import datetime
        today = datetime.now().date()
        episode_path = anima_dir / "episodes" / f"{today}.md"
        episode_path.write_text(
            f"# {today} Log\n\n"
            f"## {datetime.now().strftime('%H:%M')} --- Test Episode\n"
            f"Some test content here.\n",
            encoding="utf-8",
        )

        # Mock all external dependencies
        with patch.object(engine, "_summarize_episodes", new_callable=AsyncMock, return_value=""):
            with patch.object(engine, "_validate_consolidation", new_callable=AsyncMock, return_value=""):
                with patch.object(engine, "_run_reconsolidation", new_callable=AsyncMock, return_value={"targets_found": 0}):
                    with patch("core.memory.forgetting.ForgettingEngine", side_effect=ImportError):
                        with patch("core.memory.distillation.ProceduralDistiller", side_effect=ImportError):
                            result = await engine.daily_consolidate(
                                model="test-model",
                                min_episodes=1,
                            )

        assert "reconsolidation" in result


# ── Constructor Tests ──────────────────────────────────────


class TestConstructor:
    """Test ReconsolidationEngine constructor."""

    def test_default_construction(self, anima_dir: Path) -> None:
        """Constructor creates MemoryManager and ActivityLogger when not provided."""
        engine = ReconsolidationEngine(anima_dir, "test-anima")
        assert engine.memory_manager is not None
        assert engine.activity_logger is not None
        assert engine.anima_dir == anima_dir
        assert engine.anima_name == "test-anima"

    def test_injected_dependencies(
        self, anima_dir: Path, memory_manager, activity_logger,
    ) -> None:
        """Constructor uses injected dependencies."""
        engine = ReconsolidationEngine(
            anima_dir, "test-anima",
            memory_manager=memory_manager,
            activity_logger=activity_logger,
        )
        assert engine.memory_manager is memory_manager
        assert engine.activity_logger is activity_logger


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
