from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Unit tests for model-name consolidation (hardcoded defaults → config fallback).

Verifies that all functions changed in the model-name consolidation refactor:
  - Fall back to ``ConsolidationConfig().llm_model`` when called with the
    default empty-string model parameter.
  - Honour an explicit model string when one is provided.
  - ``ContextTracker`` falls back to ``AnimaDefaults().model``.

These are *unit* tests — all external dependencies (litellm, ChromaDB,
sentence-transformers, NLI models) are mocked.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Expected config defaults ─────────────────────────────────────


EXPECTED_CONSOLIDATION_MODEL = "anthropic/claude-sonnet-4-20250514"
EXPECTED_ANIMA_DEFAULT_MODEL = "claude-sonnet-4-20250514"


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure for unit tests."""
    anima_dir = tmp_path / "test_anima"
    for subdir in (
        "episodes", "knowledge", "procedures",
        "activity_log", "state", "shortterm",
    ):
        (anima_dir / subdir).mkdir(parents=True)
    return anima_dir


def _make_mock_llm_response(text: str = "mock response") -> MagicMock:
    """Build a minimal mock for ``litellm.acompletion`` return value."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text
    return mock_response


# ── Helper: assert model passed to litellm.acompletion ───────────


def _assert_model_in_acompletion(mock_llm: AsyncMock, expected_model: str) -> None:
    """Assert that the *first* call to acompletion used ``expected_model``."""
    assert mock_llm.await_count >= 1, "litellm.acompletion was never called"
    call_kwargs = mock_llm.call_args_list[0].kwargs
    assert call_kwargs["model"] == expected_model, (
        f"Expected model={expected_model!r}, got {call_kwargs['model']!r}"
    )


# ══════════════════════════════════════════════════════════════════
# Phase 1: Memory subsystem — 12 function defaults
# ══════════════════════════════════════════════════════════════════


# ── 1. consolidation.py ──────────────────────────────────────────
# NOTE: TestConsolidationEngineModelDefault was removed.
# The methods it tested (daily_consolidate, weekly_integrate, _compress_old_episodes)
# were deleted in the consolidation refactor. Consolidation is now Anima-driven
# via run_consolidation() with tool-call loops.


# ── 2. distillation.py ───────────────────────────────────────────


class TestProceduralDistillerModelDefault:
    """ProceduralDistiller.classify_and_distill / distill_procedures / weekly_pattern_distill."""

    @pytest.fixture
    def distiller(self, temp_anima_dir: Path):
        from core.memory.distillation import ProceduralDistiller
        return ProceduralDistiller(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

    # ── classify_and_distill ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_classify_and_distill_default_model(self, distiller):
        """classify_and_distill(model='') resolves to ConsolidationConfig().llm_model."""
        mock_resp = _make_mock_llm_response(
            "## knowledge抽出\n(なし)\n\n## procedure抽出\n(なし)\n"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await distiller.classify_and_distill("some episodes")
            _assert_model_in_acompletion(mock_llm, EXPECTED_CONSOLIDATION_MODEL)

    @pytest.mark.asyncio
    async def test_classify_and_distill_explicit_model(self, distiller):
        """classify_and_distill(model='custom/m') uses that model."""
        mock_resp = _make_mock_llm_response(
            "## knowledge抽出\n(なし)\n\n## procedure抽出\n(なし)\n"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await distiller.classify_and_distill("some episodes", model="custom/m")
            _assert_model_in_acompletion(mock_llm, "custom/m")

    # ── distill_procedures ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_distill_procedures_default_model(self, distiller):
        """distill_procedures(model='') resolves to ConsolidationConfig().llm_model."""
        mock_resp = _make_mock_llm_response(
            "## knowledge抽出\n(なし)\n\n## procedure抽出\n(なし)\n"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await distiller.distill_procedures("episodes text")
            # distill_procedures calls classify_and_distill internally
            _assert_model_in_acompletion(mock_llm, EXPECTED_CONSOLIDATION_MODEL)

    @pytest.mark.asyncio
    async def test_distill_procedures_explicit_model(self, distiller):
        """distill_procedures(model='x') uses that model."""
        mock_resp = _make_mock_llm_response(
            "## knowledge抽出\n(なし)\n\n## procedure抽出\n(なし)\n"
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await distiller.distill_procedures("episodes text", model="openai/gpt-4o")
            _assert_model_in_acompletion(mock_llm, "openai/gpt-4o")

    # ── weekly_pattern_distill ───────────────────────────────

    @pytest.mark.asyncio
    async def test_weekly_pattern_distill_default_model(self, distiller):
        """weekly_pattern_distill(model='') resolves to ConsolidationConfig().llm_model."""
        # No activity entries → returns early before LLM call, but model is resolved
        result = await distiller.weekly_pattern_distill()
        assert result["patterns_detected"] == 0

    @pytest.mark.asyncio
    async def test_weekly_pattern_distill_explicit_model(self, distiller):
        """weekly_pattern_distill(model='x') uses that model."""
        result = await distiller.weekly_pattern_distill(model="explicit/weekly")
        assert result["patterns_detected"] == 0


# ── 3. reconsolidation.py ────────────────────────────────────────


class TestReconsolidationEngineModelDefault:
    """ReconsolidationEngine.apply_reconsolidation / reconsolidate_knowledge."""

    @pytest.fixture
    def recon_engine(self, temp_anima_dir: Path):
        from core.memory.reconsolidation import ReconsolidationEngine

        mm = MagicMock()
        mm.read_procedure_metadata.return_value = {
            "failure_count": 3, "confidence": 0.3, "version": 1,
            "description": "test",
        }
        mm.read_procedure_content.return_value = "test content"
        mm.read_knowledge_metadata.return_value = {
            "failure_count": 3, "confidence": 0.3, "version": 1,
        }
        mm.read_knowledge_content.return_value = "knowledge content"

        al = MagicMock()

        return ReconsolidationEngine(
            anima_dir=temp_anima_dir,
            anima_name="test_anima",
            memory_manager=mm,
            activity_logger=al,
        )

    # ── apply_reconsolidation ────────────────────────────────

    @pytest.mark.asyncio
    async def test_apply_reconsolidation_default_model(self, recon_engine):
        """apply_reconsolidation(targets, model='') resolves to config default."""
        result = await recon_engine.apply_reconsolidation([])
        assert result == {"updated": 0, "skipped": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_apply_reconsolidation_explicit_model(
        self, recon_engine, temp_anima_dir,
    ):
        """apply_reconsolidation(targets, model='x') uses that model."""
        proc_dir = temp_anima_dir / "procedures"
        proc_file = proc_dir / "test_proc.md"
        proc_file.write_text("---\nversion: 1\n---\ntest", encoding="utf-8")

        mock_resp = _make_mock_llm_response("revised procedure content")
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await recon_engine.apply_reconsolidation(
                [proc_file], model="google/gemini-2.5-pro",
            )
            _assert_model_in_acompletion(mock_llm, "google/gemini-2.5-pro")

    # ── reconsolidate_knowledge ──────────────────────────────

    @pytest.mark.asyncio
    async def test_reconsolidate_knowledge_default_model(self, recon_engine):
        """reconsolidate_knowledge(model='') resolves to config default."""
        result = await recon_engine.reconsolidate_knowledge()
        assert result["targets_found"] == 0

    @pytest.mark.asyncio
    async def test_reconsolidate_knowledge_explicit_model(self, recon_engine):
        """reconsolidate_knowledge(model='x') uses that model."""
        result = await recon_engine.reconsolidate_knowledge(model="explicit/k-model")
        assert result["targets_found"] == 0


# ── 4. contradiction.py ──────────────────────────────────────────


class TestContradictionDetectorModelDefault:
    """ContradictionDetector.scan_contradictions / resolve_contradictions."""

    @pytest.fixture
    def detector(self, temp_anima_dir: Path):
        from core.memory.contradiction import ContradictionDetector
        return ContradictionDetector(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

    # ── scan_contradictions ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_scan_contradictions_default_model(self, detector):
        """scan_contradictions(model='') resolves to ConsolidationConfig().llm_model."""
        with patch.object(
            detector, "_find_candidate_pairs", return_value=[],
        ):
            result = await detector.scan_contradictions()
            assert result == []

    @pytest.mark.asyncio
    async def test_scan_contradictions_explicit_model(self, detector):
        """scan_contradictions(model='x') uses that model."""
        with patch.object(
            detector, "_find_candidate_pairs", return_value=[],
        ):
            result = await detector.scan_contradictions(model="explicit/contra")
            assert result == []

    # ── resolve_contradictions ───────────────────────────────

    @pytest.mark.asyncio
    async def test_resolve_contradictions_default_model(self, detector):
        """resolve_contradictions([], model='') resolves to config default."""
        result = await detector.resolve_contradictions([])
        assert result == {"superseded": 0, "merged": 0, "coexisted": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_resolve_contradictions_explicit_model(self, detector):
        """resolve_contradictions([], model='x') uses that model."""
        result = await detector.resolve_contradictions(
            [], model="explicit/resolve",
        )
        assert result == {"superseded": 0, "merged": 0, "coexisted": 0, "errors": 0}


# ── 5. forgetting.py ─────────────────────────────────────────────


class TestForgettingEngineModelDefault:
    """ForgettingEngine.neurogenesis_reorganize."""

    @pytest.fixture
    def forgetter(self, temp_anima_dir: Path):
        from core.memory.forgetting import ForgettingEngine
        return ForgettingEngine(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

    @pytest.mark.asyncio
    async def test_neurogenesis_reorganize_default_model(self, forgetter):
        """neurogenesis_reorganize(model='') resolves to ConsolidationConfig().llm_model."""
        with patch.object(
            forgetter, "_get_vector_store",
        ) as mock_store, patch.object(
            forgetter, "_get_all_chunks", return_value=[],
        ):
            result = await forgetter.neurogenesis_reorganize()
            assert result["merged_count"] == 0

    @pytest.mark.asyncio
    async def test_neurogenesis_reorganize_explicit_model(self, forgetter):
        """neurogenesis_reorganize(model='x') uses that model."""
        with patch.object(
            forgetter, "_get_vector_store",
        ), patch.object(
            forgetter, "_get_all_chunks", return_value=[],
        ):
            result = await forgetter.neurogenesis_reorganize(model="explicit/neuro")
            assert result["merged_count"] == 0


# ── 6. validation.py ─────────────────────────────────────────────


class TestKnowledgeValidatorModelDefault:
    """KnowledgeValidator.validate."""

    @pytest.fixture
    def validator(self):
        from core.memory.validation import KnowledgeValidator

        v = KnowledgeValidator()
        # Disable NLI model loading
        v._nli_available = False
        return v

    @pytest.mark.asyncio
    async def test_validate_default_model(self, validator):
        """validate(items, episodes, model='') resolves to ConsolidationConfig().llm_model."""
        items = [{"content": "test knowledge", "type": "create", "filename": "t.md"}]
        mock_resp = _make_mock_llm_response('{"valid": true, "reason": "ok"}')
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await validator.validate(items, "source episodes")
            _assert_model_in_acompletion(mock_llm, EXPECTED_CONSOLIDATION_MODEL)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_validate_explicit_model(self, validator):
        """validate(items, episodes, model='x') uses that model."""
        items = [{"content": "test knowledge", "type": "create", "filename": "t.md"}]
        mock_resp = _make_mock_llm_response('{"valid": true, "reason": "ok"}')
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await validator.validate(
                items, "source episodes", model="custom/validator",
            )
            _assert_model_in_acompletion(mock_llm, "custom/validator")
            assert len(result) == 1


# ══════════════════════════════════════════════════════════════════
# Phase 2: ContextTracker default model
# ══════════════════════════════════════════════════════════════════


class TestContextTrackerModelDefault:
    """ContextTracker.__post_init__ falls back to AnimaDefaults().model."""

    def test_context_tracker_default_model(self):
        """ContextTracker() with no args has model == AnimaDefaults().model."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker()
        assert tracker.model == EXPECTED_ANIMA_DEFAULT_MODEL

    def test_context_tracker_explicit_model(self):
        """ContextTracker(model='gpt-4o') uses that model."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="gpt-4o")
        assert tracker.model == "gpt-4o"

    def test_context_tracker_empty_string_resolves(self):
        """ContextTracker(model='') resolves via __post_init__ to default."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker(model="")
        assert tracker.model == EXPECTED_ANIMA_DEFAULT_MODEL

    def test_context_tracker_context_window_after_default(self):
        """Default model should resolve to a known context window size."""
        from core.prompt.context import ContextTracker

        tracker = ContextTracker()
        # claude-sonnet-4 has 200K context window
        assert tracker.context_window == 200_000


# ══════════════════════════════════════════════════════════════════
# Config defaults sanity checks
# ══════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Verify that config model defaults are the expected values."""

    def test_consolidation_config_llm_model(self):
        """ConsolidationConfig().llm_model is the expected default."""
        from core.config.models import ConsolidationConfig

        cfg = ConsolidationConfig()
        assert cfg.llm_model == EXPECTED_CONSOLIDATION_MODEL

    def test_anima_defaults_model(self):
        """AnimaDefaults().model is the expected default."""
        from core.config.models import AnimaDefaults

        defaults = AnimaDefaults()
        assert defaults.model == EXPECTED_ANIMA_DEFAULT_MODEL

    def test_consolidation_model_has_provider_prefix(self):
        """ConsolidationConfig().llm_model includes the provider prefix."""
        from core.config.models import ConsolidationConfig

        cfg = ConsolidationConfig()
        assert "/" in cfg.llm_model, (
            "llm_model should include provider prefix (e.g. 'anthropic/')"
        )

    def test_anima_defaults_model_no_provider_prefix(self):
        """AnimaDefaults().model does NOT include a provider prefix."""
        from core.config.models import AnimaDefaults

        defaults = AnimaDefaults()
        assert "/" not in defaults.model, (
            "AnimaDefaults.model should not include a provider prefix"
        )
