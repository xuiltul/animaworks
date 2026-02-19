from __future__ import annotations

"""E2E tests for memory evaluation ablation pipeline.

Runs the full pipeline: dataset generation -> ablation experiments -> report
in mock mode with a temporary directory for isolation.
"""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────


def _chromadb_available() -> bool:
    """Check if ChromaDB is importable."""
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


# ── Dataset Generation ───────────────────────────────────────────


class TestDatasetGeneration:
    """E2E tests for dataset generation step."""

    def test_generate_all_creates_expected_structure(self, tmp_path):
        """Full dataset generation should create all expected directories."""
        from experiments.memory_eval.dataset import AblationDatasetGenerator

        gen = AblationDatasetGenerator(output_dir=tmp_path, seed=42)
        counts = gen.generate_all()

        # Verify counts
        assert counts["knowledge"] == 30
        assert counts["episodes"] == 15
        assert counts["procedures"] == 5
        assert counts["skills"] == 5
        assert counts["noise"] == 100
        assert counts["flawed_procedures"] == 3
        assert counts["fixed_procedures"] == 3

        # Verify directory structure
        assert (tmp_path / "knowledge").is_dir()
        assert (tmp_path / "episodes").is_dir()
        assert (tmp_path / "procedures").is_dir()
        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / "noise").is_dir()
        assert (tmp_path / "flawed_procedures").is_dir()
        assert (tmp_path / "fixed_procedures").is_dir()
        assert (tmp_path / "queries.json").is_file()

        # Verify queries structure
        with open(tmp_path / "queries.json") as f:
            data = json.load(f)
        assert len(data["queries"]) == 20

        # Verify all queries have ground truth
        for q in data["queries"]:
            assert "relevant_files" in q
            assert len(q["relevant_files"]) > 0


# ── Reconsolidation Ablation E2E ─────────────────────────────────


class TestReconsolidationAblationE2E:
    """E2E test for the reconsolidation ablation (no ChromaDB needed)."""

    @pytest.mark.asyncio
    async def test_full_reconsolidation_pipeline(self, tmp_path):
        """Run reconsolidation ablation end-to-end in mock mode."""
        from experiments.memory_eval.dataset import AblationDatasetGenerator
        from experiments.memory_eval.ablation.reconsolidation import (
            ReconsolidationAblation,
            ReconsolidationAblationResult,
        )

        # Step 1: Generate dataset
        dataset_dir = tmp_path / "dataset"
        gen = AblationDatasetGenerator(output_dir=dataset_dir, seed=42)
        gen.generate_all()

        # Step 2: Run reconsolidation ablation (mock mode, no LLM needed)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        ablation = ReconsolidationAblation(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            use_mock=True,
        )
        result = await ablation.run(work_dir)

        # Step 3: Verify results
        assert isinstance(result, ReconsolidationAblationResult)
        assert result.n_procedures == 3  # 3 flawed procedures
        assert result.use_mock is True

        # ON condition should improve
        assert result.avg_improvement_on >= result.avg_improvement_off

        # Result file should exist
        result_file = output_dir / "reconsolidation_ablation_result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["n_procedures"] == 3
        assert len(data["on_results"]) == 3
        assert len(data["off_results"]) == 3

        # Verify each procedure result has required fields
        for pr in data["on_results"]:
            assert "name" in pr
            assert "round1_success_rate" in pr
            assert "round2_success_rate" in pr
            assert "was_reconsolidated" in pr
            assert 0.0 <= pr["round1_success_rate"] <= 1.0
            assert 0.0 <= pr["round2_success_rate"] <= 1.0


# ── Report Generation E2E ────────────────────────────────────────


class TestReportGenerationE2E:
    """E2E test for report generation from ablation results."""

    def test_report_from_sample_results(self, tmp_path):
        """Generate report from sample results and verify structure."""
        from experiments.memory_eval.analysis.report import generate_report

        # Create sample results
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        priming = {
            "on": {"precision_at_3": 0.70, "precision_at_5": 0.60,
                    "recall_at_5": 0.75, "avg_priming_tokens": 300},
            "off": {"precision_at_3": 0.50, "precision_at_5": 0.40,
                    "recall_at_5": 0.55},
            "n_queries": 20,
        }
        forgetting = {
            "on": {"precision_at_3": 0.65, "precision_at_5": 0.55,
                    "recall_at_5": 0.70, "memory_count_before": 130,
                    "memory_count_after": 85},
            "off": {"precision_at_3": 0.45, "precision_at_5": 0.35,
                    "recall_at_5": 0.50, "memory_count_before": 130,
                    "memory_count_after": 130},
            "n_queries": 20,
        }
        reconsolidation = {
            "on": {"rounds": [{"success_rate": 0.2}, {"success_rate": 0.8}],
                    "overall_success_rate": 0.8},
            "off": {"rounds": [{"success_rate": 0.2}, {"success_rate": 0.2}],
                    "overall_success_rate": 0.2},
            "n_procedures": 5,
        }

        (results_dir / "priming_results.json").write_text(json.dumps(priming))
        (results_dir / "forgetting_results.json").write_text(json.dumps(forgetting))
        (results_dir / "reconsolidation_results.json").write_text(
            json.dumps(reconsolidation),
        )

        # Generate report
        report_path = tmp_path / "report.md"
        generate_report(results_dir, report_path)

        # Verify
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 200

        # Should contain all three ablation sections
        content_lower = content.lower()
        assert "priming" in content_lower
        assert "forgetting" in content_lower
        assert "reconsolidation" in content_lower

        # Should contain tables
        assert "|" in content


# ── Full Pipeline E2E ────────────────────────────────────────────


class TestFullPipelineE2E:
    """E2E test for the complete pipeline (dataset + all ablations + report)."""

    @pytest.mark.skipif(
        not _chromadb_available(),
        reason="ChromaDB not available",
    )
    @pytest.mark.asyncio
    async def test_full_pipeline_mock_mode(self, tmp_path):
        """Run the entire pipeline in mock mode."""
        from experiments.memory_eval.run_all import run_all

        output_dir = tmp_path / "pipeline_output"
        output_dir.mkdir()

        await run_all(
            output_dir=output_dir,
            use_mock=True,
        )

        # Verify meta.json
        meta_path = output_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["mode"] == "mock"

        # Verify dataset was generated
        dataset_dir = output_dir / "dataset"
        assert dataset_dir.exists()
        assert (dataset_dir / "queries.json").exists()

        # Verify at least some result files exist
        # (individual ablations may fail due to ChromaDB issues,
        # but the pipeline should still produce output files)
        result_files = list(output_dir.glob("*_results.json"))
        assert len(result_files) > 0

        # Verify report was generated
        report_path = output_dir / "evaluation_results.md"
        assert report_path.exists()

    @pytest.mark.asyncio
    async def test_single_ablation_reconsolidation(self, tmp_path):
        """Run only reconsolidation ablation (no ChromaDB needed)."""
        from experiments.memory_eval.dataset import AblationDatasetGenerator

        # Pre-generate dataset
        dataset_dir = tmp_path / "dataset"
        gen = AblationDatasetGenerator(output_dir=dataset_dir, seed=42)
        gen.generate_all()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from experiments.memory_eval.run_all import run_reconsolidation_ablation

        result = await run_reconsolidation_ablation(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            use_mock=True,
        )

        assert "on" in result
        assert "off" in result
        assert "rounds" in result["on"]
        assert len(result["on"]["rounds"]) == 2

        # Verify result file
        result_file = output_dir / "reconsolidation_results.json"
        assert result_file.exists()

    @pytest.mark.asyncio
    async def test_result_normalization(self, tmp_path):
        """Verify that result normalization produces expected structure."""
        from experiments.memory_eval.run_all import (
            _normalize_priming_result,
            _normalize_forgetting_result,
            _normalize_reconsolidation_result,
        )
        from experiments.memory_eval.ablation.priming import PrimingAblationResult
        from experiments.memory_eval.ablation.forgetting import ForgettingAblationResult
        from experiments.memory_eval.ablation.reconsolidation import (
            ReconsolidationAblationResult,
        )

        # Test priming normalization
        priming = PrimingAblationResult(
            avg_priming_precision_at_3=0.7,
            avg_priming_precision_at_5=0.6,
            avg_priming_recall_at_5=0.8,
            avg_priming_tokens=300,
            avg_baseline_precision_at_3=0.5,
            avg_baseline_precision_at_5=0.4,
            avg_baseline_recall_at_5=0.6,
        )
        norm_p = _normalize_priming_result(priming)
        assert norm_p["on"]["precision_at_3"] == 0.7
        assert norm_p["off"]["precision_at_3"] == 0.5

        # Test forgetting normalization
        forgetting = ForgettingAblationResult(
            forgetting_precision_at_3=0.65,
            forgetting_precision_at_5=0.55,
            forgetting_recall_at_5=0.7,
            baseline_precision_at_3=0.45,
            baseline_precision_at_5=0.35,
            baseline_recall_at_5=0.5,
            total_before_forgetting=130,
            total_after_forgetting=85,
        )
        norm_f = _normalize_forgetting_result(forgetting)
        assert norm_f["on"]["memory_count_before"] == 130
        assert norm_f["on"]["memory_count_after"] == 85
        assert norm_f["off"]["memory_count_after"] == 130  # No change

        # Test reconsolidation normalization
        recon = ReconsolidationAblationResult(
            avg_round1_success_rate=0.2,
            avg_round2_success_rate_on=0.85,
            avg_round2_success_rate_off=0.2,
        )
        norm_r = _normalize_reconsolidation_result(recon)
        assert norm_r["on"]["rounds"][0]["success_rate"] == 0.2
        assert norm_r["on"]["rounds"][1]["success_rate"] == 0.85
        assert norm_r["off"]["rounds"][1]["success_rate"] == 0.2
