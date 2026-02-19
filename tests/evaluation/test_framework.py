# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Unit tests for the memory evaluation framework.

This module tests the core components of the evaluation framework:
- Configuration validation
- Metrics calculation
- Logging functionality
- Experiment orchestration (with mocked agents)
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytest.importorskip("numpy")

from framework import (
    ConversationLength,
    Domain,
    ExperimentConfig,
    MemoryExperiment,
    MemorySize,
    MetricsCollector,
    Scenario,
    ScenarioType,
    SearchConfig,
    SearchMethod,
    Turn,
)
from framework.logger import ExperimentLogger

# ── Configuration Tests ─────────────────────────────────────────────────────


class TestExperimentConfig:
    """Test ExperimentConfig validation and factory methods."""

    def test_create_condition_a(self):
        """Test Condition A (BM25 only) configuration."""
        config = ExperimentConfig.create_condition_a(
            experiment_id="test_001",
            participants=30,
        )

        assert config.condition == "A"
        assert config.search_config.method == SearchMethod.BM25
        assert config.search_config.priming_enabled is False
        assert config.participants == 30

    def test_create_condition_b(self):
        """Test Condition B (Vector only) configuration."""
        config = ExperimentConfig.create_condition_b(
            experiment_id="test_002",
            participants=30,
        )

        assert config.condition == "B"
        assert config.search_config.method == SearchMethod.VECTOR
        assert config.search_config.priming_enabled is False

    def test_create_condition_c(self):
        """Test Condition C (Hybrid) configuration."""
        config = ExperimentConfig.create_condition_c(
            experiment_id="test_003",
            participants=30,
        )

        assert config.condition == "C"
        assert config.search_config.method == SearchMethod.HYBRID
        assert config.search_config.priming_enabled is False
        assert config.search_config.weights["vector"] == 0.5
        assert config.search_config.weights["bm25"] == 0.3
        assert config.search_config.weights["recency"] == 0.2

    def test_create_condition_d(self):
        """Test Condition D (Hybrid + Priming) configuration."""
        config = ExperimentConfig.create_condition_d(
            experiment_id="test_004",
            participants=30,
            priming_budget=2000,
        )

        assert config.condition == "D"
        assert config.search_config.method == SearchMethod.HYBRID_PRIMING
        assert config.search_config.priming_enabled is True
        assert config.search_config.priming_budget == 2000

    def test_invalid_condition_method_mismatch(self):
        """Test that condition must match search method."""
        with pytest.raises(ValueError, match="Condition A requires method bm25"):
            ExperimentConfig(
                experiment_id="invalid",
                condition="A",
                participants=30,
                memory_size=MemorySize.SMALL,
                conversation_length=ConversationLength.SHORT,
                domain=Domain.BUSINESS,
                search_config=SearchConfig(method=SearchMethod.VECTOR),  # Wrong method
            )

    def test_invalid_participants(self):
        """Test that participants must be positive."""
        with pytest.raises(ValueError, match="participants must be positive"):
            ExperimentConfig.create_condition_a(
                experiment_id="invalid",
                participants=0,
            )

    def test_invalid_temperature(self):
        """Test that temperature must be in [0, 1]."""
        config = ExperimentConfig.create_condition_a(experiment_id="test")
        with pytest.raises(ValueError, match="temperature must be in"):
            config.temperature = 1.5
            config.__post_init__()


class TestSearchConfig:
    """Test SearchConfig validation."""

    def test_priming_auto_enabled_for_condition_d(self):
        """Test that priming is auto-enabled for HYBRID_PRIMING method."""
        config = SearchConfig(method=SearchMethod.HYBRID_PRIMING)
        assert config.priming_enabled is True

    def test_weights_validation(self):
        """Test that hybrid weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            SearchConfig(
                method=SearchMethod.HYBRID,
                weights={"vector": 0.5, "bm25": 0.3, "recency": 0.1},  # Sum = 0.9
            )

    def test_priming_budget_validation(self):
        """Test that priming budget must be positive when enabled."""
        with pytest.raises(ValueError, match="priming_budget must be positive"):
            SearchConfig(
                method=SearchMethod.HYBRID_PRIMING,
                priming_enabled=True,
                priming_budget=0,
            )


# ── Metrics Tests ───────────────────────────────────────────────────────────


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_precision_recall_f1_calculation(self):
        """Test precision, recall, F1 calculation."""
        collector = MetricsCollector()

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc3", "doc4"]

        precision, recall, f1 = collector.calculate_precision_recall(retrieved, relevant, k=3)

        # TP = 2 (doc1, doc3), FP = 1 (doc2), FN = 1 (doc4)
        assert precision == pytest.approx(2 / 3, rel=1e-3)
        assert recall == pytest.approx(2 / 3, rel=1e-3)
        assert f1 == pytest.approx(2 / 3, rel=1e-3)

    def test_precision_recall_empty_retrieved(self):
        """Test precision/recall when no documents retrieved."""
        collector = MetricsCollector()

        precision, recall, f1 = collector.calculate_precision_recall(
            retrieved=[],
            relevant=["doc1", "doc2"],
            k=3,
        )

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_token_counting(self):
        """Test token counting heuristic."""
        collector = MetricsCollector()

        text = "This is a test sentence."
        tokens = collector.count_tokens(text)

        # Rough estimate: ~6 tokens (1 token ≈ 4 chars)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        collector = MetricsCollector()

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        percentiles = collector.calculate_percentiles(values, [50, 95, 99])

        assert percentiles["p50"] == pytest.approx(5.5, rel=1e-1)
        assert percentiles["p95"] == pytest.approx(9.55, rel=1e-1)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        collector = MetricsCollector()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = collector.calculate_summary_stats(values)

        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test async latency measurement."""
        collector = MetricsCollector()

        async def dummy_func(delay: float) -> str:
            await asyncio.sleep(delay)
            return "done"

        result, latency = await collector.measure_latency_async(dummy_func, 0.05)

        assert result == "done"
        assert latency >= 50.0  # At least 50ms
        assert latency < 100.0  # Should be less than 100ms


# ── Logger Tests ────────────────────────────────────────────────────────────


class TestExperimentLogger:
    """Test ExperimentLogger functionality."""

    def test_logger_initialization(self):
        """Test logger creates directory structure."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            logger = ExperimentLogger(
                experiment_id="test_exp",
                output_dir=output_dir,
                condition="A",
                participant_id=1,
            )

            # Check directory structure
            assert (output_dir / "test_exp").exists()
            assert (output_dir / "test_exp" / "conversations").exists()
            assert (output_dir / "test_exp" / "turns").exists()
            assert (output_dir / "test_exp" / "metadata.json").exists()

            # Check metadata content
            with open(output_dir / "test_exp" / "metadata.json") as f:
                metadata = json.load(f)
                assert metadata["experiment_id"] == "test_exp"
                assert metadata["condition"] == "A"
                assert metadata["participant_id"] == 1

    def test_priming_logging(self):
        """Test priming metrics logging."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            logger = ExperimentLogger(
                experiment_id="test_exp",
                output_dir=output_dir,
                condition="D",
                participant_id=1,
            )

            logger.start_conversation("conv_001")
            logger.log_priming(turn_id="turn_001", latency_ms=50.0, tokens=100)

            # Check priming log exists and has data
            priming_log = output_dir / "test_exp" / "priming.jsonl"
            assert priming_log.exists()

            with open(priming_log) as f:
                lines = f.readlines()
                assert len(lines) == 1
                entry = json.loads(lines[0])
                assert entry["turn_id"] == "turn_001"
                assert entry["latency_ms"] == 50.0
                assert entry["tokens"] == 100

    def test_error_logging(self):
        """Test error logging."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            logger = ExperimentLogger(
                experiment_id="test_exp",
                output_dir=output_dir,
                condition="A",
                participant_id=1,
            )

            logger.log_error(
                error_type="TestError",
                error_message="This is a test error",
                context={"test_key": "test_value"},
            )

            # Check error log exists
            error_log = output_dir / "test_exp" / "errors.jsonl"
            assert error_log.exists()

            with open(error_log) as f:
                lines = f.readlines()
                assert len(lines) == 1
                entry = json.loads(lines[0])
                assert entry["error_type"] == "TestError"
                assert entry["error_message"] == "This is a test error"


# ── Framework Tests ─────────────────────────────────────────────────────────


class TestMemoryExperiment:
    """Test MemoryExperiment orchestration."""

    def test_experiment_initialization(self):
        """Test experiment initialization."""
        with TemporaryDirectory() as tmpdir:
            config = ExperimentConfig.create_condition_a(experiment_id="test")
            experiment = MemoryExperiment(config, output_dir=Path(tmpdir))

            assert experiment.config.experiment_id == "test"
            assert experiment.config.condition == "A"
            assert experiment.output_dir.exists()

    def test_validation_fails_without_scenarios(self):
        """Test validation fails when no scenarios loaded."""
        with TemporaryDirectory() as tmpdir:
            config = ExperimentConfig.create_condition_a(experiment_id="test")
            experiment = MemoryExperiment(config, output_dir=Path(tmpdir))

            assert experiment.validate_setup() is False

    def test_validation_succeeds_with_scenarios(self):
        """Test validation succeeds with scenarios."""
        with TemporaryDirectory() as tmpdir:
            config = ExperimentConfig.create_condition_a(experiment_id="test")
            scenarios = [
                Scenario(
                    scenario_id="test_001",
                    scenario_type=ScenarioType.FACTUAL_RECALL,
                    domain="business",
                    turns=[
                        Turn(
                            turn_id="turn_001",
                            message="Test message",
                            relevant_memories=["knowledge/test.md"],
                        )
                    ],
                )
            ]
            experiment = MemoryExperiment(config, output_dir=Path(tmpdir), scenarios=scenarios)

            assert experiment.validate_setup() is True

    @pytest.mark.asyncio
    async def test_run_scenario_placeholder(self):
        """Test scenario execution with placeholder agent."""
        with TemporaryDirectory() as tmpdir:
            config = ExperimentConfig.create_condition_a(experiment_id="test")
            scenario = Scenario(
                scenario_id="test_001",
                scenario_type=ScenarioType.FACTUAL_RECALL,
                domain="business",
                turns=[
                    Turn(
                        turn_id="turn_001",
                        message="Test message",
                        relevant_memories=["knowledge/test.md"],
                    )
                ],
            )
            experiment = MemoryExperiment(config, output_dir=Path(tmpdir), scenarios=[scenario])

            exp_logger = ExperimentLogger(
                experiment_id="test",
                output_dir=Path(tmpdir),
                condition="A",
                participant_id=1,
            )

            # Run scenario with None agent (uses placeholder responses)
            agent = None
            conv_metrics = await experiment.run_scenario(agent, scenario, exp_logger)

            assert conv_metrics.conversation_id == "test_001"
            assert conv_metrics.turn_count == 1
            assert conv_metrics.total_latency > 0


# ── Run Tests ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
