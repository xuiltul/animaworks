"""Tests for visualization module."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from tests.evaluation.framework.visualization import ExperimentVisualizer


class TestExperimentVisualizer:
    """Test suite for ExperimentVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return ExperimentVisualizer(style="publication")

    @pytest.fixture
    def tmp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "figures"
        output_dir.mkdir()
        return output_dir

    # ── Setup Tests ─────────────────────────────────────────────────────

    def test_publication_style(self):
        """Test publication style configuration."""
        viz = ExperimentVisualizer(style="publication")

        import matplotlib.pyplot as plt

        # Check that DPI is set for publication quality
        assert plt.rcParams["figure.dpi"] == 300
        assert plt.rcParams["savefig.dpi"] == 300

    def test_presentation_style(self):
        """Test presentation style configuration."""
        viz = ExperimentVisualizer(style="presentation")

        import matplotlib.pyplot as plt

        # Check that font sizes are larger for presentation
        assert plt.rcParams["font.size"] >= 14

    def test_colorblind_palette(self, visualizer):
        """Test that colorblind-friendly colors are used."""
        # Should have 4 condition colors
        assert len(visualizer.condition_colors) == 4
        assert "bm25" in visualizer.condition_colors
        assert "vector" in visualizer.condition_colors
        assert "hybrid" in visualizer.condition_colors
        assert "hybrid_priming" in visualizer.condition_colors

    # ── Latency Comparison Tests ────────────────────────────────────────

    def test_plot_latency_comparison(self, visualizer, tmp_output_dir):
        """Test latency comparison plot generation."""
        # Create sample data
        np.random.seed(42)
        data = []

        for condition in ["bm25", "vector", "hybrid", "hybrid_priming"]:
            # Different mean latencies for each condition
            base_latency = {"bm25": 5.0, "vector": 4.5, "hybrid": 4.0, "hybrid_priming": 2.8}[condition]

            latencies = np.random.gamma(shape=5, scale=base_latency / 5, size=50)
            for lat in latencies:
                data.append({"condition": condition, "latency": lat})

        df = pd.DataFrame(data)

        # Generate plot
        output_path = tmp_output_dir / "latency_comparison"
        visualizer.plot_latency_comparison(df, output_path, formats=["png", "pdf"])

        # Check that files were created
        assert (tmp_output_dir / "latency_comparison.png").exists()
        assert (tmp_output_dir / "latency_comparison.pdf").exists()

    def test_plot_latency_single_format(self, visualizer, tmp_output_dir):
        """Test latency plot with single output format."""
        np.random.seed(42)
        data = []
        for condition in ["bm25", "vector"]:
            for _ in range(20):
                data.append({"condition": condition, "latency": np.random.gamma(5, 1)})

        df = pd.DataFrame(data)

        output_path = tmp_output_dir / "latency_single"
        visualizer.plot_latency_comparison(df, output_path, formats=["png"])

        assert (tmp_output_dir / "latency_single.png").exists()
        assert not (tmp_output_dir / "latency_single.pdf").exists()

    # ── Precision-Recall Curve Tests ────────────────────────────────────

    def test_plot_precision_recall_curves(self, visualizer, tmp_output_dir):
        """Test P-R curve generation."""
        # Sample P-R data for different conditions
        results = {
            "bm25": {"precision": [0.9, 0.7, 0.5, 0.3], "recall": [0.3, 0.5, 0.7, 0.9]},
            "vector": {"precision": [0.95, 0.8, 0.6, 0.4], "recall": [0.3, 0.5, 0.7, 0.9]},
            "hybrid": {"precision": [0.98, 0.85, 0.7, 0.5], "recall": [0.3, 0.5, 0.7, 0.9]},
            "hybrid_priming": {
                "precision": [0.99, 0.9, 0.75, 0.55],
                "recall": [0.3, 0.5, 0.7, 0.9],
            },
        }

        output_path = tmp_output_dir / "pr_curves"
        visualizer.plot_precision_recall_curves(results, output_path, formats=["png"])

        assert (tmp_output_dir / "pr_curves.png").exists()

    # ── Scalability Tests ───────────────────────────────────────────────

    def test_plot_scalability(self, visualizer, tmp_output_dir):
        """Test scalability plot generation."""
        memory_sizes = [50, 500, 5000]

        # Simulate different scaling behaviors
        latencies = {
            "bm25": [0.1, 0.15, 0.25],  # Roughly O(n^0.3)
            "vector": [0.1, 0.12, 0.15],  # Roughly O(log n)
            "hybrid": [0.12, 0.14, 0.18],  # Between vector and BM25
            "hybrid_priming": [0.08, 0.09, 0.11],  # Best scaling
        }

        output_path = tmp_output_dir / "scalability"
        visualizer.plot_scalability(memory_sizes, latencies, output_path, formats=["png"])

        assert (tmp_output_dir / "scalability.png").exists()

    # ── Retention Rate Tests ────────────────────────────────────────────

    def test_plot_retention_rate(self, visualizer, tmp_output_dir):
        """Test retention rate plot generation."""
        # Sample retention data
        data = []

        for memory_type in ["episodic", "semantic", "procedural"]:
            for day in [7, 30]:
                # Different retention rates by type and time
                if memory_type == "semantic":
                    base_rate = 0.85 if day == 7 else 0.75
                elif memory_type == "procedural":
                    base_rate = 0.80 if day == 7 else 0.70
                else:  # episodic
                    base_rate = 0.70 if day == 7 else 0.55

                data.append({"memory_type": memory_type, "day": day, "retention_rate": base_rate})

        df = pd.DataFrame(data)

        output_path = tmp_output_dir / "retention_rate"
        visualizer.plot_retention_rate(df, output_path, formats=["png"])

        assert (tmp_output_dir / "retention_rate.png").exists()

    # ── Token Consumption Tests ─────────────────────────────────────────

    def test_plot_token_consumption(self, visualizer, tmp_output_dir):
        """Test token consumption bar chart."""
        conditions = ["bm25", "vector", "hybrid", "hybrid_priming"]
        tokens = [5000, 4500, 3500, 2000]  # Decreasing consumption

        output_path = tmp_output_dir / "token_consumption"
        visualizer.plot_token_consumption(conditions, tokens, output_path, formats=["png"])

        assert (tmp_output_dir / "token_consumption.png").exists()

    # ── Hypothesis Results Tests ────────────────────────────────────────

    def test_plot_hypothesis_results(self, visualizer, tmp_output_dir):
        """Test integrated hypothesis results plot."""
        # Mock hypothesis test results
        h1_result = {
            "mean_hybrid": 4.0,
            "mean_priming": 2.8,
            "p_value": 0.001,
            "effect_size": 1.2,
            "significant": True,
            "interpretation": "large",
        }

        h2_result = {
            "mean_precision": {"bm25": 0.60, "vector": 0.70, "hybrid": 0.80},
            "p_value": 0.003,
            "eta_squared": 0.15,
            "significant": True,
        }

        h3_result = {
            "coefficients": {"auto_consolidation": 1.5, "days_elapsed": -0.02, "importance": 0.3},
            "odds_ratios": {"auto_consolidation": 4.5, "days_elapsed": 0.98, "importance": 1.35},
            "p_values": {"auto_consolidation": 0.001, "days_elapsed": 0.15, "importance": 0.05},
            "significant_predictors": ["auto_consolidation", "importance"],
        }

        output_path = tmp_output_dir / "hypothesis_results"
        visualizer.plot_hypothesis_results(h1_result, h2_result, h3_result, output_path, formats=["png"])

        assert (tmp_output_dir / "hypothesis_results.png").exists()

    # ── Generate All Figures Tests ──────────────────────────────────────

    def test_generate_all_figures_empty_dir(self, visualizer, tmp_output_dir):
        """Test generate_all_figures with empty results directory."""
        results_dir = tmp_output_dir / "results"
        results_dir.mkdir()

        output_dir = tmp_output_dir / "output"

        # Should handle empty directory gracefully
        generated = visualizer.generate_all_figures(results_dir, output_dir)

        # Should create output directory even if no files generated
        assert output_dir.exists()
        assert len(generated) == 0

    def test_generate_all_figures_with_data(self, visualizer, tmp_output_dir):
        """Test generate_all_figures with sample data files."""
        results_dir = tmp_output_dir / "results"
        results_dir.mkdir()

        # Create sample latency data
        latency_data = pd.DataFrame(
            {
                "condition": ["bm25"] * 10 + ["vector"] * 10,
                "latency": list(np.random.gamma(5, 1, 10)) + list(np.random.gamma(4, 1, 10)),
            }
        )
        latency_data.to_csv(results_dir / "latency_data.csv", index=False)

        # Create sample retention data
        retention_data = pd.DataFrame(
            {
                "memory_type": ["episodic", "semantic"] * 2,
                "day": [7, 7, 30, 30],
                "retention_rate": [0.7, 0.85, 0.55, 0.75],
            }
        )
        retention_data.to_csv(results_dir / "retention.csv", index=False)

        output_dir = tmp_output_dir / "output"

        # Generate all figures
        generated = visualizer.generate_all_figures(results_dir, output_dir)

        # Should have generated latency and retention plots
        assert len(generated) > 0
        assert any("latency" in str(p) for p in generated)
        assert any("retention" in str(p) for p in generated)


# ── Integration Tests ─────────────────────────────────────────────────────


class TestVisualizationIntegration:
    """Integration tests for visualization with realistic data."""

    def test_end_to_end_figure_generation(self, tmp_path):
        """Test complete workflow from data to figures."""
        visualizer = ExperimentVisualizer(style="publication")

        # Create realistic experimental data
        np.random.seed(42)

        # 1. Latency data
        latency_data = []
        for condition, base_latency in [
            ("bm25", 5.0),
            ("vector", 4.5),
            ("hybrid", 4.0),
            ("hybrid_priming", 2.8),
        ]:
            latencies = np.random.gamma(shape=5, scale=base_latency / 5, size=50)
            for lat in latencies:
                latency_data.append({"condition": condition, "latency": lat})

        latency_df = pd.DataFrame(latency_data)

        # 2. Scalability data
        memory_sizes = [50, 500, 5000]
        scalability_latencies = {
            "bm25": [0.1, 0.15, 0.25],
            "vector": [0.1, 0.12, 0.15],
            "hybrid": [0.12, 0.14, 0.18],
            "hybrid_priming": [0.08, 0.09, 0.11],
        }

        # 3. Retention data
        retention_data = []
        for memory_type in ["episodic", "semantic", "procedural"]:
            for day in [7, 30]:
                rate = np.random.beta(8, 2) if day == 7 else np.random.beta(6, 4)
                retention_data.append({"memory_type": memory_type, "day": day, "retention_rate": rate})

        retention_df = pd.DataFrame(retention_data)

        # Generate all plots
        output_dir = tmp_path / "figures"

        visualizer.plot_latency_comparison(latency_df, output_dir / "latency", formats=["png", "pdf"])

        visualizer.plot_scalability(
            memory_sizes, scalability_latencies, output_dir / "scalability", formats=["png"]
        )

        visualizer.plot_retention_rate(retention_df, output_dir / "retention", formats=["png"])

        # Verify all files created
        assert (output_dir / "latency.png").exists()
        assert (output_dir / "latency.pdf").exists()
        assert (output_dir / "scalability.png").exists()
        assert (output_dir / "retention.png").exists()

    def test_svg_output_format(self, tmp_path):
        """Test SVG output for vector graphics."""
        visualizer = ExperimentVisualizer()

        data = pd.DataFrame(
            {"condition": ["bm25", "vector"] * 10, "latency": list(np.random.gamma(5, 1, 20))}
        )

        output_path = tmp_path / "latency_svg"
        visualizer.plot_latency_comparison(data, output_path, formats=["svg"])

        assert (tmp_path / "latency_svg.svg").exists()
