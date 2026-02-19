"""Tests for statistical analysis module."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from tests.evaluation.framework.analysis import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StatisticalAnalyzer(alpha=0.05)

    # ── Hypothesis H1 Tests ─────────────────────────────────────────────

    def test_h1_significant_effect(self, analyzer):
        """Test H1 with significant priming effect."""
        # Hybrid: mean=5.0, std=1.0
        latencies_hybrid = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 4.8]
        # Hybrid+Priming: mean=3.5 (30% reduction), std=0.8
        latencies_priming = [3.5, 3.6, 3.4, 3.7, 3.3, 3.8, 3.2, 3.5, 3.6, 3.4]

        result = analyzer.hypothesis_h1_priming_effect(
            latencies_hybrid, latencies_priming
        )

        # Should be significant
        assert result["significant"] is True
        assert result["p_value"] < 0.05

        # Effect size should be large (d > 0.8)
        assert result["effect_size"] > 0.8
        assert result["interpretation"] == "large"

        # Reduction should be around 30%
        assert 25 < result["mean_reduction_pct"] < 35

        # Confidence interval should not include 0
        ci_lower, ci_upper = result["confidence_interval"]
        assert ci_lower > 0  # Both bounds positive (priming reduces latency)

    def test_h1_no_effect(self, analyzer):
        """Test H1 with no priming effect."""
        # Same distribution for both conditions
        np.random.seed(42)
        latencies_hybrid = np.random.normal(5.0, 1.0, 30).tolist()
        latencies_priming = np.random.normal(5.0, 1.0, 30).tolist()

        result = analyzer.hypothesis_h1_priming_effect(
            latencies_hybrid, latencies_priming
        )

        # Should not be significant
        assert result["significant"] is False
        assert result["p_value"] > 0.05

        # Effect size should be small
        assert abs(result["effect_size"]) < 0.2
        assert result["interpretation"] == "negligible"

    def test_h1_paired_samples_required(self, analyzer):
        """Test that H1 requires equal sample sizes."""
        latencies_hybrid = [5.0, 5.2, 4.8]
        latencies_priming = [3.5, 3.6]  # Different size

        with pytest.raises(ValueError, match="equal sample sizes"):
            analyzer.hypothesis_h1_priming_effect(latencies_hybrid, latencies_priming)

    # ── Hypothesis H2 Tests ─────────────────────────────────────────────

    def test_h2_hybrid_superiority(self, analyzer):
        """Test H2 with hybrid search showing superiority."""
        # BM25: mean=0.60
        precision_bm25 = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.60] * 5

        # Vector: mean=0.65
        precision_vector = [0.65, 0.67, 0.63, 0.66, 0.64, 0.68, 0.62, 0.65] * 5

        # Hybrid: mean=0.75 (superior)
        precision_hybrid = [0.75, 0.77, 0.73, 0.76, 0.74, 0.78, 0.72, 0.75] * 5

        result = analyzer.hypothesis_h2_hybrid_search(
            precision_bm25, precision_vector, precision_hybrid
        )

        # Should be significant
        assert result["significant"] is True
        assert result["p_value"] < 0.05

        # Effect size should be substantial
        assert result["eta_squared"] > 0.10  # Medium to large effect

        # Mean precision for hybrid should be highest
        assert result["mean_precision"]["hybrid"] > result["mean_precision"]["bm25"]
        assert result["mean_precision"]["hybrid"] > result["mean_precision"]["vector"]

        # Tukey HSD should show hybrid > bm25 and hybrid > vector
        assert result["tukey_results"]["bm25_vs_hybrid"]["reject"] is True
        assert result["tukey_results"]["hybrid_vs_vector"]["reject"] is True

    def test_h2_no_difference(self, analyzer):
        """Test H2 with no difference between methods."""
        # All methods have same precision
        np.random.seed(42)
        precision_bm25 = np.random.normal(0.70, 0.05, 30).tolist()
        precision_vector = np.random.normal(0.70, 0.05, 30).tolist()
        precision_hybrid = np.random.normal(0.70, 0.05, 30).tolist()

        result = analyzer.hypothesis_h2_hybrid_search(
            precision_bm25, precision_vector, precision_hybrid
        )

        # Should not be significant
        assert result["significant"] is False
        assert result["p_value"] > 0.05

        # Effect size should be small
        assert result["eta_squared"] < 0.06

    # ── Hypothesis H3 Tests ─────────────────────────────────────────────

    def test_h3_consolidation_effect(self, analyzer):
        """Test H3 with consolidation improving retention."""
        # Create synthetic retention data
        np.random.seed(42)

        n_samples = 200

        # Auto-consolidation improves retention
        data = []
        for _ in range(n_samples):
            auto_consolidation = np.random.choice([0, 1])
            days_elapsed = np.random.choice([7, 30])
            importance = np.random.randint(1, 6)

            # Probability of recall increases with auto-consolidation
            base_prob = 0.5
            if auto_consolidation == 1:
                base_prob += 0.3  # 30% boost
            if days_elapsed == 7:
                base_prob += 0.1  # Better retention at 7 days
            base_prob += importance * 0.05  # Higher importance helps

            recalled = 1 if np.random.random() < base_prob else 0

            data.append(
                {
                    "recalled": recalled,
                    "auto_consolidation": auto_consolidation,
                    "days_elapsed": days_elapsed,
                    "importance": importance,
                }
            )

        df = pd.DataFrame(data)

        result = analyzer.hypothesis_h3_consolidation(df)

        # Auto-consolidation should be a significant predictor
        assert "auto_consolidation" in result["significant_predictors"]

        # Odds ratio should be > 1 (improves retention)
        assert result["odds_ratios"]["auto_consolidation"] > 1.0

        # P-value for auto-consolidation should be < 0.05
        assert result["p_values"]["auto_consolidation"] < 0.05

        # Model should have reasonable accuracy
        assert result["accuracy"] > 0.50

    def test_h3_with_memory_types(self, analyzer):
        """Test H3 with different memory types."""
        np.random.seed(42)

        data = []
        for _ in range(150):
            auto_consolidation = np.random.choice([0, 1])
            days_elapsed = np.random.choice([7, 30])
            importance = np.random.randint(1, 6)
            memory_type = np.random.choice(["episodic", "semantic", "procedural"])

            # Different retention rates by type
            base_prob = 0.5
            if auto_consolidation == 1:
                base_prob += 0.2
            if memory_type == "semantic":
                base_prob += 0.15  # Semantic memories last longer
            elif memory_type == "procedural":
                base_prob += 0.10

            recalled = 1 if np.random.random() < base_prob else 0

            data.append(
                {
                    "recalled": recalled,
                    "auto_consolidation": auto_consolidation,
                    "days_elapsed": days_elapsed,
                    "importance": importance,
                    "memory_type": memory_type,
                }
            )

        df = pd.DataFrame(data)
        result = analyzer.hypothesis_h3_consolidation(df)

        # Should have memory type dummies in results
        assert any("type_" in k for k in result["coefficients"].keys())

    # ── Power Analysis Tests ────────────────────────────────────────────

    def test_power_analysis_medium_effect(self, analyzer):
        """Test power analysis for medium effect size."""
        result = analyzer.calculate_power_analysis(
            effect_size=0.5, alpha=0.05, power=0.80
        )

        # Medium effect should require ~64 per group
        assert 60 <= result["required_n_per_group"] <= 70
        assert result["interpretation"] == "medium"

    def test_power_analysis_large_effect(self, analyzer):
        """Test power analysis for large effect size."""
        result = analyzer.calculate_power_analysis(
            effect_size=0.8, alpha=0.05, power=0.80
        )

        # Large effect requires fewer samples
        assert result["required_n_per_group"] < 35
        assert result["interpretation"] == "large"

    # ── Inter-Rater Reliability Tests ───────────────────────────────────

    def test_cohens_kappa_perfect_agreement(self, analyzer):
        """Test Cohen's κ with perfect agreement."""
        annotator1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        annotator2 = [1, 2, 3, 1, 2, 3, 1, 2, 3]

        kappa = analyzer.calculate_cohens_kappa(annotator1, annotator2)

        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_cohens_kappa_no_agreement(self, analyzer):
        """Test Cohen's κ with systematic disagreement."""
        annotator1 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        annotator2 = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        kappa = analyzer.calculate_cohens_kappa(annotator1, annotator2)

        # Should be negative or near 0 (worse than chance)
        assert kappa < 0.1

    def test_cohens_kappa_moderate_agreement(self, analyzer):
        """Test Cohen's κ with moderate agreement."""
        # 70% agreement
        annotator1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        annotator2 = [1, 2, 3, 1, 2, 1, 1, 3, 3, 2]

        kappa = analyzer.calculate_cohens_kappa(annotator1, annotator2)

        # Should be in moderate range (0.4-0.6)
        assert 0.3 < kappa < 0.7

    def test_krippendorff_alpha_interval(self, analyzer):
        """Test Krippendorff's α for interval data."""
        # 3 annotators, 5 items, 1-5 scale
        ratings = np.array(
            [
                [1, 2, 3, 4, 5],  # Annotator 1
                [1, 2, 3, 4, 5],  # Annotator 2
                [1, 2, 3, 4, 5],  # Annotator 3
            ]
        )

        alpha = analyzer.calculate_krippendorff_alpha(
            ratings, level_of_measurement="interval"
        )

        # Perfect agreement
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_krippendorff_alpha_with_missing(self, analyzer):
        """Test Krippendorff's α with missing values."""
        # 3 annotators, 5 items, some missing
        ratings = np.array(
            [
                [1, 2, np.nan, 4, 5],
                [1, 2, 3, 4, np.nan],
                [1, np.nan, 3, 4, 5],
            ]
        )

        alpha = analyzer.calculate_krippendorff_alpha(
            ratings, level_of_measurement="interval"
        )

        # Should still compute despite missing values
        assert 0 <= alpha <= 1

    # ── Descriptive Statistics Tests ────────────────────────────────────

    def test_descriptive_stats(self, analyzer):
        """Test comprehensive descriptive statistics."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        stats = analyzer.calculate_descriptive_stats(data)

        assert stats["mean"] == 5.5
        assert stats["median"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["p25"] == 3.25
        assert stats["p75"] == 7.75
        assert stats["n"] == 10

    def test_descriptive_stats_percentiles(self, analyzer):
        """Test that percentiles are calculated correctly."""
        # Create data where percentiles are predictable
        data = list(range(1, 101))  # 1 to 100

        stats = analyzer.calculate_descriptive_stats(data)

        # Check key percentiles
        assert stats["p50"] == pytest.approx(50.5, abs=1)
        assert stats["p95"] == pytest.approx(95.5, abs=1)
        assert stats["p99"] == pytest.approx(99.5, abs=1)


# ── Integration Tests ─────────────────────────────────────────────────────


class TestStatisticalAnalysisIntegration:
    """Integration tests using realistic experimental data."""

    def test_full_h1_analysis(self):
        """Test complete H1 analysis workflow."""
        analyzer = StatisticalAnalyzer()

        # Simulate realistic latency data
        np.random.seed(42)
        # Condition C: Hybrid search
        latencies_c = np.random.gamma(shape=5, scale=1.0, size=30).tolist()
        # Condition D: Hybrid + Priming (35% faster)
        latencies_d = (np.random.gamma(shape=5, scale=0.65, size=30)).tolist()

        # Run H1 analysis
        h1_result = analyzer.hypothesis_h1_priming_effect(latencies_c, latencies_d)

        # Verify result structure
        assert "t_statistic" in h1_result
        assert "p_value" in h1_result
        assert "effect_size" in h1_result
        assert "interpretation" in h1_result
        assert "significant" in h1_result
        assert "mean_reduction_pct" in h1_result
        assert "confidence_interval" in h1_result

        # Should detect significant reduction
        assert h1_result["significant"] is True
        assert h1_result["mean_reduction_pct"] > 20

    def test_full_h2_analysis(self):
        """Test complete H2 analysis workflow."""
        analyzer = StatisticalAnalyzer()

        np.random.seed(42)
        # Three conditions with distinct precision levels
        precision_bm25 = np.random.beta(a=6, b=4, size=30).tolist()  # ~0.60
        precision_vector = np.random.beta(a=7, b=3, size=30).tolist()  # ~0.70
        precision_hybrid = np.random.beta(a=8, b=2, size=30).tolist()  # ~0.80

        # Run H2 analysis
        h2_result = analyzer.hypothesis_h2_hybrid_search(
            precision_bm25, precision_vector, precision_hybrid
        )

        # Verify result structure
        assert "f_statistic" in h2_result
        assert "p_value" in h2_result
        assert "eta_squared" in h2_result
        assert "tukey_results" in h2_result
        assert "significant" in h2_result

        # Should detect significant differences
        assert h2_result["significant"] is True

    def test_full_h3_analysis(self):
        """Test complete H3 analysis workflow."""
        analyzer = StatisticalAnalyzer()

        # Create realistic retention dataset
        np.random.seed(42)
        data = []

        for _ in range(200):
            auto_consolidation = np.random.choice([0, 1], p=[0.5, 0.5])
            days_elapsed = np.random.choice([7, 30])
            importance = np.random.randint(1, 6)

            # Realistic retention probability
            logit = -1.0  # Base (50% retention)
            logit += 1.2 * auto_consolidation  # Strong positive effect
            logit -= 0.05 * days_elapsed  # Decay over time
            logit += 0.3 * importance  # Importance helps

            prob = 1 / (1 + np.exp(-logit))
            recalled = 1 if np.random.random() < prob else 0

            data.append(
                {
                    "recalled": recalled,
                    "auto_consolidation": auto_consolidation,
                    "days_elapsed": days_elapsed,
                    "importance": importance,
                }
            )

        df = pd.DataFrame(data)

        # Run H3 analysis
        h3_result = analyzer.hypothesis_h3_consolidation(df)

        # Verify result structure
        assert "coefficients" in h3_result
        assert "odds_ratios" in h3_result
        assert "p_values" in h3_result
        assert "aic" in h3_result
        assert "bic" in h3_result
        assert "accuracy" in h3_result
        assert "significant_predictors" in h3_result

        # Auto-consolidation should be significant
        assert "auto_consolidation" in h3_result["significant_predictors"]
        assert h3_result["odds_ratios"]["auto_consolidation"] > 1.0
