# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Statistical analysis tools for memory performance evaluation.

This module implements hypothesis testing and effect size calculations
for the AnimaWorks memory evaluation experiments.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower

logger = logging.getLogger(__name__)


# ── Statistical Analyzer ──────────────────────────────────────────────────


class StatisticalAnalyzer:
    """Hypothesis testing and effect size calculations for memory experiments."""

    def __init__(self, alpha: float = 0.05):
        """Initialize analyzer.

        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha

    # ── Hypothesis Testing ──────────────────────────────────────────────

    def hypothesis_h1_priming_effect(
        self,
        latencies_hybrid: list[float],
        latencies_hybrid_priming: list[float],
    ) -> dict[str, Any]:
        """Test H1: Priming effect on response latency.

        Uses paired t-test (same agents in both conditions) and Cohen's d.

        H0: No difference in latency between hybrid and hybrid+priming
        H1: Hybrid+priming has 30%+ lower latency than hybrid alone

        Args:
            latencies_hybrid: Response latencies for condition C (hybrid)
            latencies_hybrid_priming: Response latencies for condition D (hybrid+priming)

        Returns:
            Dictionary with test results:
                - t_statistic: t-test statistic
                - p_value: two-tailed p-value
                - effect_size: Cohen's d
                - interpretation: 'small', 'medium', or 'large'
                - significant: True if p < alpha
                - mean_reduction_pct: Percentage reduction in latency
                - confidence_interval: 95% CI for mean difference
        """
        # Input validation
        if len(latencies_hybrid) != len(latencies_hybrid_priming):
            raise ValueError("Paired t-test requires equal sample sizes")

        if len(latencies_hybrid) < 2:
            raise ValueError("Need at least 2 samples for t-test")

        # Convert to numpy arrays
        hybrid = np.array(latencies_hybrid)
        hybrid_priming = np.array(latencies_hybrid_priming)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(hybrid, hybrid_priming)

        # Cohen's d (paired samples)
        differences = hybrid - hybrid_priming
        pooled_std = np.std(differences, ddof=1)
        effect_size = np.mean(differences) / pooled_std

        # Effect size interpretation (Cohen, 1988)
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        # Mean reduction percentage
        mean_hybrid = np.mean(hybrid)
        mean_priming = np.mean(hybrid_priming)
        reduction_pct = ((mean_hybrid - mean_priming) / mean_hybrid) * 100

        # 95% Confidence interval for mean difference
        se_diff = stats.sem(differences)
        df = len(differences) - 1
        ci = stats.t.interval(0.95, df, loc=np.mean(differences), scale=se_diff)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "interpretation": interpretation,
            "significant": bool(p_value < self.alpha),
            "mean_reduction_pct": float(reduction_pct),
            "confidence_interval": (float(ci[0]), float(ci[1])),
            "sample_size": len(latencies_hybrid),
            "mean_hybrid": float(mean_hybrid),
            "mean_priming": float(mean_priming),
        }

    def hypothesis_h2_hybrid_search(
        self,
        precision_bm25: list[float],
        precision_vector: list[float],
        precision_hybrid: list[float],
    ) -> dict[str, Any]:
        """Test H2: Hybrid search superiority.

        Uses one-way ANOVA and Tukey HSD post-hoc tests.

        H0: No difference in precision across search methods
        H1: Hybrid search has 15-25% higher precision

        Args:
            precision_bm25: Precision@k for BM25-only search
            precision_vector: Precision@k for vector-only search
            precision_hybrid: Precision@k for hybrid search

        Returns:
            Dictionary with test results:
                - f_statistic: ANOVA F-statistic
                - p_value: ANOVA p-value
                - eta_squared: Effect size (proportion of variance explained)
                - tukey_results: Pairwise comparison results
                - significant: True if p < alpha
                - mean_precision: Mean precision for each condition
        """
        # Input validation
        if len(precision_bm25) < 2 or len(precision_vector) < 2 or len(precision_hybrid) < 2:
            raise ValueError("Need at least 2 samples per group for ANOVA")

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(precision_bm25, precision_vector, precision_hybrid)

        # Prepare data for Tukey HSD
        all_values = list(precision_bm25) + list(precision_vector) + list(precision_hybrid)
        all_labels = (
            ["bm25"] * len(precision_bm25)
            + ["vector"] * len(precision_vector)
            + ["hybrid"] * len(precision_hybrid)
        )

        data = pd.DataFrame({"precision": all_values, "condition": all_labels})

        # Tukey HSD post-hoc test
        tukey = pairwise_tukeyhsd(
            endog=data["precision"], groups=data["condition"], alpha=self.alpha
        )

        # Parse Tukey results
        tukey_df = pd.DataFrame(
            data=tukey.summary().data[1:],  # Skip header row
            columns=tukey.summary().data[0],
        )

        tukey_results = {}
        for _, row in tukey_df.iterrows():
            pair = f"{row['group1']}_vs_{row['group2']}"
            tukey_results[pair] = {
                "meandiff": float(row["meandiff"]),
                "p_adj": float(row["p-adj"]),
                "lower": float(row["lower"]),
                "upper": float(row["upper"]),
                "reject": bool(row["reject"]),
            }

        # Calculate eta-squared (effect size for ANOVA)
        # η² = SS_between / SS_total
        grand_mean = np.mean(all_values)
        ss_between = sum(
            len(group) * (np.mean(group) - grand_mean) ** 2
            for group in [precision_bm25, precision_vector, precision_hybrid]
        )
        ss_total = sum((x - grand_mean) ** 2 for x in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "eta_squared": float(eta_squared),
            "tukey_results": tukey_results,
            "significant": bool(p_value < self.alpha),
            "mean_precision": {
                "bm25": float(np.mean(precision_bm25)),
                "vector": float(np.mean(precision_vector)),
                "hybrid": float(np.mean(precision_hybrid)),
            },
            "sample_sizes": {
                "bm25": len(precision_bm25),
                "vector": len(precision_vector),
                "hybrid": len(precision_hybrid),
            },
        }

    def hypothesis_h3_consolidation(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Test H3: Consolidation effect on memory retention.

        Uses logistic regression to model retention probability as a function
        of auto-consolidation, days elapsed, importance, and optionally
        memory type.

        H0: Auto-consolidation does not affect retention
        H1: Auto-consolidation improves retention

        Args:
            data: DataFrame with columns:
                - recalled: Binary outcome (0/1)
                - auto_consolidation: Whether auto-consolidation was applied (0/1)
                - days_elapsed: Days since encoding
                - importance: Importance score (1-5)
                - memory_type: (optional) Type of memory

        Returns:
            Dictionary with logistic regression results:
                - significant: Whether auto_consolidation is significant
                - significant_predictors: List of significant predictor names
                - coefficients: Dict of predictor -> coefficient
                - odds_ratios: Dict of predictor -> odds ratio
                - p_values: Dict of predictor -> p-value
                - accuracy: Classification accuracy
                - sample_size: Number of observations
                - aic: Akaike Information Criterion (if statsmodels available)
                - bic: Bayesian Information Criterion (if statsmodels available)
        """
        import statsmodels.api as sm

        # Build feature matrix
        feature_cols = ["auto_consolidation", "days_elapsed", "importance"]

        # Handle memory_type via dummy encoding if present
        if "memory_type" in data.columns:
            dummies = pd.get_dummies(data["memory_type"], prefix="type", drop_first=True).astype(int)
            X = pd.concat([data[feature_cols], dummies], axis=1)
        else:
            X = data[feature_cols].copy()

        y = data["recalled"]

        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        # Fit logistic regression
        model = sm.Logit(y, X_with_const)
        result = model.fit(disp=0)

        # Extract coefficients (exclude constant)
        predictor_names = [c for c in X_with_const.columns if c != "const"]
        coefficients: dict[str, float] = {}
        odds_ratios: dict[str, float] = {}
        p_values: dict[str, float] = {}
        significant_predictors: list[str] = []

        for name in predictor_names:
            coef = float(result.params[name])
            pval = float(result.pvalues[name])
            coefficients[name] = coef
            odds_ratios[name] = float(np.exp(coef))
            p_values[name] = pval
            if pval < self.alpha:
                significant_predictors.append(name)

        # Calculate accuracy
        predicted_probs = result.predict(X_with_const)
        predicted_classes = (predicted_probs >= 0.5).astype(int)
        accuracy = float((predicted_classes == y).mean())

        return {
            "significant": "auto_consolidation" in significant_predictors,
            "significant_predictors": significant_predictors,
            "coefficients": coefficients,
            "odds_ratios": odds_ratios,
            "p_values": p_values,
            "accuracy": accuracy,
            "sample_size": len(data),
            "aic": float(result.aic),
            "bic": float(result.bic),
        }

    # ── Power Analysis ──────────────────────────────────────────────────

    def calculate_power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.80
    ) -> dict[str, Any]:
        """Calculate required sample size for given effect size and power.

        Args:
            effect_size: Expected Cohen's d
            alpha: Significance level (default: 0.05)
            power: Desired statistical power (default: 0.80)

        Returns:
            Dictionary with:
                - required_n_per_group: Sample size needed per group
                - effect_size: Input effect size
                - alpha: Input alpha
                - power: Input power
                - interpretation: Effect size interpretation
        """
        analysis = TTestIndPower()
        required_n = analysis.solve_power(
            effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"
        )

        # Effect size interpretation
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "required_n_per_group": int(np.ceil(required_n)),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "interpretation": interpretation,
        }

    # ── Inter-Rater Reliability ────────────────────────────────────────

    def calculate_cohens_kappa(
        self, annotator1: list[Any], annotator2: list[Any]
    ) -> float:
        """Calculate Cohen's κ for inter-annotator agreement.

        Args:
            annotator1: Ratings from annotator 1
            annotator2: Ratings from annotator 2

        Returns:
            Cohen's κ value (-1 to 1, where 1 = perfect agreement)
        """
        if len(annotator1) != len(annotator2):
            raise ValueError("Annotator lists must have equal length")

        # Create confusion matrix
        labels = sorted(set(annotator1) | set(annotator2))
        n = len(annotator1)

        # Observed agreement
        p_o = sum(a1 == a2 for a1, a2 in zip(annotator1, annotator2, strict=False)) / n

        # Expected agreement by chance
        p_e = 0
        for label in labels:
            p1 = sum(a == label for a in annotator1) / n
            p2 = sum(a == label for a in annotator2) / n
            p_e += p1 * p2

        # Cohen's κ
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

        return float(kappa)

    def calculate_krippendorff_alpha(
        self, ratings: np.ndarray, level_of_measurement: str = "interval"
    ) -> float:
        """Calculate Krippendorff's α for multi-rater agreement.

        Args:
            ratings: 2D array (annotators × items). Use np.nan for missing values.
            level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'

        Returns:
            Krippendorff's α (-1 to 1, where 1 = perfect agreement)

        Reference:
            Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology.
        """
        # Convert to float array and transpose (items × annotators)
        reliability_data = np.array(ratings, dtype=float).T

        n_items, n_coders = reliability_data.shape

        # Calculate coincidence matrix
        values = np.unique(reliability_data[~np.isnan(reliability_data)])
        n_values = len(values)
        coincidence_matrix = np.zeros((n_values, n_values))

        for item_ratings in reliability_data:
            valid_ratings = item_ratings[~np.isnan(item_ratings)]
            n_valid = len(valid_ratings)

            if n_valid < 2:
                continue

            for i, val1 in enumerate(valid_ratings):
                for j, val2 in enumerate(valid_ratings):
                    if i != j:
                        idx1 = np.where(values == val1)[0][0]
                        idx2 = np.where(values == val2)[0][0]
                        coincidence_matrix[idx1, idx2] += 1 / (n_valid - 1)

        # Calculate observed disagreement
        n_c = np.sum(coincidence_matrix)
        if n_c == 0:
            return 0.0

        delta = self._krippendorff_delta(values, level_of_measurement)
        d_o = np.sum(coincidence_matrix * delta) / n_c

        # Calculate expected disagreement
        n_k = np.sum(coincidence_matrix, axis=1)
        d_e = 0.0
        for i in range(n_values):
            for j in range(n_values):
                if i != j:
                    d_e += n_k[i] * n_k[j] * delta[i, j]
        d_e /= n_c * (n_c - 1)

        # Krippendorff's α
        alpha = 1 - (d_o / d_e) if d_e > 0 else 1.0

        return float(alpha)

    def _krippendorff_delta(
        self, values: np.ndarray, level: str
    ) -> np.ndarray:
        """Calculate difference function for Krippendorff's α.

        Args:
            values: Unique rating values
            level: Level of measurement

        Returns:
            Distance matrix
        """
        n = len(values)
        delta = np.zeros((n, n))

        if level == "nominal":
            # Nominal: 0 if same, 1 if different
            for i in range(n):
                for j in range(n):
                    delta[i, j] = 0 if i == j else 1

        elif level == "ordinal":
            # Ordinal: Squared differences in cumulative frequencies
            # Simplified version for now
            for i in range(n):
                for j in range(n):
                    delta[i, j] = (values[i] - values[j]) ** 2

        elif level in ["interval", "ratio"]:
            # Interval/Ratio: Squared differences
            for i in range(n):
                for j in range(n):
                    delta[i, j] = (values[i] - values[j]) ** 2

        return delta

    # ── Descriptive Statistics ─────────────────────────────────────────

    def calculate_descriptive_stats(self, data: list[float]) -> dict[str, float]:
        """Calculate comprehensive descriptive statistics.

        Args:
            data: List of numeric values

        Returns:
            Dictionary with mean, median, std, min, max, percentiles
        """
        arr = np.array(data)

        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "n": len(arr),
        }
