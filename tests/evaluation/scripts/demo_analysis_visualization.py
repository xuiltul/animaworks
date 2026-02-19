#!/usr/bin/env python3

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Demo script for Phase 3-4: Statistical Analysis and Visualization.

This script demonstrates the statistical analysis and visualization capabilities
for memory performance evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run demo of statistical analysis and visualization."""
    logger.info("=== Phase 3-4 Demo: Statistical Analysis & Visualization ===")

    # Import modules
    try:
        import sys
        from pathlib import Path

        # Add the tests directory to the Python path
        tests_dir = Path(__file__).parent.parent.parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))

        from evaluation.framework.analysis import StatisticalAnalyzer
        from evaluation.framework.visualization import ExperimentVisualizer
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install -e '.[evaluation]'")
        import traceback

        traceback.print_exc()
        return

    # Create output directory
    output_dir = Path("tests/evaluation/demo_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")

    # ── Step 1: Generate Synthetic Experimental Data ──────────────────

    logger.info("\n--- Step 1: Generating Synthetic Experimental Data ---")

    np.random.seed(42)

    # 1.1. Latency Data (4 conditions × 30 samples)
    logger.info("Generating latency data...")
    latency_data = []
    condition_params = {
        "bm25": {"shape": 5, "scale": 1.0},  # Mean ~5.0s
        "vector": {"shape": 5, "scale": 0.9},  # Mean ~4.5s
        "hybrid": {"shape": 5, "scale": 0.8},  # Mean ~4.0s
        "hybrid_priming": {"shape": 5, "scale": 0.56},  # Mean ~2.8s (44% reduction)
    }

    for condition, params in condition_params.items():
        latencies = np.random.gamma(shape=params["shape"], scale=params["scale"], size=30)
        for lat in latencies:
            latency_data.append({"condition": condition, "latency": lat})

    latency_df = pd.DataFrame(latency_data)
    logger.info(f"  Generated {len(latency_df)} latency samples")

    # 1.2. Precision Data (3 methods × 30 samples)
    logger.info("Generating precision data...")
    precision_data = {
        "bm25": np.random.beta(a=6, b=4, size=30).tolist(),  # Mean ~0.60
        "vector": np.random.beta(a=7, b=3, size=30).tolist(),  # Mean ~0.70
        "hybrid": np.random.beta(a=8, b=2, size=30).tolist(),  # Mean ~0.80
    }

    # 1.3. Retention Data (200 samples with logistic model)
    logger.info("Generating retention data...")
    retention_data = []

    for _ in range(200):
        auto_consolidation = np.random.choice([0, 1])
        days_elapsed = np.random.choice([7, 30])
        importance = np.random.randint(1, 6)
        memory_type = np.random.choice(["episodic", "semantic", "procedural"])

        # Logistic model
        logit = -1.0  # Base
        logit += 1.5 * auto_consolidation  # Strong positive effect
        logit -= 0.03 * days_elapsed  # Time decay
        logit += 0.4 * importance  # Importance helps

        # Memory type effects
        if memory_type == "semantic":
            logit += 0.5
        elif memory_type == "procedural":
            logit += 0.3

        prob = 1 / (1 + np.exp(-logit))
        recalled = 1 if np.random.random() < prob else 0

        retention_data.append(
            {
                "recalled": recalled,
                "auto_consolidation": auto_consolidation,
                "days_elapsed": days_elapsed,
                "importance": importance,
                "memory_type": memory_type,
            }
        )

    retention_df = pd.DataFrame(retention_data)
    logger.info(f"  Generated {len(retention_df)} retention samples")

    # ── Step 2: Statistical Analysis ──────────────────────────────────

    logger.info("\n--- Step 2: Running Statistical Analyses ---")

    analyzer = StatisticalAnalyzer(alpha=0.05)

    # 2.1. Hypothesis H1: Priming Effect
    logger.info("\n[H1] Testing priming effect on latency...")

    latencies_hybrid = latency_df[latency_df["condition"] == "hybrid"]["latency"].tolist()
    latencies_priming = latency_df[latency_df["condition"] == "hybrid_priming"]["latency"].tolist()

    h1_result = analyzer.hypothesis_h1_priming_effect(latencies_hybrid, latencies_priming)

    logger.info(f"  t-statistic: {h1_result['t_statistic']:.3f}")
    logger.info(f"  p-value: {h1_result['p_value']:.6f}")
    logger.info(f"  Effect size (Cohen's d): {h1_result['effect_size']:.3f} ({h1_result['interpretation']})")
    logger.info(f"  Mean reduction: {h1_result['mean_reduction_pct']:.1f}%")
    logger.info(f"  95% CI: ({h1_result['confidence_interval'][0]:.3f}, {h1_result['confidence_interval'][1]:.3f})")
    logger.info(f"  Significant: {h1_result['significant']} (α=0.05)")

    # 2.2. Hypothesis H2: Hybrid Search Superiority
    logger.info("\n[H2] Testing hybrid search superiority...")

    h2_result = analyzer.hypothesis_h2_hybrid_search(
        precision_data["bm25"], precision_data["vector"], precision_data["hybrid"]
    )

    logger.info(f"  F-statistic: {h2_result['f_statistic']:.3f}")
    logger.info(f"  p-value: {h2_result['p_value']:.6f}")
    logger.info(f"  Effect size (η²): {h2_result['eta_squared']:.3f}")
    logger.info(f"  Mean Precision:")
    for method, precision in h2_result["mean_precision"].items():
        logger.info(f"    {method}: {precision:.3f}")
    logger.info(f"  Significant: {h2_result['significant']} (α=0.05)")

    logger.info(f"  Post-hoc (Tukey HSD):")
    for pair, result in h2_result["tukey_results"].items():
        if result["reject"]:
            logger.info(f"    {pair}: SIGNIFICANT (p={result['p_adj']:.4f})")

    # 2.3. Hypothesis H3: Consolidation Effect
    logger.info("\n[H3] Testing auto-consolidation effect...")

    h3_result = analyzer.hypothesis_h3_consolidation(retention_df)

    logger.info(f"  Model AIC: {h3_result['aic']:.1f}")
    logger.info(f"  Model BIC: {h3_result['bic']:.1f}")
    logger.info(f"  Accuracy: {h3_result['accuracy']:.3f}")
    logger.info(f"  Significant predictors: {', '.join(h3_result['significant_predictors'])}")

    if "auto_consolidation" in h3_result["odds_ratios"]:
        or_value = h3_result["odds_ratios"]["auto_consolidation"]
        p_value = h3_result["p_values"]["auto_consolidation"]
        logger.info(f"  Auto-consolidation:")
        logger.info(f"    Odds Ratio: {or_value:.3f}")
        logger.info(f"    p-value: {p_value:.6f}")
        logger.info(f"    Interpretation: {or_value:.1f}x higher odds of retention")

    # 2.4. Power Analysis
    logger.info("\n[Power Analysis] Sample size calculation...")

    power_result = analyzer.calculate_power_analysis(effect_size=0.5, alpha=0.05, power=0.80)

    logger.info(f"  For medium effect size (d=0.5):")
    logger.info(f"    Required n per group: {power_result['required_n_per_group']}")
    logger.info(f"    Our sample size: 30 per group ✓")

    # 2.5. Inter-Rater Reliability Demo
    logger.info("\n[Inter-Rater Reliability] Example calculation...")

    # Simulate 2 annotators rating 20 items
    annotator1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
    annotator2 = [1, 2, 3, 1, 2, 2, 1, 3, 3, 1, 2, 3, 1, 1, 3, 1, 2, 3, 2, 2]  # ~80% agreement

    kappa = analyzer.calculate_cohens_kappa(annotator1, annotator2)
    logger.info(f"  Cohen's κ: {kappa:.3f}")

    if kappa > 0.80:
        logger.info(f"  Interpretation: Excellent agreement")
    elif kappa > 0.60:
        logger.info(f"  Interpretation: Good agreement")
    else:
        logger.info(f"  Interpretation: Moderate agreement")

    # ── Step 3: Visualization ─────────────────────────────────────────

    logger.info("\n--- Step 3: Generating Visualizations ---")

    visualizer = ExperimentVisualizer(style="publication")

    # 3.1. Latency Comparison
    logger.info("\nGenerating latency comparison plot...")
    visualizer.plot_latency_comparison(
        latency_df, output_dir / "fig1_latency_comparison", formats=["png", "pdf"]
    )
    logger.info(f"  ✓ Saved: {output_dir}/fig1_latency_comparison.png")
    logger.info(f"  ✓ Saved: {output_dir}/fig1_latency_comparison.pdf")

    # 3.2. Precision-Recall Curves
    logger.info("\nGenerating precision-recall curves...")
    pr_data = {
        "bm25": {"precision": [0.9, 0.7, 0.5, 0.3, 0.2], "recall": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "vector": {"precision": [0.95, 0.8, 0.65, 0.45, 0.3], "recall": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "hybrid": {"precision": [0.98, 0.88, 0.75, 0.55, 0.4], "recall": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "hybrid_priming": {
            "precision": [0.99, 0.92, 0.82, 0.65, 0.5],
            "recall": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
    }

    visualizer.plot_precision_recall_curves(pr_data, output_dir / "fig2_precision_recall", formats=["png", "pdf"])
    logger.info(f"  ✓ Saved: {output_dir}/fig2_precision_recall.png")

    # 3.3. Scalability
    logger.info("\nGenerating scalability plot...")
    memory_sizes = [50, 500, 5000]
    scalability_latencies = {
        "bm25": [0.1, 0.15, 0.25],
        "vector": [0.1, 0.12, 0.15],
        "hybrid": [0.12, 0.14, 0.18],
        "hybrid_priming": [0.08, 0.09, 0.11],
    }

    visualizer.plot_scalability(
        memory_sizes, scalability_latencies, output_dir / "fig3_scalability", formats=["png"]
    )
    logger.info(f"  ✓ Saved: {output_dir}/fig3_scalability.png")

    # 3.4. Retention Rate
    logger.info("\nGenerating retention rate plot...")
    retention_plot_data = []
    for memory_type in ["episodic", "semantic", "procedural"]:
        for day in [7, 30]:
            type_day_data = retention_df[
                (retention_df["memory_type"] == memory_type) & (retention_df["days_elapsed"] == day)
            ]
            retention_rate = type_day_data["recalled"].mean()
            retention_plot_data.append({"memory_type": memory_type, "day": day, "retention_rate": retention_rate})

    retention_plot_df = pd.DataFrame(retention_plot_data)

    visualizer.plot_retention_rate(retention_plot_df, output_dir / "fig4_retention_rate", formats=["png"])
    logger.info(f"  ✓ Saved: {output_dir}/fig4_retention_rate.png")

    # 3.5. Token Consumption
    logger.info("\nGenerating token consumption plot...")
    token_conditions = ["bm25", "vector", "hybrid", "hybrid_priming"]
    token_consumption = [5000, 4500, 3500, 2000]

    visualizer.plot_token_consumption(
        token_conditions, token_consumption, output_dir / "fig5_token_consumption", formats=["png"]
    )
    logger.info(f"  ✓ Saved: {output_dir}/fig5_token_consumption.png")

    # 3.6. Hypothesis Results Summary
    logger.info("\nGenerating hypothesis results summary...")
    visualizer.plot_hypothesis_results(
        h1_result, h2_result, h3_result, output_dir / "fig6_hypothesis_results", formats=["png", "pdf"]
    )
    logger.info(f"  ✓ Saved: {output_dir}/fig6_hypothesis_results.png")

    # ── Step 4: Summary Statistics ────────────────────────────────────

    logger.info("\n--- Step 4: Summary Statistics ---")

    logger.info("\nDescriptive Statistics (Latency by Condition):")
    for condition in ["bm25", "vector", "hybrid", "hybrid_priming"]:
        cond_data = latency_df[latency_df["condition"] == condition]["latency"].tolist()
        stats = analyzer.calculate_descriptive_stats(cond_data)

        logger.info(f"\n  {condition}:")
        logger.info(f"    Mean: {stats['mean']:.3f}s")
        logger.info(f"    Median: {stats['median']:.3f}s")
        logger.info(f"    Std Dev: {stats['std']:.3f}s")
        logger.info(f"    P95: {stats['p95']:.3f}s")
        logger.info(f"    P99: {stats['p99']:.3f}s")

    # ── Step 5: Save Results ──────────────────────────────────────────

    logger.info("\n--- Step 5: Saving Results ---")

    # Save data files for future use
    latency_df.to_csv(output_dir / "latency_data.csv", index=False)
    retention_df.to_csv(output_dir / "retention_data.csv", index=False)
    retention_plot_df.to_csv(output_dir / "retention_plot_data.csv", index=False)

    logger.info(f"  ✓ Saved data files to {output_dir}/")

    # Save statistical results as JSON
    import json

    # Convert numpy/pandas types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy/pandas types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    results_summary = {
        "h1_priming_effect": convert_to_json_serializable(h1_result),
        "h2_hybrid_search": convert_to_json_serializable(h2_result),
        "h3_consolidation": convert_to_json_serializable(h3_result),
        "power_analysis": convert_to_json_serializable(power_result),
        "inter_rater_reliability": {"cohens_kappa": float(kappa)},
    }

    with open(output_dir / "statistical_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"  ✓ Saved statistical results to {output_dir}/statistical_results.json")

    # ── Final Summary ─────────────────────────────────────────────────

    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nAll outputs saved to: {output_dir.absolute()}")
    logger.info("\nGenerated Files:")
    logger.info("  Figures:")
    logger.info("    - fig1_latency_comparison.png/pdf")
    logger.info("    - fig2_precision_recall.png/pdf")
    logger.info("    - fig3_scalability.png")
    logger.info("    - fig4_retention_rate.png")
    logger.info("    - fig5_token_consumption.png")
    logger.info("    - fig6_hypothesis_results.png/pdf")
    logger.info("  Data:")
    logger.info("    - latency_data.csv")
    logger.info("    - retention_data.csv")
    logger.info("    - statistical_results.json")

    logger.info("\nKey Findings:")
    logger.info(f"  ✓ H1: Priming reduces latency by {h1_result['mean_reduction_pct']:.1f}% (p={h1_result['p_value']:.4f})")
    logger.info(f"  ✓ H2: Hybrid search improves precision (p={h2_result['p_value']:.4f})")
    logger.info(f"  ✓ H3: Auto-consolidation improves retention {h3_result['odds_ratios'].get('auto_consolidation', 1.0):.1f}x")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
