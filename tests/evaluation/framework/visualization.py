# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Visualization tools for memory performance evaluation.

This module generates publication-quality figures for research papers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

logger = logging.getLogger(__name__)


# ── Experiment Visualizer ─────────────────────────────────────────────────


class ExperimentVisualizer:
    """Generate publication-quality graphs for memory experiments."""

    def __init__(self, style: str = "publication"):
        """Initialize visualizer with publication settings.

        Args:
            style: 'publication' or 'presentation'
        """
        self.style = style
        self.setup_publication_style()

    def setup_publication_style(self):
        """Configure matplotlib/seaborn for publication-quality output."""
        # Set seaborn style
        sns.set_theme(style="whitegrid", context="paper")

        # Publication settings
        if self.style == "publication":
            rcParams.update(
                {
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "DejaVu Serif"],
                    "font.size": 10,
                    "axes.labelsize": 10,
                    "axes.titlesize": 11,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 9,
                    "figure.titlesize": 12,
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "savefig.bbox": "tight",
                    "savefig.pad_inches": 0.1,
                    "axes.linewidth": 0.8,
                    "grid.linewidth": 0.5,
                    "lines.linewidth": 1.5,
                    "patch.linewidth": 0.5,
                    "xtick.major.width": 0.8,
                    "ytick.major.width": 0.8,
                }
            )
        else:  # presentation
            rcParams.update(
                {
                    "font.size": 14,
                    "axes.labelsize": 16,
                    "axes.titlesize": 18,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "legend.fontsize": 14,
                    "figure.titlesize": 20,
                    "figure.dpi": 150,
                }
            )

        # Colorblind-friendly palette
        self.colors = sns.color_palette("colorblind")
        self.condition_colors = {
            "bm25": self.colors[0],  # Blue
            "vector": self.colors[1],  # Orange
            "hybrid": self.colors[2],  # Green
            "hybrid_priming": self.colors[3],  # Red
        }

    # ── Latency Comparison ──────────────────────────────────────────────

    def plot_latency_comparison(
        self,
        data: pd.DataFrame,
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot latency comparison across conditions.

        Creates a combined box plot and violin plot showing P50, P95, P99.

        Args:
            data: DataFrame with columns ['condition', 'latency']
            output_path: Path to save figure (without extension)
            formats: List of formats to save ['png', 'pdf', 'svg']
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Determine conditions from data
        conditions = data["condition"].unique().tolist()

        # Build labels for tick marks
        label_map = {
            "bm25": "BM25",
            "vector": "Vector",
            "hybrid": "Hybrid",
            "hybrid_priming": "Hybrid+Priming",
        }
        tick_labels = [label_map.get(c, c) for c in conditions]

        # Box plot
        ax1 = axes[0]
        sns.boxplot(
            data=data,
            x="condition",
            y="latency",
            ax=ax1,
            palette=self.condition_colors,
            showfliers=False,
        )
        ax1.set_xlabel("Search Strategy")
        ax1.set_ylabel("Response Latency (seconds)")
        ax1.set_title("Response Latency Distribution")
        ax1.set_xticks(range(len(conditions)))
        ax1.set_xticklabels(tick_labels)

        # Add percentile markers
        for i, condition in enumerate(conditions):
            cond_data = data[data["condition"] == condition]["latency"]
            if len(cond_data) == 0:
                continue
            p95 = np.percentile(cond_data, 95)
            p99 = np.percentile(cond_data, 99)
            ax1.plot(i, p95, "r^", markersize=6, label="P95" if i == 0 else "")
            ax1.plot(i, p99, "rv", markersize=6, label="P99" if i == 0 else "")

        ax1.legend()

        # Violin plot
        ax2 = axes[1]
        sns.violinplot(
            data=data,
            x="condition",
            y="latency",
            ax=ax2,
            palette=self.condition_colors,
            inner="quartile",
        )
        ax2.set_xlabel("Search Strategy")
        ax2.set_ylabel("Response Latency (seconds)")
        ax2.set_title("Latency Density Distribution")
        ax2.set_xticks(range(len(conditions)))
        ax2.set_xticklabels(tick_labels)

        plt.tight_layout()

        # Save in multiple formats
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved latency comparison to {output_file}")

        plt.close()

    # ── Precision-Recall Curves ─────────────────────────────────────────

    def plot_precision_recall_curves(
        self,
        results: dict[str, dict[str, list[float]]],
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot Precision-Recall curves for different search strategies.

        Args:
            results: Dictionary mapping condition -> {'precision': [...], 'recall': [...]}
            output_path: Path to save figure
            formats: List of formats to save
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, ax = plt.subplots(figsize=(6, 5))

        condition_labels = {
            "bm25": "BM25 Only",
            "vector": "Vector Only",
            "hybrid": "Hybrid Search",
            "hybrid_priming": "Hybrid + Priming",
        }

        for condition, values in results.items():
            precision = values["precision"]
            recall = values["recall"]
            color = self.condition_colors.get(condition, "gray")
            label = condition_labels.get(condition, condition)

            ax.plot(recall, precision, marker="o", color=color, label=label, linewidth=2)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved P-R curves to {output_file}")

        plt.close()

    # ── Scalability Analysis ────────────────────────────────────────────

    def plot_scalability(
        self,
        memory_sizes: list[int],
        latencies: dict[str, list[float]],
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot scalability: memory size vs latency.

        Args:
            memory_sizes: List of memory sizes (number of files)
            latencies: Dictionary mapping condition -> [latency at each size]
            output_path: Path to save figure
            formats: List of formats to save
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, ax = plt.subplots(figsize=(7, 5))

        condition_labels = {
            "bm25": "BM25 Only",
            "vector": "Vector Only",
            "hybrid": "Hybrid Search",
            "hybrid_priming": "Hybrid + Priming",
        }

        for condition, values in latencies.items():
            color = self.condition_colors.get(condition, "gray")
            label = condition_labels.get(condition, condition)

            ax.plot(
                memory_sizes,
                values,
                marker="o",
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Memory Size (number of files)")
        ax.set_ylabel("Mean Response Latency (seconds)")
        ax.set_title("Scalability: Latency vs Memory Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        # Add reference lines for complexity classes
        x_range = np.array(memory_sizes)
        # O(1) - constant
        ax.plot(x_range, [values[0]] * len(x_range), "--", alpha=0.3, label="O(1)")
        # O(log n)
        if len(x_range) > 1:
            log_ref = values[0] * np.log(x_range) / np.log(x_range[0])
            ax.plot(x_range, log_ref, "--", alpha=0.3, label="O(log n)")

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved scalability plot to {output_file}")

        plt.close()

    # ── Retention Rate ──────────────────────────────────────────────────

    def plot_retention_rate(
        self,
        data: pd.DataFrame,
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot memory retention rate over time.

        Args:
            data: DataFrame with columns ['memory_type', 'day', 'retention_rate']
            output_path: Path to save figure
            formats: List of formats to save
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, ax = plt.subplots(figsize=(7, 5))

        memory_types = data["memory_type"].unique()
        days = sorted(data["day"].unique())

        type_colors = {
            "episodic": self.colors[0],
            "semantic": self.colors[1],
            "procedural": self.colors[2],
        }

        for mem_type in memory_types:
            type_data = data[data["memory_type"] == mem_type]
            retention_by_day = []

            for day in days:
                day_data = type_data[type_data["day"] == day]
                if len(day_data) > 0:
                    retention_by_day.append(day_data["retention_rate"].mean())
                else:
                    retention_by_day.append(np.nan)

            color = type_colors.get(mem_type, "gray")
            ax.plot(
                days,
                retention_by_day,
                marker="o",
                color=color,
                label=mem_type.capitalize(),
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Days After Encoding")
        ax.set_ylabel("Retention Rate")
        ax.set_title("Memory Retention Rate Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Add reference line at 80%
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80% target")

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved retention rate plot to {output_file}")

        plt.close()

    # ── Token Consumption ───────────────────────────────────────────────

    def plot_token_consumption(
        self,
        conditions: list[str],
        tokens: list[float],
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot token consumption comparison.

        Args:
            conditions: List of condition names
            tokens: Mean tokens consumed per condition
            output_path: Path to save figure
            formats: List of formats to save
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, ax = plt.subplots(figsize=(7, 5))

        colors = [self.condition_colors.get(cond, "gray") for cond in conditions]
        labels = [
            {"bm25": "BM25", "vector": "Vector", "hybrid": "Hybrid", "hybrid_priming": "Hybrid+Priming"}.get(
                cond, cond
            )
            for cond in conditions
        ]

        bars = ax.bar(range(len(conditions)), tokens, color=colors)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Search Strategy")
        ax.set_ylabel("Mean Tokens Consumed")
        ax.set_title("Token Consumption by Condition")
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, tokens, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.0f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved token consumption plot to {output_file}")

        plt.close()

    # ── Hypothesis Results ──────────────────────────────────────────────

    def plot_hypothesis_results(
        self,
        h1_result: dict[str, Any],
        h2_result: dict[str, Any],
        h3_result: dict[str, Any],
        output_path: Path,
        formats: list[str] | None = None,
    ):
        """Plot integrated view of all hypothesis test results.

        Args:
            h1_result: Results from hypothesis_h1_priming_effect()
            h2_result: Results from hypothesis_h2_hybrid_search()
            h3_result: Results from hypothesis_h3_consolidation()
            output_path: Path to save figure
            formats: List of formats to save
        """
        if formats is None:
            formats = ["png", "pdf"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # H1: Priming effect on latency
        ax1 = axes[0, 0]
        conditions = ["Hybrid", "Hybrid+Priming"]
        latencies = [h1_result["mean_hybrid"], h1_result["mean_priming"]]
        bars = ax1.bar(conditions, latencies, color=[self.colors[2], self.colors[3]])
        ax1.set_ylabel("Mean Latency (s)")
        ax1.set_title(f"H1: Priming Effect\n(p={h1_result['p_value']:.4f}, d={h1_result['effect_size']:.2f})")
        ax1.grid(True, axis="y", alpha=0.3)

        # Add significance indicator
        if h1_result["significant"]:
            y_max = max(latencies) * 1.1
            ax1.plot([0, 1], [y_max, y_max], "k-", linewidth=1)
            ax1.text(0.5, y_max * 1.02, "***", ha="center", fontsize=14)

        # H2: Hybrid search superiority
        ax2 = axes[0, 1]
        search_methods = ["BM25", "Vector", "Hybrid"]
        precisions = [
            h2_result["mean_precision"]["bm25"],
            h2_result["mean_precision"]["vector"],
            h2_result["mean_precision"]["hybrid"],
        ]
        colors_h2 = [self.colors[0], self.colors[1], self.colors[2]]
        ax2.bar(search_methods, precisions, color=colors_h2)
        ax2.set_ylabel("Mean Precision@k")
        ax2.set_title(f"H2: Hybrid Search\n(p={h2_result['p_value']:.4f}, η²={h2_result['eta_squared']:.3f})")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_ylim([0, 1.0])

        # H3: Consolidation effect
        ax3 = axes[1, 0]
        if "auto_consolidation" in h3_result["coefficients"]:
            coef = h3_result["coefficients"]["auto_consolidation"]
            odds = h3_result["odds_ratios"]["auto_consolidation"]
            p_val = h3_result["p_values"]["auto_consolidation"]

            ax3.barh(["Coefficient", "Odds Ratio"], [coef, odds], color=self.colors[3])
            ax3.set_xlabel("Value")
            ax3.set_title(f"H3: Auto-Consolidation Effect\n(p={p_val:.4f})")
            ax3.grid(True, axis="x", alpha=0.3)
            ax3.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="OR=1 (no effect)")
            ax3.legend()

        # Effect sizes comparison
        ax4 = axes[1, 1]
        effect_names = [
            "H1: Priming\n(Cohen's d)",
            "H2: Hybrid\n(η²)",
            "H3: Consolidation\n(OR)",
        ]
        effect_values = [
            h1_result["effect_size"],
            h2_result["eta_squared"],
            h3_result["odds_ratios"].get("auto_consolidation", 1.0),
        ]
        bars = ax4.barh(effect_names, effect_values, color=[self.colors[3], self.colors[2], self.colors[1]])
        ax4.set_xlabel("Effect Size")
        ax4.set_title("Summary: Effect Sizes")
        ax4.grid(True, axis="x", alpha=0.3)

        # Add interpretation labels
        for i, (bar, interp) in enumerate(
            zip(
                bars,
                [
                    h1_result["interpretation"],
                    "medium" if h2_result["eta_squared"] > 0.06 else "small",
                    "significant" if h3_result["odds_ratios"].get("auto_consolidation", 1.0) > 1.5 else "modest",
                ],
                strict=False,
            )
        ):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height() / 2, f"  {interp}", va="center")

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            output_file = output_path.with_suffix(f".{fmt}")
            plt.savefig(output_file, format=fmt)
            logger.info(f"Saved hypothesis results to {output_file}")

        plt.close()

    # ── Generate All Figures ────────────────────────────────────────────

    def generate_all_figures(
        self, results_dir: Path, output_dir: Path
    ) -> list[Path]:
        """Generate all publication figures from results directory.

        Args:
            results_dir: Directory containing processed results
            output_dir: Directory to save figures

        Returns:
            List of generated figure paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        logger.info(f"Generating figures from {results_dir} to {output_dir}")

        # Load results (assuming specific file names)
        try:
            # Latency data
            latency_file = results_dir / "latency_data.csv"
            if latency_file.exists():
                latency_data = pd.read_csv(latency_file)
                fig_path = output_dir / "fig1_latency_comparison"
                self.plot_latency_comparison(latency_data, fig_path)
                generated_files.extend([fig_path.with_suffix(".png"), fig_path.with_suffix(".pdf")])

            # Precision-recall data
            pr_file = results_dir / "precision_recall.json"
            if pr_file.exists():
                import json

                with open(pr_file) as f:
                    pr_data = json.load(f)
                fig_path = output_dir / "fig2_precision_recall"
                self.plot_precision_recall_curves(pr_data, fig_path)
                generated_files.extend([fig_path.with_suffix(".png"), fig_path.with_suffix(".pdf")])

            # Scalability data
            scalability_file = results_dir / "scalability.csv"
            if scalability_file.exists():
                scal_data = pd.read_csv(scalability_file)
                memory_sizes = scal_data["memory_size"].unique().tolist()
                latencies_by_condition = {}
                for condition in scal_data["condition"].unique():
                    cond_data = scal_data[scal_data["condition"] == condition]
                    latencies_by_condition[condition] = cond_data["latency"].tolist()

                fig_path = output_dir / "fig3_scalability"
                self.plot_scalability(memory_sizes, latencies_by_condition, fig_path)
                generated_files.extend([fig_path.with_suffix(".png"), fig_path.with_suffix(".pdf")])

            # Retention data
            retention_file = results_dir / "retention.csv"
            if retention_file.exists():
                retention_data = pd.read_csv(retention_file)
                fig_path = output_dir / "fig4_retention_rate"
                self.plot_retention_rate(retention_data, fig_path)
                generated_files.extend([fig_path.with_suffix(".png"), fig_path.with_suffix(".pdf")])

            logger.info(f"Generated {len(generated_files)} figure files")

        except Exception as e:
            logger.error(f"Error generating figures: {e}")

        return generated_files
