# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Metrics collection and calculation tools.

This module provides tools for measuring performance metrics including:
- Latency measurement (priming, search, response)
- Precision, Recall, F1 calculation
- Token counting
- Statistical aggregations
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Latency Measurement ─────────────────────────────────────────────────────


class LatencyMeasurer:
    """Measures execution latency with high precision."""

    @staticmethod
    def measure_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
        """Measure latency of a synchronous function.

        Args:
            func: Function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms

    @staticmethod
    async def measure_async(
        func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> tuple[T, float]:
        """Measure latency of an asynchronous function.

        Args:
            func: Async function to measure
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return result, latency_ms


# ── Search Quality Metrics ──────────────────────────────────────────────────


class SearchMetrics:
    """Calculates search quality metrics (Precision, Recall, F1)."""

    @staticmethod
    def calculate_precision_recall_f1(
        retrieved: list[str], relevant: list[str], k: int | None = None
    ) -> tuple[float, float, float]:
        """Calculate Precision@k, Recall@k, and F1@k.

        Args:
            retrieved: List of retrieved document IDs (in ranked order)
            relevant: List of relevant document IDs (ground truth)
            k: Top-k cutoff (if None, use all retrieved)

        Returns:
            Tuple of (precision, recall, f1)

        Examples:
            >>> SearchMetrics.calculate_precision_recall_f1(
            ...     retrieved=["doc1", "doc2", "doc3"],
            ...     relevant=["doc1", "doc3", "doc4"],
            ...     k=3
            ... )
            (0.6666..., 0.6666..., 0.6666...)
        """
        if k is not None:
            retrieved = retrieved[:k]

        if not retrieved:
            return 0.0, 0.0, 0.0

        retrieved_set = set(retrieved)
        relevant_set = set(relevant)

        true_positives = len(retrieved_set & relevant_set)

        # Precision@k = TP / k
        precision = true_positives / len(retrieved) if retrieved else 0.0

        # Recall@k = TP / |relevant|
        recall = true_positives / len(relevant) if relevant else 0.0

        # F1@k = 2 * (P * R) / (P + R)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    @staticmethod
    def calculate_mean_average_precision(
        retrieved: list[str], relevant: list[str]
    ) -> float:
        """Calculate Mean Average Precision (MAP).

        Args:
            retrieved: List of retrieved document IDs (in ranked order)
            relevant: List of relevant document IDs (ground truth)

        Returns:
            Mean Average Precision score

        Examples:
            >>> SearchMetrics.calculate_mean_average_precision(
            ...     retrieved=["doc1", "doc2", "doc3"],
            ...     relevant=["doc1", "doc3"]
            ... )
            0.8333...
        """
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        precision_at_k = []
        num_relevant_seen = 0

        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_k.append(num_relevant_seen / i)

        if not precision_at_k:
            return 0.0

        return sum(precision_at_k) / len(relevant)

    @staticmethod
    def calculate_ndcg(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@k).

        Args:
            retrieved: List of retrieved document IDs (in ranked order)
            relevant: List of relevant document IDs (ground truth)
            k: Top-k cutoff (if None, use all retrieved)

        Returns:
            NDCG@k score

        Examples:
            >>> SearchMetrics.calculate_ndcg(
            ...     retrieved=["doc1", "doc2", "doc3"],
            ...     relevant=["doc1", "doc3"],
            ...     k=3
            ... )
            0.7613...
        """
        if k is not None:
            retrieved = retrieved[:k]

        if not retrieved or not relevant:
            return 0.0

        relevant_set = set(relevant)

        # DCG: sum(rel_i / log2(i + 1)) for i in 1..k
        dcg = sum(
            (1.0 / np.log2(i + 1)) if doc_id in relevant_set else 0.0
            for i, doc_id in enumerate(retrieved, start=1)
        )

        # IDCG: DCG of perfect ranking
        ideal_length = min(len(retrieved), len(relevant))
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_length + 1))

        return dcg / idcg if idcg > 0 else 0.0


# ── Token Counting ──────────────────────────────────────────────────────────


class TokenCounter:
    """Counts tokens in text.

    Uses a simple heuristic: 1 token ≈ 4 characters (as per OpenAI estimate).
    For more accurate counting, integrate with tiktoken or the actual model's tokenizer.
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count using character count heuristic.

        Args:
            text: Input text

        Returns:
            Estimated token count

        Notes:
            This is a rough estimate. For production use, integrate with
            the actual tokenizer (e.g., tiktoken for Claude/OpenAI models).
        """
        return max(1, len(text) // 4)


# ── Statistical Aggregations ────────────────────────────────────────────────


class StatisticalAggregator:
    """Calculates statistical summaries of metrics."""

    @staticmethod
    def calculate_percentiles(values: list[float], percentiles: list[int]) -> dict[str, float]:
        """Calculate percentile statistics.

        Args:
            values: List of numeric values
            percentiles: List of percentiles to calculate (e.g., [50, 95, 99])

        Returns:
            Dictionary mapping percentile names to values

        Examples:
            >>> StatisticalAggregator.calculate_percentiles([1, 2, 3, 4, 5], [50, 95])
            {'p50': 3.0, 'p95': 4.8}
        """
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}

        return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}

    @staticmethod
    def calculate_summary_stats(values: list[float]) -> dict[str, float]:
        """Calculate summary statistics (mean, median, std, min, max).

        Args:
            values: List of numeric values

        Returns:
            Dictionary of summary statistics

        Examples:
            >>> stats = StatisticalAggregator.calculate_summary_stats([1, 2, 3, 4, 5])
            >>> stats['mean']
            3.0
            >>> stats['median']
            3.0
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }


# ── Main Metrics Collector ──────────────────────────────────────────────────


class MetricsCollector:
    """Main metrics collector integrating all measurement tools.

    This class provides a unified interface for collecting all experiment metrics:
    - Latency measurements
    - Search quality metrics
    - Token counting
    - Statistical aggregations

    Examples:
        >>> collector = MetricsCollector()
        >>> # Measure async function latency
        >>> result, latency = await collector.measure_latency_async(some_async_func, arg1, arg2)
        >>> # Calculate search metrics
        >>> p, r, f1 = collector.calculate_precision_recall(retrieved, relevant, k=3)
    """

    def __init__(self):
        self.latency_measurer = LatencyMeasurer()
        self.search_metrics = SearchMetrics()
        self.token_counter = TokenCounter()
        self.aggregator = StatisticalAggregator()

    # ── Latency Measurement ─────────────────────────────────────────────────

    def measure_latency_sync(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> tuple[T, float]:
        """Measure latency of a synchronous function (returns result, latency_ms)."""
        return self.latency_measurer.measure_sync(func, *args, **kwargs)

    async def measure_latency_async(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> tuple[T, float]:
        """Measure latency of an asynchronous function (returns result, latency_ms)."""
        return await self.latency_measurer.measure_async(func, *args, **kwargs)

    # ── Search Quality ──────────────────────────────────────────────────────

    def calculate_precision_recall(
        self, retrieved: list[str], relevant: list[str], k: int = 3
    ) -> tuple[float, float, float]:
        """Calculate Precision@k, Recall@k, F1@k.

        Args:
            retrieved: Retrieved document IDs
            relevant: Ground truth relevant document IDs
            k: Top-k cutoff

        Returns:
            Tuple of (precision, recall, f1)
        """
        return self.search_metrics.calculate_precision_recall_f1(retrieved, relevant, k)

    def calculate_map(self, retrieved: list[str], relevant: list[str]) -> float:
        """Calculate Mean Average Precision."""
        return self.search_metrics.calculate_mean_average_precision(retrieved, relevant)

    def calculate_ndcg(self, retrieved: list[str], relevant: list[str], k: int = 3) -> float:
        """Calculate NDCG@k."""
        return self.search_metrics.calculate_ndcg(retrieved, relevant, k)

    # ── Token Counting ──────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Estimate token count in text."""
        return self.token_counter.estimate_tokens(text)

    # ── Statistical Aggregations ────────────────────────────────────────────

    def calculate_percentiles(
        self, values: list[float], percentiles: list[int] | None = None
    ) -> dict[str, float]:
        """Calculate percentile statistics."""
        if percentiles is None:
            percentiles = [50, 95, 99]
        return self.aggregator.calculate_percentiles(values, percentiles)

    def calculate_summary_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate summary statistics."""
        return self.aggregator.calculate_summary_stats(values)
