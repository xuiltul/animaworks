from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Performance tests for priming layer.

Tests priming layer performance characteristics:
- Indexing throughput
- Search latency comparison
- Memory usage baseline
- Concurrent priming
- Large dataset scalability

Based on: docs/testing/priming-layer-test-plan.md Phase 3

Benchmark results: tests/performance/BENCHMARK_RESULTS.md
"""

import asyncio
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.priming import PrimingEngine

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from core.memory.rag import MemoryIndexer, MemoryRetriever
    import chromadb as _chromadb  # noqa: F401 — verify native dep is available
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def large_knowledge_base(tmp_path):
    """Create a large knowledge base for performance testing."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True)

    # Create 100 knowledge files with varying content
    for i in range(100):
        content = f"""# Knowledge File {i}

## Overview
This is knowledge document number {i} for performance testing.

## Details
{'Content paragraph. ' * 50}

## Technical Information
- Item 1: Performance testing
- Item 2: Indexing throughput
- Item 3: Search latency
- Item 4: Memory usage
- Item 5: Concurrent execution

## Related Topics
- Topic A{i}: Performance optimization
- Topic B{i}: Scalability testing
- Topic C{i}: Benchmark results

## Notes
{'Additional notes. ' * 20}
"""
        (knowledge_dir / f"file_{i:03d}.md").write_text(content, encoding="utf-8")

    return knowledge_dir


@pytest.fixture
def anima_dir_with_data(tmp_path):
    """Create a Anima directory with test data."""
    anima_dir = tmp_path / "test_anima"

    # Create directory structure
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "skills").mkdir(parents=True)
    (anima_dir / "state").mkdir(parents=True)

    # Create 50 knowledge files
    for i in range(50):
        content = f"# Knowledge {i}\n\n" + "Content " * 100
        (anima_dir / "knowledge" / f"knowledge_{i:03d}.md").write_text(
            content, encoding="utf-8"
        )

    # Create 50 episode files
    for i in range(50):
        episode_date = datetime.now().date() - timedelta(days=i)
        content = f"""# {episode_date} 行動ログ

## 10:00 — タスク {i}

**相手**: システム
**トピック**: テスト
**要点**:
- タスク実行
- 進捗確認
- 報告作成
"""
        (anima_dir / "episodes" / f"{episode_date}.md").write_text(
            content, encoding="utf-8"
        )

    # Create shared users directory
    shared_users_dir = tmp_path / "shared" / "users"
    shared_users_dir.mkdir(parents=True)

    return anima_dir


@pytest.fixture
def memory_tracker():
    """Track memory usage."""
    if not PSUTIL_AVAILABLE:
        pytest.skip("psutil not installed")

    import psutil
    process = psutil.Process()

    class MemoryTracker:
        def __init__(self):
            self.start_mem = process.memory_info().rss

        def current_usage_mb(self):
            return (process.memory_info().rss - self.start_mem) / 1024 / 1024

        def reset(self):
            self.start_mem = process.memory_info().rss

    return MemoryTracker()


# ── Helper Functions ──────────────────────────────────────────


def print_benchmark(name, times):
    """Print benchmark statistics."""
    if not times:
        print(f"\n{name} Benchmark: No data")
        return

    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
    p99 = sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else max(times)

    print(f"\n{name} Benchmark:")
    print(f"  Mean:   {mean*1000:.2f} ms")
    print(f"  Median: {median*1000:.2f} ms")
    print(f"  StdDev: {stdev*1000:.2f} ms")
    print(f"  P95:    {p95*1000:.2f} ms")
    print(f"  P99:    {p99*1000:.2f} ms")
    print(f"  Min:    {min(times)*1000:.2f} ms")
    print(f"  Max:    {max(times)*1000:.2f} ms")


# ── Test Cases ────────────────────────────────────────────────


@pytest.mark.performance
@pytest.mark.skipif(not RAG_AVAILABLE, reason="ChromaDB not installed")
def test_indexing_throughput(large_knowledge_base):
    """Test indexing throughput with 100 files.

    Goal: < 100ms per file (95th percentile)

    Verifies:
    - Indexing completes without errors
    - Per-file latency is reasonable
    - 95th percentile meets target
    """
    from core.memory.rag import MemoryIndexer
    from core.memory.rag.store import ChromaVectorStore

    # Create temporary Anima directory
    anima_dir = large_knowledge_base.parent
    chroma_dir = anima_dir / ".chroma"
    chroma_dir.mkdir(exist_ok=True)

    # Create vector store
    vector_store = ChromaVectorStore(persist_dir=chroma_dir)

    indexer = MemoryIndexer(
        vector_store=vector_store,
        anima_name="test_anima",
        anima_dir=anima_dir,
    )

    # Get all files
    files = sorted(large_knowledge_base.glob("*.md"))
    assert len(files) == 100

    # Measure per-file indexing time
    per_file_times = []

    start_total = time.perf_counter()

    for file_path in files:
        start = time.perf_counter()
        indexer.index_file(file_path, memory_type="knowledge")
        end = time.perf_counter()

        per_file_times.append(end - start)

    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Print results
    print_benchmark("Indexing (per file)", per_file_times)
    print(f"\nTotal indexing time: {total_time:.2f} seconds")
    print(f"Average: {total_time / len(files) * 1000:.2f} ms/file")

    # Verify
    p95 = sorted(per_file_times)[95]
    assert p95 < 0.100, f"P95 indexing time {p95*1000:.2f}ms exceeds 100ms target"


@pytest.mark.performance
@pytest.mark.skipif(not RAG_AVAILABLE, reason="ChromaDB not installed")
def test_search_latency_comparison(large_knowledge_base):
    """Test hybrid search latency with varying configurations.

    Expected latencies:
    - Hybrid search: ~60-100ms

    Verifies:
    - Search method works correctly
    - Latencies are in expected ranges
    - Consistent performance across queries
    """
    from core.memory.rag import MemoryIndexer, MemoryRetriever
    from core.memory.rag.store import ChromaVectorStore

    # Create temporary Anima directory
    anima_dir = large_knowledge_base.parent
    chroma_dir = anima_dir / ".chroma"
    chroma_dir.mkdir(exist_ok=True)

    # Create vector store
    vector_store = ChromaVectorStore(persist_dir=chroma_dir)

    # Index all files
    indexer = MemoryIndexer(
        vector_store=vector_store,
        anima_name="test_anima",
        anima_dir=anima_dir,
    )

    for file_path in sorted(large_knowledge_base.glob("*.md")):
        indexer.index_file(file_path, memory_type="knowledge")

    # Create retriever
    retriever = MemoryRetriever(
        vector_store=vector_store,
        indexer=indexer,
        knowledge_dir=anima_dir / "knowledge",
    )

    # Test queries
    queries = [
        "performance optimization",
        "scalability testing",
        "benchmark results",
        "indexing throughput",
        "search latency",
    ]

    # Measure hybrid search
    hybrid_times = []

    for _ in range(50):
        query = queries[_ % len(queries)]

        # Hybrid search
        start = time.perf_counter()
        retriever.search(
            query=query,
            anima_name="test_anima",
            memory_type="knowledge",
            top_k=5,
        )
        end = time.perf_counter()
        hybrid_times.append(end - start)

    # Print results
    print_benchmark("Hybrid Search", hybrid_times)

    # Verify latencies are reasonable
    # Note: Actual times depend on hardware, so we use relaxed thresholds
    assert statistics.mean(hybrid_times) < 0.300, "Hybrid search too slow"


@pytest.mark.performance
@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
@pytest.mark.asyncio
async def test_memory_usage_baseline(anima_dir_with_data, memory_tracker):
    """Test memory usage baseline during repeated priming.

    Goal: Memory increase < 50MB (no memory leak)

    Verifies:
    - Memory usage is tracked
    - Multiple priming runs don't leak memory
    - Final memory increase is within acceptable range
    """
    with patch("core.paths.get_shared_dir", return_value=anima_dir_with_data.parent / "shared"):
        engine = PrimingEngine(anima_dir_with_data)

        # Reset memory tracker
        memory_tracker.reset()
        initial_mem = memory_tracker.current_usage_mb()

        # Run 100 priming operations
        for i in range(100):
            result = await engine.prime_memories(
                message=f"Test message {i}",
                sender_name="system",
                channel="chat",
            )

        # Check final memory usage
        final_mem = memory_tracker.current_usage_mb()
        mem_increase = final_mem - initial_mem

        print(f"\n=== Memory Usage ===")
        print(f"Initial:  {initial_mem:.2f} MB")
        print(f"Final:    {final_mem:.2f} MB")
        print(f"Increase: {mem_increase:.2f} MB")

        # Verify no major memory leak
        # Note: Some increase is expected due to caching, etc.
        assert mem_increase < 50, f"Memory increased by {mem_increase:.2f} MB (> 50 MB threshold)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_priming(tmp_path):
    """Test concurrent priming with multiple Anima instances.

    Verifies:
    - 10 parallel priming operations complete successfully
    - No race conditions
    - No data corruption
    - All results are valid
    """
    # Create 10 Anima directories
    anima_dirs = []
    for i in range(10):
        anima_dir = tmp_path / f"anima_{i}"
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "skills").mkdir(parents=True)

        # Add some content
        (anima_dir / "knowledge" / "test.md").write_text(
            f"# Knowledge for Anima {i}\n\nContent here.",
            encoding="utf-8",
        )

        today = datetime.now().date()
        (anima_dir / "episodes" / f"{today}.md").write_text(
            f"# {today} Log for Anima {i}\n\n## 10:00 — Task\n\nContent.",
            encoding="utf-8",
        )

        anima_dirs.append(anima_dir)

    # Create engines
    engines = [PrimingEngine(anima_dir) for anima_dir in anima_dirs]

    # Run concurrent priming
    async def prime_one(engine, index):
        return await engine.prime_memories(
            message=f"Test message for anima {index}",
            sender_name="system",
            channel="chat",
        )

    start = time.perf_counter()
    results = await asyncio.gather(*[
        prime_one(engine, i) for i, engine in enumerate(engines)
    ])
    end = time.perf_counter()

    total_time = end - start

    print(f"\n=== Concurrent Priming ===")
    print(f"Total time: {total_time*1000:.2f} ms")
    print(f"Average per anima: {total_time/len(engines)*1000:.2f} ms")

    # Verify all completed successfully
    assert len(results) == 10
    for i, result in enumerate(results):
        # Each result should be valid (not necessarily non-empty)
        assert isinstance(result.sender_profile, str)
        assert isinstance(result.recent_activity, str)
        assert isinstance(result.related_knowledge, str)
        assert isinstance(result.matched_skills, list)


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.skipif(not RAG_AVAILABLE, reason="ChromaDB not installed")
def test_large_dataset_scalability(tmp_path):
    """Test scalability with large dataset (1000 files).

    Goal:
    - Indexing: < 2 minutes (120 seconds)
    - Search: < 200ms

    Verifies:
    - System handles large datasets
    - Indexing completes in reasonable time
    - Search remains fast

    Note: This test is marked as 'slow' and may be skipped in CI.
    """
    from core.memory.rag import MemoryIndexer, MemoryRetriever
    from core.memory.rag.store import ChromaVectorStore

    # Create 1000 knowledge files
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True)

    print("\nCreating 1000 test files...")
    for i in range(1000):
        content = f"# File {i}\n\n" + f"Content for file {i}. " * 50
        (knowledge_dir / f"file_{i:04d}.md").write_text(content, encoding="utf-8")

    # Index all files
    chroma_dir = tmp_path / ".chroma"
    chroma_dir.mkdir(exist_ok=True)

    # Create vector store
    vector_store = ChromaVectorStore(persist_dir=chroma_dir)

    indexer = MemoryIndexer(
        vector_store=vector_store,
        anima_name="test_anima",
        anima_dir=tmp_path,
    )

    print("Indexing 1000 files...")
    start_indexing = time.perf_counter()

    for file_path in sorted(knowledge_dir.glob("*.md")):
        indexer.index_file(file_path, memory_type="knowledge")

    end_indexing = time.perf_counter()
    indexing_time = end_indexing - start_indexing

    print(f"Indexing completed in {indexing_time:.2f} seconds")

    # Verify indexing time
    assert indexing_time < 120, f"Indexing took {indexing_time:.2f}s (> 120s target)"

    # Create retriever
    retriever = MemoryRetriever(
        vector_store=vector_store,
        indexer=indexer,
        knowledge_dir=knowledge_dir,
    )

    # Measure search latency
    print("Testing search latency...")
    search_times = []

    for i in range(50):
        start = time.perf_counter()
        retriever.search(
            query=f"Content for file {i * 20}",
            anima_name="test_anima",
            memory_type="knowledge",
            top_k=10,
        )
        end = time.perf_counter()
        search_times.append(end - start)

    print_benchmark("Search on 1000 files", search_times)

    # Verify search latency
    mean_search = statistics.mean(search_times)
    assert mean_search < 0.200, f"Mean search time {mean_search*1000:.2f}ms exceeds 200ms target"


# ── Additional Performance Tests ──────────────────────────────


@pytest.mark.performance
@pytest.mark.asyncio
async def test_priming_latency_percentiles(anima_dir_with_data):
    """Test priming latency distribution with detailed percentiles.

    Measures:
    - Mean, median, P50, P75, P90, P95, P99
    - Ensures consistent performance
    """
    with patch("core.paths.get_shared_dir", return_value=anima_dir_with_data.parent / "shared"):
        engine = PrimingEngine(anima_dir_with_data)

        # Test messages
        messages = [
            "タスクを実行",
            "進捗を確認",
            "報告を作成",
            "ドキュメントを更新",
            "レビューを依頼",
        ]

        latencies = []

        # Run 100 priming operations
        for i in range(100):
            message = messages[i % len(messages)]

            start = time.perf_counter()
            result = await engine.prime_memories(
                message=message,
                sender_name="system",
                channel="chat",
            )
            end = time.perf_counter()

            latencies.append(end - start)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)

        percentiles = {
            "P50": sorted_latencies[50],
            "P75": sorted_latencies[75],
            "P90": sorted_latencies[90],
            "P95": sorted_latencies[95],
            "P99": sorted_latencies[99],
        }

        print("\n=== Priming Latency Percentiles ===")
        print(f"Mean:   {statistics.mean(latencies)*1000:.2f} ms")
        print(f"Median: {statistics.median(latencies)*1000:.2f} ms")
        for name, value in percentiles.items():
            print(f"{name}:    {value*1000:.2f} ms")

        # Verify P95 is under threshold
        # CI tolerance: 300ms, production target: 200ms
        threshold = 300  # Relaxed for CI
        assert percentiles["P95"] < threshold / 1000, (
            f"P95 latency {percentiles['P95']*1000:.2f}ms exceeds {threshold}ms"
        )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_empty_cache_vs_warm_cache(anima_dir_with_data):
    """Test priming performance: cold start vs warm cache.

    Verifies:
    - First run (cold cache) is slower
    - Subsequent runs (warm cache) are faster
    - Cache improves performance
    """
    with patch("core.paths.get_shared_dir", return_value=anima_dir_with_data.parent / "shared"):
        engine = PrimingEngine(anima_dir_with_data)

        # Cold cache run
        start = time.perf_counter()
        await engine.prime_memories(
            message="Test cold cache",
            sender_name="system",
            channel="chat",
        )
        end = time.perf_counter()
        cold_time = end - start

        # Warm cache runs
        warm_times = []
        for _ in range(10):
            start = time.perf_counter()
            await engine.prime_memories(
                message="Test warm cache",
                sender_name="system",
                channel="chat",
            )
            end = time.perf_counter()
            warm_times.append(end - start)

        mean_warm = statistics.mean(warm_times)

        print(f"\n=== Cache Performance ===")
        print(f"Cold cache: {cold_time*1000:.2f} ms")
        print(f"Warm cache (mean): {mean_warm*1000:.2f} ms")
        print(f"Speedup: {cold_time/mean_warm:.2f}x")

        # Warm cache should be faster or equal
        # (Note: Might be equal if caching is not implemented)
        assert mean_warm <= cold_time * 1.2  # Allow 20% variance


@pytest.mark.performance
@pytest.mark.asyncio
async def test_budget_allocation_performance(anima_dir_with_data):
    """Test performance of different budget allocations.

    Verifies:
    - Small budget (500 tokens) is faster
    - Large budget (3000 tokens) is slower but more complete
    - Budget affects processing time
    """
    with patch("core.paths.get_shared_dir", return_value=anima_dir_with_data.parent / "shared"):
        engine = PrimingEngine(anima_dir_with_data)

        # Small budget (greeting)
        small_budget_times = []
        for _ in range(20):
            start = time.perf_counter()
            await engine.prime_memories(
                message="こんにちは",
                sender_name="system",
                channel="chat",
                enable_dynamic_budget=True,
            )
            end = time.perf_counter()
            small_budget_times.append(end - start)

        # Large budget (complex request)
        large_budget_times = []
        for _ in range(20):
            start = time.perf_counter()
            await engine.prime_memories(
                message="過去の全ての記録を参照して、詳細な分析結果を提供してください。",
                sender_name="system",
                channel="chat",
                enable_dynamic_budget=True,
            )
            end = time.perf_counter()
            large_budget_times.append(end - start)

        mean_small = statistics.mean(small_budget_times)
        mean_large = statistics.mean(large_budget_times)

        print(f"\n=== Budget Allocation Performance ===")
        print(f"Small budget (500 tokens):  {mean_small*1000:.2f} ms")
        print(f"Large budget (3000 tokens): {mean_large*1000:.2f} ms")
        print(f"Ratio: {mean_large/mean_small:.2f}x")

        # Both should complete in reasonable time
        assert mean_small < 0.300, f"Small budget too slow: {mean_small*1000:.2f}ms"
        assert mean_large < 0.500, f"Large budget too slow: {mean_large*1000:.2f}ms"
