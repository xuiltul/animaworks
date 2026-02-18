from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for file watcher."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from core.memory.rag.watcher import FileWatcher


class MockIndexer:
    """Mock indexer for testing."""

    def __init__(self):
        self.anima_name = "test_anima"
        self.indexed_files = []

    def index_file(self, file_path, memory_type, force):
        """Mock index_file method."""
        self.indexed_files.append((file_path, memory_type))
        return 1  # Return chunk count


@pytest.fixture
def temp_anima_dir():
    """Create temporary anima directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir)

        # Create memory directories
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "episodes").mkdir()
        (anima_dir / "procedures").mkdir()
        (anima_dir / "skills").mkdir()

        yield anima_dir


@pytest.mark.asyncio
async def test_file_watcher_start_stop(temp_anima_dir):
    """Test starting and stopping file watcher."""
    indexer = MockIndexer()
    watcher = FileWatcher(temp_anima_dir, indexer)

    # Start watcher
    watcher.start()
    assert watcher._running is True

    # Wait a bit
    await asyncio.sleep(0.1)

    # Stop watcher
    watcher.stop()
    assert watcher._running is False


@pytest.mark.asyncio
async def test_file_creation_triggers_indexing(temp_anima_dir):
    """Test that file creation triggers indexing."""
    indexer = MockIndexer()
    watcher = FileWatcher(temp_anima_dir, indexer)

    # Start watcher
    watcher.start()

    # Create a file
    test_file = temp_anima_dir / "knowledge" / "test.md"
    test_file.write_text("# Test\n\nContent")

    # Wait for debounce + processing
    await asyncio.sleep(1.0)

    # Stop watcher
    watcher.stop()

    # Check that file was indexed
    assert len(indexer.indexed_files) > 0
    assert any(fp == test_file for fp, _ in indexer.indexed_files)


@pytest.mark.asyncio
async def test_debouncing(temp_anima_dir):
    """Test that rapid file modifications are debounced."""
    indexer = MockIndexer()
    watcher = FileWatcher(temp_anima_dir, indexer)

    # Start watcher
    watcher.start()

    # Create and modify file rapidly
    test_file = temp_anima_dir / "knowledge" / "test.md"
    for i in range(5):
        test_file.write_text(f"# Test\n\nContent {i}")
        await asyncio.sleep(0.05)  # 50ms between writes

    # Wait for debounce + processing
    await asyncio.sleep(1.0)

    # Stop watcher
    watcher.stop()

    # Should only be indexed once due to debouncing
    # (or at most twice if timing is unlucky)
    indexing_count = sum(1 for fp, _ in indexer.indexed_files if fp == test_file)
    assert indexing_count <= 2


def test_queue_file(temp_anima_dir):
    """Test manual file queuing."""
    indexer = MockIndexer()
    watcher = FileWatcher(temp_anima_dir, indexer)

    test_file = temp_anima_dir / "knowledge" / "test.md"
    test_file.write_text("# Test\n\nContent")

    # Queue file manually
    watcher.queue_file(test_file)

    # Check that file is in queue
    assert test_file in watcher._queue


def test_memory_type_detection(temp_anima_dir):
    """Test memory type detection from file path."""
    indexer = MockIndexer()
    watcher = FileWatcher(temp_anima_dir, indexer)

    # Test different memory types
    assert watcher._get_memory_type(temp_anima_dir / "knowledge" / "test.md") == "knowledge"
    assert watcher._get_memory_type(temp_anima_dir / "episodes" / "test.md") == "episodes"
    assert watcher._get_memory_type(temp_anima_dir / "procedures" / "test.md") == "procedures"
    assert watcher._get_memory_type(temp_anima_dir / "skills" / "test.md") == "skills"

    # Test file outside anima_dir
    assert watcher._get_memory_type(Path("/tmp/test.md")) is None
