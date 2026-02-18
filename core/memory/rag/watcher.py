from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""File watcher for automatic memory indexing.

Monitors memory directories for file changes and triggers incremental indexing.

Based on: docs/design/priming-layer-design.md Phase 3
"""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("animaworks.rag.watcher")

# ── Configuration ───────────────────────────────────────────────────

# Debounce delay (ms) - wait for file operations to settle
DEBOUNCE_DELAY_MS = 500

# Batch size for indexing
BATCH_SIZE = 10


# ── FileWatcher ─────────────────────────────────────────────────────


class MemoryFileHandler(FileSystemEventHandler):
    """Handler for memory file system events."""

    def __init__(self, watcher: FileWatcher) -> None:
        """Initialize handler.

        Args:
            watcher: Parent FileWatcher instance
        """
        super().__init__()
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File created: %s", event.src_path)
            self.watcher.queue_file(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File modified: %s", event.src_path)
            self.watcher.queue_file(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File deleted: %s", event.src_path)
            # Note: For deletion, we'd need to remove from vector store
            # This is a Phase 4 feature
            pass


class FileWatcher:
    """File system watcher for automatic memory indexing.

    Monitors memory directories and triggers incremental indexing
    with debouncing to avoid redundant operations.
    """

    def __init__(
        self,
        anima_dir: Path,
        indexer,  # MemoryIndexer instance
        knowledge_graph=None,  # KnowledgeGraph instance (optional)
        anima_name: str | None = None,
        *,
        extra_watch_dirs: list[tuple[Path, str]] | None = None,
    ) -> None:
        """Initialize file watcher.

        Args:
            anima_dir: Path to anima's memory directory
            indexer: MemoryIndexer instance for indexing operations
            knowledge_graph: Optional KnowledgeGraph for incremental updates
            anima_name: Anima name (required if knowledge_graph is provided)
            extra_watch_dirs: Additional (directory, memory_type) pairs to watch.
                Used for common_knowledge/ watching with a shared indexer.
        """
        self.anima_dir = anima_dir
        self.indexer = indexer
        self.knowledge_graph = knowledge_graph
        self.anima_name = anima_name
        self._extra_watch_dirs = extra_watch_dirs or []
        self.observer: Observer | None = None
        self._running = False

        # Debounce queue: file_path -> last_modified_time
        self._queue: dict[Path, float] = {}
        self._queue_lock = asyncio.Lock()

        # Background task for processing queue
        self._processor_task: asyncio.Task | None = None

    # ── Start/Stop ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start file watcher."""
        if self._running:
            logger.warning("FileWatcher already running")
            return

        logger.info("Starting FileWatcher for %s", self.anima_dir)

        # Create observer
        self.observer = Observer()
        handler = MemoryFileHandler(self)

        # Watch memory directories
        watch_dirs = [
            self.anima_dir / "knowledge",
            self.anima_dir / "episodes",
            self.anima_dir / "procedures",
            self.anima_dir / "skills",
        ]

        # Add extra watch dirs (e.g., common_knowledge/)
        for extra_dir, _memory_type in self._extra_watch_dirs:
            watch_dirs.append(extra_dir)

        for watch_dir in watch_dirs:
            if watch_dir.is_dir():
                self.observer.schedule(handler, str(watch_dir), recursive=False)
                logger.debug("Watching directory: %s", watch_dir)

        # Start observer
        self.observer.start()
        self._running = True

        # Start background processor
        try:
            loop = asyncio.get_running_loop()
            self._processor_task = loop.create_task(self._process_queue_loop())
        except RuntimeError:
            # No running event loop - skip background processing
            logger.warning("No running event loop, background processing disabled")

        logger.info("FileWatcher started")

    def stop(self) -> None:
        """Stop file watcher."""
        if not self._running:
            return

        logger.info("Stopping FileWatcher")

        # Stop observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.observer = None

        # Stop background processor
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()

        self._running = False
        logger.info("FileWatcher stopped")

    # ── Queue management ────────────────────────────────────────────

    def queue_file(self, file_path: Path) -> None:
        """Add file to indexing queue.

        Args:
            file_path: Path to file to index
        """
        # Update queue with current timestamp
        self._queue[file_path] = time.time()
        logger.debug("Queued file: %s (queue size: %d)", file_path, len(self._queue))

    async def _process_queue_loop(self) -> None:
        """Background loop to process indexing queue with debouncing."""
        logger.debug("Queue processor started")

        while self._running:
            try:
                await asyncio.sleep(DEBOUNCE_DELAY_MS / 1000.0)

                # Get files ready for processing (debounce period elapsed)
                now = time.time()
                ready_files: list[tuple[Path, str]] = []

                async with self._queue_lock:
                    for file_path, queued_time in list(self._queue.items()):
                        # Check if debounce period has elapsed
                        if (now - queued_time) >= (DEBOUNCE_DELAY_MS / 1000.0):
                            # Determine memory type from directory name
                            memory_type = self._get_memory_type(file_path)
                            if memory_type:
                                ready_files.append((file_path, memory_type))
                                del self._queue[file_path]

                # Process ready files
                if ready_files:
                    logger.debug("Processing %d queued files", len(ready_files))
                    await self._process_batch(ready_files)

            except asyncio.CancelledError:
                logger.debug("Queue processor cancelled")
                break
            except Exception as e:
                logger.error("Queue processor error: %s", e)

        logger.debug("Queue processor stopped")

    async def _process_batch(self, files: list[tuple[Path, str]]) -> None:
        """Process a batch of files for indexing and graph update.

        Args:
            files: List of (file_path, memory_type) tuples
        """
        knowledge_changed_files: list[Path] = []

        # Index files sequentially (could be parallelized in future)
        for file_path, memory_type in files:
            if not file_path.exists():
                logger.debug("File no longer exists, skipping: %s", file_path)
                continue

            try:
                # Run indexing in executor to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self.indexer.index_file,
                    file_path,
                    memory_type,
                    False,  # force=False (incremental)
                )
                logger.debug("Indexed file: %s (type=%s)", file_path, memory_type)

                # Track changed knowledge files for graph update
                if memory_type == "knowledge":
                    knowledge_changed_files.append(file_path)

            except Exception as e:
                logger.error("Failed to index file %s: %s", file_path, e)

        # Update knowledge graph incrementally if available
        if knowledge_changed_files and self.knowledge_graph and self.anima_name:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self.knowledge_graph.update_graph_incremental,
                    knowledge_changed_files,
                    self.anima_name,
                )
                # Save updated graph
                cache_dir = self.anima_dir / "vectordb"
                await loop.run_in_executor(
                    None,
                    self.knowledge_graph.save_graph,
                    cache_dir,
                )
                logger.debug(
                    "Graph incrementally updated and saved for %d knowledge files",
                    len(knowledge_changed_files),
                )
            except Exception as e:
                logger.error("Failed to update knowledge graph: %s", e)

    def _get_memory_type(self, file_path: Path) -> str | None:
        """Determine memory type from file path.

        Args:
            file_path: Path to memory file

        Returns:
            Memory type or None if not recognized
        """
        # Check extra watch dirs first (e.g., common_knowledge/)
        for extra_dir, memory_type in self._extra_watch_dirs:
            try:
                file_path.relative_to(extra_dir)
                return memory_type
            except ValueError:
                continue

        try:
            rel_path = file_path.relative_to(self.anima_dir)
            parent_dir = rel_path.parts[0]

            # Map directory name to memory type
            memory_type_map = {
                "knowledge": "knowledge",
                "episodes": "episodes",
                "procedures": "procedures",
                "skills": "skills",
            }

            return memory_type_map.get(parent_dir)

        except ValueError:
            # File is outside anima_dir
            logger.warning("File outside anima directory: %s", file_path)
            return None


# ── Public API ──────────────────────────────────────────────────────


def create_file_watcher(
    anima_dir: Path,
    indexer,
    knowledge_graph=None,
    anima_name: str | None = None,
    *,
    extra_watch_dirs: list[tuple[Path, str]] | None = None,
) -> FileWatcher:
    """Create a file watcher for an anima's memory directory.

    Args:
        anima_dir: Path to anima directory
        indexer: MemoryIndexer instance
        knowledge_graph: Optional KnowledgeGraph for incremental updates
        anima_name: Anima name (required if knowledge_graph is provided)
        extra_watch_dirs: Additional (directory, memory_type) pairs to watch.

    Returns:
        FileWatcher instance (not started)
    """
    return FileWatcher(
        anima_dir,
        indexer,
        knowledge_graph,
        anima_name,
        extra_watch_dirs=extra_watch_dirs,
    )
