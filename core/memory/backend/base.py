from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for memory backends."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

VALID_SCOPES: frozenset[str] = frozenset(
    {
        "knowledge",
        "episodes",
        "procedures",
        "common_knowledge",
        "skills",
        "activity_log",
        "all",
    }
)

# ── Data Models ────────────────────────────────────────────────────────────


@dataclass
class RetrievedMemory:
    """A single memory chunk returned from retrieval."""

    content: str
    score: float
    source: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    trust: str = "medium"


# ── Abstract Base ──────────────────────────────────────────────────────────


class MemoryBackend(ABC):
    """Unified interface for memory storage and retrieval.

    All memory backends (legacy ChromaDB, future Neo4j, etc.) implement
    this interface to provide pluggable storage behind the existing
    MemoryManager / PrimingEngine.
    """

    @abstractmethod
    async def ingest_file(self, path: Path) -> int:
        """Index a file into the memory store.

        Args:
            path: Filesystem path to the file to ingest.

        Returns:
            Number of chunks created from the file.
        """

    @abstractmethod
    async def ingest_text(
        self,
        text: str,
        source: str,
        metadata: dict | None = None,
    ) -> int:
        """Index raw text into the memory store.

        Args:
            text: The text content to ingest.
            source: Logical source identifier (e.g. ``"knowledge/topic.md"``).
            metadata: Optional metadata to attach to created chunks.

        Returns:
            Number of chunks created.
        """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve relevant memories for a query.

        Args:
            query: Natural-language search query.
            scope: One of :data:`VALID_SCOPES` to restrict search.
            limit: Maximum number of results.
            min_score: Minimum relevance score threshold.

        Returns:
            Matching memories sorted by descending relevance.
        """

    @abstractmethod
    async def delete(self, source: str) -> None:
        """Remove all chunks originating from *source*.

        Args:
            source: The source identifier whose chunks should be deleted.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Drop all stored data and reinitialise the backend."""

    @abstractmethod
    async def stats(self) -> dict[str, int | float]:
        """Return backend statistics.

        Returns:
            Dict with at least ``"total_chunks"`` and ``"total_sources"``.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the backend is healthy and reachable."""

    # ── Concrete defaults (Phase 0 compatibility) ──────────────────────────

    async def get_important_chunks(self, limit: int = 20) -> list[RetrievedMemory]:
        """Return high-importance chunks for distillation/priming.

        Default implementation returns an empty list; backends that track
        access frequency or importance scores should override.
        """
        return []

    async def record_access(self, results: list[RetrievedMemory]) -> None:  # noqa: B027
        """Record that *results* were accessed (for forgetting heuristics).

        Default implementation is a no-op.
        """

    async def rebuild_index(self, scope: str | None = None) -> int:
        """Rebuild search indices, optionally scoped.

        Args:
            scope: Restrict rebuild to a single scope, or ``None`` for all.

        Returns:
            Number of chunks re-indexed.  Default returns ``0``.
        """
        return 0

    async def get_community_context(
        self,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedMemory]:
        """Return community summaries relevant to *query*.

        Default returns empty list.  Neo4j backend overrides with
        actual community search.
        """
        return []

    async def get_recent_facts(
        self,
        query: str,
        *,
        hours: int = 24,
        limit: int = 10,
    ) -> list[RetrievedMemory]:
        """Return recently created/valid facts matching *query*.

        Default returns empty list.  Backends with temporal
        awareness should override.
        """
        return []
