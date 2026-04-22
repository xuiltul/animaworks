from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-based memory backend following Graphiti architecture.

Phase 1 skeleton: only health_check, reset, and stats are functional.
ingest_*/retrieve/delete raise NotImplementedError (Issue #3-6).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.memory.backend.base import MemoryBackend, RetrievedMemory

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)

# ── Neo4jGraphBackend ──────────────────────────────────────────────────────


class Neo4jGraphBackend(MemoryBackend):
    """Neo4j-based memory backend following Graphiti architecture.

    Phase 1 skeleton: only health_check, reset, and stats are functional.
    ingest_*/retrieve/delete raise NotImplementedError (Issue #3-6).
    """

    def __init__(
        self,
        anima_dir: Path,
        *,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "animaworks",
        database: str = "neo4j",
        group_id: str | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_dir.name
        self._group_id = group_id or self._anima_name
        self._driver: Neo4jDriver | None = None
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._schema_ensured = False

    # ── Driver lifecycle ───────────────────────────────────────────────────

    async def _ensure_driver(self) -> Neo4jDriver:
        """Lazy-init and return the driver, ensuring schema exists."""
        if self._driver is None:
            from core.memory.graph.driver import Neo4jDriver

            self._driver = Neo4jDriver(self._uri, self._user, self._password, self._database)
            await self._driver.connect()
        if not self._schema_ensured:
            from core.memory.graph.schema import ensure_schema

            await ensure_schema(self._driver)
            self._schema_ensured = True
        return self._driver

    # ── Implemented methods ────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return ``True`` if Neo4j is reachable."""
        try:
            driver = await self._ensure_driver()
            return await driver.health_check()
        except Exception:
            logger.warning("Neo4j health check failed", exc_info=True)
            return False

    async def reset(self) -> None:
        """Drop all stored data for this Anima's group_id."""
        driver = await self._ensure_driver()
        from core.memory.graph.queries import DELETE_ALL_BY_GROUP

        await driver.execute_write(DELETE_ALL_BY_GROUP, {"group_id": self._group_id})
        logger.info("Reset Neo4j data for group_id=%s", self._group_id)

    async def stats(self) -> dict[str, int | float]:
        """Return node/edge counts for this Anima's group_id."""
        try:
            driver = await self._ensure_driver()
            from core.memory.graph.queries import (
                COUNT_EDGES_BY_GROUP,
                COUNT_NODES_BY_GROUP,
            )

            node_rows = await driver.execute_query(COUNT_NODES_BY_GROUP, {"group_id": self._group_id})
            edge_rows = await driver.execute_query(COUNT_EDGES_BY_GROUP, {"group_id": self._group_id})
            result: dict[str, int | float] = {
                "total_chunks": 0,
                "total_sources": 0,
            }
            for row in node_rows:
                label = row.get("label", "unknown")
                cnt = row.get("cnt", 0)
                result[f"nodes_{label}"] = cnt
                result["total_chunks"] += cnt  # type: ignore[operator]
            for row in edge_rows:
                rel_type = row.get("rel_type", "unknown")
                result[f"edges_{rel_type}"] = row.get("cnt", 0)
            return result
        except Exception:
            logger.warning("Neo4j stats failed", exc_info=True)
            return {"total_chunks": 0, "total_sources": 0}

    # ── Stub methods (NotImplementedError) ─────────────────────────────────

    async def ingest_file(self, path: Path) -> int:
        """Raise NotImplementedError until Issue #3."""
        raise NotImplementedError("Neo4j ingest_file will be implemented in Issue #3")

    async def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        """Raise NotImplementedError until Issue #3."""
        raise NotImplementedError("Neo4j ingest_text will be implemented in Issue #3")

    async def retrieve(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Raise NotImplementedError until Issue #6."""
        raise NotImplementedError("Neo4j retrieve will be implemented in Issue #6")

    async def delete(self, source: str) -> None:
        """Raise NotImplementedError until Issue #3."""
        raise NotImplementedError("Neo4j delete will be implemented in Issue #3")

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Shut down the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
