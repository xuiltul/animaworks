from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Async Neo4j driver wrapper with lazy imports and connection pooling."""

import logging

logger = logging.getLogger(__name__)


# ── Neo4jDriver ──────────


class Neo4jDriver:
    """Manages async Neo4j connection pool.

    The ``neo4j`` package is imported lazily so that the rest of the
    framework can load without it installed.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

    # ── lifecycle ──────────

    async def connect(self) -> None:
        """Create driver and verify connectivity."""
        AsyncGraphDatabase = _import_neo4j()
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j connected: %s (db=%s)", self._uri, self._database)

    async def close(self) -> None:
        """Close the driver connection pool."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    async def health_check(self) -> bool:
        """Return True if the DB is reachable."""
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
        except Exception:
            return False
        return True

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    # ── query execution ──────────

    async def execute_query(
        self,
        query: str,
        parameters: dict | None = None,
        *,
        database: str | None = None,
    ) -> list[dict]:
        """Execute a Cypher query and return results as dicts."""
        self._ensure_connected()
        result = await self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=database or self._database,
        )
        return [record.data() for record in result.records]

    async def execute_write(
        self,
        query: str,
        parameters: dict | None = None,
    ) -> None:
        """Execute a write Cypher query."""
        self._ensure_connected()
        await self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=self._database,
        )

    # ── internal ──────────

    def _ensure_connected(self) -> None:
        if self._driver is None:
            raise RuntimeError("Neo4jDriver is not connected. Call connect() first.")


# ── helpers ──────────


def _import_neo4j():  # noqa: ANN202
    """Lazy-import ``neo4j.AsyncGraphDatabase``."""
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        raise ImportError("Neo4j driver not installed. Install with: pip install animaworks[neo4j]") from None
    return AsyncGraphDatabase
