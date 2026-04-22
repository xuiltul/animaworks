from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-based memory backend following Graphiti architecture.

Supports Episode/Entity/Fact ingestion via LLM extraction pipeline.
retrieve/delete are still stubs (Issue #6).
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from core.memory.backend.base import MemoryBackend, RetrievedMemory

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)

# ── Neo4jGraphBackend ──────────────────────────────────────────────────────


class Neo4jGraphBackend(MemoryBackend):
    """Neo4j-based memory backend with Episode/Entity/Fact ingestion.

    Supports ingest_text/ingest_file via LLM extraction pipeline.
    retrieve/delete are stubs until Issue #6.
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
        self._ingest_semaphore = asyncio.Semaphore(2)
        self._extractor: object | None = None

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

    # ── Ingest methods ──────────────────────────────────────────────────────

    async def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        """Ingest text: create Episode, extract Entities + Facts via LLM, store in Neo4j."""
        from core.memory.graph.queries import (
            CREATE_ENTITY,
            CREATE_EPISODE,
            CREATE_FACT,
            CREATE_MENTION,
        )

        async with self._ingest_semaphore:
            driver = await self._ensure_driver()
            now_str = datetime.now(tz=UTC).isoformat()
            episode_uuid = str(uuid4())

            await driver.execute_write(
                CREATE_EPISODE,
                {
                    "uuid": episode_uuid,
                    "content": text[:10000],
                    "source": source,
                    "source_description": (metadata or {}).get("description", source),
                    "group_id": self._group_id,
                    "created_at": now_str,
                    "valid_at": (metadata or {}).get("valid_at", now_str),
                },
            )

            try:
                extractor = self._get_extractor()
                entities = await extractor.extract_entities(text)
                facts = await extractor.extract_facts(text, entities)
            except Exception:
                logger.warning("Extraction failed, Episode-only fallback", exc_info=True)
                return 1

            entity_count = 0
            entity_uuid_map: dict[str, str] = {}
            for ent in entities:
                if not ent.name.strip():
                    continue
                ent_uuid = str(uuid4())
                entity_uuid_map[ent.name] = ent_uuid
                try:
                    await driver.execute_write(
                        CREATE_ENTITY,
                        {
                            "uuid": ent_uuid,
                            "name": ent.name,
                            "summary": ent.summary,
                            "group_id": self._group_id,
                            "created_at": now_str,
                            "name_embedding": [],
                        },
                    )
                    entity_count += 1
                    await driver.execute_write(
                        CREATE_MENTION,
                        {
                            "episode_uuid": episode_uuid,
                            "entity_uuid": ent_uuid,
                            "uuid": str(uuid4()),
                            "created_at": now_str,
                        },
                    )
                except Exception:
                    logger.warning("Entity creation failed: %s", ent.name, exc_info=True)

            fact_count = 0
            for fact in facts:
                src_uuid = entity_uuid_map.get(fact.source_entity)
                tgt_uuid = entity_uuid_map.get(fact.target_entity)
                if not src_uuid or not tgt_uuid:
                    continue
                try:
                    await driver.execute_write(
                        CREATE_FACT,
                        {
                            "source_uuid": src_uuid,
                            "target_uuid": tgt_uuid,
                            "uuid": str(uuid4()),
                            "fact": fact.fact,
                            "fact_embedding": [],
                            "group_id": self._group_id,
                            "created_at": now_str,
                            "valid_at": fact.valid_at or now_str,
                            "source_episode_uuids": [episode_uuid],
                        },
                    )
                    fact_count += 1
                except Exception:
                    logger.warning("Fact creation failed: %s", fact.fact, exc_info=True)

            total = 1 + entity_count + fact_count
            logger.info(
                "Ingested: 1 episode, %d entities, %d facts (group=%s)",
                entity_count,
                fact_count,
                self._group_id,
            )
            return total

    async def ingest_file(self, path: Path) -> int:
        """Read a file and ingest its content as one or more episodes."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read file: %s", path, exc_info=True)
            return 0

        if not content.strip():
            return 0

        source = str(path.name)
        sections = self._split_sections(content)
        total = 0
        for section in sections:
            if section.strip():
                total += await self.ingest_text(section, source=source)
        return total

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_extractor(self):  # noqa: ANN202 – lazy import avoids circular
        """Create or return cached FactExtractor."""
        if self._extractor is None:
            model = self._resolve_background_model()
            locale = self._resolve_locale()
            from core.memory.extraction.extractor import FactExtractor

            self._extractor = FactExtractor(model=model, locale=locale)
        return self._extractor

    @staticmethod
    def _resolve_background_model() -> str:
        """Resolve the background model for extraction."""
        try:
            from core.config.models import load_config

            cfg = load_config()
            return cfg.anima_defaults.background_model or cfg.anima_defaults.model
        except Exception:
            return "claude-sonnet-4-6"

    @staticmethod
    def _resolve_locale() -> str:
        try:
            from core.config.models import load_config

            return load_config().locale
        except Exception:
            return "ja"

    @staticmethod
    def _split_sections(content: str, max_chars: int = 8000) -> list[str]:
        """Split content into sections by markdown headers or by size."""
        sections = re.split(r"\n(?=##?\s)", content)
        result: list[str] = []
        for section in sections:
            if len(section) <= max_chars:
                result.append(section)
            else:
                for i in range(0, len(section), max_chars):
                    result.append(section[i : i + max_chars])
        return result if result else [content]

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
