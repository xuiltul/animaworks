from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-based memory backend following Graphiti architecture.

Supports Episode/Entity/Fact ingestion via LLM extraction pipeline
and hybrid retrieval (BM25 + Vector + BFS + cross-encoder reranking).
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

    Supports ingest_text/ingest_file via LLM extraction pipeline
    and hybrid retrieval (BM25 + Vector + BFS + cross-encoder reranking).
    delete is still a stub (Issue #3).
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
        self._embedding_available: bool | None = None

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

    # ── Embedding ──────────────────────────────────────────────────────────

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the shared singleton model.

        Falls back to empty lists when sentence-transformers is unavailable
        or embedding generation fails for any reason.
        """
        if not texts:
            return []
        if self._embedding_available is False:
            return [[] for _ in texts]
        try:
            from core.memory.rag.singleton import generate_embeddings

            result = await asyncio.to_thread(generate_embeddings, texts)
            self._embedding_available = True
            return result
        except Exception:
            if self._embedding_available is None:
                logger.warning("Embedding generation unavailable, vectors will be empty", exc_info=True)
            self._embedding_available = False
            return [[] for _ in texts]

    # ── Ingest methods ──────────────────────────────────────────────────────

    async def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        """Ingest text: create Episode, extract Entities + Facts via LLM, store in Neo4j.

        The EntityResolver session cache persists across multiple ``ingest_text``
        calls so that entities mentioned in different episodes within the same
        batch are correctly de-duplicated.  Call :meth:`clear_resolver_cache`
        explicitly when a logical batch is complete.
        """
        from core.memory.graph.queries import (
            CHECK_EPISODE_EXISTS,
            CREATE_ENTITY,
            CREATE_EPISODE,
            CREATE_FACT,
            CREATE_MENTION,
        )

        async with self._ingest_semaphore:
            driver = await self._ensure_driver()
            now_str = datetime.now(tz=UTC).isoformat()
            episode_uuid = (metadata or {}).get("episode_uuid") or str(uuid4())

            existing = await driver.execute_query(
                CHECK_EPISODE_EXISTS,
                {"uuid": episode_uuid, "group_id": self._group_id},
            )
            if existing:
                logger.debug("Episode %s already exists, skipping", episode_uuid)
                return 0

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

            # Generate episode embedding for vector search
            ep_embeddings = await self._embed_texts([text[:2000]])
            if ep_embeddings and ep_embeddings[0]:
                try:
                    await driver.execute_write(
                        "MATCH (e:Episode {uuid: $uuid}) SET e.content_embedding = $embedding",
                        {"uuid": episode_uuid, "embedding": ep_embeddings[0]},
                    )
                except Exception:
                    logger.debug("Episode embedding update failed", exc_info=True)

            try:
                extractor = self._get_extractor()
                entities = await extractor.extract_entities(text)
                facts = await extractor.extract_facts(text, entities)
            except Exception:
                logger.warning("Extraction failed, Episode-only fallback", exc_info=True)
                return 1

            # 3. Batch-generate entity embeddings
            valid_entities = [e for e in entities if e.name.strip()]
            entity_texts = [f"{e.name}: {e.summary}" for e in valid_entities]
            entity_embeddings = await self._embed_texts(entity_texts)

            # 4. Resolve + Create Entity nodes
            entity_count = 0
            entity_uuid_map: dict[str, str] = {}
            new_entity_uuids: list[str] = []
            resolver = self._get_resolver()

            for idx, ent in enumerate(valid_entities):
                emb = entity_embeddings[idx] if idx < len(entity_embeddings) else []
                try:
                    resolved = await resolver.resolve(ent, name_embedding=emb)
                    entity_uuid_map[ent.name] = resolved.uuid

                    if resolved.is_new:
                        await driver.execute_write(
                            CREATE_ENTITY,
                            {
                                "uuid": resolved.uuid,
                                "name": resolved.name,
                                "entity_type": ent.entity_type,
                                "summary": resolved.summary,
                                "group_id": self._group_id,
                                "created_at": now_str,
                                "name_embedding": emb,
                            },
                        )
                        new_entity_uuids.append(resolved.uuid)
                        entity_count += 1
                    else:
                        from core.memory.graph.queries import UPDATE_ENTITY_SUMMARY

                        await driver.execute_write(
                            UPDATE_ENTITY_SUMMARY,
                            {
                                "uuid": resolved.uuid,
                                "summary": resolved.summary,
                            },
                        )

                    await driver.execute_write(
                        CREATE_MENTION,
                        {
                            "episode_uuid": episode_uuid,
                            "entity_uuid": resolved.uuid,
                            "uuid": str(uuid4()),
                            "created_at": now_str,
                        },
                    )
                except Exception:
                    logger.warning("Entity resolution/creation failed: %s", ent.name, exc_info=True)

            # 5. Batch-generate fact embeddings
            valid_facts = [
                f for f in facts if entity_uuid_map.get(f.source_entity) and entity_uuid_map.get(f.target_entity)
            ]
            fact_texts = [f"{f.source_entity} → {f.target_entity}: {f.fact}" for f in valid_facts]
            fact_embeddings = await self._embed_texts(fact_texts)

            fact_count = 0
            for idx, fact in enumerate(valid_facts):
                src_uuid = entity_uuid_map[fact.source_entity]
                tgt_uuid = entity_uuid_map[fact.target_entity]
                f_emb = fact_embeddings[idx] if idx < len(fact_embeddings) else []
                try:
                    fact_uuid = str(uuid4())
                    await driver.execute_write(
                        CREATE_FACT,
                        {
                            "source_uuid": src_uuid,
                            "target_uuid": tgt_uuid,
                            "uuid": fact_uuid,
                            "fact": fact.fact,
                            "fact_embedding": f_emb,
                            "group_id": self._group_id,
                            "created_at": now_str,
                            "valid_at": fact.valid_at or now_str,
                            "source_episode_uuids": [episode_uuid],
                        },
                    )
                    fact_count += 1

                    try:
                        invalidator = self._get_invalidator()
                        invalidated = await invalidator.find_and_invalidate(
                            new_fact_uuid=fact_uuid,
                            source_entity_uuid=src_uuid,
                            target_entity_uuid=tgt_uuid,
                            new_fact_text=fact.fact,
                            new_valid_at=fact.valid_at or now_str,
                        )
                        if invalidated:
                            logger.info("Invalidated %d facts during ingest", len(invalidated))
                    except Exception:
                        logger.debug("Invalidation check failed", exc_info=True)
                except Exception:
                    logger.warning("Fact creation failed: %s", fact.fact, exc_info=True)

            total = 1 + entity_count + fact_count
            logger.info(
                "Ingested: 1 episode, %d entities, %d facts (group=%s)",
                entity_count,
                fact_count,
                self._group_id,
            )

            # 6. Dynamic community update for new entities
            if new_entity_uuids:
                try:
                    await self._dynamic_community_update(driver, new_entity_uuids)
                except Exception:
                    logger.debug("Dynamic community update failed, continuing", exc_info=True)

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

    def _get_resolver(self):  # noqa: ANN202 – lazy import avoids circular
        """Create or return cached EntityResolver."""
        if not hasattr(self, "_resolver") or self._resolver is None:
            from core.memory.extraction.resolver import EntityResolver

            model = self._resolve_background_model()
            locale = self._resolve_locale()
            self._resolver = EntityResolver(
                self._driver,
                self._group_id,
                model=model,
                locale=locale,
            )
        return self._resolver

    def clear_resolver_cache(self) -> None:
        """Clear the session cache after an ingest batch."""
        if hasattr(self, "_resolver") and self._resolver is not None:
            self._resolver.clear_cache()

    def _get_invalidator(self):  # noqa: ANN202 – lazy import avoids circular
        """Create or return cached EdgeInvalidator."""
        if not hasattr(self, "_invalidator") or self._invalidator is None:
            from core.memory.extraction.invalidator import EdgeInvalidator

            model = self._resolve_background_model()
            locale = self._resolve_locale()
            self._invalidator = EdgeInvalidator(
                self._driver,
                self._group_id,
                model=model,
                locale=locale,
            )
        return self._invalidator

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

    async def _dynamic_community_update(
        self,
        driver: Neo4jDriver,
        new_entity_uuids: list[str],
    ) -> None:
        """Assign newly created entities to existing communities via majority vote."""
        try:
            from core.memory.graph.community import CommunityDetector
            from core.memory.graph.queries import FIND_ENTITY_NEIGHBORS

            detector = CommunityDetector(
                driver,
                self._group_id,
                model=self._resolve_background_model(),
                locale=self._resolve_locale(),
            )

            for entity_uuid in new_entity_uuids:
                try:
                    rows = await driver.execute_query(
                        FIND_ENTITY_NEIGHBORS,
                        {"entity_uuid": entity_uuid, "group_id": self._group_id, "limit": 20},
                    )
                    neighbor_uuids = [r["uuid"] for r in rows if r.get("uuid")]
                    if neighbor_uuids:
                        assigned = await detector.dynamic_update(entity_uuid, neighbor_uuids)
                        if assigned:
                            logger.debug("Entity %s assigned to community %s", entity_uuid, assigned)
                except Exception:
                    logger.debug("Dynamic community update failed for %s", entity_uuid, exc_info=True)
        except Exception:
            logger.debug("Community module unavailable, skipping dynamic update", exc_info=True)

    async def retrieve(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve memories using hybrid search (BM25 + Vector + BFS + reranker)."""
        if scope == "community":
            return await self._retrieve_communities(query, limit, min_score)

        driver = await self._ensure_driver()

        query_embeddings = await self._embed_texts([query])
        query_embedding = query_embeddings[0] if query_embeddings else []

        from core.memory.graph.search import HybridSearch

        search = HybridSearch(driver, self._group_id)

        try:
            results = await search.search(
                query,
                scope=scope,
                limit=limit,
                query_embedding=query_embedding,
            )
        except ValueError:
            return []
        except Exception:
            logger.warning("Hybrid search failed", exc_info=True)
            return []

        memories: list[RetrievedMemory] = []
        for r in results:
            score = float(r.get("ce_score", r.get("rrf_score", r.get("score", 0.0))))
            if score < min_score:
                continue

            if scope == "entity":
                content = f"{r.get('name', '')}: {r.get('summary', '')}"
                source = f"entity:{r.get('uuid', '')}"
            elif scope == "episode":
                content = r.get("content", "")
                source = f"episode:{r.get('uuid', '')}"
            elif scope == "community":
                content = f"[{r.get('name', '')}] {r.get('summary', '')}"
                source = f"community:{r.get('uuid', '')}"
            else:
                content = f"{r.get('source_name', '')} → {r.get('target_name', '')}: {r.get('fact', '')}"
                source = f"fact:{r.get('uuid', '')}"

            memories.append(
                RetrievedMemory(
                    content=content,
                    score=score,
                    source=source,
                    metadata={k: v for k, v in r.items() if isinstance(v, (str, int, float, bool))},
                    trust="medium",
                )
            )

        return memories

    async def _retrieve_communities(self, query: str, limit: int, min_score: float) -> list[RetrievedMemory]:
        """Retrieve communities by simple text match."""
        driver = await self._ensure_driver()
        from core.memory.graph.queries import SEARCH_COMMUNITIES

        rows = await driver.execute_query(
            SEARCH_COMMUNITIES,
            {"group_id": self._group_id, "limit": limit},
        )

        memories: list[RetrievedMemory] = []
        for r in rows:
            content = f"[{r.get('name', '')}] {r.get('summary', '')}"
            memories.append(
                RetrievedMemory(
                    content=content,
                    score=1.0,
                    source=f"community:{r.get('uuid', '')}",
                    metadata={k: v for k, v in r.items() if isinstance(v, (str, int, float, bool))},
                    trust="medium",
                )
            )
        return memories

    async def get_community_context(
        self,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedMemory]:
        """Return community summaries from Neo4j."""
        try:
            return await self._retrieve_communities(query, limit, min_score=0.0)
        except Exception:
            logger.debug("get_community_context failed", exc_info=True)
            return []

    async def get_recent_facts(
        self,
        query: str,
        *,
        hours: int = 24,
        limit: int = 10,
    ) -> list[RetrievedMemory]:
        """Return recently valid facts from Neo4j."""
        try:
            driver = await self._ensure_driver()
            from datetime import timedelta

            from core.memory.graph.queries import FIND_RECENT_FACTS

            cutoff = (datetime.now(tz=UTC) - timedelta(hours=hours)).isoformat()
            rows = await driver.execute_query(
                FIND_RECENT_FACTS,
                {"group_id": self._group_id, "since": cutoff, "limit": limit},
            )
            return [
                RetrievedMemory(
                    content=f"{r.get('source_name', '')} → {r.get('target_name', '')}: {r.get('fact', '')}",
                    score=1.0,
                    source=f"fact:{r.get('uuid', '')}",
                    metadata={k: v for k, v in r.items() if isinstance(v, (str, int, float, bool))},
                    trust="medium",
                )
                for r in rows
            ]
        except Exception:
            logger.debug("get_recent_facts failed", exc_info=True)
            return []

    async def delete(self, source: str) -> None:
        """Soft-delete an episode, entity, or fact by prefixed ID.

        Accepted formats:
            ``episode:{uuid}`` — soft-delete episode + remove MENTIONS
            ``entity:{uuid}``  — soft-delete entity + invalidate facts
            ``fact:{uuid}``    — soft-delete fact (RELATES_TO)

        Unprefixed IDs are treated as episode UUIDs for backward compat.
        """
        driver = await self._ensure_driver()
        now_str = datetime.now(tz=UTC).isoformat()

        if ":" in source:
            prefix, uuid = source.split(":", 1)
        else:
            prefix, uuid = "episode", source

        from core.memory.graph.queries import (
            SOFT_DELETE_ENTITY,
            SOFT_DELETE_EPISODE,
            SOFT_DELETE_FACT,
        )

        query_map = {
            "episode": SOFT_DELETE_EPISODE,
            "entity": SOFT_DELETE_ENTITY,
            "fact": SOFT_DELETE_FACT,
        }

        query = query_map.get(prefix)
        if not query:
            logger.warning("Unknown delete prefix: %s", prefix)
            return

        try:
            await driver.execute_write(
                query,
                {"uuid": uuid, "group_id": self._group_id, "deleted_at": now_str},
            )
            logger.info("Soft-deleted %s:%s (group=%s)", prefix, uuid, self._group_id)
        except Exception:
            logger.warning("Failed to delete %s:%s", prefix, uuid, exc_info=True)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Shut down the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
