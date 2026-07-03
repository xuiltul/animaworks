from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-based memory backend following Graphiti architecture.

Supports Episode/Entity/Fact ingestion via LLM extraction pipeline
and hybrid retrieval (BM25 + Vector + BFS + cross-encoder reranking).
"""

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from core.memory.backend.base import MemoryBackend, RetrievedMemory

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)

# ── Neo4jGraphBackend ──────────────────────────────────────────────────────


class Neo4jGraphBackend(MemoryBackend):
    """Neo4j-based memory backend with Episode/Entity/Fact ingestion.

    Supports ingest_text/ingest_file via LLM extraction pipeline
    and hybrid retrieval (BM25 + Vector + BFS + cross-encoder reranking).
    delete() soft-deletes an episode/entity/fact by prefixed ID.
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

    async def _embed_texts(
        self,
        texts: list[str],
        *,
        purpose: Literal["document", "query"] = "document",
        priority: Literal["interactive", "bulk"] | None = None,
    ) -> list[list[float]]:
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

            resolved_priority = priority or ("interactive" if purpose == "query" else "bulk")
            result = await asyncio.to_thread(
                generate_embeddings,
                texts,
                purpose=purpose,
                priority=resolved_priority,
            )
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
            metadata = metadata or {}
            episode_uuid = self._resolve_episode_uuid(metadata)

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
                    "source_description": metadata.get("description", source),
                    "group_id": self._group_id,
                    "created_at": now_str,
                    "valid_at": metadata.get("valid_at", now_str),
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
                episode_valid_at = metadata.get("valid_at")
                facts = await extractor.extract_facts(text, entities, reference_time=episode_valid_at)
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
                except Exception:
                    logger.warning("Entity resolution failed (resolve): %s", ent.name, exc_info=True)
                    continue

                entity_uuid_map[ent.name] = resolved.uuid

                try:
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
                except Exception:
                    logger.warning(
                        "Entity write failed (create/update): %s",
                        ent.name,
                        exc_info=True,
                    )
                    continue

                try:
                    await driver.execute_write(
                        CREATE_MENTION,
                        {
                            "episode_uuid": episode_uuid,
                            "entity_uuid": resolved.uuid,
                            "uuid": str(uuid4()),
                            "group_id": self._group_id,
                            "created_at": now_str,
                        },
                    )
                except Exception:
                    logger.warning(
                        "Mention creation failed: %s -> %s",
                        episode_uuid[:8],
                        ent.name,
                        exc_info=True,
                    )

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
                    raw_edge_type = getattr(fact, "raw_edge_type", None)
                    if not isinstance(raw_edge_type, str) or not raw_edge_type.strip():
                        raw_edge_type = None
                    await driver.execute_write(
                        CREATE_FACT,
                        {
                            "source_uuid": src_uuid,
                            "target_uuid": tgt_uuid,
                            "uuid": fact_uuid,
                            "fact": fact.fact,
                            "fact_embedding": f_emb,
                            "edge_type": getattr(fact, "edge_type", "RELATES_TO") or "RELATES_TO",
                            "raw_edge_type": raw_edge_type,
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

        normalized_source_path = self._normalize_source_path(path)
        source = f"file:{normalized_source_path}"
        sections = self._split_sections(content)
        total = 0
        for section_index, section in enumerate(sections):
            if section.strip():
                source_hash = hashlib.sha256(section.encode("utf-8")).hexdigest()
                total += await self.ingest_text(
                    section,
                    source=source,
                    metadata={
                        "stable_key": f"file:{normalized_source_path}:{section_index}:{source_hash}",
                        "source_path": normalized_source_path,
                        "source_hash": source_hash,
                        "section_index": section_index,
                    },
                )
        return total

    # ── Helpers ────────────────────────────────────────────────────────────

    def _stable_episode_uuid(self, stable_key: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"animaworks:neo4j:{self._group_id}:{stable_key}"))

    def _resolve_episode_uuid(self, metadata: dict) -> str:
        explicit_uuid = metadata.get("episode_uuid")
        if explicit_uuid:
            return str(explicit_uuid)
        stable_key = metadata.get("stable_key")
        if stable_key:
            return self._stable_episode_uuid(str(stable_key))
        return str(uuid4())

    def _normalize_source_path(self, path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self._anima_dir.resolve()).as_posix()
        except ValueError:
            return resolved_path.as_posix()

    def _get_resolver(self):  # noqa: ANN202 – lazy import avoids circular
        """Create or return cached EntityResolver."""
        if not hasattr(self, "_resolver") or self._resolver is None:
            from core.memory.extraction.resolver import EntityResolver

            model, llm_extra = self._resolve_extraction_config()
            locale = self._resolve_locale()
            self._resolver = EntityResolver(
                self._driver,
                self._group_id,
                model=model,
                locale=locale,
                llm_extra=llm_extra,
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

            model, llm_extra = self._resolve_extraction_config()
            locale = self._resolve_locale()
            self._invalidator = EdgeInvalidator(
                self._driver,
                self._group_id,
                model=model,
                locale=locale,
                llm_extra=llm_extra,
            )
        return self._invalidator

    def _get_extractor(self):  # noqa: ANN202 – lazy import avoids circular
        """Create or return cached FactExtractor."""
        if self._extractor is None:
            model, llm_extra = self._resolve_extraction_config()
            locale = self._resolve_locale()
            from core.memory.extraction.extractor import FactExtractor

            self._extractor = FactExtractor(
                model=model,
                locale=locale,
                llm_extra=llm_extra,
                anima_dir=self._anima_dir,
            )
        return self._extractor

    def _resolve_extraction_config(self) -> tuple[str, dict[str, object]]:
        """Resolve the model and LLM kwargs for extraction.

        Endpoint override fields (api base / api key / extra body) in
        status.json are deliberately NOT trusted, matching the legacy
        hardening in ``core.memory.fact_config``: status.json is writable
        by the anima process itself, so honoring endpoint overrides from it
        would let a tampered file redirect extraction requests and leak
        API keys.

        Returns:
            ``(model_name, llm_extra)`` where *llm_extra* may contain
            ``timeout`` only.

        Resolution order for model:
            1. Per-anima status.json ``extraction_model``
            2. Per-anima status.json ``background_model``
            3. Global config ``anima_defaults.background_model``
            4. Global config ``anima_defaults.model``
            5. Fallback: ``claude-sonnet-4-6``
        """
        import json

        from core.memory.fact_config import _coerce_timeout_seconds

        llm_extra: dict[str, object] = {}
        try:
            status_path = self._anima_dir / "status.json"
            if status_path.is_file():
                data = json.loads(status_path.read_text(encoding="utf-8"))
                if data.get("extraction_timeout"):
                    timeout = _coerce_timeout_seconds(data["extraction_timeout"], 0)
                    if timeout > 0:
                        llm_extra["timeout"] = timeout
                if data.get("extraction_model"):
                    return data["extraction_model"], llm_extra
                if data.get("background_model"):
                    return data["background_model"], llm_extra
        except Exception as e:
            logger.debug("neo4j_graph: failed to read status.json for extraction config: %s", e)
        try:
            from core.config.models import load_config

            cfg = load_config()
            return (cfg.anima_defaults.background_model or cfg.anima_defaults.model), llm_extra
        except Exception:
            return "claude-sonnet-4-6", llm_extra

    def _resolve_background_model(self) -> str:
        """Return the model name used for background / community LLM tasks."""
        model, _ = self._resolve_extraction_config()
        return model

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
                        else:
                            logger.debug(
                                "Entity %s not assigned to an existing community; batch detection remains authoritative",
                                entity_uuid,
                            )
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
        edge_type_filter: str | None = None,
        as_of_time: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve memories using hybrid search (BM25 + Vector + BFS + reranker)."""
        if scope == "community":
            return await self._retrieve_communities(query, limit, min_score)

        driver = await self._ensure_driver()

        query_embeddings = await self._embed_texts([query], purpose="query")
        query_embedding = query_embeddings[0] if query_embeddings else []

        from core.memory.graph.search import HybridSearch

        search = HybridSearch(driver, self._group_id)

        try:
            results = await search.search(
                query,
                scope=scope,
                limit=limit,
                as_of_time=as_of_time,
                query_embedding=query_embedding,
                edge_type_filter=edge_type_filter,
                time_start=time_start,
                time_end=time_end,
            )
        except ValueError:
            return []
        except Exception:
            logger.warning("Hybrid search failed", exc_info=True)
            return []

        memories: list[RetrievedMemory] = []
        for r in results:
            score = self._result_score(r)
            if score < min_score:
                continue

            result_type = r.get("type") or scope

            if result_type == "entity":
                content = f"{r.get('name', '')}: {r.get('summary', '')}"
                source = f"entity:{r.get('uuid', '')}"
            elif result_type == "episode":
                content = r.get("content", "")
                source = f"episode:{r.get('uuid', '')}"
            elif result_type == "community":
                content = f"[{r.get('name', '')}] {r.get('summary', '')}"
                source = f"community:{r.get('uuid', '')}"
            else:
                edge_label = r.get("edge_type", "RELATES_TO")
                content = (
                    f"{r.get('source_name', '')} -[{edge_label}]-> {r.get('target_name', '')}: {r.get('fact', '')}"
                )
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
        """Retrieve communities by query relevance or recency fallback."""
        driver = await self._ensure_driver()
        from core.memory.graph.queries import FULLTEXT_SEARCH_COMMUNITIES, SEARCH_COMMUNITIES

        if query.strip():
            try:
                rows = await driver.execute_query(
                    FULLTEXT_SEARCH_COMMUNITIES,
                    {
                        "query": query,
                        "group_id": self._group_id,
                        "top_k": max(limit * 3, limit, 1),
                        "limit": limit,
                    },
                )
            except Exception:
                logger.debug("Community fulltext search failed", exc_info=True)
                return []
        else:
            rows = await driver.execute_query(
                SEARCH_COMMUNITIES,
                {"group_id": self._group_id, "limit": limit},
            )

        memories: list[RetrievedMemory] = []
        for r in rows:
            score = self._result_score(r) if query.strip() else 1.0
            if score < min_score:
                continue
            content = f"[{r.get('name', '')}] {r.get('summary', '')}"
            memories.append(
                RetrievedMemory(
                    content=content,
                    score=score,
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

            from core.memory.graph.queries import FIND_RECENT_FACTS

            cutoff_dt = datetime.now(tz=UTC) - timedelta(hours=hours)
            cutoff = cutoff_dt.isoformat()

            if query.strip():
                from core.memory.graph.search import HybridSearch

                query_embeddings = await self._embed_texts([query], purpose="query")
                query_embedding = query_embeddings[0] if query_embeddings else []
                rows = await HybridSearch(driver, self._group_id).search(
                    query,
                    scope="fact",
                    limit=max(limit * 3, limit),
                    as_of_time=datetime.now(tz=UTC).isoformat(),
                    query_embedding=query_embedding,
                )
                rows = [r for r in rows if self._row_created_at_is_recent(r, cutoff_dt)][:limit]
            else:
                rows = await driver.execute_query(
                    FIND_RECENT_FACTS,
                    {"group_id": self._group_id, "since": cutoff, "limit": limit},
                )

            memories: list[RetrievedMemory] = []
            for row in rows:
                score = self._result_score(row) if query.strip() else 1.0
                memories.append(self._fact_row_to_memory(row, score=score))
            return memories
        except Exception:
            logger.debug("get_recent_facts failed", exc_info=True)
            return []

    @staticmethod
    def _result_score(row: dict) -> float:
        for key in ("ce_score", "rrf_score", "score"):
            value = row.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _parse_graph_datetime(value: object) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @classmethod
    def _row_created_at_is_recent(cls, row: dict, cutoff: datetime) -> bool:
        created_at = cls._parse_graph_datetime(row.get("created_at"))
        if created_at is None:
            return False
        return created_at >= cutoff

    @staticmethod
    def _fact_row_to_memory(row: dict, *, score: float) -> RetrievedMemory:
        return RetrievedMemory(
            content=(
                f"{row.get('source_name', '')} "
                f"-[{row.get('edge_type', 'RELATES_TO')}]-> "
                f"{row.get('target_name', '')}: {row.get('fact', '')}"
            ),
            score=score,
            source=f"fact:{row.get('uuid', '')}",
            metadata={k: v for k, v in row.items() if isinstance(v, (str, int, float, bool))},
            trust="medium",
        )

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
