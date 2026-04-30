# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""Entity Resolution: Vector + MinHash + LLM 3-step deduplication."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

from core.memory.ontology.default import ExtractedEntity

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    uuid: str
    name: str
    summary: str
    entity_type: str
    is_new: bool
    merged_with_uuid: str | None = None


# ── EntityResolver ─────────────────────────────────────────


class EntityResolver:
    """3-step entity resolver: Vector -> MinHash -> LLM."""

    def __init__(
        self,
        driver: Neo4jDriver,
        group_id: str,
        *,
        model: str = "claude-sonnet-4-6",
        locale: str = "ja",
        vector_top_k: int = 10,
        vector_min_score: float = 0.5,
        jaccard_threshold: float = 0.4,
        llm_extra: dict[str, object] | None = None,
    ) -> None:
        self._driver = driver
        self._group_id = group_id
        self._model = model
        self._locale = locale
        self._vector_top_k = vector_top_k
        self._vector_min_score = vector_min_score
        self._jaccard_threshold = jaccard_threshold
        self._llm_extra = llm_extra or {}
        self._session_cache: dict[str, ResolvedEntity] = {}

    def clear_cache(self) -> None:
        """Clear the session-level resolution cache."""
        self._session_cache.clear()

    async def resolve(
        self,
        entity: ExtractedEntity,
        *,
        name_embedding: list[float] | None = None,
    ) -> ResolvedEntity:
        """Resolve an entity against existing graph nodes.

        Args:
            entity: Newly extracted entity to resolve.
            name_embedding: Pre-computed embedding for vector search.

        Returns:
            ResolvedEntity with is_new=True for new entities,
            is_new=False with merged_with_uuid for duplicates.
        """
        cache_key = f"{entity.name.lower().strip()}::{entity.entity_type}"
        if cache_key in self._session_cache:
            logger.debug("Session cache hit: %s", cache_key)
            return self._session_cache[cache_key]

        from uuid import uuid4

        new_uuid = str(uuid4())

        # Step 1: Vector candidate search
        candidates = await self._find_vector_candidates(entity, name_embedding)
        if not candidates:
            result = ResolvedEntity(
                uuid=new_uuid,
                name=entity.name,
                summary=entity.summary,
                entity_type=entity.entity_type,
                is_new=True,
            )
            self._session_cache[cache_key] = result
            return result

        # Step 2: MinHash Jaccard filter
        candidates = self._filter_by_jaccard(entity, candidates)
        if not candidates:
            result = ResolvedEntity(
                uuid=new_uuid,
                name=entity.name,
                summary=entity.summary,
                entity_type=entity.entity_type,
                is_new=True,
            )
            self._session_cache[cache_key] = result
            return result

        # Step 3: LLM judgment
        try:
            llm_result = await self._llm_judge(entity, candidates)
        except Exception:
            logger.warning("LLM dedupe failed, creating new entity", exc_info=True)
            llm_result = None

        if llm_result and llm_result.get("duplicate_of_uuid"):
            existing_uuid = llm_result["duplicate_of_uuid"]
            merged_summary = llm_result.get("merged_summary", entity.summary)
            result = ResolvedEntity(
                uuid=existing_uuid,
                name=entity.name,
                summary=merged_summary,
                entity_type=entity.entity_type,
                is_new=False,
                merged_with_uuid=existing_uuid,
            )
        else:
            result = ResolvedEntity(
                uuid=new_uuid,
                name=entity.name,
                summary=entity.summary,
                entity_type=entity.entity_type,
                is_new=True,
            )

        self._session_cache[cache_key] = result
        return result

    # ── Step 1: Vector search ──────────────────────────────

    async def _find_vector_candidates(
        self,
        entity: ExtractedEntity,
        name_embedding: list[float] | None,
    ) -> list[dict]:
        """Find candidate entities using vector similarity search."""
        if not name_embedding:
            from core.memory.graph.queries import FIND_ENTITIES_BY_NAME

            results = await self._driver.execute_query(
                FIND_ENTITIES_BY_NAME,
                {
                    "group_id": self._group_id,
                    "name_pattern": f"(?i).*{entity.name}.*",
                    "limit": self._vector_top_k,
                },
            )
            return results

        from core.memory.graph.queries import FIND_ENTITIES_BY_VECTOR

        results = await self._driver.execute_query(
            FIND_ENTITIES_BY_VECTOR,
            {
                "group_id": self._group_id,
                "entity_type": entity.entity_type,
                "embedding": name_embedding,
                "top_k": self._vector_top_k,
                "min_score": self._vector_min_score,
            },
        )
        return results

    # ── Step 2: MinHash filter ─────────────────────────────

    def _filter_by_jaccard(self, entity: ExtractedEntity, candidates: list[dict]) -> list[dict]:
        """Filter candidates by MinHash Jaccard similarity."""
        from core.memory.extraction.minhash import text_similarity

        entity_text = f"{entity.name} {entity.summary}"
        filtered = []
        for c in candidates:
            cand_text = f"{c.get('name', '')} {c.get('summary', '')}"
            sim = text_similarity(entity_text, cand_text)
            if sim >= self._jaccard_threshold:
                c["jaccard_score"] = sim
                filtered.append(c)

        filtered.sort(key=lambda x: x.get("jaccard_score", 0), reverse=True)
        return filtered

    # ── Step 3: LLM judgment ───────────────────────────────

    async def _llm_judge(self, entity: ExtractedEntity, candidates: list[dict]) -> dict | None:
        """Ask LLM whether entity duplicates any candidate."""
        import litellm

        prompts = self._select_prompts()
        candidates_json = json.dumps(
            [
                {
                    "uuid": c.get("uuid", ""),
                    "name": c.get("name", ""),
                    "summary": c.get("summary", ""),
                    "entity_type": c.get("entity_type", ""),
                }
                for c in candidates[:5]
            ],
            ensure_ascii=False,
        )
        user_prompt = prompts.DEDUPE_USER.format(
            new_entity_name=entity.name,
            new_entity_type=entity.entity_type,
            new_entity_summary=entity.summary,
            candidates_json=candidates_json,
        )

        extra = dict(self._llm_extra)
        effective_timeout = extra.pop("timeout", 30)
        response = await litellm.acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": prompts.DEDUPE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=effective_timeout,
            **extra,
        )

        text = response.choices[0].message.content or ""
        return self._parse_dedupe_response(text)

    @staticmethod
    def _parse_dedupe_response(text: str) -> dict | None:
        """Parse LLM dedupe response JSON."""
        import re

        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        body = fence.group(1) if fence else text
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _select_prompts(self):
        """Select prompt module by locale."""
        if self._locale == "en":
            from core.memory.extraction.prompts import en as p
        else:
            from core.memory.extraction.prompts import ja as p
        return p
