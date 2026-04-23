# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""Temporal fact invalidation — detect contradictions and set invalid_at."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)


# ── EdgeInvalidator ───────────────────────────────────────


class EdgeInvalidator:
    """Detects contradictions between new and existing facts via LLM.

    When a new fact contradicts an existing one on the same entity pair,
    the older fact's invalid_at is set to the new fact's valid_at.
    """

    def __init__(
        self,
        driver: Neo4jDriver,
        group_id: str,
        *,
        model: str = "claude-sonnet-4-6",
        locale: str = "ja",
    ) -> None:
        self._driver = driver
        self._group_id = group_id
        self._model = model
        self._locale = locale

    async def find_and_invalidate(
        self,
        new_fact_uuid: str,
        source_entity_uuid: str,
        target_entity_uuid: str,
        new_fact_text: str,
        new_valid_at: str,
    ) -> list[str]:
        """Find existing facts that contradict *new_fact* and invalidate them.

        Args:
            new_fact_uuid: UUID of the newly created fact.
            source_entity_uuid: Source entity UUID.
            target_entity_uuid: Target entity UUID.
            new_fact_text: The new fact's natural language description.
            new_valid_at: ISO datetime of when the new fact became valid.

        Returns:
            List of UUIDs of invalidated facts.
        """
        from core.memory.graph.queries import FIND_ACTIVE_FACTS_FOR_PAIR

        candidates = await self._driver.execute_query(
            FIND_ACTIVE_FACTS_FOR_PAIR,
            {
                "source_uuid": source_entity_uuid,
                "target_uuid": target_entity_uuid,
                "new_valid_at": new_valid_at,
                "new_fact_uuid": new_fact_uuid,
            },
        )

        if not candidates:
            candidates = []

        from core.memory.graph.queries import FIND_ACTIVE_FACTS_FOR_PAIR_REVERSE

        reverse_candidates = await self._driver.execute_query(
            FIND_ACTIVE_FACTS_FOR_PAIR_REVERSE,
            {
                "source_uuid": source_entity_uuid,
                "target_uuid": target_entity_uuid,
                "new_valid_at": new_valid_at,
                "new_fact_uuid": new_fact_uuid,
            },
        )
        candidates.extend(reverse_candidates)

        if not candidates:
            return []

        try:
            contradicted_uuids = await self._judge_contradictions(new_fact_text, candidates)
        except Exception:
            logger.warning("LLM contradiction check failed, keeping all facts", exc_info=True)
            return []

        if not contradicted_uuids:
            return []

        from core.memory.graph.queries import INVALIDATE_FACT

        invalidated: list[str] = []
        for fact_uuid in contradicted_uuids:
            try:
                await self._driver.execute_write(
                    INVALIDATE_FACT,
                    {"uuid": fact_uuid, "invalid_at": new_valid_at, "group_id": self._group_id},
                )
                invalidated.append(fact_uuid)
                logger.info(
                    "Invalidated fact %s (contradicted by %s)",
                    fact_uuid,
                    new_fact_uuid,
                )
            except Exception:
                logger.warning("Failed to invalidate fact %s", fact_uuid, exc_info=True)

        return invalidated

    async def expire_fact(self, fact_uuid: str, expired_at: str) -> bool:
        """Mark a fact as expired (time-based lifecycle, distinct from contradiction)."""
        from core.memory.graph.queries import EXPIRE_FACT

        try:
            await self._driver.execute_write(
                EXPIRE_FACT,
                {"uuid": fact_uuid, "expired_at": expired_at, "group_id": self._group_id},
            )
            logger.info("Expired fact %s at %s", fact_uuid, expired_at)
            return True
        except Exception:
            logger.warning("Failed to expire fact %s", fact_uuid, exc_info=True)
            return False

    # ── LLM judgment ──────────────────────────────────────

    async def _judge_contradictions(
        self,
        new_fact_text: str,
        candidates: list[dict],
    ) -> list[str]:
        """Ask LLM which candidates contradict the new fact."""
        import litellm

        prompts = self._select_prompts()
        candidates_json = json.dumps(
            [
                {
                    "uuid": c.get("uuid", ""),
                    "fact": c.get("fact", ""),
                    "valid_at": c.get("valid_at", ""),
                }
                for c in candidates
            ],
            ensure_ascii=False,
        )

        user_prompt = prompts.INVALIDATE_USER.format(
            new_fact=new_fact_text,
            existing_facts_json=candidates_json,
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": prompts.INVALIDATE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=30,
        )

        text = response.choices[0].message.content or ""
        return self._parse_invalidation_response(text)

    # ── Response parsing ──────────────────────────────────

    @staticmethod
    def _parse_invalidation_response(text: str) -> list[str]:
        """Parse LLM response to extract list of contradicted fact UUIDs."""
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        body = fence.group(1) if fence else text

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return []

        if isinstance(data, dict):
            uuids = data.get("contradicted_uuids", data.get("contradicted", []))
            if isinstance(uuids, list):
                return [str(u) for u in uuids if u]
        if isinstance(data, list):
            return [str(u) for u in data if u]
        return []

    def _select_prompts(self):
        """Return the prompt module matching the configured locale."""
        if self._locale == "en":
            from core.memory.extraction.prompts import en as p
        else:
            from core.memory.extraction.prompts import ja as p
        return p
