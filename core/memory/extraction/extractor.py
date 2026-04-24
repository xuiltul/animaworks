# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""LLM-based entity and fact extraction pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, get_args

from core.memory.ontology.default import (
    DEFAULT_EDGE_TYPE,
    EDGE_TYPE_DESCRIPTIONS,
    EDGE_TYPES,
    ENTITY_TYPES,
    EntityExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    FactExtractionResult,
)

logger = logging.getLogger(__name__)

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(get_args(ENTITY_TYPES))
_VALID_EDGE_TYPES: frozenset[str] = frozenset(get_args(EDGE_TYPES))
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


# ── FactExtractor ──────────────────────────────────────────


class FactExtractor:
    """LLM-based entity and fact extraction pipeline.

    3-step process following Graphiti architecture:

    1. Create Episode node (caller responsibility)
    2. Extract entities via LLM
    3. Extract facts (relationships) via LLM
    """

    def __init__(
        self,
        model: str,
        *,
        locale: str = "ja",
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._locale = locale
        self._max_retries = max_retries
        self._timeout = timeout

    # ── Public API ─────────────────────────────────────────

    async def extract_entities(
        self,
        content: str,
        previous_entities: list[dict[str, str]] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from text content using LLM.

        Args:
            content: Text to extract from.
            previous_entities: List of ``{"name": ..., "summary": ...}`` for
                dedup hints.

        Returns:
            List of extracted entities. Empty list on failure.
        """
        prompts = self._select_prompts()
        prev_str = json.dumps(previous_entities, ensure_ascii=False) if previous_entities else "[]"
        user_prompt = prompts.ENTITY_USER.format(
            content=content,
            previous_entities=prev_str,
        )

        try:
            raw = await self._call_llm(prompts.ENTITY_SYSTEM, user_prompt)
        except Exception:
            logger.warning("Entity extraction LLM call failed", exc_info=True)
            return []

        result = self._parse_json_response(raw, EntityExtractionResult)
        if not isinstance(result, EntityExtractionResult):
            return []

        entities: list[ExtractedEntity] = []
        for ent in result.entities:
            if not ent.name or not ent.name.strip():
                continue
            if ent.entity_type not in _VALID_ENTITY_TYPES:
                ent = ent.model_copy(update={"entity_type": "Concept"})
            entities.append(ent)

        logger.debug("Extracted %d entities from text", len(entities))
        return entities

    async def extract_facts(
        self,
        content: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedFact]:
        """Extract facts (relationships) between entities using LLM.

        Args:
            content: Original text.
            entities: Previously extracted entities.

        Returns:
            List of extracted facts. Empty list on failure.
        """
        if not entities:
            return []

        prompts = self._select_prompts()
        entities_json = json.dumps(
            [e.model_dump(mode="json") for e in entities],
            ensure_ascii=False,
        )
        edge_types_list = "\n".join(
            f"- `{k}`: {v}" for k, v in EDGE_TYPE_DESCRIPTIONS.items()
        )
        user_prompt = prompts.FACT_USER.format(
            content=content,
            entities_json=entities_json,
            edge_types_list=edge_types_list,
        )

        try:
            raw = await self._call_llm(prompts.FACT_SYSTEM, user_prompt)
        except Exception:
            logger.warning("Fact extraction LLM call failed", exc_info=True)
            return []

        result = self._parse_json_response(raw, FactExtractionResult)
        if not isinstance(result, FactExtractionResult):
            return []

        entity_names = {e.name for e in entities}
        facts: list[ExtractedFact] = []
        for fact in result.facts:
            if fact.source_entity not in entity_names:
                logger.debug("Dropping fact: source %r not in entities", fact.source_entity)
                continue
            if fact.target_entity not in entity_names:
                logger.debug("Dropping fact: target %r not in entities", fact.target_entity)
                continue
            if fact.edge_type not in _VALID_EDGE_TYPES:
                fact = fact.model_copy(update={"edge_type": DEFAULT_EDGE_TYPE})
            facts.append(fact)

        logger.debug("Extracted %d facts from text", len(facts))
        return facts

    # ── LLM call ───────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call LLM with retry logic.

        Uses litellm.acompletion for async calls.  Retries up to
        ``max_retries`` on failure.

        Returns:
            Raw text response.

        Raises:
            Exception: If all retry attempts fail.
        """
        import litellm

        from core.memory._llm_utils import get_llm_kwargs_for_model

        llm_kwargs = get_llm_kwargs_for_model(self._model)
        resolved_model = llm_kwargs.pop("model", self._model)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await litellm.acompletion(
                    model=resolved_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=self._timeout,
                    **llm_kwargs,
                )
                text: str = response.choices[0].message.content or ""
                return text
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise last_exc  # type: ignore[misc]

    # ── JSON parsing ───────────────────────────────────────

    @staticmethod
    def _parse_json_response(text: str, model_cls: type[Any]) -> Any:
        """Parse LLM response text into a Pydantic model.

        Handles JSON wrapped in markdown code fences, raw JSON,
        and partial recovery.

        Args:
            text: Raw LLM output.
            model_cls: Pydantic model class to validate against.

        Returns:
            Parsed model instance, or an empty default on failure.
        """
        if not text or not text.strip():
            return model_cls()

        body = text
        fence_match = _CODE_FENCE_RE.search(text)
        if fence_match:
            body = fence_match.group(1)

        try:
            data = json.loads(body)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            data = json.loads(text)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

        logger.debug("Failed to parse LLM JSON response: %.200s", text)
        return model_cls()

    # ── Prompt selection ───────────────────────────────────

    def _select_prompts(self) -> Any:
        """Return the prompt module for the configured locale."""
        if self._locale == "en":
            from core.memory.extraction.prompts import en as prompts
        else:
            from core.memory.extraction.prompts import ja as prompts
        return prompts
