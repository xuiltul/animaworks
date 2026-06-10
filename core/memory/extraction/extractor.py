# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""LLM-based entity and fact extraction pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, get_args

from core.memory.fact_observability import warn_rate_limited
from core.memory.ontology.default import (
    ENTITY_TYPES,
    EntityExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    FactExtractionResult,
    allowed_edge_types,
    canonicalize_edge_type,
    format_edge_types_for_prompt,
)
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(get_args(ENTITY_TYPES))
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
        llm_extra: dict[str, object] | None = None,
        anima_dir: Path | None = None,
        credential: str = "",
    ) -> None:
        self._model = model
        self._locale = locale
        self._max_retries = max_retries
        self._timeout = timeout
        self._llm_extra = llm_extra or {}
        self._anima_dir = Path(anima_dir) if anima_dir is not None else None
        self._credential = credential
        self.last_failure_stage = ""
        self.last_failure_reason = ""

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
        self._clear_failure()
        prompts = self._select_prompts()
        prev_str = json.dumps(previous_entities, ensure_ascii=False) if previous_entities else "[]"
        user_prompt = prompts.ENTITY_USER.format(
            content=content,
            previous_entities=prev_str,
        )

        try:
            raw = await self._call_llm(prompts.ENTITY_SYSTEM, user_prompt)
        except Exception as exc:
            self._record_failure(
                "entity_llm",
                f"{type(exc).__name__}: {exc}",
                "Entity extraction LLM call failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return []

        result = self._parse_json_response(raw, EntityExtractionResult, stage="entity")
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
        *,
        reference_time: str | None = None,
    ) -> list[ExtractedFact]:
        """Extract facts (relationships) between entities using LLM.

        Args:
            content: Original text.
            entities: Previously extracted entities.
            reference_time: ISO timestamp for temporal grounding; defaults to now.

        Returns:
            List of extracted facts. Empty list on failure.
        """
        self._clear_failure()
        if not entities:
            return []

        prompts = self._select_prompts()
        entities_json = json.dumps(
            [e.model_dump(mode="json") for e in entities],
            ensure_ascii=False,
        )
        edge_types_list = format_edge_types_for_prompt(self._anima_dir)
        user_prompt = prompts.FACT_USER.format(
            content=content,
            entities_json=entities_json,
            edge_types_list=edge_types_list,
            reference_time=reference_time or now_iso(),
        )

        try:
            raw = await self._call_llm(prompts.FACT_SYSTEM, user_prompt)
        except Exception as exc:
            self._record_failure(
                "fact_llm",
                f"{type(exc).__name__}: {exc}",
                "Fact extraction LLM call failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return []

        result = self._parse_json_response(raw, FactExtractionResult, stage="fact")
        if not isinstance(result, FactExtractionResult):
            return []

        entity_names = {e.name for e in entities}
        allowed = allowed_edge_types(self._anima_dir)
        facts: list[ExtractedFact] = []
        for fact in result.facts:
            if fact.source_entity not in entity_names:
                logger.debug("Dropping fact: source %r not in entities", fact.source_entity)
                continue
            if fact.target_entity not in entity_names:
                logger.debug("Dropping fact: target %r not in entities", fact.target_entity)
                continue
            edge_type, raw_edge_type = canonicalize_edge_type(fact.edge_type, allowed)
            fact = fact.model_copy(update={"edge_type": edge_type, "raw_edge_type": raw_edge_type})
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

        from core.memory._llm_utils import get_memory_llm_kwargs_for_model

        llm_kwargs = get_memory_llm_kwargs_for_model(
            self._model,
            self._llm_extra,
            credential=self._credential,
        )
        resolved_model = llm_kwargs.pop("model", self._model)
        effective_timeout = llm_kwargs.pop("timeout", self._timeout)

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
                    timeout=effective_timeout,
                    **llm_kwargs,
                )
                text: str = response.choices[0].message.content or ""
                return text
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    exc_info=True,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise last_exc  # type: ignore[misc]

    # ── JSON parsing ───────────────────────────────────────

    def _parse_json_response(self, text: str, model_cls: type[Any], *, stage: str) -> Any:
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
            self._record_failure(
                f"{stage}_parse",
                "empty_response",
                "%s extraction returned an empty LLM response",
                stage.capitalize(),
            )
            return model_cls()

        body = text
        fence_match = _CODE_FENCE_RE.search(text)
        if fence_match:
            body = fence_match.group(1)

        first_exc: Exception | None = None
        try:
            data = json.loads(body)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            first_exc = exc

        try:
            data = json.loads(text)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            if first_exc is None:
                first_exc = exc

        self._record_failure(
            f"{stage}_parse",
            f"{type(first_exc).__name__}: {first_exc}" if first_exc else "invalid_json",
            "Failed to parse %s extraction LLM JSON response: %.200s",
            stage,
            text,
            exc_info=(type(first_exc), first_exc, first_exc.__traceback__) if first_exc else False,
        )
        return model_cls()

    def _clear_failure(self) -> None:
        self.last_failure_stage = ""
        self.last_failure_reason = ""

    def _record_failure(
        self,
        stage: str,
        reason: str,
        message: str,
        *args: object,
        exc_info: bool | tuple[type[BaseException], BaseException, Any] = False,
    ) -> None:
        self.last_failure_stage = stage
        self.last_failure_reason = reason
        warn_rate_limited(
            logger,
            f"fact_extraction.{stage}",
            message,
            *args,
            exc_info=exc_info,
        )

    # ── Prompt selection ───────────────────────────────────

    def _select_prompts(self) -> Any:
        """Return the prompt module for the configured locale."""
        if self._locale == "en":
            from core.memory.extraction.prompts import en as prompts
        else:
            from core.memory.extraction.prompts import ja as prompts
        return prompts
