from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy atomic-fact extraction helpers."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory.fact_config import _resolve_extraction_config
from core.memory.fact_invalidation import ReconcileAction, ReconcileResult, reconcile_new_fact
from core.memory.fact_observability import warn_rate_limited
from core.memory.facts import FactRecord, append_fact_records, fact_file_for_record
from core.memory.ontology.default import ExtractedEntity, ExtractedFact
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.memory.fact_extraction")


@dataclass(frozen=True)
class FactExtractionOutcome:
    """Fact extraction/storage outcome with operational failure counters."""

    records: list[FactRecord]
    failed: bool = False
    failure_stage: str = ""
    failure_reason: str = ""

    @property
    def facts_extracted(self) -> int:
        return len(self.records)

    @property
    def facts_failed(self) -> int:
        return int(self.failed)


def _facts_extraction_enabled() -> bool:
    try:
        from core.config import load_config

        return bool(getattr(load_config().rag, "facts_extraction_enabled", True))
    except Exception:
        logger.debug("Failed to load facts_extraction_enabled; defaulting to enabled", exc_info=True)
        return True


def _entity_registry_enabled() -> bool:
    try:
        from core.config import load_config

        return bool(getattr(load_config().rag, "entity_registry_enabled", True))
    except Exception:
        logger.debug("Failed to load entity_registry_enabled; defaulting to enabled", exc_info=True)
        return True


def _facts_reconcile_enabled() -> bool:
    try:
        from core.config import load_config

        return bool(getattr(load_config().rag, "facts_reconcile_enabled", True))
    except Exception:
        logger.debug("Failed to load facts_reconcile_enabled; defaulting to enabled", exc_info=True)
        return True


def format_turns_for_fact_extraction(turns: list[Any]) -> str:
    lines: list[str] = []
    for turn in turns:
        role = getattr(turn, "role", "")
        content = getattr(turn, "content", "")
        timestamp = getattr(turn, "timestamp", "")
        if content:
            prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{prefix}{role}: {content}")
    return "\n".join(lines)


def records_from_extraction(
    entities: list[ExtractedEntity],
    facts: list[ExtractedFact],
    *,
    source_episode: str,
    source_session_id: str = "",
    recorded_at: str | None = None,
    confidence: float = 0.85,
) -> list[FactRecord]:
    entity_names = [entity.name for entity in entities if entity.name.strip()]
    out: list[FactRecord] = []
    recorded = recorded_at or now_iso()
    for fact in facts:
        out.append(
            FactRecord(
                text=fact.fact,
                source_entity=fact.source_entity,
                target_entity=fact.target_entity,
                edge_type=fact.edge_type,
                raw_edge_type=fact.raw_edge_type or "",
                valid_at=fact.valid_at or "",
                recorded_at=recorded,
                entities=entity_names,
                source_episode=source_episode,
                source_session_id=source_session_id,
                confidence=confidence,
            )
        )
    return out


async def extract_fact_records(
    anima_dir: Path,
    text: str,
    *,
    source_episode: str,
    source_session_id: str = "",
    reference_time: str | None = None,
    extractor: Any | None = None,
    model: str | None = None,
    locale: str | None = None,
    llm_extra: dict[str, object] | None = None,
    enabled: bool | None = None,
) -> list[FactRecord]:
    return (
        await extract_fact_records_with_outcome(
            anima_dir,
            text,
            source_episode=source_episode,
            source_session_id=source_session_id,
            reference_time=reference_time,
            extractor=extractor,
            model=model,
            locale=locale,
            llm_extra=llm_extra,
            enabled=enabled,
        )
    ).records


async def extract_fact_records_with_outcome(
    anima_dir: Path,
    text: str,
    *,
    source_episode: str,
    source_session_id: str = "",
    reference_time: str | None = None,
    extractor: Any | None = None,
    model: str | None = None,
    locale: str | None = None,
    llm_extra: dict[str, object] | None = None,
    enabled: bool | None = None,
) -> FactExtractionOutcome:
    if not text.strip():
        return FactExtractionOutcome([])
    if enabled is False or (enabled is None and not _facts_extraction_enabled()):
        return FactExtractionOutcome([])

    try:
        if extractor is None:
            from core.memory.extraction.extractor import FactExtractor

            resolved_model, resolved_extra, resolved_locale, timeout, credential = _resolve_extraction_config(anima_dir)
            extractor = FactExtractor(
                model=model or resolved_model,
                locale=locale or resolved_locale,
                timeout=timeout,
                llm_extra=llm_extra or resolved_extra,
                anima_dir=anima_dir,
                credential="" if model else credential,
            )
        entities = await extractor.extract_entities(text)
        failure_stage = str(getattr(extractor, "last_failure_stage", "") or "")
        failure_reason = str(getattr(extractor, "last_failure_reason", "") or "")
        if failure_stage:
            return FactExtractionOutcome([], True, failure_stage, failure_reason)
        resolved_reference_time = reference_time or now_iso()
        facts = await extractor.extract_facts(
            text,
            entities,
            reference_time=resolved_reference_time,
        )
        failure_stage = str(getattr(extractor, "last_failure_stage", "") or "")
        failure_reason = str(getattr(extractor, "last_failure_reason", "") or "")
        records = records_from_extraction(
            entities,
            facts,
            source_episode=source_episode,
            source_session_id=source_session_id,
            recorded_at=resolved_reference_time,
        )
        return FactExtractionOutcome(records, bool(failure_stage), failure_stage, failure_reason)
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.extract",
            "Atomic fact extraction failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return FactExtractionOutcome([], True, "extract", f"{type(exc).__name__}: {exc}")


async def extract_and_store_facts(
    anima_dir: Path,
    text: str,
    *,
    source_episode: str,
    source_session_id: str = "",
    reference_time: str | None = None,
    origin: str = "conversation",
    extractor: Any | None = None,
    enabled: bool | None = None,
) -> list[FactRecord]:
    """Extract facts, append new records, and update the facts RAG index."""
    records = await extract_fact_records(
        anima_dir,
        text,
        source_episode=source_episode,
        source_session_id=source_session_id,
        reference_time=reference_time,
        extractor=extractor,
        enabled=enabled,
    )
    return (
        await _store_fact_records(
            anima_dir,
            records,
            reference_time=reference_time,
            origin=origin,
        )
    ).records


async def extract_and_store_facts_with_outcome(
    anima_dir: Path,
    text: str,
    *,
    source_episode: str,
    source_session_id: str = "",
    reference_time: str | None = None,
    origin: str = "conversation",
    extractor: Any | None = None,
    enabled: bool | None = None,
) -> FactExtractionOutcome:
    """Extract/store facts and return counters for lifecycle completion logs."""

    extraction = await extract_fact_records_with_outcome(
        anima_dir,
        text,
        source_episode=source_episode,
        source_session_id=source_session_id,
        reference_time=reference_time,
        extractor=extractor,
        enabled=enabled,
    )
    if not extraction.records:
        return extraction
    return await _store_fact_records(
        anima_dir,
        extraction.records,
        reference_time=reference_time,
        origin=origin,
        initial_failed=extraction.failed,
        initial_stage=extraction.failure_stage,
        initial_reason=extraction.failure_reason,
    )


async def _store_fact_records(
    anima_dir: Path,
    records: list[FactRecord],
    *,
    reference_time: str | None,
    origin: str,
    initial_failed: bool = False,
    initial_stage: str = "",
    initial_reason: str = "",
) -> FactExtractionOutcome:
    if not records:
        return FactExtractionOutcome(records, initial_failed, initial_stage, initial_reason)

    records_to_append, reconciled_stored, affected_paths, updated_records = await asyncio.to_thread(
        _reconcile_extracted_facts,
        anima_dir,
        records,
        as_of_time=reference_time,
    )
    if not records_to_append and not affected_paths:
        return FactExtractionOutcome(reconciled_stored, initial_failed, initial_stage, initial_reason)

    try:
        stored = [*reconciled_stored, *append_fact_records(anima_dir, records_to_append)]
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.append",
            "Failed to append atomic facts",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return FactExtractionOutcome([], True, "append", f"{type(exc).__name__}: {exc}")

    side_effect_failed = initial_failed
    failure_stage = initial_stage
    failure_reason = initial_reason
    if stored or affected_paths:
        entity_registry_enabled = _entity_registry_enabled()
        entity_registry = None
        entity_keys = None
        registry_records = [*stored, *[record for record in updated_records if record.is_active()]]
        if entity_registry_enabled and registry_records:
            entity_registry = _upsert_fact_entities(anima_dir, registry_records)
            entity_keys = _entity_keys_for_records(entity_registry, registry_records) if entity_registry else None
            if entity_registry is None:
                side_effect_failed = True
                failure_stage = failure_stage or "entity_registry"
                failure_reason = failure_reason or "entity_registry_update_failed"
        index_ok = _index_fact_records(
            anima_dir,
            stored,
            origin=origin,
            sync_entities=entity_registry_enabled,
            entity_registry=entity_registry,
            entity_keys=entity_keys,
            extra_paths=affected_paths,
        )
        if index_ok is False:
            side_effect_failed = True
            failure_stage = failure_stage or "index"
            failure_reason = failure_reason or "fact_index_update_failed"
    return FactExtractionOutcome(stored, side_effect_failed, failure_stage, failure_reason)


def _reconcile_extracted_facts(
    anima_dir: Path,
    records: list[FactRecord],
    *,
    as_of_time: str | None,
) -> tuple[list[FactRecord], list[FactRecord], set[Path], list[FactRecord]]:
    if not _facts_reconcile_enabled():
        return list(records), [], set(), []

    to_append: list[FactRecord] = []
    stored: list[FactRecord] = []
    affected_paths: set[Path] = set()
    updated_records: list[FactRecord] = []

    for record in records:
        try:
            result = reconcile_new_fact(anima_dir, record, as_of_time=as_of_time)
        except Exception as exc:
            warn_rate_limited(
                logger,
                "fact_extraction.reconcile",
                "Fact reconciliation failed; appending extracted fact",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            result = ReconcileResult(
                action=ReconcileAction.ADD,
                fact=record,
                should_append=True,
                reason="reconcile_exception",
            )

        affected_paths.update(result.affected_paths)
        updated_records.extend(result.updated_records)
        stored.extend(result.appended_records)
        if result.should_append:
            to_append.append(record)

    return to_append, stored, affected_paths, updated_records


def _upsert_fact_entities(anima_dir: Path, records: list[FactRecord]) -> dict[str, Any] | None:
    try:
        from core.memory.entity_index import upsert_entities_from_facts

        return upsert_entities_from_facts(anima_dir, records)
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.entity_registry",
            "Failed to update entity registry from facts",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return None


def _entity_keys_for_records(registry: dict[str, Any], records: list[FactRecord]) -> set[str]:
    try:
        from core.memory.entity_index import entity_keys_for_records

        return entity_keys_for_records(registry, records)
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.entity_keys",
            "Failed to resolve entity keys for sync",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return set()


def _index_fact_records(
    anima_dir: Path,
    records: list[FactRecord],
    *,
    origin: str,
    sync_entities: bool = True,
    entity_registry: dict[str, Any] | None = None,
    entity_keys: set[str] | None = None,
    extra_paths: set[Path] | None = None,
) -> bool | None:
    paths = sorted({fact_file_for_record(anima_dir, record) for record in records} | set(extra_paths or set()))
    if not paths:
        return None
    try:
        from core.memory.manager import MemoryManager

        memory = MemoryManager(anima_dir)
        for path in paths:
            memory._rag.index_file(path, "facts", force=True, origin=origin)
        if sync_entities:
            return _sync_entity_collection(anima_dir, memory, registry=entity_registry, entity_keys=entity_keys)
        return True
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.index",
            "Failed to index atomic facts",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


def _sync_entity_collection(
    anima_dir: Path,
    memory: Any,
    *,
    registry: dict[str, Any] | None = None,
    entity_keys: set[str] | None = None,
) -> bool:
    try:
        from core.memory.entity_index import sync_entity_collection

        indexer = memory._rag._get_indexer()
        vector_store = getattr(indexer, "vector_store", None) if indexer is not None else None
        if vector_store is not None:
            sync_entity_collection(anima_dir, registry=registry, entity_keys=entity_keys, vector_store=vector_store)
        return True
    except Exception as exc:
        warn_rate_limited(
            logger,
            "fact_extraction.entity_collection",
            "Failed to sync entity collection",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return True
