from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy atomic-fact extraction helpers."""

import json
import logging
from pathlib import Path
from typing import Any

from core.memory.facts import FactRecord, append_fact_records, fact_file_for_record
from core.memory.ontology.default import ExtractedEntity, ExtractedFact
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.memory.fact_extraction")


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


def _resolve_extraction_config(anima_dir: Path) -> tuple[str, dict[str, object], str, int]:
    """Resolve extraction model and timeout without trusting endpoint data from status.json."""
    llm_extra: dict[str, object] = {}
    timeout = 30

    try:
        status_path = Path(anima_dir) / "status.json"
        if status_path.is_file():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            if data.get("extraction_timeout"):
                timeout = int(data["extraction_timeout"])
            if data.get("extraction_model"):
                return str(data["extraction_model"]), llm_extra, _resolve_locale(), timeout
            if data.get("background_model"):
                return str(data["background_model"]), llm_extra, _resolve_locale(), timeout
    except Exception:
        logger.debug("Failed to read status.json for fact extraction", exc_info=True)

    try:
        from core.config.models import load_config

        cfg = load_config()
        model = cfg.anima_defaults.background_model or cfg.anima_defaults.model
        return model, llm_extra, cfg.locale, timeout
    except Exception:
        return "claude-sonnet-4-6", llm_extra, "ja", timeout


def _resolve_locale() -> str:
    try:
        from core.config.models import load_config

        return str(load_config().locale or "ja")
    except Exception:
        return "ja"


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
    if not text.strip():
        return []
    if enabled is False or (enabled is None and not _facts_extraction_enabled()):
        return []

    try:
        if extractor is None:
            from core.memory.extraction.extractor import FactExtractor

            resolved_model, resolved_extra, resolved_locale, timeout = _resolve_extraction_config(anima_dir)
            extractor = FactExtractor(
                model=model or resolved_model,
                locale=locale or resolved_locale,
                timeout=timeout,
                llm_extra=llm_extra or resolved_extra,
                anima_dir=anima_dir,
            )
        entities = await extractor.extract_entities(text)
        facts = await extractor.extract_facts(
            text,
            entities,
            reference_time=reference_time or now_iso(),
        )
        return records_from_extraction(
            entities,
            facts,
            source_episode=source_episode,
            source_session_id=source_session_id,
        )
    except Exception:
        logger.warning("Atomic fact extraction failed", exc_info=True)
        return []


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
    if not records:
        return []

    try:
        stored = append_fact_records(anima_dir, records)
    except Exception:
        logger.warning("Failed to append atomic facts", exc_info=True)
        return []

    if stored:
        entity_registry_enabled = _entity_registry_enabled()
        entity_registry = None
        entity_keys = None
        if entity_registry_enabled:
            entity_registry = _upsert_fact_entities(anima_dir, stored)
            entity_keys = _entity_keys_for_records(entity_registry, stored) if entity_registry else None
        _index_fact_records(
            anima_dir,
            stored,
            origin=origin,
            sync_entities=entity_registry_enabled,
            entity_registry=entity_registry,
            entity_keys=entity_keys,
        )
    return stored


def _upsert_fact_entities(anima_dir: Path, records: list[FactRecord]) -> dict[str, Any] | None:
    try:
        from core.memory.entity_index import upsert_entities_from_facts

        return upsert_entities_from_facts(anima_dir, records)
    except Exception:
        logger.debug("Failed to update entity registry from facts", exc_info=True)
        return None


def _entity_keys_for_records(registry: dict[str, Any], records: list[FactRecord]) -> set[str]:
    try:
        from core.memory.entity_index import entity_keys_for_records

        return entity_keys_for_records(registry, records)
    except Exception:
        logger.debug("Failed to resolve entity keys for sync", exc_info=True)
        return set()


def _index_fact_records(
    anima_dir: Path,
    records: list[FactRecord],
    *,
    origin: str,
    sync_entities: bool = True,
    entity_registry: dict[str, Any] | None = None,
    entity_keys: set[str] | None = None,
) -> None:
    paths = sorted({fact_file_for_record(anima_dir, record) for record in records})
    if not paths:
        return
    try:
        from core.memory.manager import MemoryManager

        memory = MemoryManager(anima_dir)
        for path in paths:
            memory._rag.index_file(path, "facts", origin=origin)
        if sync_entities:
            _sync_entity_collection(anima_dir, memory, registry=entity_registry, entity_keys=entity_keys)
    except Exception:
        logger.debug("Failed to index atomic facts", exc_info=True)


def _sync_entity_collection(
    anima_dir: Path,
    memory: Any,
    *,
    registry: dict[str, Any] | None = None,
    entity_keys: set[str] | None = None,
) -> None:
    try:
        from core.memory.entity_index import sync_entity_collection

        indexer = memory._rag._get_indexer()
        vector_store = getattr(indexer, "vector_store", None) if indexer is not None else None
        if vector_store is not None:
            sync_entity_collection(anima_dir, registry=registry, entity_keys=entity_keys, vector_store=vector_store)
    except Exception:
        logger.debug("Failed to sync entity collection", exc_info=True)
