from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy entity registry and best-effort entity vector collection sync."""

import contextlib
import json
import logging
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.memory.facts import FactRecord, fact_entity_names, iter_fact_records
from core.platform.locks import acquire_file_lock, release_file_lock
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.memory.entity_index")

REGISTRY_VERSION = 1
_NORMALIZE_RE = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff+&'-]+")
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def normalize_entity_key(value: object) -> str:
    """Normalize an entity/alias into the stable registry key space."""
    text = str(value or "").replace("’", "'")
    text = _NORMALIZE_RE.sub(" ", text).strip().casefold()
    return " ".join(part for part in text.split() if part)


def entity_registry_path(anima_dir: Path) -> Path:
    return Path(anima_dir) / "state" / "entity_registry.json"


def empty_registry() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "entities": {}}


def load_entity_registry(anima_dir: Path, *, rebuild_on_corrupt: bool = True) -> dict[str, Any]:
    """Load the registry, rebuilding from facts when corrupt."""
    with _locked_registry(anima_dir):
        return _load_entity_registry_unlocked(anima_dir, rebuild_on_corrupt=rebuild_on_corrupt)


def save_entity_registry(anima_dir: Path, registry: dict[str, Any]) -> None:
    with _locked_registry(anima_dir):
        _save_entity_registry_unlocked(anima_dir, registry)


def upsert_entities_from_facts(anima_dir: Path, records: list[FactRecord]) -> dict[str, Any]:
    """Upsert fact entities into the authoritative JSON registry."""
    with _locked_registry(anima_dir):
        registry = _load_entity_registry_unlocked(anima_dir)
        entities: dict[str, Any] = registry.setdefault("entities", {})
        alias_owner = _alias_owner_map(entities)
        changed = False

        for record in records:
            source_fact_id = record.fact_id
            seen_for_record: set[str] = set()
            for raw_entity in record_entities(record):
                key = normalize_entity_key(raw_entity)
                if not key or key in seen_for_record:
                    continue
                seen_for_record.add(key)
                existing_key = alias_owner.get(key, key)
                entry = entities.get(existing_key)
                if not isinstance(entry, dict):
                    entry = _new_entry(str(raw_entity), source_fact_id)
                    entities[existing_key] = entry
                    alias_owner[existing_key] = existing_key
                    alias_owner[key] = existing_key
                    changed = True
                changed = _merge_entry(entry, raw_entity, source_fact_id) or changed

        if changed:
            _save_entity_registry_unlocked(anima_dir, registry)
        return registry


def rebuild_entity_registry(anima_dir: Path) -> dict[str, Any]:
    """Rebuild registry from active facts JSONL files."""
    with _locked_registry(anima_dir):
        return _rebuild_entity_registry_unlocked(anima_dir)


def _load_entity_registry_unlocked(anima_dir: Path, *, rebuild_on_corrupt: bool = True) -> dict[str, Any]:
    path = entity_registry_path(anima_dir)
    if not path.is_file():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("entities"), dict):
            raise ValueError("entity registry must contain an entities object")
        data["version"] = int(data.get("version") or REGISTRY_VERSION)
        return data
    except Exception:
        logger.warning("Entity registry is corrupt: %s", path, exc_info=True)
        if not rebuild_on_corrupt:
            return empty_registry()
        _archive_corrupt_registry(path)
        try:
            return _rebuild_entity_registry_unlocked(anima_dir)
        except Exception:
            logger.warning("Failed to rebuild entity registry from facts", exc_info=True)
            return empty_registry()


def _save_entity_registry_unlocked(anima_dir: Path, registry: dict[str, Any]) -> None:
    registry = _normalize_registry(registry)
    path = entity_registry_path(anima_dir)
    atomic_write_text(path, json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _rebuild_entity_registry_unlocked(anima_dir: Path) -> dict[str, Any]:
    registry = empty_registry()
    records = list(iter_fact_records(anima_dir))
    if records:
        entities = registry["entities"]
        alias_owner: dict[str, str] = {}
        for record in records:
            seen_for_record: set[str] = set()
            for raw_entity in record_entities(record):
                key = normalize_entity_key(raw_entity)
                if not key or key in seen_for_record:
                    continue
                seen_for_record.add(key)
                existing_key = alias_owner.get(key, key)
                entry = entities.get(existing_key)
                if not isinstance(entry, dict):
                    entry = _new_entry(str(raw_entity), record.fact_id)
                    entities[existing_key] = entry
                    alias_owner[existing_key] = existing_key
                    alias_owner[key] = existing_key
                _merge_entry(entry, raw_entity, record.fact_id)
    _save_entity_registry_unlocked(anima_dir, registry)
    return registry


def match_query_entities(
    anima_dir: Path,
    query: str,
    *,
    vector_store: Any | None = None,
    embedding_fn: Any | None = None,
    top_k: int = 5,
    min_score: float = 0.35,
) -> set[str]:
    """Return registry entity keys matched by deterministic query phrases."""
    if not query.strip():
        return set()
    registry = load_entity_registry(anima_dir, rebuild_on_corrupt=True)
    entities: dict[str, Any] = registry.get("entities", {})
    alias_owner = _alias_owner_map(entities)
    try:
        from core.memory.retrieval.entity import extract_entities

        query_entities = extract_entities(query)
    except Exception:
        query_entities = {query}
    matches: set[str] = set()
    for value in query_entities:
        key = normalize_entity_key(value)
        owner = alias_owner.get(key)
        if owner:
            matches.add(owner)
    if matches:
        return matches
    collection_matches = _match_query_entities_from_collection(
        anima_dir,
        query,
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        top_k=top_k,
        min_score=min_score,
    )
    matches.update(key for key in collection_matches if key in entities)
    return matches


def sync_entity_collection(
    anima_dir: Path,
    *,
    registry: dict[str, Any] | None = None,
    entity_keys: set[str] | None = None,
    vector_store: Any | None = None,
    embedding_fn: Any | None = None,
) -> bool:
    """Best-effort sync of registry entries into ``{anima}_entities``."""
    registry = registry or load_entity_registry(anima_dir)
    entities = registry.get("entities", {})
    if not isinstance(entities, dict) or not entities:
        return True
    try:
        if vector_store is None:
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store(Path(anima_dir).name)
        if vector_store is None:
            return False
        if embedding_fn is None:
            from core.memory.rag.singleton import generate_embeddings

            embedding_fn = generate_embeddings
        from core.memory.rag.store import Document

        docs: list[Any] = []
        sorted_items = sorted(entities.items())
        if entity_keys is not None:
            wanted = {normalize_entity_key(value) for value in entity_keys if normalize_entity_key(value)}
            sorted_items = [(key, entry) for key, entry in sorted_items if normalize_entity_key(key) in wanted]
            if not sorted_items:
                return True
        contents = [_entity_document_content(entry) for _key, entry in sorted_items]
        embeddings = embedding_fn(contents)
        if len(embeddings) != len(contents):
            return False
        for index, (key, entry) in enumerate(sorted_items):
            docs.append(
                Document(
                    id=f"{Path(anima_dir).name}/entity/{key}",
                    content=contents[index],
                    embedding=embeddings[index],
                    metadata=_entity_document_metadata(key, entry),
                )
            )
        collection = f"{Path(anima_dir).name}_entities"
        vector_store.create_collection(collection)
        return bool(vector_store.upsert(collection, docs))
    except Exception:
        logger.warning("Failed to sync entity collection for %s", anima_dir, exc_info=True)
        return False


def entity_keys_for_records(registry: dict[str, Any], records: list[FactRecord]) -> set[str]:
    """Resolve record-carried entity names to canonical registry keys."""
    entities = registry.get("entities", {})
    if not isinstance(entities, dict):
        return set()
    alias_owner = _alias_owner_map(entities)
    keys: set[str] = set()
    for record in records:
        for raw_entity in record_entities(record):
            owner = alias_owner.get(normalize_entity_key(raw_entity))
            if owner:
                keys.add(owner)
    return keys


def iter_entity_registry_entries(anima_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate normalized entity registry entries for graph construction."""
    registry = load_entity_registry(anima_dir, rebuild_on_corrupt=True)
    entities = registry.get("entities", {})
    if not isinstance(entities, dict):
        return
    for key, entry in sorted(entities.items()):
        if isinstance(entry, dict):
            yield normalize_entity_key(key), entry


def _match_query_entities_from_collection(
    anima_dir: Path,
    query: str,
    *,
    vector_store: Any | None,
    embedding_fn: Any | None,
    top_k: int,
    min_score: float,
) -> set[str]:
    try:
        if vector_store is None:
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store(Path(anima_dir).name)
        if vector_store is None:
            return set()
        if embedding_fn is None:
            from core.memory.rag.singleton import generate_embeddings

            embedding_fn = generate_embeddings
        embeddings = embedding_fn([query])
        if not embeddings:
            return set()
        collection = f"{Path(anima_dir).name}_entities"
        results = vector_store.query(collection, embeddings[0], top_k=top_k)
    except Exception:
        logger.debug("Failed to query entity collection for %s", anima_dir, exc_info=True)
        return set()

    matches: set[str] = set()
    for result in results or []:
        try:
            score = float(getattr(result, "score", 0.0) or 0.0)
            if score < min_score:
                continue
            document = getattr(result, "document", None)
            metadata = getattr(document, "metadata", {}) if document is not None else {}
            key = normalize_entity_key(metadata.get("entity_key") or metadata.get("canonical") or "")
            if key:
                matches.add(key)
        except Exception:
            logger.debug("Failed to parse entity collection result", exc_info=True)
    return matches


def record_entities(record: FactRecord) -> list[str]:
    values = fact_entity_names(record)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_entity_key(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _process_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        if resolved not in _LOCKS:
            _LOCKS[resolved] = threading.Lock()
        return _LOCKS[resolved]


@contextlib.contextmanager
def _locked_registry(anima_dir: Path) -> Iterator[None]:
    path = entity_registry_path(anima_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    thread_lock = _process_lock(lock_path)
    with thread_lock, lock_path.open("a+", encoding="utf-8") as lock_file:
        locked = False
        try:
            acquire_file_lock(lock_file, exclusive=True)
            locked = True
        except OSError:
            logger.debug("OS file lock unavailable for %s", lock_path, exc_info=True)
        try:
            yield
        finally:
            if locked:
                try:
                    release_file_lock(lock_file)
                except OSError:
                    logger.debug("Failed to release registry lock %s", lock_path, exc_info=True)


def _new_entry(canonical: str, source_fact_id: str) -> dict[str, Any]:
    ts = now_iso()
    key = normalize_entity_key(canonical)
    return {
        "canonical": str(canonical or "").strip(),
        "aliases": [key] if key else [],
        "entity_type": "",
        "mention_count": 0,
        "first_seen_at": ts,
        "last_seen_at": ts,
        "source_fact_ids": [],
    }


def _merge_entry(entry: dict[str, Any], raw_entity: object, source_fact_id: str) -> bool:
    changed = False
    alias = normalize_entity_key(raw_entity)
    aliases = [str(value) for value in entry.get("aliases", []) if str(value).strip()]
    if alias and alias not in {normalize_entity_key(value) for value in aliases}:
        aliases.append(alias)
        entry["aliases"] = aliases
        changed = True
    fact_ids = [str(value) for value in entry.get("source_fact_ids", []) if str(value).strip()]
    if source_fact_id and source_fact_id not in fact_ids:
        fact_ids.append(source_fact_id)
        entry["source_fact_ids"] = fact_ids
        entry["mention_count"] = len(fact_ids)
        entry.setdefault("first_seen_at", now_iso())
        entry["last_seen_at"] = now_iso()
        changed = True
    elif "mention_count" not in entry:
        entry["mention_count"] = len(fact_ids)
        changed = True
    return changed


def _alias_owner_map(entities: dict[str, Any]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for key, entry in entities.items():
        norm_key = normalize_entity_key(key)
        owners.setdefault(norm_key, key)
        if isinstance(entry, dict):
            owners.setdefault(normalize_entity_key(entry.get("canonical")), key)
            for alias in entry.get("aliases", []):
                alias_key = normalize_entity_key(alias)
                if alias_key:
                    owners.setdefault(alias_key, key)
    return owners


def _normalize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "entities": dict(registry.get("entities", {}))}


def _archive_corrupt_registry(path: Path) -> None:
    if not path.exists():
        return
    stamp = now_iso().replace(":", "").replace("+", "_").replace(".", "_")
    dest = path.with_name(f"{path.name}.corrupt.{stamp}")
    try:
        path.rename(dest)
    except OSError:
        logger.debug("Failed to archive corrupt entity registry %s", path, exc_info=True)


def _entity_document_content(entry: dict[str, Any]) -> str:
    aliases = ", ".join(str(value) for value in entry.get("aliases", []) if str(value).strip())
    entity_type = str(entry.get("entity_type", "") or "")
    return "\n".join(
        value
        for value in (
            f"Entity: {entry.get('canonical', '')}",
            f"Aliases: {aliases}" if aliases else "",
            f"Type: {entity_type}" if entity_type else "",
        )
        if value
    )


def _entity_document_metadata(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_type": "entities",
        "entity_key": key,
        "canonical": str(entry.get("canonical", "") or ""),
        "aliases": [str(value) for value in entry.get("aliases", []) if str(value).strip()],
        "entity_type": str(entry.get("entity_type", "") or ""),
        "mention_count": int(entry.get("mention_count", 0) or 0),
        "first_seen_at": str(entry.get("first_seen_at", "") or ""),
        "last_seen_at": str(entry.get("last_seen_at", "") or ""),
        "source_fact_ids": [str(value) for value in entry.get("source_fact_ids", []) if str(value).strip()],
    }
