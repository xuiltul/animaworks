from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Entity/fact graph layers for Legacy NetworkX spreading activation."""

import logging
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from core.time_utils import ensure_aware, now_local

logger = logging.getLogger("animaworks.rag.entity_graph")

ENTITY_AWARE_EDGE_BASE = {
    "mentions_entity": 0.45,
    "fact_source": 0.65,
    "fact_entity": 0.75,
    "co_mention": 0.25,
}
_RECENCY_HALF_LIFE_DAYS = 30.0
_RECENCY_MAX_MULTIPLIER = 1.25


def entity_node_id(entity_key: str) -> str:
    return f"entity:{entity_key}"


def fact_node_id(fact_id: str) -> str:
    return f"fact:{fact_id}"


def add_entity_aware_layers(
    graph: nx.DiGraph,
    anima_dir: Path,
    *,
    graph_entity_edge_cap: int,
    graph_inverse_fan_enabled: bool,
    graph_recency_weight_enabled: bool,
    as_of_time: str | datetime | None,
    make_node_id: Callable[[str, str], str],
    resolve_link_target: Callable[[nx.DiGraph, str], str | None],
) -> None:
    """Add active fact/entity nodes and associative edges to a memory graph."""
    try:
        from core.memory.entity_index import iter_entity_registry_entries, normalize_entity_key
        from core.memory.facts import fact_entity_names, iter_active_fact_records
    except Exception:
        logger.debug("Entity-aware graph dependencies unavailable", exc_info=True)
        return

    entities = {key: entry for key, entry in iter_entity_registry_entries(anima_dir)}
    if not entities:
        logger.debug("Entity-aware graph skipped: registry missing or empty")
        return

    alias_owner = _alias_owner_map(entities, normalize_entity_key)
    for key, entry in entities.items():
        canonical = str(entry.get("canonical", key) or key)
        graph.add_node(
            entity_node_id(key),
            node_type="entity",
            memory_type="entities",
            entity_key=key,
            canonical=canonical,
            aliases=[str(value) for value in entry.get("aliases", []) if str(value).strip()],
        )

    facts = list(iter_active_fact_records(anima_dir, as_of_time=as_of_time))
    entity_carriers: dict[str, list[tuple[str, datetime | None]]] = {key: [] for key in entities}
    memory_mentions: list[tuple[str, set[str], datetime | None]] = []
    fact_mentions: list[tuple[Any, datetime | None, list[str]]] = []

    for node_id, attrs in list(graph.nodes(data=True)):
        if attrs.get("node_type") != "memory_file":
            continue
        path = Path(str(attrs.get("path", "")))
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        mentioned = _mentioned_entity_keys(content, entities, alias_owner, normalize_entity_key)
        recency = _memory_node_time(attrs)
        if not mentioned:
            continue
        memory_mentions.append((node_id, mentioned, recency))
        for entity_key in mentioned:
            entity_carriers[entity_key].append((node_id, recency))

    for record in facts:
        owners: list[str] = []
        for raw_entity in fact_entity_names(record):
            owner = alias_owner.get(normalize_entity_key(raw_entity))
            if owner and owner not in owners:
                owners.append(owner)
        fact_time = _parse_time(record.valid_at or record.recorded_at)
        fact_node = fact_node_id(record.fact_id)
        fact_mentions.append((record, fact_time, owners))
        for owner in owners:
            entity_carriers.setdefault(owner, []).append((fact_node, fact_time))

    entity_fanout = {
        key: max(1, len({carrier for carrier, _recency in carriers}))
        for key, carriers in entity_carriers.items()
    }

    for node_id, mentioned, recency in memory_mentions:
        for entity_key in mentioned:
            entity_node = entity_node_id(entity_key)
            fanout = entity_fanout.get(entity_key, 1)
            weight = _weighted_similarity(
                ENTITY_AWARE_EDGE_BASE["mentions_entity"],
                fanout=fanout,
                recency=recency,
                inverse_fan=graph_inverse_fan_enabled,
                recency_weight=graph_recency_weight_enabled,
            )
            _add_typed_edge(graph, node_id, entity_node, "mentions_entity", weight, fanout)
            _add_typed_edge(graph, entity_node, node_id, "mentions_entity", weight, fanout)

    for record, fact_time, owners in fact_mentions:
        fact_node = fact_node_id(record.fact_id)
        graph.add_node(
            fact_node,
            node_type="fact",
            memory_type="facts",
            fact_id=record.fact_id,
            content=record.text,
            source_file=f"facts/{record.fact_id}",
            source_episode=record.source_episode,
            source_session_id=record.source_session_id,
            valid_at_iso=record.valid_at,
            valid_until=record.valid_until,
            edge_type=record.edge_type,
            source_entity=record.source_entity,
            target_entity=record.target_entity,
            entities=list(record.entities),
        )

        source_node = _resolve_source_memory_node(
            graph,
            record.source_episode,
            make_node_id=make_node_id,
            resolve_link_target=resolve_link_target,
        )
        if source_node:
            weight = _weighted_similarity(
                ENTITY_AWARE_EDGE_BASE["fact_source"],
                fanout=1,
                recency=fact_time,
                inverse_fan=False,
                recency_weight=graph_recency_weight_enabled,
            )
            _add_typed_edge(graph, fact_node, source_node, "fact_source", weight, 1)
            _add_typed_edge(graph, source_node, fact_node, "fact_source", weight, 1)

        for owner in owners:
            entity_node = entity_node_id(owner)
            fanout = entity_fanout.get(owner, 1)
            weight = _weighted_similarity(
                ENTITY_AWARE_EDGE_BASE["fact_entity"],
                fanout=fanout,
                recency=fact_time,
                inverse_fan=graph_inverse_fan_enabled,
                recency_weight=graph_recency_weight_enabled,
            )
            _add_typed_edge(graph, fact_node, entity_node, "fact_entity", weight, fanout)
            _add_typed_edge(graph, entity_node, fact_node, "fact_entity", weight, fanout)

    for _entity_key, carriers in entity_carriers.items():
        unique: dict[str, datetime | None] = {}
        for node_id, recency in carriers:
            unique.setdefault(node_id, recency)
        selected = list(unique.items())[:graph_entity_edge_cap]
        fanout = max(1, len(unique))
        for index, (left, left_time) in enumerate(selected):
            for right, right_time in selected[index + 1 :]:
                if left == right:
                    continue
                recency = max((value for value in (left_time, right_time) if value is not None), default=None)
                weight = _weighted_similarity(
                    ENTITY_AWARE_EDGE_BASE["co_mention"],
                    fanout=fanout,
                    recency=recency,
                    inverse_fan=graph_inverse_fan_enabled,
                    recency_weight=graph_recency_weight_enabled,
                )
                _add_typed_edge(graph, left, right, "co_mention", weight, fanout)
                _add_typed_edge(graph, right, left, "co_mention", weight, fanout)


def graph_diagnostics(graph: nx.DiGraph) -> dict[str, dict[str, int]]:
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for _node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("node_type", "memory_file"))
        node_types[node_type] = node_types.get(node_type, 0) + 1
    for _source, _target, attrs in graph.edges(data=True):
        edge_type = str(attrs.get("link_type", ""))
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    return {"node_types": node_types, "edge_types": edge_types}


def _alias_owner_map(entities: dict[str, dict[str, Any]], normalize_fn) -> dict[str, str]:
    owners: dict[str, str] = {}
    for key, entry in entities.items():
        owners.setdefault(normalize_fn(key), key)
        owners.setdefault(normalize_fn(entry.get("canonical", "")), key)
        for alias in entry.get("aliases", []):
            alias_key = normalize_fn(alias)
            if alias_key:
                owners.setdefault(alias_key, key)
    return owners


def _mentioned_entity_keys(
    content: str,
    entities: dict[str, dict[str, Any]],
    alias_owner: dict[str, str],
    normalize_fn,
) -> set[str]:
    normalized_content = normalize_fn(content)
    mentioned: set[str] = set()
    for key, entry in entities.items():
        candidates = [key, str(entry.get("canonical", "") or ""), *entry.get("aliases", [])]
        for candidate in candidates:
            normalized = normalize_fn(candidate)
            if normalized and normalized in normalized_content:
                mentioned.add(alias_owner.get(normalized, key))
                break
    return mentioned


def _parse_time(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def _memory_node_time(attrs: dict[str, Any]) -> datetime | None:
    path = Path(str(attrs.get("path", "") or ""))
    if path.is_file():
        try:
            return ensure_aware(datetime.fromtimestamp(path.stat().st_mtime, tz=now_local().tzinfo))
        except OSError:
            return None
    return _parse_time(attrs.get("valid_at") or attrs.get("event_time_iso"))


def _weighted_similarity(
    base: float,
    *,
    fanout: int,
    recency: datetime | None,
    inverse_fan: bool,
    recency_weight: bool,
) -> float:
    weight = float(base)
    if inverse_fan:
        weight *= 1.0 / math.sqrt(max(1, int(fanout)))
    if recency_weight and recency is not None:
        age_days = max(0.0, (now_local() - ensure_aware(recency)).total_seconds() / 86400.0)
        recency_multiplier = 1.0 + (_RECENCY_MAX_MULTIPLIER - 1.0) * (
            0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)
        )
        weight *= recency_multiplier
    return max(0.0, weight)


def _add_typed_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    edge_type: str,
    similarity: float,
    fanout: int,
) -> None:
    if source == target:
        return
    existing = graph.get_edge_data(source, target)
    if existing and float(existing.get("similarity", 0.0) or 0.0) >= similarity:
        return
    graph.add_edge(
        source,
        target,
        link_type=edge_type,
        similarity=similarity,
        base_similarity=ENTITY_AWARE_EDGE_BASE.get(edge_type, similarity),
        fanout=fanout,
    )


def _resolve_source_memory_node(
    graph: nx.DiGraph,
    source_episode: str,
    *,
    make_node_id: Callable[[str, str], str],
    resolve_link_target: Callable[[nx.DiGraph, str], str | None],
) -> str | None:
    source = str(source_episode or "").strip().split("#", 1)[0]
    if not source:
        return None
    parts = Path(source).parts
    if parts and parts[0] in {"knowledge", "episodes", "procedures", "skills", "common_knowledge"}:
        memory_type = parts[0]
        rel_key = str(Path(*parts[1:]).with_suffix("")) if len(parts) > 1 else ""
        candidate = make_node_id(rel_key, memory_type)
        if candidate in graph:
            return candidate
    stem = Path(source).stem
    return resolve_link_target(graph, stem)
