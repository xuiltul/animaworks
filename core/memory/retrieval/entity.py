from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Opt-in entity/phrase ranking helpers for memory retrieval."""

import json
import re
from dataclasses import dataclass
from typing import Any

_QUOTE_RE = re.compile(r"[\"']([^\"']{3,80})[\"']")
_CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+&'.-]*|[A-Z]{2,}[A-Za-z0-9+&'.-]*)"
    r"(?:\s+(?:of|the|and|for|to|by|in|on|at|with|"
    r"[A-Z][A-Za-z0-9+&'.-]*|[A-Z]{2,}[A-Za-z0-9+&'.-]*)){0,4}",
)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]{2,}")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+'-]*|\d{4}|[\u3040-\u30ff\u3400-\u9fff]{2,}")

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "both",
        "by",
        "conversation",
        "did",
        "do",
        "does",
        "done",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "in",
        "is",
        "it",
        "its",
        "kind",
        "like",
        "many",
        "mentioned",
        "of",
        "on",
        "or",
        "question",
        "recently",
        "session",
        "she",
        "some",
        "speaker",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "type",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "why",
        "with",
    },
)


@dataclass(frozen=True)
class EntityBoostConfig:
    """Configuration for additive entity/phrase candidate scoring."""

    enabled: bool = False
    boost: float = 0.20
    max_boost: float = 0.80
    category: int | None = None
    categories: tuple[int, ...] = (1, 4)
    ignored_entities: tuple[str, ...] = ()
    query_entities: tuple[str, ...] = ()
    require_query_entities: bool = False
    prefer_candidate_metadata: bool = True


def extract_entities(text: str, *, ignored_entities: tuple[str, ...] = ()) -> set[str]:
    """Extract deterministic entity-like phrases without external NLP dependencies."""
    ignored = {_normalize_entity(value) for value in ignored_entities}
    ignored.discard("")

    entities: set[str] = set()
    for match in _QUOTE_RE.finditer(text):
        _add_entity(entities, match.group(1), ignored)
    for match in _CAPITALIZED_RE.finditer(text):
        _add_entity(entities, match.group(0), ignored)
    for match in _CJK_RE.finditer(text):
        _add_entity(entities, match.group(0), ignored)

    tokens = _content_tokens(text, ignored)
    entities.update(tokens)
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            _add_entity(entities, phrase, ignored)
    return entities


def apply_entity_boost(
    query: str,
    candidates: list[dict[str, Any]],
    config: EntityBoostConfig | None,
) -> list[dict[str, Any]]:
    """Apply a capped additive boost when query and candidate share entity phrases."""
    if config is None or not config.enabled:
        return candidates
    if config.category is not None and config.category not in config.categories:
        return candidates

    ignored = {_normalize_entity(v) for v in config.ignored_entities}
    ignored.discard("")
    query_entities = {
        _normalize_entity(value)
        for value in config.query_entities
        if _valid_entity(_normalize_entity(value), ignored)
    }
    if config.require_query_entities and not query_entities:
        return candidates
    if not config.require_query_entities:
        query_entities.update(extract_entities(query, ignored_entities=config.ignored_entities))
    if not query_entities:
        return candidates

    boosted: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_entities = _candidate_entities(candidate, config)
        overlap = query_entities & candidate_entities
        if not overlap:
            boosted.append(candidate)
            continue

        row = dict(candidate)
        current_score = float(row.get("score", 0.0) or 0.0)
        row.setdefault("base_score", float(row.get("rrf_score", current_score) or current_score))
        entity_boost = min(max(0.0, float(config.max_boost)), max(0.0, float(config.boost)) * len(overlap))
        row["entity_boost"] = entity_boost
        row["entity_overlap"] = sorted(overlap)[:20]
        row["query_entities"] = sorted(query_entities)[:30]
        row["candidate_entities"] = sorted(candidate_entities)[:30]
        row["score"] = current_score + entity_boost
        boosted.append(row)

    boosted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return boosted


def _candidate_entities(candidate: dict[str, Any], config: EntityBoostConfig) -> set[str]:
    if config.prefer_candidate_metadata:
        metadata_entities = _metadata_entities(candidate, ignored_entities=config.ignored_entities)
        if metadata_entities:
            return metadata_entities
    return extract_entities(
        str(candidate.get("content", "") or ""),
        ignored_entities=config.ignored_entities,
    )


def _metadata_entities(candidate: dict[str, Any], *, ignored_entities: tuple[str, ...]) -> set[str]:
    ignored = {_normalize_entity(value) for value in ignored_entities}
    ignored.discard("")
    values: list[Any] = []
    for source in (candidate, candidate.get("metadata", {})):
        if not isinstance(source, dict):
            continue
        raw = source.get("entities")
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values.extend(parsed)
                else:
                    values.append(raw)
            except json.JSONDecodeError:
                values.extend(part.strip() for part in raw.split(","))
        elif isinstance(raw, (list, tuple, set)):
            values.extend(raw)
    entities: set[str] = set()
    for value in values:
        _add_entity(entities, str(value), ignored)
    return entities


def _add_entity(target: set[str], value: str, ignored: set[str]) -> None:
    entity = _normalize_entity(value)
    if _valid_entity(entity, ignored):
        target.add(entity)


def _normalize_entity(value: str) -> str:
    value = value.replace("’", "'")
    value = re.sub(r"[*_`#\[\](){}<>]", " ", value)
    value = re.sub(r"\b([A-Za-z]+)'s\b", r"\1", value)
    value = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff+&'-]+", " ", value)
    tokens = [token.strip("-'&+ ").lower() for token in value.split()]
    tokens = [token for token in tokens if token]
    while tokens and tokens[0] in _STOPWORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _STOPWORDS:
        tokens.pop()
    return " ".join(tokens)


def _valid_entity(entity: str, ignored: set[str]) -> bool:
    if not entity or entity in ignored or entity in _STOPWORDS:
        return False
    parts = entity.split()
    if any(part in ignored for part in parts):
        return False
    if len(parts) == 1:
        token = parts[0]
        return len(token) >= 3 and token not in _STOPWORDS
    return any(part not in _STOPWORDS and len(part) >= 3 for part in parts)


def _content_tokens(text: str, ignored: set[str]) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = _normalize_entity(match.group(0))
        if _valid_entity(token, ignored):
            tokens.append(token)
    return tokens[:120]
