from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Opt-in entity/phrase ranking helpers for memory retrieval."""

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
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

# mtime-based registry cache for search hot path (no rebuild, no locks beyond this).
_REGISTRY_CACHE_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, _CachedRegistry] = {}


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
    use_content_tokens: bool = True
    require_multi_token_overlap: bool = False
    # When set, load entity_registry.json for alias expansion + 1-hop related boost.
    anima_dir: str | Path | None = None
    # Additive per related (1-hop) entity match; default is half of ``boost`` when None.
    related_boost: float | None = None


@dataclass(frozen=True)
class EntityAliasIndex:
    """Lightweight view of the entity registry for retrieval-time alias resolution."""

    alias_owner: dict[str, str]
    synonyms: dict[str, frozenset[str]]
    related: dict[str, frozenset[str]]


@dataclass
class _CachedRegistry:
    mtime_ns: int
    index: EntityAliasIndex


def extract_entities(
    text: str,
    *,
    ignored_entities: tuple[str, ...] = (),
    use_content_tokens: bool = True,
) -> set[str]:
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

    if use_content_tokens:
        tokens = _content_tokens(text, ignored)
        entities.update(tokens)
        for size in (2, 3):
            for index in range(0, max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[index : index + size])
                _add_entity(entities, phrase, ignored)
    return entities


def expand_alias_terms(
    text: str,
    alias_map: dict[str, tuple[str, ...]],
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    """Return deterministic alias terms whose trigger phrases appear in text."""
    normalized = _normalize_entity(text)
    aliases: list[str] = []
    seen: set[str] = set()
    for trigger, values in alias_map.items():
        clean_trigger = _normalize_entity(trigger)
        if not clean_trigger or clean_trigger not in normalized:
            continue
        for value in values:
            alias = str(value or "").strip()
            key = alias.casefold()
            if not alias or key in seen:
                continue
            aliases.append(alias)
            seen.add(key)
            if len(aliases) >= limit:
                return tuple(aliases)
    return tuple(aliases)


def load_entity_alias_index(anima_dir: str | Path | None) -> EntityAliasIndex | None:
    """Load (or return cached) alias index from ``anima_dir/state/entity_registry.json``.

    Uses mtime-based invalidation. Missing/empty/corrupt registries return None so
    callers can fall back to regex-only entity extraction.
    """
    if anima_dir is None:
        return None
    try:
        path = Path(anima_dir) / "state" / "entity_registry.json"
        if not path.is_file():
            return None
        stat = path.stat()
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
        cache_key = str(path.resolve())
    except OSError:
        return None

    with _REGISTRY_CACHE_LOCK:
        cached = _REGISTRY_CACHE.get(cache_key)
        if cached is not None and cached.mtime_ns == mtime_ns:
            return cached.index

    index = _build_alias_index_from_path(path)
    if index is None:
        return None

    with _REGISTRY_CACHE_LOCK:
        _REGISTRY_CACHE[cache_key] = _CachedRegistry(mtime_ns=mtime_ns, index=index)
    return index


def clear_entity_alias_index_cache() -> None:
    """Drop the process-local registry cache (tests / hot-reload)."""
    with _REGISTRY_CACHE_LOCK:
        _REGISTRY_CACHE.clear()


def apply_entity_boost(
    query: str,
    candidates: list[dict[str, Any]],
    config: EntityBoostConfig | None,
) -> list[dict[str, Any]]:
    """Apply a capped additive boost when query and candidate share entity phrases.

    When ``config.anima_dir`` points at a populated entity registry, query phrases
    are resolved through canonical/aliases so alternate surface forms still match,
    and candidates that mention 1-hop related entities receive a smaller boost.
    Without a registry the historical regex-only path is used unchanged.
    """
    if config is None or not config.enabled:
        return candidates
    if config.category is not None and config.category not in config.categories:
        return candidates

    ignored = {_normalize_entity(v) for v in config.ignored_entities}
    ignored.discard("")
    query_entities = {
        _normalize_entity(value) for value in config.query_entities if _valid_entity(_normalize_entity(value), ignored)
    }
    if config.require_query_entities and not query_entities:
        return candidates
    if not config.require_query_entities:
        query_entities.update(
            extract_entities(
                query,
                ignored_entities=config.ignored_entities,
                use_content_tokens=config.use_content_tokens,
            ),
        )
    if not query_entities:
        return candidates

    alias_index = load_entity_alias_index(config.anima_dir)
    # Substring match against full query text (CJK extract often glues particles).
    query_keys = _match_registry_keys_in_text(query, alias_index) if alias_index else set()
    query_keys |= _resolve_entity_keys(query_entities, alias_index) if alias_index else set()
    related_of_query: set[str] = set()
    if alias_index and query_keys:
        for key in query_keys:
            related_of_query.update(alias_index.related.get(key, ()))
        related_of_query -= query_keys

    # Surface forms of query entities (incl. aliases) for reporting / phrase fallback.
    query_surfaces = set(query_entities)
    if alias_index and query_keys:
        for key in query_keys:
            query_surfaces.update(alias_index.synonyms.get(key, ()))

    related_boost_value = (
        float(config.related_boost)
        if config.related_boost is not None
        else max(0.0, float(config.boost)) * 0.5
    )
    primary_boost = max(0.0, float(config.boost))
    max_boost = max(0.0, float(config.max_boost))

    boosted: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_entities = _candidate_entities(candidate, config)
        content = str(candidate.get("content", "") or "")
        candidate_keys = set()
        if alias_index:
            candidate_keys |= _match_registry_keys_in_text(content, alias_index)
            candidate_keys |= _resolve_entity_keys(candidate_entities, alias_index)
            # Metadata entity lists (may not appear in content text).
            for value in candidate_entities:
                candidate_keys |= _resolve_entity_keys({value}, alias_index)
        candidate_surfaces = set(candidate_entities)
        if alias_index and candidate_keys:
            for key in candidate_keys:
                candidate_surfaces.update(alias_index.synonyms.get(key, ()))

        primary_keys: set[str] = set()
        related_keys_hit: set[str] = set()
        phrase_overlap: set[str] = set()

        if alias_index:
            primary_keys = query_keys & candidate_keys
            related_keys_hit = related_of_query & candidate_keys
            related_keys_hit -= primary_keys
            # Unregistered phrases keep the historical intersection path.
            registered_surfaces = set(alias_index.alias_owner)
            unreg_query = {p for p in query_entities if p not in registered_surfaces}
            unreg_candidate = {p for p in candidate_entities if p not in registered_surfaces}
            phrase_overlap = unreg_query & unreg_candidate
        else:
            phrase_overlap = query_entities & candidate_entities

        if config.require_multi_token_overlap:
            phrase_overlap = {entity for entity in phrase_overlap if len(entity.split()) > 1}
            if alias_index:
                primary_keys = {
                    key
                    for key in primary_keys
                    if any(len(s.split()) > 1 for s in alias_index.synonyms.get(key, ()))
                }
                related_keys_hit = {
                    key
                    for key in related_keys_hit
                    if any(len(s.split()) > 1 for s in alias_index.synonyms.get(key, ()))
                }

        primary_count = len(primary_keys) + len(phrase_overlap)
        related_count = len(related_keys_hit)
        if primary_count == 0 and related_count == 0:
            boosted.append(candidate)
            continue

        overlap_labels = set(phrase_overlap) | set(primary_keys)
        if primary_keys and alias_index:
            for key in primary_keys:
                overlap_labels.update(alias_index.synonyms.get(key, ()) & (query_surfaces | candidate_surfaces))
            overlap_labels |= primary_keys

        row = dict(candidate)
        current_score = float(row.get("score", 0.0) or 0.0)
        row.setdefault("base_score", float(row.get("rrf_score", current_score) or current_score))
        entity_boost = min(
            max_boost,
            primary_boost * primary_count + related_boost_value * related_count,
        )
        row["entity_boost"] = entity_boost
        row["entity_overlap"] = sorted(overlap_labels)[:20]
        if related_keys_hit:
            row["entity_related_overlap"] = sorted(related_keys_hit)[:20]
        row["query_entities"] = sorted(query_surfaces)[:30]
        row["candidate_entities"] = sorted(candidate_surfaces)[:30]
        row["score"] = current_score + entity_boost
        boosted.append(row)

    boosted.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return boosted


def _build_alias_index_from_path(path: Path) -> EntityAliasIndex | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    entities = data.get("entities")
    if not isinstance(entities, dict) or not entities:
        return None
    return _build_alias_index(entities)


def _build_alias_index(entities: dict[str, Any]) -> EntityAliasIndex:
    from core.memory.entity_index import normalize_entity_key

    alias_owner: dict[str, str] = {}
    synonyms: dict[str, set[str]] = {}
    fact_to_entities: dict[str, set[str]] = {}

    for raw_key, entry in entities.items():
        if not isinstance(entry, dict):
            continue
        key = normalize_entity_key(raw_key) or str(raw_key).strip().casefold()
        if not key:
            continue
        surfaces: set[str] = {key}
        surfaces.add(_normalize_entity(key))
        for value in (raw_key, entry.get("canonical", ""), *entry.get("aliases", [])):
            text = str(value or "").strip()
            if not text:
                continue
            for form in (normalize_entity_key(text), _normalize_entity(text), text.casefold()):
                if form:
                    surfaces.add(form)
        surfaces.discard("")
        synonyms[key] = surfaces
        for surface in surfaces:
            alias_owner.setdefault(surface, key)
        for fact_id in entry.get("source_fact_ids", []) or []:
            fid = str(fact_id or "").strip()
            if fid:
                fact_to_entities.setdefault(fid, set()).add(key)

    related: dict[str, set[str]] = {key: set() for key in synonyms}
    for group in fact_to_entities.values():
        if len(group) < 2:
            continue
        for entity_key in group:
            related.setdefault(entity_key, set()).update(group - {entity_key})

    return EntityAliasIndex(
        alias_owner=alias_owner,
        synonyms={key: frozenset(values) for key, values in synonyms.items()},
        related={key: frozenset(values) for key, values in related.items()},
    )


def _resolve_entity_keys(phrases: set[str], index: EntityAliasIndex | None) -> set[str]:
    if not index or not phrases:
        return set()
    from core.memory.entity_index import normalize_entity_key

    keys: set[str] = set()
    for phrase in phrases:
        for form in (_normalize_entity(phrase), normalize_entity_key(phrase), phrase.casefold()):
            owner = index.alias_owner.get(form)
            if owner:
                keys.add(owner)
                break
        # Also allow alias as substring of a glued CJK phrase (e.g. なつめのタスク).
        if index:
            keys |= _match_registry_keys_in_text(phrase, index)
    return keys


def _match_registry_keys_in_text(text: str, index: EntityAliasIndex | None) -> set[str]:
    """Return registry keys whose canonical/alias surface appears in text."""
    if not index or not text:
        return set()
    from core.memory.entity_index import normalize_entity_key

    haystacks = {
        normalize_entity_key(text),
        _normalize_entity(text),
        text.casefold(),
    }
    haystacks.discard("")
    if not haystacks:
        return set()

    matched: set[str] = set()
    for key, synonyms in index.synonyms.items():
        # Longer surfaces first so more specific aliases win diagnostics.
        for surface in sorted(synonyms, key=len, reverse=True):
            if len(surface) < 2:
                continue
            if any(surface in hay for hay in haystacks):
                matched.add(key)
                break
    return matched


def _candidate_entities(candidate: dict[str, Any], config: EntityBoostConfig) -> set[str]:
    if config.prefer_candidate_metadata:
        metadata_entities = _metadata_entities(candidate, ignored_entities=config.ignored_entities)
        if metadata_entities:
            return metadata_entities
    return extract_entities(
        str(candidate.get("content", "") or ""),
        ignored_entities=config.ignored_entities,
        use_content_tokens=config.use_content_tokens,
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
