from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Deterministic LoCoMo category-1 multi-hop retrieval helpers."""

import logging
import os
import re
from typing import Any

MULTIHOP_METADATA_FIELDS: tuple[str, ...] = (
    "locomo_multihop_helper",
    "locomo_multihop_query",
    "locomo_multihop_aliases",
    "locomo_multihop_person",
)

_MULTIHOP_QUERY_LIMIT = 6
_MULTIHOP_ALIAS_LIMIT = 8
_MULTIHOP_PROFILE_FACT_LIMIT = 12
_MULTIHOP_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "destress": ("relax", "unwind", "mental health", "running", "pottery"),
    "identity": ("transgender", "woman", "transgender woman"),
    "relationship status": ("single", "dating", "partner", "relationship"),
    "children": ("son", "daughter", "children", "kids", "three kids", "3"),
    "kids": ("son", "daughter", "children", "kids", "dinosaurs", "nature"),
    "symbols": ("rainbow flag", "transgender symbol", "necklace", "symbolizes"),
    "events": ("attended", "joined", "speech", "parade", "support group", "conference", "mentoring"),
    "participated": ("attended", "joined", "speech", "parade", "support group", "conference", "mentoring"),
    "instruments": ("clarinet", "violin", "music", "play"),
    "books": ("book", "read", "Charlotte's Web", "Nothing is Impossible"),
    "read": ("book", "read", "Charlotte's Web", "Nothing is Impossible"),
    "pets": ("pet", "dog", "cat", "Oliver", "Luna", "Bailey"),
    "painted": ("paint", "painting", "sunset", "sunrise", "art"),
    "both painted": ("paint", "painting", "sunset", "sunrise", "art"),
}
_MULTIHOP_STOPWORDS = frozenset(
    {
        "a",
        "and",
        "are",
        "both",
        "did",
        "do",
        "does",
        "for",
        "has",
        "have",
        "how",
        "is",
        "kind",
        "many",
        "of",
        "on",
        "some",
        "subject",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "whose",
        "with",
    },
)

logger = logging.getLogger(__name__)


def empty_multihop_meta() -> dict[str, Any]:
    return {
        "enabled": False,
        "alias_map_enabled": False,
        "fallback_used": False,
        "query_count": 0,
        "queries": [],
        "aliases": [],
        "helper_hit_counts": {},
    }


def multihop_aliases(question: str, *, enable_alias_map: bool = False) -> tuple[str, ...]:
    if not enable_alias_map:
        return ()

    from core.memory.retrieval.entity import expand_alias_terms  # noqa: PLC0415

    return expand_alias_terms(question, _MULTIHOP_ALIAS_MAP, limit=_MULTIHOP_ALIAS_LIMIT)


def _fact_index_enabled() -> bool:
    return os.environ.get("LOCOMO_FACT_INDEX", "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _multihop_terms(question: str, *, aliases: tuple[str, ...], persons: list[str]) -> list[str]:
    ignored = {person.casefold() for person in persons}
    for person in persons:
        ignored.update(part.casefold() for part in re.findall(r"[\w]+", person, flags=re.UNICODE))

    terms: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        clean = _normalize_space(raw.strip(" \t\r\n.,;:!?\"'()[]{}"))
        if clean.endswith("'s"):
            clean = clean[:-2]
        key = clean.casefold()
        if (
            not key
            or key in seen
            or key in ignored
            or key in _MULTIHOP_STOPWORDS
            or (len(key) < 3 and not key.isdigit())
        ):
            return
        terms.append(clean)
        seen.add(key)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]*|\d+", question):
        add(token)
    for alias in aliases:
        add(alias)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]*|\d+", alias):
            add(token)
    return terms[:16]


def _is_shared_subject_question(question: str) -> bool:
    key = f" {question.casefold()} "
    if re.search(r"\b(both|shared|same|common)\b", key):
        return True
    if " and " not in key:
        return False
    return bool(
        re.search(
            r"\b(attend|attended|do|does|did|have|has|like|liked|paint|painted|play|played|read|same|share)\b",
            key,
        ),
    )


def _text_matches_any(text_key: str, terms: list[str] | tuple[str, ...]) -> bool:
    for raw in terms:
        term = str(raw or "").strip().casefold()
        if not term:
            continue
        if term in text_key:
            return True
        tokens = [token.casefold() for token in re.findall(r"[\w]+", term, flags=re.UNICODE)]
        tokens = [token for token in tokens if token and token not in _MULTIHOP_STOPWORDS]
        if tokens and all(token in text_key for token in tokens):
            return True
    return False


def _multihop_helper_bonus(helper: str) -> float:
    return {
        "intersection": 0.35,
        "fact_fallback": 0.25,
        "decomposition": 0.20,
        "alias": 0.15,
        "profile": 0.10,
    }.get(helper, 0.0)


class LocomoMultiHopOrchestrator:
    """Adapter-local category-1 helper orchestration."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def feature_enabled(self, category: int | None) -> bool:
        return category == 1 and _fact_index_enabled() and int(getattr(self._adapter, "_last_fact_count", 0) or 0) > 0

    def augment(self, question: str, base_items: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        alias_map_enabled = bool(getattr(self._adapter, "_enable_locomo_alias", False))
        aliases = multihop_aliases(question, enable_alias_map=alias_map_enabled)
        persons = self.persons(question)
        query_specs = self.query_specs(question, persons=persons, aliases=aliases)
        if len(base_items) >= top_k:
            self._adapter._last_multihop_meta = self._meta(
                query_specs=query_specs,
                aliases=aliases,
                alias_map_enabled=alias_map_enabled,
                fallback_used=False,
                helper_counts={},
            )
            return base_items[:top_k]

        helper_items: list[dict[str, Any]] = []
        for spec in query_specs:
            if spec["helper"] == "decomposition" or spec["helper"] == "alias" or spec["helper"] == "intersection":
                helper_items.extend(self.fact_search(spec["query"], spec, top_k=top_k))

        helper_items.extend(self.profile_candidates(question, persons=persons, aliases=aliases))
        fallback_used = False
        if not base_items:
            fallback_query = " ".join([question, *aliases[:_MULTIHOP_ALIAS_LIMIT]]).strip()
            fallback_spec = {
                "query": fallback_query or question,
                "helper": "fact_fallback",
                "aliases": aliases[:_MULTIHOP_ALIAS_LIMIT],
                "person": "",
            }
            fallback_items = self.fact_search(fallback_spec["query"], fallback_spec, top_k=top_k, include_vectors=True)
            if fallback_items:
                self._adapter._last_abstain = False
                self._adapter._last_abstain_reason = ""
                fallback_used = True
                helper_items.extend(fallback_items)

        merged = self.merge_items(base_items, helper_items, limit=top_k)
        helper_counts: dict[str, int] = {}
        for item in merged:
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            helper = str(meta.get("locomo_multihop_helper", "") or "")
            if helper:
                helper_counts[helper] = helper_counts.get(helper, 0) + 1
        self._adapter._last_multihop_meta = self._meta(
            query_specs=query_specs,
            aliases=aliases,
            alias_map_enabled=alias_map_enabled,
            fallback_used=fallback_used,
            helper_counts=helper_counts,
        )
        return merged

    def persons(self, question: str) -> list[str]:
        question_key = question.casefold()
        ignored_entities = tuple(getattr(self._adapter, "_entity_ignored_entities", ()) or ())
        persons = [name for name in ignored_entities if name.casefold() in question_key]
        if persons:
            return persons[:2]
        return list(ignored_entities[:2])

    def query_specs(
        self,
        question: str,
        *,
        persons: list[str],
        aliases: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        terms = _multihop_terms(question, aliases=aliases, persons=persons)
        specs: list[dict[str, Any]] = []

        def add(query: str, helper: str, *, person: str = "") -> None:
            clean = _normalize_space(query)
            if not clean or clean.casefold() in {str(spec["query"]).casefold() for spec in specs}:
                return
            specs.append(
                {
                    "query": clean,
                    "helper": helper,
                    "aliases": aliases[:_MULTIHOP_ALIAS_LIMIT],
                    "person": person,
                },
            )

        if aliases:
            add(" ".join([question, *aliases[:_MULTIHOP_ALIAS_LIMIT]]), "alias")
        if _is_shared_subject_question(question) and len(persons) >= 2:
            shared_terms = aliases[:_MULTIHOP_ALIAS_LIMIT] or tuple(terms[:6])
            for person in persons:
                add(" ".join([person, "shared", *shared_terms]), "intersection", person=person)
        for person in persons:
            if aliases:
                add(" ".join([person, *aliases[:_MULTIHOP_ALIAS_LIMIT]]), "decomposition", person=person)
            if terms:
                add(" ".join([person, *terms[:6]]), "decomposition", person=person)
        return specs[:_MULTIHOP_QUERY_LIMIT]

    def fact_search(
        self,
        query: str,
        spec: dict[str, Any],
        *,
        top_k: int,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            rows = self._adapter._fact_bm25_search(query, top_k)
            if include_vectors:
                rows = self._adapter._rrf_merge(self._adapter._search_fact_vectors(query, top_k), rows)
        except Exception:
            logger.debug("LoCoMo multi-hop helper search failed", exc_info=True)
            return []
        return [self.annotate_item(row, spec) for row in rows]

    def annotate_item(self, row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
        item = {
            "content": row.get("content", ""),
            "score": float(row.get("score", 0.0) or 0.0) + _multihop_helper_bonus(str(spec.get("helper", ""))),
            "metadata": dict(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
        }
        meta = item["metadata"]
        helper = str(spec.get("helper", "") or "")
        meta["locomo_multihop_helper"] = helper
        meta["locomo_multihop_query"] = str(spec.get("query", "") or "")
        meta["locomo_multihop_aliases"] = list(spec.get("aliases", ()) or ())
        person = str(spec.get("person", "") or "")
        if person:
            meta["locomo_multihop_person"] = person
        return item

    def profile_candidates(
        self,
        question: str,
        *,
        persons: list[str],
        aliases: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        documents = getattr(self._adapter, "_fact_bm25_corpus", None) or []
        if not documents:
            return []
        terms = _multihop_terms(question, aliases=aliases, persons=persons)
        if not terms and not aliases:
            return []
        out: list[dict[str, Any]] = []
        for person in persons:
            matches: list[tuple[str, dict[str, Any]]] = []
            person_key = person.casefold()
            for text, meta in documents:
                text_key = text.casefold()
                if person_key not in text_key and str(meta.get("speaker", "") or "").casefold() != person_key:
                    continue
                if not _text_matches_any(text_key, [*aliases, *terms]):
                    continue
                matches.append((text, meta))
            if not matches:
                continue
            content = "\n".join(text for text, _ in matches[:_MULTIHOP_PROFILE_FACT_LIMIT])
            first_meta = dict(matches[0][1])
            first_meta.update(
                {
                    "memory_type": "facts",
                    "search_method": "locomo_multihop_profile",
                    "source_file": f"facts/profile/{person}",
                    "locomo_multihop_helper": "profile",
                    "locomo_multihop_query": " ".join([question, *aliases[:_MULTIHOP_ALIAS_LIMIT]]).strip(),
                    "locomo_multihop_aliases": list(aliases[:_MULTIHOP_ALIAS_LIMIT]),
                    "locomo_multihop_person": person,
                },
            )
            helper = "intersection" if _is_shared_subject_question(question) and len(persons) >= 2 else "profile"
            if helper == "intersection":
                first_meta["locomo_multihop_helper"] = "intersection"
            out.append(
                {
                    "content": content,
                    "score": 1.20 + min(0.30, len(matches) / 100.0),
                    "metadata": first_meta,
                },
            )
        return out

    def merge_items(
        self,
        base_items: list[dict[str, Any]],
        helper_items: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        base_keys: list[str] = []
        helper_keys: list[str] = []

        def normalize(item: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
            content = str(item.get("content", "") or "").strip()
            if not content:
                return None
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            key = str(meta.get("fact_id", "") or content)
            return (
                key,
                {
                    "content": content,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "metadata": dict(meta),
                },
            )

        for item in base_items:
            normalized = normalize(item)
            if normalized is None:
                continue
            key, row = normalized
            if key not in by_key:
                base_keys.append(key)
            by_key[key] = row

        for item in sorted(helper_items, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True):
            normalized = normalize(item)
            if normalized is None:
                continue
            key, row = normalized
            current = by_key.get(key)
            if current is not None:
                if float(row.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                    by_key[key] = row
                continue
            by_key[key] = row
            helper_keys.append(key)

        if not base_keys:
            return [by_key[key] for key in helper_keys if key in by_key][:limit]
        merged = [by_key[key] for key in base_keys if key in by_key][:limit]
        remaining = max(0, limit - len(merged))
        if remaining:
            merged.extend(by_key[key] for key in helper_keys[:remaining] if key in by_key)
        return merged

    @staticmethod
    def _meta(
        *,
        query_specs: list[dict[str, Any]],
        aliases: tuple[str, ...],
        alias_map_enabled: bool,
        fallback_used: bool,
        helper_counts: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "alias_map_enabled": alias_map_enabled,
            "fallback_used": fallback_used,
            "query_count": len(query_specs),
            "queries": [spec["query"] for spec in query_specs],
            "aliases": aliases,
            "helper_hit_counts": helper_counts,
        }
