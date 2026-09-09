from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified Legacy memory retrieval orchestration.

This module keeps Legacy retrieval policy in one place while reusing the
existing RAG search helpers for vector, graph, keyword, and activity sources.
"""

import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from core.memory.bm25 import search_activity_log
except ImportError:
    search_activity_log = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_TOOL_ALL_SCOPES: tuple[str, ...] = (
    "facts",
    "episodes",
    "knowledge",
    "procedures",
    "common_knowledge",
    "skills",
    "conversation_summary",
    "activity_log",
)
_EXPLICIT_SCOPES = _TOOL_ALL_SCOPES

_ITERATIVE_TRIGGERS = frozenset({"task", "tool"})
_ENGLISH_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
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
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
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
_JAPANESE_QUERY_WORDS_RE = re.compile(r"(?:どちら|どなた|いかが|どんな|どれ|どこ|いつ|だれ|誰|なぜ|どう|どの|なん|何)")
_JAPANESE_AUXILIARIES_RE = re.compile(
    r"(?:について|における|に関する|による|という|でした|ません|ました|ます|です|ください)"
)
_JAPANESE_PARTICLES_RE = re.compile(r"[はがをにへでとのもや]")
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+'-]*|\d+(?:[.-]\d+)*|[\u3040-\u30ff\u3400-\u9fffー]+")
_QUOTED_PHRASE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|「([^」\n]+)」|『([^』\n]+)』|(?<!\w)\'([^\'\n]+)\'(?!\w)')
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+&'.-]*|[A-Z]{2,}[A-Za-z0-9+&'.-]*)"
    r"(?:\s+(?:of|the|and|for|to|by|in|on|at|with|"
    r"[A-Z][A-Za-z0-9+&'.-]*|[A-Z]{2,}[A-Za-z0-9+&'.-]*)){0,4}"
)


def build_iterative_queries(query: str) -> list[str]:
    """Build deterministic, LLM-free fallback queries for a second retrieval round.

    The first query removes common English stopwords and Japanese question,
    auxiliary, and particle forms.  The second retains quoted phrases and
    English proper-name-like phrases.  Empty queries and exact duplicates of
    the original are omitted.
    """
    original = unicodedata.normalize("NFKC", str(query or "")).strip()
    if not original:
        return []

    keyword_parts: list[str] = []
    for token in _QUERY_TOKEN_RE.findall(original):
        if token[0].isascii():
            if token.casefold() not in _ENGLISH_QUERY_STOPWORDS:
                keyword_parts.append(token)
            continue
        cleaned = _JAPANESE_QUERY_WORDS_RE.sub(" ", token)
        cleaned = _JAPANESE_AUXILIARIES_RE.sub(" ", cleaned)
        cleaned = _JAPANESE_PARTICLES_RE.sub(" ", cleaned)
        keyword_parts.extend(part for part in cleaned.split() if part and part != "か")
    keyword_query = " ".join(keyword_parts)

    entity_spans: list[tuple[int, int, str]] = []
    quoted_spans: list[tuple[int, int]] = []
    for match in _QUOTED_PHRASE_RE.finditer(original):
        phrase = next((group for group in match.groups() if group is not None), "").strip()
        if phrase:
            entity_spans.append((match.start(), match.end(), phrase))
            quoted_spans.append((match.start(), match.end()))
    for match in _CAPITALIZED_PHRASE_RE.finditer(original):
        if any(start <= match.start() and match.end() <= end for start, end in quoted_spans):
            continue
        phrase = match.group(0).strip()
        if phrase.casefold() in _ENGLISH_QUERY_STOPWORDS:
            continue
        entity_spans.append((match.start(), match.end(), phrase))

    entity_parts: list[str] = []
    seen_entities: set[str] = set()
    for _start, _end, phrase in sorted(entity_spans, key=lambda item: (item[0], item[1])):
        key = phrase.casefold()
        if key not in seen_entities:
            entity_parts.append(phrase)
            seen_entities.add(key)
    entity_query = " ".join(entity_parts)

    transformed: list[str] = []
    seen = {original.casefold()}
    for candidate in (keyword_query, entity_query):
        candidate = " ".join(candidate.split()).strip()
        key = candidate.casefold()
        if candidate and key not in seen:
            transformed.append(candidate)
            seen.add(key)
    return transformed


@dataclass(frozen=True)
class TriggerPolicy:
    """Legacy retrieval policy selected by recall trigger."""

    pool_k: int
    rerank: bool
    scopes: tuple[str, ...]
    confidence_gate: bool = True


TRIGGER_POLICIES: dict[str, TriggerPolicy] = {
    "chat": TriggerPolicy(
        pool_k=50,
        rerank=True,
        scopes=("facts", "episodes", "knowledge", "procedures", "activity_log"),
    ),
    "inbox": TriggerPolicy(
        pool_k=30,
        rerank=True,
        scopes=("facts", "episodes", "activity_log"),
    ),
    "heartbeat": TriggerPolicy(
        pool_k=20,
        rerank=False,
        scopes=("episodes", "activity_log"),
    ),
    "task": TriggerPolicy(
        pool_k=30,
        rerank=True,
        scopes=("facts", "procedures", "knowledge"),
    ),
    "cron": TriggerPolicy(
        pool_k=30,
        rerank=True,
        scopes=("facts", "episodes", "knowledge", "activity_log"),
    ),
    "tool": TriggerPolicy(
        pool_k=50,
        rerank=True,
        scopes=_TOOL_ALL_SCOPES,
    ),
}


def _explicit_time_range(*, time_start: str | None, time_end: str | None) -> Any | None:
    """Convert schema ISO bounds into the same inclusive range used by query extraction."""
    if not time_start and not time_end:
        return None

    from core.memory.retrieval.time_expr import TimeRange

    start = _parse_time_bound(time_start, end_of_day=False)
    end = _parse_time_bound(time_end, end_of_day=True)
    if start is None and end is None:
        return None
    if start is not None and end is not None and start > end:
        start, end = end, start
    return TimeRange(start=start, end=end)


def _parse_time_bound(value: str | None, *, end_of_day: bool) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed_date = date.fromisoformat(text)
    except ValueError:
        normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    return datetime.combine(parsed_date, time.max if end_of_day else time.min)


class UnifiedMemorySearch:
    """Legacy-only retrieval orchestrator shared by tools, backend, and priming."""

    def __init__(
        self,
        anima_dir: Path,
        *,
        common_knowledge_dir: Path | None = None,
        common_skills_dir: Path | None = None,
        rag_search: Any | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        if common_knowledge_dir is None or common_skills_dir is None:
            inferred_data_dir = anima_dir.parent.parent if len(anima_dir.parents) >= 2 else anima_dir.parent
            if (inferred_data_dir / "common_knowledge").is_dir() or (inferred_data_dir / "common_skills").is_dir():
                data_dir = inferred_data_dir
            else:
                try:
                    from core.paths import get_data_dir

                    data_dir = get_data_dir()
                except Exception:
                    data_dir = inferred_data_dir
            common_knowledge_dir = common_knowledge_dir or (data_dir / "common_knowledge")
            common_skills_dir = common_skills_dir or (data_dir / "common_skills")
        self._common_knowledge_dir = common_knowledge_dir
        self._common_skills_dir = common_skills_dir
        self._rag_search = rag_search
        self._last_search_meta: dict[str, object] = {}
        # Thread-local so search_many can run queries in parallel without
        # clobbering per-query abstain metadata across worker threads.
        self._search_meta_tls = threading.local()

    def _set_last_search_meta(self, meta: dict[str, object]) -> None:
        self._last_search_meta = meta
        self._search_meta_tls.meta = meta

    @property
    def last_search_meta(self) -> dict[str, object]:
        """Metadata from the latest pipeline run (this thread)."""
        meta = getattr(self._search_meta_tls, "meta", None)
        if isinstance(meta, dict):
            return dict(meta)
        return dict(self._last_search_meta)

    def search(
        self,
        query: str,
        *,
        scope: str,
        limit: int,
        trigger: str,
        offset: int = 0,
        min_score: float = 0.0,
        time_start: str | None = None,
        time_end: str | None = None,
        scope_override: tuple[str, ...] | None = None,
        pipeline_settings: dict[str, object] | None = None,
        temporal_boost: Any | None = None,
        entity_boost: Any | None = None,
        reference_time: Any | None = None,
        _allow_iterative: bool = True,
        skip_bm25_validation: bool = False,
        _access_batch: Any | None = None,
        _flush_access_batch: bool = True,
    ) -> list[dict[str, Any]]:
        """Search Legacy memories through a trigger-aware shared policy."""
        search_started = perf_counter()
        limit = max(0, int(limit))
        if limit <= 0:
            self._set_last_search_meta({"abstain": False, "abstain_reason": ""})
            return []
        offset = max(0, min(int(offset), 50)) if trigger == "tool" else 0
        policy = self._policy_for(trigger)
        scopes = self._target_scopes(scope, policy, scope_override=scope_override)
        rag = self._ensure_rag_search()
        settings = dict(pipeline_settings or rag._load_rag_pipeline_settings())
        iterative_enabled = bool(settings.get("iterative_retrieval_enabled", True))
        try:
            iterative_min_results = max(0, int(settings.get("iterative_min_results", 2) or 0))
        except (TypeError, ValueError):
            iterative_min_results = 2
        pool_k = max(int(settings.get("rerank_candidate_pool", policy.pool_k) or policy.pool_k), offset + limit)
        if pipeline_settings is None:
            pool_k = max(policy.pool_k, offset + limit)
        rerank_enabled = bool(settings.get("rerank_enabled", policy.rerank)) and policy.rerank
        confidence_enabled = bool(settings.get("abstain_on_low_confidence", policy.confidence_gate))
        confidence_enabled = confidence_enabled and policy.confidence_gate

        from core.memory.retrieval.query_expansion import (
            coerce_reference_time,
            expand_query,
            filter_ranked_lists_by_time_hint,
        )

        coerced_reference_time = coerce_reference_time(reference_time)
        expanded = expand_query(query, reference_time=coerced_reference_time)
        # Sparse (BM25/keyword) query carries expanded ISO dates and lowercased
        # tokens; dense (vector/graph) query keeps natural text plus quoted
        # phrases only. See F19.
        search_query = expanded.search_text or query
        dense_query = expanded.dense_text or query
        time_hint_start = time_start or expanded.time_hint_start
        time_hint_end = time_end or expanded.time_hint_end
        iterative_entity_boost = entity_boost
        iterative_temporal_boost = temporal_boost
        if entity_boost is None:
            entity_boost = rag._build_entity_boost_config(dense_query, settings)
        if temporal_boost is None:
            temporal_boost = self._build_temporal_boost_config(
                query,
                settings,
                time_start=time_start,
                time_end=time_end,
                reference_time=coerced_reference_time,
            )
        access_boost = None
        access_boost_builder = getattr(rag, "_build_access_boost_config", None)
        if callable(access_boost_builder):
            access_boost = access_boost_builder(settings)

        try:
            indexer = rag._get_indexer()
        except Exception:
            logger.debug("Unified search indexer init failed", exc_info=True)
            indexer = None

        from core.memory.rag.retriever import AccessBatch

        access_batch = _access_batch or AccessBatch()

        embedding = None
        embedding_started = perf_counter()
        generate_embeddings = getattr(indexer, "_generate_embeddings", None)
        if callable(generate_embeddings):
            try:
                embedding = generate_embeddings([dense_query], purpose="query")[0]
            except Exception:
                logger.debug("Unified query embedding prefetch failed", exc_info=True)
        logger.info(
            "Unified search embedding: scope=%s query_chars=%d elapsed=%.3fs available=%s",
            scope,
            len(query),
            perf_counter() - embedding_started,
            embedding is not None,
        )

        collect_started = perf_counter()
        ranked_lists = self._collect_ranked_lists(
            rag,
            dense_query=dense_query,
            sparse_query=search_query,
            scopes=scopes,
            pool_k=pool_k,
            entity_boost=entity_boost,
            embedding=embedding,
            indexer=indexer,
            access_batch=access_batch,
            skip_bm25_validation=skip_bm25_validation,
            time_start=time_start,
            time_end=time_end,
        )
        logger.info(
            "Unified search retrieval: scope=%s query_chars=%d elapsed=%.3fs lists=%d candidates=%d",
            scope,
            len(query),
            perf_counter() - collect_started,
            len(ranked_lists),
            sum(len(items) for items in ranked_lists),
        )
        if _flush_access_batch:
            flush_started = perf_counter()
            access_batch.flush(getattr(indexer, "vector_store", None))
            logger.info(
                "Unified search access flush: scope=%s elapsed=%.3fs",
                scope,
                perf_counter() - flush_started,
            )
        ranked_lists = filter_ranked_lists_by_time_hint(
            ranked_lists,
            time_hint_start=time_hint_start,
            time_hint_end=time_hint_end,
        )

        if not ranked_lists:
            self._set_last_search_meta(
                {
                    "abstain": False,
                    "abstain_reason": "",
                    "query_expansion": {
                        "original": expanded.original,
                        "search_text": search_query,
                        "time_hint_start": time_hint_start,
                        "time_hint_end": time_hint_end,
                    },
                }
            )
            items = self._maybe_iterative_search(
                [],
                query=query,
                scope=scope,
                limit=limit,
                trigger=trigger,
                offset=offset,
                min_score=min_score,
                time_start=time_start,
                time_end=time_end,
                scope_override=scope_override,
                pipeline_settings=pipeline_settings,
                temporal_boost=iterative_temporal_boost,
                entity_boost=iterative_entity_boost,
                reference_time=coerced_reference_time,
                enabled=iterative_enabled,
                min_results=iterative_min_results,
                allow_iterative=_allow_iterative,
            )
            logger.info(
                "Unified search complete: scope=%s mode=empty elapsed=%.3fs results=%d",
                scope,
                perf_counter() - search_started,
                len(items),
            )
            return items
        if self._is_keyword_only_fallback(ranked_lists):
            self._set_last_search_meta(
                {
                    "abstain": False,
                    "abstain_reason": "",
                    "query_expansion": {
                        "original": expanded.original,
                        "search_text": search_query,
                        "time_hint_start": time_hint_start,
                        "time_hint_end": time_hint_end,
                    },
                }
            )
            items = ranked_lists[0]
            if min_score > 0.0:
                items = [item for item in items if float(item.get("score", 0.0) or 0.0) >= min_score]
            items = self._soft_source_collapse(items)[offset : offset + limit]
            items = self._maybe_iterative_search(
                items,
                query=query,
                scope=scope,
                limit=limit,
                trigger=trigger,
                offset=offset,
                min_score=min_score,
                time_start=time_start,
                time_end=time_end,
                scope_override=scope_override,
                pipeline_settings=pipeline_settings,
                temporal_boost=iterative_temporal_boost,
                entity_boost=iterative_entity_boost,
                reference_time=coerced_reference_time,
                enabled=iterative_enabled,
                min_results=iterative_min_results,
                allow_iterative=_allow_iterative,
            )
            items = self._soft_source_collapse(items)[:limit]
            logger.info(
                "Unified search complete: scope=%s mode=keyword-only elapsed=%.3fs results=%d",
                scope,
                perf_counter() - search_started,
                len(items),
            )
            return items

        from core.memory.retrieval.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            cross_encoder_model=str(settings.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")),
        )
        pipeline_started = perf_counter()
        result = pipeline.run(
            dense_query,
            ranked_lists,
            limit=pool_k,
            pool_k=pool_k,
            rerank_enabled=rerank_enabled,
            abstain_on_low_confidence=confidence_enabled,
            confidence_threshold=float(settings.get("confidence_threshold", 0.35)),
            rrf_confidence_threshold=float(settings.get("rrf_confidence_threshold", 0.02)),
            temporal_boost=temporal_boost,
            entity_boost=entity_boost,
            access_boost=access_boost,
        )
        logger.info(
            "Unified search pipeline: scope=%s elapsed=%.3fs rerank_enabled=%s",
            scope,
            perf_counter() - pipeline_started,
            rerank_enabled,
        )
        self._set_last_search_meta(
            {
                "abstain": result.abstain,
                "abstain_reason": result.abstain_reason,
                "query_expansion": {
                    "original": expanded.original,
                    "search_text": search_query,
                    "time_hint_start": time_hint_start,
                    "time_hint_end": time_hint_end,
                },
            }
        )

        items = result.items
        # Only apply min_score to reranked results: after cross-encoder rerank
        # the score is a CE logit that min_score is calibrated against. In RRF
        # order the score is a tiny fusion value (~0.03 max) that min_score
        # (default 0.3) would wipe out entirely; the confidence gate already
        # guards quality there. See F2.
        if min_score > 0.0 and self._rerank_was_applied(items):
            items = [item for item in items if float(item.get("score", 0.0) or 0.0) >= min_score]
        items = self._soft_source_collapse(items)[offset : offset + limit]
        items = self._maybe_iterative_search(
            items,
            query=query,
            scope=scope,
            limit=limit,
            trigger=trigger,
            offset=offset,
            min_score=min_score,
            time_start=time_start,
            time_end=time_end,
            scope_override=scope_override,
            pipeline_settings=pipeline_settings,
            temporal_boost=iterative_temporal_boost,
            entity_boost=iterative_entity_boost,
            reference_time=coerced_reference_time,
            enabled=iterative_enabled,
            min_results=iterative_min_results,
            allow_iterative=_allow_iterative,
        )
        items = self._soft_source_collapse(items)[:limit]
        logger.info(
            "Unified search complete: scope=%s mode=hybrid elapsed=%.3fs results=%d",
            scope,
            perf_counter() - search_started,
            len(items),
        )
        return items

    def _maybe_iterative_search(
        self,
        first_round: list[dict[str, Any]],
        *,
        query: str,
        scope: str,
        limit: int,
        trigger: str,
        offset: int,
        min_score: float,
        time_start: str | None,
        time_end: str | None,
        scope_override: tuple[str, ...] | None,
        pipeline_settings: dict[str, object] | None,
        temporal_boost: Any | None,
        entity_boost: Any | None,
        reference_time: datetime | None,
        enabled: bool,
        min_results: int,
        allow_iterative: bool,
    ) -> list[dict[str, Any]]:
        """Run and merge an optional, strictly non-recursive second round."""
        normalized_trigger = (trigger or "chat").strip().lower()
        if (
            not allow_iterative
            or not enabled
            or normalized_trigger not in _ITERATIVE_TRIGGERS
            or len(first_round) >= min_results
        ):
            return first_round

        queries = self._build_iterative_queries(query)
        if not queries:
            return first_round

        first_meta = self.last_search_meta
        second_round = self.search_many(
            queries,
            scope=scope,
            limit=limit,
            trigger=trigger,
            offset=offset,
            min_score=min_score,
            time_start=time_start,
            time_end=time_end,
            scope_override=scope_override,
            pipeline_settings=pipeline_settings,
            temporal_boost=temporal_boost,
            entity_boost=entity_boost,
            reference_time=reference_time,
            _allow_iterative=False,
        )
        second_meta = self.last_search_meta

        best: dict[str, dict[str, Any]] = {}
        for item in first_round:
            key = self._result_key(item)
            current = best.get(key)
            if current is None or float(item.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                best[key] = item
        for item in second_round:
            marked = dict(item)
            marked["retrieval_round"] = 2
            key = self._result_key(marked)
            current = best.get(key)
            if current is None or float(marked.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                best[key] = marked

        merged = self._soft_source_collapse(
            sorted(
                best.values(),
                key=lambda item: float(item.get("score", 0.0) or 0.0),
                reverse=True,
            )
        )[:limit]
        self._set_last_search_meta(
            {
                **first_meta,
                "abstain": (bool(first_meta.get("abstain", False)) or bool(second_meta.get("abstain", False)))
                and not merged,
                "abstain_reason": (
                    str(second_meta.get("abstain_reason", "") or first_meta.get("abstain_reason", ""))
                    if not merged
                    else ""
                ),
                "iterative_retrieval": {
                    "attempted": True,
                    "queries": queries,
                    "second_round_results": len(second_round),
                },
            }
        )
        return merged

    def _build_iterative_queries(self, query: str) -> list[str]:
        """Build keyword/entity transformations plus one registry alias variant."""
        queries = build_iterative_queries(query)
        alias_query = self._alias_substitution_query(query)
        seen = {str(query or "").strip().casefold(), *(item.casefold() for item in queries)}
        if alias_query and alias_query.casefold() not in seen:
            queries.append(alias_query)
        return queries[:3]

    def _alias_substitution_query(self, query: str) -> str | None:
        """Replace matched registry surfaces with deterministic alternate aliases."""
        try:
            from core.memory.retrieval.entity import load_entity_alias_index

            index = load_entity_alias_index(self._anima_dir)
        except Exception:
            logger.debug("Iterative retrieval alias index load failed", exc_info=True)
            return None
        if index is None:
            return None

        original = str(query or "").strip()
        normalized = original.casefold()
        replacements: list[tuple[str, str, str]] = []
        used_owners: set[str] = set()
        for surface, owner in sorted(index.alias_owner.items(), key=lambda item: (-len(item[0]), item[0])):
            if owner in used_owners or len(surface) < 2 or surface not in normalized:
                continue
            alternatives = sorted(
                (synonym for synonym in index.synonyms.get(owner, ()) if synonym.casefold() != surface.casefold()),
                key=lambda value: (value.casefold(), value),
            )
            if not alternatives:
                continue
            replacements.append((surface, alternatives[0], owner))
            used_owners.add(owner)

        transformed = original
        for surface, replacement, _owner in replacements:
            transformed = re.sub(re.escape(surface), replacement, transformed, count=1, flags=re.IGNORECASE)
        transformed = " ".join(transformed.split()).strip()
        if not transformed or transformed.casefold() == original.casefold():
            return None
        return transformed

    @staticmethod
    def _build_temporal_boost_config(
        query: str,
        settings: dict[str, object],
        *,
        time_start: str | None,
        time_end: str | None,
        reference_time: datetime | None,
    ) -> Any | None:
        """Build automatic temporal ranking config from explicit or query time intent."""
        if not bool(settings.get("temporal_boost_enabled", True)):
            return None

        from core.memory.retrieval.temporal import TemporalBoostConfig
        from core.memory.retrieval.time_expr import TimeRange, extract_time_range

        now = reference_time or datetime.now()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        explicit = _explicit_time_range(time_start=time_start, time_end=time_end)
        resolved = explicit or extract_time_range(query, now=now)
        if resolved is None:
            return None
        return TemporalBoostConfig(
            enabled=True,
            boost=float(settings.get("temporal_boost", 0.05) or 0.0),
            max_boost=float(settings.get("temporal_boost_max", 0.10) or 0.0),
            category=None,
            time_range=TimeRange(
                start=resolved.start,
                end=resolved.end,
                recency=resolved.recency,
            ),
            recency=resolved.recency,
            half_life_days=float(settings.get("temporal_half_life_days", 7.0) or 7.0),
            now=now,
        )

    @staticmethod
    def _rerank_was_applied(items: list[dict[str, Any]]) -> bool:
        """Match pipeline.py used_rerank: any cross-encoder row means reranked."""
        return any(str(item.get("search_method", "")) == "cross_encoder" for item in items)

    def search_many(
        self,
        queries: list[str],
        *,
        scope: str,
        limit: int,
        trigger: str,
        offset: int = 0,
        min_score: float = 0.0,
        time_start: str | None = None,
        time_end: str | None = None,
        scope_override: tuple[str, ...] | None = None,
        pipeline_settings: dict[str, object] | None = None,
        temporal_boost: Any | None = None,
        entity_boost: Any | None = None,
        reference_time: Any | None = None,
        _allow_iterative: bool = False,
        rerank_after_merge: bool = False,
        skip_bm25_validation: bool = False,
    ) -> list[dict[str, Any]]:
        """Run multiple queries and merge by stable document identity."""
        from core.memory.rag.retriever import AccessBatch

        search_many_started = perf_counter()
        merge_rerank_enabled = rerank_after_merge and len(queries) > 1
        merged_pipeline_settings = pipeline_settings
        if merge_rerank_enabled:
            policy = self._policy_for(trigger)
            rag = self._ensure_rag_search()
            merged_pipeline_settings = dict(pipeline_settings or rag._load_rag_pipeline_settings())
            if pipeline_settings is None:
                merged_pipeline_settings["rerank_candidate_pool"] = max(policy.pool_k, limit)
            merge_rerank_enabled = policy.rerank and bool(merged_pipeline_settings.get("rerank_enabled", policy.rerank))
            merged_pipeline_settings["rerank_enabled"] = False
        search_kwargs: dict[str, Any] = {
            "scope": scope,
            "limit": limit,
            "trigger": trigger,
            "offset": offset,
            "min_score": min_score,
            "time_start": time_start,
            "time_end": time_end,
            "scope_override": scope_override,
            "pipeline_settings": merged_pipeline_settings,
            "temporal_boost": temporal_boost,
            "entity_boost": entity_boost,
            "reference_time": reference_time,
            "_allow_iterative": _allow_iterative,
            "skip_bm25_validation": skip_bm25_validation,
        }

        def _run_one(query: str) -> tuple[list[dict[str, Any]], dict[str, object], AccessBatch]:
            started = perf_counter()
            query_access_batch = AccessBatch()
            results = self.search(
                query,
                **search_kwargs,
                _access_batch=query_access_batch,
                _flush_access_batch=False,
            )
            logger.info(
                "Unified search_many query complete: scope=%s query_chars=%d elapsed=%.3fs results=%d",
                scope,
                len(query),
                perf_counter() - started,
                len(results),
            )
            return results, self.last_search_meta, query_access_batch

        if len(queries) <= 1:
            per_query = [_run_one(query) for query in queries]
        else:
            # Parallelize independent queries (priming C/F can issue up to 3).
            with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                per_query = list(pool.map(_run_one, queries))

        if queries:
            access_batch = AccessBatch()
            for _results, _meta, query_access_batch in per_query:
                access_batch.absorb(query_access_batch)
            flush_started = perf_counter()
            try:
                indexer = self._ensure_rag_search()._get_indexer()
            except Exception:
                logger.debug("Unified search_many indexer init failed", exc_info=True)
                indexer = None
            access_batch.flush(getattr(indexer, "vector_store", None))
            logger.info(
                "Unified search_many access flush: scope=%s queries=%d elapsed=%.3fs",
                scope,
                len(queries),
                perf_counter() - flush_started,
            )

        best: dict[str, dict[str, Any]] = {}
        saw_abstain = False
        abstain_reason = ""
        for results, meta, _access_batch in per_query:
            if bool(meta.get("abstain", False)):
                saw_abstain = True
                abstain_reason = str(meta.get("abstain_reason", "") or abstain_reason)
            for item in results:
                key = self._result_key(item)
                existing = best.get(key)
                if existing is None or float(item.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    best[key] = item

        merged = sorted(best.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        if merge_rerank_enabled and len(merged) >= 2:
            from core.memory.retrieval.reranker import get_reranker

            rerank_started = perf_counter()
            model_name = str(
                (merged_pipeline_settings or {}).get(
                    "cross_encoder_model",
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                )
            )
            merged = get_reranker(model_name).rerank_sync(queries[0], merged, top_k=len(merged))
            if min_score > 0.0 and self._rerank_was_applied(merged):
                merged = [item for item in merged if float(item.get("score", 0.0) or 0.0) >= min_score]
            logger.info(
                "Unified search_many merged rerank: scope=%s query_chars=%d candidates=%d elapsed=%.3fs results=%d",
                scope,
                len(queries[0]),
                len(best),
                perf_counter() - rerank_started,
                len(merged),
            )
        merged = self._soft_source_collapse(merged)[:limit]
        self._set_last_search_meta(
            {
                "abstain": saw_abstain and not merged,
                "abstain_reason": abstain_reason if saw_abstain and not merged else "",
            }
        )
        logger.info(
            "Unified search_many complete: scope=%s queries=%d elapsed=%.3fs results=%d",
            scope,
            len(queries),
            perf_counter() - search_many_started,
            len(merged),
        )
        return merged

    @staticmethod
    def _soft_source_collapse(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prefer one chunk per Markdown source, then append deferred chunks."""
        first: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            source = str(item.get("source_file", "") or item.get("source", ""))
            memory_type = str(item.get("memory_type", "")).lower()
            try:
                total_chunks = int(item.get("total_chunks", 1) or 1)
            except (TypeError, ValueError):
                total_chunks = 1
            collapsible = (
                total_chunks > 1 and source.lower().endswith(".md") and memory_type not in {"activity_log", "facts"}
            )
            if collapsible and source in seen:
                deferred.append(item)
            else:
                first.append(item)
                if collapsible:
                    seen.add(source)
        return first + deferred

    def _ensure_rag_search(self) -> Any:
        if self._rag_search is None:
            from core.memory.rag_search import RAGMemorySearch

            self._rag_search = RAGMemorySearch(
                self._anima_dir,
                self._common_knowledge_dir,
                self._common_skills_dir,
            )
        return self._rag_search

    def _policy_for(self, trigger: str) -> TriggerPolicy:
        normalized = (trigger or "chat").strip().lower()
        policy = TRIGGER_POLICIES.get(normalized)
        if policy is None:
            logger.debug("Unknown memory search trigger %r; using chat policy", trigger)
            return TRIGGER_POLICIES["chat"]
        return policy

    def _target_scopes(
        self,
        scope: str,
        policy: TriggerPolicy,
        *,
        scope_override: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if scope_override is not None:
            return tuple(s for s in scope_override if s in _EXPLICIT_SCOPES)
        normalized = (scope or "all").strip().lower()
        if normalized == "all":
            return policy.scopes
        if normalized in _EXPLICIT_SCOPES:
            return (normalized,)
        logger.debug("Unknown memory search scope %r; using knowledge", scope)
        return ("knowledge",)

    def _collect_ranked_lists(
        self,
        rag: Any,
        *,
        dense_query: str,
        sparse_query: str,
        scopes: tuple[str, ...],
        pool_k: int,
        entity_boost: Any | None,
        embedding: list[float] | None,
        indexer: Any | None,
        access_batch: Any,
        skip_bm25_validation: bool,
        time_start: str | None,
        time_end: str | None,
    ) -> list[list[dict[str, Any]]]:
        # Vector and graph retrieval use the dense query; BM25-backed
        # activity_log and keyword fallbacks use the sparse query. See F19.
        ranked_lists: list[list[dict[str, Any]]] = []
        vector_scopes = [scope for scope in scopes if scope != "activity_log"]
        remaining_vector_scopes = vector_scopes
        if vector_scopes:
            first_scope = vector_scopes[0]
            first_hits = self._vector_hits(
                rag,
                dense_query,
                first_scope,
                pool_k,
                entity_boost=entity_boost,
                embedding=embedding,
                access_batch=access_batch,
            )
            if first_hits:
                ranked_lists.append(first_hits)
            remaining_vector_scopes = vector_scopes[1:]

        vector_groups: list[tuple[list[str], bool]] = []
        graph_enabled = bool(rag._load_rag_pipeline_settings().get("enable_spreading_activation", True))
        grouped: set[str] = set()
        for scope in remaining_vector_scopes:
            if scope in grouped:
                continue
            group = [scope]
            if scope == "knowledge" and "common_knowledge" in remaining_vector_scopes:
                group.append("common_knowledge")
            grouped.update(group)
            vector_groups.append((group, graph_enabled and "episodes" in group))
        if graph_enabled and "episodes" in scopes and "episodes" not in remaining_vector_scopes:
            vector_groups.append(([], True))

        def _run_vector_group(group: list[str], include_graph: bool):
            hits = {
                scope: self._vector_hits(
                    rag,
                    dense_query,
                    scope,
                    pool_k,
                    entity_boost=entity_boost,
                    embedding=embedding,
                    access_batch=access_batch,
                )
                for scope in group
            }
            graph_hits = (
                self._graph_hits(
                    rag,
                    dense_query,
                    pool_k,
                    embedding=embedding,
                    indexer=indexer,
                    access_batch=access_batch,
                )
                if include_graph
                else []
            )
            return hits, graph_hits

        with ThreadPoolExecutor(max_workers=len(vector_groups) + 2) as pool:
            activity_future = (
                pool.submit(
                    search_activity_log,
                    self._anima_dir,
                    sparse_query,
                    top_k=pool_k,
                    offset=0,
                    time_start=time_start,
                    time_end=time_end,
                )
                if "activity_log" in scopes and search_activity_log is not None
                else None
            )
            keyword_future = pool.submit(
                self._keyword_hits,
                rag,
                sparse_query,
                vector_scopes,
                pool_k,
                entity_boost=entity_boost,
                skip_bm25_validation=skip_bm25_validation,
            )
            vector_futures = [pool.submit(_run_vector_group, *group) for group in vector_groups]
            vector_hits: dict[str, list[dict[str, Any]]] = {}
            graph_hits: list[dict[str, Any]] = []
            for future in vector_futures:
                group_hits, group_graph_hits = future.result()
                vector_hits.update(group_hits)
                if group_graph_hits:
                    graph_hits = group_graph_hits
            for vector_scope in remaining_vector_scopes:
                hits = vector_hits.get(vector_scope, [])
                if hits:
                    ranked_lists.append(hits)

            if graph_hits:
                ranked_lists.append(graph_hits)

            if activity_future is not None:
                try:
                    activity_hits = activity_future.result()
                    if activity_hits:
                        ranked_lists.append(activity_hits)
                except Exception:
                    logger.debug("Unified activity_log search failed", exc_info=True)

            keyword_hits = keyword_future.result()
        if keyword_hits:
            ranked_lists.append(keyword_hits)
        return ranked_lists

    def _vector_hits(
        self,
        rag: Any,
        query: str,
        scope: str,
        pool_k: int,
        *,
        entity_boost: Any | None,
        embedding: list[float] | None,
        access_batch: Any,
    ) -> list[dict[str, Any]]:
        try:
            return rag._vector_search_primary(
                query,
                scope,
                offset=0,
                knowledge_dir=self._anima_dir / "knowledge",
                result_limit=pool_k,
                entity_boost=entity_boost,
                embedding=embedding,
                access_batch=access_batch,
            )
        except Exception:
            logger.debug("Unified vector search failed for scope=%s", scope, exc_info=True)
            return []

    def _graph_hits(
        self,
        rag: Any,
        query: str,
        pool_k: int,
        *,
        embedding: list[float] | None,
        indexer: Any | None,
        access_batch: Any,
    ) -> list[dict[str, Any]]:
        try:
            return rag._graph_episodes_search(
                query,
                pool_k,
                self._anima_dir / "knowledge",
                embedding=embedding,
                indexer=indexer,
                access_batch=access_batch,
            )
        except Exception:
            logger.debug("Unified graph episode search failed", exc_info=True)
            return []

    def _keyword_hits(
        self,
        rag: Any,
        query: str,
        scopes: list[str],
        pool_k: int,
        *,
        entity_boost: Any | None,
        skip_bm25_validation: bool,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for scope in scopes:
            self._merge_keyword_scope(
                rag,
                query,
                scope,
                pool_k,
                entity_boost=entity_boost,
                skip_bm25_validation=skip_bm25_validation,
                merged=merged,
            )
        return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[:pool_k]

    def _merge_keyword_scope(
        self,
        rag: Any,
        query: str,
        scope: str,
        pool_k: int,
        *,
        entity_boost: Any | None,
        skip_bm25_validation: bool,
        merged: dict[str, dict[str, Any]],
    ) -> None:
        try:
            hits = rag._keyword_search_fallback(
                query,
                scope,
                0,
                knowledge_dir=self._anima_dir / "knowledge",
                episodes_dir=self._anima_dir / "episodes",
                procedures_dir=self._anima_dir / "procedures",
                common_knowledge_dir=self._common_knowledge_dir,
                result_limit=pool_k,
                entity_boost=entity_boost,
                skip_bm25_validation=skip_bm25_validation,
            )
        except Exception:
            logger.debug("Unified keyword search failed for scope=%s", scope, exc_info=True)
            return
        for hit in hits:
            key = self._result_key(hit)
            current = merged.get(key)
            if current is None or float(hit.get("score", 0.0) or 0.0) > float(current.get("score", 0.0) or 0.0):
                merged[key] = hit

    @staticmethod
    def _result_key(item: dict[str, Any]) -> str:
        doc_id = str(item.get("doc_id", "") or "")
        if doc_id:
            return doc_id
        source = str(item.get("source_file", "") or item.get("source", "") or "")
        chunk = str(item.get("chunk_index", "") or "")
        fact_id = str(item.get("fact_id", "") or "")
        if source or chunk or fact_id:
            return f"{source}#{chunk}:{fact_id}"
        return str(hash((item.get("content", ""), item.get("memory_type", ""))))

    @staticmethod
    def _is_keyword_only_fallback(ranked_lists: list[list[dict[str, Any]]]) -> bool:
        if len(ranked_lists) != 1 or not ranked_lists[0]:
            return False
        return all(str(item.get("search_method", "")).startswith("keyword") for item in ranked_lists[0])
