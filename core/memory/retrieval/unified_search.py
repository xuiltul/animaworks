from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified Legacy memory retrieval orchestration.

This module keeps Legacy retrieval policy in one place while reusing the
existing RAG search helpers for vector, graph, keyword, and activity sources.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
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

    @property
    def last_search_meta(self) -> dict[str, object]:
        """Metadata from the latest pipeline run."""
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
    ) -> list[dict[str, Any]]:
        """Search Legacy memories through a trigger-aware shared policy."""
        limit = max(0, int(limit))
        if limit <= 0:
            self._last_search_meta = {"abstain": False, "abstain_reason": ""}
            return []
        offset = max(0, min(int(offset), 50)) if trigger == "tool" else 0
        policy = self._policy_for(trigger)
        scopes = self._target_scopes(scope, policy, scope_override=scope_override)
        rag = self._ensure_rag_search()
        settings = dict(pipeline_settings or rag._load_rag_pipeline_settings())
        pool_k = max(int(settings.get("rerank_candidate_pool", policy.pool_k) or policy.pool_k), limit)
        if pipeline_settings is None:
            pool_k = max(policy.pool_k, limit)
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
            rag._get_indexer()
        except Exception:
            logger.debug("Unified search indexer init failed", exc_info=True)

        ranked_lists = self._collect_ranked_lists(
            rag,
            dense_query=dense_query,
            sparse_query=search_query,
            scopes=scopes,
            pool_k=pool_k,
            entity_boost=entity_boost,
        )
        ranked_lists = filter_ranked_lists_by_time_hint(
            ranked_lists,
            time_hint_start=time_hint_start,
            time_hint_end=time_hint_end,
        )

        if not ranked_lists:
            self._last_search_meta = {
                "abstain": False,
                "abstain_reason": "",
                "query_expansion": {
                    "original": expanded.original,
                    "search_text": search_query,
                    "time_hint_start": time_hint_start,
                    "time_hint_end": time_hint_end,
                },
            }
            return []
        if self._is_keyword_only_fallback(ranked_lists):
            self._last_search_meta = {
                "abstain": False,
                "abstain_reason": "",
                "query_expansion": {
                    "original": expanded.original,
                    "search_text": search_query,
                    "time_hint_start": time_hint_start,
                    "time_hint_end": time_hint_end,
                },
            }
            items = ranked_lists[0][offset : offset + limit]
            if min_score > 0.0:
                items = [item for item in items if float(item.get("score", 0.0) or 0.0) >= min_score]
            return items

        from core.memory.retrieval.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            cross_encoder_model=str(settings.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")),
        )
        result = pipeline.run(
            dense_query,
            ranked_lists,
            limit=offset + limit,
            pool_k=pool_k,
            rerank_enabled=rerank_enabled,
            abstain_on_low_confidence=confidence_enabled,
            confidence_threshold=float(settings.get("confidence_threshold", 0.35)),
            rrf_confidence_threshold=float(settings.get("rrf_confidence_threshold", 0.02)),
            temporal_boost=temporal_boost,
            entity_boost=entity_boost,
            access_boost=access_boost,
        )
        self._last_search_meta = {
            "abstain": result.abstain,
            "abstain_reason": result.abstain_reason,
            "query_expansion": {
                "original": expanded.original,
                "search_text": search_query,
                "time_hint_start": time_hint_start,
                "time_hint_end": time_hint_end,
            },
        }

        items = result.items[offset : offset + limit]
        # Only apply min_score to reranked results: after cross-encoder rerank
        # the score is a CE logit that min_score is calibrated against. In RRF
        # order the score is a tiny fusion value (~0.03 max) that min_score
        # (default 0.3) would wipe out entirely; the confidence gate already
        # guards quality there. See F2.
        if min_score > 0.0 and self._rerank_was_applied(items):
            items = [item for item in items if float(item.get("score", 0.0) or 0.0) >= min_score]
        return items

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
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Run multiple queries and merge by stable document identity."""
        best: dict[str, dict[str, Any]] = {}
        saw_abstain = False
        abstain_reason = ""
        for query in queries:
            results = self.search(
                query,
                scope=scope,
                limit=limit,
                trigger=trigger,
                min_score=min_score,
            )
            meta = self.last_search_meta
            if bool(meta.get("abstain", False)):
                saw_abstain = True
                abstain_reason = str(meta.get("abstain_reason", "") or abstain_reason)
            for item in results:
                key = self._result_key(item)
                existing = best.get(key)
                if existing is None or float(item.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    best[key] = item

        merged = sorted(best.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[:limit]
        self._last_search_meta = {
            "abstain": saw_abstain and not merged,
            "abstain_reason": abstain_reason if saw_abstain and not merged else "",
        }
        return merged

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
    ) -> list[list[dict[str, Any]]]:
        # Vector and graph retrieval use the dense query; BM25-backed
        # activity_log and keyword fallbacks use the sparse query. See F19.
        ranked_lists: list[list[dict[str, Any]]] = []
        vector_scopes = [scope for scope in scopes if scope != "activity_log"]
        for vector_scope in vector_scopes:
            hits = self._vector_hits(rag, dense_query, vector_scope, pool_k, entity_boost=entity_boost)
            if hits:
                ranked_lists.append(hits)

        if "episodes" in scopes:
            graph_hits = self._graph_hits(rag, dense_query, pool_k)
            if graph_hits:
                ranked_lists.append(graph_hits)

        if "activity_log" in scopes and search_activity_log is not None:
            try:
                activity_hits = search_activity_log(
                    self._anima_dir,
                    sparse_query,
                    top_k=pool_k,
                    offset=0,
                )
                if activity_hits:
                    ranked_lists.append(activity_hits)
            except Exception:
                logger.debug("Unified activity_log search failed", exc_info=True)

        keyword_hits = self._keyword_hits(rag, sparse_query, vector_scopes, pool_k, entity_boost=entity_boost)
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
    ) -> list[dict[str, Any]]:
        try:
            return rag._vector_search_primary(
                query,
                scope,
                offset=0,
                knowledge_dir=self._anima_dir / "knowledge",
                result_limit=pool_k,
                entity_boost=entity_boost,
            )
        except Exception:
            logger.debug("Unified vector search failed for scope=%s", scope, exc_info=True)
            return []

    def _graph_hits(self, rag: Any, query: str, pool_k: int) -> list[dict[str, Any]]:
        try:
            return rag._graph_episodes_search(
                query,
                pool_k,
                self._anima_dir / "knowledge",
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
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for scope in scopes:
            self._merge_keyword_scope(
                rag,
                query,
                scope,
                pool_k,
                entity_boost=entity_boost,
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
