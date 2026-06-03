from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Retrieval pipeline: RRF merge → optional rerank → confidence gate."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.memory.retrieval.confidence_gate import apply_confidence_gate
from core.memory.retrieval.entity import EntityBoostConfig, apply_entity_boost
from core.memory.retrieval.reranker import CrossEncoderReranker, get_reranker
from core.memory.retrieval.rrf import legacy_result_key, rrf_merge
from core.memory.retrieval.temporal import TemporalBoostConfig, apply_temporal_boost

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final pipeline output."""

    items: list[dict[str, Any]]
    abstain: bool = False
    abstain_reason: str = ""


class RetrievalPipeline:
    """Backend-agnostic hybrid retrieval post-processing."""

    def __init__(
        self,
        *,
        reranker: CrossEncoderReranker | None = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> None:
        self._reranker = reranker or get_reranker(cross_encoder_model)

    def run(
        self,
        query: str,
        ranked_lists: list[list[dict[str, Any]]],
        *,
        limit: int = 10,
        pool_k: int = 50,
        rrf_k: int = 60,
        key_fn: Callable[[dict[str, Any]], str] | None = None,
        rerank_enabled: bool = True,
        rerank_text_field: str | Callable[[dict], str] = "content",
        min_candidates_for_rerank: int = 2,
        abstain_on_low_confidence: bool = True,
        confidence_threshold: float = 0.35,
        rrf_confidence_threshold: float = 0.02,
        temporal_boost: TemporalBoostConfig | None = None,
        entity_boost: EntityBoostConfig | None = None,
    ) -> PipelineResult:
        """Merge, rerank, and optionally gate candidates."""
        non_empty = [lst for lst in ranked_lists if lst]
        if not non_empty:
            if abstain_on_low_confidence:
                gated = apply_confidence_gate([], threshold=confidence_threshold)
                return PipelineResult(
                    items=gated.candidates,
                    abstain=gated.abstain,
                    abstain_reason=gated.reason,
                )
            return PipelineResult(items=[])

        merged = rrf_merge(
            non_empty,
            key_fn=key_fn or legacy_result_key,
            k=rrf_k,
            top_k=pool_k,
        )
        candidates = merged[:pool_k]
        if temporal_boost is not None:
            candidates = apply_temporal_boost(query, candidates, temporal_boost)
        if entity_boost is not None:
            candidates = apply_entity_boost(query, candidates, entity_boost)

        used_rerank = False
        if rerank_enabled and len(candidates) >= min_candidates_for_rerank:
            try:
                candidates = self._reranker.rerank_sync(
                    query,
                    candidates,
                    text_field=rerank_text_field,
                    top_k=min(pool_k, max(limit, len(candidates))),
                    min_candidates=min_candidates_for_rerank,
                )
                used_rerank = any(c.get("search_method") == "cross_encoder" for c in candidates)
            except Exception:
                logger.warning("Rerank stage failed; using RRF order", exc_info=True)

        if temporal_boost is not None and used_rerank:
            candidates = apply_temporal_boost(query, candidates, temporal_boost)
        if entity_boost is not None and used_rerank:
            candidates = apply_entity_boost(query, candidates, entity_boost)
        candidates = candidates[:limit]

        if abstain_on_low_confidence:
            threshold = confidence_threshold if used_rerank else rrf_confidence_threshold
            gated = apply_confidence_gate(candidates, threshold=threshold)
            return PipelineResult(
                items=gated.candidates[:limit],
                abstain=gated.abstain,
                abstain_reason=gated.reason,
            )

        return PipelineResult(items=candidates)
