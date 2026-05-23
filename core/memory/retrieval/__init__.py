from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Backend-agnostic memory retrieval pipeline (RRF, rerank, confidence gate)."""

from core.memory.retrieval.confidence_gate import GateResult, apply_confidence_gate
from core.memory.retrieval.pipeline import PipelineResult, RetrievalPipeline
from core.memory.retrieval.reranker import CrossEncoderReranker, get_reranker
from core.memory.retrieval.rrf import legacy_result_key, reciprocal_rank_fusion, rrf_merge
from core.memory.retrieval.types import RetrievalCandidate

__all__ = [
    "CrossEncoderReranker",
    "GateResult",
    "PipelineResult",
    "RetrievalCandidate",
    "RetrievalPipeline",
    "apply_confidence_gate",
    "get_reranker",
    "legacy_result_key",
    "reciprocal_rank_fusion",
    "rrf_merge",
]
