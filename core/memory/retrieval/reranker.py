from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Cross-encoder reranker for hybrid search results."""

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderReranker:
    """Reranker using sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._available = True

    def _ensure_model(self) -> bool:
        if not self._available:
            return False
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder loaded: %s", self._model_name)
            return True
        except Exception:
            logger.warning("Cross-encoder unavailable: %s", self._model_name, exc_info=True)
            self._available = False
            return False

    def _score_sync(self, query: str, texts: list[str]) -> list[float] | None:
        if not self._ensure_model():
            return None
        try:
            pairs = [[query, t] for t in texts]
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]
        except Exception:
            logger.warning("Cross-encoder scoring failed", exc_info=True)
            return None

    def rerank_sync(
        self,
        query: str,
        items: list[dict],
        *,
        text_field: str | Callable[[dict], str] = "content",
        top_k: int = 10,
        min_candidates: int = 2,
    ) -> list[dict]:
        """Synchronous rerank for Legacy RAG paths."""
        if not items:
            return []
        if len(items) < min_candidates:
            return [dict(item) for item in items[:top_k]]

        if callable(text_field):
            texts = [str(text_field(item)) for item in items]
        else:
            texts = [str(item.get(text_field, "")) for item in items]

        scores = self._score_sync(query, texts)
        if scores is None:
            return [dict(item) for item in items[:top_k]]

        scored = list(zip(items, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        result: list[dict] = []
        for item, score in scored[:top_k]:
            row = dict(item)
            row["ce_score"] = score
            row["score"] = score
            row["search_method"] = "cross_encoder"
            result.append(row)
        return result

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_field: str | Callable[[dict], str] = "fact",
        top_k: int = 10,
    ) -> list[dict]:
        """Async rerank for Neo4j hybrid search."""
        if not items:
            return []

        if callable(text_field):
            texts = [str(text_field(item)) for item in items]
        else:
            texts = [str(item.get(text_field, "")) for item in items]

        scores = await asyncio.to_thread(self._score_sync, query, texts)
        if scores is None:
            return [dict(item) for item in items[:top_k]]

        scored = list(zip(items, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        result: list[dict] = []
        for item, score in scored[:top_k]:
            row = dict(item)
            row["ce_score"] = score
            row["score"] = score
            row["search_method"] = "cross_encoder"
            result.append(row)
        return result


_reranker: CrossEncoderReranker | None = None


def get_reranker(model_name: str = _DEFAULT_MODEL) -> CrossEncoderReranker:
    """Get or create singleton reranker instance."""
    global _reranker  # noqa: PLW0603
    if _reranker is None or _reranker._model_name != model_name:
        _reranker = CrossEncoderReranker(model_name)
    return _reranker
