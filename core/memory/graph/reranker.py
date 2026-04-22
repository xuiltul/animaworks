# Copyright 2026 AnimaWorks contributors — Apache-2.0
"""Cross-encoder reranker for hybrid search results."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


# ── CrossEncoderReranker ────────────────────────────────────


class CrossEncoderReranker:
    """Reranker using sentence-transformers cross-encoder.

    Lazily loads the model on first use. Falls back gracefully
    if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._available = True

    def _ensure_model(self) -> bool:
        """Load model if not already loaded. Returns True if available."""
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
            logger.warning(
                "Cross-encoder unavailable: %s",
                self._model_name,
                exc_info=True,
            )
            self._available = False
            return False

    def _score_sync(self, query: str, texts: list[str]) -> list[float]:
        """Score query-text pairs synchronously."""
        if not self._ensure_model():
            return [0.0] * len(texts)
        try:
            pairs = [[query, t] for t in texts]
            scores = self._model.predict(pairs)
            return [float(s) for s in scores]
        except Exception:
            logger.warning("Cross-encoder scoring failed", exc_info=True)
            return [0.0] * len(texts)

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_field: str = "fact",
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank items by cross-encoder score.

        Args:
            query: Search query.
            items: Candidate items with text_field.
            text_field: Key containing text to score against query.
            top_k: Max results to return.

        Returns:
            Items sorted by cross-encoder score, with 'ce_score' added.
            Falls back to input order if model unavailable.
        """
        if not items:
            return []

        texts = [str(item.get(text_field, "")) for item in items]

        scores = await asyncio.to_thread(self._score_sync, query, texts)

        scored = list(zip(items, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for item, score in scored[:top_k]:
            d = dict(item)
            d["ce_score"] = score
            result.append(d)

        return result


# ── Singleton ───────────────────────────────────────────────

_reranker: CrossEncoderReranker | None = None


def get_reranker(model_name: str = _DEFAULT_MODEL) -> CrossEncoderReranker:
    """Get or create singleton reranker instance."""
    global _reranker  # noqa: PLW0603
    if _reranker is None or _reranker._model_name != model_name:
        _reranker = CrossEncoderReranker(model_name)
    return _reranker
