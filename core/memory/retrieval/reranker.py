from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Cross-encoder reranker for hybrid search results.

When ``ANIMAWORKS_RERANK_URL`` is set (child processes), scoring delegates
to the server's ``/api/internal/rerank`` endpoint instead of loading the
CrossEncoder model locally.  HTTP failures skip rerank (original order is
kept) rather than falling back to a local model load.
"""

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_BATCH_LIMIT = 1000
_HTTP_TIMEOUT = 180.0


def _construct_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def _predict(model, pairs: list[list[str]]):
    return model.predict(pairs, show_progress_bar=False)


class CrossEncoderReranker:
    """Reranker using sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        started = time.perf_counter()
        self._model_name = model_name
        self._model = None
        self._available = True
        from core.memory.rag.endpoints import get_endpoints

        self._rerank_url = get_endpoints().rerank_url
        if self._rerank_url:
            self._device = "cpu"
        else:
            from core.infra.gpu import is_component_degraded, resolve_device

            self._device = "cpu" if is_component_degraded("reranker") else resolve_device("reranker")
        self._lock = threading.Lock()
        logger.info(
            "Cross-encoder client initialized: model=%s device=%s http=%s elapsed=%.3fs",
            self._model_name,
            self._device,
            bool(self._rerank_url),
            time.perf_counter() - started,
        )

    def _ensure_model(self) -> bool:
        if not self._available:
            return False
        if self._model is not None:
            return True
        with self._lock:
            return self._load_model_locked()

    def _load_model_locked(self) -> bool:
        # Double-check under the lock: concurrent first-use callers used to
        # each load their own CrossEncoder (observed as triple simultaneous
        # "Loading weights" bursts per process), leaking the extra copies.
        if not self._available:
            return False
        if self._model is not None:
            return True
        started = time.perf_counter()
        from core.infra.gpu import record_component_device
        from core.infra.inference import run_inference

        try:
            self._model = run_inference(
                "reranker", self._device, _construct_cross_encoder, self._model_name, self._device
            )
            record_component_device("reranker", self._device)
            logger.info(
                "Cross-encoder loaded: model=%s device=%s elapsed=%.3fs",
                self._model_name,
                self._device,
                time.perf_counter() - started,
            )
            return True
        except Exception as exc:
            if self._device != "cpu":
                from core.infra.gpu import record_gpu_failure

                record_gpu_failure("reranker", exc)
                try:
                    self._device = "cpu"
                    self._model = run_inference("reranker", "cpu", _construct_cross_encoder, self._model_name, "cpu")
                    record_component_device("reranker", "cpu")
                    logger.warning("Cross-encoder GPU load failed; falling back to CPU: %s", exc)
                    return True
                except Exception:
                    logger.warning("Cross-encoder CPU fallback unavailable: %s", self._model_name, exc_info=True)
            else:
                logger.warning("Cross-encoder unavailable: %s", self._model_name, exc_info=True)
            self._available = False
            return False

    def _score_local(self, query: str, texts: list[str]) -> list[float] | None:
        started = time.perf_counter()
        if not self._ensure_model():
            return None
        from core.infra.inference import run_inference

        scoring_device = self._device
        try:
            pairs = [[query, t] for t in texts]
            lock_started = time.perf_counter()
            with self._lock:
                lock_wait = time.perf_counter() - lock_started
                predict_started = time.perf_counter()
                scoring_device = self._device
                scores = run_inference("reranker", scoring_device, _predict, self._model, pairs)
                predict_elapsed = time.perf_counter() - predict_started
            result = [float(s) for s in scores]
            logger.info(
                "Cross-encoder local score: query_chars=%d documents=%d elapsed=%.3fs lock_wait=%.3fs predict=%.3fs",
                len(query),
                len(texts),
                time.perf_counter() - started,
                lock_wait,
                predict_elapsed,
            )
            return result
        except Exception as exc:
            from core.infra.gpu import is_gpu_failure, record_component_device, record_gpu_failure, release_mps_cache

            if scoring_device != "cpu" and is_gpu_failure(exc):
                logger.error("GPU failure detected - falling back to CPU reranker", exc_info=True)
                record_gpu_failure("reranker", exc)
                try:
                    pairs = [[query, t] for t in texts]
                    with self._lock:
                        # Another caller may have fallen back while this one waited.
                        failed_device = self._device
                        if failed_device != "cpu":
                            self._device = "cpu"
                            self._model = run_inference(
                                "reranker", "cpu", _construct_cross_encoder, self._model_name, "cpu"
                            )
                            record_component_device("reranker", "cpu")
                            if failed_device == "mps":
                                run_inference("reranker", failed_device, release_mps_cache)
                        scores = run_inference("reranker", "cpu", _predict, self._model, pairs)
                    return [float(s) for s in scores]
                except Exception:
                    logger.warning("Cross-encoder CPU fallback scoring failed", exc_info=True)
            logger.warning("Cross-encoder scoring failed", exc_info=True)
            return None

    def _score_http(self, query: str, texts: list[str], rerank_url: str) -> list[float] | None:
        """Score via server's /api/internal/rerank. On failure return None (skip)."""
        from core.memory.rag.embedding import _shared_http_client

        started = time.perf_counter()
        all_scores: list[float] = []
        try:
            for i in range(0, len(texts), _BATCH_LIMIT):
                batch = texts[i : i + _BATCH_LIMIT]
                request_started = time.perf_counter()
                resp = _shared_http_client().post(
                    rerank_url,
                    json={"query": query, "documents": batch},
                    timeout=_HTTP_TIMEOUT,
                )
                resp.raise_for_status()
                scores = resp.json()["scores"]
                if len(scores) != len(batch):
                    logger.warning(
                        "HTTP rerank returned %d scores for %d documents; skipping",
                        len(scores),
                        len(batch),
                    )
                    return None
                all_scores.extend(float(s) for s in scores)
                logger.info(
                    "Cross-encoder HTTP batch: query_chars=%d documents=%d elapsed=%.3fs",
                    len(query),
                    len(batch),
                    time.perf_counter() - request_started,
                )
        except Exception:
            logger.warning(
                "HTTP rerank failed (url=%s); skipping rerank, keeping original order",
                rerank_url,
                exc_info=True,
            )
            return None
        logger.info(
            "Cross-encoder HTTP score: query_chars=%d documents=%d elapsed=%.3fs",
            len(query),
            len(texts),
            time.perf_counter() - started,
        )
        return all_scores

    def _score_sync(self, query: str, texts: list[str]) -> list[float] | None:
        """Score documents; HTTP-delegate when ANIMAWORKS_RERANK_URL is set."""
        if not texts:
            return []
        if self._rerank_url:
            return self._score_http(query, texts, self._rerank_url)
        return self._score_local(query, texts)

    def score_sync(self, query: str, texts: list[str]) -> list[float] | None:
        """Public scoring entry used by the central /api/internal/rerank endpoint.

        Always uses the local model (server process). Child processes should
        call the HTTP endpoint rather than this method.
        """
        if not texts:
            return []
        return self._score_local(query, texts)

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


_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(model_name: str = _DEFAULT_MODEL) -> CrossEncoderReranker:
    """Get or create singleton reranker instance."""
    global _reranker  # noqa: PLW0603
    with _reranker_lock:
        if _reranker is None or _reranker._model_name != model_name:
            _reranker = CrossEncoderReranker(model_name)
        return _reranker


def _reset_for_testing() -> None:
    """Reset singleton for test isolation."""
    global _reranker  # noqa: PLW0603
    with _reranker_lock:
        _reranker = None
