from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding model lifecycle and prioritized embedding generation."""

import logging
import os
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_BATCH_LIMIT = 1000
_embedding_lock = threading.Lock()
_embedding_encode_lock = threading.Lock()
_embedding_model: SentenceTransformer | None = None
_embedding_model_name: str | None = None
_embedding_model_device: str | None = None
EmbeddingPurpose = Literal["document", "query"]
EmbeddingPriority = Literal["interactive", "bulk"]
_priority_condition = threading.Condition(threading.Lock())
_interactive_waiters = 0
_bulk_yield_count = 0


def _get_configured_model_name() -> str:
    """Read embedding model name from config.json, falling back to default."""
    try:
        from core.config import load_config

        config = load_config()
        return config.rag.embedding_model
    except Exception:
        return "intfloat/multilingual-e5-small"


def _get_embedding_prefix_settings() -> tuple[bool, str, str]:
    """Return e5-prefix settings from config, preserving legacy defaults on failure."""
    try:
        from core.config import load_config

        rag = load_config().rag
        return (
            bool(getattr(rag, "embedding_e5_prefix_enabled", False)),
            str(getattr(rag, "embedding_query_prefix", "query: ") or ""),
            str(getattr(rag, "embedding_document_prefix", "passage: ") or ""),
        )
    except Exception:
        return False, "query: ", "passage: "


def _prefix_texts_for_embedding(texts: list[str], *, purpose: EmbeddingPurpose) -> list[str]:
    """Apply configured E5 query/document prefixes to embedding input text."""
    enabled, query_prefix, document_prefix = _get_embedding_prefix_settings()
    if not enabled:
        return texts
    prefix = query_prefix if purpose == "query" else document_prefix
    if not prefix:
        return texts
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in texts]


def _get_embedding_batch_size() -> int:
    try:
        from core.config import load_config

        batch_size = int(getattr(load_config().gpu, "embedding_batch_size", 32))
        return batch_size if batch_size > 0 else 32
    except Exception:
        return 32


def _get_bulk_yield_batches() -> int:
    try:
        from core.config import load_config

        yield_batches = int(getattr(load_config().gpu, "embedding_bulk_yield_batches", 5))
        return yield_batches if yield_batches > 0 else 5
    except Exception:
        return 5


def _construct_embedding_model(resolved_name: str, cache_dir: str, device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(resolved_name, cache_folder=cache_dir, device=device)


def _load_embedding_model_on_device(resolved_name: str, device: str) -> SentenceTransformer:
    global _embedding_model, _embedding_model_device, _embedding_model_name

    from core.infra.gpu import record_component_device
    from core.infra.inference import run_inference
    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading embedding model (singleton): %s on %s", resolved_name, device)
    _embedding_model = run_inference(
        "embedding", device, _construct_embedding_model, resolved_name, str(cache_dir), device
    )
    try:
        from core.config import load_config

        max_seq = int(getattr(load_config().rag, "embedding_max_seq_length", 2048))
    except Exception:
        max_seq = 2048
    model_max_seq = getattr(_embedding_model, "max_seq_length", None)
    if max_seq > 0 and isinstance(model_max_seq, int) and model_max_seq > max_seq:
        _embedding_model.max_seq_length = max_seq
        logger.info("Embedding max_seq_length capped to %d", max_seq)
    _embedding_model_name = resolved_name
    _embedding_model_device = device
    record_component_device("embedding", device)
    logger.info("Embedding model loaded (singleton)")
    return _embedding_model


def _reload_embedding_model_on_device(resolved_name: str, device: str) -> SentenceTransformer:
    with _embedding_lock:
        # Callers that failed together must not each load a model: torch never
        # returns freed arenas to the OS (see get_embedding_model).
        if _embedding_model is not None and (_embedding_model_name, _embedding_model_device) == (resolved_name, device):
            return _embedding_model
        return _load_embedding_model_on_device(resolved_name, device)


def _model_device_type(model: SentenceTransformer) -> str:
    """Return the device a model's weights are on, which picks its inference thread.

    This asks the model rather than module state: a caller can still hold the
    MPS model after another thread has fallen back to CPU, and that model must
    stay on the MPS thread.
    """
    device_type = getattr(getattr(model, "device", None), "type", None)
    return device_type if isinstance(device_type, str) else "cpu"


def _encode_on_inference_thread(model: SentenceTransformer, texts: list[str], **kwargs) -> object:
    from core.infra.inference import run_inference

    return run_inference("embedding", _model_device_type(model), model.encode, texts, **kwargs)


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Return process-level singleton SentenceTransformer model.

    Args:
        model_name: Explicit model name override.  When ``None``,
            the model is resolved from ``config.json``
            (``rag.embedding_model``).

    If the cached model was loaded with a different name, it is
    discarded and reloaded with the requested model.
    """
    global _embedding_model, _embedding_model_name

    resolved_name = model_name or _get_configured_model_name()
    from core.infra.gpu import is_component_degraded, record_gpu_failure, resolve_device

    device = "cpu" if is_component_degraded("embedding") else resolve_device("embedding")

    # Fast path: already loaded with the same model name.
    #
    # The device is deliberately NOT part of this check: ``resolve_device``
    # re-probes CUDA on every call, and a flapping probe (driver stress,
    # sandboxed child) used to discard and reload the model on each flip —
    # torch never returns freed arenas to the OS, so RSS ratcheted up by
    # ~0.5GB per reload (2026-07-17 OOM incident). Device changes now only
    # happen through the explicit GPU-failure fallback below and in
    # ``thread_safe_encode``.
    if _embedding_model is not None and _embedding_model_name == resolved_name:
        return _embedding_model

    with _embedding_lock:
        # Double-check after acquiring lock
        if _embedding_model is not None and _embedding_model_name == resolved_name:
            return _embedding_model

        # Different model requested → discard and reload
        if _embedding_model is not None and _embedding_model_name != resolved_name:
            logger.info(
                "Embedding model changed: %s/%s -> %s/%s; reloading",
                _embedding_model_name,
                _embedding_model_device,
                resolved_name,
                device,
            )
        try:
            return _load_embedding_model_on_device(resolved_name, device)
        except Exception as exc:
            if device != "cpu":
                record_gpu_failure("embedding", exc)
                logger.warning("GPU embedding model load failed, falling back to CPU: %s", exc)
                return _load_embedding_model_on_device(resolved_name, "cpu")
            raise


def _coerce_embeddings(embeddings) -> list[list[float]]:
    import numpy as np

    if isinstance(embeddings, np.ndarray):
        return [emb.tolist() for emb in embeddings]
    return embeddings


def thread_safe_encode(
    texts: list[str],
    *,
    convert_to_numpy: bool = True,
    show_progress_bar: bool = False,
    purpose: EmbeddingPurpose = "document",
    priority: EmbeddingPriority = "interactive",
) -> list[list[float]]:
    """Serialize embedding calls and prioritize interactive work.

    Bulk callers release the lock between configured model batches so
    waiting interactive callers can run first.
    """
    from core.infra.gpu import is_gpu_failure, record_gpu_failure, release_mps_cache
    from core.infra.inference import run_inference

    model = get_embedding_model()
    prefixed_texts = _prefix_texts_for_embedding(texts, purpose=purpose)
    batch_size = _get_embedding_batch_size()
    try:
        return _encode_with_priority(
            model,
            prefixed_texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            priority=priority,
        )
    except Exception as exc:
        if not is_gpu_failure(exc):
            raise
        logger.error("GPU failure detected - falling back to CPU embedding", exc_info=True)
        record_gpu_failure("embedding", exc)
        failed_device = _model_device_type(model)
        cpu_model = _reload_embedding_model_on_device(_get_configured_model_name(), "cpu")
    # Out here the exception's traceback no longer holds the GPU model, so its
    # memory can actually be released.
    del model
    if failed_device == "mps":
        run_inference("embedding", failed_device, release_mps_cache)
    return _encode_with_priority(
        cpu_model,
        prefixed_texts,
        convert_to_numpy=convert_to_numpy,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
        priority=priority,
    )


def _encode_with_priority(
    model: SentenceTransformer,
    texts: list[str],
    *,
    convert_to_numpy: bool,
    show_progress_bar: bool,
    batch_size: int,
    priority: EmbeddingPriority,
) -> list[list[float]]:
    if priority == "interactive":
        with _interactive_native_slot():
            embeddings = _encode_on_inference_thread(
                model,
                texts,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
            )
        return _coerce_embeddings(embeddings)

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        _wait_for_bulk_turn()
        batch = texts[start : start + batch_size]
        with _embedding_encode_lock:
            embeddings = _encode_on_inference_thread(
                model,
                batch,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
            )
        all_embeddings.extend(_coerce_embeddings(embeddings))
        with _priority_condition:
            _priority_condition.notify_all()
    return all_embeddings


class _interactive_native_slot:
    def __enter__(self) -> None:
        global _interactive_waiters
        with _priority_condition:
            _interactive_waiters += 1
            _priority_condition.notify_all()
        _embedding_encode_lock.acquire()
        with _priority_condition:
            _interactive_waiters = max(0, _interactive_waiters - 1)
            _priority_condition.notify_all()

    def __exit__(self, exc_type, exc, tb) -> None:
        _embedding_encode_lock.release()
        with _priority_condition:
            _priority_condition.notify_all()


def _wait_for_bulk_turn() -> None:
    global _bulk_yield_count
    max_yields = _get_bulk_yield_batches()
    with _priority_condition:
        while _interactive_waiters > 0 and _bulk_yield_count < max_yields:
            _bulk_yield_count += 1
            _priority_condition.wait()
        if _interactive_waiters == 0 or _bulk_yield_count >= max_yields:
            _bulk_yield_count = 0


def generate_embeddings(
    texts: list[str],
    *,
    purpose: EmbeddingPurpose = "document",
    priority: EmbeddingPriority = "interactive",
) -> list[list[float]]:
    """Generate embeddings via HTTP server or local model.

    When ``ANIMAWORKS_EMBED_URL`` is set, delegates to the server's
    ``/api/internal/embed`` endpoint.  Otherwise, uses the local
    SentenceTransformer singleton.
    """
    if not texts:
        return []
    from core.memory.rag.endpoints import get_endpoints

    embed_url = get_endpoints().embed_url
    if embed_url:
        return _generate_embeddings_http(texts, embed_url, purpose=purpose, priority=priority)
    return _generate_embeddings_local(texts, purpose=purpose, priority=priority)


_HTTP_CLIENT = None
_HTTP_CLIENT_PID = None


def _shared_http_client():
    """Process-wide authenticated client for batched internal API calls."""
    global _HTTP_CLIENT, _HTTP_CLIENT_PID

    pid = os.getpid()
    if _HTTP_CLIENT is None or pid != _HTTP_CLIENT_PID:
        from core.internal_api import host_api

        _HTTP_CLIENT = host_api.http_client(timeout=180.0)
        _HTTP_CLIENT_PID = pid
    return _HTTP_CLIENT


def _generate_embeddings_http(
    texts: list[str],
    embed_url: str,
    *,
    purpose: EmbeddingPurpose = "document",
    priority: EmbeddingPriority = "interactive",
) -> list[list[float]]:
    """Call server's /api/internal/embed endpoint."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[i : i + _BATCH_LIMIT]
        resp = _shared_http_client().post(
            embed_url,
            json={"texts": batch, "purpose": purpose, "priority": priority},
            # Bulk indexing batches can queue behind interactive embeds on a
            # busy GPU (e.g. right after a fleet restart); 30s was too tight.
            timeout=180.0,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings


def _generate_embeddings_local(
    texts: list[str],
    *,
    purpose: EmbeddingPurpose = "document",
    priority: EmbeddingPriority = "interactive",
) -> list[list[float]]:
    """Use local SentenceTransformer model."""
    return thread_safe_encode(texts, purpose=purpose, priority=priority)


def get_embedding_model_name() -> str:
    """Return the name of the currently loaded (or configured) embedding model."""
    if _embedding_model_name is not None:
        return _embedding_model_name
    return _get_configured_model_name()


def get_embedding_e5_prefix_enabled() -> bool:
    """Return whether configured E5 prefixes are part of the embedding input."""
    enabled, _query_prefix, _document_prefix = _get_embedding_prefix_settings()
    return enabled


def _reset_for_testing() -> None:
    """Reset embedding process state for test isolation."""
    global _embedding_model, _embedding_model_name, _embedding_model_device
    global _interactive_waiters, _bulk_yield_count, _HTTP_CLIENT, _HTTP_CLIENT_PID

    with _embedding_lock:
        _embedding_model = None
        _embedding_model_name = None
        _embedding_model_device = None
        _HTTP_CLIENT = None
        _HTTP_CLIENT_PID = None
    with _priority_condition:
        _interactive_waiters = 0
        _bulk_yield_count = 0
        _priority_condition.notify_all()
    from core.infra.gpu import reset_gpu_status_for_testing
    from core.infra.inference import reset_inference_for_testing

    reset_gpu_status_for_testing()
    reset_inference_for_testing()
