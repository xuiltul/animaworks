from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-level singletons for RAG components.

Ensures ChromaVectorStore and SentenceTransformer embedding model
are initialized only once per process, avoiding costly repeated
model loading (~6 seconds per initialization).

When ``ANIMAWORKS_EMBED_URL`` is set (child processes), embedding
generation delegates to the server's ``/api/internal/embed`` endpoint
via HTTP, eliminating per-process GPU model loading.
"""

import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from core.memory.rag.http_store import HttpVectorStore
    from core.memory.rag.store import VectorStore

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 1000

_lock = threading.Lock()
_native_ops_lock = threading.Lock()
_vector_stores: dict[str | None, VectorStore | None] = {}
_http_stores: dict[tuple[str, str | None], HttpVectorStore] = {}
_embedding_model: SentenceTransformer | None = None
_embedding_model_name: str | None = None
_init_failed: bool = False
_direct_disabled_warned: bool = False
_cuda_unavailable_warned: bool = False


def _get_http_store(base_url: str, anima_name: str | None) -> HttpVectorStore:
    """Return cached HttpVectorStore for the given base_url and anima_name."""
    normalized_url = base_url.rstrip("/")
    key = (normalized_url, anima_name)
    if key not in _http_stores:
        with _lock:
            if key not in _http_stores:
                from core.memory.rag.http_store import HttpVectorStore

                _http_stores[key] = HttpVectorStore(base_url=normalized_url, anima_name=anima_name)
    return _http_stores[key]


def get_vector_store(anima_name: str | None = None) -> VectorStore | None:
    """Return process-level singleton VectorStore per anima.

    When ``ANIMAWORKS_VECTOR_URL`` is set, delegates to the server's
    vector API via HttpVectorStore. Otherwise uses ChromaVectorStore
    with local ChromaDB.

    Args:
        anima_name: Anima name for per-anima DB isolation.
            When ``None``, uses the legacy shared directory.

    Returns:
        VectorStore instance (ChromaVectorStore or HttpVectorStore),
        or ``None`` if ChromaDB failed to initialize
        (e.g., Python 3.14 + pydantic.v1 incompatibility).
    """
    global _direct_disabled_warned, _init_failed

    vector_url = os.environ.get("ANIMAWORKS_VECTOR_URL")
    if vector_url:
        return _get_http_store(vector_url, anima_name)

    from core.memory.rag.direct_access import direct_chroma_allowed

    if not direct_chroma_allowed():
        if not _direct_disabled_warned:
            logger.warning("Vector store unavailable: direct ChromaDB access is disabled outside the vector worker")
            _direct_disabled_warned = True
        return None

    # Fast path: already known to be broken — skip without locking
    if _init_failed:
        return None

    if anima_name not in _vector_stores:
        with _lock:
            if _init_failed:
                return None
            if anima_name not in _vector_stores:
                try:
                    from core.memory.rag.store import create_chroma_vector_store

                    if anima_name:
                        from core.paths import get_anima_vectordb_dir

                        persist_dir = get_anima_vectordb_dir(anima_name)
                    else:
                        persist_dir = None  # ChromaVectorStore defaults to ~/.animaworks/vectordb
                    store = create_chroma_vector_store(persist_dir=persist_dir)
                    store.anima_name = anima_name
                    _vector_stores[anima_name] = store
                except Exception as exc:
                    _init_failed = True
                    import sys

                    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                    logger.warning(
                        "ChromaDB initialization failed (Python %s). "
                        "RAG features (semantic search, vector indexing) will be disabled. "
                        "Other memory features continue to work normally. "
                        "Error: %s",
                        py_ver,
                        exc,
                    )
                    if sys.version_info >= (3, 14):
                        logger.warning(
                            "This is likely caused by a known pydantic.v1 compatibility issue "
                            "with Python 3.14+. See: https://github.com/chroma-core/chroma/issues/5996"
                        )
                    return None
    return _vector_stores.get(anima_name)


def _get_configured_model_name() -> str:
    """Read embedding model name from config.json, falling back to default."""
    try:
        from core.config import load_config

        config = load_config()
        return config.rag.embedding_model
    except Exception:
        return "intfloat/multilingual-e5-small"


def _cuda_available_safely() -> bool:
    """Return whether CUDA can be initialized without raising.

    Some hosts expose CUDA libraries while the installed driver/GPU pair is
    incompatible.  In that state PyTorch can raise ``cudaGetDeviceCount``
    during lazy CUDA initialization.  Treat that as "CUDA unavailable" so RAG
    embeddings continue on CPU instead of surfacing a 500 from the internal
    embedding endpoint.
    """
    global _cuda_unavailable_warned

    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:
        if not _cuda_unavailable_warned:
            logger.warning("CUDA unavailable for embedding model; using CPU: %s", exc)
            _cuda_unavailable_warned = True
        return False


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

    # Fast path: already loaded with the same model name
    if _embedding_model is not None and _embedding_model_name == resolved_name:
        return _embedding_model

    with _lock:
        # Double-check after acquiring lock
        if _embedding_model is not None and _embedding_model_name == resolved_name:
            return _embedding_model

        # Different model requested → discard and reload
        if _embedding_model is not None and _embedding_model_name != resolved_name:
            logger.info(
                "Embedding model changed: %s → %s; reloading",
                _embedding_model_name,
                resolved_name,
            )

        from sentence_transformers import SentenceTransformer

        from core.paths import get_data_dir

        cache_dir = get_data_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from core.config import load_config

            _use_gpu = load_config().rag.use_gpu
        except Exception:
            _use_gpu = False
        device = "cuda" if _use_gpu and _cuda_available_safely() else "cpu"
        logger.info("Loading embedding model (singleton): %s on %s", resolved_name, device)
        try:
            _embedding_model = SentenceTransformer(resolved_name, cache_folder=str(cache_dir), device=device)
        except Exception as e:
            if device == "cuda":
                logger.warning("GPU embedding model load failed, falling back to CPU: %s", e)
                _embedding_model = SentenceTransformer(resolved_name, cache_folder=str(cache_dir), device="cpu")
            else:
                raise
        _embedding_model_name = resolved_name
        logger.info("Embedding model loaded (singleton)")
    return _embedding_model


def thread_safe_encode(
    texts: list[str],
    *,
    convert_to_numpy: bool = True,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """Serialize SentenceTransformer.encode() via _native_ops_lock.

    PyTorch models and ChromaDB Rust bindings both use native code
    that can SEGV when run concurrently under glibc 2.43+.  All native
    operations share a single lock to prevent parallel native execution.
    """
    import numpy as np

    model = get_embedding_model()
    with _native_ops_lock:
        embeddings = model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
        )
    if isinstance(embeddings, np.ndarray):
        return [emb.tolist() for emb in embeddings]
    return embeddings


def native_ops_lock() -> threading.Lock:
    """Return the global lock for native operations.

    Used by internal API endpoints to serialize ChromaDB Rust calls.
    """
    return _native_ops_lock


def reset_vector_store(anima_name: str | None = None) -> None:
    """Drop cached vector-store clients for runtime repair.

    ``anima_name`` targets one per-anima store.  ``None`` resets the
    legacy/shared store keyed by ``None``.  Any cached HTTP client is
    closed when possible so callers can reconnect after server-side
    repair.
    """
    global _init_failed

    with _lock:
        store = _vector_stores.pop(anima_name, None)
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close vector store for %s", anima_name, exc_info=True)

        http_keys = [key for key in _http_stores if key[1] == anima_name]
        for key in http_keys:
            store = _http_stores.pop(key, None)
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close vector store for %s", anima_name, exc_info=True)
        _init_failed = False


def get_embedding_dimension() -> int:
    """Return the dimensionality of the current embedding model.

    Loads the model if not yet initialized, then queries it for
    the sentence embedding dimension.
    """
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError("Embedding model did not report a dimension")
    return dim


# ── Unified embedding generation ─────────────────────────────────


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via HTTP server or local model.

    When ``ANIMAWORKS_EMBED_URL`` is set, delegates to the server's
    ``/api/internal/embed`` endpoint.  Otherwise, uses the local
    SentenceTransformer singleton.
    """
    if not texts:
        return []
    embed_url = os.environ.get("ANIMAWORKS_EMBED_URL")
    if embed_url:
        return _generate_embeddings_http(texts, embed_url)
    return _generate_embeddings_local(texts)


def _generate_embeddings_http(texts: list[str], embed_url: str) -> list[list[float]]:
    """Call server's /api/internal/embed endpoint."""
    import httpx

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[i : i + _BATCH_LIMIT]
        resp = httpx.post(
            embed_url,
            json={"texts": batch},
            timeout=30.0,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings


def _generate_embeddings_local(texts: list[str]) -> list[list[float]]:
    """Use local SentenceTransformer model."""
    return thread_safe_encode(texts)


def get_embedding_model_name() -> str:
    """Return the name of the currently loaded (or configured) embedding model."""
    if _embedding_model_name is not None:
        return _embedding_model_name
    return _get_configured_model_name()


def _reset_for_testing():
    """Reset singletons for test isolation."""
    global _cuda_unavailable_warned, _direct_disabled_warned, _embedding_model, _embedding_model_name, _init_failed
    with _lock:
        _vector_stores.clear()
        _http_stores.clear()
        _embedding_model = None
        _embedding_model_name = None
        _init_failed = False
        _direct_disabled_warned = False
        _cuda_unavailable_warned = False
