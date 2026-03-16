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

    from core.memory.rag.store import ChromaVectorStore

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 1000

_lock = threading.Lock()
_vector_stores: dict[str | None, ChromaVectorStore | None] = {}
_embedding_model: SentenceTransformer | None = None
_embedding_model_name: str | None = None
_init_failed: bool = False


def get_vector_store(anima_name: str | None = None) -> ChromaVectorStore | None:
    """Return process-level singleton ChromaVectorStore per anima.

    Args:
        anima_name: Anima name for per-anima DB isolation.
            When ``None``, uses the legacy shared directory.

    Returns:
        ChromaVectorStore instance, or ``None`` if ChromaDB failed to
        initialize (e.g., Python 3.14 + pydantic.v1 incompatibility).
    """
    global _init_failed

    # Fast path: already known to be broken — skip without locking
    if _init_failed:
        return None

    if anima_name not in _vector_stores:
        with _lock:
            if _init_failed:
                return None
            if anima_name not in _vector_stores:
                try:
                    from core.memory.rag.store import ChromaVectorStore

                    if anima_name:
                        from core.paths import get_anima_vectordb_dir

                        persist_dir = get_anima_vectordb_dir(anima_name)
                    else:
                        persist_dir = None  # ChromaVectorStore defaults to ~/.animaworks/vectordb
                    _vector_stores[anima_name] = ChromaVectorStore(persist_dir=persist_dir)
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
        device = "cuda" if _use_gpu else "cpu"
        logger.info("Loading embedding model (singleton): %s on %s", resolved_name, device)
        try:
            _embedding_model = SentenceTransformer(resolved_name, cache_folder=str(cache_dir), device=device)
        except Exception as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                logger.warning("GPU OOM loading embedding model, falling back to CPU: %s", e)
                _embedding_model = SentenceTransformer(resolved_name, cache_folder=str(cache_dir), device="cpu")
            else:
                raise
        _embedding_model_name = resolved_name
        logger.info("Embedding model loaded (singleton)")
    return _embedding_model


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
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def get_embedding_model_name() -> str:
    """Return the name of the currently loaded (or configured) embedding model."""
    if _embedding_model_name is not None:
        return _embedding_model_name
    return _get_configured_model_name()


def _reset_for_testing():
    """Reset singletons for test isolation."""
    global _embedding_model, _embedding_model_name, _init_failed
    with _lock:
        _vector_stores.clear()
        _embedding_model = None
        _embedding_model_name = None
        _init_failed = False
