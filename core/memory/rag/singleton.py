from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-level singletons for RAG components.

Ensures ChromaVectorStore and SentenceTransformer embedding model
are initialized only once per process, avoiding costly repeated
model loading (~6 seconds per initialization).
"""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_vector_stores: dict[str | None, object] = {}
_embedding_model = None
_embedding_model_name: str | None = None


def get_vector_store(anima_name: str | None = None):
    """Return process-level singleton ChromaVectorStore per anima.

    Args:
        anima_name: Anima name for per-anima DB isolation.
            When ``None``, uses the legacy shared directory.
    """
    if anima_name not in _vector_stores:
        with _lock:
            if anima_name not in _vector_stores:
                from core.memory.rag.store import ChromaVectorStore
                if anima_name:
                    from core.paths import get_anima_vectordb_dir
                    persist_dir = get_anima_vectordb_dir(anima_name)
                else:
                    persist_dir = None  # ChromaVectorStore defaults to ~/.animaworks/vectordb
                _vector_stores[anima_name] = ChromaVectorStore(persist_dir=persist_dir)
    return _vector_stores[anima_name]


def _get_configured_model_name() -> str:
    """Read embedding model name from config.json, falling back to default."""
    try:
        from core.config import load_config
        config = load_config()
        return config.rag.embedding_model
    except Exception:
        return "intfloat/multilingual-e5-small"


def get_embedding_model(model_name: str | None = None):
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
                _embedding_model_name, resolved_name,
            )

        from sentence_transformers import SentenceTransformer
        from core.paths import get_data_dir
        cache_dir = get_data_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading embedding model (singleton): %s", resolved_name)
        _embedding_model = SentenceTransformer(
            resolved_name, cache_folder=str(cache_dir)
        )
        _embedding_model_name = resolved_name
        logger.info("Embedding model loaded (singleton)")
    return _embedding_model


def get_embedding_dimension() -> int:
    """Return the dimensionality of the current embedding model.

    Loads the model if not yet initialized, then queries it for
    the sentence embedding dimension.
    """
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


def get_embedding_model_name() -> str:
    """Return the name of the currently loaded (or configured) embedding model."""
    if _embedding_model_name is not None:
        return _embedding_model_name
    return _get_configured_model_name()


def _reset_for_testing():
    """Reset singletons for test isolation."""
    global _embedding_model, _embedding_model_name
    with _lock:
        _vector_stores.clear()
        _embedding_model = None
        _embedding_model_name = None
