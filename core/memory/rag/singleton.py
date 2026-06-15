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
from typing import TYPE_CHECKING, Literal

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
_embedding_model_device: str | None = None
_init_failed: bool = False
_direct_disabled_warned: bool = False

EmbeddingPurpose = Literal["document", "query"]
EmbeddingPriority = Literal["interactive", "bulk"]

_priority_condition = threading.Condition(threading.Lock())
_interactive_waiters = 0
_bulk_yield_count = 0


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
                # Importing the ChromaDB store is an environment-level concern
                # (e.g. the known Python 3.14 + pydantic.v1 incompatibility). A
                # failure here means ChromaDB can never work in this process, so
                # latch the global flag and disable RAG everywhere.
                try:
                    from core.memory.rag.store import create_chroma_vector_store
                except Exception as exc:
                    _init_failed = True
                    import sys

                    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                    logger.warning(
                        "ChromaDB import failed (Python %s). "
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

                # Opening one anima's on-disk DB is a per-anima concern. A corrupt
                # chroma.sqlite3 (e.g. a schema-less stub left by an interrupted
                # repair: "no such table: tenants/databases") must NOT latch the
                # global flag — doing so would take vector writes down for every
                # other anima too. Return None for this anima only; a subsequent
                # repair/reset lets it retry without poisoning the worker.
                try:
                    if anima_name:
                        from core.paths import get_anima_vectordb_dir

                        persist_dir = get_anima_vectordb_dir(anima_name)
                    else:
                        persist_dir = None  # ChromaVectorStore defaults to ~/.animaworks/vectordb
                    store = create_chroma_vector_store(persist_dir=persist_dir, anima_name=anima_name)
                    _vector_stores[anima_name] = store
                except Exception as exc:
                    logger.warning(
                        "Vector store init failed for %s; RAG disabled for this anima only "
                        "(other animas unaffected). Error: %s",
                        anima_name or "shared",
                        exc,
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


def _load_embedding_model_on_device(resolved_name: str, device: str) -> SentenceTransformer:
    global _embedding_model, _embedding_model_device, _embedding_model_name

    from sentence_transformers import SentenceTransformer

    from core.gpu import record_component_device
    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading embedding model (singleton): %s on %s", resolved_name, device)
    _embedding_model = SentenceTransformer(resolved_name, cache_folder=str(cache_dir), device=device)
    _embedding_model_name = resolved_name
    _embedding_model_device = device
    record_component_device("embedding", device)
    logger.info("Embedding model loaded (singleton)")
    return _embedding_model


def _reload_embedding_model_on_device(resolved_name: str, device: str) -> SentenceTransformer:
    with _lock:
        return _load_embedding_model_on_device(resolved_name, device)


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
    from core.gpu import is_component_degraded, record_gpu_failure, resolve_device

    device = "cpu" if is_component_degraded("embedding") else resolve_device("embedding")

    # Fast path: already loaded with the same model name
    if _embedding_model is not None and _embedding_model_name == resolved_name and _embedding_model_device == device:
        return _embedding_model

    with _lock:
        # Double-check after acquiring lock
        if (
            _embedding_model is not None
            and _embedding_model_name == resolved_name
            and _embedding_model_device == device
        ):
            return _embedding_model

        # Different model requested → discard and reload
        if _embedding_model is not None and (
            _embedding_model_name != resolved_name or _embedding_model_device != device
        ):
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
            if device == "cuda":
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
    """Serialize SentenceTransformer.encode() via _native_ops_lock.

    PyTorch models and ChromaDB Rust bindings both use native code
    that can SEGV when run concurrently under glibc 2.43+.  All native
    operations share a single lock to prevent parallel native execution.
    Bulk callers release the lock between configured model batches so
    waiting interactive callers can run first.
    """
    from core.gpu import is_cuda_failure, record_gpu_failure

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
        if not is_cuda_failure(exc):
            raise
        logger.error("GPU failure detected - falling back to CPU embedding", exc_info=True)
        record_gpu_failure("embedding", exc)
        cpu_model = _reload_embedding_model_on_device(_get_configured_model_name(), "cpu")
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
            embeddings = model.encode(
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
        with _native_ops_lock:
            embeddings = model.encode(
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
        _native_ops_lock.acquire()
        with _priority_condition:
            _interactive_waiters = max(0, _interactive_waiters - 1)
            _priority_condition.notify_all()

    def __exit__(self, exc_type, exc, tb) -> None:
        _native_ops_lock.release()
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

    Closing a native ChromaDB client destroys chromadb's process-global
    system cache (verified with chromadb 1.5.9), which silently invalidates
    every *other* live PersistentClient in this process. If we closed only the
    target, the sibling animas' cached stores would keep pointing at the
    destroyed cache and their next access would fail with "disk I/O error" /
    "file is not a database" — corrupting their DBs. This is what turned a
    single rebuild's reset into a worker-wide corruption cascade. So whenever a
    native store is closed, drop (and close) every cached native store; each is
    recreated lazily with a fresh client and cache on next use.
    """
    global _init_failed

    with _lock:
        target = _vector_stores.pop(anima_name, None)
        closed_native = target is not None
        _close_store(target, anima_name)

        if closed_native:
            siblings = list(_vector_stores.items())
            _vector_stores.clear()
            for sibling_name, sibling_store in siblings:
                _close_store(sibling_store, sibling_name)
            _clear_chroma_system_cache()

        http_keys = [key for key in _http_stores if key[1] == anima_name]
        for key in http_keys:
            store = _http_stores.pop(key, None)
            _close_store(store, anima_name)
        _init_failed = False


def close_all_vector_stores() -> None:
    """Close all cached vector-store clients before process shutdown."""
    global _init_failed

    with _lock:
        stores = list(_vector_stores.items())
        http_stores = list(_http_stores.items())
        _vector_stores.clear()
        _http_stores.clear()
        _init_failed = False

    for anima_name, store in stores:
        _close_store(store, anima_name)
    if stores:
        _clear_chroma_system_cache()
    for (_base_url, anima_name), store in http_stores:
        _close_store(store, anima_name)


def _close_store(store, anima_name: str | None) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("Failed to close vector store for %s", anima_name, exc_info=True)


def _clear_chroma_system_cache() -> None:
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        logger.debug("Failed to clear ChromaDB system cache", exc_info=True)


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
    embed_url = os.environ.get("ANIMAWORKS_EMBED_URL")
    if embed_url:
        return _generate_embeddings_http(texts, embed_url, purpose=purpose, priority=priority)
    return _generate_embeddings_local(texts, purpose=purpose, priority=priority)


def _generate_embeddings_http(
    texts: list[str],
    embed_url: str,
    *,
    purpose: EmbeddingPurpose = "document",
    priority: EmbeddingPriority = "interactive",
) -> list[list[float]]:
    """Call server's /api/internal/embed endpoint."""
    import httpx

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[i : i + _BATCH_LIMIT]
        resp = httpx.post(
            embed_url,
            json={"texts": batch, "purpose": purpose, "priority": priority},
            timeout=30.0,
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


def _reset_for_testing():
    """Reset singletons for test isolation."""
    global _bulk_yield_count, _direct_disabled_warned, _embedding_model, _embedding_model_device, _embedding_model_name
    global _init_failed, _interactive_waiters
    from core.gpu import reset_gpu_status_for_testing

    with _lock:
        _vector_stores.clear()
        _http_stores.clear()
        _embedding_model = None
        _embedding_model_name = None
        _embedding_model_device = None
        _init_failed = False
        _direct_disabled_warned = False
        with _priority_condition:
            _interactive_waiters = 0
            _bulk_yield_count = 0
            _priority_condition.notify_all()
        reset_gpu_status_for_testing()
