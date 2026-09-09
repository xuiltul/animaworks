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
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from core.memory.rag.http_store import HttpVectorStore
    from core.memory.rag.ipc_store import IpcVectorStore, MemoryRequester
    from core.memory.rag.store import VectorStore

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 1000

_lock = threading.Lock()
_native_ops_lock = threading.Lock()
_vector_stores: dict[str | None, VectorStore | None] = {}
_vector_store_init_failed: set[str | None] = set()
_http_stores: dict[tuple[str, str | None], HttpVectorStore] = {}
_ipc_stores: dict[tuple[str, str | None], IpcVectorStore] = {}
_ipc_vector_requester: MemoryRequester | None = None
_ipc_vector_owner: str | None = None
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

_ERROR_RESET_COOLDOWN_ENV = "ANIMAWORKS_VECTOR_ERROR_RESET_COOLDOWN_SECONDS"
_DEFAULT_ERROR_RESET_COOLDOWN_SECONDS = 60.0
_error_reset_lock = threading.Lock()
_last_error_reset_monotonic: float | None = None

_VECTOR_STORE_CLOSE_TIMEOUT_SECONDS = 30.0


class _VectorStoreLifecycleGate:
    """A writer-preferring shared/exclusive gate for native store handles.

    Chroma's client cache is process-global: closing one native client can
    invalidate all of them.  Consequently a reset must wait for every active
    native operation, while ordinary operations can continue concurrently.
    Writer preference prevents a steady stream of reads from starving a
    requested reset indefinitely.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._upgraded_readers: dict[int, int] = {}
        self._writer_active = False
        self._writers_waiting = 0
        self._condemned_requests = 0
        self._deferred_closes: list[Callable[[], None]] = []

    def acquire_shared(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            while self._writer_active or self._writers_waiting or self._condemned_requests or self._deferred_closes:
                self._condition.wait()
            self._readers += 1
            self._reader_depths[thread_id] = self._reader_depths.get(thread_id, 0) + 1

    def current_thread_holds_shared(self) -> bool:
        """Return whether waiting for reader drain would deadlock this thread."""
        with self._condition:
            return self._reader_depths.get(threading.get_ident(), 0) > 0

    def release_shared(self) -> None:
        thread_id = threading.get_ident()
        drain_deferred = False
        with self._condition:
            depth = self._reader_depths.get(thread_id, 0)
            if depth <= 1:
                self._reader_depths.pop(thread_id, None)
            else:
                self._reader_depths[thread_id] = depth - 1
            self._readers -= 1
            if self._readers == 0:
                if self._deferred_closes and not self._writer_active:
                    self._writer_active = True
                    drain_deferred = True
                self._condition.notify_all()
        if drain_deferred:
            self._drain_deferred_closes()

    def acquire_exclusive(self, timeout: float) -> bool:
        thread_id = threading.get_ident()
        deadline = time.monotonic() + timeout
        with self._condition:
            # An operation's existing self-heal can synchronously request a
            # reset. Temporarily remove this thread's reader slots so a
            # shared→exclusive upgrade does not wait on itself.
            upgraded_readers = self._reader_depths.get(thread_id, 0)
            self._readers -= upgraded_readers
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers or self._condemned_requests:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._readers += upgraded_readers
                        # Reserve a condemned slot before returning so no new
                        # operation can reach the old handles while the caller
                        # registers its delayed close.
                        self._condemned_requests += 1
                        return False
                    self._condition.wait(timeout=remaining)
                self._writer_active = True
                if upgraded_readers:
                    self._upgraded_readers[thread_id] = upgraded_readers
                return True
            finally:
                self._writers_waiting -= 1
                self._condition.notify_all()

    def defer_close(self, close_action: Callable[[], None]) -> None:
        """Queue one timed-out lifecycle action until all readers drain."""
        drain_deferred = False
        with self._condition:
            self._deferred_closes.append(close_action)
            self._condemned_requests -= 1
            if self._readers == 0 and not self._writer_active:
                self._writer_active = True
                drain_deferred = True
            self._condition.notify_all()
        if drain_deferred:
            self._drain_deferred_closes()

    def release_exclusive(self, *, acquired: bool, condemned_handled: bool) -> None:
        thread_id = threading.get_ident()
        drain_deferred = False
        upgraded_readers = 0
        with self._condition:
            if acquired:
                upgraded_readers = self._upgraded_readers.pop(thread_id, 0)
                if self._deferred_closes:
                    drain_deferred = True
                else:
                    self._readers += upgraded_readers
                    self._writer_active = False
            elif not condemned_handled:
                self._condemned_requests -= 1
            self._condition.notify_all()
        if drain_deferred:
            self._drain_deferred_closes(restored_readers=upgraded_readers)

    def _drain_deferred_closes(self, *, restored_readers: int = 0) -> None:
        """Run queued closes while holding the gate's writer slot."""
        while True:
            with self._condition:
                while not self._deferred_closes and self._condemned_requests:
                    self._condition.wait()
                if not self._deferred_closes:
                    self._readers += restored_readers
                    self._writer_active = False
                    self._condition.notify_all()
                    return
                close_actions = self._deferred_closes
                self._deferred_closes = []

            for close_action in close_actions:
                try:
                    close_action()
                except Exception:
                    logger.debug("Failed to run deferred vector-store close", exc_info=True)


_vector_store_lifecycle_gate = _VectorStoreLifecycleGate()


@contextmanager
def vector_store_operation(anima_name: str | None = None) -> Iterator[None]:
    """Keep a native vector-store operation alive while reset/close is excluded.

    ``anima_name`` is retained for lifecycle call-site clarity. Native Chroma
    clients share a process-wide cache, so a reset for one owner must exclude
    operations for every owner before it can safely close sibling clients.
    """
    del anima_name
    _vector_store_lifecycle_gate.acquire_shared()
    try:
        yield
    finally:
        _vector_store_lifecycle_gate.release_shared()


@contextmanager
def _vector_store_close_gate(operation: str) -> Iterator[Callable[[Callable[[], None]], None]]:
    """Run a lifecycle action exclusively, or defer it until readers drain."""
    acquired = _vector_store_lifecycle_gate.acquire_exclusive(_VECTOR_STORE_CLOSE_TIMEOUT_SECONDS)
    if not acquired:
        logger.warning(
            "Timed out after %.0fs waiting for in-flight vector-store operations during %s; deferring close",
            _VECTOR_STORE_CLOSE_TIMEOUT_SECONDS,
            operation,
        )
    condemned_handled = False

    def close_or_defer(close_action: Callable[[], None]) -> None:
        nonlocal condemned_handled
        if acquired:
            close_action()
        else:
            _vector_store_lifecycle_gate.defer_close(close_action)
            condemned_handled = True

    try:
        yield close_or_defer
    finally:
        _vector_store_lifecycle_gate.release_exclusive(
            acquired=acquired,
            condemned_handled=condemned_handled,
        )


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


def configure_ipc_vector_requester(requester: MemoryRequester | None, *, anima_name: str | None = None) -> None:
    """Install the task runner's root-memory requester for phase3 operations."""
    global _ipc_vector_requester, _ipc_vector_owner

    with _lock:
        _ipc_vector_requester = requester
        _ipc_vector_owner = anima_name if requester is not None else None
        _ipc_stores.clear()


def _get_ipc_store(base_url: str, anima_name: str | None) -> IpcVectorStore | None:
    requester = _ipc_vector_requester
    if requester is None:
        return None
    if _ipc_vector_owner is not None and anima_name != _ipc_vector_owner:
        logger.warning("Root vector store owner mismatch: requested=%s owner=%s", anima_name, _ipc_vector_owner)
        return None
    normalized_url = base_url.rstrip("/")
    key = (normalized_url, anima_name)
    if key not in _ipc_stores:
        with _lock:
            if key not in _ipc_stores:
                from core.memory.rag.ipc_store import IpcVectorStore

                _ipc_stores[key] = IpcVectorStore(normalized_url, anima_name, requester)
    return _ipc_stores[key]


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
    # A phase3 root also executes inbox/tool work locally. Its registered
    # requester must win over the HTTP proxy to avoid a server round trip to
    # this same owner. The env flag is inherited by disposable task children.
    if _ipc_vector_requester is not None:
        return _get_ipc_store(vector_url or "", anima_name)
    if os.environ.get("ANIMAWORKS_MEMORY_VIA_ROOT") == "1":
        if vector_url:
            return _get_ipc_store(vector_url, anima_name)
        logger.warning("IPC vector store unavailable: ANIMAWORKS_VECTOR_URL is missing")
        return None
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

    if anima_name in _vector_store_init_failed:
        return None

    if anima_name not in _vector_stores:
        with _lock:
            if _init_failed:
                return None
            if anima_name in _vector_store_init_failed:
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
                    _vector_store_init_failed.add(anima_name)
                    logger.warning(
                        "Vector store init failed for %s; RAG disabled for this anima only "
                        "(other animas unaffected). Error: %s",
                        anima_name or "shared",
                        exc,
                    )
                    if anima_name is not None:
                        try:
                            from core.memory.rag.repair import record_chroma_error

                            record_chroma_error(
                                anima_name=anima_name,
                                collection=f"{anima_name}_knowledge",
                                error=f"store_init_failed: {exc}",
                                source="vector_store_init",
                            )
                        except Exception:
                            logger.debug(
                                "Failed to persist vector-store init failure signal for %s",
                                anima_name,
                                exc_info=True,
                            )
                    return None
    return _vector_stores.get(anima_name)


def is_global_vector_store_init_failed() -> bool:
    """Return whether vector-store initialization is globally latched off."""
    with _lock:
        return _init_failed


def is_vector_store_init_failed(anima_name: str | None) -> bool:
    """Return whether native vector-store initialization failed for one owner."""
    with _lock:
        return anima_name in _vector_store_init_failed


def list_vector_store_init_failed_animas() -> list[str]:
    """Return sorted per-anima vector-store initialization failure latches."""
    with _lock:
        return sorted(name for name in _vector_store_init_failed if name is not None)


def clear_vector_store_init_failed(anima_name: str) -> bool:
    """Clear only the per-owner vector-store init failure latch."""
    if anima_name is None:
        return False
    with _lock:
        was_failed = anima_name in _vector_store_init_failed
        _vector_store_init_failed.discard(anima_name)
        return was_failed


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

    # Fast path: already loaded with the same model name.
    #
    # The device is deliberately NOT part of this check: ``resolve_device``
    # re-probes CUDA on every call, and a flapping probe (driver stress,
    # sandboxed child) used to discard and reload the model on each flip —
    # torch never returns freed arenas to the OS, so RSS ratcheted up by
    # ~0.5GB per reload (2026-07-17 OOM incident). Device changes now only
    # happen through the explicit CUDA-failure fallback below and in
    # ``thread_safe_encode``.
    if _embedding_model is not None and _embedding_model_name == resolved_name:
        return _embedding_model

    with _lock:
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

    close_finished = threading.Event()
    with _vector_store_close_gate("reset") as close_or_defer:
        with _lock:
            _vector_store_init_failed.discard(anima_name)
            target = _vector_stores.pop(anima_name, None)
            native_stores = [(anima_name, target)] if target is not None else []

            if native_stores:
                native_stores.extend(_vector_stores.items())
                _vector_stores.clear()

            http_keys = [key for key in _http_stores if key[1] == anima_name]
            http_stores = [(key[1], _http_stores.pop(key, None)) for key in http_keys]
            _init_failed = False

        def close_removed_stores() -> None:
            try:
                for owner, store in native_stores:
                    _close_store(store, owner)
                if native_stores:
                    _clear_chroma_system_cache()
                for owner, store in http_stores:
                    _close_store(store, owner)
            finally:
                close_finished.set()

        close_or_defer(close_removed_stores)

    # Repair callers move or delete the DB immediately after reset. They must
    # not proceed until a timed-out close has drained. A self-heal running under
    # this thread's shared slot cannot wait for itself, so it leaves the close
    # condemned for release_shared() to finish.
    if not _vector_store_lifecycle_gate.current_thread_holds_shared():
        close_finished.wait()


def _error_reset_cooldown_seconds() -> float:
    try:
        value = float(os.environ.get(_ERROR_RESET_COOLDOWN_ENV, _DEFAULT_ERROR_RESET_COOLDOWN_SECONDS))
    except ValueError:
        return _DEFAULT_ERROR_RESET_COOLDOWN_SECONDS
    return max(0.0, value)


def reset_vector_store_after_error(anima_name: str | None = None, *, source: str = "") -> bool:
    """Rate-limited ``reset_vector_store`` for error-recovery paths.

    Every error-triggered reset closes ALL cached native clients (see
    ``reset_vector_store``), interrupting chromadb's in-flight compactions on
    sibling DBs. Under an error storm those interruptions surface fresh
    transient 522/"Failed to get segments" errors, which trigger further
    resets — the self-sustaining churn that has repeatedly escalated into real
    on-disk corruption and fleet-wide quarantines. A process-wide cooldown
    caps that feedback loop. Deliberate resets (rebuild swaps, explicit
    ``/reset-store``) must keep calling ``reset_vector_store`` directly and
    are never throttled.

    Returns True when the reset ran, False when suppressed by the cooldown.
    """
    global _last_error_reset_monotonic

    cooldown = _error_reset_cooldown_seconds()
    with _error_reset_lock:
        now = time.monotonic()
        last = _last_error_reset_monotonic
        if last is not None and (now - last) < cooldown:
            logger.warning(
                "Skipping error-triggered vector store reset for %s (source=%s): cooldown %.0fs, %.0fs remaining",
                anima_name or "shared",
                source or "unknown",
                cooldown,
                cooldown - (now - last),
            )
            return False
        _last_error_reset_monotonic = now
    reset_vector_store(anima_name)
    return True


def close_all_vector_stores() -> None:
    """Close all cached vector-store clients before process shutdown."""
    global _init_failed

    close_finished = threading.Event()
    with _vector_store_close_gate("close-all") as close_or_defer:
        with _lock:
            stores = list(_vector_stores.items())
            http_stores = list(_http_stores.items())
            _vector_stores.clear()
            _http_stores.clear()
            _vector_store_init_failed.clear()
            _init_failed = False

        def close_removed_stores() -> None:
            try:
                for anima_name, store in stores:
                    _close_store(store, anima_name)
                if stores:
                    _clear_chroma_system_cache()
                for (_base_url, anima_name), store in http_stores:
                    _close_store(store, anima_name)
            finally:
                close_finished.set()

        close_or_defer(close_removed_stores)

    if not _vector_store_lifecycle_gate.current_thread_holds_shared():
        close_finished.wait()


def _close_vector_store_with_gate(store, anima_name: str | None, *, operation: str) -> None:
    """Close one non-cached store without racing native operations."""
    close_finished = threading.Event()

    def close_store() -> None:
        try:
            _close_store(store, anima_name)
        finally:
            close_finished.set()

    with _vector_store_close_gate(operation) as close_or_defer:
        close_or_defer(close_store)

    if not _vector_store_lifecycle_gate.current_thread_holds_shared():
        close_finished.wait()


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


_HTTP_CLIENT = None
_HTTP_CLIENT_PID = None


def _shared_http_client():
    """Process-wide keep-alive client — per-call httpx.post() reopens a TCP
    connection each time (measured ~12 calls/search)."""
    global _HTTP_CLIENT, _HTTP_CLIENT_PID
    import os

    import httpx

    pid = os.getpid()
    if _HTTP_CLIENT is None or pid != _HTTP_CLIENT_PID:
        _HTTP_CLIENT = httpx.Client(timeout=180.0)
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


def _reset_for_testing():
    """Reset singletons for test isolation."""
    global _bulk_yield_count, _direct_disabled_warned, _embedding_model, _embedding_model_device, _embedding_model_name
    global _init_failed, _interactive_waiters, _ipc_vector_requester, _ipc_vector_owner, _last_error_reset_monotonic
    global _vector_store_lifecycle_gate
    from core.gpu import reset_gpu_status_for_testing

    with _error_reset_lock:
        _last_error_reset_monotonic = None
    _vector_store_lifecycle_gate = _VectorStoreLifecycleGate()
    with _lock:
        _vector_stores.clear()
        _vector_store_init_failed.clear()
        _http_stores.clear()
        _ipc_stores.clear()
        _ipc_vector_requester = None
        _ipc_vector_owner = None
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
