from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense vector retrieval system with temporal decay and spreading activation.

Implements:
- Dense vector similarity search (semantic)
- Temporal decay scoring (newer documents ranked higher)
- Spreading activation via knowledge graph (optional)
"""

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.memory.rag.store import SearchResult
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.rag.retriever")

# ── Configuration ───────────────────────────────────────────────────

WEIGHT_RECENCY = 0.2

# Temporal decay half-life (days)
RECENCY_HALF_LIFE_DAYS = 30.0

WEIGHT_FREQUENCY = 0.1

# Hard cap for frequency boost to prevent unbounded score inflation.
# log1p(19) ≈ 3.0, so access_count < 19 behaves identically to before.
FREQUENCY_LOG_CAP = 3.0

# Importance boost for [IMPORTANT]-tagged chunks (amygdala model:
# lowers activation threshold for emotionally significant memories)
WEIGHT_IMPORTANCE = 0.20

# Metadata key prefix for per-anima access counts on shared collections.
_PER_ANIMA_AC_PREFIX = "ac_"


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A retrieved document with combined score."""

    doc_id: str
    content: str
    score: float
    metadata: dict[str, str | int | float | list[str]]
    source_scores: dict[str, float]  # Debug info: individual scores


# ── MemoryRetriever ────────────────────────────────────────────────


class MemoryRetriever:
    """Dense vector search with temporal decay and spreading activation.

    Pipeline:
      Query → Dense Vector Search → Temporal Decay → Sort → Spreading Activation → Results
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        indexer,  # MemoryIndexer instance
        knowledge_dir: Path,
    ) -> None:
        """Initialize memory retriever.

        Args:
            vector_store: VectorStore instance
            indexer: MemoryIndexer instance (for embedding generation)
            knowledge_dir: Path to knowledge directory (for spreading activation)
        """
        self.vector_store = vector_store
        self.indexer = indexer
        self.knowledge_dir = knowledge_dir
        self._knowledge_graph = None  # Lazy initialization
        self._graph_lock = threading.Lock()

    # ── Main search API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        anima_name: str,
        memory_type: str = "knowledge",
        top_k: int = 3,
        enable_spreading_activation: bool | None = None,
        *,
        include_shared: bool = False,
        include_superseded: bool = False,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Perform dense vector search with temporal decay.

        Args:
            query: Search query text
            anima_name: Anima name (for collection selection)
            memory_type: Memory type (knowledge, episodes, etc.)
            top_k: Number of results to return
            enable_spreading_activation: Enable graph-based spreading activation.
                ``None`` reads from ``config.rag.enable_spreading_activation``
                (C方式); explicit ``True``/``False`` overrides.
            include_shared: Also search ``shared_common_knowledge`` collection
                and merge results by score.
            include_superseded: If False (default), exclude knowledge that has
                been superseded (``valid_until`` is non-empty). Set to True
                to include all knowledge regardless of validity.
            min_score: If set, filter out results whose raw vector similarity
                score (before temporal decay / frequency boost) is below this
                threshold.  ``None`` (default) disables filtering.  Spreading
                activation results (no ``"vector"`` key) are never filtered.

        Returns:
            List of retrieval results sorted by combined score
        """
        if enable_spreading_activation is None:
            try:
                _cfg = self._load_config()
                enable_spreading_activation = getattr(
                    _cfg.rag,
                    "enable_spreading_activation",
                    False,
                )
                spreading_types = tuple(getattr(_cfg.rag, "spreading_memory_types", ("knowledge", "episodes")))
            except Exception:
                enable_spreading_activation = False
                spreading_types = ("knowledge", "episodes")
        else:
            spreading_types = self._get_spreading_memory_types()

        logger.debug(
            "Vector search: query='%s', anima=%s, type=%s, top_k=%d, spreading=%s, shared=%s",
            query,
            anima_name,
            memory_type,
            top_k,
            enable_spreading_activation,
            include_shared,
        )

        # Build metadata filter for superseded knowledge exclusion
        filter_metadata: dict[str, str | int | float] | None = None
        if not include_superseded and memory_type == "knowledge":
            filter_metadata = {"valid_until": ""}

        # 1. Dense Vector search (personal collection)
        vector_results = self._vector_search(
            query,
            anima_name,
            memory_type,
            top_k * 2,
            filter_metadata=filter_metadata,
        )

        # 1b. Shared collection search (if requested)
        _SHARED_COLLECTION_MAP: dict[str, str] = {
            "knowledge": "shared_common_knowledge",
            "skills": "shared_common_skills",
        }
        if include_shared:
            shared_collection = _SHARED_COLLECTION_MAP.get(memory_type)
            if shared_collection:
                shared_results = self._vector_search_collection(
                    query,
                    shared_collection,
                    top_k * 2,
                    filter_metadata=filter_metadata,
                )
                vector_results.extend(shared_results)

        # 2. Convert to RetrievalResult
        results = [
            RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score,
                metadata=metadata,
                source_scores={"vector": score},
            )
            for doc_id, content, score, metadata in vector_results
        ]

        # 3. Apply temporal decay and frequency boost
        results = self._apply_score_adjustments(results, anima_name)

        # 3b. Filter by minimum vector score
        if min_score is not None:
            results = [r for r in results if r.source_scores.get("vector", 1.0) >= min_score]

        # 4. Sort & top_k
        results.sort(key=lambda r: r.score, reverse=True)
        initial_results = results[:top_k]

        # 5. Apply spreading activation if enabled
        if enable_spreading_activation and memory_type in spreading_types:
            try:
                expanded = self._apply_spreading_activation(initial_results, anima_name)
                return expanded
            except Exception as e:
                logger.warning("Spreading activation failed, returning initial results: %s", e)
                return initial_results

        return initial_results

    def get_important_chunks(
        self,
        anima_name: str,
        include_shared: bool = True,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Collect all [IMPORTANT] chunks regardless of query context."""
        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        personal = self.vector_store.get_by_metadata(
            f"{anima_name}_knowledge",
            {"importance": "important"},
            limit=limit,
        )
        for r in personal:
            if r.document.id not in seen_ids:
                seen_ids.add(r.document.id)
                results.append(r)

        if include_shared:
            shared = self.vector_store.get_by_metadata(
                "shared_common_knowledge",
                {"importance": "important"},
                limit=limit,
            )
            for r in shared:
                if r.document.id not in seen_ids:
                    seen_ids.add(r.document.id)
                    results.append(r)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ── ActionRule Search ──────────

    def search_action_rules(
        self,
        tool_name: str,
        query: str,
        anima_name: str,
        top_k: int = 3,
        min_score: float = 0.80,
    ) -> list[RetrievalResult]:
        """Search for action rules relevant to the given tool and query.

        Args:
            tool_name: The tool about to be executed (e.g. ``call_human``).
            query: Search query (typically tool_name + tool_input summary).
            anima_name: The anima performing the action.
            top_k: Maximum results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching action rule chunks, filtered by ``trigger_tools``
            and sorted by score descending.
        """
        collection_name = f"{anima_name}_knowledge"
        filter_metadata: dict[str, str | int | float] = {"type": "action_rule"}
        vector_rows = self._vector_search_collection(
            query,
            collection_name,
            top_k * 2,
            filter_metadata=filter_metadata,
        )

        tool_lower = tool_name.lower()
        results: list[RetrievalResult] = []
        for doc_id, content, score, metadata in vector_rows:
            raw_triggers = metadata.get("trigger_tools")
            if raw_triggers is None:
                continue
            trigger_parts = [p.strip().lower() for p in str(raw_triggers).split(",") if p.strip()]
            if tool_lower not in trigger_parts:
                continue
            if score < min_score:
                continue
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                    source_scores={"vector": score},
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ── Search methods ──────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        anima_name: str,
        memory_type: str,
        top_k: int,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Perform vector similarity search on an anima collection.

        Returns:
            List of (doc_id, content, score, metadata) tuples
        """
        collection_name = f"{anima_name}_{memory_type}"
        return self._vector_search_collection(
            query,
            collection_name,
            top_k,
            filter_metadata=filter_metadata,
        )

    def _vector_search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Perform vector similarity search on a named collection.

        Args:
            query: Search query text
            collection_name: Name of the ChromaDB collection
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (exact match).
                Used to exclude superseded knowledge via ``valid_until``.

        Returns:
            List of (doc_id, content, score, metadata) tuples
        """
        # Generate query embedding
        embedding = self.indexer._generate_embeddings([query])[0]

        # Query vector store
        results = self.vector_store.query(
            collection=collection_name,
            embedding=embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        return [(r.document.id, r.document.content, r.score, r.document.metadata) for r in results]

    # ── Score adjustment ────────────────────────────────────────────

    def _apply_score_adjustments(
        self,
        results: list[RetrievalResult],
        anima_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Apply temporal decay, frequency boost, and importance boost.

        - Temporal decay: exponential decay based on document age
        - Frequency boost: log-scaled boost based on access count (Hebbian LTP),
          capped at ``WEIGHT_FREQUENCY * FREQUENCY_LOG_CAP``.
          For shared chunks, per-anima access count (``ac_{anima_name}``) is
          used instead of the global ``access_count``.
        - Importance boost: flat boost for [IMPORTANT]-tagged chunks (amygdala model)
        """
        now = now_local()
        cap = WEIGHT_FREQUENCY * FREQUENCY_LOG_CAP

        for result in results:
            # --- Temporal decay ---
            # Prefer valid_at (event time) over updated_at (file modification time)
            valid_at_raw = result.metadata.get("valid_at")
            if valid_at_raw is not None:
                try:
                    valid_at_ts = float(valid_at_raw)
                    age_days = (now.timestamp() - valid_at_ts) / 86400.0
                    decay_factor = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                except (ValueError, TypeError):
                    decay_factor = 0.5
            else:
                updated_at_str = result.metadata.get("updated_at")
                if not updated_at_str:
                    decay_factor = 0.5
                else:
                    try:
                        updated_at = ensure_aware(datetime.fromisoformat(str(updated_at_str)))
                        age_days = (now - updated_at).total_seconds() / 86400.0
                        decay_factor = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                    except (ValueError, TypeError):
                        decay_factor = 0.5

            recency_score = WEIGHT_RECENCY * decay_factor
            result.score = result.score + recency_score
            result.source_scores["recency"] = recency_score

            # --- Frequency boost (Hebbian LTP analog) ---
            is_shared = result.metadata.get("anima") == "shared"
            if is_shared and anima_name:
                ac_key = f"{_PER_ANIMA_AC_PREFIX}{anima_name}"
                access_count = int(str(result.metadata.get(ac_key, 0)))
            else:
                access_count = int(str(result.metadata.get("access_count", 0)))
            frequency_boost = min(WEIGHT_FREQUENCY * math.log1p(access_count), cap)
            result.score += frequency_boost
            result.source_scores["frequency"] = frequency_boost

            # --- Importance boost (amygdala model) ---
            if result.metadata.get("importance") == "important":
                result.score += WEIGHT_IMPORTANCE
                result.source_scores["importance"] = WEIGHT_IMPORTANCE

        return results

    def record_access(self, results: list[RetrievalResult], anima_name: str) -> None:
        """Record access for retrieved results (LTP analog).

        For personal collections: increments ``access_count``.
        For shared collections: increments per-anima ``ac_{anima_name}``
        and the global ``access_count`` (debug/audit only).
        """
        if not results:
            return

        now_iso_str = now_iso()

        # Group by (collection, is_shared) so we can branch logic
        _Batch = dict[str, list[str]]  # collection → [doc_ids]
        personal_batches: _Batch = {}
        shared_batches: _Batch = {}

        for r in results:
            memory_type = r.metadata.get("memory_type", "knowledge")
            source = r.metadata.get("anima", anima_name)
            collection = f"{source}_{memory_type}"
            if source == "shared":
                shared_batches.setdefault(collection, []).append(r.doc_id)
            else:
                personal_batches.setdefault(collection, []).append(r.doc_id)

        for collection, ids in personal_batches.items():
            try:
                current = self._read_metadata_field(collection, ids, "access_count")
                metas = [
                    {
                        "access_count": current.get(doc_id, 0) + 1,
                        "last_accessed_at": now_iso_str,
                    }
                    for doc_id in ids
                ]
                self.vector_store.update_metadata(collection, ids, metas)
                logger.debug("Recorded access for %d chunks in %s", len(ids), collection)
            except Exception as e:
                logger.warning("Failed to record access for %s: %s", collection, e)

        ac_key = f"{_PER_ANIMA_AC_PREFIX}{anima_name}"
        for collection, ids in shared_batches.items():
            try:
                current_pa = self._read_metadata_field(collection, ids, ac_key)
                current_global = self._read_metadata_field(collection, ids, "access_count")
                metas = [
                    {
                        ac_key: current_pa.get(doc_id, 0) + 1,
                        "access_count": current_global.get(doc_id, 0) + 1,
                        "last_accessed_at": now_iso_str,
                    }
                    for doc_id in ids
                ]
                self.vector_store.update_metadata(collection, ids, metas)
                logger.debug(
                    "Recorded per-anima access (%s) for %d chunks in %s",
                    ac_key,
                    len(ids),
                    collection,
                )
            except Exception as e:
                logger.warning("Failed to record access for %s: %s", collection, e)

    def _read_metadata_field(
        self,
        collection: str,
        ids: list[str],
        field: str = "access_count",
    ) -> dict[str, int]:
        """Read an integer metadata field from the vector store."""
        try:
            docs = self.vector_store.get_by_ids(collection, ids)
            return {doc.id: int(str(doc.metadata.get(field, 0))) for doc in docs}
        except Exception:
            return {}

    def reset_shared_access_counts(self) -> dict[str, int]:
        """Reset access_count and per-anima ac_* fields in shared collections.

        Returns:
            Dict mapping collection name to number of chunks reset.
        """
        _SHARED_COLLECTIONS = ("shared_common_knowledge", "shared_common_skills")
        result: dict[str, int] = {}

        for collection_name in _SHARED_COLLECTIONS:
            try:
                all_results = self.vector_store.get_by_metadata(collection_name, {}, limit=100000)
                if not all_results:
                    continue

                all_ids = [r.document.id for r in all_results]
                reset_metas: list[dict[str, str | int | float]] = []
                for r in all_results:
                    patch: dict[str, str | int | float] = {
                        "access_count": 0,
                        "last_accessed_at": "",
                    }
                    for key in r.document.metadata:
                        if key.startswith(_PER_ANIMA_AC_PREFIX):
                            patch[key] = 0
                    reset_metas.append(patch)

                self.vector_store.update_metadata(collection_name, all_ids, reset_metas)
                result[collection_name] = len(all_ids)
                logger.info("Reset access counts for %d chunks in %s", len(all_ids), collection_name)
            except Exception as e:
                logger.warning("Failed to reset %s: %s", collection_name, e)

        return result

    # ── Config helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_config():
        """Load AnimaWorks config (cached internally by load_config)."""
        from core.config.models import load_config

        return load_config()

    def _get_spreading_memory_types(self) -> tuple[str, ...]:
        """Return spreading memory types from config with fallback."""
        try:
            _cfg = self._load_config()
            return tuple(getattr(_cfg.rag, "spreading_memory_types", ("knowledge", "episodes")))
        except Exception:
            return ("knowledge", "episodes")

    # ── Spreading activation ────────────────────────────────────────

    def _apply_spreading_activation(
        self,
        initial_results: list[RetrievalResult],
        anima_name: str,
    ) -> list[RetrievalResult]:
        """Apply spreading activation to expand search results.

        Builds a graph from all configured memory types (knowledge +
        episodes by default).  Tries loading from cache first, then
        falls back to a full build.

        Args:
            initial_results: Initial search results
            anima_name: Anima name

        Returns:
            Expanded results with activated neighbors
        """
        with self._graph_lock:
            if self._knowledge_graph is None:
                try:
                    from core.memory.rag.graph import KnowledgeGraph

                    self._knowledge_graph = KnowledgeGraph(
                        self.vector_store,
                        self.indexer,
                    )

                    cache_dir = self.knowledge_dir.parent / "vectordb"
                    _cfg = self._safe_load_config()
                    threshold = (
                        getattr(
                            _cfg.rag,
                            "implicit_link_threshold",
                            0.75,
                        )
                        if _cfg
                        else 0.75
                    )
                    if not self._knowledge_graph.load_graph(cache_dir):
                        memory_dirs = self._collect_spreading_dirs()
                        self._knowledge_graph.build_graph(
                            anima_name,
                            self.knowledge_dir,
                            memory_dirs=memory_dirs,
                            implicit_link_threshold=threshold,
                        )
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        self._knowledge_graph.save_graph(cache_dir)

                except Exception as e:
                    logger.warning("Failed to initialize knowledge graph: %s", e)
                    return initial_results

        max_hops = self._get_config_max_hops()
        return self._knowledge_graph.expand_search_results(
            initial_results,
            max_hops=max_hops,
        )

    def _safe_load_config(self):
        """Try loading config; return config or None on failure."""
        try:
            return self._load_config()
        except Exception:
            return None

    def _get_config_max_hops(self) -> int:
        """Read ``rag.max_graph_hops`` from config with fallback."""
        try:
            return getattr(self._load_config().rag, "max_graph_hops", 2)
        except Exception:
            return 2

    def _collect_spreading_dirs(self) -> dict[str, Path]:
        """Collect additional memory directories for spreading activation.

        Reads ``rag.spreading_memory_types`` from config and maps each
        type (excluding ``knowledge`` which is the primary directory)
        to its filesystem path.
        """
        anima_dir = self.knowledge_dir.parent
        extra: dict[str, Path] = {}

        try:
            config = self._load_config()
            memory_types = config.rag.spreading_memory_types
        except Exception:
            memory_types = ["knowledge", "episodes"]

        for mt in memory_types:
            if mt == "knowledge":
                continue
            candidate = anima_dir / mt
            if candidate.is_dir():
                extra[mt] = candidate

        return extra
