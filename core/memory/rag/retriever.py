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
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from core.memory.rag.store import SearchResult
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.rag.retriever")

# ── Configuration ───────────────────────────────────────────────────

WEIGHT_RECENCY = 0.2

# Temporal decay half-life (days)
RECENCY_HALF_LIFE_DAYS = 30.0

WEIGHT_FREQUENCY = 0.1


@lru_cache(maxsize=2048)
def _load_skill_document_by_signature(path: Path, mtime_ns: int, size: int):
    del mtime_ns, size
    from core.skills.loader import load_skill_document

    return load_skill_document(path)


def _load_skill_document_cached(path: Path):
    stat = path.stat()
    return _load_skill_document_by_signature(path, stat.st_mtime_ns, stat.st_size)


# Hard cap for frequency boost to prevent unbounded score inflation.
# log1p(19) ≈ 3.0, so access_count < 19 behaves identically to before.
FREQUENCY_LOG_CAP = 3.0

# Importance boost for [IMPORTANT]-tagged chunks (amygdala model:
# lowers activation threshold for emotionally significant memories)
WEIGHT_IMPORTANCE = 0.20

# Metadata key prefix for per-anima access counts on shared collections.
_PER_ANIMA_AC_PREFIX = "ac_"
_ACCESS_WEIGHT_BY_KIND: dict[str, float] = {
    "retrieved": 0.2,
    "used": 1.0,
}


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A retrieved document with combined score."""

    doc_id: str
    content: str
    score: float
    metadata: dict[str, str | int | float | list[str]]
    source_scores: dict[str, float]  # Debug info: individual scores


class AccessBatch:
    """Combine retrieved-access writes without changing in-search boosts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._patches: dict[tuple[str, str], dict[str, str | int | float]] = {}
        self._increments: dict[tuple[str, str], dict[str, str | int | float]] = {}
        self._episode_graph_results: dict[tuple[str, int], tuple[list[RetrievalResult], bool]] = {}
        self._records: list[tuple[list[RetrievalResult], str, str]] = []

    def remember_episode_graph_results(
        self,
        query: str,
        top_k: int,
        results: list[RetrievalResult],
        *,
        expanded: bool,
    ) -> None:
        """Keep episode results for the graph ranker in this retrieval request."""
        with self._lock:
            self._episode_graph_results[(query, top_k)] = (results, expanded)

    def take_episode_graph_results(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[RetrievalResult], bool] | None:
        """Return and discard episode results saved for the graph ranker."""
        with self._lock:
            return self._episode_graph_results.pop((query, top_k), None)

    def overlay(self, collection: str, doc_id: str, metadata: dict) -> dict:
        with self._lock:
            patch = self._patches.get((collection, doc_id))
            return {**metadata, **patch} if patch else metadata

    def record(self, results: list[RetrievalResult], anima_name: str, *, kind: str) -> None:
        weight = _ACCESS_WEIGHT_BY_KIND.get(kind, _ACCESS_WEIGHT_BY_KIND["used"])
        timestamp = now_iso()
        with self._lock:
            self._records.append((list(results), anima_name, kind))
            for result in results:
                memory_type = result.metadata.get("memory_type", "knowledge")
                source = result.metadata.get("anima", anima_name)
                collection = f"{source}_{memory_type}"
                key = (collection, result.doc_id)
                current = {**result.metadata, **self._patches.get(key, {})}
                self._patches[key] = MemoryRetriever._access_metadata_patch(
                    current,
                    kind,
                    weight,
                    timestamp,
                    per_anima_access_key=f"{_PER_ANIMA_AC_PREFIX}{anima_name}" if source == "shared" else None,
                )
                increment = self._increments.setdefault(
                    key,
                    {
                        "access_delta": 0.0,
                        "retrieved_delta": 0,
                        "used_delta": 0,
                        "last_accessed_at": timestamp,
                    },
                )
                increment["access_delta"] = float(increment["access_delta"]) + weight
                increment[f"{kind}_delta"] = int(increment[f"{kind}_delta"]) + 1
                increment["last_accessed_at"] = timestamp
                increment[f"last_{kind}_at"] = timestamp
                if source == "shared":
                    increment["per_anima_access_key"] = f"{_PER_ANIMA_AC_PREFIX}{anima_name}"

    def absorb(self, other: AccessBatch) -> None:
        """Replay another query's records without sharing its score overlays."""
        with other._lock:
            records = other._records
            other._records = []
        for results, anima_name, kind in records:
            self.record(results, anima_name, kind=kind)

    def flush(self, vector_store) -> None:
        with self._lock:
            pending = self._patches
            self._patches = {}
            self._records = []
            increments = self._increments
            self._increments = {}
        if not pending:
            return
        enqueue = getattr(type(vector_store), "enqueue_access_updates", None)
        if callable(enqueue):
            operations = [
                {"collection": collection, "doc_id": doc_id, **increment}
                for (collection, doc_id), increment in increments.items()
            ]
            if enqueue(vector_store, operations):
                return
        grouped: dict[str, list[tuple[str, dict[str, str | int | float]]]] = {}
        for (collection, doc_id), metadata in pending.items():
            grouped.setdefault(collection, []).append((doc_id, metadata))

        for collection, rows in grouped.items():
            vector_store.update_metadata(
                collection,
                [doc_id for doc_id, _metadata in rows],
                [metadata for _doc_id, metadata in rows],
            )


def _metadata_number(metadata: dict[str, object], field: str, *, default: float = 0.0) -> float:
    """Read a non-negative numeric metadata value, accepting legacy strings."""
    try:
        return max(0.0, float(str(metadata.get(field, default))))
    except (TypeError, ValueError):
        return max(0.0, default)


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
        self._knowledge_graph_signature: tuple[str, bool, int, bool, bool] | None = None
        self._graph_lock = threading.Lock()

    def clear_search_cache(self) -> None:
        """Drop data that is valid only for one retrieval request."""
        if self._knowledge_graph is not None:
            self._knowledge_graph.clear_search_cache()

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
        embedding: list[float] | None = None,
        access_batch: AccessBatch | None = None,
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

        # Build metadata filter for superseded knowledge exclusion.
        # Facts need post-filtering because future valid_until values remain active.
        filter_metadata: dict[str, str | int | float] | None = None
        if not include_superseded and memory_type == "knowledge":
            filter_metadata = {"valid_until": ""}

        vector_started = perf_counter()

        # 1. Dense Vector search (personal collection)
        fetch_multiplier = 4 if memory_type == "facts" and not include_superseded else 2
        vector_results = self._vector_search(
            query,
            anima_name,
            memory_type,
            top_k * fetch_multiplier,
            filter_metadata=filter_metadata,
            embedding=embedding,
            access_batch=access_batch,
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
                    embedding=embedding,
                    access_batch=access_batch,
                )
                vector_results.extend(shared_results)

        if memory_type == "skills":
            vector_results = self._filter_loadable_skill_vector_results(vector_results)
        if memory_type == "facts" and not include_superseded:
            vector_results = self._filter_active_fact_vector_results(vector_results)
            if len(vector_results) < top_k:
                vector_results = self._refetch_active_fact_vector_results(
                    query,
                    anima_name,
                    top_k,
                    initial_fetch_k=top_k * fetch_multiplier,
                    embedding=embedding,
                    access_batch=access_batch,
                )

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
        logger.info(
            "Memory retrieval vector phase: anima=%s type=%s top_k=%d results=%d elapsed=%.3fs",
            anima_name,
            memory_type,
            top_k,
            len(initial_results),
            perf_counter() - vector_started,
        )
        # 5. Apply spreading activation if enabled
        if enable_spreading_activation and memory_type in spreading_types:
            spreading_started = perf_counter()
            try:
                expanded = self._apply_spreading_activation(initial_results, anima_name)
                logger.info(
                    "Memory retrieval graph phase: anima=%s type=%s seeds=%d results=%d elapsed=%.3fs",
                    anima_name,
                    memory_type,
                    len(initial_results),
                    len(expanded),
                    perf_counter() - spreading_started,
                )
                if memory_type == "episodes" and access_batch is not None:
                    access_batch.remember_episode_graph_results(query, top_k, expanded, expanded=True)
                return expanded
            except Exception as e:
                logger.warning("Spreading activation failed, returning initial results: %s", e)
                if memory_type == "episodes" and access_batch is not None:
                    access_batch.remember_episode_graph_results(query, top_k, initial_results, expanded=False)
                return initial_results

        if memory_type == "episodes" and access_batch is not None:
            access_batch.remember_episode_graph_results(query, top_k, initial_results, expanded=False)
        return initial_results

    def expand_search_results(
        self,
        initial_results: list[RetrievalResult],
        anima_name: str,
    ) -> list[RetrievalResult]:
        """Apply graph spreading to already-fetched vector seeds."""
        spreading_started = perf_counter()
        try:
            expanded = self._apply_spreading_activation(initial_results, anima_name)
        except Exception as e:
            logger.warning("Spreading activation failed, returning initial results: %s", e)
            return initial_results
        logger.info(
            "Memory retrieval graph phase: anima=%s type=episodes seeds=%d results=%d elapsed=%.3fs reused_seeds=true",
            anima_name,
            len(initial_results),
            len(expanded),
            perf_counter() - spreading_started,
        )
        return expanded

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
        """Search personal and shared action rules relevant to the tool and query.

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
        filter_metadata: dict[str, str | int | float] = {"type": "action_rule"}
        vector_rows: list[tuple[str, str, float, dict]] = []
        for collection_name in (f"{anima_name}_knowledge", "shared_common_knowledge"):
            vector_rows.extend(
                self._vector_search_collection(
                    query,
                    collection_name,
                    top_k * 2,
                    filter_metadata=filter_metadata,
                )
            )

        tool_lower = tool_name.lower()
        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        for doc_id, content, score, metadata in vector_rows:
            if doc_id in seen_ids:
                continue
            raw_triggers = metadata.get("trigger_tools")
            if raw_triggers is None:
                continue
            trigger_parts = [p.strip().lower() for p in str(raw_triggers).split(",") if p.strip()]
            if tool_lower not in trigger_parts:
                continue
            if score < min_score:
                continue
            seen_ids.add(doc_id)
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
        embedding: list[float] | None = None,
        access_batch: AccessBatch | None = None,
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
            embedding=embedding,
            access_batch=access_batch,
        )

    def _vector_search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_metadata: dict[str, str | int | float] | None = None,
        embedding: list[float] | None = None,
        access_batch: AccessBatch | None = None,
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
        if embedding is None:
            embedding = self.indexer._generate_embeddings([query], purpose="query")[0]

        # Query vector store
        results = self.vector_store.query(
            collection=collection_name,
            embedding=embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        return [
            (
                result.document.id,
                result.document.content,
                result.score,
                access_batch.overlay(collection_name, result.document.id, result.document.metadata)
                if access_batch is not None
                else result.document.metadata,
            )
            for result in results
        ]

    def _filter_loadable_skill_vector_results(
        self,
        results: list[tuple[str, str, float, dict]],
    ) -> list[tuple[str, str, float, dict]]:
        """Drop archived/blocked/deleted skill chunks that may still exist in vectors."""
        try:
            from core.skills.curator import curator_allows_access, replay_curator_state

            replay = replay_curator_state(self.indexer.anima_dir)
        except Exception:
            logger.debug("Failed to replay curator state for vector skill filtering", exc_info=True)
            return results

        resolved: list[Path | None] = []
        for _doc_id, _content, _score, metadata in results:
            source_file = metadata.get("source_file") if isinstance(metadata, dict) else None
            if not isinstance(source_file, str):
                resolved.append(None)
            else:
                resolved.append(self._resolve_skill_source_file(source_file))

        unique_paths = list(dict.fromkeys(path for path in resolved if path is not None))

        def _is_allowed(skill_path: Path) -> bool:
            try:
                meta, _content = _load_skill_document_cached(skill_path)
                allowed, _reason = curator_allows_access(meta, replay=replay)
            except Exception:
                logger.debug("Failed to evaluate vector skill result %s", skill_path, exc_info=True)
                return False
            return allowed

        with ThreadPoolExecutor(max_workers=min(16, len(unique_paths) or 1)) as pool:
            allowed_paths = {
                path for path, allowed in zip(unique_paths, pool.map(_is_allowed, unique_paths), strict=True) if allowed
            }
        return [result for result, path in zip(results, resolved, strict=True) if path in allowed_paths]

    def _filter_active_fact_vector_results(
        self,
        results: list[tuple[str, str, float, dict]],
    ) -> list[tuple[str, str, float, dict]]:
        try:
            from core.memory.facts import is_valid_until_active
        except Exception:
            logger.debug("Failed to import fact validity helper", exc_info=True)
            return results

        filtered: list[tuple[str, str, float, dict]] = []
        for doc_id, content, score, metadata in results:
            valid_until = str((metadata or {}).get("valid_until", "") or "")
            if is_valid_until_active(valid_until):
                filtered.append((doc_id, content, score, metadata))
        return filtered

    def _refetch_active_fact_vector_results(
        self,
        query: str,
        anima_name: str,
        top_k: int,
        *,
        initial_fetch_k: int,
        embedding: list[float] | None = None,
        access_batch: AccessBatch | None = None,
    ) -> list[tuple[str, str, float, dict]]:
        filtered: list[tuple[str, str, float, dict]] = []
        fetch_sizes = (
            max(initial_fetch_k, top_k * 8, 50),
            min(max(top_k * 16, 100), 1000),
        )
        for fetch_k in fetch_sizes:
            if embedding is None:
                if access_batch is None:
                    results = self._vector_search(query, anima_name, "facts", fetch_k)
                else:
                    results = self._vector_search(query, anima_name, "facts", fetch_k, access_batch=access_batch)
            else:
                results = self._vector_search(
                    query,
                    anima_name,
                    "facts",
                    fetch_k,
                    embedding=embedding,
                    access_batch=access_batch,
                )
            filtered = self._filter_active_fact_vector_results(results)
            if len(filtered) >= top_k:
                break
        return filtered

    def _resolve_skill_source_file(self, source_file: str) -> Path | None:
        path = Path(source_file)
        if path.parts[:1] == ("skills",):
            return self.indexer.anima_dir / path
        if path.parts[:1] == ("common_skills",):
            try:
                from core.paths import get_data_dir

                return get_data_dir() / path
            except Exception:
                logger.debug("Failed to resolve data dir for shared skill source", exc_info=True)
                return None
        if path.parts[:1] == ("companies",):
            try:
                from core.company_resources import get_company_resources, infer_data_dir

                resources = get_company_resources(self.indexer.anima_dir)
                if resources is None or path.parts[:3] != ("companies", resources.name, "skills"):
                    return None
                candidate = (infer_data_dir(self.indexer.anima_dir) / path).resolve()
                candidate.relative_to(resources.skills_dir)
                return candidate
            except (OSError, ValueError):
                return None
        return None

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
                access_count = _metadata_number(result.metadata, ac_key)
            else:
                access_count = _metadata_number(result.metadata, "access_count")
            frequency_boost = min(WEIGHT_FREQUENCY * math.log1p(access_count), cap)
            result.score += frequency_boost
            result.source_scores["frequency"] = frequency_boost

            # --- Importance boost (amygdala model) ---
            if result.metadata.get("importance") == "important":
                result.score += WEIGHT_IMPORTANCE
                result.source_scores["importance"] = WEIGHT_IMPORTANCE

        return results

    def record_access(
        self,
        results: list[RetrievalResult],
        anima_name: str,
        *,
        kind: str = "used",
        use_result_metadata: bool = False,
    ) -> None:
        """Record access for results (LTP analog).

        ``kind="retrieved"`` is used for automatic search/priming hits and
        applies weak weight. ``kind="used"`` is explicit use
        (``read_memory_file`` / outcome report) and applies full weight.
        Existing legacy ``access_count`` values are migrated into
        ``used_count`` when split counters are absent.

        For personal collections: increments weighted ``access_count``.
        For shared collections: increments per-anima ``ac_{anima_name}``
        and the global ``access_count`` (debug/audit only).
        """
        if not results:
            return

        if kind not in _ACCESS_WEIGHT_BY_KIND:
            logger.debug("Unknown record_access kind=%r; defaulting to used", kind)
            kind = "used"
        weight = _ACCESS_WEIGHT_BY_KIND[kind]
        now_iso_str = now_iso()

        # Group by (collection, is_shared) so we can branch logic
        _Batch = dict[str, list[str]]  # collection → [doc_ids]
        personal_batches: _Batch = {}
        shared_batches: _Batch = {}
        result_metadata: dict[str, dict[str, dict[str, object]]] = {}

        for r in results:
            memory_type = r.metadata.get("memory_type", "knowledge")
            source = r.metadata.get("anima", anima_name)
            collection = f"{source}_{memory_type}"
            if use_result_metadata:
                result_metadata.setdefault(collection, {})[r.doc_id] = dict(r.metadata)
            if source == "shared":
                shared_batches.setdefault(collection, []).append(r.doc_id)
            else:
                personal_batches.setdefault(collection, []).append(r.doc_id)

        for collection, ids in personal_batches.items():
            try:
                current = (
                    result_metadata[collection] if use_result_metadata else self._read_metadata_fields(collection, ids)
                )
                metas = [
                    self._access_metadata_patch(current.get(doc_id, {}), kind, weight, now_iso_str) for doc_id in ids
                ]
                self.vector_store.update_metadata(collection, ids, metas)
                logger.debug("Recorded %s access for %d chunks in %s", kind, len(ids), collection)
            except Exception as e:
                logger.warning("Failed to record access for %s: %s", collection, e)

        ac_key = f"{_PER_ANIMA_AC_PREFIX}{anima_name}"
        for collection, ids in shared_batches.items():
            try:
                current = (
                    result_metadata[collection] if use_result_metadata else self._read_metadata_fields(collection, ids)
                )
                metas = [
                    self._access_metadata_patch(
                        current.get(doc_id, {}),
                        kind,
                        weight,
                        now_iso_str,
                        per_anima_access_key=ac_key,
                    )
                    for doc_id in ids
                ]
                self.vector_store.update_metadata(collection, ids, metas)
                logger.debug(
                    "Recorded per-anima %s access (%s) for %d chunks in %s",
                    kind,
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
    ) -> dict[str, float]:
        """Read a numeric metadata field from the vector store."""
        return {
            doc_id: _metadata_number(meta, field)
            for doc_id, meta in self._read_metadata_fields(collection, ids).items()
        }

    def _read_metadata_fields(
        self,
        collection: str,
        ids: list[str],
    ) -> dict[str, dict[str, object]]:
        """Read metadata for documents from the vector store."""
        try:
            docs = self.vector_store.get_by_ids(collection, ids)
            return {doc.id: dict(doc.metadata) for doc in docs}
        except Exception:
            return {}

    @staticmethod
    def _access_metadata_patch(
        current: dict[str, object],
        kind: str,
        weight: float,
        now_iso_str: str,
        *,
        per_anima_access_key: str | None = None,
    ) -> dict[str, str | int | float]:
        access_count = _metadata_number(current, "access_count") + weight
        retrieved_count = _metadata_number(current, "retrieved_count")
        used_count = _metadata_number(
            current,
            "used_count",
            default=_metadata_number(current, "access_count"),
        )

        patch: dict[str, str | int | float] = {
            "access_count": access_count,
            "retrieved_count": retrieved_count,
            "used_count": used_count,
            "last_accessed_at": now_iso_str,
        }
        if per_anima_access_key is not None:
            patch[per_anima_access_key] = _metadata_number(current, per_anima_access_key) + weight
        if kind == "retrieved":
            patch["retrieved_count"] = retrieved_count + 1
            patch["last_retrieved_at"] = now_iso_str
        else:
            patch["used_count"] = used_count + 1
            patch["last_used_at"] = now_iso_str
        return patch

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
                        "retrieved_count": 0,
                        "used_count": 0,
                        "last_accessed_at": "",
                        "last_retrieved_at": "",
                        "last_used_at": "",
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

    @staticmethod
    def _env_flag_enabled(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    def _entity_aware_graph_settings(self, config) -> dict[str, bool | int]:
        """Resolve optional entity-aware graph settings from config/env."""
        rag = getattr(config, "rag", None)
        enabled = bool(getattr(rag, "entity_aware_graph_enabled", False))
        if self._env_flag_enabled("LOCOMO_ENTITY_AWARE_GRAPH"):
            enabled = True
        return {
            "enabled": enabled,
            "edge_cap": int(getattr(rag, "graph_entity_edge_cap", 8) or 8),
            "inverse_fan": bool(getattr(rag, "graph_inverse_fan_enabled", True)),
            "recency_weight": bool(getattr(rag, "graph_recency_weight_enabled", True)),
        }

    # ── Spreading activation ────────────────────────────────────────

    def _apply_spreading_activation(
        self,
        initial_results: list[RetrievalResult],
        anima_name: str,
    ) -> list[RetrievalResult]:
        """Apply spreading activation to expand search results.

        Uses an already-built, schema-compatible graph cache. A cache miss
        retains the retrieved seeds: full graph construction embeds and queries
        every source file and belongs to explicit/background maintenance, never
        a latency-bounded search (whose cancelled worker thread would keep
        building the graph after its caller has timed out).

        Args:
            initial_results: Initial search results
            anima_name: Anima name

        Returns:
            Expanded results with activated neighbors
        """
        _cfg = self._safe_load_config()
        graph_settings = self._entity_aware_graph_settings(_cfg)
        graph_signature = (
            anima_name,
            bool(graph_settings["enabled"]),
            int(graph_settings["edge_cap"]),
            bool(graph_settings["inverse_fan"]),
            bool(graph_settings["recency_weight"]),
        )

        with self._graph_lock:
            if self._knowledge_graph_signature != graph_signature:
                self._knowledge_graph = None

            if self._knowledge_graph is None:
                try:
                    from core.memory.rag.graph import GRAPH_SCHEMA_VERSION, KnowledgeGraph

                    graph = KnowledgeGraph(
                        self.vector_store,
                        self.indexer,
                    )

                    cache_dir = self.knowledge_dir.parent / "vectordb"
                    cache_enabled = bool(getattr(_cfg.rag, "graph_cache_enabled", True)) if _cfg else True
                    loaded = False
                    if cache_enabled:
                        loaded = graph.load_graph(
                            cache_dir,
                            expected_schema_version=GRAPH_SCHEMA_VERSION,
                            entity_aware_graph_enabled=bool(graph_settings["enabled"]),
                        )
                    if not loaded:
                        logger.info(
                            "Graph expansion skipped: no usable prebuilt cache for %s; retaining retrieval seeds",
                            anima_name,
                        )
                        return initial_results
                    self._knowledge_graph = graph
                    self._knowledge_graph_signature = graph_signature

                except Exception as e:
                    logger.warning("Failed to initialize knowledge graph: %s", e)
                    self._knowledge_graph = None
                    self._knowledge_graph_signature = None
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
