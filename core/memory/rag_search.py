from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.company_resources import (
    CompanyResources,
    company_resource_pointer,
    get_company_resources,
    get_company_resources_for_company,
    infer_data_dir,
)
from core.config.models import read_anima_company_checked
from core.memory.fact_observability import warn_rate_limited
from core.memory.rag.shared_check_registry import (
    SharedCheckOutcome,
    make_shared_check_key,
    run_shared_check,
)
from core.memory.rag.shared_meta import read_shared_hash, reset_shared_for_company_change, write_shared_hash
from core.memory.rag.store import CollectionExistence

logger = logging.getLogger("animaworks.memory")

_INDEX_RETRY_SECONDS = 30.0

try:
    from core.memory.bm25 import search_activity_log, search_longterm_memory_bm25
except ImportError:
    search_activity_log = None  # type: ignore[assignment,misc]
    search_longterm_memory_bm25 = None  # type: ignore[assignment,misc]

_EPISODES_TOP_K = 10
_DEFAULT_TOP_K = 5
WEIGHT_TOKEN_OVERLAP = 0.1


def _keyword_token_matches(token: str, content_lower: str) -> bool:
    """Return True when a keyword token matches directly or by bounded CJK slices."""
    if token in content_lower:
        return True
    if len(token) < 4:
        return False
    if any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in token):
        for size in (8, 6, 4):
            for start in range(0, max(0, len(token) - size + 1)):
                if token[start : start + size] in content_lower:
                    return True
    return False


@lru_cache(maxsize=4096)
def _read_keyword_file_by_signature(path: Path, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return path.read_text(encoding="utf-8")


def _read_keyword_file(path: Path) -> tuple[str, os.stat_result]:
    stat = path.stat()
    return _read_keyword_file_by_signature(path, stat.st_mtime_ns, stat.st_size), stat


# ── Shared-index change detection helpers ─────────────────


def _compute_dir_hash(dir_path: Path, glob_pattern: str = "*.md") -> str:
    """Compute a SHA-256 hash over file relative paths + mtimes in *dir_path*.

    The hash changes whenever a file is added, removed, or modified.
    """
    entries: list[tuple[str, float]] = []
    for f in dir_path.rglob(glob_pattern):
        if f.is_file():
            entries.append((str(f.relative_to(dir_path)), f.stat().st_mtime))
    entries.sort()
    h = hashlib.sha256(repr(entries).encode()).hexdigest()
    return h


def _shared_collection_exists(vector_store, collection_name: str) -> CollectionExistence:
    """Return the checked existence state for a shared collection."""
    return vector_store.collection_exists(collection_name)


# ── RAGMemorySearch ───────────────────────────────────────


class RAGMemorySearch:
    """RAG vector search and indexer management."""

    def __init__(
        self,
        anima_dir: Path,
        common_knowledge_dir: Path,
        common_skills_dir: Path,
    ) -> None:
        self._anima_dir = anima_dir
        self._common_knowledge_dir = common_knowledge_dir
        self._common_skills_dir = common_skills_dir
        self._indexer = None
        self._retriever = None
        self._indexer_initialized = False
        self._indexer_init_lock = threading.Lock()
        self._index_retry_at: float | None = None
        self._auto_index_on_access = not os.environ.get("ANIMAWORKS_TASK_IPC_PATH", "").strip()
        self._last_search_meta: dict[str, object] = {}

    def _init_indexer(self) -> None:
        """Initialize RAG indexer if dependencies are available.

        Called lazily by ``_get_indexer()`` on first access.
        Uses process-level singletons for ChromaVectorStore and embedding
        model to avoid costly repeated initialization.
        """
        self._indexer_initialized = True
        self._index_retry_at = None
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            anima_name = self._anima_dir.name
            vector_store = get_vector_store(anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, indexer disabled")
                self._schedule_index_retry()
                return
            self._indexer = MemoryIndexer(vector_store, anima_name, self._anima_dir)
            logger.debug("RAG indexer initialized for anima=%s", anima_name)

            if not self._auto_index_on_access:
                return

            # Cold catch-up: index personal memory dirs that have sources.
            # index_directory uses index_meta hash + collection existence checks
            # so unchanged files are skipped on subsequent inits (no full re-embed).
            # knowledge/episodes/skills were previously only covered by CLI full
            # index or daily scheduler; without those they never entered dense search.
            cold_sources: list[tuple[str, Path, str]] = [
                ("knowledge", self._anima_dir / "knowledge", "*.md"),
                ("episodes", self._anima_dir / "episodes", "*.md"),
                ("procedures", self._anima_dir / "procedures", "*.md"),
                ("skills", self._anima_dir / "skills", "SKILL.md"),
                ("facts", self._anima_dir / "facts", "*.jsonl"),
            ]
            for memory_type, memory_dir, pattern in cold_sources:
                if not memory_dir.is_dir() or not any(memory_dir.rglob(pattern)):
                    continue
                try:
                    indexed = self._indexer.index_directory(memory_dir, memory_type)
                    if indexed.files_failed or indexed.files_unprocessed:
                        self._schedule_index_retry()
                    if indexed.chunks_indexed > 0:
                        logger.debug(
                            "Indexed %d chunks from %s/",
                            indexed.chunks_indexed,
                            memory_type,
                        )
                except Exception as e:
                    self._schedule_index_retry()
                    if memory_type == "facts":
                        warn_rate_limited(
                            logger,
                            "fact_extraction.rag_search_facts_index",
                            "Failed to index facts for anima=%s",
                            anima_name,
                            exc_info=(type(e), e, e.__traceback__),
                        )
                    else:
                        logger.warning("Failed to index %s: %s", memory_type, e)

            # Index conversation summary (compressed_summary)
            state_dir = self._anima_dir / "state"
            conv_file = state_dir / "conversation.json"
            if conv_file.is_file():
                try:
                    indexed = self._indexer.index_conversation_summary(
                        state_dir,
                        anima_name,
                    )
                    outcome = getattr(self._indexer, "_last_index_file_outcome", None)
                    if getattr(outcome, "status", None) == "failed":
                        self._schedule_index_retry()
                    if indexed > 0:
                        logger.debug(
                            "Indexed %d chunks from conversation_summary",
                            indexed,
                        )
                except Exception as e:
                    self._schedule_index_retry()
                    warn_rate_limited(
                        logger,
                        "fact_extraction.conversation_summary_index",
                        "Failed to index conversation_summary for anima=%s",
                        anima_name,
                        exc_info=(type(e), e, e.__traceback__),
                    )

        except ImportError:
            logger.debug("RAG dependencies not installed, indexing disabled")
        except Exception as e:
            self._schedule_index_retry()
            logger.warning("Failed to initialize RAG indexer: %s", e)
        finally:
            if self._index_retry_at is not None:
                # A long failed catch-up must still cool down before a queued
                # caller retries it. Single-file failures do not postpone an
                # already scheduled retry in _schedule_index_retry().
                self._index_retry_at = time.monotonic() + _INDEX_RETRY_SECONDS

    def _schedule_index_retry(self) -> None:
        """Retry failed catch-up on later access without a hot outage loop."""
        if self._index_retry_at is None:
            self._index_retry_at = time.monotonic() + _INDEX_RETRY_SECONDS

    # ── Shared collection change detection ────────────────

    def _check_shared_collections(self) -> None:
        """Re-index shared common_knowledge / common_skills if changed.

        Called on every ``_get_indexer()`` access so that file changes are
        picked up even after the initial ``_init_indexer()`` run.  Uses a
        SHA-256 hash of (relative_path, mtime) tuples stored in the
        per-anima ``shared_index_meta.json`` to skip re-indexing when unchanged.
        """
        if self._indexer is None:
            return
        try:
            vector_store = self._indexer.vector_store
            company_valid, company = self._reset_shared_for_company_change(vector_store)
            if not company_valid:
                return
            company_resources = get_company_resources_for_company(
                company,
                data_dir=infer_data_dir(self._anima_dir),
            )
            self._ensure_shared_knowledge_indexed(vector_store)
            self._ensure_shared_skills_indexed(vector_store)
            self._ensure_company_knowledge_indexed(vector_store, company_resources)
            self._ensure_company_skills_indexed(vector_store, company_resources)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Shared collection check failed: %s", e)

    def _reset_shared_for_company_change(self, vector_store) -> tuple[bool, str | None]:
        """Drop stale company content only after a checked membership read."""
        valid, company = read_anima_company_checked(self._anima_dir)
        if not valid:
            logger.warning("Company membership unavailable for %s; skipping shared indexing", self._anima_dir.name)
            return False, None
        if not reset_shared_for_company_change(self._anima_dir, vector_store, company or ""):
            return False, company
        return True, company

    @staticmethod
    def _shared_check_timing() -> tuple[float, float, float]:
        """Load shared-check timing with schema defaults as a fail-safe."""
        try:
            from core.config import load_config

            rag = load_config().rag
        except Exception:
            from core.config.models import RAGConfig

            rag = RAGConfig()
        return (
            rag.shared_check_ttl_seconds,
            rag.shared_check_backoff_initial_seconds,
            rag.shared_check_backoff_max_seconds,
        )

    def _ensure_shared_directory_indexed(
        self,
        vector_store,
        directory: Path,
        memory_type: str,
        glob: str,
        meta_key: str,
        *,
        shared_anima_dir: Path,
        missing_log: str | None = None,
        success_log: str | None = None,
    ) -> None:
        """Check and index one shared source through the process-wide registry."""
        current_hash = _compute_dir_hash(directory, glob)
        collection = f"shared_{memory_type}"
        key = make_shared_check_key(
            self._anima_dir.name,
            vector_store,
            collection,
            f"{meta_key}:{current_hash}",
        )

        def check() -> SharedCheckOutcome:
            stored_hash = read_shared_hash(self._anima_dir, meta_key)
            force = False
            if current_hash == stored_hash:
                existence = _shared_collection_exists(vector_store, collection)
                if existence is CollectionExistence.EXISTS:
                    return SharedCheckOutcome.SUCCESS
                if existence is CollectionExistence.UNAVAILABLE:
                    return SharedCheckOutcome.TRANSIENT
                if missing_log:
                    logger.info(missing_log)
                force = True

            from core.memory.rag import MemoryIndexer

            indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=shared_anima_dir,
                collection_prefix="shared",
                embedding_model=self._indexer.embedding_model if self._indexer else None,
            )
            result = indexer.index_directory(directory, memory_type, force=force)
            if result.files_failed == 0:
                write_shared_hash(self._anima_dir, meta_key, current_hash)
                if success_log and result.chunks_indexed > 0:
                    logger.info(success_log, result.chunks_indexed)
                return SharedCheckOutcome.SUCCESS
            if result.transient_failures > 0:
                return SharedCheckOutcome.TRANSIENT
            return SharedCheckOutcome.FAILED

        ttl, backoff_initial, backoff_max = self._shared_check_timing()
        run_shared_check(
            key,
            check,
            ttl_seconds=ttl,
            backoff_initial_seconds=backoff_initial,
            backoff_max_seconds=backoff_max,
        )

    def _ensure_shared_knowledge_indexed(self, vector_store) -> None:
        """Index common_knowledge/ into ``shared_common_knowledge`` collection.

        Skips re-indexing when the directory hash matches the stored value
        AND the target collection still exists in the vector store.  When
        the collection is missing (e.g. vectordb was wiped), forces a full
        re-index so it gets recreated.
        """
        ck_dir = self._common_knowledge_dir
        if not ck_dir.is_dir() or not any(ck_dir.rglob("*.md")):
            logger.debug("No common_knowledge files found, skipping shared indexing")
            return

        try:
            from core.paths import get_data_dir

            self._ensure_shared_directory_indexed(
                vector_store,
                ck_dir,
                "common_knowledge",
                "*.md",
                "shared_common_knowledge_hash",
                shared_anima_dir=get_data_dir(),
                missing_log="shared_common_knowledge collection missing despite tracked hash, forcing re-index",
                success_log="Indexed %d chunks into shared_common_knowledge",
            )
        except Exception as e:
            logger.warning("Failed to index shared common_knowledge: %s", e)

    def _ensure_shared_skills_indexed(self, vector_store) -> None:
        """Index common_skills/ into ``shared_common_skills`` collection.

        Skips re-indexing when the directory hash matches the stored value
        AND the target collection still exists in the vector store.  When
        the collection is missing (e.g. vectordb was wiped), forces a full
        re-index so it gets recreated.
        """
        cs_dir = self._common_skills_dir
        if not cs_dir.is_dir() or not any(cs_dir.rglob("SKILL.md")):
            logger.debug("No common_skills files found, skipping shared skills indexing")
            return

        try:
            from core.paths import get_data_dir

            self._ensure_shared_directory_indexed(
                vector_store,
                cs_dir,
                "common_skills",
                "SKILL.md",
                "shared_common_skills_hash",
                shared_anima_dir=get_data_dir(),
                missing_log="shared_common_skills collection missing despite tracked hash, forcing re-index",
                success_log="Indexed %d chunks into shared_common_skills",
            )
        except Exception as e:
            logger.warning("Failed to index shared common_skills: %s", e)

    def _ensure_company_knowledge_indexed(
        self,
        vector_store,
        resources: CompanyResources | None,
    ) -> None:
        if resources is None or not resources.knowledge_dir.is_dir() or not any(resources.knowledge_dir.rglob("*.md")):
            return
        self._index_company_directory(
            vector_store,
            resources.knowledge_dir,
            "common_knowledge",
            "*.md",
            "shared_company_knowledge_hash",
        )

    def _ensure_company_skills_indexed(
        self,
        vector_store,
        resources: CompanyResources | None,
    ) -> None:
        if resources is None or not resources.skills_dir.is_dir() or not any(resources.skills_dir.rglob("SKILL.md")):
            return
        self._index_company_directory(
            vector_store,
            resources.skills_dir,
            "common_skills",
            "SKILL.md",
            "shared_company_skills_hash",
        )

    def _index_company_directory(
        self,
        vector_store,
        directory: Path,
        memory_type: str,
        glob: str,
        meta_key: str,
    ) -> None:
        try:
            self._ensure_shared_directory_indexed(
                vector_store,
                directory,
                memory_type,
                glob,
                meta_key,
                shared_anima_dir=infer_data_dir(self._anima_dir),
            )
        except Exception as exc:
            logger.warning("Failed to index company %s: %s", memory_type, exc)

    def _get_indexer(self):
        """Return the RAG indexer, initializing it on first call.

        Also checks shared collections for changes on every root-process call.
        Task runners initialize only the read-capable indexer; root preflight,
        consolidation, and cron remain responsible for automatic indexing.
        """
        with self._indexer_init_lock:
            if not self._indexer_initialized or (
                self._index_retry_at is not None and time.monotonic() >= self._index_retry_at
            ):
                self._init_indexer()
        if self._auto_index_on_access:
            self._check_shared_collections()
        return self._indexer

    def _get_retriever(self, indexer, knowledge_dir: Path):
        """Reuse the retriever so its loaded graph survives across searches."""
        from core.memory.rag.retriever import MemoryRetriever

        if (
            self._retriever is None
            or self._retriever.indexer is not indexer
            or self._retriever.knowledge_dir != knowledge_dir
        ):
            self._retriever = MemoryRetriever(indexer.vector_store, indexer, knowledge_dir)
        return self._retriever

    # ── Search methods ────────────────────────────────────

    def search_memory_text(
        self,
        query: str,
        scope: str = "all",
        *,
        offset: int = 0,
        context_window: int = 128_000,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
        result_limit: int | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> list[dict]:
        """Search memory through the unified Legacy retrieval policy.

        Returns ranked results as dicts with score, content, and metadata while
        preserving the legacy tool result shape.
        """
        offset = max(0, min(offset, 50))
        self._last_search_meta = {}
        if self._retriever is not None:
            self._retriever.clear_search_cache()

        if scope == "activity_log":
            if search_activity_log is None:
                return []
            return search_activity_log(
                self._anima_dir,
                query,
                top_k=10,
                offset=offset,
                time_start=time_start,
                time_end=time_end,
            )

        if scope in (
            "all",
            "facts",
            "knowledge",
            "episodes",
            "procedures",
            "common_knowledge",
            "skills",
            "conversation_summary",
        ):
            from core.memory.retrieval.unified_search import UnifiedMemorySearch

            searcher = UnifiedMemorySearch(
                self._anima_dir,
                common_knowledge_dir=common_knowledge_dir,
                common_skills_dir=self._common_skills_dir,
                rag_search=self,
            )
            results = searcher.search(
                query,
                scope=scope,
                limit=result_limit or 10,
                trigger="tool",
                offset=offset,
                time_start=time_start,
                time_end=time_end,
            )
            self._last_search_meta = searcher.last_search_meta
            return results

        indexer = self._get_indexer()
        primary_results: list[dict] = []
        entity_boost = self._build_entity_boost_config(query)
        if indexer is not None:
            try:
                primary_results = self._vector_search_primary(
                    query,
                    scope,
                    offset,
                    knowledge_dir,
                    result_limit=result_limit,
                    entity_boost=entity_boost,
                )
            except Exception as e:
                logger.debug("Vector search failed, falling back to keyword: %s", e)
                primary_results = self._keyword_search_fallback(
                    query,
                    scope,
                    offset,
                    knowledge_dir=knowledge_dir,
                    episodes_dir=episodes_dir,
                    procedures_dir=procedures_dir,
                    common_knowledge_dir=common_knowledge_dir,
                    result_limit=result_limit,
                    entity_boost=entity_boost,
                )
        else:
            primary_results = self._keyword_search_fallback(
                query,
                scope,
                offset,
                knowledge_dir=knowledge_dir,
                episodes_dir=episodes_dir,
                procedures_dir=procedures_dir,
                common_knowledge_dir=common_knowledge_dir,
                result_limit=result_limit,
                entity_boost=entity_boost,
            )

        return primary_results

    @property
    def last_search_meta(self) -> dict[str, object]:
        """Metadata from the most recent search (e.g. abstain flag)."""
        return dict(self._last_search_meta)

    def _load_rag_pipeline_settings(self) -> dict[str, object]:
        """Resolve RAG pipeline knobs from config with safe defaults."""
        defaults: dict[str, object] = {
            "rerank_enabled": True,
            "rerank_candidate_pool": 50,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "abstain_on_low_confidence": True,
            "confidence_threshold": 0.35,
            "rrf_confidence_threshold": 0.02,
            "enable_spreading_activation": True,
            "iterative_retrieval_enabled": True,
            "iterative_min_results": 2,
            "entity_registry_enabled": True,
            "entity_boost_enabled": True,
            "entity_boost": 0.20,
            "entity_boost_cap": 0.80,
            "temporal_boost_enabled": True,
            "temporal_boost": 0.05,
            "temporal_boost_max": 0.10,
            "temporal_half_life_days": 7.0,
            "access_boost_enabled": True,
            "access_boost_weight": 0.05,
            "access_boost_cap": 0.25,
            "access_boost_half_life_days": 30.0,
        }
        try:
            from core.config import load_config

            rag = load_config().rag
            defaults.update(
                {
                    "enable_spreading_activation": rag.enable_spreading_activation,
                    "rerank_enabled": rag.rerank_enabled,
                    "rerank_candidate_pool": rag.rerank_candidate_pool,
                    "cross_encoder_model": rag.cross_encoder_model,
                    "abstain_on_low_confidence": rag.abstain_on_low_confidence,
                    "confidence_threshold": rag.confidence_threshold,
                    "rrf_confidence_threshold": rag.rrf_confidence_threshold,
                    "iterative_retrieval_enabled": getattr(rag, "iterative_retrieval_enabled", True),
                    "iterative_min_results": getattr(rag, "iterative_min_results", 2),
                    "entity_registry_enabled": getattr(rag, "entity_registry_enabled", True),
                    "entity_boost_enabled": getattr(rag, "entity_boost_enabled", True),
                    "entity_boost": getattr(rag, "entity_boost", 0.20),
                    "entity_boost_cap": getattr(rag, "entity_boost_cap", 0.80),
                    "temporal_boost_enabled": getattr(rag, "temporal_boost_enabled", True),
                    "temporal_boost": getattr(rag, "temporal_boost", 0.05),
                    "temporal_boost_max": getattr(rag, "temporal_boost_max", 0.10),
                    "temporal_half_life_days": getattr(rag, "temporal_half_life_days", 7.0),
                    "access_boost_enabled": getattr(rag, "access_boost_enabled", True),
                    "access_boost_weight": getattr(rag, "access_boost_weight", 0.05),
                    "access_boost_cap": getattr(rag, "access_boost_cap", 0.25),
                    "access_boost_half_life_days": getattr(rag, "access_boost_half_life_days", 30.0),
                }
            )
        except Exception:
            logger.debug("Using default RAG pipeline settings", exc_info=True)
        return defaults

    def _build_entity_boost_config(self, query: str, settings: dict[str, object] | None = None):
        settings = settings or self._load_rag_pipeline_settings()
        if not bool(settings.get("entity_boost_enabled", True)):
            return None
        registry_enabled = bool(settings.get("entity_registry_enabled", True))
        query_entities: tuple[str, ...] = ()
        if registry_enabled:
            try:
                from core.memory.entity_index import match_query_entities

                query_entities = tuple(sorted(match_query_entities(self._anima_dir, query)))
            except Exception:
                logger.debug("Failed to match query entities from registry", exc_info=True)
        from core.memory.retrieval.entity import EntityBoostConfig

        related_boost_raw = settings.get("entity_related_boost")
        related_boost = float(related_boost_raw) if related_boost_raw is not None else None
        return EntityBoostConfig(
            enabled=True,
            boost=float(settings.get("entity_boost", 0.20) or 0.0),
            max_boost=float(settings.get("entity_boost_cap", 0.80) or 0.0),
            category=None,
            query_entities=query_entities,
            require_query_entities=registry_enabled,
            anima_dir=self._anima_dir if registry_enabled else None,
            related_boost=related_boost,
        )

    def _build_access_boost_config(self, settings: dict[str, object] | None = None):
        settings = settings or self._load_rag_pipeline_settings()
        if not bool(settings.get("access_boost_enabled", True)):
            return None
        from core.memory.retrieval.access_boost import AccessBoostConfig

        return AccessBoostConfig(
            enabled=True,
            weight=float(settings.get("access_boost_weight", 0.05) or 0.0),
            cap=float(settings.get("access_boost_cap", 0.25) or 0.0),
            half_life_days=float(settings.get("access_boost_half_life_days", 30.0) or 30.0),
        )

    def _graph_episodes_search(
        self,
        query: str,
        pool_k: int,
        knowledge_dir: Path,
        *,
        embedding: list[float] | None = None,
        indexer: Any | None = None,
        access_batch=None,
    ) -> list[dict]:
        """Episodes vector search with graph spreading activation."""
        if not self._load_rag_pipeline_settings().get("enable_spreading_activation", True):
            return []
        if indexer is None:
            indexer = self._get_indexer()
        if indexer is None:
            return []

        anima_name = self._anima_dir.name
        retriever = self._get_retriever(indexer, knowledge_dir)
        try:
            saved = access_batch.take_episode_graph_results(query, pool_k) if access_batch is not None else None
            if saved is not None:
                results, already_expanded = saved
                rag_results = results if already_expanded else retriever.expand_search_results(results, anima_name)
            else:
                rag_results = retriever.search(
                    query=query,
                    anima_name=anima_name,
                    memory_type="episodes",
                    top_k=pool_k,
                    enable_spreading_activation=True,
                    embedding=embedding,
                    access_batch=access_batch,
                )
        except Exception:
            logger.debug("graph episodes search failed", exc_info=True)
            return []

        out: list[dict] = []
        for r in rag_results:
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            item = {
                "doc_id": r.doc_id,
                "source_file": meta.get("source_file", r.doc_id),
                "content": r.content,
                "score": r.score,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "total_chunks": int(meta.get("total_chunks", 1)),
                "memory_type": str(meta.get("memory_type", "episodes") or "episodes"),
                "search_method": "vector_graph",
            }
            for key in (
                "fact_id",
                "edge_type",
                "source_entity",
                "target_entity",
                "valid_at_iso",
                "valid_at",
                "event_time_iso",
                "event_time_text",
                "event_time_parse_error",
                "valid_until",
                "source_episode",
                "source_session_id",
                "access_count",
                "retrieved_count",
                "used_count",
                "last_accessed_at",
                "last_retrieved_at",
                "last_used_at",
                "anima",
                "created_at",
                "updated_at",
                "recorded_at",
                "origin",
                "confidence",
            ):
                if key in meta:
                    item[key] = meta[key]
            out.append(item)
        return out

    def _vector_search_primary(
        self,
        query: str,
        scope: str,
        offset: int,
        knowledge_dir: Path,
        *,
        result_limit: int | None = None,
        entity_boost=None,
        embedding: list[float] | None = None,
        access_batch=None,
    ) -> list[dict]:
        """Perform vector search as primary result source."""
        if self._indexer is None:
            return []

        anima_name = self._anima_dir.name
        retriever = self._get_retriever(self._indexer, knowledge_dir)

        include_shared = scope in ("common_knowledge", "skills", "all")
        all_results: list[dict] = []
        tokens = [tok for tok in query.lower().split() if tok]
        page_size = result_limit if result_limit is not None else 10

        for memory_type in self._resolve_search_types(scope):
            if result_limit is not None:
                fetch_k = result_limit
            else:
                per_type = _EPISODES_TOP_K if memory_type == "episodes" else _DEFAULT_TOP_K
                fetch_k = offset + per_type

            rag_results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type=memory_type,
                top_k=fetch_k,
                include_shared=include_shared,
                embedding=embedding,
                access_batch=access_batch,
            )
            rag_results = [
                result
                for result in rag_results
                if self._company_source_is_visible(str(result.metadata.get("source_file", "")))
            ]

            if rag_results:
                if access_batch is None:
                    retriever.record_access(
                        rag_results,
                        anima_name,
                        kind="retrieved",
                        use_result_metadata=True,
                    )
                else:
                    access_batch.record(rag_results, anima_name, kind="retrieved")

            for r in rag_results:
                score = r.score
                if tokens:
                    content_lower = r.content.lower()
                    matched = sum(1 for tok in tokens if _keyword_token_matches(tok, content_lower))
                    overlap_ratio = matched / len(tokens)
                    score += WEIGHT_TOKEN_OVERLAP * overlap_ratio

                item = {
                    "doc_id": r.doc_id,
                    "source_file": r.metadata.get("source_file", r.doc_id),
                    "content": r.content,
                    "score": score,
                    "chunk_index": int(r.metadata.get("chunk_index", 0)),
                    "total_chunks": int(r.metadata.get("total_chunks", 1)),
                    "memory_type": r.metadata.get("memory_type", memory_type),
                    "search_method": "vector",
                }
                for key in (
                    "fact_id",
                    "edge_type",
                    "source_entity",
                    "target_entity",
                    "valid_at_iso",
                    "valid_at",
                    "event_time_iso",
                    "event_time_text",
                    "event_time_parse_error",
                    "valid_until",
                    "source_episode",
                    "source_session_id",
                    "entities",
                    "access_count",
                    "retrieved_count",
                    "used_count",
                    "last_accessed_at",
                    "last_retrieved_at",
                    "last_used_at",
                    "anima",
                    "created_at",
                    "updated_at",
                    "recorded_at",
                    "origin",
                    "confidence",
                ):
                    if key in r.metadata:
                        item[key] = r.metadata[key]
                all_results.append(item)

        if entity_boost is not None:
            from core.memory.retrieval.entity import apply_entity_boost

            all_results = apply_entity_boost(query, all_results, entity_boost)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        if result_limit is not None:
            return all_results[:result_limit]
        return all_results[offset : offset + page_size]

    def _company_source_is_visible(self, source_file: str) -> bool:
        path = Path(source_file)
        if path.parts[:1] != ("companies",):
            return True
        resources = get_company_resources(self._anima_dir)
        return (
            resources is not None
            and ".." not in path.parts
            and path.parts[:3]
            in {
                ("companies", resources.name, "knowledge"),
                ("companies", resources.name, "skills"),
            }
        )

    def _keyword_search_fallback(
        self,
        query: str,
        scope: str,
        offset: int,
        *,
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
        result_limit: int | None = None,
        entity_boost=None,
        skip_bm25_validation: bool = False,
    ) -> list[dict]:
        """Sparse keyword search used alongside vectors and as fallback.

        Long-term personal memory uses the persisted BM25 corpus. Shared
        common_knowledge, skills/common_skills, facts, and conversation summary
        keep the legacy file scan because they are outside the per-anima
        long-term BM25 index.
        """
        dirs: list[tuple[Path, str]] = []
        longterm_types: list[str] = []
        if scope in ("knowledge", "all"):
            longterm_types.append("knowledge")
        if scope in ("episodes", "all"):
            longterm_types.append("episodes")
        if scope in ("procedures", "all"):
            longterm_types.append("procedures")
        if scope in ("common_knowledge", "all"):
            if common_knowledge_dir.is_dir():
                dirs.append((common_knowledge_dir, "common_knowledge"))
            company_resources = get_company_resources(self._anima_dir)
            if company_resources is not None and company_resources.knowledge_dir.is_dir():
                dirs.append((company_resources.knowledge_dir, "common_knowledge"))

        tokens = [tok for tok in query.lower().split() if tok]
        if not tokens:
            return []

        page_size = result_limit if result_limit is not None else 10
        fetch_limit = offset + page_size
        file_scores: dict[str, dict] = {}

        bm25_hits: list[dict] = []
        if longterm_types and search_longterm_memory_bm25 is not None:
            if not skip_bm25_validation:
                self._maybe_rebuild_dirty_longterm_bm25()
            try:
                bm25_hits = search_longterm_memory_bm25(
                    self._anima_dir,
                    query,
                    memory_types=tuple(longterm_types),
                    top_k=fetch_limit,
                    offset=0,
                    validate_sources=not skip_bm25_validation,
                )
            except Exception:
                logger.debug("Long-term BM25 search failed", exc_info=True)
        for hit in bm25_hits:
            key = f"{hit.get('source_file', '')}#{hit.get('chunk_index', '')}"
            if key not in file_scores or float(file_scores[key].get("score", 0.0) or 0.0) < float(
                hit.get("score", 0.0) or 0.0
            ):
                file_scores[key] = hit

        if not bm25_hits:
            for memory_type in longterm_types:
                if memory_type == "knowledge":
                    dirs.append((knowledge_dir, "knowledge"))
                elif memory_type == "episodes":
                    dirs.append((episodes_dir, "episodes"))
                elif memory_type == "procedures":
                    dirs.append((procedures_dir, "procedures"))

        for d, memory_type in dirs:
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
                try:
                    content, stat = _read_keyword_file(f)
                except OSError:
                    continue
                if memory_type == "knowledge" and self._knowledge_file_is_superseded(f, content=content):
                    continue
                content_lower = content.lower()
                matched = sum(1 for tok in tokens if _keyword_token_matches(tok, content_lower))
                if matched == 0:
                    continue
                score = matched / len(tokens)
                rel_path = company_resource_pointer(f) or f"{memory_type}/{f.name}"
                if rel_path not in file_scores or file_scores[rel_path]["score"] < score:
                    lines = content.split("\n")
                    preview = "\n".join(lines[:30])
                    file_scores[rel_path] = {
                        "source_file": rel_path,
                        "content": preview,
                        "score": score,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "memory_type": memory_type,
                        "search_method": "keyword_fallback",
                        **self._keyword_file_metadata(f, memory_type, content=content, stat=stat),
                    }

        if scope in ("skills", "all"):
            for hit in self._keyword_search_skills(tokens):
                rel_path = hit["source_file"]
                if rel_path not in file_scores or file_scores[rel_path]["score"] < hit["score"]:
                    file_scores[rel_path] = hit

        if scope in ("facts", "all"):
            for hit in self._keyword_search_facts(query, tokens):
                rel_path = f"{hit['source_file']}:{hit.get('fact_id', '')}"
                if rel_path not in file_scores or file_scores[rel_path]["score"] < hit["score"]:
                    file_scores[rel_path] = hit

        if scope in ("all", "conversation_summary"):
            conv_file = self._anima_dir / "state" / "conversation.json"
            if conv_file.is_file():
                try:
                    conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                    summary = conv_data.get("compressed_summary", "")
                    if summary:
                        content_lower = summary.lower()
                        matched = sum(1 for tok in tokens if _keyword_token_matches(tok, content_lower))
                        if matched > 0:
                            score = matched / len(tokens)
                            file_scores["conversation_summary"] = {
                                "source_file": "state/conversation.json",
                                "content": summary[:2000],
                                "score": score,
                                "chunk_index": 0,
                                "total_chunks": 1,
                                "memory_type": "conversation_summary",
                                "search_method": "keyword_fallback",
                            }
                except Exception as e:
                    logger.debug("Failed to read conversation summary: %s", e)

        results = list(file_scores.values())
        if entity_boost is not None:
            from core.memory.retrieval.entity import apply_entity_boost

            results = apply_entity_boost(query, results, entity_boost)
        else:
            results.sort(key=lambda x: x["score"], reverse=True)
        return results[offset : offset + page_size]

    @staticmethod
    def _knowledge_file_is_superseded(path: Path, *, content: str | None = None) -> bool:
        try:
            from core.memory.frontmatter import parse_frontmatter

            meta, _ = parse_frontmatter(content if content is not None else path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to inspect knowledge validity for keyword search: %s", path, exc_info=True)
            return False
        return bool(meta.get("valid_until"))

    @staticmethod
    def _resolve_search_types(scope: str) -> list[str]:
        """Map scope to memory_type list for vector search."""
        if scope == "knowledge":
            return ["knowledge"]
        if scope == "episodes":
            return ["episodes"]
        if scope == "procedures":
            return ["procedures"]
        if scope == "common_knowledge":
            return ["knowledge"]
        if scope == "skills":
            return ["skills"]
        if scope == "conversation_summary":
            return ["conversation_summary"]
        if scope == "facts":
            return ["facts"]
        if scope == "all":
            return ["facts", "knowledge", "episodes", "procedures", "skills", "conversation_summary"]
        return ["knowledge"]

    def _keyword_search_skills(self, tokens: list[str]) -> list[dict]:
        """Sparse keyword search over personal and shared SKILL.md files.

        Used when vector search is unavailable. Applies the same curator deny
        constraints as the vector path (indexer skip + retriever filter).
        """
        if not tokens:
            return []

        try:
            from core.memory.rag.retriever import _load_skill_document_cached
            from core.skills.curator import curator_allows_access, replay_curator_state
        except ImportError:
            return []

        try:
            replay = replay_curator_state(self._anima_dir)
        except Exception:
            logger.debug("Failed to replay curator state for skill keyword search", exc_info=True)
            replay = None

        skill_roots: list[tuple[Path, str, Path]] = []
        personal_skills = self._anima_dir / "skills"
        if personal_skills.is_dir():
            skill_roots.append((personal_skills, "skills", self._anima_dir))
        if self._common_skills_dir.is_dir():
            # source_file for shared skills is ``common_skills/<name>/SKILL.md``
            # relative to the data dir parent of common_skills/.
            skill_roots.append((self._common_skills_dir, "common_skills", self._common_skills_dir.parent))
        company_resources = get_company_resources(self._anima_dir)
        if company_resources is not None and company_resources.skills_dir.is_dir():
            skill_roots.append((company_resources.skills_dir, "common_skills", infer_data_dir(self._anima_dir)))

        results: list[dict] = []
        for root_dir, memory_type, rel_base in skill_roots:
            for skill_path in sorted(root_dir.rglob("SKILL.md")):
                if not skill_path.is_file():
                    continue
                try:
                    meta, content = _load_skill_document_cached(skill_path)
                    allowed, _reason = curator_allows_access(meta, replay=replay)
                except Exception:
                    logger.debug(
                        "Failed to evaluate skill curator access for keyword search: %s",
                        skill_path,
                        exc_info=True,
                    )
                    continue
                if not allowed:
                    continue
                content_lower = content.lower()
                matched = sum(1 for tok in tokens if _keyword_token_matches(tok, content_lower))
                if matched == 0:
                    continue
                score = matched / len(tokens)
                try:
                    rel_path = str(skill_path.relative_to(rel_base)).replace("\\", "/")
                except ValueError:
                    rel_path = f"{memory_type}/{skill_path.parent.name}/SKILL.md"
                lines = content.split("\n")
                preview = "\n".join(lines[:30])
                results.append(
                    {
                        "source_file": rel_path,
                        "content": preview,
                        "score": score,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "memory_type": memory_type,
                        "search_method": "keyword_fallback",
                        **self._keyword_file_metadata(skill_path, memory_type),
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _keyword_search_facts(self, query: str, tokens: list[str] | None = None) -> list[dict]:
        """Keyword search over active legacy facts JSONL records."""
        del query
        tokens = tokens or []
        if not tokens:
            return []

        try:
            from core.memory.facts import FactRecord
        except ImportError:
            return []

        facts_dir = self._anima_dir / "facts"
        if not facts_dir.is_dir():
            return []

        results: list[dict] = []
        for path in sorted(facts_dir.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    record = FactRecord.from_json_line(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not record.is_active():
                    continue
                searchable = "\n".join(
                    [
                        record.text,
                        record.source_entity,
                        record.target_entity,
                        record.edge_type,
                        " ".join(record.entities),
                    ]
                ).lower()
                matched = sum(1 for tok in tokens if tok in searchable)
                if matched == 0:
                    continue
                score = matched / len(tokens)
                results.append(
                    {
                        "source_file": f"facts/{path.name}",
                        "content": record.text,
                        "score": score,
                        "chunk_index": line_no - 1,
                        "total_chunks": len(lines),
                        "memory_type": "facts",
                        "search_method": "keyword_fallback",
                        "fact_id": record.fact_id,
                        "edge_type": record.edge_type,
                        "source_entity": record.source_entity,
                        "target_entity": record.target_entity,
                        "valid_at_iso": record.valid_at,
                        "event_time_iso": record.valid_at,
                        "valid_until": record.valid_until,
                        "recorded_at": record.recorded_at,
                        "source_episode": record.source_episode,
                        "source_session_id": record.source_session_id,
                        "entities": list(record.entities),
                        "confidence": record.confidence,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def _keyword_file_metadata(
        path: Path,
        memory_type: str,
        *,
        content: str | None = None,
        stat: os.stat_result | None = None,
    ) -> dict[str, str]:
        metadata: dict[str, str] = {}
        try:
            metadata["updated_at"] = datetime.fromtimestamp(
                stat.st_mtime if stat is not None else path.stat().st_mtime,
                tz=UTC,
            ).isoformat()
        except OSError:
            return metadata
        if memory_type not in {"knowledge", "common_knowledge"}:
            return metadata
        try:
            from core.memory.frontmatter import parse_frontmatter

            meta, _ = parse_frontmatter(content if content is not None else path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to inspect knowledge metadata for keyword search: %s", path, exc_info=True)
            return metadata
        for key in ("updated_at", "origin"):
            raw_value = meta.get(key, "")
            value = raw_value.isoformat() if hasattr(raw_value, "isoformat") else str(raw_value or "").strip()
            if value:
                metadata[key] = value
        return metadata

    def search_knowledge(self, query: str, knowledge_dir: Path) -> list[tuple[str, str]]:
        """Search knowledge/ by keyword (OR-split on whitespace tokens)."""
        results: list[tuple[str, str]] = []
        tokens = [tok for tok in query.lower().split() if tok]
        if not tokens:
            return results
        for f in knowledge_dir.glob("*.md"):
            for line in f.read_text(encoding="utf-8").splitlines():
                line_lower = line.lower()
                if any(tok in line_lower for tok in tokens):
                    results.append((f.name, line.strip()))
        logger.debug("search_knowledge query='%s' results=%d", query, len(results))
        return results

    def search_procedures(
        self,
        query: str,
        procedures_dir: Path,
    ) -> list[tuple[str, str]]:
        """Search procedures/ by keyword (delegates to search_memory_text)."""
        return self.search_memory_text(
            query,
            scope="procedures",
            knowledge_dir=procedures_dir.parent / "knowledge",
            episodes_dir=procedures_dir.parent / "episodes",
            procedures_dir=procedures_dir,
            common_knowledge_dir=self._common_knowledge_dir,
        )

    def index_file(self, path: Path, memory_type: str, *, force: bool = False, origin: str = "") -> None:
        """Index a single file if indexer is available."""
        indexer = self._get_indexer()
        if indexer:
            try:
                indexer.index_file(path, memory_type, force=force, origin=origin)
                outcome = getattr(indexer, "_last_index_file_outcome", None)
                if self._auto_index_on_access and getattr(outcome, "status", None) == "failed":
                    self._schedule_index_retry()
            except Exception as e:
                if self._auto_index_on_access:
                    self._schedule_index_retry()
                logger.warning("Failed to index %s file: %s", memory_type, e)
        self._update_longterm_bm25_source(path, memory_type)

    def _update_longterm_bm25_source(self, path: Path, memory_type: str) -> None:
        try:
            from core.memory.bm25 import LONGTERM_BM25_MEMORY_TYPES, update_longterm_bm25_source

            if memory_type in LONGTERM_BM25_MEMORY_TYPES:
                update_longterm_bm25_source(self._anima_dir, str(path.relative_to(self._anima_dir)))
        except Exception:
            logger.debug("Failed to update long-term BM25 index for %s", self._anima_dir.name, exc_info=True)

    def _maybe_rebuild_dirty_longterm_bm25(self) -> None:
        """Rebuild the long-term BM25 index before searching if it is stale (F14).

        The dirty marker is written on every memory save but was otherwise never
        consumed, leaving the BM25 corpus up to a full day out of date. A
        cooldown inside ``maybe_rebuild_dirty_longterm_bm25`` keeps this from
        rebuilding on every query.
        """
        try:
            from core.memory.bm25 import maybe_rebuild_dirty_longterm_bm25

            maybe_rebuild_dirty_longterm_bm25(self._anima_dir)
        except Exception:
            logger.debug("On-demand long-term BM25 rebuild check failed for %s", self._anima_dir.name, exc_info=True)
