from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from core.memory.fact_observability import warn_rate_limited

logger = logging.getLogger("animaworks.memory")

try:
    from core.memory.bm25 import search_activity_log, search_longterm_memory_bm25
except ImportError:
    search_activity_log = None  # type: ignore[assignment,misc]
    search_longterm_memory_bm25 = None  # type: ignore[assignment,misc]

_EPISODES_TOP_K = 10
_DEFAULT_TOP_K = 5
WEIGHT_TOKEN_OVERLAP = 0.1


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


def _read_shared_hash(meta_path: Path, key: str) -> str | None:
    """Read a stored shared-index hash from *meta_path* (index_meta.json)."""
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get(key)
    except (json.JSONDecodeError, OSError):
        return None


def _write_shared_hash(meta_path: Path, key: str, value: str) -> None:
    """Write a shared-index hash into *meta_path* (index_meta.json)."""
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    meta[key] = value
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _shared_collection_exists(vector_store, collection_name: str) -> bool:
    """Return True if *collection_name* exists in *vector_store*.

    Used to verify shared collections (``shared_common_knowledge`` /
    ``shared_common_skills``) before short-circuiting on hash match.
    Returns ``True`` on listing failure (be conservative — preserve
    legacy behavior on transient errors rather than triggering full
    re-indexing across all animas).
    """
    try:
        return collection_name in vector_store.list_collections()
    except Exception as exc:
        logger.debug("Failed to list collections for existence check: %s", exc)
        return True


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
        self._indexer_initialized = False
        self._last_search_meta: dict[str, object] = {}

    def _init_indexer(self) -> None:
        """Initialize RAG indexer if dependencies are available.

        Called lazily by ``_get_indexer()`` on first access.
        Uses process-level singletons for ChromaVectorStore and embedding
        model to avoid costly repeated initialization.
        """
        self._indexer_initialized = True
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            anima_name = self._anima_dir.name
            vector_store = get_vector_store(anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, indexer disabled")
                return
            self._indexer = MemoryIndexer(vector_store, anima_name, self._anima_dir)
            logger.debug("RAG indexer initialized for anima=%s", anima_name)

            # Index procedures directory alongside knowledge
            procedures_dir = self._anima_dir / "procedures"
            if procedures_dir.is_dir() and any(procedures_dir.glob("*.md")):
                try:
                    indexed = self._indexer.index_directory(
                        procedures_dir,
                        "procedures",
                    )
                    if indexed > 0:
                        logger.debug(
                            "Indexed %d chunks from procedures/",
                            indexed,
                        )
                except Exception as e:
                    logger.warning("Failed to index procedures: %s", e)

            facts_dir = self._anima_dir / "facts"
            if facts_dir.is_dir() and any(facts_dir.glob("*.jsonl")):
                try:
                    indexed = self._indexer.index_directory(
                        facts_dir,
                        "facts",
                    )
                    if indexed > 0:
                        logger.debug(
                            "Indexed %d chunks from facts/",
                            indexed,
                        )
                except Exception as e:
                    warn_rate_limited(
                        logger,
                        "fact_extraction.rag_search_facts_index",
                        "Failed to index facts for anima=%s",
                        anima_name,
                        exc_info=(type(e), e, e.__traceback__),
                    )

            # Index conversation summary (compressed_summary)
            state_dir = self._anima_dir / "state"
            conv_file = state_dir / "conversation.json"
            if conv_file.is_file():
                try:
                    indexed = self._indexer.index_conversation_summary(
                        state_dir,
                        anima_name,
                    )
                    if indexed > 0:
                        logger.debug(
                            "Indexed %d chunks from conversation_summary",
                            indexed,
                        )
                except Exception as e:
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
            logger.warning("Failed to initialize RAG indexer: %s", e)

    # ── Shared collection change detection ────────────────

    def _check_shared_collections(self) -> None:
        """Re-index shared common_knowledge / common_skills if changed.

        Called on every ``_get_indexer()`` access so that file changes are
        picked up even after the initial ``_init_indexer()`` run.  Uses a
        SHA-256 hash of (relative_path, mtime) tuples stored in the
        per-anima ``index_meta.json`` to skip re-indexing when unchanged.
        """
        if self._indexer is None:
            return
        try:
            vector_store = self._indexer.vector_store
            self._ensure_shared_knowledge_indexed(vector_store)
            self._ensure_shared_skills_indexed(vector_store)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Shared collection check failed: %s", e)

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

        meta_path = self._anima_dir / "index_meta.json"
        current_hash = _compute_dir_hash(ck_dir, "*.md")
        stored_hash = _read_shared_hash(meta_path, "shared_common_knowledge_hash")
        force = False
        if current_hash == stored_hash:
            # Verify the shared collection still exists in this anima's
            # vector store; if missing, fall through with force=True so
            # the collection gets recreated.
            if _shared_collection_exists(vector_store, "shared_common_knowledge"):
                logger.debug("common_knowledge unchanged (hash match), skipping")
                return
            logger.info("shared_common_knowledge collection missing despite tracked hash, forcing re-index")
            force = True

        try:
            from core.memory.rag import MemoryIndexer
            from core.paths import get_data_dir

            data_dir = get_data_dir()
            shared_indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=data_dir,
                collection_prefix="shared",
                embedding_model=self._indexer.embedding_model if self._indexer else None,
            )
            indexed = shared_indexer.index_directory(ck_dir, "common_knowledge", force=force)
            _write_shared_hash(meta_path, "shared_common_knowledge_hash", current_hash)
            if indexed > 0:
                logger.info(
                    "Indexed %d chunks into shared_common_knowledge",
                    indexed,
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

        meta_path = self._anima_dir / "index_meta.json"
        current_hash = _compute_dir_hash(cs_dir, "SKILL.md")
        stored_hash = _read_shared_hash(meta_path, "shared_common_skills_hash")
        force = False
        if current_hash == stored_hash:
            if _shared_collection_exists(vector_store, "shared_common_skills"):
                logger.debug("common_skills unchanged (hash match), skipping")
                return
            logger.info("shared_common_skills collection missing despite tracked hash, forcing re-index")
            force = True

        try:
            from core.memory.rag import MemoryIndexer
            from core.paths import get_data_dir

            data_dir = get_data_dir()
            shared_indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=data_dir,
                collection_prefix="shared",
                embedding_model=self._indexer.embedding_model if self._indexer else None,
            )
            # shared_common_skills is a global collection. Per-anima curator
            # archive/block/delete state cannot be applied destructively here
            # without hiding common skills from other Animas, so personal
            # enforcement happens at retrieval time in MemoryRetriever.
            # Trust/security metadata remains enforced by MemoryIndexer.
            indexed = shared_indexer.index_directory(cs_dir, "common_skills", force=force)
            _write_shared_hash(meta_path, "shared_common_skills_hash", current_hash)
            if indexed > 0:
                logger.info(
                    "Indexed %d chunks into shared_common_skills",
                    indexed,
                )
        except Exception as e:
            logger.warning("Failed to index shared common_skills: %s", e)

    def _get_indexer(self):
        """Return the RAG indexer, initializing it on first call.

        Also checks shared collections for changes on every call.
        """
        if not self._indexer_initialized:
            self._init_indexer()
        self._check_shared_collections()
        return self._indexer

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
    ) -> list[dict]:
        """Search memory through the unified Legacy retrieval policy.

        Returns ranked results as dicts with score, content, and metadata while
        preserving the legacy tool result shape.
        """
        offset = max(0, min(offset, 50))
        self._last_search_meta = {}

        if scope == "activity_log":
            if search_activity_log is None:
                return []
            return search_activity_log(
                self._anima_dir,
                query,
                top_k=10,
                offset=offset,
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
            "entity_registry_enabled": True,
            "entity_boost_enabled": False,
            "entity_boost": 0.20,
            "entity_boost_cap": 0.80,
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
                    "rerank_enabled": rag.rerank_enabled,
                    "rerank_candidate_pool": rag.rerank_candidate_pool,
                    "cross_encoder_model": rag.cross_encoder_model,
                    "abstain_on_low_confidence": rag.abstain_on_low_confidence,
                    "confidence_threshold": rag.confidence_threshold,
                    "rrf_confidence_threshold": rag.rrf_confidence_threshold,
                    "entity_registry_enabled": getattr(rag, "entity_registry_enabled", True),
                    "entity_boost_enabled": getattr(rag, "entity_boost_enabled", False),
                    "entity_boost": getattr(rag, "entity_boost", 0.20),
                    "entity_boost_cap": getattr(rag, "entity_boost_cap", 0.80),
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
        if not bool(settings.get("entity_boost_enabled", False)):
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

        return EntityBoostConfig(
            enabled=True,
            boost=float(settings.get("entity_boost", 0.20) or 0.0),
            max_boost=float(settings.get("entity_boost_cap", 0.80) or 0.0),
            category=None,
            query_entities=query_entities,
            require_query_entities=registry_enabled,
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
    ) -> list[dict]:
        """Episodes vector search with graph spreading activation."""
        from core.memory.rag.retriever import MemoryRetriever

        indexer = self._get_indexer()
        if indexer is None:
            return []

        anima_name = self._anima_dir.name
        retriever = MemoryRetriever(
            indexer.vector_store,
            indexer,
            knowledge_dir,
        )
        try:
            rag_results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type="episodes",
                top_k=pool_k,
                enable_spreading_activation=True,
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
    ) -> list[dict]:
        """Perform vector search as primary result source."""
        from core.memory.rag.retriever import MemoryRetriever

        if self._indexer is None:
            return []

        anima_name = self._anima_dir.name
        retriever = MemoryRetriever(
            self._indexer.vector_store,
            self._indexer,
            knowledge_dir,
        )

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
            )

            if rag_results:
                retriever.record_access(rag_results, anima_name, kind="retrieved")

            for r in rag_results:
                score = r.score
                if tokens:
                    content_lower = r.content.lower()
                    matched = sum(1 for tok in tokens if tok in content_lower)
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
    ) -> list[dict]:
        """Sparse keyword search used alongside vectors and as fallback.

        Long-term personal memory uses the persisted BM25 corpus. Shared
        common_knowledge, facts, and conversation summary keep the legacy file
        scan because they are outside the per-anima long-term BM25 index.
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

        tokens = [tok for tok in query.lower().split() if tok]
        if not tokens:
            return []

        page_size = result_limit if result_limit is not None else 10
        fetch_limit = offset + page_size
        file_scores: dict[str, dict] = {}

        bm25_hits: list[dict] = []
        if longterm_types and search_longterm_memory_bm25 is not None:
            try:
                bm25_hits = search_longterm_memory_bm25(
                    self._anima_dir,
                    query,
                    memory_types=tuple(longterm_types),
                    top_k=fetch_limit,
                    offset=0,
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
                if memory_type == "knowledge" and self._knowledge_file_is_superseded(f):
                    continue
                try:
                    content = f.read_text(encoding="utf-8")
                except OSError:
                    continue
                content_lower = content.lower()
                matched = sum(1 for tok in tokens if tok in content_lower)
                if matched == 0:
                    continue
                score = matched / len(tokens)
                rel_path = f"{memory_type}/{f.name}"
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
                        **self._keyword_file_metadata(f, memory_type),
                    }

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
                        matched = sum(1 for tok in tokens if tok in content_lower)
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
    def _knowledge_file_is_superseded(path: Path) -> bool:
        try:
            from core.memory.frontmatter import parse_frontmatter

            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
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
    def _keyword_file_metadata(path: Path, memory_type: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        try:
            metadata["updated_at"] = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        except OSError:
            return metadata
        if memory_type not in {"knowledge", "common_knowledge"}:
            return metadata
        try:
            from core.memory.frontmatter import parse_frontmatter

            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
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
            except Exception as e:
                logger.warning("Failed to index %s file: %s", memory_type, e)
        self._mark_longterm_bm25_dirty(memory_type)

    def _mark_longterm_bm25_dirty(self, memory_type: str) -> None:
        try:
            from core.memory.bm25 import LONGTERM_BM25_MEMORY_TYPES, mark_longterm_bm25_dirty

            if memory_type in LONGTERM_BM25_MEMORY_TYPES:
                mark_longterm_bm25_dirty(self._anima_dir, reason=f"index_file:{memory_type}")
        except Exception:
            logger.debug("Failed to mark long-term BM25 index dirty for %s", self._anima_dir.name, exc_info=True)
