from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("animaworks.memory")

try:
    from core.memory.bm25 import reciprocal_rank_fusion, search_activity_log
except ImportError:
    search_activity_log = None  # type: ignore[assignment,misc]
    reciprocal_rank_fusion = None  # type: ignore[assignment,misc]

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
                    logger.warning("Failed to index conversation_summary: %s", e)

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

        Skips re-indexing when the directory hash matches the stored value.
        """
        ck_dir = self._common_knowledge_dir
        if not ck_dir.is_dir() or not any(ck_dir.rglob("*.md")):
            logger.debug("No common_knowledge files found, skipping shared indexing")
            return

        meta_path = self._anima_dir / "index_meta.json"
        current_hash = _compute_dir_hash(ck_dir, "*.md")
        stored_hash = _read_shared_hash(meta_path, "shared_common_knowledge_hash")
        if current_hash == stored_hash:
            logger.debug("common_knowledge unchanged (hash match), skipping")
            return

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
            indexed = shared_indexer.index_directory(ck_dir, "common_knowledge")
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

        Skips re-indexing when the directory hash matches the stored value.
        """
        cs_dir = self._common_skills_dir
        if not cs_dir.is_dir() or not any(cs_dir.rglob("SKILL.md")):
            logger.debug("No common_skills files found, skipping shared skills indexing")
            return

        meta_path = self._anima_dir / "index_meta.json"
        current_hash = _compute_dir_hash(cs_dir, "SKILL.md")
        stored_hash = _read_shared_hash(meta_path, "shared_common_skills_hash")
        if current_hash == stored_hash:
            logger.debug("common_skills unchanged (hash match), skipping")
            return

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
            indexed = shared_indexer.index_directory(cs_dir, "common_skills")
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
    ) -> list[dict]:
        """Search memory by vector similarity with keyword fallback.

        Returns ranked results as dicts with score, content, and metadata.
        Vector search is primary; keyword OR search is fallback only
        (activated when ChromaDB is unavailable).
        """
        offset = max(0, min(offset, 50))

        if scope == "activity_log":
            if search_activity_log is None:
                return []
            return search_activity_log(
                self._anima_dir,
                query,
                top_k=10,
                offset=offset,
            )

        primary_results: list[dict] = []
        indexer = self._get_indexer()
        if indexer is not None:
            try:
                primary_results = self._vector_search_primary(
                    query,
                    scope,
                    offset,
                    knowledge_dir,
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
            )

        if scope == "all" and reciprocal_rank_fusion is not None and search_activity_log is not None:
            try:
                bm25_results = search_activity_log(
                    self._anima_dir,
                    query,
                    top_k=10,
                    offset=0,
                )
            except Exception:
                logger.debug("activity_log BM25 search failed", exc_info=True)
                bm25_results = []

            if bm25_results:
                return reciprocal_rank_fusion(
                    primary_results,
                    bm25_results,
                    k=60,
                )[:10]

        return primary_results

    def _vector_search_primary(
        self,
        query: str,
        scope: str,
        offset: int,
        knowledge_dir: Path,
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

        include_shared = scope in ("common_knowledge", "all")
        all_results: list[dict] = []
        tokens = [tok for tok in query.lower().split() if tok]

        for memory_type in self._resolve_search_types(scope):
            top_k = _EPISODES_TOP_K if memory_type == "episodes" else _DEFAULT_TOP_K
            fetch_k = offset + top_k

            rag_results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type=memory_type,
                top_k=fetch_k,
                include_shared=include_shared,
            )

            if rag_results:
                retriever.record_access(rag_results, anima_name)

            for r in rag_results:
                score = r.score
                if tokens:
                    content_lower = r.content.lower()
                    matched = sum(1 for tok in tokens if tok in content_lower)
                    overlap_ratio = matched / len(tokens)
                    score += WEIGHT_TOKEN_OVERLAP * overlap_ratio

                all_results.append(
                    {
                        "source_file": r.metadata.get("source_file", r.doc_id),
                        "content": r.content,
                        "score": score,
                        "chunk_index": int(r.metadata.get("chunk_index", 0)),
                        "total_chunks": int(r.metadata.get("total_chunks", 1)),
                        "memory_type": r.metadata.get("memory_type", memory_type),
                        "search_method": "vector",
                    }
                )

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[offset : offset + 10]

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
    ) -> list[dict]:
        """Keyword OR search with scoring. Used only when vector search is unavailable."""
        dirs: list[tuple[Path, str]] = []
        if scope in ("knowledge", "all"):
            dirs.append((knowledge_dir, "knowledge"))
        if scope in ("episodes", "all"):
            dirs.append((episodes_dir, "episodes"))
        if scope in ("procedures", "all"):
            dirs.append((procedures_dir, "procedures"))
        if scope in ("common_knowledge", "all"):
            if common_knowledge_dir.is_dir():
                dirs.append((common_knowledge_dir, "common_knowledge"))

        tokens = [tok for tok in query.lower().split() if tok]
        if not tokens:
            return []

        file_scores: dict[str, dict] = {}

        for d, memory_type in dirs:
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
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
                    }

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

        results = sorted(file_scores.values(), key=lambda x: x["score"], reverse=True)
        return results[offset : offset + 10]

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
        if scope == "conversation_summary":
            return ["conversation_summary"]
        if scope == "all":
            return ["knowledge", "episodes", "procedures", "conversation_summary"]
        return ["knowledge"]

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

    def index_file(self, path: Path, memory_type: str, *, origin: str = "") -> None:
        """Index a single file if indexer is available."""
        indexer = self._get_indexer()
        if indexer:
            try:
                indexer.index_file(path, memory_type, origin=origin)
            except Exception as e:
                logger.warning("Failed to index %s file: %s", memory_type, e)
