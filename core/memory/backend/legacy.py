from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy backend wrapping the existing ChromaDB + RAG memory system."""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.memory.backend.base import MemoryBackend, RetrievedMemory

if TYPE_CHECKING:
    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever
    from core.memory.rag.store import VectorStore
    from core.memory.rag_search import RAGMemorySearch

logger = logging.getLogger(__name__)

# ── Scope mapping ──────────────────────────────────────────────────────────

_SCOPE_TO_MEMORY_TYPE: dict[str, str] = {
    "knowledge": "knowledge",
    "episodes": "episodes",
    "procedures": "procedures",
    "common_knowledge": "common_knowledge",
    "skills": "skills",
    "all": "all",
}

_PATH_PART_TO_MEMORY_TYPE: dict[str, str] = {
    "episodes": "episodes",
    "knowledge": "knowledge",
    "procedures": "procedures",
    "skills": "skills",
    "common_knowledge": "common_knowledge",
    "common_skills": "common_skills",
}


# ── LegacyRAGBackend ──────────────────────────────────────────────────────


class LegacyRAGBackend(MemoryBackend):
    """Legacy backend wrapping existing ChromaDB + RAG.

    Delegates to :class:`RAGMemorySearch`, :class:`MemoryRetriever`, and
    :class:`MemoryIndexer` while exposing the unified :class:`MemoryBackend`
    interface.  All heavy components are lazily initialised to avoid startup
    costs and circular imports.
    """

    def __init__(
        self,
        anima_dir: Path,
        *,
        common_knowledge_dir: Path | None = None,
        common_skills_dir: Path | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_dir.name

        from core.paths import get_data_dir

        data_dir = get_data_dir()
        self._common_knowledge_dir = common_knowledge_dir or (data_dir / "common_knowledge")
        self._common_skills_dir = common_skills_dir or (data_dir / "common_skills")

        self._rag_search: RAGMemorySearch | None = None
        self._retriever: MemoryRetriever | None = None
        self._indexer: MemoryIndexer | None = None
        self._vector_store: VectorStore | None = None

    # ── Lazy initialisation helpers ────────────────────────────────────────

    def _ensure_rag_search(self) -> RAGMemorySearch:
        """Return a lazily initialised :class:`RAGMemorySearch`."""
        if self._rag_search is None:
            from core.memory.rag_search import RAGMemorySearch

            self._rag_search = RAGMemorySearch(
                self._anima_dir,
                self._common_knowledge_dir,
                self._common_skills_dir,
            )
        return self._rag_search

    def _ensure_vector_store(self) -> VectorStore | None:
        """Return a lazily initialised :class:`VectorStore` (or ``None``)."""
        if self._vector_store is None:
            try:
                from core.memory.rag.singleton import get_vector_store

                self._vector_store = get_vector_store(self._anima_name)
            except Exception:
                logger.debug("Vector store unavailable", exc_info=True)
        return self._vector_store

    def _ensure_indexer(self) -> MemoryIndexer | None:
        """Return a lazily initialised :class:`MemoryIndexer` (or ``None``)."""
        if self._indexer is None:
            vs = self._ensure_vector_store()
            if vs is None:
                return None
            try:
                from core.memory.rag.indexer import MemoryIndexer

                self._indexer = MemoryIndexer(vs, self._anima_name, self._anima_dir)
            except Exception:
                logger.warning("Failed to create MemoryIndexer", exc_info=True)
        return self._indexer

    def _ensure_retriever(self) -> MemoryRetriever | None:
        """Return a lazily initialised :class:`MemoryRetriever` (or ``None``)."""
        if self._retriever is None:
            vs = self._ensure_vector_store()
            indexer = self._ensure_indexer()
            if vs is None or indexer is None:
                return None
            try:
                from core.memory.rag.retriever import MemoryRetriever

                knowledge_dir = self._anima_dir / "knowledge"
                self._retriever = MemoryRetriever(vs, indexer, knowledge_dir)
            except Exception:
                logger.warning("Failed to create MemoryRetriever", exc_info=True)
        return self._retriever

    # ── MemoryBackend implementation ───────────────────────────────────────

    async def ingest_file(self, path: Path) -> int:
        """Index a file via the existing RAG indexer."""
        memory_type = self._infer_memory_type(path)
        try:
            rag = self._ensure_rag_search()
            count = await asyncio.to_thread(rag.index_file, path, memory_type, origin="")
            return count if isinstance(count, int) else 0
        except Exception:
            logger.warning("ingest_file failed for %s", path, exc_info=True)
            return 0

    async def ingest_text(
        self,
        text: str,
        source: str,
        metadata: dict | None = None,
    ) -> int:
        """Write text to a temp file under the appropriate directory, then index."""
        memory_type = self._infer_memory_type_from_source(source)
        target_dir = self._anima_dir / memory_type
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = source.replace("/", "_").replace("\\", "_")
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        target_path = target_dir / safe_name

        try:
            await asyncio.to_thread(target_path.write_text, text, encoding="utf-8")
            return await self.ingest_file(target_path)
        except Exception:
            logger.warning("ingest_text failed for source=%s", source, exc_info=True)
            return 0

    async def retrieve(
        self,
        query: str,
        *,
        scope: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievedMemory]:
        """Retrieve memories, delegating to retriever or search_memory_text."""
        memory_type = _SCOPE_TO_MEMORY_TYPE.get(scope, "knowledge")

        if scope in ("all", "activity_log"):
            return await self._retrieve_via_search_text(query, scope, limit, min_score)

        retriever = self._ensure_retriever()
        if retriever is None:
            return await self._retrieve_via_search_text(query, scope, limit, min_score)

        try:
            include_shared = scope in ("common_knowledge", "skills", "all")
            rag_min = min_score if min_score > 0.0 else None
            results = await asyncio.to_thread(
                retriever.search,
                query,
                self._anima_name,
                memory_type=memory_type,
                top_k=limit,
                include_shared=include_shared,
                min_score=rag_min,
            )

            if results:
                try:
                    await asyncio.to_thread(retriever.record_access, results, self._anima_name)
                except Exception:
                    logger.debug("record_access failed", exc_info=True)

            return [self._retrieval_result_to_memory(r) for r in results]
        except Exception:
            logger.warning("retrieve via retriever failed, falling back", exc_info=True)
            return await self._retrieve_via_search_text(query, scope, limit, min_score)

    async def delete(self, source: str) -> None:
        """Remove all chunks whose ``source_file`` metadata matches *source*."""
        vs = self._ensure_vector_store()
        if vs is None:
            return

        try:
            collections = await asyncio.to_thread(vs.list_collections)
            prefix = f"{self._anima_name}_"
            for coll_name in collections:
                if not coll_name.startswith(prefix):
                    continue
                try:
                    hits = await asyncio.to_thread(
                        vs.get_by_metadata,
                        coll_name,
                        {"source_file": source},
                        limit=10_000,
                    )
                    if hits:
                        ids = [h.document.id for h in hits]
                        await asyncio.to_thread(vs.delete_documents, coll_name, ids)
                        logger.info(
                            "Deleted %d chunks from %s (source=%s)",
                            len(ids),
                            coll_name,
                            source,
                        )
                except Exception:
                    logger.debug("delete scan failed for %s", coll_name, exc_info=True)
        except Exception:
            logger.warning("delete failed for source=%s", source, exc_info=True)

    async def reset(self) -> None:
        """Delete all collections belonging to this Anima."""
        vs = self._ensure_vector_store()
        if vs is None:
            return

        try:
            collections = await asyncio.to_thread(vs.list_collections)
            prefix = f"{self._anima_name}_"
            for coll_name in collections:
                if coll_name.startswith(prefix):
                    await asyncio.to_thread(vs.delete_collection, coll_name)
                    logger.info("Deleted collection %s", coll_name)
        except Exception:
            logger.warning("reset failed", exc_info=True)

    async def stats(self) -> dict[str, int | float]:
        """Return chunk counts per collection for this Anima."""
        vs = self._ensure_vector_store()
        if vs is None:
            return {"total_chunks": 0, "total_sources": 0}

        total_chunks = 0
        total_sources: set[str] = set()

        try:
            collections = await asyncio.to_thread(vs.list_collections)
            prefix = f"{self._anima_name}_"
            for coll_name in collections:
                if not coll_name.startswith(prefix):
                    continue
                try:
                    docs = await asyncio.to_thread(vs.get_by_metadata, coll_name, {}, limit=100_000)
                    total_chunks += len(docs)
                    for d in docs:
                        src = d.document.metadata.get("source_file")
                        if src:
                            total_sources.add(str(src))
                except Exception:
                    logger.debug("stats scan failed for %s", coll_name, exc_info=True)
        except Exception:
            logger.warning("stats failed", exc_info=True)

        return {"total_chunks": total_chunks, "total_sources": len(total_sources)}

    async def health_check(self) -> bool:
        """ChromaDB is always local — return True if vector store initialises."""
        try:
            vs = self._ensure_vector_store()
            return vs is not None
        except Exception:
            return False

    # ── Optional overrides ─────────────────────────────────────────────────

    async def get_important_chunks(self, limit: int = 20) -> list[RetrievedMemory]:
        """Return ``[IMPORTANT]``-tagged chunks via the retriever."""
        retriever = self._ensure_retriever()
        if retriever is None:
            return []

        try:
            results = await asyncio.to_thread(
                retriever.get_important_chunks,
                self._anima_name,
                include_shared=True,
                limit=limit,
            )
            return [
                RetrievedMemory(
                    content=r.document.content,
                    score=r.score,
                    source=str(r.document.metadata.get("source_file", r.document.id)),
                    metadata={k: v for k, v in r.document.metadata.items() if isinstance(v, (str, int, float, bool))},
                    trust="medium",
                )
                for r in results
            ]
        except Exception:
            logger.warning("get_important_chunks failed", exc_info=True)
            return []

    async def record_access(self, results: list[RetrievedMemory]) -> None:
        """Record access for Hebbian LTP scoring."""
        retriever = self._ensure_retriever()
        if retriever is None or not results:
            return

        try:
            from core.memory.rag.retriever import RetrievalResult

            rag_results = [
                RetrievalResult(
                    doc_id=r.metadata.get("doc_id", r.source),
                    content=r.content,
                    score=r.score,
                    metadata=dict(r.metadata),
                    source_scores={},
                )
                for r in results
            ]
            await asyncio.to_thread(retriever.record_access, rag_results, self._anima_name)
        except Exception:
            logger.debug("record_access failed", exc_info=True)

    async def rebuild_index(self, scope: str | None = None) -> int:
        """Re-index directories for the given scope (or all)."""
        indexer = self._ensure_indexer()
        if indexer is None:
            return 0

        scopes = [scope] if scope else ["knowledge", "episodes", "procedures"]
        total = 0

        for s in scopes:
            target_dir = self._anima_dir / s
            if not target_dir.is_dir():
                continue
            try:
                count = await asyncio.to_thread(indexer.index_directory, target_dir, s, True)
                total += count
            except Exception:
                logger.warning("rebuild_index failed for scope=%s", s, exc_info=True)

        return total

    async def get_community_context(
        self,
        query: str,
        limit: int = 3,
    ) -> list[RetrievedMemory]:
        """Legacy has no community concept — always empty."""
        return []

    async def get_recent_facts(
        self,
        query: str,
        *,
        hours: int = 24,
        limit: int = 10,
    ) -> list[RetrievedMemory]:
        """Search recent activity_log via BM25 as a rough proxy for recent facts."""
        try:
            from core.memory.bm25 import search_activity_log

            results = await asyncio.to_thread(
                search_activity_log,
                query,
                self._anima_dir / "activity_log",
                top_k=limit,
            )
            return [
                RetrievedMemory(
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", "activity_log"),
                    metadata={"search_method": "bm25_activity"},
                    trust="medium",
                )
                for r in results
            ]
        except Exception:
            logger.debug("get_recent_facts via BM25 failed", exc_info=True)
            return []

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _retrieve_via_search_text(
        self,
        query: str,
        scope: str,
        limit: int,
        min_score: float,
    ) -> list[RetrievedMemory]:
        """Fall back to :meth:`RAGMemorySearch.search_memory_text`."""
        rag = self._ensure_rag_search()
        try:
            raw_results = await asyncio.to_thread(
                rag.search_memory_text,
                query,
                scope,
                offset=0,
                context_window=128_000,
                knowledge_dir=self._anima_dir / "knowledge",
                episodes_dir=self._anima_dir / "episodes",
                procedures_dir=self._anima_dir / "procedures",
                common_knowledge_dir=self._common_knowledge_dir,
            )
        except Exception:
            logger.warning("search_memory_text failed", exc_info=True)
            return []

        memories: list[RetrievedMemory] = []
        for d in raw_results[:limit]:
            score = float(d.get("score", 0.0))
            if score < min_score:
                continue
            memories.append(
                RetrievedMemory(
                    content=d.get("content", ""),
                    score=score,
                    source=d.get("source_file", ""),
                    metadata={
                        "memory_type": d.get("memory_type", ""),
                        "search_method": d.get("search_method", ""),
                        "chunk_index": d.get("chunk_index", 0),
                    },
                    trust="medium",
                )
            )
        return memories

    @staticmethod
    def _infer_memory_type(path: Path) -> str:
        """Determine memory_type from file path parts."""
        parts = path.parts
        for part in reversed(parts):
            if part in _PATH_PART_TO_MEMORY_TYPE:
                return _PATH_PART_TO_MEMORY_TYPE[part]
        return "knowledge"

    @staticmethod
    def _infer_memory_type_from_source(source: str) -> str:
        """Determine memory_type from a logical source identifier."""
        for key, mt in _PATH_PART_TO_MEMORY_TYPE.items():
            if key in source:
                return mt
        return "knowledge"

    @staticmethod
    def _retrieval_result_to_memory(r) -> RetrievedMemory:
        """Convert a :class:`RetrievalResult` to :class:`RetrievedMemory`."""
        meta = dict(r.metadata) if r.metadata else {}
        flat_meta: dict[str, str | int | float | bool] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                flat_meta[k] = v
        flat_meta["doc_id"] = r.doc_id
        return RetrievedMemory(
            content=r.content,
            score=r.score,
            source=str(meta.get("source_file", r.doc_id)),
            metadata=flat_meta,
            trust="medium",
        )
