from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("animaworks.memory")


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
        knowledge_dir: Path,
        episodes_dir: Path,
        procedures_dir: Path,
        common_knowledge_dir: Path,
    ) -> list[tuple[str, str]]:
        """Search memory files by keyword and optional vector similarity.

        Returns ``(filename, matching_line)`` pairs.

        *scope* can be ``"knowledge"``, ``"episodes"``, ``"procedures"``,
        ``"common_knowledge"``, or ``"all"`` (default).
        """
        dirs: list[Path] = []
        if scope in ("knowledge", "all"):
            dirs.append(knowledge_dir)
        if scope in ("episodes", "all"):
            dirs.append(episodes_dir)
        if scope in ("procedures", "all"):
            dirs.append(procedures_dir)
        if scope in ("common_knowledge", "all"):
            if common_knowledge_dir.is_dir():
                dirs.append(common_knowledge_dir)

        # Keyword search — OR-split: match any whitespace-separated token
        results: list[tuple[str, str]] = []
        tokens = [tok for tok in query.lower().split() if tok]
        if not tokens:
            return results
        for d in dirs:
            for f in d.glob("*.md"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    line_lower = line.lower()
                    if any(tok in line_lower for tok in tokens):
                        results.append((f.name, line.strip()))

        # Search compressed_summary from conversation.json
        if scope in ("all", "conversation_summary"):
            conv_file = self._anima_dir / "state" / "conversation.json"
            if conv_file.is_file():
                try:
                    conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                    summary = conv_data.get("compressed_summary", "")
                    if summary:
                        for line in summary.splitlines():
                            line_lower = line.lower()
                            if any(tok in line_lower for tok in tokens) and line.strip():
                                results.append(("conversation_summary", line.strip()))
                except Exception as e:
                    logger.debug("Failed to search conversation_summary: %s", e)

        # Hybrid: append vector search results when RAG is available
        if self._indexer is not None and scope in (
            "knowledge",
            "episodes",
            "common_knowledge",
            "procedures",
            "conversation_summary",
            "all",
        ):
            try:
                vector_hits = self._vector_search_memory(query, scope, knowledge_dir)
                seen_files = {r[0] for r in results}
                for fname, snippet in vector_hits:
                    if fname not in seen_files:
                        results.append((fname, snippet))
                        seen_files.add(fname)
            except Exception as e:
                logger.debug("Vector search augmentation failed: %s", e)

        return results

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

    def _vector_search_memory(
        self,
        query: str,
        scope: str,
        knowledge_dir: Path,
    ) -> list[tuple[str, str]]:
        """Perform vector search to augment keyword results."""
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
        hits: list[tuple[str, str]] = []

        for memory_type in self._resolve_search_types(scope):
            rag_results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type=memory_type,
                top_k=5,
                include_shared=include_shared,
            )

            # Record access (Hebbian LTP)
            if rag_results:
                retriever.record_access(rag_results, anima_name)

            for r in rag_results:
                source = r.metadata.get("source_file", r.doc_id)
                first_line = r.content.split("\n", 1)[0].strip()
                hits.append((str(source), first_line))

        return hits

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
