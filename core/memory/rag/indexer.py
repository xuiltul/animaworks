from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory indexing system - converts memory files to vector embeddings.

Handles:
- Chunking strategies (Markdown sections, time-based episodes, whole files)
- Embedding generation (local sentence-transformers)
- Incremental indexing (only update changed files)
- Metadata extraction (tags, importance, timestamps)
"""

import fnmatch
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from core.memory.rag import indexer_delete
from core.memory.rag.episode_time import apply_episode_heading_event_time
from core.memory.rag.facts_chunker import chunk_facts_jsonl
from core.time_utils import ensure_aware, now_iso

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("animaworks.rag.indexer")

INDEX_META_FILE = "index_meta.json"
UPSERT_FAILURE_STATE_FILE = "rag_upsert_failures.json"
STALE_RECONCILIATION_LIMIT = 500


def access_tracking_metadata() -> dict[str, str | int | float]:
    """Return default split access-tracking metadata for new chunks."""
    return {
        "access_count": 0,
        "retrieved_count": 0,
        "used_count": 0,
        "last_accessed_at": "",
        "last_retrieved_at": "",
        "last_used_at": "",
    }


@dataclass
class MemoryChunk:
    """A chunk of memory content ready for indexing."""

    id: str
    content: str
    metadata: dict[str, str | int | float | bool | list[str]]


@dataclass(frozen=True)
class IndexDirectoryResult:
    """Structured outcome for one directory indexing run."""

    chunks_indexed: int = 0
    files_indexed: int = 0
    files_failed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    transient_failures: int = 0
    failed_sources: tuple[str, ...] = ()
    files_reconciled: int = 0

    @property
    def transient(self) -> bool:
        """Return True when every failed file was caused by a transient circuit."""
        return self.files_failed > 0 and self.transient_failures == self.files_failed


@dataclass(frozen=True)
class _IndexFileOutcome:
    status: Literal["indexed", "failed", "unchanged", "skipped"]
    transient: bool = False


# ── MemoryIndexer ───────────────────────────────────────────────────


class MemoryIndexer:
    """Indexes memory files into vector store.

    Manages chunking, embedding generation, and incremental updates.
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        anima_name: str,
        anima_dir: Path,
        embedding_model_name: str | None = None,
        *,
        collection_prefix: str | None = None,
        embedding_model: SentenceTransformer | None = None,
        upsert_quarantine_failure_threshold: int | None = None,
    ) -> None:
        """Initialize indexer.

        Args:
            vector_store: VectorStore instance (e.g., ChromaVectorStore)
            anima_name: Anima name (for collection naming)
            anima_dir: Path to anima's memory directory
            embedding_model_name: Sentence-transformers model name
            collection_prefix: Override for collection name prefix.
                Defaults to anima_name.  Use ``"shared"`` for
                common_knowledge indexing so collection becomes
                ``shared_common_knowledge``.
            embedding_model: Pre-initialized SentenceTransformer instance.
                When provided, ``_init_embedding_model()`` is skipped,
                avoiding redundant model loading.
        """
        self.vector_store = vector_store
        self.anima_name = anima_name
        self.anima_dir = anima_dir
        self.collection_prefix = collection_prefix or anima_name
        self._embedding_model_name_override = embedding_model_name
        if upsert_quarantine_failure_threshold is None:
            try:
                from core.config import load_config

                upsert_quarantine_failure_threshold = load_config().rag.upsert_quarantine_failure_threshold
            except Exception:
                from core.config.models import RAGConfig

                upsert_quarantine_failure_threshold = RAGConfig().upsert_quarantine_failure_threshold
        self.upsert_quarantine_failure_threshold = max(1, upsert_quarantine_failure_threshold)
        self.upsert_failure_state_path = anima_dir / "state" / UPSERT_FAILURE_STATE_FILE

        # Use injected embedding model or initialize via singleton.
        # When ANIMAWORKS_EMBED_URL is set (child processes), skip local
        # model loading — generate_embeddings() handles HTTP delegation.
        import os

        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif os.environ.get("ANIMAWORKS_EMBED_URL"):
            self.embedding_model = None  # type: ignore[assignment]
        else:
            self._init_embedding_model()

        # Load index metadata
        self.meta_path = anima_dir / INDEX_META_FILE
        self.index_meta = self._load_index_meta()
        self._skill_curator_replay = None
        self._skill_curator_state_marker: tuple[int, int] | None = None

        # Cache of collection names known to exist in the vector store.
        # Populated lazily by ``_collection_exists()`` so that hash-based
        # skip optimization can verify the collection still exists before
        # short-circuiting (recovery from a wiped/recreated vectordb).
        self._known_collections: set[str] | None = None

    def _collection_exists(self, name: str) -> bool:
        """Return True if *name* is present in the vector store.

        Lazily populates ``self._known_collections`` from
        ``vector_store.list_collections()`` on first access.  Subsequent
        calls reuse the cache; callers add to the cache after successful
        ``create_collection()`` / ``upsert()`` to avoid repeated listing.

        Returns ``True`` on listing failure to preserve legacy behavior
        (skip when uncertain).
        """
        if self._known_collections is None:
            try:
                self._known_collections = set(self.vector_store.list_collections())
            except Exception as exc:
                logger.debug("Failed to list collections for cache: %s", exc)
                # Be conservative: assume it exists rather than triggering
                # spurious re-indexing of every file on a transient error.
                return True
        return name in self._known_collections

    def _mark_collection_known(self, name: str) -> None:
        """Record that *name* is present in the vector store.

        Called after successful ``create_collection()`` / ``upsert()`` so
        that subsequent ``_collection_exists()`` checks do not need to
        re-list collections.
        """
        if self._known_collections is None:
            try:
                self._known_collections = set(self.vector_store.list_collections())
            except Exception:
                self._known_collections = set()
        self._known_collections.add(name)

    def _init_embedding_model(self) -> None:
        """Initialize sentence-transformers model via process-level singleton."""
        from core.memory.rag.singleton import get_embedding_model

        self.embedding_model = get_embedding_model(self._embedding_model_name_override)

    def _load_index_meta(self) -> dict[str, dict[str, str | int]]:
        """Load index metadata (file hashes and timestamps)."""
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load index metadata: %s", e)
            return {}

    _ragignore_cache: ClassVar[tuple[float, list[str]] | None] = None
    _ragignore_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _load_ragignore(cls) -> list[str]:
        """Load .ragignore patterns from data_dir with mtime caching."""
        from core.paths import get_data_dir

        ragignore_path = get_data_dir() / ".ragignore"
        if not ragignore_path.is_file():
            return []
        try:
            mtime = ragignore_path.stat().st_mtime
            with cls._ragignore_lock:
                if cls._ragignore_cache and cls._ragignore_cache[0] == mtime:
                    return cls._ragignore_cache[1]
            patterns = []
            for line in ragignore_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    patterns.append(stripped)
            with cls._ragignore_lock:
                cls._ragignore_cache = (mtime, patterns)
            return patterns
        except Exception as e:
            logger.warning("Failed to load .ragignore: %s", e)
            return []

    @classmethod
    def is_ragignored(cls, file_path: Path) -> bool:
        """Check if a file matches any .ragignore pattern."""
        patterns = cls._load_ragignore()
        if not patterns:
            return False
        name = file_path.name
        rel_str = str(file_path).replace("\\", "/")
        return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel_str, p) for p in patterns)

    def _save_index_meta(self) -> None:
        """Save index metadata."""
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.index_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save index metadata: %s", e)

    def _load_upsert_failure_state(self) -> dict:
        """Load persistent per-file upsert failures and quarantine history."""
        state_path = getattr(
            self,
            "upsert_failure_state_path",
            self.anima_dir / "state" / UPSERT_FAILURE_STATE_FILE,
        )
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("failures", {})
                data.setdefault("quarantined", [])
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"failures": {}, "quarantined": []}

    def _save_upsert_failure_state(self, state: dict) -> None:
        """Persist failure counters and the inspectable quarantine list."""
        state_path = getattr(
            self,
            "upsert_failure_state_path",
            self.anima_dir / "state" / UPSERT_FAILURE_STATE_FILE,
        )
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(state_path)
        except OSError:
            logger.warning("Failed to save RAG upsert failure state", exc_info=True)

    def _clear_upsert_failure(self, source_file: str, collection_name: str | None = None) -> None:
        """Reset the consecutive counter after a successful upsert."""
        state = self._load_upsert_failure_state()
        if state["failures"].pop(source_file, None) is not None:
            self._save_upsert_failure_state(state)
        if collection_name is not None:
            failures_by_collection = getattr(self, "_run_upsert_failures", None)
            if failures_by_collection is not None:
                failures_by_collection.pop(collection_name, None)

    def _is_upsert_quarantined(self, source_file: str) -> bool:
        """Return whether a source is on the persistent non-destructive skip list."""
        state = self._load_upsert_failure_state()
        return any(item.get("source_file") == source_file for item in state["quarantined"])

    def _record_upsert_failure(self, collection_name: str, source_file: str, file_path: Path) -> None:
        """Increment a persistent counter and add repeat offenders to the skip list.

        An open HTTP write circuit is a service-wide/transient signal and is
        never attributed to an individual file. Two distinct failures in the
        same collection during one indexing run are also treated as a global
        outage; counters recorded by that run are rolled back conservatively.
        """
        transient_probe = getattr(self.vector_store, "is_transient_write_failure", None)
        if callable(transient_probe) and transient_probe(collection_name):
            logger.info("Not counting transient vector service failure for %s", file_path)
            return

        state = self._load_upsert_failure_state()
        if getattr(self, "_index_directory_active", False):
            failures_by_collection = self._run_upsert_failures
            failed_sources = failures_by_collection.setdefault(collection_name, set())
            failed_sources.add(source_file)
            if len(failed_sources) > 1:
                for failed_source in failed_sources:
                    state["failures"].pop(failed_source, None)
                # The first failure in this run may have reached the threshold
                # before a second source proved the outage was collection-wide.
                # Roll back only entries for sources observed in this run;
                # previously quarantined sources are skipped before indexing and
                # therefore cannot appear in ``failed_sources``.
                state["quarantined"] = [
                    item for item in state["quarantined"] if item.get("source_file") not in failed_sources
                ]
                self._save_upsert_failure_state(state)
                logger.info(
                    "Not counting likely service-wide upsert outage: collection=%s failed_sources=%d",
                    collection_name,
                    len(failed_sources),
                )
                return

        previous = state["failures"].get(source_file, {})
        count = int(previous.get("consecutive_failures", 0)) + 1
        state["failures"][source_file] = {
            "consecutive_failures": count,
            "last_failed_at": now_iso(),
        }

        threshold = getattr(self, "upsert_quarantine_failure_threshold", 3)
        if count < threshold:
            self._save_upsert_failure_state(state)
            return

        state["failures"].pop(source_file, None)
        state["quarantined"].append(
            {
                "source_file": source_file,
                "failure_count": count,
                "quarantined_at": now_iso(),
                "reason": "consecutive_upsert_failures",
            }
        )
        self._save_upsert_failure_state(state)
        logger.warning(
            "Quarantined RAG source from indexing after %d consecutive upsert failures (source retained): %s",
            count,
            file_path,
        )

    # ── Main indexing API ───────────────────────────────────────────

    def _finish_index_file(
        self,
        chunks: int,
        status: Literal["indexed", "failed", "unchanged", "skipped"],
        *,
        transient: bool = False,
    ) -> int:
        self._last_index_file_outcome = _IndexFileOutcome(status=status, transient=transient)
        return chunks

    def index_file(
        self,
        file_path: Path,
        memory_type: str,
        force: bool = False,
        origin: str = "",
    ) -> int:
        """Index a single memory file.

        Args:
            file_path: Path to the memory file
            memory_type: Memory type (knowledge, episodes, procedures, skills, shared_users)
            force: Force re-indexing even if file hasn't changed
            origin: Provenance origin category (e.g. "consolidation", "external_platform").
                Stored in chunk metadata for trust-level resolution at retrieval time.

        Returns:
            Number of chunks indexed
        """
        from core import startup_progress

        self._last_index_file_outcome = _IndexFileOutcome(status="failed")
        startup_progress.raise_if_cancelled()
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
            return self._finish_index_file(0, "failed")

        collection_name = f"{self.collection_prefix}_{memory_type}"
        try:
            file_key = str(file_path.relative_to(self.anima_dir))
        except ValueError:
            file_key = str(file_path)

        if self._is_upsert_quarantined(file_key):
            logger.debug("Skipping quarantined RAG source: %s", file_path)
            return self._finish_index_file(0, "skipped")

        # Check .ragignore exclusion. Remove any previously-indexed chunks so
        # a file that matches .ragignore only after indexing does not linger
        # in the collection (mirrors the curator-denied path below).
        if self.is_ragignored(file_path):
            logger.debug("Skipping ragignored file: %s", file_path)
            self.delete_indexed_file(file_path, memory_type)
            return self._finish_index_file(0, "skipped")

        if memory_type in ("skills", "common_skills") and file_path.name == "SKILL.md":
            try:
                from core.skills.curator import curator_allows_access, replay_curator_state
                from core.skills.loader import load_skill_metadata

                meta = load_skill_metadata(file_path)
                curator_state_path = self.anima_dir / "state" / "skill_curator.jsonl"
                state_marker = None
                if curator_state_path.exists():
                    state_stat = curator_state_path.stat()
                    state_marker = (state_stat.st_mtime_ns, state_stat.st_size)
                if self._skill_curator_replay is None or self._skill_curator_state_marker != state_marker:
                    self._skill_curator_replay = replay_curator_state(self.anima_dir)
                    self._skill_curator_state_marker = state_marker
                allowed, reason = curator_allows_access(meta, replay=self._skill_curator_replay)
                if not allowed:
                    logger.info("Skipping non-loadable skill from RAG index: %s (%s)", file_path, reason)
                    self.delete_indexed_file(file_path, memory_type)
                    return self._finish_index_file(0, "skipped")
            except Exception:
                logger.debug("Failed to evaluate skill curator access for %s", file_path, exc_info=True)

        # Check if file has changed
        file_hash = self._compute_file_hash(file_path)

        if not force and file_key in self.index_meta:
            if self.index_meta[file_key].get("hash") == file_hash:
                # Verify the collection still exists in the vector store
                # before short-circuiting.  If the vectordb was wiped or
                # recreated since the last index, the meta hash would
                # otherwise cause us to silently skip and the collection
                # would never be re-created.
                if self._collection_exists(collection_name):
                    logger.debug("File unchanged, skipping: %s", file_path)
                    return self._finish_index_file(0, "unchanged")
                logger.info(
                    "Collection '%s' missing despite tracked hash, forcing re-index of %s",
                    collection_name,
                    file_path,
                )

        logger.info("Indexing file: %s (type=%s)", file_path, memory_type)

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return self._finish_index_file(0, "failed")

        # Chunk the content
        chunks = self._chunk_file(file_path, content, memory_type, origin=origin)

        if not chunks:
            logger.debug("No chunks extracted from %s", file_path)
            self.delete_indexed_file(file_path, memory_type)
            return self._finish_index_file(0, "indexed")

        source_mtime_ns = file_path.stat().st_mtime_ns
        for chunk in chunks:
            chunk.metadata["source_hash"] = file_hash
            chunk.metadata["source_mtime_ns"] = source_mtime_ns

        # Generate embeddings
        embeddings = self._generate_embeddings([chunk.content for chunk in chunks])

        # Build documents
        from core.memory.rag.store import Document

        documents = [
            Document(
                id=chunk.id,
                content=chunk.content,
                embedding=embeddings[i],
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]

        if not indexer_delete.upsert_file_documents(self, collection_name, file_key, file_path, documents):
            transient_probe = getattr(self.vector_store, "is_transient_write_failure", None)
            transient = bool(callable(transient_probe) and transient_probe(collection_name))
            return self._finish_index_file(0, "failed", transient=transient)

        self.index_meta[file_key] = {
            "hash": file_hash,
            "indexed_at": now_iso(),
            "chunks": len(chunks),
        }
        self._save_index_meta()

        logger.info("Indexed %d chunks from %s", len(chunks), file_path)
        return self._finish_index_file(len(chunks), "indexed")

    def delete_indexed_file(self, file_path: Path, memory_type: str) -> int:
        return indexer_delete.delete_indexed_file(self, file_path, memory_type)

    def _delete_indexed_file_documents(self, collection_name: str, source_file: str) -> int | None:
        return indexer_delete.delete_indexed_file_documents(self, collection_name, source_file)

    def index_directory(
        self,
        directory: Path,
        memory_type: str,
        force: bool = False,
    ) -> IndexDirectoryResult:
        """Index all .md files in a directory.

        Args:
            directory: Path to memory directory
            memory_type: Memory type
            force: Force re-indexing

        Returns:
            Structured counts that distinguish unchanged files from failures.
        """
        collection_name = f"{self.collection_prefix}_{memory_type}"
        if not directory.is_dir():
            logger.warning("Directory not found: %s", directory)
            result = IndexDirectoryResult()
            self._log_index_directory_summary(collection_name, result)
            return result

        patterns = {"facts": "*.jsonl", "skills": "SKILL.md", "common_skills": "SKILL.md"}
        md_files = sorted(directory.rglob(patterns.get(memory_type, "*.md")))
        total_chunks = 0
        files_indexed = 0
        files_failed = 0
        files_unchanged = 0
        files_skipped = 0
        files_reconciled = 0
        transient_failures = 0
        failed_sources: list[str] = []
        try:
            from core import startup_progress

            track_startup = startup_progress.is_active()
        except Exception:
            startup_progress = None  # type: ignore[assignment]
            track_startup = False

        if track_startup and startup_progress is not None:
            startup_progress.set_phase(
                "indexing",
                detail=str(directory),
                done_count=0,
                total_count=len(md_files),
            )

        previous_active = getattr(self, "_index_directory_active", False)
        previous_failures = getattr(self, "_run_upsert_failures", {})
        self._index_directory_active = True
        self._run_upsert_failures = {}
        try:
            for index, md_file in enumerate(md_files):
                if startup_progress is not None:
                    startup_progress.raise_if_cancelled()
                    if track_startup:
                        startup_progress.update_progress(
                            detail=str(md_file),
                            done_count=index,
                            total_count=len(md_files),
                        )
                try:
                    total_chunks += self.index_file(md_file, memory_type, force=force)
                except Exception:
                    files_failed += 1
                    failed_sources.append(self._directory_source(md_file, directory))
                    raise
                outcome = getattr(self, "_last_index_file_outcome", _IndexFileOutcome(status="failed"))
                if outcome.status == "indexed":
                    files_indexed += 1
                elif outcome.status == "failed":
                    files_failed += 1
                    transient_failures += int(outcome.transient)
                    failed_sources.append(self._directory_source(md_file, directory))
                elif outcome.status == "unchanged":
                    files_unchanged += 1
                else:
                    files_skipped += 1
                if track_startup and startup_progress is not None:
                    startup_progress.update_progress(
                        detail=str(md_file),
                        done_count=index + 1,
                        total_count=len(md_files),
                    )
            files_reconciled = self._reconcile_stale_entries(directory, memory_type)
        finally:
            self._index_directory_active = previous_active
            self._run_upsert_failures = previous_failures
            result = IndexDirectoryResult(
                chunks_indexed=total_chunks,
                files_indexed=files_indexed,
                files_failed=files_failed,
                files_unchanged=files_unchanged,
                files_skipped=files_skipped,
                files_reconciled=files_reconciled,
                transient_failures=transient_failures,
                failed_sources=tuple(failed_sources),
            )
            self._log_index_directory_summary(collection_name, result)
        return result

    def _reconcile_stale_entries(self, directory: Path, memory_type: str) -> int:
        """Remove stale per-anima sources lexically below *directory*.

        Shared indexers are intentionally excluded: their single shared meta
        file cannot represent stale-chunk cleanup across multiple per-anima
        vector stores safely.
        """
        if self.collection_prefix == "shared":
            return 0
        anima_dir = getattr(self, "anima_dir", None)
        if anima_dir is None:
            return 0
        try:
            directory_key = directory.relative_to(anima_dir)
        except ValueError:
            logger.warning("Skipping reconciliation outside anima directory: %s", directory)
            return 0

        stale_sources: list[str] = []
        index_meta = getattr(self, "index_meta", {})
        for source_file in list(index_meta):
            source_key = Path(source_file)
            if source_key.is_absolute() or ".." in source_key.parts:
                continue
            if not source_key.is_relative_to(directory_key):
                continue
            source_path = anima_dir / source_key
            if not os.path.lexists(source_path) or self.is_ragignored(source_path):
                stale_sources.append(source_file)
                if len(stale_sources) >= STALE_RECONCILIATION_LIMIT:
                    break

        collection_name = f"{self.collection_prefix}_{memory_type}"
        reconciled_sources: list[str] = []
        for source_file in stale_sources:
            try:
                deleted_count = self._delete_indexed_file_documents(collection_name, source_file)
            except Exception:
                logger.warning("Failed to reconcile indexed source %s", source_file, exc_info=True)
                continue
            if deleted_count is not None:
                reconciled_sources.append(source_file)

        for source_file in reconciled_sources:
            index_meta.pop(source_file, None)
        if reconciled_sources:
            self._save_index_meta()
        return len(reconciled_sources)

    @staticmethod
    def _directory_source(file_path: Path, directory: Path) -> str:
        try:
            return str(file_path.relative_to(directory))
        except ValueError:
            return str(file_path)

    @staticmethod
    def _log_index_directory_summary(collection_name: str, result: IndexDirectoryResult) -> None:
        examples = list(result.failed_sources[:3])
        logger.info(
            "Index directory summary: collection=%s successful=%d failed=%d "
            "failed_sources=%s transient=%s unchanged=%d skipped=%d reconciled=%d chunks=%d",
            collection_name,
            result.files_indexed,
            result.files_failed,
            examples,
            result.transient,
            result.files_unchanged,
            result.files_skipped,
            result.files_reconciled,
            result.chunks_indexed,
        )

    def index_conversation_summary(
        self,
        conversation_path: Path,
        anima_name: str,
        force: bool = False,
    ) -> int:
        """Index compressed_summary from conversation.json into RAG.

        Reads the ``compressed_summary`` field from ``conversation.json``,
        chunks it by ``### `` headings (the format used by conversation
        compression), and indexes each chunk into the
        ``{anima_name}_conversation_summary`` collection with
        ``source: "conversation_gist"`` metadata.

        Args:
            conversation_path: Path to the ``state/`` directory containing
                ``conversation.json``
            anima_name: Anima name (for collection naming)
            force: Force re-indexing even if content hasn't changed

        Returns:
            Number of chunks indexed
        """
        conv_file = conversation_path / "conversation.json"
        if not conv_file.exists():
            logger.debug("conversation.json not found at %s", conv_file)
            return 0

        try:
            with open(conv_file, encoding="utf-8") as f:
                conv_data = json.load(f)
        except Exception as e:
            logger.warning("Failed to read conversation.json: %s", e)
            return 0

        summary = conv_data.get("compressed_summary", "")
        if not summary or len(summary) < 50:
            logger.debug("compressed_summary too short or empty, skipping")
            return 0

        # Check if content has changed via hash
        content_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()
        meta_key = "conversation_summary"
        collection_name = f"{self.collection_prefix}_conversation_summary"
        if not force and meta_key in self.index_meta:
            if self.index_meta[meta_key].get("hash") == content_hash:
                # Verify the collection still exists; force re-index if
                # the vectordb was wiped/recreated.
                if self._collection_exists(collection_name):
                    logger.debug("compressed_summary unchanged, skipping")
                    return 0
                logger.info(
                    "Collection '%s' missing despite tracked hash, forcing re-index of conversation_summary",
                    collection_name,
                )

        logger.info("Indexing compressed_summary for %s", anima_name)

        # Chunk by ### headings
        source_id = f"{self.collection_prefix}/conversation_summary"
        chunks = self._chunk_markdown_text(summary, source_id)

        if not chunks:
            logger.debug("No chunks extracted from compressed_summary")
            return 0

        # Generate embeddings
        embeddings = self._generate_embeddings([c.content for c in chunks])

        # Build documents and upsert
        from core.memory.rag.store import Document

        self.vector_store.create_collection(collection_name)

        documents = [
            Document(
                id=chunk.id,
                content=chunk.content,
                embedding=embeddings[i],
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]
        if not self.vector_store.upsert(collection_name, documents):
            logger.warning("Upsert failed for conversation_summary, skipping index_meta update")
            return 0

        self._mark_collection_known(collection_name)

        self.index_meta[meta_key] = {
            "hash": content_hash,
            "indexed_at": now_iso(),
            "chunks": len(chunks),
        }
        self._save_index_meta()

        logger.info("Indexed %d conversation_summary chunks for %s", len(chunks), anima_name)
        return len(chunks)

    def _chunk_markdown_text(
        self,
        text: str,
        source_id: str,
    ) -> list[MemoryChunk]:
        """Chunk a markdown text string by ``### `` headings.

        Used for compressed_summary which uses ``### heading`` sections.
        Returns MemoryChunk list compatible with ``_generate_embeddings()``
        and vector store upsert.

        Args:
            text: Markdown text to chunk
            source_id: Base ID for chunk naming (e.g. ``anima/conversation_summary``)

        Returns:
            List of MemoryChunk instances
        """
        chunks: list[MemoryChunk] = []
        sections = re.split(r"\n(###\s+.+)", text)

        chunk_idx = 0

        # Preamble (content before first ### heading)
        preamble = sections[0].strip()
        if preamble and len(preamble) > 50:
            chunk_id = f"{source_id}#{chunk_idx}"
            metadata: dict[str, str | int | float | list[str]] = {
                "anima": self.collection_prefix,
                "memory_type": "conversation_summary",
                "source": "conversation_gist",
                "source_file": "state/conversation.json",
                "chunk_index": chunk_idx,
                "importance": "normal",
                **access_tracking_metadata(),
                "activation_level": "normal",
                "low_activation_since": "",
                "valid_until": "",
            }
            chunks.append(MemoryChunk(id=chunk_id, content=preamble, metadata=metadata))
            chunk_idx += 1

        # Heading sections
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i].strip()
                body = sections[i + 1].strip()
                section_content = f"{heading}\n\n{body}"

                if section_content.strip():
                    chunk_id = f"{source_id}#{chunk_idx}"
                    metadata = {
                        "anima": self.collection_prefix,
                        "memory_type": "conversation_summary",
                        "source": "conversation_gist",
                        "source_file": "state/conversation.json",
                        "chunk_index": chunk_idx,
                        "importance": "normal",
                        **access_tracking_metadata(),
                        "activation_level": "normal",
                        "low_activation_since": "",
                        "valid_until": "",
                    }
                    chunks.append(MemoryChunk(id=chunk_id, content=section_content, metadata=metadata))
                    chunk_idx += 1

        # Fallback: if no ### headings found, treat entire text as one chunk
        if not chunks and text.strip():
            chunk_id = f"{source_id}#0"
            metadata = {
                "anima": self.collection_prefix,
                "memory_type": "conversation_summary",
                "source": "conversation_gist",
                "source_file": "state/conversation.json",
                "chunk_index": 0,
                "importance": "normal",
                **access_tracking_metadata(),
                "activation_level": "normal",
                "low_activation_since": "",
                "valid_until": "",
            }
            chunks.append(MemoryChunk(id=chunk_id, content=text.strip(), metadata=metadata))

        return chunks

    # ── Chunking strategies ─────────────────────────────────────────

    def _chunk_file(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        *,
        origin: str = "",
    ) -> list[MemoryChunk]:
        """Chunk file based on memory type.

        Markdown memory is split by headings/time headings; facts use JSONL
        records; procedures, skills, and shared users are indexed whole.
        """
        if memory_type == "facts":
            return chunk_facts_jsonl(
                file_path,
                content,
                anima_dir=self.anima_dir,
                collection_prefix=self.collection_prefix,
                make_chunk_id=self._make_chunk_id,
                chunk_factory=MemoryChunk,
                origin=origin,
            )
        if memory_type in ("knowledge", "common_knowledge"):
            return self._chunk_by_markdown_headings(file_path, content, memory_type, origin=origin)
        if memory_type == "episodes":
            time_chunks = self._chunk_by_time_headings(file_path, content, memory_type, origin=origin)
            if time_chunks:
                return time_chunks
            return self._chunk_by_markdown_headings(file_path, content, memory_type, origin=origin)
        # procedures, skills, shared_users
        return self._chunk_whole_file(file_path, content, memory_type, origin=origin)

    def _chunk_by_markdown_headings(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        *,
        origin: str = "",
    ) -> list[MemoryChunk]:
        """Split by Markdown ## headings.

        Chunks are numbered sequentially starting from 0.  The preamble
        (content before the first ``##`` heading) is emitted first when
        it exceeds 50 characters, followed by each heading section.

        YAML frontmatter (``---`` delimited) is stripped before chunking
        to avoid polluting vector embeddings with metadata.  The parsed
        frontmatter is passed to ``_extract_metadata`` so that fields
        like ``valid_until`` are included in chunk metadata.
        """
        frontmatter = self._parse_frontmatter(content)
        content = self._strip_frontmatter(content)
        chunks: list[MemoryChunk] = []
        sections = re.split(r"\n(##\s+.+)", f"\n{content}")

        preamble = sections[0].strip()
        chunk_idx = 0

        # 1. Preamble (content before first ## heading)
        if preamble and len(preamble) > 50:
            chunk_id = self._make_chunk_id(file_path, memory_type, chunk_idx)
            metadata = self._extract_metadata(
                file_path,
                preamble,
                memory_type,
                chunk_idx,
                1,
                frontmatter=frontmatter,
                origin=origin,
            )
            chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    content=preamble,
                    metadata=metadata,
                )
            )
            chunk_idx += 1

        # 2. Heading sections (sequential)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i].strip()
                body = sections[i + 1].strip()
                section_content = f"{heading}\n\n{body}"

                if section_content.strip():
                    chunk_id = self._make_chunk_id(file_path, memory_type, chunk_idx)
                    metadata = self._extract_metadata(
                        file_path,
                        section_content,
                        memory_type,
                        chunk_idx,
                        1,
                        frontmatter=frontmatter,
                        origin=origin,
                    )
                    if memory_type == "episodes":
                        apply_episode_heading_event_time(metadata, heading)
                    chunks.append(
                        MemoryChunk(
                            id=chunk_id,
                            content=section_content,
                            metadata=metadata,
                        )
                    )
                    chunk_idx += 1

        return chunks

    def _chunk_by_time_headings(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        *,
        origin: str = "",
    ) -> list[MemoryChunk]:
        """Split by time headings (## HH:MM format) with ``valid_at`` metadata."""
        frontmatter = self._parse_frontmatter(content)
        content = self._strip_frontmatter(content)
        chunks: list[MemoryChunk] = []

        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", file_path.stem)
        base_date_str = date_match.group(1) if date_match else None

        # Match headings like ## 09:30, ## 14:15 optional — title
        sections = re.split(r"\n(##\s+\d{1,2}:\d{2}.*)", f"\n{content}")

        if len(sections) <= 1:
            return []

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i].strip()
                body = sections[i + 1].strip()
                section_content = f"{heading}\n\n{body}"

                if section_content.strip():
                    chunk_id = self._make_chunk_id(file_path, memory_type, i // 2)
                    metadata = self._extract_metadata(
                        file_path,
                        section_content,
                        memory_type,
                        i // 2,
                        (i // 2) + 1,
                        frontmatter=frontmatter,
                        origin=origin,
                    )

                    if base_date_str:
                        time_match = re.match(r"##\s+(\d{1,2}):(\d{2})", heading)
                        if time_match:
                            try:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2))
                                dt = ensure_aware(
                                    datetime.fromisoformat(f"{base_date_str}T{hour:02d}:{minute:02d}:00"),
                                )
                                metadata["valid_at"] = dt.timestamp()
                            except (ValueError, TypeError):
                                pass

                    chunks.append(
                        MemoryChunk(
                            id=chunk_id,
                            content=section_content,
                            metadata=metadata,
                        )
                    )

        return chunks

    def _chunk_whole_file(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        *,
        origin: str = "",
    ) -> list[MemoryChunk]:
        """Return entire file as single chunk.

        YAML frontmatter (``---`` delimited) is stripped before chunking
        to avoid polluting vector embeddings with metadata.  The parsed
        frontmatter is passed to ``_extract_metadata`` so that fields
        like ``valid_until`` are included in chunk metadata.
        """
        frontmatter = self._parse_frontmatter(content)
        content = self._strip_frontmatter(content)
        if not content.strip():
            return []

        chunk_id = self._make_chunk_id(file_path, memory_type, 0)
        metadata = self._extract_metadata(
            file_path,
            content,
            memory_type,
            0,
            1,
            frontmatter=frontmatter,
            origin=origin,
        )

        return [
            MemoryChunk(
                id=chunk_id,
                content=content,
                metadata=metadata,
            )
        ]

    # ── Frontmatter handling ─────────────────────────────────────────

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Strip YAML frontmatter from content.

        Delegates to the canonical line-based parser to avoid false
        splits when YAML values contain ``---``.

        Args:
            content: File content potentially starting with ``---`` YAML block

        Returns:
            Content without frontmatter, or original content if none found
        """
        from core.memory.frontmatter import strip_frontmatter

        return strip_frontmatter(content).strip()

    # ── Helpers ─────────────────────────────────────────────────────

    def _make_chunk_id(self, file_path: Path, memory_type: str, index: int) -> str:
        """Generate unique chunk ID.

        Uses ``{collection_prefix}/{rel_path}#{index}`` format.
        ``rel_path`` already contains the directory hierarchy
        (e.g. ``knowledge/file.md``), so ``memory_type`` is intentionally
        **not** embedded in the ID to avoid path duplication.
        """
        rel_path = file_path.relative_to(self.anima_dir)
        return f"{self.collection_prefix}/{rel_path}#{index}"

    @staticmethod
    def _parse_frontmatter(raw_content: str) -> dict:
        """Parse YAML frontmatter from raw file content.

        Delegates to the canonical line-based parser to avoid false
        splits when YAML values contain ``---``.

        Args:
            raw_content: Full file content potentially starting with ``---``

        Returns:
            Parsed frontmatter dict, or empty dict if absent/unparseable
        """
        from core.memory.frontmatter import parse_frontmatter

        meta, _ = parse_frontmatter(raw_content)
        return meta

    def _extract_metadata(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        chunk_index: int,
        total_chunks: int,
        frontmatter: dict | None = None,
        origin: str = "",
    ) -> dict[str, str | int | float | bool | list[str]]:
        """Extract metadata from file and content.

        Args:
            file_path: Path to the source file
            content: Chunk content text
            memory_type: Memory type identifier
            chunk_index: Index of this chunk within the file
            total_chunks: Total number of chunks from this file
            frontmatter: Pre-parsed YAML frontmatter from the file.
                If provided, ``valid_until`` is extracted from it.
            origin: Provenance origin category for trust resolution.
        """
        metadata: dict[str, str | int | float | bool | list[str]] = {
            "anima": self.collection_prefix,
            "memory_type": memory_type,
            "source_file": str(file_path.relative_to(self.anima_dir)),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }

        # File timestamps
        stat = file_path.stat()
        metadata["created_at"] = ensure_aware(datetime.fromtimestamp(stat.st_ctime)).isoformat()
        metadata["updated_at"] = ensure_aware(datetime.fromtimestamp(stat.st_mtime)).isoformat()

        # Importance detection
        if "[IMPORTANT]" in content or "[重要]" in content:
            metadata["importance"] = "important"
        else:
            metadata["importance"] = "normal"

        # ── ActionRule ──────────
        if "[ACTION-RULE]" in content:
            lines = content.splitlines()
            heading_idx = next((i for i, ln in enumerate(lines) if "[ACTION-RULE]" in ln), None)
            if heading_idx is not None:
                meta_lines: list[str] = []
                sep_idx: int | None = None
                for j in range(heading_idx + 1, len(lines)):
                    if lines[j].strip() == "---":
                        sep_idx = j
                        break
                if sep_idx is not None:
                    meta_lines = lines[heading_idx + 1 : sep_idx]
                else:
                    for j in range(heading_idx + 1, len(lines)):
                        ln = lines[j]
                        if re.match(r"^\s*(trigger_tools|keywords)\s*:", ln, re.IGNORECASE):
                            meta_lines.append(ln)
                        elif not ln.strip():
                            continue
                        else:
                            break
                trigger_tools_re = re.compile(r"^\s*trigger_tools\s*:\s*(.*)$", re.IGNORECASE)
                keywords_re = re.compile(r"^\s*keywords\s*:\s*(.*)$", re.IGNORECASE)
                trigger_val: str | None = None
                keywords_val: str | None = None
                for mln in meta_lines:
                    m = trigger_tools_re.match(mln)
                    if m:
                        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                        trigger_val = ",".join(parts) if parts else None
                        continue
                    m = keywords_re.match(mln)
                    if m:
                        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
                        keywords_val = ",".join(parts) if parts else None
                if trigger_val:
                    metadata["type"] = "action_rule"
                    metadata["trigger_tools"] = trigger_val
                    if keywords_val:
                        metadata["action_rule_keywords"] = keywords_val
                else:
                    logger.warning(
                        "[ACTION-RULE] missing trigger_tools after heading in %s (chunk %s)",
                        file_path,
                        chunk_index,
                    )

        # Tag extraction (simple pattern: #tag or [tag])
        tags = re.findall(r"#(\w+)|「(\w+)」", content)
        flattened_tags = [t for group in tags for t in group if t]
        if flattened_tags:
            metadata["tags"] = flattened_tags[:10]  # Limit to 10 tags

        # Access tracking (Hebbian LTP analog)
        metadata.update(access_tracking_metadata())

        # Activation level (for forgetting mechanism)
        metadata["activation_level"] = "normal"
        metadata["low_activation_since"] = ""

        # Supersession tracking: valid_until from frontmatter
        # Legacy migration: rename superseded_at → valid_until
        fm = frontmatter or {}
        if "superseded_at" in fm and "valid_until" not in fm:
            fm["valid_until"] = fm.pop("superseded_at")
        metadata["valid_until"] = str(fm.get("valid_until", "") or "")

        # valid_at: event timestamp (preferred over file timestamps for recency)
        valid_at_str = str(fm.get("valid_from", "") or "")
        if valid_at_str:
            try:
                vat = ensure_aware(datetime.fromisoformat(valid_at_str))
                metadata["valid_at"] = vat.timestamp()
            except (ValueError, TypeError):
                pass
        if "valid_at" not in metadata:
            metadata["valid_at"] = float(stat.st_mtime)

        if fm.get("summary"):
            metadata["summary"] = str(fm["summary"])[:200]

        # Failure tracking fields from frontmatter (knowledge + procedures)
        if fm:
            for field in ("success_count", "failure_count", "version"):
                if field in fm:
                    try:
                        metadata[field] = int(fm[field])
                    except (ValueError, TypeError):
                        # Skip non-integer values (e.g. "archived_2026-03-28")
                        pass
            if "confidence" in fm:
                try:
                    metadata["confidence"] = float(fm["confidence"])
                except (ValueError, TypeError):
                    pass
            if "last_used" in fm and fm["last_used"]:
                metadata["last_used"] = str(fm["last_used"])

        if origin:
            metadata["origin"] = origin

        return metadata

    def _generate_embeddings(
        self,
        texts: list[str],
        *,
        purpose: Literal["document", "query"] = "document",
        priority: Literal["interactive", "bulk"] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Delegates to ``generate_embeddings()`` which routes to the HTTP
        server (child processes) or local model (server/test) automatically.
        """
        if not texts:
            return []

        logger.debug("Generating embeddings for %d texts", len(texts))
        from core.memory.rag.singleton import generate_embeddings

        resolved_priority = priority or ("interactive" if purpose == "query" else "bulk")
        return generate_embeddings(texts, purpose=purpose, priority=resolved_priority)

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
