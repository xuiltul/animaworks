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
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from core.time_utils import ensure_aware, now_iso

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("animaworks.rag.indexer")

# ── Configuration ───────────────────────────────────────────────────

# Index metadata file
INDEX_META_FILE = "index_meta.json"


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class MemoryChunk:
    """A chunk of memory content ready for indexing."""

    id: str
    content: str
    metadata: dict[str, str | int | float | list[str]]


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

        # Use injected embedding model or initialize via singleton
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self._init_embedding_model()

        # Load index metadata
        self.meta_path = anima_dir / INDEX_META_FILE
        self.index_meta = self._load_index_meta()

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

    # ── Main indexing API ───────────────────────────────────────────

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
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
            return 0

        # Check .ragignore exclusion
        if self.is_ragignored(file_path):
            logger.debug("Skipping ragignored file: %s", file_path)
            return 0

        # Check if file has changed
        file_hash = self._compute_file_hash(file_path)
        file_key = str(file_path.relative_to(self.anima_dir))

        if not force and file_key in self.index_meta:
            if self.index_meta[file_key].get("hash") == file_hash:
                logger.debug("File unchanged, skipping: %s", file_path)
                return 0

        logger.info("Indexing file: %s (type=%s)", file_path, memory_type)

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return 0

        # Chunk the content
        chunks = self._chunk_file(file_path, content, memory_type, origin=origin)

        if not chunks:
            logger.debug("No chunks extracted from %s", file_path)
            return 0

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

        # Upsert to vector store
        from core.memory.rag.singleton import get_embedding_dimension

        collection_name = f"{self.collection_prefix}_{memory_type}"
        self.vector_store.create_collection(collection_name, get_embedding_dimension())
        self.vector_store.upsert(collection_name, documents)

        # Update index metadata
        self.index_meta[file_key] = {
            "hash": file_hash,
            "indexed_at": now_iso(),
            "chunks": len(chunks),
        }
        self._save_index_meta()

        logger.info("Indexed %d chunks from %s", len(chunks), file_path)
        return len(chunks)

    def index_directory(
        self,
        directory: Path,
        memory_type: str,
        force: bool = False,
    ) -> int:
        """Index all .md files in a directory.

        Args:
            directory: Path to memory directory
            memory_type: Memory type
            force: Force re-indexing

        Returns:
            Total number of chunks indexed
        """
        if not directory.is_dir():
            logger.warning("Directory not found: %s", directory)
            return 0

        if memory_type in ("skills", "common_skills"):
            pattern = "*/SKILL.md"
        else:
            pattern = "*.md"
        total_chunks = 0
        for md_file in sorted(directory.glob(pattern)):
            total_chunks += self.index_file(md_file, memory_type, force=force)

        logger.info(
            "Indexed directory %s: %d total chunks (type=%s)",
            directory,
            total_chunks,
            memory_type,
        )
        return total_chunks

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
        if not force and meta_key in self.index_meta:
            if self.index_meta[meta_key].get("hash") == content_hash:
                logger.debug("compressed_summary unchanged, skipping")
                return 0

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
        from core.memory.rag.singleton import get_embedding_dimension
        from core.memory.rag.store import Document

        collection_name = f"{self.collection_prefix}_conversation_summary"
        self.vector_store.create_collection(collection_name, get_embedding_dimension())

        documents = [
            Document(
                id=chunk.id,
                content=chunk.content,
                embedding=embeddings[i],
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]
        self.vector_store.upsert(collection_name, documents)

        # Update index metadata
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
                "access_count": 0,
                "last_accessed_at": "",
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
                        "access_count": 0,
                        "last_accessed_at": "",
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
                "access_count": 0,
                "last_accessed_at": "",
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

        Strategies:
        - knowledge / common_knowledge: Markdown heading sections
        - episodes: Time-based sections (## HH:MM)
        - procedures: Whole file (don't split procedures)
        - skills / common_skills: Whole file
        - shared_users: Whole file
        """
        if memory_type in ("knowledge", "common_knowledge"):
            return self._chunk_by_markdown_headings(file_path, content, memory_type, origin=origin)
        elif memory_type == "episodes":
            return self._chunk_by_time_headings(file_path, content, memory_type, origin=origin)
        else:  # procedures, skills, shared_users
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
        sections = re.split(r"\n(##\s+.+)", content)

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
        """Split by time headings (## HH:MM format)."""
        frontmatter = self._parse_frontmatter(content)
        chunks: list[MemoryChunk] = []
        # Match headings like ## 09:30, ## 14:15 — タイトル
        sections = re.split(r"\n(##\s+\d{1,2}:\d{2}.*)", content)

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
    ) -> dict[str, str | int | float | list[str]]:
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
        metadata: dict[str, str | int | float | list[str]] = {
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

        # Tag extraction (simple pattern: #tag or [tag])
        tags = re.findall(r"#(\w+)|「(\w+)」", content)
        flattened_tags = [t for group in tags for t in group if t]
        if flattened_tags:
            metadata["tags"] = flattened_tags[:10]  # Limit to 10 tags

        # Access tracking (Hebbian LTP analog)
        metadata["access_count"] = 0
        metadata["last_accessed_at"] = ""

        # Activation level (for forgetting mechanism)
        metadata["activation_level"] = "normal"
        metadata["low_activation_since"] = ""

        # Supersession tracking: valid_until from frontmatter
        # Legacy migration: rename superseded_at → valid_until
        fm = frontmatter or {}
        if "superseded_at" in fm and "valid_until" not in fm:
            fm["valid_until"] = fm.pop("superseded_at")
        metadata["valid_until"] = str(fm.get("valid_until", "") or "")

        # Failure tracking fields from frontmatter (knowledge + procedures)
        if fm:
            for field in ("success_count", "failure_count", "version"):
                if field in fm:
                    metadata[field] = int(fm[field])
            if "confidence" in fm:
                metadata["confidence"] = float(fm["confidence"])
            if "last_used" in fm and fm["last_used"]:
                metadata["last_used"] = str(fm["last_used"])

        if origin:
            metadata["origin"] = origin

        return metadata

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        logger.debug("Generating embeddings for %d texts", len(texts))
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
