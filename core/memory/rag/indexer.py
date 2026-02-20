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

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.time_utils import ensure_aware, now_iso

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
        embedding_model: object | None = None,
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

    def _load_index_meta(self) -> dict[str, dict[str, str]]:
        """Load index metadata (file hashes and timestamps)."""
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load index metadata: %s", e)
            return {}

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
    ) -> int:
        """Index a single memory file.

        Args:
            file_path: Path to the memory file
            memory_type: Memory type (knowledge, episodes, procedures, skills, shared_users)
            force: Force re-indexing even if file hasn't changed

        Returns:
            Number of chunks indexed
        """
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
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
        chunks = self._chunk_file(file_path, content, memory_type)

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

        total_chunks = 0
        for md_file in sorted(directory.glob("*.md")):
            total_chunks += self.index_file(md_file, memory_type, force=force)

        logger.info(
            "Indexed directory %s: %d total chunks (type=%s)",
            directory,
            total_chunks,
            memory_type,
        )
        return total_chunks

    # ── Chunking strategies ─────────────────────────────────────────

    def _chunk_file(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
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
            return self._chunk_by_markdown_headings(file_path, content, memory_type)
        elif memory_type == "episodes":
            return self._chunk_by_time_headings(file_path, content, memory_type)
        else:  # procedures, skills, shared_users
            return self._chunk_whole_file(file_path, content, memory_type)

    def _chunk_by_markdown_headings(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
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
                file_path, preamble, memory_type, chunk_idx, 1,
                frontmatter=frontmatter,
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
                        file_path, section_content, memory_type, chunk_idx, 1,
                        frontmatter=frontmatter,
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
                        file_path, section_content, memory_type, i // 2, (i // 2) + 1,
                        frontmatter=frontmatter,
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
            file_path, content, memory_type, 0, 1,
            frontmatter=frontmatter,
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

        Args:
            content: File content potentially starting with ``---`` YAML block

        Returns:
            Content without frontmatter, or original content if none found
        """
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content

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

        Args:
            raw_content: Full file content potentially starting with ``---``

        Returns:
            Parsed frontmatter dict, or empty dict if absent/unparseable
        """
        if raw_content.startswith("---"):
            parts = raw_content.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    fm = yaml.safe_load(parts[1])
                    return fm if isinstance(fm, dict) else {}
                except Exception:
                    return {}
        return {}

    def _extract_metadata(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        chunk_index: int,
        total_chunks: int,
        frontmatter: dict | None = None,
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
