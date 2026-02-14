from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

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

logger = logging.getLogger("animaworks.rag.indexer")

# ── Configuration ───────────────────────────────────────────────────

# Embedding model (sentence-transformers)
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384

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
        person_name: str,
        person_dir: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize indexer.

        Args:
            vector_store: VectorStore instance (e.g., ChromaVectorStore)
            person_name: Person name (for collection naming)
            person_dir: Path to person's memory directory
            embedding_model: Sentence-transformers model name
        """
        self.vector_store = vector_store
        self.person_name = person_name
        self.person_dir = person_dir
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        self._init_embedding_model()

        # Load index metadata
        self.meta_path = person_dir / INDEX_META_FILE
        self.index_meta = self._load_index_meta()

    def _init_embedding_model(self) -> None:
        """Initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e

        logger.info("Loading embedding model: %s", self.embedding_model_name)

        # Cache model in ~/.animaworks/models/
        from core.paths import get_data_dir

        cache_dir = get_data_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            cache_folder=str(cache_dir),
        )
        logger.info("Embedding model loaded (dimension=%d)", EMBEDDING_DIMENSION)

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
        file_key = str(file_path.relative_to(self.person_dir))

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
        collection_name = f"{self.person_name}_{memory_type}"
        self.vector_store.create_collection(collection_name, EMBEDDING_DIMENSION)
        self.vector_store.upsert(collection_name, documents)

        # Update index metadata
        self.index_meta[file_key] = {
            "hash": file_hash,
            "indexed_at": datetime.now().isoformat(),
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
        - knowledge: Markdown heading sections
        - episodes: Time-based sections (## HH:MM)
        - procedures: Whole file (don't split procedures)
        - skills: Whole file
        - shared_users: Whole file
        """
        if memory_type == "knowledge":
            return self._chunk_by_markdown_headings(file_path, content)
        elif memory_type == "episodes":
            return self._chunk_by_time_headings(file_path, content)
        else:  # procedures, skills, shared_users
            return self._chunk_whole_file(file_path, content, memory_type)

    def _chunk_by_markdown_headings(
        self,
        file_path: Path,
        content: str,
    ) -> list[MemoryChunk]:
        """Split by Markdown ## headings."""
        chunks: list[MemoryChunk] = []
        sections = re.split(r"\n(##\s+.+)", content)

        current_section = sections[0].strip()  # Before first heading

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i].strip()
                body = sections[i + 1].strip()
                section_content = f"{heading}\n\n{body}"

                if section_content.strip():
                    chunk_id = self._make_chunk_id(file_path, "knowledge", i // 2)
                    metadata = self._extract_metadata(
                        file_path, section_content, "knowledge", i // 2, (i // 2) + 1
                    )
                    chunks.append(
                        MemoryChunk(
                            id=chunk_id,
                            content=section_content,
                            metadata=metadata,
                        )
                    )

        # Add content before first heading as chunk 0 if substantial
        if current_section and len(current_section) > 50:
            chunk_id = self._make_chunk_id(file_path, "knowledge", 0)
            metadata = self._extract_metadata(
                file_path, current_section, "knowledge", 0, len(chunks) + 1
            )
            chunks.insert(
                0,
                MemoryChunk(
                    id=chunk_id,
                    content=current_section,
                    metadata=metadata,
                ),
            )

        return chunks

    def _chunk_by_time_headings(
        self,
        file_path: Path,
        content: str,
    ) -> list[MemoryChunk]:
        """Split by time headings (## HH:MM format)."""
        chunks: list[MemoryChunk] = []
        # Match headings like ## 09:30, ## 14:15 — タイトル
        sections = re.split(r"\n(##\s+\d{1,2}:\d{2}.*)", content)

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i].strip()
                body = sections[i + 1].strip()
                section_content = f"{heading}\n\n{body}"

                if section_content.strip():
                    chunk_id = self._make_chunk_id(file_path, "episodes", i // 2)
                    metadata = self._extract_metadata(
                        file_path, section_content, "episodes", i // 2, (i // 2) + 1
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
        """Return entire file as single chunk."""
        if not content.strip():
            return []

        chunk_id = self._make_chunk_id(file_path, memory_type, 0)
        metadata = self._extract_metadata(file_path, content, memory_type, 0, 1)

        return [
            MemoryChunk(
                id=chunk_id,
                content=content,
                metadata=metadata,
            )
        ]

    # ── Helpers ─────────────────────────────────────────────────────

    def _make_chunk_id(self, file_path: Path, memory_type: str, index: int) -> str:
        """Generate unique chunk ID."""
        rel_path = file_path.relative_to(self.person_dir)
        return f"{self.person_name}/{memory_type}/{rel_path}#{index}"

    def _extract_metadata(
        self,
        file_path: Path,
        content: str,
        memory_type: str,
        chunk_index: int,
        total_chunks: int,
    ) -> dict[str, str | int | float | list[str]]:
        """Extract metadata from file and content."""
        metadata: dict[str, str | int | float | list[str]] = {
            "person": self.person_name,
            "memory_type": memory_type,
            "source_file": str(file_path.relative_to(self.person_dir)),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }

        # File timestamps
        stat = file_path.stat()
        metadata["created_at"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata["updated_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

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
