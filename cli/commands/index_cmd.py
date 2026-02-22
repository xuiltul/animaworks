from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Index command - build or rebuild RAG vector indexes."""

import argparse
import logging
from pathlib import Path

from core.paths import get_data_dir

logger = logging.getLogger("animaworks.cli.index")


def setup_index_command(subparsers: argparse._SubParsersAction) -> None:
    """Setup the 'index' subcommand."""
    parser = subparsers.add_parser(
        "index",
        help="Build or rebuild RAG vector indexes for memory search",
        description="Index memory files into vector database for hybrid search.",
    )
    parser.add_argument(
        "--anima",
        type=str,
        help="Index only this anima's memories (default: all animas)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full re-indexing (delete existing index and rebuild)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    parser.set_defaults(func=index_command)


def _check_model_change(base_dir: Path, full: bool) -> str:
    """Check if the configured embedding model differs from the last indexed model.

    Args:
        base_dir: AnimaWorks data directory.
        full: Whether ``--full`` rebuild was requested.

    Returns:
        The current configured model name.

    Raises:
        SystemExit: If the model changed but ``--full`` was not specified.
    """
    import json
    import sys

    from core.memory.rag.singleton import get_embedding_model_name

    current_model = get_embedding_model_name()
    meta_path = base_dir / "index_meta.json"

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            previous_model = meta.get("embedding_model")
        except (json.JSONDecodeError, OSError):
            previous_model = None

        if previous_model and previous_model != current_model and not full:
            logger.error(
                "Embedding model changed: %s → %s.  "
                "Run 'animaworks index --full' to rebuild the index.",
                previous_model,
                current_model,
            )
            sys.exit(1)

    return current_model


def _save_global_index_meta(base_dir: Path, model_name: str) -> None:
    """Write the embedding model name to the global index_meta.json."""
    import json

    meta_path = base_dir / "index_meta.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    meta["embedding_model"] = model_name
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def index_command(args: argparse.Namespace) -> None:
    """Execute the index command."""
    try:
        from core.memory.rag import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore
    except ImportError:
        logger.error(
            "RAG dependencies not installed. Run: pip install 'animaworks[rag]'"
        )
        return

    base_dir = get_data_dir()
    animas_dir = base_dir / "animas"

    if not animas_dir.is_dir():
        logger.error("Animas directory not found: %s", animas_dir)
        logger.info("Run 'animaworks init' first to set up the environment")
        return

    # Check for embedding model change before proceeding
    current_model = _check_model_change(base_dir, args.full)
    logger.info("Embedding model: %s", current_model)

    # Determine which animas to index
    if args.anima:
        anima_dirs = [animas_dir / args.anima]
        if not anima_dirs[0].is_dir():
            logger.error("Anima not found: %s", args.anima)
            return
    else:
        anima_dirs = [p for p in sorted(animas_dir.iterdir()) if p.is_dir()]

    if not anima_dirs:
        logger.warning("No animas found to index")
        return

    from core.paths import get_anima_vectordb_dir

    # Index each anima
    total_chunks = 0
    for anima_dir in anima_dirs:
        anima_name = anima_dir.name
        logger.info("=" * 60)
        logger.info("Indexing anima: %s", anima_name)
        logger.info("=" * 60)

        # Per-anima vector store
        vector_store = ChromaVectorStore(persist_dir=get_anima_vectordb_dir(anima_name))

        # Full rebuild: delete existing collections for this anima
        if args.full and not args.dry_run:
            logger.info("Full rebuild: deleting collections for %s", anima_name)
            for collection in vector_store.list_collections():
                vector_store.delete_collection(collection)

        # Initialize indexer
        indexer = MemoryIndexer(vector_store, anima_name, anima_dir)

        # Index each memory type
        memory_types = [
            ("knowledge", anima_dir / "knowledge"),
            ("episodes", anima_dir / "episodes"),
            ("procedures", anima_dir / "procedures"),
            ("skills", anima_dir / "skills"),
        ]

        for memory_type, memory_dir in memory_types:
            if not memory_dir.is_dir():
                logger.debug("Skipping %s (directory not found)", memory_type)
                continue

            logger.info("Indexing %s...", memory_type)

            if args.dry_run:
                # Just count files
                md_files = list(memory_dir.glob("*.md"))
                logger.info("  Would index %d files in %s/", len(md_files), memory_type)
                continue

            chunks = indexer.index_directory(
                memory_dir,
                memory_type,
                force=args.full,
            )
            total_chunks += chunks
            logger.info("  Indexed %d chunks from %s/", chunks, memory_type)

    # Index shared user memories
    shared_users_dir = base_dir / "shared" / "users"
    if shared_users_dir.is_dir() and not args.anima:
        logger.info("=" * 60)
        logger.info("Indexing shared user memories")
        logger.info("=" * 60)

        # Use shared vector store for user memories
        shared_store = ChromaVectorStore()  # defaults to ~/.animaworks/vectordb
        indexer = MemoryIndexer(
            shared_store, "shared", shared_users_dir.parent
        )

        if args.dry_run:
            user_dirs = [d for d in shared_users_dir.iterdir() if d.is_dir()]
            logger.info("  Would index %d user profiles", len(user_dirs))
        else:
            chunks = indexer.index_directory(
                shared_users_dir,
                "shared_users",
                force=args.full,
            )
            total_chunks += chunks
            logger.info("  Indexed %d user profile chunks", chunks)

    # Summary
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("Dry run complete (no actual indexing performed)")
    else:
        logger.info("Indexing complete: %d total chunks indexed", total_chunks)
        # Record the embedding model used for future change detection
        _save_global_index_meta(base_dir, current_model)
