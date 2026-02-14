from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

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
        "--person",
        type=str,
        help="Index only this person's memories (default: all persons)",
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
    persons_dir = base_dir / "persons"

    if not persons_dir.is_dir():
        logger.error("Persons directory not found: %s", persons_dir)
        logger.info("Run 'animaworks init' first to set up the environment")
        return

    # Determine which persons to index
    if args.person:
        person_dirs = [persons_dir / args.person]
        if not person_dirs[0].is_dir():
            logger.error("Person not found: %s", args.person)
            return
    else:
        person_dirs = [p for p in sorted(persons_dir.iterdir()) if p.is_dir()]

    if not person_dirs:
        logger.warning("No persons found to index")
        return

    # Initialize vector store
    vector_store = ChromaVectorStore()

    # Full rebuild: delete existing collections
    if args.full and not args.dry_run:
        logger.info("Full rebuild requested, deleting existing collections")
        for collection in vector_store.list_collections():
            vector_store.delete_collection(collection)

    # Index each person
    total_chunks = 0
    for person_dir in person_dirs:
        person_name = person_dir.name
        logger.info("=" * 60)
        logger.info("Indexing person: %s", person_name)
        logger.info("=" * 60)

        # Initialize indexer
        indexer = MemoryIndexer(vector_store, person_name, person_dir)

        # Index each memory type
        memory_types = [
            ("knowledge", person_dir / "knowledge"),
            ("episodes", person_dir / "episodes"),
            ("procedures", person_dir / "procedures"),
            ("skills", person_dir / "skills"),
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
    if shared_users_dir.is_dir() and not args.person:
        logger.info("=" * 60)
        logger.info("Indexing shared user memories")
        logger.info("=" * 60)

        # Use a dummy person name for shared memories
        # (each person will access the same shared_users collection)
        indexer = MemoryIndexer(
            vector_store, "shared", shared_users_dir.parent
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
        logger.info("Vector database: %s", vector_store.persist_dir)
