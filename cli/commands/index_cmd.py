from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Index command - build or rebuild RAG vector indexes."""

import argparse
import atexit
import logging
import os
from pathlib import Path
from typing import Any

from core.paths import get_data_dir

logger = logging.getLogger("animaworks.cli.index")


def _setup_server_delegation() -> bool:
    """Detect running server and configure HTTP delegation.

    When the server is running, sets ``ANIMAWORKS_VECTOR_URL``,
    ``ANIMAWORKS_EMBED_URL``, and ``ANIMAWORKS_RERANK_URL`` so that
    ``get_vector_store()`` returns ``HttpVectorStore`` and embeddings /
    rerank are generated server-side.  This prevents unsafe concurrent
    ChromaDB access and per-process model loads.

    Returns:
        True if delegation was activated (server is running).
    """
    from cli.commands.server import _is_process_alive, _read_pid

    pid = _read_pid()
    if pid is None or not _is_process_alive(pid):
        return False

    try:
        from core.config import load_config

        port = load_config().server.port
    except Exception:
        port = 18500

    base = f"http://127.0.0.1:{port}/api"
    os.environ.setdefault("ANIMAWORKS_VECTOR_URL", f"{base}/internal/vector")
    os.environ.setdefault("ANIMAWORKS_EMBED_URL", f"{base}/internal/embed")
    os.environ.setdefault("ANIMAWORKS_RERANK_URL", f"{base}/internal/rerank")
    logger.info(
        "Server detected (pid=%d). Using HTTP delegation for safe ChromaDB access.",
        pid,
    )
    return True


def _setup_offline_vector_worker_if_needed(server_mode: bool) -> Any | None:
    """Start a temporary vector worker when no server delegation exists."""
    if server_mode or os.environ.get("ANIMAWORKS_VECTOR_URL"):
        return None
    from core.memory.rag.vector_worker_client import start_temporary_vector_worker

    worker = start_temporary_vector_worker()
    atexit.register(worker.stop)
    return worker


def _stop_offline_vector_worker(worker: Any | None) -> None:
    if worker is None:
        return
    try:
        atexit.unregister(worker.stop)
    except Exception:
        pass
    worker.stop()


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
        "--shared",
        action="store_true",
        help="Index shared collections (common_knowledge + common_skills) into each enabled anima's DB",
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

    from core.memory.rag.singleton import get_embedding_e5_prefix_enabled, get_embedding_model_name

    current_model = get_embedding_model_name()
    current_e5_prefix = get_embedding_e5_prefix_enabled()
    meta_path = base_dir / "index_meta.json"

    if meta_path.is_file():
        from core.i18n import t
        from core.memory.rag.index_signature import index_signature_error

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            signature_error = index_signature_error(meta, current_model, current_e5_prefix)
        except (json.JSONDecodeError, OSError):
            signature_error = t("rag.signature_unreadable")
        if signature_error and not full:
            logger.error(t("rag.indexing_blocked", reason=signature_error))
            sys.exit(1)

    return current_model


def _save_global_index_meta(base_dir: Path, model_name: str) -> None:
    """Write the embedding index signature to the global index_meta.json."""
    import json

    from core.memory.rag.singleton import get_embedding_e5_prefix_enabled

    meta_path = base_dir / "index_meta.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    meta["embedding_model"] = model_name
    meta["embedding_e5_prefix"] = get_embedding_e5_prefix_enabled()
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_anima_enabled(anima_dir: Path) -> bool:
    """Check whether an anima is enabled via its status.json."""
    import json

    status_file = anima_dir / "status.json"
    if not status_file.is_file():
        return True
    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
        return data.get("enabled", True)
    except (json.JSONDecodeError, OSError):
        return True


def _uses_root_vector_store(anima_dir: Path) -> bool:
    from core.config.resolver import resolve_process_model_config

    config = resolve_process_model_config(anima_dir)
    return config.valid and config.process_model == "phase3"


def _index_shared_collections(
    anima_dirs: list[Path],
    base_dir: Path,
    *,
    full: bool,
    dry_run: bool,
) -> int:
    """Index common_knowledge + common_skills into each anima's per-anima DB.

    Returns total chunks indexed across all animas.
    """
    from core.company_resources import get_company_resources_for_company
    from core.config.models import read_anima_company_checked
    from core.memory.rag import MemoryIndexer
    from core.memory.rag.repair import is_repair_locked
    from core.memory.rag.shared_meta import read_shared_hash, reset_shared_for_company_change, write_shared_hash
    from core.memory.rag.singleton import get_vector_store
    from core.memory.rag_search import _compute_dir_hash

    ck_dir = base_dir / "common_knowledge"
    cs_dir = base_dir / "common_skills"

    shared_dirs: list[tuple[str, Path, str, str]] = []
    if ck_dir.is_dir() and any(ck_dir.rglob("*.md")):
        shared_dirs.append(("common_knowledge", ck_dir, "*.md", "shared_common_knowledge_hash"))
    if cs_dir.is_dir() and any(cs_dir.rglob("SKILL.md")):
        shared_dirs.append(("common_skills", cs_dir, "SKILL.md", "shared_common_skills_hash"))

    total = 0
    for anima_dir in anima_dirs:
        anima_name = anima_dir.name
        if _uses_root_vector_store(anima_dir):
            logger.error(
                "Cannot index phase3 anima %s from the CLI; use the running server's daily root indexing",
                anima_name,
            )
            continue
        if is_repair_locked(anima_name):
            logger.warning("  %s: skipping shared indexing because RAG repair lock is held", anima_name)
            continue
        company_valid, company = read_anima_company_checked(anima_dir)
        if not company_valid:
            logger.warning("  %s: company membership unavailable, skipping shared indexing", anima_name)
            continue
        current_company = company or ""
        company_resources = get_company_resources_for_company(company, data_dir=base_dir)
        vector_store = get_vector_store(anima_name)
        if vector_store is None:
            logger.warning("Vector store unavailable for %s, skipping shared indexing", anima_name)
            continue

        anima_shared_dirs = list(shared_dirs)
        if company_resources is not None:
            if company_resources.knowledge_dir.is_dir() and any(company_resources.knowledge_dir.rglob("*.md")):
                anima_shared_dirs.append(
                    (
                        "common_knowledge",
                        company_resources.knowledge_dir,
                        "*.md",
                        "shared_company_knowledge_hash",
                    )
                )
            if company_resources.skills_dir.is_dir() and any(company_resources.skills_dir.rglob("SKILL.md")):
                anima_shared_dirs.append(
                    (
                        "common_skills",
                        company_resources.skills_dir,
                        "SKILL.md",
                        "shared_company_skills_hash",
                    )
                )

        if not dry_run and not reset_shared_for_company_change(anima_dir, vector_store, current_company):
            logger.warning("  %s: shared reset incomplete, skipping indexing", anima_name)
            continue

        if not anima_shared_dirs:
            logger.info("  %s: no shared knowledge/skills files found, skipping", anima_name)
            continue

        for label, src_dir, glob, meta_key in anima_shared_dirs:
            current_hash = _compute_dir_hash(src_dir, glob)
            stored_hash = read_shared_hash(anima_dir, meta_key)

            if not full and current_hash == stored_hash:
                logger.info("  %s: %s unchanged, skipping", anima_name, label)
                continue

            logger.info("  %s: indexing %s...", anima_name, label)
            if dry_run:
                md_files = list(src_dir.rglob(glob))
                logger.info("    Would index %d files", len(md_files))
                continue

            shared_indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=base_dir,
                collection_prefix="shared",
            )
            result = shared_indexer.index_directory(src_dir, label, force=full)
            total += result.chunks_indexed
            if result.files_failed == 0:
                write_shared_hash(anima_dir, meta_key, current_hash)
            logger.info("    Indexed %d chunks", result.chunks_indexed)

    return total


def index_command(args: argparse.Namespace) -> None:
    """Execute the index command."""
    try:
        from core.memory.bm25 import rebuild_longterm_bm25_index
        from core.memory.rag import MemoryIndexer
        from core.memory.rag.repair import is_repair_locked
        from core.memory.rag.singleton import get_vector_store
    except ImportError:
        logger.error("RAG dependencies not installed. Run: pip install 'animaworks[rag]'")
        return

    base_dir = get_data_dir()
    animas_dir = base_dir / "animas"

    if not animas_dir.is_dir():
        logger.error("Animas directory not found: %s", animas_dir)
        logger.info("Run 'animaworks init' first to set up the environment")
        return

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

    phase3_dirs = [anima_dir for anima_dir in anima_dirs if _uses_root_vector_store(anima_dir)]
    for anima_dir in phase3_dirs:
        logger.error(
            "Cannot index phase3 anima %s from the CLI; use the running server's daily root indexing",
            anima_dir.name,
        )
    anima_dirs = [anima_dir for anima_dir in anima_dirs if anima_dir not in phase3_dirs]
    if not anima_dirs:
        return

    server_mode = _setup_server_delegation()
    temp_worker = _setup_offline_vector_worker_if_needed(server_mode)

    current_model = _check_model_change(base_dir, args.full)
    logger.info("Embedding model: %s", current_model)

    include_shared = args.shared or (not args.anima)

    total_chunks = 0
    for anima_dir in anima_dirs:
        anima_name = anima_dir.name
        logger.info("=" * 60)
        logger.info("Indexing anima: %s", anima_name)
        logger.info("=" * 60)

        if is_repair_locked(anima_name):
            logger.warning("Skipping %s: RAG repair lock is held", anima_name)
            continue

        vector_store = get_vector_store(anima_name)
        if vector_store is None:
            logger.warning("Vector store unavailable for %s, skipping", anima_name)
            continue

        if args.full and not args.dry_run:
            logger.info(
                "Full rebuild: deleting collections for %s (will recreate with cosine similarity)",
                anima_name,
            )
            collections = vector_store.list_collections_checked()
            if collections is None:
                logger.warning(
                    "Cannot list collections for %s; skipping full rebuild to avoid partial deletion",
                    anima_name,
                )
                continue
            for collection in collections:
                vector_store.delete_collection(collection)

        indexer = MemoryIndexer(vector_store, anima_name, anima_dir)

        memory_types = [
            ("knowledge", anima_dir / "knowledge"),
            ("episodes", anima_dir / "episodes"),
            ("procedures", anima_dir / "procedures"),
            ("skills", anima_dir / "skills"),
            ("facts", anima_dir / "facts"),
        ]

        for memory_type, memory_dir in memory_types:
            if not memory_dir.is_dir():
                logger.debug("Skipping %s (directory not found)", memory_type)
                continue

            logger.info("Indexing %s...", memory_type)

            if args.dry_run:
                if memory_type in ("skills", "common_skills"):
                    md_files = list(memory_dir.rglob("SKILL.md"))
                elif memory_type == "facts":
                    md_files = list(memory_dir.rglob("*.jsonl"))
                else:
                    md_files = list(memory_dir.rglob("*.md"))
                logger.info("  Would index %d files in %s/", len(md_files), memory_type)
                continue

            result = indexer.index_directory(
                memory_dir,
                memory_type,
                force=args.full,
            )
            total_chunks += result.chunks_indexed
            logger.info("  Indexed %d chunks from %s/", result.chunks_indexed, memory_type)

        state_dir = anima_dir / "state"
        conv_file = state_dir / "conversation.json"
        if conv_file.is_file():
            logger.info("Indexing conversation_summary...")
            if args.dry_run:
                logger.info("  Would index compressed_summary from conversation.json")
            else:
                chunks = indexer.index_conversation_summary(
                    state_dir,
                    anima_name,
                    force=args.full,
                )
                total_chunks += chunks
                logger.info("  Indexed %d chunks from conversation_summary", chunks)

        if not args.dry_run:
            from core.memory.entity_index import rebuild_entity_collection

            logger.info("Rebuilding entity collection...")
            if rebuild_entity_collection(anima_dir, vector_store=vector_store):
                logger.info("  Entity collection rebuild complete")
            else:
                logger.warning("  Failed to rebuild entity collection for %s", anima_name)

            bm25_result = rebuild_longterm_bm25_index(anima_dir)
            logger.info(
                "  Rebuilt long-term BM25 index with %d documents",
                bm25_result.documents,
            )

    if include_shared:
        enabled_dirs = [d for d in anima_dirs if _is_anima_enabled(d)]
        if enabled_dirs:
            logger.info("=" * 60)
            logger.info("Indexing shared collections (common_knowledge + common_skills)")
            logger.info("=" * 60)
            total_chunks += _index_shared_collections(
                enabled_dirs,
                base_dir,
                full=args.full,
                dry_run=args.dry_run,
            )

    shared_users_dir = base_dir / "shared" / "users"
    if shared_users_dir.is_dir() and not args.anima:
        logger.info("=" * 60)
        logger.info("Indexing shared user memories")
        logger.info("=" * 60)

        shared_store = get_vector_store(None)
        if shared_store is None:
            logger.warning("Shared vector store unavailable, skipping user memories")
        elif args.dry_run:
            user_dirs = [d for d in shared_users_dir.iterdir() if d.is_dir()]
            logger.info("  Would index %d user profiles", len(user_dirs))
        else:
            indexer = MemoryIndexer(shared_store, "shared", shared_users_dir.parent)
            result = indexer.index_directory(
                shared_users_dir,
                "shared_users",
                force=args.full,
            )
            total_chunks += result.chunks_indexed
            logger.info("  Indexed %d user profile chunks", result.chunks_indexed)

    logger.info("=" * 60)
    if args.dry_run:
        logger.info("Dry run complete (no actual indexing performed)")
    else:
        logger.info("Indexing complete: %d total chunks indexed", total_chunks)
        _save_global_index_meta(base_dir, current_model)
    _stop_offline_vector_worker(temp_worker)
