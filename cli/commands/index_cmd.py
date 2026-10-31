from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Index command - build or rebuild RAG vector indexes."""

import argparse
import logging
from pathlib import Path
from typing import Any

from core.paths import get_data_dir
from core.platform.atomic_io import atomic_write_json

logger = logging.getLogger("animaworks.cli.index")


def _setup_server_delegation() -> bool:
    """Detect running server and configure HTTP delegation.

    When the server is running, configures vector/embed/rerank URLs so that
    ``get_vector_store(anima_name)`` delegates to the server and embeddings /
    rerank are generated server-side.  This prevents unsafe concurrent
    ChromaDB access and per-process model loads.

    Returns:
        True if delegation was activated (server is running).
    """
    from core.platform.pid import read_server_pid
    from core.platform.process import is_process_alive

    pid = read_server_pid()
    from core.memory.rag.endpoints import RagEndpoints, configure_endpoints

    if pid is None or not is_process_alive(pid):
        configure_endpoints(RagEndpoints())
        return False

    from core.platform.env import server_port

    port = server_port()
    configure_endpoints(RagEndpoints.for_server(port))
    logger.info(
        "Server detected (pid=%d). Using HTTP delegation for safe ChromaDB access.",
        pid,
    )
    return True


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

    from core.memory.rag.embedding import get_embedding_e5_prefix_enabled, get_embedding_model_name

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

    from core.memory.rag.embedding import get_embedding_e5_prefix_enabled

    meta_path = base_dir / "index_meta.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    meta["embedding_model"] = model_name
    meta["embedding_e5_prefix"] = get_embedding_e5_prefix_enabled()
    atomic_write_json(meta_path, meta, indent=2, ensure_ascii=False)


def _is_anima_enabled(anima_dir: Path) -> bool:
    """Check whether an anima is enabled via its status.json."""
    from core.platform.status_store import read_status

    return read_status(anima_dir).get("enabled", True)


def _index_shared_for_anima(
    anima_dir: Path,
    base_dir: Path,
    vector_store: Any,
    *,
    full: bool,
    dry_run: bool,
) -> int:
    """Index shared collections for one anima using its already-open vector store."""
    from core.config.models import read_anima_company_checked
    from core.memory.rag import MemoryIndexer
    from core.memory.rag.shared_meta import read_shared_hash, reset_shared_for_company_change, write_shared_hash
    from core.memory.retrieval.rag_search import _compute_dir_hash
    from core.org.company_resources import get_company_resources_for_company

    anima_name = anima_dir.name
    company_valid, company = read_anima_company_checked(anima_dir)
    if not company_valid:
        logger.warning("  %s: company membership unavailable, skipping shared indexing", anima_name)
        return 0
    current_company = company or ""
    company_resources = get_company_resources_for_company(company, data_dir=base_dir)

    ck_dir = base_dir / "common_knowledge"
    cs_dir = base_dir / "common_skills"
    anima_shared_dirs: list[tuple[str, Path, str, str]] = []
    if ck_dir.is_dir() and any(ck_dir.rglob("*.md")):
        anima_shared_dirs.append(("common_knowledge", ck_dir, "*.md", "shared_common_knowledge_hash"))
    if cs_dir.is_dir() and any(cs_dir.rglob("SKILL.md")):
        anima_shared_dirs.append(("common_skills", cs_dir, "SKILL.md", "shared_common_skills_hash"))
    if company_resources is not None:
        if company_resources.knowledge_dir.is_dir() and any(company_resources.knowledge_dir.rglob("*.md")):
            anima_shared_dirs.append(
                ("common_knowledge", company_resources.knowledge_dir, "*.md", "shared_company_knowledge_hash")
            )
        if company_resources.skills_dir.is_dir() and any(company_resources.skills_dir.rglob("SKILL.md")):
            anima_shared_dirs.append(
                ("common_skills", company_resources.skills_dir, "SKILL.md", "shared_company_skills_hash")
            )

    if not anima_shared_dirs:
        logger.info("  %s: no shared knowledge/skills files found, skipping", anima_name)
        return 0
    if not dry_run and not reset_shared_for_company_change(anima_dir, vector_store, current_company):
        logger.warning("  %s: shared reset incomplete, skipping indexing", anima_name)
        return 0

    total = 0
    for label, src_dir, glob, meta_key in anima_shared_dirs:
        current_hash = _compute_dir_hash(src_dir, glob)
        stored_hash = read_shared_hash(anima_dir, meta_key)
        if not full and current_hash == stored_hash:
            logger.info("  %s: %s unchanged, skipping", anima_name, label)
            continue

        logger.info("  %s: indexing %s...", anima_name, label)
        if dry_run:
            files = list(src_dir.rglob(glob))
            logger.info("    Would index %d files", len(files))
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


def _index_anima(
    args: argparse.Namespace,
    anima_dir: Path,
    indexer_cls: Any,
    rebuild_bm25: Any,
    *,
    base_dir: Path | None = None,
    include_shared: bool = False,
) -> int:
    """Index one anima while holding exactly one vector owner access."""
    from core.memory.rag.cli_access import open_vector_access
    from core.memory.rag.repair import is_repair_locked

    anima_name = anima_dir.name
    if is_repair_locked(anima_name):
        logger.warning("Skipping %s: RAG repair lock is held", anima_name)
        return 0
    if base_dir is None:
        base_dir = get_data_dir()

    total_chunks = 0
    with open_vector_access(anima_name, anima_dir, purpose="index") as access:
        vector_store = access.store
        if args.full and not args.dry_run:
            logger.info("Full rebuild: rebuilding all vector collections for %s", anima_name)
            repair_result = access.repair(include_shared=True)
            if not repair_result.get("ok", repair_result.get("status") in {"success", "healthy"}):
                raise RuntimeError(f"Full RAG rebuild failed for {anima_name}: {repair_result}")

        indexer = indexer_cls(vector_store, anima_name, anima_dir)
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
                if memory_type == "skills":
                    files = list(memory_dir.rglob("SKILL.md"))
                elif memory_type == "facts":
                    files = list(memory_dir.rglob("*.jsonl"))
                else:
                    files = list(memory_dir.rglob("*.md"))
                logger.info("  Would index %d files in %s/", len(files), memory_type)
                continue

            result = indexer.index_directory(memory_dir, memory_type, force=args.full)
            total_chunks += result.chunks_indexed
            logger.info("  Indexed %d chunks from %s/", result.chunks_indexed, memory_type)

        state_dir = anima_dir / "state"
        if (state_dir / "conversation.json").is_file():
            logger.info("Indexing conversation_summary...")
            if args.dry_run:
                logger.info("  Would index compressed_summary from conversation.json")
            else:
                chunks = indexer.index_conversation_summary(state_dir, anima_name, force=args.full)
                total_chunks += chunks
                logger.info("  Indexed %d chunks from conversation_summary", chunks)

        if not args.dry_run:
            from core.memory.facts.entity_index import rebuild_entity_collection

            logger.info("Rebuilding entity collection...")
            if rebuild_entity_collection(anima_dir, vector_store=vector_store):
                logger.info("  Entity collection rebuild complete")
            else:
                logger.warning("  Failed to rebuild entity collection for %s", anima_name)
            bm25_result = rebuild_bm25(anima_dir)
            logger.info("  Rebuilt long-term BM25 index with %d documents", bm25_result.documents)

        if include_shared and _is_anima_enabled(anima_dir):
            total_chunks += _index_shared_for_anima(
                anima_dir,
                base_dir,
                vector_store,
                full=args.full,
                dry_run=args.dry_run,
            )

    return total_chunks


def index_command(args: argparse.Namespace) -> None:
    """Execute the index command."""
    try:
        from core.memory.rag import MemoryIndexer
        from core.memory.retrieval.bm25 import rebuild_longterm_bm25_index
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

    current_model = _check_model_change(base_dir, args.full)
    logger.info("Embedding model: %s", current_model)

    include_shared = args.shared or (not args.anima)

    total_chunks = 0
    errors: list[str] = []
    for anima_dir in anima_dirs:
        anima_name = anima_dir.name
        logger.info("=" * 60)
        logger.info("Indexing anima: %s", anima_name)
        logger.info("=" * 60)
        try:
            total_chunks += _index_anima(
                args,
                anima_dir,
                MemoryIndexer,
                rebuild_longterm_bm25_index,
                base_dir=base_dir,
                include_shared=include_shared,
            )
        except Exception as exc:
            from core.memory.rag.owner_lock import VectorOwnerBusy

            if not isinstance(exc, VectorOwnerBusy):
                logger.exception("Indexing failed for %s", anima_name)
            else:
                logger.error("Cannot access vector store for %s: %s", anima_name, exc)
            errors.append(anima_name)

    logger.info("=" * 60)
    if args.dry_run:
        logger.info("Dry run complete (no actual indexing performed)")
    else:
        logger.info("Indexing complete: %d total chunks indexed", total_chunks)
        _save_global_index_meta(base_dir, current_model)
    if errors:
        raise SystemExit(1)
