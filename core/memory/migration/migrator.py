from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory migrator — Legacy RAG to Neo4j backend."""

import logging
from collections.abc import Callable
from pathlib import Path

from core.memory.migration.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ── MemoryMigrator ─────────────────────────────────────────────


class MemoryMigrator:
    """Migrate Anima memory from Legacy (ChromaDB) to Neo4j backend."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._animas_dir = data_dir / "animas"

    # ── Discovery ──────────────────────────────────────────────

    def list_animas(self) -> list[str]:
        """List all Anima names in data directory."""
        if not self._animas_dir.is_dir():
            return []
        return sorted(d.name for d in self._animas_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    def count_files(self, anima_name: str) -> dict[str, int]:
        """Count memory files for an Anima (for dry-run)."""
        anima_dir = self._animas_dir / anima_name
        counts: dict[str, int] = {}
        for subdir in ("knowledge", "episodes", "procedures", "skills"):
            d = anima_dir / subdir
            if d.is_dir():
                counts[subdir] = sum(1 for f in d.rglob("*.md") if f.is_file())
        return counts

    # ── Migration ──────────────────────────────────────────────

    async def migrate_anima(
        self,
        anima_name: str,
        *,
        checkpoint_manager: CheckpointManager | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> dict[str, int]:
        """Migrate a single Anima's memory files to Neo4j.

        Args:
            anima_name: Name of the Anima to migrate.
            checkpoint_manager: Optional ``CheckpointManager`` for resume.
            on_progress: Optional callback ``(file_path, status)``.

        Returns:
            Dict with counts:
            ``{"files": N, "entities": N, "facts": N, "skipped": N, "errors": N}``
        """
        anima_dir = self._animas_dir / anima_name
        if not anima_dir.is_dir():
            raise FileNotFoundError(f"Anima directory not found: {anima_dir}")

        from core.memory.backend.registry import get_backend

        backend = get_backend("neo4j", anima_dir)

        stats = {
            "files": 0,
            "entities": 0,
            "facts": 0,
            "skipped": 0,
            "errors": 0,
        }

        for subdir in ("knowledge", "episodes", "procedures"):
            mem_dir = anima_dir / subdir
            if not mem_dir.is_dir():
                continue

            for md_file in sorted(mem_dir.rglob("*.md")):
                key = f"{anima_name}:{md_file.relative_to(anima_dir)}"

                if checkpoint_manager and checkpoint_manager.is_done(key):
                    stats["skipped"] += 1
                    continue

                try:
                    count = await backend.ingest_file(md_file)
                    stats["files"] += 1
                    stats["entities"] += count

                    if checkpoint_manager:
                        checkpoint_manager.mark_done(key, anima=anima_name, file=str(md_file))
                    if on_progress:
                        on_progress(str(md_file), "done")

                except Exception as e:
                    stats["errors"] += 1
                    logger.warning("Migration failed for %s: %s", md_file, e)
                    if checkpoint_manager:
                        checkpoint_manager.mark_error(
                            key,
                            str(e),
                            anima=anima_name,
                            file=str(md_file),
                        )
                    if on_progress:
                        on_progress(str(md_file), f"error: {e}")

        try:
            await backend.close()
        except Exception as e:
            logger.debug("backend.close() failed (ignored): %s", e)

        return stats

    # ── Cost estimation ────────────────────────────────────────

    def estimate_cost(self, anima_name: str, *, tokens_per_file: int = 2000) -> dict[str, int | float]:
        """Estimate migration cost for an Anima.

        Args:
            anima_name: Anima to estimate.
            tokens_per_file: Estimated tokens per extraction call.

        Returns:
            Dict with ``estimated_files``, ``estimated_tokens``,
            ``estimated_llm_calls``, and ``file_counts``.
        """
        counts = self.count_files(anima_name)
        total_files = sum(counts.values())
        llm_calls = total_files * 2
        tokens = total_files * tokens_per_file * 2
        return {
            "estimated_files": total_files,
            "estimated_llm_calls": llm_calls,
            "estimated_tokens": tokens,
            "file_counts": counts,
        }
