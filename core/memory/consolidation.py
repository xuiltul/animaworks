from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Memory consolidation engine — pre/post-processing helpers.

The actual consolidation (episode summarisation, knowledge extraction,
contradiction checks, etc.) is now performed by the Anima itself through
its tool-call loop (see ``Anima.run_consolidation()``).

This module retains:
- Episode and resolved-event collection (pre-processing for the Anima)
- RAG index updates and rebuilds (post-processing after the Anima finishes)
- Monthly forgetting (lifecycle.py post-processing)
- Legacy knowledge migration
- LLM output sanitisation (shared utility used by reconsolidation.py)
"""

import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.time_utils import ensure_aware, now_iso, now_jst

logger = logging.getLogger("animaworks.consolidation")


# ── ConsolidationEngine ────────────────────────────────────────


class ConsolidationEngine:
    """Pre/post-processing helpers for memory consolidation.

    The Anima itself now drives the consolidation loop via tool calls.
    This class provides:
    - **Pre-processing**: episode collection, resolved-event collection
    - **Post-processing**: RAG index updates/rebuilds, monthly forgetting
    - **Utilities**: knowledge file listing, LLM output sanitisation,
      legacy knowledge migration
    """

    def __init__(self, anima_dir: Path, anima_name: str, *, rag_store: Any | None = None) -> None:
        """Initialize consolidation engine.

        Args:
            anima_dir: Path to anima's directory (~/.animaworks/animas/{name})
            anima_name: Name of the anima for logging
            rag_store: Optional shared RAG vector store instance.
                When provided, avoids re-creating the singleton internally.
        """
        self.anima_dir = anima_dir
        self.anima_name = anima_name
        self._rag_store = rag_store
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # ── Legacy Migration ─────────────────────────────────────────

    def _migrate_legacy_knowledge(self) -> int:
        """Migrate legacy knowledge files to YAML frontmatter format.

        Detects knowledge files without frontmatter, creates backups, then
        rewrites them with ``---`` YAML frontmatter containing estimated
        metadata.  Controlled by a ``.migrated`` marker file so it runs
        only once per anima.

        Returns:
            Number of files migrated
        """
        marker = self.knowledge_dir / ".migrated"
        if marker.exists():
            return 0

        from core.memory.manager import MemoryManager

        # Use a lightweight MemoryManager to access frontmatter helpers
        mm = MemoryManager(self.anima_dir)

        backup_dir = self.anima_dir / "archive" / "pre_migration"
        migrated = 0

        for path in sorted(self.knowledge_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")

                # Skip files that already have frontmatter
                if text.startswith("---"):
                    continue

                # Create backup
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_dir / path.name)

                # Try to extract created_at from [AUTO-CONSOLIDATED: YYYY-MM-DD HH:MM]
                created_at = now_iso()
                ts_match = re.search(
                    r"\[AUTO-CONSOLIDATED:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]",
                    text,
                )
                if ts_match:
                    try:
                        parsed = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M")
                        created_at = parsed.isoformat()
                    except ValueError:
                        logger.debug("Failed to parse consolidation timestamp", exc_info=True)

                # Strip code fences that LLM may have wrapped around content
                content = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.MULTILINE)
                content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)
                content = content.strip()

                metadata = {
                    "created_at": created_at,
                    "confidence": 0.5,
                    "auto_consolidated": True,
                    "migrated_from_legacy": True,
                    "success_count": 0,
                    "failure_count": 0,
                    "version": 1,
                    "last_used": "",
                }

                mm.write_knowledge_with_meta(path, content, metadata)
                migrated += 1
                logger.info("Migrated legacy knowledge file: %s", path.name)

            except Exception:
                logger.exception("Failed to migrate knowledge file: %s", path.name)
                continue

        # Write marker
        marker.write_text(
            now_iso() + "\n",
            encoding="utf-8",
        )
        if migrated > 0:
            logger.info(
                "Legacy knowledge migration complete for anima=%s: migrated=%d",
                self.anima_name, migrated,
            )
        return migrated

    # ── Episode Collection ─────────────────────────────────────

    def _collect_recent_episodes(self, hours: int = 24) -> list[dict[str, str]]:
        """Collect episode entries from the past N hours.

        Supports both standard (YYYY-MM-DD.md) and suffixed
        (YYYY-MM-DD_xxx.md) episode filenames.  Files without
        ``## HH:MM — Title`` headers are treated as single entries
        using the file's mtime for timestamp.

        Args:
            hours: Number of hours to look back

        Returns:
            List of episode entries, each with 'date', 'time', 'content'
        """
        cutoff = now_jst() - timedelta(hours=hours)
        entries: list[dict[str, str]] = []

        # Check today and yesterday's episode files
        for day_offset in range(2):
            target_date = now_jst().date() - timedelta(days=day_offset)
            episode_files = sorted(self.episodes_dir.glob(f"{target_date}*.md"))

            for episode_file in episode_files:
                content = episode_file.read_text(encoding="utf-8")

                # Parse episode entries (format: ## HH:MM — Title)
                found_entries = list(re.finditer(
                    r"^## (\d{2}:\d{2})\s*—\s*(.+?)(?=^##|\Z)",
                    content,
                    re.MULTILINE | re.DOTALL,
                ))

                if found_entries:
                    for match in found_entries:
                        time_str = match.group(1)
                        entry_content = match.group(2).strip()

                        # Parse timestamp
                        try:
                            entry_dt = ensure_aware(datetime.strptime(
                                f"{target_date} {time_str}",
                                "%Y-%m-%d %H:%M",
                            ))

                            # Only include if within time window
                            if entry_dt >= cutoff:
                                entries.append({
                                    "date": str(target_date),
                                    "time": time_str,
                                    "content": entry_content,
                                })
                        except ValueError:
                            logger.warning(
                                "Failed to parse episode timestamp: %s %s",
                                target_date, time_str,
                            )
                else:
                    # Fallback: treat entire file as a single entry using mtime
                    file_mtime = ensure_aware(datetime.fromtimestamp(
                        episode_file.stat().st_mtime,
                    ))
                    if file_mtime >= cutoff:
                        entries.append({
                            "date": str(target_date),
                            "time": file_mtime.strftime("%H:%M"),
                            "content": content.strip(),
                        })

        # Deduplicate by content prefix (first 200 chars)
        seen: set[str] = set()
        unique_entries: list[dict[str, str]] = []
        for entry in entries:
            dedup_key = entry["content"][:200].strip()
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_entries.append(entry)
        entries = unique_entries

        # Sort by datetime (newest first)
        entries.sort(
            key=lambda e: datetime.strptime(f"{e['date']} {e['time']}", "%Y-%m-%d %H:%M"),
            reverse=True,
        )

        return entries

    def _collect_resolved_events(self, hours: int = 24) -> list[dict]:
        """Collect issue_resolved events from activity log."""
        try:
            from core.memory.activity import ActivityLogger
            activity = ActivityLogger(self.anima_dir)
            entries = activity.recent(days=1, limit=50, types=["issue_resolved"])
            return [
                {"ts": e.ts, "content": e.content, "summary": e.summary, "meta": e.meta or {}}
                for e in entries
            ]
        except Exception:
            logger.debug("Failed to collect resolved events", exc_info=True)
            return []

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def _sanitize_llm_output(text: str) -> str:
        """Remove code fences from LLM output.

        LLMs sometimes wrap their entire response in ```markdown fences.
        This method strips those wrapper fences while preserving any
        intentional code blocks within the content.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned text with wrapper code fences removed
        """
        text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    def _list_knowledge_files(self) -> list[str]:
        """List all existing knowledge files.

        Returns:
            List of knowledge file paths (relative to knowledge/)
        """
        if not self.knowledge_dir.exists():
            return []

        files = []
        for path in self.knowledge_dir.rglob("*.md"):
            rel_path = path.relative_to(self.knowledge_dir)
            files.append(str(rel_path))

        return sorted(files)

    # ── RAG Index ────────────────────────────────────────────────

    def _update_rag_index(self, filenames: list[str]) -> None:
        """Update RAG index for the specified knowledge files.

        Args:
            filenames: List of knowledge file names (relative to knowledge/)
        """
        if not filenames:
            return

        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store(self.anima_name)
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)

            for filename in filenames:
                filepath = self.knowledge_dir / filename
                if filepath.exists():
                    indexer.index_file(filepath, memory_type="knowledge")
                    logger.debug("Updated RAG index for: %s", filename)

        except ImportError:
            logger.debug("RAG not available, skipping index update")
        except Exception as e:
            logger.warning("Failed to update RAG index: %s", e)

    def _rebuild_rag_index(self) -> None:
        """Rebuild RAG index for all knowledge and episode files."""
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store(self.anima_name)
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)

            # Re-index all knowledge files
            for knowledge_file in self.knowledge_dir.rglob("*.md"):
                indexer.index_file(knowledge_file, memory_type="knowledge")
                logger.debug("Re-indexed knowledge: %s", knowledge_file.name)

            # Re-index all episode files
            for episode_file in self.episodes_dir.glob("*.md"):
                indexer.index_file(episode_file, memory_type="episodes")
                logger.debug("Re-indexed episode: %s", episode_file.name)

            logger.info("RAG index rebuild complete for anima=%s", self.anima_name)

        except ImportError:
            logger.debug("RAG not available, skipping index rebuild")
        except Exception:
            logger.exception("Failed to rebuild RAG index")

    # ── Monthly Forgetting ──────────────────────────────────────

    async def monthly_forget(self) -> dict[str, Any]:
        """Perform monthly forgetting: archive and remove forgotten memories.

        This is the final stage of the forgetting pipeline, removing
        memories that have remained at low activation for extended periods.
        Also cleans up old procedure version archives.
        """
        logger.info("Starting monthly forgetting for anima=%s", self.anima_name)
        try:
            from core.memory.forgetting import ForgettingEngine
            forgetter = ForgettingEngine(self.anima_dir, self.anima_name)
            result = forgetter.complete_forgetting()

            # Clean up old procedure version archives
            try:
                archive_result = forgetter.cleanup_procedure_archives()
                result["procedure_archive_cleanup"] = archive_result
                logger.info(
                    "Procedure archive cleanup for anima=%s: "
                    "deleted=%d, kept=%d",
                    self.anima_name,
                    archive_result.get("deleted_count", 0),
                    archive_result.get("kept_count", 0),
                )
            except Exception:
                logger.exception(
                    "Procedure archive cleanup failed for anima=%s",
                    self.anima_name,
                )

            # Rebuild RAG index after deletions
            self._rebuild_rag_index()

            logger.info(
                "Monthly forgetting complete for anima=%s: "
                "forgotten=%d, archived=%d files",
                self.anima_name,
                result.get("forgotten_chunks", 0),
                len(result.get("archived_files", [])),
            )
            return result

        except Exception:
            logger.exception("Monthly forgetting failed for anima=%s", self.anima_name)
            return {"forgotten_chunks": 0, "archived_files": [], "error": True}
