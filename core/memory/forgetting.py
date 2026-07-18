from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Active forgetting mechanism based on synaptic homeostasis hypothesis.

Implements three stages of memory forgetting:
1. Synaptic downscaling (daily): Mark low-activation chunks
2. Neurogenesis reorganization (weekly): Merge similar low-activation chunks
3. Complete forgetting (monthly): Archive and delete forgotten memories

Based on:
- Tononi & Cirelli (2003, 2006): Synaptic homeostasis hypothesis
- Frankland et al. (2013): Hippocampal neurogenesis and active forgetting
"""

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.paths import load_prompt
from core.time_utils import ensure_aware, now_local, today_local

logger = logging.getLogger("animaworks.forgetting")

# ── Configuration ──────────────────────────────────────────────────

# Synaptic downscaling thresholds
DOWNSCALING_DAYS_THRESHOLD = 90  # Days since last access
DOWNSCALING_ACCESS_THRESHOLD = 3  # Minimum access count to avoid marking

# Neurogenesis reorganization
REORGANIZATION_SIMILARITY_THRESHOLD = 0.80  # Vector similarity for merging

# Complete forgetting
FORGETTING_LOW_ACTIVATION_DAYS = 90  # Days in low activation before deletion
FORGETTING_MAX_ACCESS_COUNT = 2  # Max access count to still be eligible for deletion

# Protected memory types (skills and shared_users are fully protected)
PROTECTED_MEMORY_TYPES = frozenset({"skills", "shared_users"})

# [IMPORTANT] safety net: after this many days without access,
# important chunks lose protection (conceptual integration should
# have happened by then via weekly consolidation prompts).
IMPORTANT_SAFETY_NET_DAYS = 365

# Procedure-specific forgetting thresholds (more lenient than knowledge)
PROCEDURE_INACTIVITY_DAYS = 180  # Days since last use (vs 90 for knowledge)
PROCEDURE_MIN_USAGE = 3  # Minimum total usage to avoid downscaling
PROCEDURE_LOW_UTILITY_THRESHOLD = 0.3  # Utility score below this is low
PROCEDURE_LOW_UTILITY_MIN_FAILURES = 3  # Min failures for utility check
PROCEDURE_ARCHIVE_KEEP_VERSIONS = 5  # Keep N most recent archive versions


@dataclass(frozen=True)
class _SourceSegment:
    """Source-file segment reconstructed with indexer-compatible chunk indexes."""

    chunk_index: int | None
    content: str


@dataclass(frozen=True)
class _SourceWritePlan:
    """Planned source-file mutation after vector merge succeeds."""

    path: Path
    source_file: str
    content: str | None
    archive: bool
    delete: bool
    label: str


# ── ForgettingEngine ───────────────────────────────────────────────


class ForgettingEngine:
    """Active forgetting based on synaptic homeostasis and neurogenesis."""

    def __init__(self, anima_dir: Path, anima_name: str) -> None:
        self.anima_dir = anima_dir
        self.anima_name = anima_name
        self.archive_dir = anima_dir / "archive" / "forgotten"

    def _is_protected(self, metadata: dict) -> bool:
        """Check if a chunk is protected from forgetting.

        Fully protected types (skills, shared_users) are always skipped.
        Procedures use utility-based protection via ``_is_protected_procedure``.
        Knowledge with ``success_count >= 2`` is protected via
        ``_is_protected_knowledge``.
        The ``importance == "important"`` tag protects any memory type,
        but with a safety-net: chunks that remain unaccessed for
        ``IMPORTANT_SAFETY_NET_DAYS`` lose their protection (conceptual
        integration via weekly consolidation should have happened by then).
        """
        if metadata.get("memory_type") in PROTECTED_MEMORY_TYPES:
            return True
        important_expired = False
        if metadata.get("importance") == "important":
            important_expired = self._important_safety_net_expired(metadata)
            if not important_expired:
                return True
        # Procedures use utility-based protection instead of blanket protection
        if metadata.get("memory_type") == "procedures":
            return self._is_protected_procedure(metadata, important_expired=important_expired)
        # Knowledge with confirmed usefulness is protected
        if metadata.get("memory_type") == "knowledge":
            return self._is_protected_knowledge(metadata, important_expired=important_expired)
        return False

    def _important_safety_net_expired(self, metadata: dict) -> bool:
        """Check if an [IMPORTANT] chunk has exceeded the safety-net window.

        Returns True when the chunk has gone unaccessed for longer than
        ``IMPORTANT_SAFETY_NET_DAYS``, meaning conceptual integration
        should have occurred by now and the raw episodic [IMPORTANT]
        can safely enter normal forgetting.
        """
        used_count = self._used_count(metadata)
        if used_count > 0:
            last_used_str = self._last_used_at(metadata)
            if last_used_str:
                try:
                    last_dt = ensure_aware(datetime.fromisoformat(str(last_used_str)))
                    days = (now_local() - last_dt).total_seconds() / 86400.0
                    return days > IMPORTANT_SAFETY_NET_DAYS
                except (ValueError, TypeError):
                    pass
            return False

        updated_str = metadata.get("updated_at", "")
        if updated_str:
            try:
                updated_dt = ensure_aware(datetime.fromisoformat(str(updated_str)))
                days = (now_local() - updated_dt).total_seconds() / 86400.0
                return days > IMPORTANT_SAFETY_NET_DAYS
            except (ValueError, TypeError):
                pass
        return False

    @staticmethod
    def _number(metadata: dict, key: str, *, default: float = 0.0) -> float:
        try:
            return max(0.0, float(str(metadata.get(key, default))))
        except (TypeError, ValueError):
            return max(0.0, default)

    def _used_count(self, metadata: dict) -> float:
        """Return combined usage: explicit uses plus automatic recalls (F11).

        Auto-recall (``access_count``) now counts toward "usage" so memories
        that keep getting retrieved by search stay protected from forgetting,
        matching the access-boost philosophy. Legacy chunks predating the
        used/access split carry only ``access_count`` (``used_count`` absent →
        0), so the sum equals the legacy value and remains backward compatible.
        Explicit uses bump both counters, which merely over-protects (never
        under-protects) already-used memories.
        """
        return self._number(metadata, "used_count") + self._number(metadata, "access_count")

    @staticmethod
    def _last_used_at(metadata: dict) -> str:
        """Return the most recent usage timestamp (F11).

        Combines the explicit-use time (``last_used_at`` / legacy ``last_used``)
        with the automatic-recall time (``last_accessed_at``) and returns the
        latest, so a memory that keeps getting retrieved registers as recently
        used. Returns an empty string when no timestamp is present.
        """
        candidates: list[str] = []
        for key in ("last_used_at", "last_used", "last_accessed_at"):
            value = str(metadata.get(key, "") or "").strip()
            if value:
                candidates.append(value)
        if not candidates:
            return ""
        return max(candidates)

    def _is_protected_knowledge(self, metadata: dict, *, important_expired: bool = False) -> bool:
        """Knowledge-specific protection check.

        Returns True (protected) if any of:
        - ``importance == "important"`` ([IMPORTANT] tag), unless the
          top-level important safety net has expired
        - ``success_count >= 2`` (knowledge confirmed useful multiple times)

        Args:
            metadata: Chunk metadata from the vector store.

        Returns:
            True if the knowledge chunk should be protected from forgetting.
        """
        if metadata.get("importance") == "important" and not important_expired:
            return True
        if int(metadata.get("success_count", 0)) >= 2:  # noqa: SIM103
            return True
        return False

    def _is_protected_procedure(self, metadata: dict, *, important_expired: bool = False) -> bool:
        """Procedure-specific protection check.

        Returns True (protected) if any of:
        - ``importance == "important"`` ([IMPORTANT] tag), unless the
          top-level important safety net has expired
        - ``protected is True`` (manual protection flag)
        - ``version >= 3`` (mature procedure that survived reconsolidation)
        """
        if metadata.get("importance") == "important" and not important_expired:
            return True
        if metadata.get("protected") is True:
            return True
        if metadata.get("version", 1) >= 3:  # noqa: SIM103
            return True
        return False

    def _should_downscale_procedure(
        self,
        metadata: dict,
        now: datetime,
    ) -> bool:
        """Procedure-specific downscaling check.

        A procedure is marked low-activation if either:
        1. Inactive for >180 days AND total usage < 3 (rarely used, old)
        2. failure_count >= 3 AND utility score < 0.3 (high failure rate)

        Args:
            metadata: Chunk metadata from the vector store.
            now: Current datetime for age calculation.

        Returns:
            True if the procedure should be marked as low-activation.
        """
        # Calculate days since last use
        last_used_str = self._last_used_at(metadata)
        if not last_used_str:
            last_used_str = metadata.get("updated_at", "")

        if last_used_str:
            try:
                last_used_dt = ensure_aware(datetime.fromisoformat(str(last_used_str)))
                days_since = (now - last_used_dt).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_since = float("inf")
        else:
            days_since = float("inf")

        success_count = int(metadata.get("success_count", 0))
        failure_count = int(metadata.get("failure_count", 0))
        total_usage = max(success_count + failure_count, int(self._used_count(metadata)))

        # Condition 1: Long inactivity + low total usage
        if days_since > PROCEDURE_INACTIVITY_DAYS and total_usage < PROCEDURE_MIN_USAGE:
            return True

        # Condition 2: High failure rate (utility < 0.3 with >= 3 failures)
        if failure_count >= PROCEDURE_LOW_UTILITY_MIN_FAILURES:
            utility = success_count / max(1, total_usage)
            if utility < PROCEDURE_LOW_UTILITY_THRESHOLD:
                return True

        return False

    def _get_vector_store(self):
        """Get vector store singleton.

        Returns:
            VectorStore instance, or ``None`` if unavailable.
        """
        from core.memory.rag.singleton import get_vector_store

        return get_vector_store(self.anima_name)

    def _get_all_chunks(self, collection_name: str) -> list[dict]:
        """Get all chunks from a collection with their metadata."""
        try:
            store = self._get_vector_store()
            if store is None:
                return []
            results = store.get_by_metadata(collection_name, {}, limit=100_000)
            return [
                {
                    "id": r.document.id,
                    "metadata": dict(r.document.metadata),
                    "content": r.document.content,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("Failed to get chunks from %s: %s", collection_name, e)
            return []

    # ── Stage 1: Synaptic Downscaling (Daily) ──────────────────────

    def synaptic_downscaling(self) -> dict[str, Any]:
        """Mark low-activation chunks (daily, runs in daily_consolidate).

        Criteria: days_since_access > 90 AND access_count < 3
        Action: Set activation_level="low", record low_activation_since
        Skip: Protected memory types, important chunks, already low
        """
        logger.info("Starting synaptic downscaling for anima=%s", self.anima_name)
        now = now_local()
        now_iso_str = now.isoformat()
        total_scanned = 0
        total_marked = 0
        store = self._get_vector_store()

        if store is None:
            logger.warning(
                "Skipping synaptic downscaling for anima=%s: RAG/ChromaDB unavailable",
                self.anima_name,
            )
            return {"scanned": 0, "marked_low": 0, "skipped_reason": "rag_unavailable"}

        # Scan all relevant collections (including procedures)
        for memory_type in ("knowledge", "episodes", "procedures"):
            collection_name = f"{self.anima_name}_{memory_type}"
            chunks = self._get_all_chunks(collection_name)
            total_scanned += len(chunks)

            ids_to_mark: list[str] = []
            metas_to_mark: list[dict] = []

            for chunk in chunks:
                meta = chunk["metadata"]

                # Skip protected
                if self._is_protected(meta):
                    continue

                # Skip already low
                if meta.get("activation_level") == "low":
                    continue

                # Procedure-specific downscaling logic
                if meta.get("memory_type") == "procedures":
                    if self._should_downscale_procedure(meta, now):
                        ids_to_mark.append(chunk["id"])
                        metas_to_mark.append(
                            {
                                "activation_level": "low",
                                "low_activation_since": now_iso_str,
                            }
                        )
                    continue

                # Check access recency
                used_count = self._used_count(meta)
                last_used_str = self._last_used_at(meta)

                if last_used_str:
                    try:
                        last_used = ensure_aware(datetime.fromisoformat(str(last_used_str)))
                        days_since = (now - last_used).total_seconds() / 86400.0
                    except (ValueError, TypeError):
                        days_since = float("inf")
                else:
                    # Never used — use updated_at as fallback
                    updated_str = meta.get("updated_at", "")
                    if updated_str:
                        try:
                            updated_at = ensure_aware(datetime.fromisoformat(str(updated_str)))
                            days_since = (now - updated_at).total_seconds() / 86400.0
                        except (ValueError, TypeError):
                            days_since = float("inf")
                    else:
                        days_since = float("inf")

                # Apply threshold
                if days_since > DOWNSCALING_DAYS_THRESHOLD and used_count < DOWNSCALING_ACCESS_THRESHOLD:
                    ids_to_mark.append(chunk["id"])
                    metas_to_mark.append(
                        {
                            "activation_level": "low",
                            "low_activation_since": now_iso_str,
                        }
                    )

            # Batch update
            if ids_to_mark:
                try:
                    store.update_metadata(collection_name, ids_to_mark, metas_to_mark)
                    total_marked += len(ids_to_mark)
                    logger.info(
                        "Marked %d chunks as low-activation in %s",
                        len(ids_to_mark),
                        collection_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to mark chunks in %s: %s",
                        collection_name,
                        e,
                    )

        result = {
            "scanned": total_scanned,
            "marked_low": total_marked,
        }
        logger.info(
            "Synaptic downscaling complete for anima=%s: scanned=%d, marked=%d",
            self.anima_name,
            total_scanned,
            total_marked,
        )
        logger.info(
            "forgetting_funnel: anima=%s stage=downscaling scanned=%d marked=%d merged=0 forgotten=0",
            self.anima_name,
            total_scanned,
            total_marked,
        )
        return result

    # ── Stage 2: Neurogenesis Reorganization (Weekly) ──────────────

    async def neurogenesis_reorganize(
        self,
        model: str = "",
    ) -> dict[str, Any]:
        """Merge similar low-activation chunks (weekly, runs in weekly_integrate).

        Criteria: activation_level=="low" AND pairwise vector similarity >= 0.8
        Action: LLM merge -> delete originals -> insert merged chunk
        """
        if not model:
            from core.memory._llm_utils import get_consolidation_llm_kwargs

            model = get_consolidation_llm_kwargs()["model"]
        logger.info("Starting neurogenesis reorganization for anima=%s", self.anima_name)
        store = self._get_vector_store()

        if store is None:
            logger.warning(
                "Skipping neurogenesis reorganization for anima=%s: RAG/ChromaDB unavailable",
                self.anima_name,
            )
            return {"merged_count": 0, "merged_pairs": [], "skipped_reason": "rag_unavailable"}

        total_merged = 0
        total_scanned = 0
        total_marked = 0
        merged_pairs: list[str] = []

        for memory_type in ("knowledge", "episodes", "procedures"):
            collection_name = f"{self.anima_name}_{memory_type}"

            # F10: purge legacy synthetic "merged" chunks. Merged content now
            # lives only in the primary source .md (re-indexed by the weekly
            # rebuild); directly-upserted source_file="merged" chunks left over
            # from the old flow duplicate that content and must be removed.
            self._purge_merged_chunks(collection_name, store)

            chunks = self._get_all_chunks(collection_name)
            total_scanned += len(chunks)

            # Filter low-activation chunks
            low_chunks = [
                c
                for c in chunks
                if c["metadata"].get("activation_level") == "low" and not self._is_protected(c["metadata"])
            ]
            total_marked += len(low_chunks)

            if len(low_chunks) < 2:
                continue

            # Find similar pairs using vector similarity
            similar_pairs = self._find_similar_pairs(
                low_chunks,
                collection_name,
                store,
            )

            if not similar_pairs:
                continue

            # Merge each pair via LLM
            for chunk_a, chunk_b, similarity in similar_pairs:
                try:
                    merged_content = await self._merge_chunks_llm(
                        chunk_a,
                        chunk_b,
                        similarity,
                        model,
                    )
                    if merged_content:
                        source_plans = self._plan_merged_source_files(
                            chunk_a,
                            chunk_b,
                            merged_content,
                        )
                        if source_plans is None:
                            logger.warning(
                                "Skipping neurogenesis merge for %s and %s because source sync was skipped",
                                chunk_a["id"],
                                chunk_b["id"],
                            )
                            continue

                        # Delete originals
                        store.delete_documents(
                            collection_name,
                            [chunk_a["id"], chunk_b["id"]],
                        )
                        # Sync source files on disk so next RAG re-index
                        # reflects the merge instead of restoring originals.
                        # The merged content is written to the primary .md here
                        # and re-indexed by the weekly rebuild (F10) — we no
                        # longer upsert a synthetic "merged" chunk that would
                        # duplicate that content after rebuild.
                        if not self._apply_source_write_plans(source_plans):
                            logger.warning(
                                "Merged vectors for %s and %s but failed to sync source files",
                                chunk_a["id"],
                                chunk_b["id"],
                            )
                            continue
                        total_merged += 1
                        merged_pairs.append(f"{chunk_a['id']} + {chunk_b['id']}")
                except Exception as e:
                    logger.warning(
                        "Failed to merge chunks %s and %s: %s",
                        chunk_a["id"],
                        chunk_b["id"],
                        e,
                    )

        result = {
            "scanned": total_scanned,
            "marked_low": total_marked,
            "merged_count": total_merged,
            "merged_pairs": merged_pairs,
        }
        logger.info(
            "Neurogenesis reorganization complete for anima=%s: merged=%d",
            self.anima_name,
            total_merged,
        )
        logger.info(
            "forgetting_funnel: anima=%s stage=reorganization scanned=%d marked=%d merged=%d forgotten=0",
            self.anima_name,
            total_scanned,
            total_marked,
            total_merged,
        )
        return result

    def _find_similar_pairs(
        self,
        chunks: list[dict],
        collection_name: str,
        store,
    ) -> list[tuple[dict, dict, float]]:
        """Find pairs of low-activation chunks with high vector similarity."""
        from core.memory.rag.singleton import generate_embeddings

        pairs: list[tuple[dict, dict, float]] = []
        processed_ids: set[str] = set()

        try:
            embeddings = generate_embeddings(
                [chunk["content"] for chunk in chunks],
                purpose="query",
                priority="bulk",
            )
            embeddings_by_id = {chunk["id"]: embedding for chunk, embedding in zip(chunks, embeddings, strict=False)}

            for _, chunk_a in enumerate(chunks):
                if chunk_a["id"] in processed_ids:
                    continue

                embedding = embeddings_by_id.get(chunk_a["id"])
                if embedding is None:
                    continue

                # Query for similar chunks
                results = store.query(
                    collection=collection_name,
                    embedding=embedding,
                    top_k=5,
                )

                for r in results:
                    other_id = r.document.id
                    if other_id == chunk_a["id"] or other_id in processed_ids:
                        continue

                    # Check if the other chunk is also low-activation
                    other_chunk = next(
                        (c for c in chunks if c["id"] == other_id),
                        None,
                    )
                    if other_chunk is None:
                        continue

                    similarity = r.score
                    if similarity >= REORGANIZATION_SIMILARITY_THRESHOLD:
                        pairs.append((chunk_a, other_chunk, similarity))
                        processed_ids.add(chunk_a["id"])
                        processed_ids.add(other_id)
                        break  # One merge per chunk per cycle

        except Exception as e:
            logger.warning("Failed to find similar pairs: %s", e)

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    async def _merge_chunks_llm(
        self,
        chunk_a: dict,
        chunk_b: dict,
        similarity: float,
        model: str,
    ) -> str | None:
        """Merge two chunks using LLM."""
        prompt = load_prompt(
            "memory/forgetting_merge",
            content_a=chunk_a["content"],
            content_b=chunk_b["content"],
            similarity=f"{similarity:.2f}",
        )

        try:
            from core.memory._llm_utils import one_shot_completion

            result = await one_shot_completion(prompt, model=model, max_tokens=1024)
            if not result:
                return None
            if result.strip().upper().startswith("MERGE_REJECT"):
                logger.info(
                    "LLM rejected neurogenesis merge for %s and %s",
                    chunk_a["id"],
                    chunk_b["id"],
                )
                return None
            return result
        except Exception as e:
            logger.warning("LLM merge failed: %s", e)
            return None

    def _purge_merged_chunks(self, collection_name: str, store: Any | None) -> int:
        """Delete leftover synthetic ``source_file="merged"`` chunks (F10).

        Older neurogenesis merges upserted a synthetic chunk with
        ``source_file="merged"`` in addition to writing the merged text into
        the primary source .md. Once the weekly rebuild re-indexes that .md,
        the synthetic chunk becomes a duplicate. Direct upsert has been removed;
        this purges any duplicates left by previous runs.

        Args:
            collection_name: Vector collection to clean.
            store: Vector store handle (``None`` when RAG is unavailable).

        Returns:
            Number of merged chunks deleted.
        """
        if store is None:
            return 0
        try:
            results = store.get_by_metadata(collection_name, {"source_file": "merged"}, limit=100_000)
            ids = [r.document.id for r in results]
            if not ids:
                return 0
            if store.delete_documents(collection_name, ids):
                logger.info("Purged %d legacy merged chunks from %s", len(ids), collection_name)
                return len(ids)
            logger.warning("Failed to purge merged chunks from %s", collection_name)
        except Exception as e:
            logger.warning("Failed to purge merged chunks from %s: %s", collection_name, e)
        return 0

    def _sync_merged_source_files(
        self,
        chunk_a: dict,
        chunk_b: dict,
        merged_content: str,
    ) -> bool:
        """Update source .md files on disk after a neurogenesis merge.

        Without this step, the next ``_rebuild_rag_index()`` would re-index
        the original (pre-merge) files and the merged chunks would reappear
        as duplicates.

        Steps:
          1. Validate source files have not changed since indexing.
          2. Replace only the target chunk in the primary source file.
          3. Remove only the absorbed chunk from the secondary source file.
          4. Archive/delete the secondary only when all indexed chunks were absorbed.

        Args:
            chunk_a: First chunk dict (kept as primary).
            chunk_b: Second chunk dict (secondary, may be removed).
            merged_content: LLM-merged text to write.

        Returns:
            True when source files were updated or safely skipped, False when
            the merge should be skipped because sources changed or boundaries
            could not be matched.
        """
        plans = self._plan_merged_source_files(chunk_a, chunk_b, merged_content)
        if plans is None:
            return False
        return self._apply_source_write_plans(plans)

    def _plan_merged_source_files(
        self,
        chunk_a: dict,
        chunk_b: dict,
        merged_content: str,
    ) -> list[_SourceWritePlan] | None:
        """Validate and plan source .md mutations for a neurogenesis merge."""
        source_a = (chunk_a.get("metadata") or {}).get("source_file", "")
        source_b = (chunk_b.get("metadata") or {}).get("source_file", "")

        # Guard: skip if source_file values are missing/empty or "merged"
        if not source_a or source_a == "merged":
            return []

        primary_path = self.anima_dir / source_a
        secondary_path = (self.anima_dir / source_b) if source_b and source_b != "merged" else None
        same_source = secondary_path == primary_path

        if self._source_changed_since_index(primary_path, [chunk_a]):
            return None
        if secondary_path and not same_source and self._source_changed_since_index(secondary_path, [chunk_b]):
            return None

        idx_a = self._chunk_index(chunk_a)
        if idx_a is None:
            logger.warning("Skipping neurogenesis source sync for %s: missing chunk_index", source_a)
            return None

        memory_type_a = self._memory_type_for_source(chunk_a, source_a)

        if same_source:
            idx_b = self._chunk_index(chunk_b)
            replacements = {idx_a: merged_content}
            removals: set[int] = set()
            if idx_b is not None and idx_b != idx_a:
                keep_idx = min(idx_a, idx_b)
                drop_idx = max(idx_a, idx_b)
                replacements = {keep_idx: merged_content}
                removals = {drop_idx}

            primary_plan = self._build_source_update(
                primary_path,
                memory_type_a,
                replacements=replacements,
                removals=removals,
            )
            if primary_plan is None:
                return None

            return [
                _SourceWritePlan(
                    path=primary_path,
                    source_file=source_a,
                    content=primary_plan[0],
                    archive=True,
                    delete=False,
                    label="primary",
                )
            ]

        primary_plan = self._build_source_update(
            primary_path,
            memory_type_a,
            replacements={idx_a: merged_content},
            removals=set(),
        )
        if primary_plan is None:
            return None

        secondary_plan: tuple[str, int] | None = None
        if secondary_path and secondary_path.exists():
            idx_b = self._chunk_index(chunk_b)
            if idx_b is None:
                logger.warning("Skipping neurogenesis source sync for %s: missing chunk_index", source_b)
                return None
            secondary_plan = self._build_source_update(
                secondary_path,
                self._memory_type_for_source(chunk_b, source_b),
                replacements={},
                removals={idx_b},
            )
            if secondary_plan is None:
                return None

        plans = [
            _SourceWritePlan(
                path=primary_path,
                source_file=source_a,
                content=primary_plan[0],
                archive=True,
                delete=False,
                label="primary",
            )
        ]
        if secondary_path and secondary_plan is not None:
            remaining_indexed_chunks = secondary_plan[1]
            if remaining_indexed_chunks == 0:
                plans.append(
                    _SourceWritePlan(
                        path=secondary_path,
                        source_file=source_b,
                        content=None,
                        archive=True,
                        delete=True,
                        label="secondary",
                    )
                )
            else:
                plans.append(
                    _SourceWritePlan(
                        path=secondary_path,
                        source_file=source_b,
                        content=secondary_plan[0],
                        archive=False,
                        delete=False,
                        label="secondary",
                    )
                )
        return plans

    def _apply_source_write_plans(self, plans: list[_SourceWritePlan]) -> bool:
        archive_dir = self.anima_dir / "archive" / "merged"
        timestamp = now_local().strftime("%Y%m%d_%H%M%S")
        for plan in plans:
            if plan.archive:
                self._archive_merged_source(plan.path, archive_dir, timestamp, plan.label)
            if plan.delete:
                try:
                    plan.path.unlink()
                    logger.debug("Removed absorbed %s source file: %s", plan.label, plan.source_file)
                except Exception as e:
                    logger.warning("Failed to remove %s source %s: %s", plan.label, plan.source_file, e)
                    return False
                continue
            if plan.content is not None and not self._write_source_text(plan.path, plan.content, plan.source_file):
                return False
        return True

    @staticmethod
    def _chunk_index(chunk: dict) -> int | None:
        raw = (chunk.get("metadata") or {}).get("chunk_index", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _memory_type_for_source(chunk: dict, source_file: str) -> str:
        meta_type = (chunk.get("metadata") or {}).get("memory_type")
        if isinstance(meta_type, str) and meta_type:
            return meta_type
        prefix = source_file.replace("\\", "/").split("/", 1)[0]
        if prefix in {"knowledge", "episodes", "procedures", "skills", "shared_users"}:
            return prefix
        return "knowledge"

    @staticmethod
    def _compute_source_hash(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _source_changed_since_index(self, path: Path, chunks: list[dict]) -> bool:
        """Return True when the on-disk source no longer matches index metadata."""
        if not path.exists():
            return False

        metadata = [c.get("metadata") or {} for c in chunks]
        expected_hash = next(
            (
                str(meta[key])
                for meta in metadata
                for key in ("source_hash", "file_hash", "content_hash")
                if meta.get(key)
            ),
            "",
        )
        if expected_hash and self._compute_source_hash(path) != expected_hash:
            logger.warning(
                "Skipping neurogenesis source sync for %s: source file content changed since indexing",
                path,
            )
            return True

        expected_mtime_ns: int | None = None
        for meta in metadata:
            if meta.get("source_mtime_ns") is None:
                continue
            try:
                expected_mtime_ns = int(meta["source_mtime_ns"])
            except (TypeError, ValueError):
                continue
            break
        if expected_mtime_ns is not None and path.stat().st_mtime_ns != expected_mtime_ns:
            logger.warning(
                "Skipping neurogenesis source sync for %s: source file mtime changed since indexing",
                path,
            )
            return True

        for meta in metadata:
            updated_at = meta.get("updated_at")
            if not updated_at:
                continue
            try:
                expected_dt = ensure_aware(datetime.fromisoformat(str(updated_at)))
                actual_dt = ensure_aware(datetime.fromtimestamp(path.stat().st_mtime))
            except (TypeError, ValueError, OSError):
                continue
            if abs((actual_dt - expected_dt).total_seconds()) > 0.001:
                logger.warning(
                    "Skipping neurogenesis source sync for %s: source file mtime changed since indexing",
                    path,
                )
                return True
        return False

    def _build_source_update(
        self,
        path: Path,
        memory_type: str,
        *,
        replacements: dict[int, str],
        removals: set[int],
    ) -> tuple[str, int] | None:
        """Build updated source text and return remaining indexed chunk count."""
        if not path.exists():
            if replacements and not removals:
                return next(iter(replacements.values())).strip(), 1
            logger.warning("Skipping neurogenesis source sync for %s: source file missing", path)
            return None

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Skipping neurogenesis source sync for %s: %s", path, e)
            return None

        prefix, segments = self._split_source_segments(raw, memory_type)
        indexed = {segment.chunk_index for segment in segments if segment.chunk_index is not None}
        target_indexes = set(replacements) | removals

        if not indexed and target_indexes == {0} and segments:
            segments = [_SourceSegment(0, segment.content) for segment in segments]
            indexed = {0}

        missing = target_indexes - indexed
        if missing:
            logger.warning(
                "Skipping neurogenesis source sync for %s: chunk index(es) not found: %s",
                path,
                sorted(missing),
            )
            return None

        updated_segments: list[_SourceSegment] = []
        remaining_indexed = 0
        for segment in segments:
            if segment.chunk_index in removals:
                continue
            content = replacements.get(segment.chunk_index, segment.content)
            updated_segments.append(_SourceSegment(segment.chunk_index, content))
            if segment.chunk_index is not None:
                remaining_indexed += 1

        return self._render_source_segments(prefix, updated_segments), remaining_indexed

    @staticmethod
    def _split_source_segments(raw: str, memory_type: str) -> tuple[str, list[_SourceSegment]]:
        prefix, body = ForgettingEngine._split_frontmatter_prefix(raw)
        body = body.strip()
        if not body:
            return prefix, []

        if memory_type not in {"knowledge", "common_knowledge", "episodes"}:
            return prefix, [_SourceSegment(0, body)]

        segments: list[_SourceSegment] = []
        sections = re.split(r"\n(##\s+.+)", f"\n{body}")
        preamble = sections[0].strip()
        chunk_idx = 0

        if preamble:
            indexed = chunk_idx if len(preamble) > 50 else None
            segments.append(_SourceSegment(indexed, preamble))
            if indexed is not None:
                chunk_idx += 1

        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                continue
            heading = sections[i].strip()
            section_body = sections[i + 1].strip()
            section_content = f"{heading}\n\n{section_body}".strip()
            if section_content:
                segments.append(_SourceSegment(chunk_idx, section_content))
                chunk_idx += 1

        return prefix, segments

    @staticmethod
    def _split_frontmatter_prefix(raw: str) -> tuple[str, str]:
        if not raw.startswith("---"):
            return "", raw
        first_newline = raw.find("\n")
        if first_newline == -1:
            return "", raw
        rest = raw[first_newline + 1 :]
        match = re.search(r"^---\s*$", rest, flags=re.MULTILINE)
        if match is None:
            return "", raw
        end = first_newline + 1 + match.end()
        body = raw[end:]
        if body.startswith("\n\n"):
            body = body[2:]
        elif body.startswith("\n"):
            body = body[1:]
        return raw[:end].rstrip(), body

    @staticmethod
    def _render_source_segments(prefix: str, segments: list[_SourceSegment]) -> str:
        body = "\n\n".join(segment.content.strip() for segment in segments if segment.content.strip()).strip()
        if prefix and body:
            return f"{prefix}\n\n{body}"
        if prefix:
            return prefix
        return body

    def _archive_merged_source(
        self,
        path: Path,
        archive_dir: Path,
        timestamp: str,
        label: str,
    ) -> Path | None:
        if not path.exists():
            return None
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
        counter = 1
        while dest.exists():
            dest = archive_dir / f"{path.stem}_{timestamp}_{counter}{path.suffix}"
            counter += 1
        try:
            shutil.copy2(str(path), str(dest))
            logger.debug("Archived merged source (%s): %s -> %s", label, path, dest.name)
            return dest
        except Exception as e:
            logger.warning("Failed to archive %s source %s: %s", label, path, e)
            return None

    @staticmethod
    def _write_source_text(path: Path, content: str, source_file: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, content)
            logger.debug("Wrote merged source content to %s", source_file)
            return True
        except Exception as e:
            logger.warning("Failed to write merged content to %s: %s", source_file, e)
            return False

    # ── Stage 3: Complete Forgetting (Monthly) ─────────────────────

    def complete_forgetting(self) -> dict[str, Any]:
        """Archive and delete chunks that remain low-activation (monthly).

        Criteria: low_activation_since > 90 days ago AND access_count <= 2
        Action: Move source file to archive/forgotten/, delete from vector index
        """
        logger.info("Starting complete forgetting for anima=%s", self.anima_name)
        now = now_local()
        store = self._get_vector_store()

        if store is None:
            logger.warning(
                "Skipping complete forgetting for anima=%s: RAG/ChromaDB unavailable",
                self.anima_name,
            )
            return {"forgotten_chunks": 0, "archived_files": [], "skipped_reason": "rag_unavailable"}

        total_forgotten = 0
        total_scanned = 0
        total_marked = 0
        archived_files: list[str] = []

        for memory_type in ("knowledge", "episodes", "procedures"):
            collection_name = f"{self.anima_name}_{memory_type}"
            chunks = self._get_all_chunks(collection_name)
            total_scanned += len(chunks)

            ids_to_delete: list[str] = []
            source_files_to_archive: set[str] = set()

            for chunk in chunks:
                meta = chunk["metadata"]

                # Skip protected
                if self._is_protected(meta):
                    continue

                # Must be low activation
                if meta.get("activation_level") != "low":
                    continue
                total_marked += 1

                # Check duration of low activation
                low_since_str = meta.get("low_activation_since", "")
                if not low_since_str:
                    continue

                try:
                    low_since = ensure_aware(datetime.fromisoformat(str(low_since_str)))
                    days_low = (now - low_since).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    continue

                # Check criteria
                used_count = self._used_count(meta)
                if days_low > FORGETTING_LOW_ACTIVATION_DAYS and used_count <= FORGETTING_MAX_ACCESS_COUNT:
                    ids_to_delete.append(chunk["id"])
                    source_file = meta.get("source_file", "")
                    if source_file and source_file != "merged":
                        source_files_to_archive.add(source_file)

            # Delete from vector index FIRST — if this fails, skip archiving
            # to avoid orphaned state (files archived but chunks still present)
            if ids_to_delete:
                try:
                    store.delete_documents(collection_name, ids_to_delete)
                    total_forgotten += len(ids_to_delete)
                    logger.info(
                        "Deleted %d forgotten chunks from %s",
                        len(ids_to_delete),
                        collection_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to delete chunks from %s: %s",
                        collection_name,
                        e,
                    )
                    continue  # Skip archiving if vector deletion failed

            # Archive source files AFTER successful vector deletion
            for source_file in source_files_to_archive:
                self._archive_source_file(source_file)
                archived_files.append(source_file)

        result = {
            "forgotten_chunks": total_forgotten,
            "archived_files": archived_files,
            "funnel": {
                "scanned": total_scanned,
                "marked": total_marked,
                "merged": 0,
                "forgotten": total_forgotten,
            },
        }
        logger.info(
            "Complete forgetting done for anima=%s: forgotten=%d, archived=%d files",
            self.anima_name,
            total_forgotten,
            len(archived_files),
        )
        logger.info(
            "forgetting_funnel: anima=%s stage=complete scanned=%d marked=%d merged=0 forgotten=%d",
            self.anima_name,
            total_scanned,
            total_marked,
            total_forgotten,
        )
        return result

    def _archive_source_file(self, relative_path: str) -> None:
        """Move source file to archive/forgotten/ directory."""
        source_path = self.anima_dir / relative_path
        if not source_path.exists():
            return

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self.archive_dir / source_path.name

        # Add timestamp suffix if destination exists
        if dest_path.exists():
            timestamp = now_local().strftime("%Y%m%d_%H%M%S")
            dest_path = self.archive_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"

        try:
            shutil.move(str(source_path), str(dest_path))
            logger.info("Archived forgotten file: %s -> %s", relative_path, dest_path.name)
        except Exception as e:
            logger.warning("Failed to archive %s: %s", relative_path, e)

    # ── Episode Retention Archival ─────────────────────────────────

    def archive_expired_episodes(
        self,
        retention_days: int,
        batch_limit: int = 0,
    ) -> dict[str, Any]:
        """Archive episode files older than the configured retention window.

        This is deterministic monthly housekeeping, independent from the
        vector low-activation criteria used by ``complete_forgetting``.
        Files are moved from ``episodes/`` to ``archive/episodes/`` and their
        existing RAG chunks are removed by ``source_file`` when a vector store
        is available.
        """
        episodes_dir = self.anima_dir / "episodes"
        if retention_days < 0:
            retention_days = 0
        if batch_limit < 0:
            batch_limit = 0
        if not episodes_dir.is_dir():
            return {
                "retention_days": retention_days,
                "batch_limit": batch_limit,
                "batch_limited": False,
                "attempted": 0,
                "remaining_count": 0,
                "scanned": 0,
                "archived_count": 0,
                "archived_files": [],
                "archive_destinations": [],
                "deleted_indexed_chunks": 0,
                "skipped_undated": 0,
                "index_delete_failures": 0,
            }

        today = today_local()
        store = self._get_vector_store()
        collection_name = f"{self.anima_name}_episodes"
        archived_files: list[str] = []
        archive_destinations: list[str] = []
        deleted_indexed_chunks = 0
        index_delete_failures = 0
        skipped_undated = 0
        scanned = 0
        expired_files: list[Path] = []

        episode_files = sorted(path for path in episodes_dir.iterdir() if path.suffix in {".md", ".jsonl"})
        for episode_file in episode_files:
            if not episode_file.is_file():
                continue
            scanned += 1
            episode_date = self._episode_file_date(episode_file)
            if episode_date is None:
                try:
                    episode_date = datetime.fromtimestamp(
                        episode_file.stat().st_mtime,
                        tz=now_local().tzinfo,
                    ).date()
                except OSError:
                    skipped_undated += 1
                    continue
            if (today - episode_date).days <= retention_days:
                continue
            expired_files.append(episode_file)

        files_to_archive = expired_files[:batch_limit] if batch_limit else expired_files
        batch_limited = bool(batch_limit and len(expired_files) > batch_limit)
        attempted = len(files_to_archive)

        for episode_file in files_to_archive:
            relative_path = str(episode_file.relative_to(self.anima_dir))
            if store is None:
                deleted = 0
            else:
                deleted = self._delete_indexed_source_file(
                    store,
                    collection_name,
                    relative_path,
                )
            if deleted is None:
                index_delete_failures += 1
            else:
                deleted_indexed_chunks += deleted

            destination = self._archive_episode_file(episode_file)
            if destination is None:
                continue
            archived_files.append(relative_path)
            archive_destinations.append(str(destination.relative_to(self.anima_dir)))

        remaining_count = sum(path.exists() for path in expired_files)

        result = {
            "retention_days": retention_days,
            "batch_limit": batch_limit,
            "batch_limited": batch_limited,
            "attempted": attempted,
            "remaining_count": remaining_count,
            "scanned": scanned,
            "archived_count": len(archived_files),
            "archived_files": archived_files,
            "archive_destinations": archive_destinations,
            "deleted_indexed_chunks": deleted_indexed_chunks,
            "skipped_undated": skipped_undated,
            "index_delete_failures": index_delete_failures,
        }
        if store is None:
            result["index_skipped_reason"] = "rag_unavailable"

        logger.info(
            "Episode retention archival for anima=%s: retention_days=%d batch_limit=%d "
            "scanned=%d attempted=%d archived=%d remaining=%d batch_limited=%s "
            "deleted_indexed_chunks=%d skipped_undated=%d index_delete_failures=%d",
            self.anima_name,
            retention_days,
            batch_limit,
            scanned,
            attempted,
            len(archived_files),
            remaining_count,
            batch_limited,
            deleted_indexed_chunks,
            skipped_undated,
            index_delete_failures,
        )
        return result

    @staticmethod
    def _episode_file_date(path: Path) -> date | None:
        """Return the date encoded in an episode filename, if present."""
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if match is None:
            return None
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None

    def _delete_indexed_source_file(
        self,
        store: Any | None,
        collection_name: str,
        source_file: str,
    ) -> int | None:
        """Delete vector documents for one source file.

        Returns the number of deleted chunks, or ``None`` when the vector
        store was unavailable or deletion failed.
        """
        if store is None:
            return None
        try:
            results = store.get_by_metadata(collection_name, {"source_file": source_file}, limit=10_000)
            ids = [result.document.id for result in results]
            if not ids:
                return 0
            if not store.delete_documents(collection_name, ids):
                logger.warning("Failed to delete indexed chunks for %s/%s", collection_name, source_file)
                return None
            return len(ids)
        except Exception:
            logger.debug(
                "Failed to delete indexed chunks for %s/%s",
                collection_name,
                source_file,
                exc_info=True,
            )
            return None

    def _archive_episode_file(self, source_path: Path) -> Path | None:
        """Move an expired episode file into ``archive/episodes/``."""
        archive_dir = self.anima_dir / "archive" / "episodes"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest_path = archive_dir / source_path.name
        if dest_path.exists():
            timestamp = now_local().strftime("%Y%m%d_%H%M%S")
            dest_path = archive_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"

        try:
            shutil.move(str(source_path), str(dest_path))
            logger.info(
                "Archived retention-expired episode: %s -> %s",
                source_path.relative_to(self.anima_dir),
                dest_path.relative_to(self.anima_dir),
            )
            return dest_path
        except Exception as e:
            logger.warning("Failed to archive retention-expired episode %s: %s", source_path, e)
            return None

    # ── Procedure Archive Cleanup ──────────────────────────────────

    def cleanup_procedure_archives(self) -> dict[str, Any]:
        """Clean up old procedure version archives (monthly).

        Keeps only the ``PROCEDURE_ARCHIVE_KEEP_VERSIONS`` most recent
        versions per procedure stem in ``archive/versions/``.

        Returns:
            Dict with ``deleted_count`` and ``kept_count`` keys.
        """
        archive_dir = self.anima_dir / "archive" / "versions"
        if not archive_dir.exists():
            return {"deleted_count": 0, "kept_count": 0}

        import re

        # Group archived files by procedure stem.
        # Naming convention from reconsolidation: {stem}_v{N}_{timestamp}.md
        stem_files: dict[str, list[Path]] = {}
        pattern = re.compile(r"^(.+?)_v\d+_\d{8}_\d{6}\.md$")

        for path in archive_dir.iterdir():
            if not path.is_file():
                continue
            m = pattern.match(path.name)
            if m:
                stem = m.group(1)
                stem_files.setdefault(stem, []).append(path)

        deleted_count = 0
        kept_count = 0

        for _, files in stem_files.items():
            # Sort by modification time descending (newest first)
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            keep = files[:PROCEDURE_ARCHIVE_KEEP_VERSIONS]
            delete = files[PROCEDURE_ARCHIVE_KEEP_VERSIONS:]

            kept_count += len(keep)
            for path in delete:
                try:
                    path.unlink()
                    deleted_count += 1
                    logger.debug("Deleted old procedure archive: %s", path.name)
                except Exception as e:
                    logger.warning("Failed to delete archive %s: %s", path.name, e)

        if deleted_count > 0:
            logger.info(
                "Procedure archive cleanup for anima=%s: deleted=%d, kept=%d",
                self.anima_name,
                deleted_count,
                kept_count,
            )

        return {"deleted_count": deleted_count, "kept_count": kept_count}
