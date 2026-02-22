from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Knowledge contradiction detection and resolution engine.

Detects contradictions between knowledge files using a two-stage pipeline:
1. Vector similarity pre-filter: Find candidate pairs via RAG
2. NLI + LLM cascade: Classify contradictions and propose resolutions

Resolution strategies:
- supersede: Newer knowledge replaces older (outdated info)
- merge: Combine complementary but conflicting knowledge
- coexist: Mark as context-dependent coexisting truths
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.paths import load_prompt
from core.time_utils import now_iso

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger

logger = logging.getLogger("animaworks.contradiction")


# ── Data structures ─────────────────────────────────────────────


@dataclass
class ContradictionResult:
    """Result of a contradiction check between two texts."""

    is_contradiction: bool
    resolution: str  # "supersede" | "merge" | "coexist"
    reason: str
    merged_content: str | None = None  # Populated only for "merge" resolution


@dataclass
class ContradictionPair:
    """A detected contradiction between two knowledge files."""

    file_a: Path
    file_b: Path
    text_a: str
    text_b: str
    confidence: float  # Contradiction confidence (0.0-1.0)
    resolution: str  # "supersede" | "merge" | "coexist"
    reason: str
    merged_content: str | None = None  # Pre-generated merge text if available


# ── ContradictionDetector ───────────────────────────────────────


class ContradictionDetector:
    """Knowledge contradiction detection and resolution engine.

    Uses a two-stage pipeline:
    1. RAG vector similarity to find candidate pairs
    2. NLI model + LLM cascade to classify and resolve contradictions

    The NLI model is shared with ``KnowledgeValidator`` to avoid
    duplicate model loading.
    """

    NLI_CONTRADICTION_THRESHOLD = 0.65
    VECTOR_SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        anima_dir: Path,
        anima_name: str,
        activity_logger: ActivityLogger | None = None,
    ) -> None:
        """Initialize contradiction detector.

        Args:
            anima_dir: Path to anima's directory (~/.animaworks/animas/{name})
            anima_name: Name of the anima for logging and RAG collection lookup
            activity_logger: Optional ActivityLogger for recording resolution events
        """
        self.anima_dir = anima_dir
        self.anima_name = anima_name
        self.knowledge_dir = anima_dir / "knowledge"
        self._nli_validator: object | None = None  # KnowledgeValidator instance
        self._activity_logger = activity_logger

    # ── NLI validator access ────────────────────────────────────

    def _get_nli_validator(self):
        """Lazily load the shared KnowledgeValidator for NLI checks.

        Returns:
            KnowledgeValidator instance, or None if unavailable
        """
        if self._nli_validator is None:
            try:
                from core.memory.validation import KnowledgeValidator

                self._nli_validator = KnowledgeValidator()
            except ImportError:
                logger.warning("KnowledgeValidator not available for NLI checks")
                return None
        return self._nli_validator

    # ── Candidate pair generation ───────────────────────────────

    def _find_candidate_pairs(
        self, target_file: Path | None = None,
    ) -> list[tuple[Path, str, Path, str]]:
        """Find candidate contradiction pairs using RAG vector similarity.

        When ``target_file`` is specified, only pairs involving that file
        are returned. Otherwise, all knowledge files are compared pairwise.

        Args:
            target_file: Optional file to check against all others

        Returns:
            List of (file_a, text_a, file_b, text_b) tuples where
            vector similarity exceeds ``VECTOR_SIMILARITY_THRESHOLD``
        """
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        if target_file is not None:
            files = [target_file] if target_file.exists() else []
        else:
            files = sorted(self.knowledge_dir.glob("*.md"))

        if len(files) == 0:
            return []

        # Read all knowledge files' content (stripping frontmatter)
        file_contents: dict[Path, str] = {}
        for f in files:
            content = mm.read_knowledge_content(f)
            if content.strip():
                file_contents[f] = content

        if target_file is not None:
            # Compare target against all other knowledge files
            other_files = [
                f for f in sorted(self.knowledge_dir.glob("*.md"))
                if f != target_file and f not in file_contents
            ]
            for f in other_files:
                content = mm.read_knowledge_content(f)
                if content.strip():
                    file_contents[f] = content

        # Try RAG-based similarity search for efficient candidate selection
        candidates = self._find_candidates_via_rag(file_contents, target_file)

        # Fallback: if RAG is unavailable, do exhaustive pairwise comparison
        if candidates is None:
            candidates = self._find_candidates_exhaustive(file_contents, target_file)

        return candidates

    def _find_candidates_via_rag(
        self,
        file_contents: dict[Path, str],
        target_file: Path | None,
    ) -> list[tuple[Path, str, Path, str]] | None:
        """Use RAG retriever to find similar knowledge file pairs.

        Returns:
            Candidate pairs, or None if RAG is unavailable
        """
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.retriever import MemoryRetriever
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store(self.anima_name)
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)
            retriever = MemoryRetriever(
                vector_store, indexer, self.knowledge_dir,
            )
        except (ImportError, Exception) as e:
            logger.debug("RAG not available for candidate search: %s", e)
            return None

        candidates: list[tuple[Path, str, Path, str]] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Determine which files to use as queries
        query_files = (
            [target_file] if target_file is not None
            else list(file_contents.keys())
        )

        for query_path in query_files:
            query_text = file_contents.get(query_path, "")
            if not query_text:
                continue

            results = retriever.search(
                query=query_text[:500],
                anima_name=self.anima_name,
                memory_type="knowledge",
                top_k=10,
            )

            for result in results:
                if result.score < self.VECTOR_SIMILARITY_THRESHOLD:
                    continue

                source_file = result.metadata.get("source_file", "")
                if not source_file:
                    continue

                # Resolve to a Path
                match_path = self.knowledge_dir / Path(source_file).name
                if not match_path.exists() or match_path == query_path:
                    continue

                # Deduplicate pairs (order-independent)
                pair_key = tuple(sorted([str(query_path), str(match_path)]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                match_text = file_contents.get(match_path, "")
                if not match_text:
                    from core.memory.manager import MemoryManager

                    mm = MemoryManager(self.anima_dir)
                    match_text = mm.read_knowledge_content(match_path)
                    if not match_text.strip():
                        continue

                candidates.append(
                    (query_path, query_text, match_path, match_text)
                )

        return candidates

    def _find_candidates_exhaustive(
        self,
        file_contents: dict[Path, str],
        target_file: Path | None,
    ) -> list[tuple[Path, str, Path, str]]:
        """Exhaustive pairwise candidate generation (RAG fallback).

        Args:
            file_contents: Mapping from file path to content text
            target_file: If set, only pairs involving this file are returned

        Returns:
            All valid pairs of knowledge files
        """
        all_files = sorted(file_contents.keys())
        candidates: list[tuple[Path, str, Path, str]] = []

        for i, file_a in enumerate(all_files):
            for file_b in all_files[i + 1:]:
                if target_file is not None:
                    if file_a != target_file and file_b != target_file:
                        continue
                candidates.append(
                    (file_a, file_contents[file_a], file_b, file_contents[file_b])
                )

        return candidates

    # ── Contradiction detection ─────────────────────────────────

    NLI_ENTAILMENT_THRESHOLD = 0.70

    async def _check_contradiction_nli(
        self, text_a: str, text_b: str,
    ) -> tuple[bool, float, bool]:
        """Check for contradiction using NLI model.

        Uses the shared KnowledgeValidator's NLI pipeline to classify
        the relationship between two texts.

        Args:
            text_a: First knowledge text
            text_b: Second knowledge text

        Returns:
            Tuple of (is_contradiction, confidence_score, is_entailment).
            ``is_entailment`` is True when either direction shows
            entailment above ``NLI_ENTAILMENT_THRESHOLD``, indicating
            the texts are consistent and no LLM check is needed.
        """
        validator = self._get_nli_validator()
        if validator is None:
            return (False, 0.0, False)

        # NLI check: text_a as premise, text_b as hypothesis
        label_ab, score_ab = validator._nli_check(
            text_b[:2000], text_a[:2000],
        )
        # Also check the reverse direction
        label_ba, score_ba = validator._nli_check(
            text_a[:2000], text_b[:2000],
        )

        # If either direction shows contradiction above threshold, flag it
        is_contradiction = False
        max_score = 0.0

        if label_ab == "contradiction" and score_ab >= self.NLI_CONTRADICTION_THRESHOLD:
            is_contradiction = True
            max_score = max(max_score, score_ab)
        if label_ba == "contradiction" and score_ba >= self.NLI_CONTRADICTION_THRESHOLD:
            is_contradiction = True
            max_score = max(max_score, score_ba)

        # Check if either direction shows clear entailment
        is_entailment = False
        if not is_contradiction:
            if (
                (label_ab == "entailment" and score_ab >= self.NLI_ENTAILMENT_THRESHOLD)
                or (label_ba == "entailment" and score_ba >= self.NLI_ENTAILMENT_THRESHOLD)
            ):
                is_entailment = True

        return (is_contradiction, max_score, is_entailment)

    async def _check_contradiction_llm(
        self,
        text_a: str,
        text_b: str,
        file_a: str,
        file_b: str,
        model: str,
    ) -> ContradictionResult:
        """Check for contradiction and propose resolution using LLM.

        Args:
            text_a: Content of the first knowledge file
            text_b: Content of the second knowledge file
            file_a: Filename of the first knowledge file
            file_b: Filename of the second knowledge file
            model: LLM model identifier (LiteLLM format)

        Returns:
            ContradictionResult with detection and resolution info
        """
        import litellm

        prompt = load_prompt(
            "memory/contradiction_detection",
            file_a=file_a,
            text_a=text_a[:3000],
            file_b=file_b,
            text_b=text_b[:3000],
        )

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return ContradictionResult(
                    is_contradiction=bool(result.get("is_contradiction", False)),
                    resolution=result.get("resolution", "coexist"),
                    reason=result.get("reason", ""),
                    merged_content=result.get("merged_content"),
                )
        except Exception as e:
            logger.warning(
                "LLM contradiction check failed for %s vs %s: %s",
                file_a, file_b, e,
            )

        # On failure, conservatively assume no contradiction
        return ContradictionResult(
            is_contradiction=False,
            resolution="coexist",
            reason="LLM check failed, conservatively assuming no contradiction",
        )

    # ── Main scan API ───────────────────────────────────────────

    async def scan_contradictions(
        self,
        target_file: Path | None = None,
        model: str = "",
    ) -> list[ContradictionPair]:
        """Scan knowledge files for contradictions.

        Uses a two-stage pipeline:
        1. RAG vector similarity to find candidate pairs
        2. NLI + LLM cascade to classify contradictions

        Args:
            target_file: If specified, only check this file against others.
                If None, scan the entire knowledge directory.
            model: LLM model for contradiction analysis

        Returns:
            List of detected contradiction pairs with resolution proposals
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model
        logger.info(
            "Starting contradiction scan for anima=%s target=%s",
            self.anima_name,
            target_file.name if target_file else "all",
        )

        # Step 1: Find candidate pairs via vector similarity
        candidates = self._find_candidate_pairs(target_file)
        if not candidates:
            logger.info("No candidate pairs found for contradiction check")
            return []

        logger.info(
            "Found %d candidate pairs for contradiction check", len(candidates),
        )

        # Step 2: Check each candidate pair
        contradictions: list[ContradictionPair] = []

        for file_a, text_a, file_b, text_b in candidates:
            # Stage 2a: NLI check (fast, local)
            is_nli_contradiction, nli_score, is_entailment = (
                await self._check_contradiction_nli(text_a, text_b)
            )

            if is_nli_contradiction:
                # NLI detected contradiction - use LLM for resolution
                llm_result = await self._check_contradiction_llm(
                    text_a, text_b, file_a.name, file_b.name, model,
                )

                if llm_result.is_contradiction:
                    contradictions.append(ContradictionPair(
                        file_a=file_a,
                        file_b=file_b,
                        text_a=text_a,
                        text_b=text_b,
                        confidence=nli_score,
                        resolution=llm_result.resolution,
                        reason=llm_result.reason,
                        merged_content=llm_result.merged_content,
                    ))
                    logger.info(
                        "Contradiction detected: %s vs %s (confidence=%.2f, "
                        "resolution=%s)",
                        file_a.name, file_b.name, nli_score,
                        llm_result.resolution,
                    )
            elif is_entailment:
                # NLI shows clear entailment — texts are consistent,
                # skip costly LLM check
                logger.debug(
                    "NLI entailment detected, skipping LLM: %s vs %s",
                    file_a.name, file_b.name,
                )
            else:
                # NLI neutral / uncertain — fall through to LLM as fallback
                # for cases where NLI may miss semantic contradictions
                llm_result = await self._check_contradiction_llm(
                    text_a, text_b, file_a.name, file_b.name, model,
                )

                if llm_result.is_contradiction:
                    contradictions.append(ContradictionPair(
                        file_a=file_a,
                        file_b=file_b,
                        text_a=text_a,
                        text_b=text_b,
                        confidence=0.5,  # Lower confidence without NLI backing
                        resolution=llm_result.resolution,
                        reason=llm_result.reason,
                        merged_content=llm_result.merged_content,
                    ))
                    logger.info(
                        "Contradiction detected (LLM-only): %s vs %s "
                        "(resolution=%s)",
                        file_a.name, file_b.name, llm_result.resolution,
                    )

        logger.info(
            "Contradiction scan complete for anima=%s: %d contradictions found",
            self.anima_name, len(contradictions),
        )

        return contradictions

    # ── Resolution execution ────────────────────────────────────

    async def resolve_contradictions(
        self,
        pairs: list[ContradictionPair],
        model: str = "",
    ) -> dict[str, int]:
        """Resolve detected contradictions by applying the proposed strategy.

        Processes each contradiction pair according to its resolution type:
        - supersede: Archive the older file, keep the newer one
        - merge: Combine both files into a unified knowledge file
        - coexist: Annotate both files with cross-references

        Args:
            pairs: List of contradiction pairs to resolve
            model: LLM model for merge text generation

        Returns:
            Summary dict with counts: superseded, merged, coexisted, errors
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model
        results = {"superseded": 0, "merged": 0, "coexisted": 0, "errors": 0}

        for pair in pairs:
            try:
                strategy = pair.resolution

                # Auto-increment failure_count on the older file (file_b)
                # BEFORE applying the resolution, since supersede/merge may
                # archive file_b making it inaccessible at its original path.
                if strategy in ("supersede", "merge"):
                    self._increment_failure_count(pair.file_b)

                if strategy == "supersede":
                    self._apply_supersede(pair)
                    results["superseded"] += 1
                elif strategy == "merge":
                    merged = await self._apply_merge(pair, model)
                    if merged:
                        results["merged"] += 1
                    else:
                        results["errors"] += 1
                        continue
                elif strategy == "coexist":
                    self._apply_coexist(pair)
                    results["coexisted"] += 1
                else:
                    logger.warning(
                        "Unknown resolution type '%s' for %s vs %s",
                        strategy, pair.file_a.name, pair.file_b.name,
                    )
                    results["errors"] += 1
                    continue

                # Persist contradiction resolution to shared JSONL history
                self._persist_contradiction_history(pair, strategy)

                # Record activity log event for successful resolutions
                self._log_resolution(pair, strategy)

            except Exception:
                logger.exception(
                    "Failed to resolve contradiction: %s vs %s",
                    pair.file_a.name, pair.file_b.name,
                )
                results["errors"] += 1

        logger.info(
            "Contradiction resolution complete for anima=%s: %s",
            self.anima_name, results,
        )

        return results

    def _persist_contradiction_history(
        self, pair: ContradictionPair, strategy: str,
    ) -> None:
        """Append a contradiction resolution entry to shared JSONL history.

        Writes to ``{shared_dir}/contradiction_history.jsonl`` in append mode.
        The shared directory is derived from ``anima_dir``'s data root.

        Args:
            pair: The resolved contradiction pair
            strategy: Resolution strategy applied (supersede/merge/coexist)
        """
        from core.paths import get_shared_dir

        history_path = get_shared_dir() / "contradiction_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "ts": now_iso(),
            "anima": self.anima_name,
            "file_a": pair.file_a.name,
            "file_b": pair.file_b.name,
            "confidence": pair.confidence,
            "resolution": strategy,
            "reason": pair.reason,
            "merged_content": pair.merged_content,
            "meta": {},
        }

        try:
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning(
                "Failed to persist contradiction history for %s vs %s",
                pair.file_a.name, pair.file_b.name,
            )

    def _increment_failure_count(self, file_path: Path) -> None:
        """Increment failure_count and recalculate confidence for a knowledge file.

        Reads the YAML frontmatter, increments ``failure_count``, recalculates
        ``confidence`` as ``success_count / (success_count + failure_count)``,
        and writes the updated frontmatter back.

        Only operates if the file still exists (it may have been archived).

        Args:
            file_path: Path to the knowledge file to update
        """
        if not file_path.exists():
            logger.debug(
                "Skipping failure_count increment — file already archived: %s",
                file_path.name,
            )
            return

        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        try:
            meta = mm.read_knowledge_metadata(file_path)
            meta["failure_count"] = meta.get("failure_count", 0) + 1
            success = meta.get("success_count", 0)
            total = success + meta["failure_count"]
            if total > 0:
                meta["confidence"] = round(success / total, 4)
            content = mm.read_knowledge_content(file_path)
            mm.write_knowledge_with_meta(file_path, content, meta)
            logger.debug(
                "Incremented failure_count for %s: failure_count=%d confidence=%.4f",
                file_path.name, meta["failure_count"], meta.get("confidence", 0),
            )
        except Exception:
            logger.warning(
                "Failed to increment failure_count for %s",
                file_path.name,
            )

    def _log_resolution(
        self, pair: ContradictionPair, strategy: str,
    ) -> None:
        """Record a knowledge_contradiction_resolved event to the activity log.

        Args:
            pair: The resolved contradiction pair
            strategy: Resolution strategy applied (supersede/merge/coexist)
        """
        if self._activity_logger is None:
            return
        try:
            self._activity_logger.log(
                event_type="knowledge_contradiction_resolved",
                content=(
                    f"矛盾解決: {pair.file_a.name} vs {pair.file_b.name}"
                    f" → 戦略: {strategy}"
                ),
                summary=f"knowledge矛盾解決({strategy})",
                meta={
                    "strategy": strategy,
                    "newer": pair.file_b.name,
                    "older": pair.file_a.name,
                },
            )
        except Exception:
            logger.warning(
                "Failed to log contradiction resolution event for %s vs %s",
                pair.file_a.name, pair.file_b.name,
            )

    def _apply_supersede(self, pair: ContradictionPair) -> None:
        """Resolve by superseding the older file with the newer one.

        Determines which file is older based on ``updated_at`` metadata,
        then archives the older file and annotates the newer one.

        Args:
            pair: Contradiction pair to resolve
        """
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        # Determine which file is newer
        meta_a = mm.read_knowledge_metadata(pair.file_a)
        meta_b = mm.read_knowledge_metadata(pair.file_b)

        ts_a = meta_a.get("updated_at") or meta_a.get("created_at", "")
        ts_b = meta_b.get("updated_at") or meta_b.get("created_at", "")

        # Default: treat file_b as newer (file_a gets superseded)
        if ts_a > ts_b:
            newer, older = pair.file_a, pair.file_b
            newer_meta = meta_a
            older_meta = meta_b
        else:
            newer, older = pair.file_b, pair.file_a
            newer_meta = meta_b
            older_meta = meta_a

        now = now_iso()

        # Update older file metadata before archiving
        older_meta["superseded_by"] = newer.name
        older_meta["valid_until"] = now
        older_content = mm.read_knowledge_content(older)
        mm.write_knowledge_with_meta(older, older_content, older_meta)

        # Move older file to archive
        archive_dir = self.anima_dir / "archive" / "superseded"
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(older), str(archive_dir / older.name))

        # Update newer file metadata with supersedes reference
        newer_meta = mm.read_knowledge_metadata(newer)
        supersedes_list = newer_meta.get("supersedes", [])
        if isinstance(supersedes_list, str):
            supersedes_list = [supersedes_list]
        supersedes_list.append(older.name)
        newer_meta["supersedes"] = supersedes_list
        newer_content = mm.read_knowledge_content(newer)
        mm.write_knowledge_with_meta(newer, newer_content, newer_meta)

        logger.info(
            "Superseded %s by %s (archived to %s)",
            older.name, newer.name, archive_dir,
        )

    async def _apply_merge(
        self, pair: ContradictionPair, model: str,
    ) -> bool:
        """Resolve by merging both files into a unified knowledge file.

        If ``pair.merged_content`` is already available (pre-generated
        during scan), it is used directly. Otherwise, the LLM generates
        a merged text.

        Args:
            pair: Contradiction pair to merge
            model: LLM model for merge text generation

        Returns:
            True if merge succeeded, False otherwise
        """
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        merged_content = pair.merged_content

        # Generate merged content via LLM if not pre-generated
        if not merged_content:
            merged_content = await self._generate_merged_content(
                pair.text_a, pair.text_b,
                pair.file_a.name, pair.file_b.name,
                model,
            )

        if not merged_content:
            logger.warning(
                "Failed to generate merged content for %s + %s",
                pair.file_a.name, pair.file_b.name,
            )
            return False

        # Derive merged filename from the topic
        topic = self._derive_merge_topic(pair.file_a.stem, pair.file_b.stem)
        merged_filename = f"_merged_{topic}.md"
        merged_path = self.knowledge_dir / merged_filename

        # Write merged file with metadata
        now = now_iso()
        metadata = {
            "created_at": now,
            "merged_from": [pair.file_a.name, pair.file_b.name],
            "merged_at": now,
            "confidence": 0.7,
            "auto_consolidated": True,
        }
        mm.write_knowledge_with_meta(merged_path, merged_content, metadata)

        # Archive original files
        archive_dir = self.anima_dir / "archive" / "merged"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for original in (pair.file_a, pair.file_b):
            if original.exists():
                shutil.move(str(original), str(archive_dir / original.name))

        logger.info(
            "Merged %s + %s into %s",
            pair.file_a.name, pair.file_b.name, merged_filename,
        )

        return True

    async def _generate_merged_content(
        self,
        text_a: str,
        text_b: str,
        file_a: str,
        file_b: str,
        model: str,
    ) -> str | None:
        """Generate merged content from two contradicting knowledge files.

        Args:
            text_a: Content of first file
            text_b: Content of second file
            file_a: Name of first file
            file_b: Name of second file
            model: LLM model identifier

        Returns:
            Merged content string, or None on failure
        """
        import litellm

        prompt = load_prompt(
            "memory/contradiction_merge",
            file_a=file_a,
            text_a=text_a[:3000],
            file_b=file_b,
            text_b=text_b[:3000],
        )

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            result = response.choices[0].message.content or ""

            # Strip code fences
            result = re.sub(r"^```(?:markdown|md)?\s*\n", "", result, flags=re.MULTILINE)
            result = re.sub(r"\n```\s*$", "", result, flags=re.MULTILINE)

            return result.strip() if result.strip() else None

        except Exception as e:
            logger.warning("Failed to generate merged content: %s", e)
            return None

    @staticmethod
    def _derive_merge_topic(stem_a: str, stem_b: str) -> str:
        """Derive a topic name from two knowledge file stems.

        Finds common words between the two stems, falling back to the
        first stem if no overlap is found.

        Args:
            stem_a: Stem of the first file
            stem_b: Stem of the second file

        Returns:
            A sanitized topic name suitable for a filename
        """
        # Split on common delimiters
        words_a = set(re.split(r"[-_]", stem_a.lower()))
        words_b = set(re.split(r"[-_]", stem_b.lower()))

        common = words_a & words_b
        # Remove short or empty tokens
        common = {w for w in common if len(w) > 1}

        if common:
            return "-".join(sorted(common))
        # Fallback: use the shorter stem
        return stem_a if len(stem_a) <= len(stem_b) else stem_b

    def _apply_coexist(self, pair: ContradictionPair) -> None:
        """Resolve by marking both files as coexisting.

        Adds ``coexists_with`` metadata to both files without moving them.

        Args:
            pair: Contradiction pair to mark as coexisting
        """
        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        # Update file_a metadata
        meta_a = mm.read_knowledge_metadata(pair.file_a)
        existing_coexists_a = meta_a.get("coexists_with", [])
        if isinstance(existing_coexists_a, str):
            existing_coexists_a = [existing_coexists_a]
        if pair.file_b.name not in existing_coexists_a:
            existing_coexists_a.append(pair.file_b.name)
        meta_a["coexists_with"] = existing_coexists_a
        content_a = mm.read_knowledge_content(pair.file_a)
        mm.write_knowledge_with_meta(pair.file_a, content_a, meta_a)

        # Update file_b metadata
        meta_b = mm.read_knowledge_metadata(pair.file_b)
        existing_coexists_b = meta_b.get("coexists_with", [])
        if isinstance(existing_coexists_b, str):
            existing_coexists_b = [existing_coexists_b]
        if pair.file_a.name not in existing_coexists_b:
            existing_coexists_b.append(pair.file_a.name)
        meta_b["coexists_with"] = existing_coexists_b
        content_b = mm.read_knowledge_content(pair.file_b)
        mm.write_knowledge_with_meta(pair.file_b, content_b, meta_b)

        logger.info(
            "Marked as coexisting: %s <-> %s",
            pair.file_a.name, pair.file_b.name,
        )
