from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Procedural memory auto-distillation engine.

Classifies episodic memories into knowledge / procedures / skip categories
using an LLM, and distills procedural episodes into reusable procedure files
with YAML frontmatter.

Pipeline:
  - Daily: LLM classifies episode sections -> writes knowledge & procedures
  - Weekly: activity_log-based pattern detection -> distill repeated patterns
"""

import json
import logging
import re
from pathlib import Path

from core.paths import load_prompt
from core.time_utils import now_iso, now_jst

logger = logging.getLogger("animaworks.distillation")

RAG_DUPLICATE_THRESHOLD = 0.85


# ── ProceduralDistiller ────────────────────────────────────────


class ProceduralDistiller:
    """Engine that distills procedural knowledge from episodic memories.

    Uses LLM-based classification to detect procedural content, then
    extracts structured, reusable procedure documents.  Saved procedures
    include YAML frontmatter with tracking metadata (confidence,
    success/failure counts, etc.).
    """

    def __init__(self, anima_dir: Path, anima_name: str) -> None:
        """Initialize the distiller.

        Args:
            anima_dir: Path to the anima's data directory.
            anima_name: Name of the anima (for logging).
        """
        self.anima_dir = anima_dir
        self.anima_name = anima_name
        self.procedures_dir = anima_dir / "procedures"
        self.knowledge_dir = anima_dir / "knowledge"
        self.episodes_dir = anima_dir / "episodes"
        self.procedures_dir.mkdir(parents=True, exist_ok=True)

    # ── LLM-based Classification & Distillation ──────────────

    async def classify_and_distill(
        self,
        episodes_text: str,
        model: str = "",
    ) -> dict:
        """Classify episodes and extract both knowledge and procedures via LLM.

        Sends all episodes to an LLM which classifies content into
        knowledge/procedures/skip categories and returns structured output
        for both.

        Args:
            episodes_text: Concatenated episode text (Markdown).
            model: LiteLLM model identifier.

        Returns:
            Dict with:
              - ``knowledge_items``: list of dicts with ``filename``, ``content``
              - ``procedure_items``: list of dicts with ``filename``,
                ``description``, ``tags``, ``content``
              - ``raw_response``: raw LLM output string
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model

        result = {
            "knowledge_items": [],
            "procedure_items": [],
            "raw_response": "",
        }

        if not episodes_text.strip():
            return result

        existing = self._load_existing_procedures()

        prompt = load_prompt(
            "memory/classification",
            episodes_text=episodes_text[:6000],
            existing_procedures=existing[:2000],
        )

        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3072,
            )
            text = response.choices[0].message.content or ""
            text = self._strip_code_fence(text)
            result["raw_response"] = text

            # Parse LLM output
            knowledge_items = self._parse_knowledge_items(text)
            procedure_items = self._parse_procedure_items(text)

            result["knowledge_items"] = knowledge_items
            result["procedure_items"] = procedure_items

            logger.info(
                "LLM classification for anima=%s: knowledge=%d procedures=%d",
                self.anima_name, len(knowledge_items), len(procedure_items),
            )

        except Exception:
            logger.exception(
                "LLM classification failed for anima=%s", self.anima_name,
            )

        return result

    # ── Daily Distillation (legacy-compatible entry point) ─────

    async def distill_procedures(
        self,
        procedural_episodes: str,
        model: str = "",
    ) -> list[dict]:
        """Extract reusable procedures from episode text via LLM classification.

        This is the main daily distillation entry point.  It sends episodes
        to the LLM for classification and returns extracted procedure items.

        Args:
            procedural_episodes: Concatenated episode text.
            model: LiteLLM model identifier.

        Returns:
            List of dicts, each with keys ``title``, ``description``,
            ``tags``, and ``content``.
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model
        if not procedural_episodes.strip():
            return []

        classification = await self.classify_and_distill(
            procedural_episodes, model=model,
        )

        # Convert procedure_items to the legacy format expected by callers
        procedures: list[dict] = []
        for item in classification["procedure_items"]:
            filename = item.get("filename", "")
            # Extract title from filename (strip procedures/ prefix and .md)
            title = filename.replace("procedures/", "").replace(".md", "")
            if not title:
                continue
            procedures.append({
                "title": title,
                "description": item.get("description", ""),
                "tags": item.get("tags", []),
                "content": item.get("content", ""),
            })

        logger.info(
            "Distilled %d procedures from episodes for anima=%s",
            len(procedures), self.anima_name,
        )
        return procedures

    def get_knowledge_items(self, classification_result: dict) -> list[dict]:
        """Extract knowledge items from a classification result.

        Used by consolidation to merge LLM-classified knowledge into the
        existing knowledge consolidation pipeline.

        Args:
            classification_result: Return value from ``classify_and_distill()``.

        Returns:
            List of knowledge item dicts with ``filename`` and ``content``.
        """
        return classification_result.get("knowledge_items", [])

    # ── Weekly Pattern Distillation ────────────────────────────

    async def weekly_pattern_distill(
        self,
        model: str = "",
        days: int = 7,
    ) -> dict:
        """Detect repeated action patterns from activity_log and distill.

        Reads activity log entries for the past *days* days, clusters
        similar activities, and uses an LLM to distill repeated patterns
        into procedure files.

        Args:
            model: LiteLLM model identifier.
            days: Look-back window in days.

        Returns:
            Dict with ``procedures_created`` (list of file paths) and
            ``patterns_detected`` (int).
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model

        # 1. Load activity entries
        entries = self._load_activity_entries(days=days)
        if not entries:
            logger.info(
                "No recent activity entries for weekly pattern distill, anima=%s",
                self.anima_name,
            )
            return {"procedures_created": [], "patterns_detected": 0}

        # 2. Filter for relevant event types
        relevant = [
            e for e in entries
            if e.get("type") in (
                "tool_use", "response_sent", "cron_executed",
                "memory_write", "issue_resolved",
            )
        ]
        if not relevant:
            logger.info(
                "No relevant activity entries for weekly pattern distill, anima=%s",
                self.anima_name,
            )
            return {"procedures_created": [], "patterns_detected": 0}

        # 3. Cluster similar activities
        clusters = self._cluster_activities(relevant, min_cluster_size=3)
        if not clusters:
            logger.info(
                "No repeated patterns detected for weekly distill, anima=%s",
                self.anima_name,
            )
            return {"procedures_created": [], "patterns_detected": 0}

        # 4. Use LLM to distill clusters into procedures
        existing = self._load_existing_procedures()
        clusters_text = self._format_clusters_for_prompt(clusters)

        prompt = load_prompt(
            "memory/weekly_pattern",
            clusters_text=clusters_text[:6000],
            existing_procedures=existing[:2000],
        )

        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            text = response.choices[0].message.content or "[]"
            procedures = self._parse_procedures(text)

            saved_paths: list[str] = []
            for item in procedures:
                path = self.save_procedure(item)
                if path is None:
                    continue
                saved_paths.append(str(path))
                logger.info(
                    "Weekly pattern distill: saved procedure '%s' for anima=%s",
                    path.name, self.anima_name,
                )

            return {
                "procedures_created": saved_paths,
                "patterns_detected": len(clusters),
            }

        except Exception:
            logger.exception(
                "Weekly pattern distill failed for anima=%s", self.anima_name,
            )
            return {"procedures_created": [], "patterns_detected": 0}

    # ── Activity Log Helpers ──────────────────────────────────

    def _load_activity_entries(self, days: int = 7) -> list[dict]:
        """Load activity log entries for the last *days* days.

        Reads JSONL files from ``{anima_dir}/activity_log/{date}.jsonl``.

        Args:
            days: Number of days to look back.

        Returns:
            List of raw entry dicts from the JSONL files.
        """
        from datetime import timedelta

        activity_dir = self.anima_dir / "activity_log"
        if not activity_dir.is_dir():
            return []

        entries: list[dict] = []
        today = now_jst().date()

        for offset in range(days):
            target_date = today - timedelta(days=offset)
            path = activity_dir / f"{target_date.isoformat()}.jsonl"
            if not path.exists():
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except OSError:
                logger.warning("Failed to read activity log: %s", path)

        return entries

    def _cluster_activities(
        self,
        entries: list[dict],
        min_cluster_size: int = 3,
    ) -> list[list[dict]]:
        """Cluster similar activities by content similarity.

        Uses a simple text-based similarity approach: groups entries that
        share significant content overlap (tool name, action summary).
        Falls back to basic grouping when vector search is unavailable.

        Args:
            entries: List of activity entry dicts.
            min_cluster_size: Minimum entries per cluster.

        Returns:
            List of clusters, each a list of entry dicts.
        """
        # Try vector-based clustering first
        try:
            return self._cluster_activities_vector(entries, min_cluster_size)
        except Exception:
            logger.debug(
                "Vector clustering unavailable, falling back to text-based",
            )

        # Fallback: group by (type, tool) pairs and content similarity
        groups: dict[str, list[dict]] = {}
        for entry in entries:
            # Build a grouping key from type + tool
            key_parts = [entry.get("type", "")]
            if entry.get("tool"):
                key_parts.append(entry["tool"])
            key = "|".join(key_parts)
            groups.setdefault(key, []).append(entry)

        # Filter to clusters with enough entries
        return [
            cluster for cluster in groups.values()
            if len(cluster) >= min_cluster_size
        ]

    def _cluster_activities_vector(
        self,
        entries: list[dict],
        min_cluster_size: int = 3,
        min_similarity: float = 0.80,
    ) -> list[list[dict]]:
        """Cluster activities using vector embeddings.

        Uses the RAG embedding model to encode activity summaries, then
        groups entries with cosine similarity >= min_similarity.

        Args:
            entries: Activity entry dicts.
            min_cluster_size: Minimum cluster size.
            min_similarity: Cosine similarity threshold.

        Returns:
            List of clusters (lists of entry dicts).

        Raises:
            ImportError: When RAG dependencies are unavailable.
            Exception: On embedding or clustering failures.
        """
        from core.memory.rag.singleton import get_embedding_model

        model = get_embedding_model()

        # Build text representations
        texts: list[str] = []
        for entry in entries:
            parts = [entry.get("type", "")]
            if entry.get("tool"):
                parts.append(entry["tool"])
            summary = entry.get("summary") or entry.get("content", "")
            if summary:
                parts.append(summary[:200])
            texts.append(" ".join(parts))

        # Generate embeddings
        import numpy as np

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        normed = embeddings / norms

        # Simple greedy clustering
        n = len(entries)
        assigned: list[bool] = [False] * n
        clusters: list[list[dict]] = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster_indices = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                sim = float(np.dot(normed[i], normed[j]))
                if sim >= min_similarity:
                    cluster_indices.append(j)
                    assigned[j] = True

            if len(cluster_indices) >= min_cluster_size:
                clusters.append([entries[idx] for idx in cluster_indices])

        return clusters

    @staticmethod
    def _format_clusters_for_prompt(clusters: list[list[dict]]) -> str:
        """Format activity clusters for the weekly pattern prompt.

        Args:
            clusters: List of entry clusters.

        Returns:
            Formatted text for LLM prompt injection.
        """
        parts: list[str] = []
        for i, cluster in enumerate(clusters, 1):
            lines = [f"### パターン {i} ({len(cluster)}回繰り返し)"]
            for entry in cluster[:10]:  # Limit entries per cluster
                ts = entry.get("ts", "")[:16]
                etype = entry.get("type", "")
                tool = entry.get("tool", "")
                summary = entry.get("summary") or entry.get("content", "")
                summary = summary[:150]
                tool_info = f" [tool: {tool}]" if tool else ""
                lines.append(f"- {ts} {etype}{tool_info}: {summary}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ── Parsing Helpers ────────────────────────────────────────

    def _parse_knowledge_items(self, text: str) -> list[dict]:
        """Parse knowledge items from LLM classification output.

        Looks for the ``## knowledge抽出`` section and extracts items
        with ``ファイル名:`` and ``内容:`` fields.

        Args:
            text: Raw LLM output (sanitized).

        Returns:
            List of dicts with ``filename`` and ``content``.
        """
        section = re.search(
            r"##\s*knowledge抽出(.+?)(?=##\s*procedure抽出|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not section:
            return []

        section_text = section.group(1)
        if "(なし)" in section_text:
            return []

        items: list[dict] = []
        for match in re.finditer(
            r"-\s*ファイル名:\s*(.+?)\s+内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
            section_text,
            re.DOTALL,
        ):
            filename = match.group(1).strip()
            content = match.group(2).strip()
            if filename and content:
                items.append({"filename": filename, "content": content})

        return items

    def _parse_procedure_items(self, text: str) -> list[dict]:
        """Parse procedure items from LLM classification output.

        Looks for the ``## procedure抽出`` section and extracts items
        with ``ファイル名:``, ``description:``, ``tags:``, ``内容:`` fields.

        Args:
            text: Raw LLM output (sanitized).

        Returns:
            List of dicts with ``filename``, ``description``, ``tags``,
            and ``content``.
        """
        section = re.search(
            r"##\s*procedure抽出(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not section:
            return []

        section_text = section.group(1)
        if "(なし)" in section_text:
            return []

        items: list[dict] = []
        for match in re.finditer(
            r"-\s*ファイル名:\s*(.+?)\s+"
            r"description:\s*(.+?)\s+"
            r"tags:\s*(.+?)\s+"
            r"内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
            section_text,
            re.DOTALL,
        ):
            filename = match.group(1).strip()
            description = match.group(2).strip()
            tags_str = match.group(3).strip()
            content = match.group(4).strip()

            # Parse tags
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            if filename and content:
                items.append({
                    "filename": filename,
                    "description": description,
                    "tags": tags,
                    "content": content,
                })

        return items

    def _parse_procedures(self, text: str) -> list[dict]:
        """Parse an LLM JSON response into a list of procedure dicts.

        Strips code fences before parsing.  Each valid item must have
        at least ``title`` and ``content`` keys.

        Args:
            text: Raw LLM output (potentially fenced JSON).

        Returns:
            List of procedure dicts with ``title``, ``content``, and
            optionally ``description`` and ``tags``.
        """
        text = self._strip_code_fence(text)

        try:
            items = json.loads(text)
            if isinstance(items, list):
                return [
                    i for i in items
                    if isinstance(i, dict)
                    and "title" in i
                    and "content" in i
                ]
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM procedure output for anima=%s",
                self.anima_name,
            )

        return []

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove Markdown code fences wrapping the entire text.

        Handles any language tag (``json``, ``markdown``, etc.) and
        preserves interior content.

        Args:
            text: Raw text potentially wrapped in code fences.

        Returns:
            Text with outer code fences removed.
        """
        text = re.sub(r"^```\w*\s*\n", "", text.strip(), count=1)
        text = re.sub(r"\n```\s*$", "", text)
        return text.strip()

    # ── Procedure I/O ──────────────────────────────────────────

    def _load_existing_procedures(self) -> str:
        """Build a summary of existing procedures for duplicate avoidance.

        Returns:
            Newline-separated list of ``- filename: description`` entries,
            or ``(なし)`` when the directory is empty.
        """
        summaries: list[str] = []
        for f in sorted(self.procedures_dir.glob("*.md")):
            meta = self._read_metadata(f)
            desc = meta.get("description", f.stem)
            summaries.append(f"- {f.stem}: {desc}")
        return "\n".join(summaries) if summaries else "(なし)"

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        """Read YAML frontmatter metadata from a procedure file.

        Args:
            path: Absolute path to the procedure Markdown file.

        Returns:
            Parsed metadata dict, or empty dict on failure.
        """
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                import yaml

                try:
                    return yaml.safe_load(parts[1]) or {}
                except Exception:
                    return {}
        return {}

    def _check_rag_duplicate(
        self, content: str, threshold: float = RAG_DUPLICATE_THRESHOLD,
    ) -> str | None:
        """Check if a similar procedure or skill already exists via RAG.

        Searches both the ``procedures`` and ``skills`` collections.
        Returns the source file path of the first match above *threshold*,
        or ``None`` if no duplicate is found.  Failures are logged and
        treated as "no duplicate" so that saving can proceed.

        Args:
            content: Procedure body text to compare.
            threshold: Minimum similarity score to consider a duplicate.

        Returns:
            Path string of the similar existing document, or None.
        """
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.retriever import MemoryRetriever
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store(self.anima_name)
            indexer = MemoryIndexer(
                vector_store, self.anima_name, self.anima_dir,
            )
            retriever = MemoryRetriever(
                vector_store, indexer, self.knowledge_dir,
            )

            for memory_type in ("procedures", "skills"):
                results = retriever.search(
                    query=content[:500],
                    anima_name=self.anima_name,
                    memory_type=memory_type,
                    top_k=3,
                )
                for r in results:
                    if r.score >= threshold:
                        return r.metadata.get("source_file", "unknown")
        except Exception as e:
            logger.warning(
                "RAG duplicate check failed (proceeding with save): %s", e,
            )
        return None

    def save_procedure(self, item: dict) -> Path | None:
        """Persist a distilled procedure with YAML frontmatter.

        Before saving, performs a RAG similarity check against existing
        procedures and skills.  If a duplicate is found (similarity
        >= ``RAG_DUPLICATE_THRESHOLD``), the save is skipped and ``None``
        is returned.

        Uses ``MemoryManager.write_procedure_with_meta()`` for consistent
        file format across the codebase.

        Args:
            item: Dict with ``title``, ``content``, and optionally
                ``description`` and ``tags``.

        Returns:
            Path to the saved procedure file, or None if skipped as
            duplicate.
        """
        content = item["content"]

        # RAG duplicate check
        existing = self._check_rag_duplicate(content)
        if existing:
            logger.info(
                "Skipping duplicate procedure '%s' (similar to %s)",
                item["title"], existing,
            )
            return None

        from core.memory.manager import MemoryManager

        title = re.sub(r"[^\w\-]", "_", item["title"])
        path = self.procedures_dir / f"{title}.md"

        metadata = {
            "description": item.get("description", ""),
            "tags": item.get("tags", []),
            "success_count": 0,
            "failure_count": 0,
            "last_used": None,
            "confidence": 0.4,
            "version": 1,
            "created_at": now_iso(),
            "auto_distilled": True,
        }

        mm = MemoryManager(self.anima_dir)
        mm.write_procedure_with_meta(path, item["content"], metadata)

        logger.info("Saved distilled procedure: %s", path.name)
        return path

    # ── Section Splitting (utility) ────────────────────────────

    @staticmethod
    def _split_into_sections(text: str) -> list[str]:
        """Split *text* on ``## `` Markdown headers.

        Args:
            text: Raw Markdown text.

        Returns:
            List of non-empty section strings.
        """
        sections = re.split(r"\n(?=##\s)", text)
        return [s.strip() for s in sections if s.strip()]
