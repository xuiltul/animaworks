from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Memory consolidation engine.

Implements daily and weekly consolidation processes that convert episodic
memories into semantic knowledge, analogous to sleep-based memory consolidation
in the human brain.
"""

import json
import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.consolidation")


# ── Path sanitization ─────────────────────────────────────────


def _sanitize_filepath(base_dir: Path, filename: str) -> Path:
    """Sanitize LLM-generated filename to prevent path traversal.

    Resolves the constructed path and verifies it stays within *base_dir*.
    If a traversal is detected, the filename is cleaned by replacing
    non-alphanumeric characters (except ``-`` and ``.``) with ``_``,
    and a ``.md`` suffix is ensured.

    Args:
        base_dir: Target directory that the file must reside in.
        filename: LLM-generated filename (may contain ``../``).

    Returns:
        Safe absolute path guaranteed to be inside *base_dir*.
    """
    filepath = (base_dir / filename).resolve()
    if not filepath.is_relative_to(base_dir.resolve()):
        logger.warning("Path traversal detected in LLM filename: %s", filename)
        safe_name = re.sub(r"[^\w\-.]", "_", Path(filename).name)
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        filepath = base_dir / safe_name
    return filepath


# ── ConsolidationEngine ────────────────────────────────────────


class ConsolidationEngine:
    """Handles automatic memory consolidation processes.

    This class implements:
    - Daily consolidation: Episodes → Knowledge (NREM sleep analog)
    - Weekly integration: Knowledge merging and episode compression
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
                created_at = datetime.now().isoformat()
                ts_match = re.search(
                    r"\[AUTO-CONSOLIDATED:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]",
                    text,
                )
                if ts_match:
                    try:
                        parsed = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M")
                        created_at = parsed.isoformat()
                    except ValueError:
                        pass

                # Strip code fences that LLM may have wrapped around content
                content = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.MULTILINE)
                content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)
                content = content.strip()

                metadata = {
                    "created_at": created_at,
                    "confidence": 0.5,
                    "auto_consolidated": True,
                    "migrated_from_legacy": True,
                }

                mm.write_knowledge_with_meta(path, content, metadata)
                migrated += 1
                logger.info("Migrated legacy knowledge file: %s", path.name)

            except Exception:
                logger.exception("Failed to migrate knowledge file: %s", path.name)
                continue

        # Write marker
        marker.write_text(
            datetime.now().isoformat() + "\n",
            encoding="utf-8",
        )
        if migrated > 0:
            logger.info(
                "Legacy knowledge migration complete for anima=%s: migrated=%d",
                self.anima_name, migrated,
            )
        return migrated

    # ── Daily Consolidation ────────────────────────────────────

    async def daily_consolidate(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        min_episodes: int = 1,
    ) -> dict[str, Any]:
        """Perform daily consolidation: Episodes → Knowledge.

        Collects episodes from the past 24 hours, extracts lessons and patterns
        via LLM, and writes them to knowledge/ directory.

        Args:
            model: LLM model to use for consolidation
            min_episodes: Minimum number of episodes required to run consolidation

        Returns:
            Dictionary with consolidation results:
            - episodes_processed: Number of episode entries processed
            - knowledge_files_created: List of new knowledge files
            - knowledge_files_updated: List of updated knowledge files
            - skipped: True if consolidation was skipped
        """
        logger.info("Starting daily consolidation for anima=%s", self.anima_name)

        # Run legacy migration (once)
        try:
            self._migrate_legacy_knowledge()
        except Exception:
            logger.exception("Legacy migration failed for anima=%s", self.anima_name)

        # Collect recent episodes
        episode_entries = self._collect_recent_episodes(hours=24)

        # Collect resolved events for consolidation prompt injection
        resolved_events = self._collect_resolved_events(hours=24)

        if len(episode_entries) < min_episodes:
            logger.info(
                "Skipping daily consolidation for anima=%s: "
                "only %d episodes found (min=%d)",
                self.anima_name, len(episode_entries), min_episodes
            )
            return {
                "episodes_processed": 0,
                "knowledge_files_created": [],
                "knowledge_files_updated": [],
                "skipped": True,
            }

        logger.info(
            "Consolidating %d episode entries for anima=%s",
            len(episode_entries), self.anima_name
        )

        # Get existing knowledge files for context
        existing_knowledge = self._list_knowledge_files()

        # Format episodes text for validation context
        episodes_text = "\n\n".join([
            f"## {e['date']} {e['time']}\n{e['content']}"
            for e in episode_entries
        ])

        # ── Procedural distillation: LLM-based classification ──
        distillation_result: dict[str, Any] = {
            "procedures_created": [],
        }
        try:
            from core.memory.distillation import ProceduralDistiller

            distiller = ProceduralDistiller(self.anima_dir, self.anima_name)
            classification = await distiller.classify_and_distill(
                episodes_text, model=model,
            )

            # Save extracted procedures
            procedures = await distiller.distill_procedures(
                episodes_text, model=model,
            ) if not classification["procedure_items"] else []

            # Use procedure_items from classification if available
            saved_paths: list[str] = []
            for item in classification["procedure_items"]:
                filename = item.get("filename", "")
                title = filename.replace("procedures/", "").replace(".md", "")
                if not title or not item.get("content"):
                    continue
                proc_item = {
                    "title": title,
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),
                    "content": item["content"],
                }
                path = distiller.save_procedure(proc_item)
                if path is not None:
                    saved_paths.append(str(path))

            distillation_result["procedures_created"] = saved_paths
            if saved_paths:
                logger.info(
                    "Daily distillation: created %d procedures for anima=%s",
                    len(saved_paths), self.anima_name,
                )

        except Exception:
            logger.exception(
                "Procedural distillation failed for anima=%s, "
                "falling back to full consolidation",
                self.anima_name,
            )

        # Generate consolidation via LLM (all episodes)
        consolidation_result = await self._summarize_episodes(
            episode_entries=episode_entries,
            existing_knowledge_files=existing_knowledge,
            model=model,
            resolved_events=resolved_events,
        )

        # Sanitize and validate, then write
        consolidation_result = self._sanitize_llm_output(consolidation_result)

        # Run validation pipeline on parsed knowledge items
        validated_result = await self._validate_consolidation(
            consolidation_result, episodes_text, model,
        )

        # Parse and write results to knowledge/
        files_created, files_updated = self._merge_to_knowledge(validated_result)

        # Update RAG index for affected files
        self._update_rag_index(files_created + files_updated)

        # Synaptic downscaling (forgetting phase 1)
        downscaling_result: dict[str, Any] = {}
        try:
            from core.memory.forgetting import ForgettingEngine
            forgetter = ForgettingEngine(self.anima_dir, self.anima_name)
            downscaling_result = forgetter.synaptic_downscaling()
            logger.info(
                "Synaptic downscaling: scanned=%d, marked=%d",
                downscaling_result.get("scanned", 0),
                downscaling_result.get("marked_low", 0),
            )
        except Exception:
            logger.exception("Synaptic downscaling failed for anima=%s", self.anima_name)

        # Reconsolidation: detect and apply prediction errors
        reconsolidation_result: dict[str, Any] = {}
        try:
            reconsolidation_result = await self._run_reconsolidation(
                episodes_text, model,
            )
        except Exception:
            logger.exception("Reconsolidation failed for anima=%s", self.anima_name)

        # Contradiction check for newly created/updated knowledge files
        contradiction_result: dict[str, int] = {}
        try:
            contradiction_result = await self._run_contradiction_check(
                files_created + files_updated, model,
            )
        except Exception:
            logger.exception(
                "Contradiction check failed for anima=%s", self.anima_name,
            )

        logger.info(
            "Daily consolidation complete for anima=%s: "
            "created=%d updated=%d",
            self.anima_name, len(files_created), len(files_updated)
        )

        return {
            "episodes_processed": len(episode_entries),
            "knowledge_files_created": files_created,
            "knowledge_files_updated": files_updated,
            "distillation": distillation_result,
            "downscaling": downscaling_result,
            "reconsolidation": reconsolidation_result,
            "contradiction": contradiction_result,
            "skipped": False,
        }

    @staticmethod
    def _filter_entries_by_text(
        entries: list[dict[str, str]],
        filtered_text: str,
    ) -> list[dict[str, str]]:
        """Keep only episode entries whose content appears in *filtered_text*.

        Used after procedural classification to pass only the semantic
        portion of episodes to the knowledge consolidation LLM.

        Args:
            entries: Original episode entries.
            filtered_text: Concatenated text of the desired category.

        Returns:
            Subset of *entries* whose content appears in *filtered_text*.
        """
        result: list[dict[str, str]] = []
        for entry in entries:
            # Use a prefix check (first 100 chars) to match sections
            prefix = entry["content"][:100].strip()
            if prefix and prefix in filtered_text:
                result.append(entry)
        return result if result else entries

    # ── Reconsolidation ─────────────────────────────────────────

    async def _run_reconsolidation(
        self,
        episodes_text: str,
        model: str,
    ) -> dict[str, Any]:
        """Run failure-count-based reconsolidation on procedures.

        Scans procedure frontmatter for files with failure_count >= 2
        and confidence < 0.6, then uses an LLM to revise them.

        Args:
            episodes_text: Concatenated episode text (unused, kept for API compat).
            model: LLM model identifier for revision.

        Returns:
            Dict with reconsolidation results (targets_found, updated, etc.).
        """
        from core.memory.reconsolidation import ReconsolidationEngine

        engine = ReconsolidationEngine(self.anima_dir, self.anima_name)
        targets = await engine.find_reconsolidation_targets()

        if not targets:
            logger.info(
                "No reconsolidation targets found for anima=%s",
                self.anima_name,
            )
            return {"targets_found": 0, "updated": 0, "skipped": 0}

        logger.info(
            "Found %d reconsolidation targets for anima=%s, applying",
            len(targets), self.anima_name,
        )

        result = await engine.apply_reconsolidation(targets, model)
        result["targets_found"] = len(targets)

        # Re-index updated procedure files
        updated_files: list[str] = [
            t.name for t in targets if t.exists()
        ]
        if updated_files:
            self._update_rag_index(updated_files)

        return result

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
        cutoff = datetime.now() - timedelta(hours=hours)
        entries: list[dict[str, str]] = []

        # Check today and yesterday's episode files
        for day_offset in range(2):
            target_date = datetime.now().date() - timedelta(days=day_offset)
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
                            entry_dt = datetime.strptime(
                                f"{target_date} {time_str}",
                                "%Y-%m-%d %H:%M",
                            )

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
                    file_mtime = datetime.fromtimestamp(
                        episode_file.stat().st_mtime,
                    )
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
            return [{"ts": e.ts, "content": e.content, "summary": e.summary} for e in entries]
        except Exception:
            logger.debug("Failed to collect resolved events", exc_info=True)
            return []

    # ── Sanitization ────────────────────────────────────────────

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

    def _fetch_related_knowledge(
        self,
        episodes_text: str,
        top_k: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """Fetch related existing knowledge via RAG for consolidation context.

        Searches the knowledge vector store for content related to the
        current episode text, returning the top matches as context for the
        consolidation LLM prompt.

        Args:
            episodes_text: Concatenated episode text to use as query
            top_k: Maximum number of knowledge chunks to retrieve
            max_tokens: Approximate token budget (chars / 3 heuristic)

        Returns:
            Formatted string of related knowledge, or empty string
        """
        try:
            from core.memory.rag.retriever import MemoryRetriever
            from core.memory.rag.singleton import get_vector_store

            from core.memory.rag import MemoryIndexer

            vector_store = self._rag_store or get_vector_store()
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)
            retriever = MemoryRetriever(
                vector_store, indexer, self.knowledge_dir,
            )

            results = retriever.search(
                query=episodes_text[:500],
                anima_name=self.anima_name,
                memory_type="knowledge",
                top_k=top_k,
            )

            if not results:
                return ""

            parts: list[str] = []
            total_chars = 0
            char_budget = max_tokens * 3  # rough chars-to-tokens heuristic

            for r in results:
                source = r.metadata.get("source_file", r.doc_id)
                snippet = r.content[:1000]
                entry = f"### {source}\n{snippet}"
                if total_chars + len(entry) > char_budget:
                    break
                parts.append(entry)
                total_chars += len(entry)

            return "\n\n".join(parts)

        except ImportError:
            logger.debug("RAG not available, skipping related knowledge fetch")
            return ""
        except Exception as e:
            logger.warning("Failed to fetch related knowledge: %s", e)
            return ""

    async def _summarize_episodes(
        self,
        episode_entries: list[dict[str, str]],
        existing_knowledge_files: list[str],
        model: str,
        resolved_events: list[dict] | None = None,
    ) -> str:
        """Use LLM to extract lessons from episodes.

        Args:
            episode_entries: List of episode entries with date, time, content
            existing_knowledge_files: List of existing knowledge file names
            model: LLM model to use
            resolved_events: Optional list of resolved issue events

        Returns:
            LLM response with structured consolidation results
        """
        # Format episodes for prompt
        episodes_text = "\n\n".join([
            f"## {e['date']} {e['time']}\n{e['content']}"
            for e in episode_entries
        ])

        # Format knowledge file list
        knowledge_list = "\n".join([f"- {f}" for f in existing_knowledge_files])
        if not knowledge_list:
            knowledge_list = "(まだ知識ファイルはありません)"

        # Fetch related knowledge via RAG for context
        related_knowledge = self._fetch_related_knowledge(episodes_text)

        # Build consolidation prompt
        prompt = f"""以下は{self.anima_name}の過去24時間のエピソード記録です。

【エピソード】
{episodes_text}

【既存の知識ファイル一覧】
{knowledge_list}
"""

        # Inject related knowledge content
        if related_knowledge:
            prompt += f"""
【関連する既存知識の内容】
{related_knowledge}
"""

        prompt += """
タスク:
以下のエピソードから新しい教訓・パターン・方針を抽出してください。

1. **新しい教訓やパターン**: エピソードから学んだことで、今後の判断に役立つもの
2. **既存知識の更新**: 既存の知識ファイルに追加・修正すべき内容
3. **新規知識ファイル**: 新しいトピックとして独立した知識ファイルが必要な場合

出力形式:
## 既存ファイル更新
- ファイル名: knowledge/xxx.md
  追加内容: (具体的な内容をMarkdown形式で記述)

## 新規ファイル作成
- ファイル名: knowledge/yyy.md
  内容: (ファイル全体の内容をMarkdown形式で記述)

注意事項:
- 以下の情報は必ず抽出してください:
  - 具体的な設定値・APIキー・認証情報の格納場所
  - ユーザーやシステムの識別情報（ID、名前、役割）
  - 手順・ワークフロー・プロセスの記録
  - チーム構成・役割分担・指揮系統
  - 技術的な判断とその理由
- 完全に同一内容の繰り返しのみスキップしてください
- 挨拶のみの会話や実質的な情報を含まないやり取りは知識化不要です
- 既存ファイルがない場合は、すべて新規ファイルとして提案してください
- ファイル名はトピックを表すわかりやすい名前にしてください（英数字とハイフン推奨）
- コードフェンス（```）で囲まないでください
"""

        # Inject resolved events into prompt
        if resolved_events:
            resolved_text = "\n".join(
                f"- {r.get('ts', '')[:16]}: {r.get('content', '')}" for r in resolved_events
            )
            prompt += f"""
【解決済み案件】
以下の案件は解決済みです。既存の知識ファイルに「未解決」「対応中」「調査中」等の
記載がある場合は、「解決済み」に更新してください。

{resolved_text}
"""

        # Call LLM
        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )

            result = response.choices[0].message.content or ""

            # Sanitize LLM output (strip code fences)
            result = self._sanitize_llm_output(result)

            # Format validation: check if parseable
            has_sections = bool(
                re.search(r"##\s*(既存ファイル更新|新規ファイル作成)", result)
            )

            if result and not has_sections:
                # Retry with explicit format instruction
                logger.info(
                    "Consolidation output missing expected sections, retrying "
                    "for anima=%s",
                    self.anima_name,
                )
                retry_prompt = (
                    "先ほどの出力が期待された形式と異なりました。\n"
                    "以下の形式で再出力してください:\n\n"
                    "## 既存ファイル更新\n"
                    "- ファイル名: knowledge/xxx.md\n"
                    "  追加内容: (内容)\n\n"
                    "## 新規ファイル作成\n"
                    "- ファイル名: knowledge/yyy.md\n"
                    "  内容: (内容)\n\n"
                    "コードフェンス（```）は使わないでください。\n\n"
                    f"元の内容:\n{result[:2000]}"
                )
                retry_response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": retry_prompt}],
                    max_tokens=2048,
                )
                retry_result = retry_response.choices[0].message.content or ""
                retry_result = self._sanitize_llm_output(retry_result)
                if re.search(r"##\s*(既存ファイル更新|新規ファイル作成)", retry_result):
                    result = retry_result
                    logger.info("Format retry succeeded for anima=%s", self.anima_name)
                else:
                    logger.warning(
                        "Format retry also failed for anima=%s, using original",
                        self.anima_name,
                    )

            logger.info(
                "Consolidation LLM response for %s (%d chars): %.500s",
                self.anima_name, len(result), result,
            )
            return result

        except Exception as e:
            logger.error(
                "Failed to call LLM for consolidation: %s",
                e,
                exc_info=True
            )
            return ""

    async def _validate_consolidation(
        self,
        consolidation_result: str,
        episodes_text: str,
        model: str,
    ) -> str:
        """Run NLI+LLM validation on parsed knowledge items.

        Extracts individual knowledge items from the consolidation LLM
        output, validates each against the source episodes, and
        reconstructs the output with only validated items.

        Args:
            consolidation_result: Raw (sanitized) consolidation LLM output
            episodes_text: Concatenated source episode text
            model: LLM model for validation fallback

        Returns:
            Filtered consolidation result with rejected items removed
        """
        if not consolidation_result.strip():
            return consolidation_result

        try:
            from core.memory.validation import KnowledgeValidator

            validator = KnowledgeValidator()

            # Extract knowledge items from the consolidation output
            items: list[dict] = []

            # Parse new file items
            create_section = re.search(
                r"##\s*新規ファイル作成(.+)",
                consolidation_result,
                re.DOTALL | re.IGNORECASE,
            )
            if create_section:
                for match in re.finditer(
                    r"-\s*ファイル名:\s*(.+?)\s+内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
                    create_section.group(1),
                    re.DOTALL,
                ):
                    items.append({
                        "filename": match.group(1).strip(),
                        "content": match.group(2).strip(),
                        "type": "create",
                    })

            # Parse update items
            update_section = re.search(
                r"##\s*既存ファイル更新(.+?)(?=##\s*新規ファイル作成|\Z)",
                consolidation_result,
                re.DOTALL | re.IGNORECASE,
            )
            if update_section:
                for match in re.finditer(
                    r"-\s*ファイル名:\s*(.+?)\s+追加内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
                    update_section.group(1),
                    re.DOTALL,
                ):
                    items.append({
                        "filename": match.group(1).strip(),
                        "content": match.group(2).strip(),
                        "type": "update",
                    })

            if not items:
                return consolidation_result

            # Validate all items
            validated = await validator.validate(items, episodes_text, model)
            validated_set = {id(item) for item in validated}

            # Reconstruct output with only validated items
            update_items = [
                item for item in validated if item["type"] == "update"
            ]
            create_items = [
                item for item in validated if item["type"] == "create"
            ]

            parts: list[str] = []
            parts.append("## 既存ファイル更新")
            if update_items:
                for item in update_items:
                    parts.append(
                        f"- ファイル名: {item['filename']}\n"
                        f"  追加内容: {item['content']}"
                    )
            else:
                parts.append("(なし)")

            parts.append("\n## 新規ファイル作成")
            if create_items:
                for item in create_items:
                    parts.append(
                        f"- ファイル名: {item['filename']}\n"
                        f"  内容: {item['content']}"
                    )
            else:
                parts.append("(なし)")

            result = "\n".join(parts)
            rejected_count = len(items) - len(validated)
            if rejected_count > 0:
                logger.info(
                    "Validation rejected %d/%d knowledge items for anima=%s",
                    rejected_count, len(items), self.anima_name,
                )
            return result

        except ImportError:
            logger.debug("Validation module not available, skipping validation")
            return consolidation_result
        except Exception:
            logger.exception(
                "Validation failed for anima=%s, using unvalidated result",
                self.anima_name,
            )
            return consolidation_result

    def _merge_to_knowledge(self, consolidation_result: str) -> tuple[list[str], list[str]]:
        """Parse consolidation result and write to knowledge files.

        Args:
            consolidation_result: LLM output with structured updates

        Returns:
            Tuple of (files_created, files_updated)
        """
        files_created: list[str] = []
        files_updated: list[str] = []

        if not consolidation_result.strip():
            logger.warning(
                "Empty consolidation LLM response for %s",
                self.anima_name,
            )
            return files_created, files_updated

        from core.memory.manager import MemoryManager

        mm = MemoryManager(self.anima_dir)

        # Parse updates to existing files
        update_section = re.search(
            r"##\s*既存ファイル更新(.+?)(?=##\s*新規ファイル作成|\Z)",
            consolidation_result,
            re.DOTALL | re.IGNORECASE
        )

        if update_section:
            for match in re.finditer(
                r"-\s*ファイル名:\s*(.+?)\s+追加内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
                update_section.group(1),
                re.DOTALL
            ):
                filename = match.group(1).strip()
                content = match.group(2).strip()

                # Normalize filename
                filename = filename.replace("knowledge/", "")
                if not filename.endswith(".md"):
                    filename += ".md"

                filepath = _sanitize_filepath(self.knowledge_dir, filename)
                filename = filepath.name

                if filepath.exists():
                    # Read existing metadata and content, then append
                    existing_meta = mm.read_knowledge_metadata(filepath)
                    existing_content = mm.read_knowledge_content(filepath)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    updated_content = (
                        f"{existing_content}\n\n"
                        f"[AUTO-CONSOLIDATED: {timestamp}]\n\n{content}"
                    )

                    # Update metadata
                    existing_meta["updated_at"] = datetime.now().isoformat()
                    mm.write_knowledge_with_meta(filepath, updated_content, existing_meta)
                    files_updated.append(filename)
                    logger.info("Updated knowledge file: %s", filename)
                else:
                    # File doesn't exist, create with frontmatter
                    timestamp = datetime.now().isoformat()
                    metadata = {
                        "created_at": timestamp,
                        "confidence": 0.7,
                        "auto_consolidated": True,
                    }
                    body = f"# {filepath.stem}\n\n{content}"
                    mm.write_knowledge_with_meta(filepath, body, metadata)
                    files_created.append(filename)
                    logger.info("Created knowledge file: %s", filename)

        # Parse new file creations
        create_section = re.search(
            r"##\s*新規ファイル作成(.+)",
            consolidation_result,
            re.DOTALL | re.IGNORECASE
        )

        if create_section:
            for match in re.finditer(
                r"-\s*ファイル名:\s*(.+?)\s+内容:\s*(.+?)(?=-\s*ファイル名:|\Z)",
                create_section.group(1),
                re.DOTALL
            ):
                filename = match.group(1).strip()
                content = match.group(2).strip()

                # Normalize filename
                filename = filename.replace("knowledge/", "")
                if not filename.endswith(".md"):
                    filename += ".md"

                filepath = _sanitize_filepath(self.knowledge_dir, filename)
                filename = filepath.name

                if not filepath.exists():
                    # Build source episode filenames
                    today = datetime.now().date()
                    source_episodes = [
                        f"{today}.md",
                        f"{(today - timedelta(days=1))}.md",
                    ]

                    timestamp = datetime.now().isoformat()
                    metadata = {
                        "created_at": timestamp,
                        "source_episodes": source_episodes,
                        "confidence": 0.7,
                        "auto_consolidated": True,
                    }
                    mm.write_knowledge_with_meta(filepath, content, metadata)
                    files_created.append(filename)
                    logger.info("Created knowledge file: %s", filename)
                else:
                    logger.warning(
                        "Skipping knowledge file creation (already exists): %s",
                        filename
                    )

        if not files_created and not files_updated:
            logger.warning(
                "No knowledge files extracted from consolidation for %s "
                "(response length=%d chars). "
                "LLM may have judged episodes as non-extractable.",
                self.anima_name, len(consolidation_result),
            )

        return files_created, files_updated

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

            vector_store = self._rag_store or get_vector_store()
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

    # ── Weekly Integration ─────────────────────────────────────

    async def weekly_integrate(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        duplicate_threshold: float = 0.85,
        episode_retention_days: int = 30,
    ) -> dict[str, Any]:
        """Perform weekly integration: Knowledge merging and episode compression.

        Args:
            model: LLM model to use for integration
            duplicate_threshold: Similarity threshold for duplicate detection (0-1)
            episode_retention_days: Days to retain uncompressed episodes

        Returns:
            Dictionary with integration results:
            - knowledge_files_merged: List of merged file pairs
            - episodes_compressed: Number of episodes compressed
            - skipped: True if integration was skipped
        """
        logger.info("Starting weekly integration for anima=%s", self.anima_name)

        results = {
            "knowledge_files_merged": [],
            "episodes_compressed": 0,
            "skipped": False,
        }

        # Step 1: Detect and merge duplicate knowledge files
        try:
            duplicates = await self._detect_duplicates(threshold=duplicate_threshold)
            if duplicates:
                logger.info(
                    "Found %d duplicate knowledge file pairs for anima=%s",
                    len(duplicates), self.anima_name
                )
                merged_files = await self._merge_knowledge_files(duplicates, model)
                results["knowledge_files_merged"] = merged_files
            else:
                logger.info("No duplicate knowledge files found for anima=%s", self.anima_name)

        except Exception:
            logger.exception("Failed to merge knowledge files for anima=%s", self.anima_name)

        # Step 2: Compress old episodes
        try:
            compressed_count = await self._compress_old_episodes(
                retention_days=episode_retention_days,
                model=model,
            )
            results["episodes_compressed"] = compressed_count
            logger.info(
                "Compressed %d old episodes for anima=%s",
                compressed_count, self.anima_name
            )

        except Exception:
            logger.exception("Failed to compress old episodes for anima=%s", self.anima_name)

        # Step 3: Rebuild RAG index for affected files
        try:
            self._rebuild_rag_index()
        except Exception:
            logger.exception("Failed to rebuild RAG index for anima=%s", self.anima_name)

        # Neurogenesis reorganization (forgetting phase 2)
        try:
            from core.memory.forgetting import ForgettingEngine
            forgetter = ForgettingEngine(self.anima_dir, self.anima_name)
            reorg_result = await forgetter.neurogenesis_reorganize(model=model)
            results["reorganization"] = reorg_result
            logger.info(
                "Neurogenesis reorganization: merged=%d",
                reorg_result.get("merged_count", 0),
            )
        except Exception:
            logger.exception("Neurogenesis reorganization failed for anima=%s", self.anima_name)

        # Step 4: Weekly procedural pattern distillation
        try:
            from core.memory.distillation import ProceduralDistiller

            distiller = ProceduralDistiller(self.anima_dir, self.anima_name)
            distill_result = await distiller.weekly_pattern_distill(model=model)
            results["weekly_distillation"] = distill_result
            logger.info(
                "Weekly pattern distillation: patterns=%d procedures=%d",
                distill_result.get("patterns_detected", 0),
                len(distill_result.get("procedures_created", [])),
            )
        except Exception:
            logger.exception(
                "Weekly pattern distillation failed for anima=%s",
                self.anima_name,
            )

        # Step 5: Full knowledge contradiction scan
        try:
            contradiction_result = await self._run_contradiction_check(
                [], model, full_scan=True,
            )
            results["contradiction"] = contradiction_result
        except Exception:
            logger.exception(
                "Weekly contradiction scan failed for anima=%s",
                self.anima_name,
            )

        logger.info(
            "Weekly integration complete for anima=%s: merged=%d compressed=%d",
            self.anima_name,
            len(results["knowledge_files_merged"]),
            results["episodes_compressed"]
        )

        return results

    async def _detect_duplicates(
        self,
        threshold: float = 0.85,
    ) -> list[tuple[str, str, float]]:
        """Detect duplicate/similar knowledge files using vector similarity.

        Args:
            threshold: Similarity threshold (0-1). Pairs above this are considered duplicates.

        Returns:
            List of (file1, file2, similarity_score) tuples, sorted by similarity desc.
        """
        try:
            from core.memory.rag.indexer import MemoryIndexer
            from core.memory.rag.retriever import MemoryRetriever
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store()
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)
            retriever = MemoryRetriever(vector_store, indexer, self.knowledge_dir)

            # Get all knowledge files
            knowledge_files = self._list_knowledge_files()
            if len(knowledge_files) < 2:
                return []

            # Compare each pair using vector similarity
            duplicates: list[tuple[str, str, float]] = []

            for i, file1 in enumerate(knowledge_files):
                for file2 in knowledge_files[i + 1:]:
                    # Read file contents
                    content1 = (self.knowledge_dir / file1).read_text(encoding="utf-8")
                    content2 = (self.knowledge_dir / file2).read_text(encoding="utf-8")

                    # Simple length check first
                    if abs(len(content1) - len(content2)) / max(len(content1), len(content2)) > 0.5:
                        continue

                    # Use RAG to find similarity
                    # Query with content1, check if content2 appears in top results
                    try:
                        results = retriever.search(
                            query=content1[:500],  # Use first 500 chars as query
                            anima_name=self.anima_name,
                            memory_type="knowledge",
                            top_k=10,
                        )

                        # Check if file2 appears in results
                        for result in results:
                            source_file = result.metadata.get("source_file", "")
                            if file2 in source_file:
                                similarity = result.score
                                if similarity >= threshold:
                                    duplicates.append((file1, file2, similarity))
                                    break

                    except Exception:
                        logger.debug("Failed to compare %s and %s", file1, file2)
                        continue

            # Sort by similarity descending
            duplicates.sort(key=lambda x: x[2], reverse=True)

            return duplicates

        except ImportError:
            logger.warning("RAG not available, cannot detect duplicates")
            return []
        except Exception:
            logger.exception("Failed to detect duplicates")
            return []

    async def _merge_knowledge_files(
        self,
        duplicates: list[tuple[str, str, float]],
        model: str,
    ) -> list[str]:
        """Merge duplicate knowledge files using LLM.

        Args:
            duplicates: List of (file1, file2, similarity) tuples
            model: LLM model to use for merging

        Returns:
            List of successfully merged file names
        """
        merged_files: list[str] = []

        for file1, file2, similarity in duplicates:
            try:
                # Read both files
                content1 = (self.knowledge_dir / file1).read_text(encoding="utf-8")
                content2 = (self.knowledge_dir / file2).read_text(encoding="utf-8")

                # Build merge prompt
                prompt = f"""以下は{self.anima_name}の類似した2つの知識ファイルです。

【ファイル1: {file1}】
{content1}

【ファイル2: {file2}】
{content2}

類似度: {similarity:.2f}

タスク:
この2つのファイルを1つに統合してください。

1. 重複する内容は1つにまとめる
2. 矛盾する記述があれば、より新しい/詳細な方を採用
3. 両方の情報を失わないように統合する

出力形式:
## 統合ファイル名
(file1とfile2を統合した適切なファイル名を提案。knowledge/は不要)

## 統合内容
(統合後のファイル全体の内容をMarkdown形式で記述)
"""

                # Call LLM
                import litellm

                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                )

                result = response.choices[0].message.content or ""

                # Parse result
                filename_match = re.search(
                    r"##\s*統合ファイル名\s*\n\s*(.+?)(?=\n##|\Z)",
                    result,
                    re.DOTALL
                )
                content_match = re.search(
                    r"##\s*統合内容\s*\n(.+)",
                    result,
                    re.DOTALL
                )

                if filename_match and content_match:
                    merged_filename = filename_match.group(1).strip()
                    merged_content = content_match.group(1).strip()

                    # Normalize filename
                    merged_filename = merged_filename.replace("knowledge/", "")
                    if not merged_filename.endswith(".md"):
                        merged_filename += ".md"

                    merged_path = _sanitize_filepath(
                        self.knowledge_dir, merged_filename,
                    )
                    merged_filename = merged_path.name

                    # Write merged file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    header = f"# {merged_path.stem}\n\n[AUTO-MERGED: {timestamp}]\n"
                    header += f"[SOURCE: {file1}, {file2}]\n\n"

                    merged_path.write_text(header + merged_content, encoding="utf-8")

                    # Archive original files (7-day retention)
                    archive_dir = self.anima_dir / "archive" / "merged"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    for orig_name in (file1, file2):
                        orig_path = self.knowledge_dir / orig_name
                        if orig_path.exists():
                            shutil.move(str(orig_path), str(archive_dir / orig_name))

                    merged_files.append(
                        f"{file1} + {file2} → {merged_filename}"
                    )

                    logger.info(
                        "Merged knowledge files: %s + %s → %s",
                        file1, file2, merged_filename
                    )

            except Exception:
                logger.exception("Failed to merge %s and %s", file1, file2)
                continue

        return merged_files

    async def _compress_old_episodes(
        self,
        retention_days: int = 30,
        model: str = "anthropic/claude-sonnet-4-20250514",
    ) -> int:
        """Compress old episodes that don't have [IMPORTANT] tags.

        Args:
            retention_days: Keep uncompressed episodes within this many days
            model: LLM model to use for compression

        Returns:
            Number of episodes compressed
        """
        cutoff_date = datetime.now().date() - timedelta(days=retention_days)
        compressed_count = 0

        # Iterate through episode files
        for episode_file in sorted(self.episodes_dir.glob("*.md")):
            try:
                # Parse date from filename — support both YYYY-MM-DD.md and YYYY-MM-DD_xxx.md
                stem = episode_file.stem
                date_str = stem[:10]  # Extract YYYY-MM-DD prefix
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # Skip if within retention period
                if file_date >= cutoff_date:
                    continue

                # Read episode content
                content = episode_file.read_text(encoding="utf-8")

                # Check for [IMPORTANT] tag
                if "[IMPORTANT]" in content or "[important]" in content.lower():
                    logger.debug("Skipping important episode: %s", episode_file.name)
                    continue

                # Compress with LLM
                prompt = f"""以下は{self.anima_name}の{date_str}のエピソード記録です。

【エピソード】
{content}

タスク:
主要な出来事のみを抽出し、簡潔に要約してください。
些細な会話や定型的なやり取りは省略してください。

出力形式:
## {date_str} 要約
- (要点1)
- (要点2)
...
"""

                import litellm

                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )

                summary = response.choices[0].message.content or ""

                if summary.strip():
                    # Backup original before compression
                    backup_dir = self.anima_dir / "archive" / "episodes"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(episode_file), str(backup_dir / episode_file.name))

                    # Replace original file with compressed version
                    compressed_header = f"# {date_str}\n\n[COMPRESSED: {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n\n"
                    episode_file.write_text(
                        compressed_header + summary,
                        encoding="utf-8"
                    )

                    compressed_count += 1
                    logger.info("Compressed episode: %s", episode_file.name)

            except Exception:
                logger.exception("Failed to compress episode: %s", episode_file.name)
                continue

        return compressed_count

    def _rebuild_rag_index(self) -> None:
        """Rebuild RAG index for all knowledge and episode files."""
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store()
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

    # ── Contradiction Detection Integration ────────────────────────

    async def _run_contradiction_check(
        self,
        affected_files: list[str],
        model: str,
        *,
        full_scan: bool = False,
    ) -> dict[str, int]:
        """Run contradiction detection and resolution on knowledge files.

        In daily mode (``full_scan=False``), checks only the newly created
        or updated files against existing knowledge.  In weekly mode
        (``full_scan=True``), scans the entire knowledge directory.

        Args:
            affected_files: List of filenames (relative to knowledge/) that
                were created or updated during this consolidation cycle
            model: LLM model for contradiction analysis
            full_scan: If True, scan all knowledge files (weekly mode)

        Returns:
            Resolution summary dict (superseded, merged, coexisted, errors)
        """
        from core.memory.contradiction import ContradictionDetector

        detector = ContradictionDetector(self.anima_dir, self.anima_name)

        if full_scan:
            # Weekly: scan entire knowledge directory
            logger.info(
                "Running full contradiction scan for anima=%s",
                self.anima_name,
            )
            contradictions = await detector.scan_contradictions(
                target_file=None, model=model,
            )
        else:
            # Daily: only check newly affected files
            if not affected_files:
                return {}

            contradictions: list = []
            for filename in affected_files:
                target = self.knowledge_dir / filename
                if not target.exists():
                    continue
                pairs = await detector.scan_contradictions(
                    target_file=target, model=model,
                )
                contradictions.extend(pairs)

        if not contradictions:
            logger.info(
                "No contradictions found for anima=%s", self.anima_name,
            )
            return {}

        # Resolve detected contradictions
        result = await detector.resolve_contradictions(contradictions, model)

        logger.info(
            "Contradiction resolution for anima=%s: %s",
            self.anima_name, result,
        )

        return result
