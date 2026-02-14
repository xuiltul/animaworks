from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Memory consolidation engine.

Implements daily and weekly consolidation processes that convert episodic
memories into semantic knowledge, analogous to sleep-based memory consolidation
in the human brain.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.consolidation")


# ── ConsolidationEngine ────────────────────────────────────────


class ConsolidationEngine:
    """Handles automatic memory consolidation processes.

    This class implements:
    - Daily consolidation: Episodes → Knowledge (NREM sleep analog)
    - Weekly integration: Knowledge merging and episode compression
    """

    def __init__(self, person_dir: Path, person_name: str) -> None:
        """Initialize consolidation engine.

        Args:
            person_dir: Path to person's directory (~/.animaworks/persons/{name})
            person_name: Name of the person for logging
        """
        self.person_dir = person_dir
        self.person_name = person_name
        self.episodes_dir = person_dir / "episodes"
        self.knowledge_dir = person_dir / "knowledge"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

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
        logger.info("Starting daily consolidation for person=%s", self.person_name)

        # Collect recent episodes
        episode_entries = self._collect_recent_episodes(hours=24)

        if len(episode_entries) < min_episodes:
            logger.info(
                "Skipping daily consolidation for person=%s: "
                "only %d episodes found (min=%d)",
                self.person_name, len(episode_entries), min_episodes
            )
            return {
                "episodes_processed": 0,
                "knowledge_files_created": [],
                "knowledge_files_updated": [],
                "skipped": True,
            }

        logger.info(
            "Consolidating %d episode entries for person=%s",
            len(episode_entries), self.person_name
        )

        # Get existing knowledge files for context
        existing_knowledge = self._list_knowledge_files()

        # Generate consolidation via LLM
        consolidation_result = await self._summarize_episodes(
            episode_entries=episode_entries,
            existing_knowledge_files=existing_knowledge,
            model=model,
        )

        # Parse and write results to knowledge/
        files_created, files_updated = self._merge_to_knowledge(consolidation_result)

        # Update RAG index for affected files
        self._update_rag_index(files_created + files_updated)

        logger.info(
            "Daily consolidation complete for person=%s: "
            "created=%d updated=%d",
            self.person_name, len(files_created), len(files_updated)
        )

        return {
            "episodes_processed": len(episode_entries),
            "knowledge_files_created": files_created,
            "knowledge_files_updated": files_updated,
            "skipped": False,
        }

    def _collect_recent_episodes(self, hours: int = 24) -> list[dict[str, str]]:
        """Collect episode entries from the past N hours.

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
            episode_file = self.episodes_dir / f"{target_date}.md"

            if not episode_file.exists():
                continue

            content = episode_file.read_text(encoding="utf-8")

            # Parse episode entries (format: ## HH:MM — Title)
            for match in re.finditer(
                r"^## (\d{2}:\d{2})\s*—\s*(.+?)(?=^##|\Z)",
                content,
                re.MULTILINE | re.DOTALL
            ):
                time_str = match.group(1)
                entry_content = match.group(2).strip()

                # Parse timestamp
                try:
                    entry_dt = datetime.strptime(
                        f"{target_date} {time_str}",
                        "%Y-%m-%d %H:%M"
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
                        target_date, time_str
                    )

        # Sort by datetime (newest first)
        entries.sort(
            key=lambda e: datetime.strptime(f"{e['date']} {e['time']}", "%Y-%m-%d %H:%M"),
            reverse=True
        )

        return entries

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

    async def _summarize_episodes(
        self,
        episode_entries: list[dict[str, str]],
        existing_knowledge_files: list[str],
        model: str,
    ) -> str:
        """Use LLM to extract lessons from episodes.

        Args:
            episode_entries: List of episode entries with date, time, content
            existing_knowledge_files: List of existing knowledge file names
            model: LLM model to use

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

        # Build consolidation prompt
        prompt = f"""以下は{self.person_name}の過去24時間のエピソード記録です。

【エピソード】
{episodes_text}

【既存の知識ファイル一覧】
{knowledge_list}

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
- 些細な会話や定型的なやり取りは知識化不要です
- 一般化できる教訓やパターンのみを抽出してください
- 既存ファイルがない場合は、すべて新規ファイルとして提案してください
- ファイル名はトピックを表すわかりやすい名前にしてください（英数字とハイフン推奨）
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
            return result

        except Exception as e:
            logger.error(
                "Failed to call LLM for consolidation: %s",
                e,
                exc_info=True
            )
            return ""

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
            return files_created, files_updated

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

                filepath = self.knowledge_dir / filename

                # Append to existing file with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                header = f"\n\n[AUTO-CONSOLIDATED: {timestamp}]\n\n"

                if filepath.exists():
                    with filepath.open("a", encoding="utf-8") as f:
                        f.write(header + content)
                    files_updated.append(filename)
                    logger.info("Updated knowledge file: %s", filename)
                else:
                    # File doesn't exist, create it
                    filepath.write_text(
                        f"# {filepath.stem}\n\n{header}{content}",
                        encoding="utf-8"
                    )
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

                filepath = self.knowledge_dir / filename

                # Add metadata header
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                header = f"# {filepath.stem}\n\n[AUTO-CONSOLIDATED: {timestamp}]\n\n"

                if not filepath.exists():
                    filepath.write_text(header + content, encoding="utf-8")
                    files_created.append(filename)
                    logger.info("Created knowledge file: %s", filename)
                else:
                    logger.warning(
                        "Skipping knowledge file creation (already exists): %s",
                        filename
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
            from core.memory.rag.store import ChromaVectorStore

            vector_store = ChromaVectorStore()
            indexer = MemoryIndexer(vector_store, self.person_name, self.person_dir)

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
        logger.info("Starting weekly integration for person=%s", self.person_name)

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
                    "Found %d duplicate knowledge file pairs for person=%s",
                    len(duplicates), self.person_name
                )
                merged_files = await self._merge_knowledge_files(duplicates, model)
                results["knowledge_files_merged"] = merged_files
            else:
                logger.info("No duplicate knowledge files found for person=%s", self.person_name)

        except Exception:
            logger.exception("Failed to merge knowledge files for person=%s", self.person_name)

        # Step 2: Compress old episodes
        try:
            compressed_count = await self._compress_old_episodes(
                retention_days=episode_retention_days,
                model=model,
            )
            results["episodes_compressed"] = compressed_count
            logger.info(
                "Compressed %d old episodes for person=%s",
                compressed_count, self.person_name
            )

        except Exception:
            logger.exception("Failed to compress old episodes for person=%s", self.person_name)

        # Step 3: Rebuild RAG index for affected files
        try:
            self._rebuild_rag_index()
        except Exception:
            logger.exception("Failed to rebuild RAG index for person=%s", self.person_name)

        logger.info(
            "Weekly integration complete for person=%s: merged=%d compressed=%d",
            self.person_name,
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
            from core.memory.rag import HybridRetriever
            from core.memory.rag.store import ChromaVectorStore

            vector_store = ChromaVectorStore()
            retriever = HybridRetriever(vector_store, self.person_name, self.person_dir)

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
                            memory_type="knowledge",
                            top_k=10,
                        )

                        # Check if file2 appears in results
                        for result in results:
                            if file2 in result.get("source_file", ""):
                                similarity = result.get("score", 0.0)
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
                prompt = f"""以下は{self.person_name}の類似した2つの知識ファイルです。

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

                    merged_path = self.knowledge_dir / merged_filename

                    # Write merged file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    header = f"# {merged_path.stem}\n\n[AUTO-MERGED: {timestamp}]\n"
                    header += f"[SOURCE: {file1}, {file2}]\n\n"

                    merged_path.write_text(header + merged_content, encoding="utf-8")

                    # Delete original files
                    (self.knowledge_dir / file1).unlink(missing_ok=True)
                    (self.knowledge_dir / file2).unlink(missing_ok=True)

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
                # Parse date from filename (format: YYYY-MM-DD.md)
                date_str = episode_file.stem
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
                prompt = f"""以下は{self.person_name}の{date_str}のエピソード記録です。

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
            from core.memory.rag.store import ChromaVectorStore

            vector_store = ChromaVectorStore()
            indexer = MemoryIndexer(vector_store, self.person_name, self.person_dir)

            # Re-index all knowledge files
            for knowledge_file in self.knowledge_dir.rglob("*.md"):
                indexer.index_file(knowledge_file, memory_type="knowledge")
                logger.debug("Re-indexed knowledge: %s", knowledge_file.name)

            # Re-index all episode files
            for episode_file in self.episodes_dir.glob("*.md"):
                indexer.index_file(episode_file, memory_type="episodes")
                logger.debug("Re-indexed episode: %s", episode_file.name)

            logger.info("RAG index rebuild complete for person=%s", self.person_name)

        except ImportError:
            logger.debug("RAG not available, skipping index rebuild")
        except Exception:
            logger.exception("Failed to rebuild RAG index")
