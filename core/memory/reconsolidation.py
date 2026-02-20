from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Failure-count-based memory reconsolidation engine.

When a procedure has accumulated enough failures (failure_count >= 2)
and its confidence has dropped below a threshold (confidence < 0.6),
the system triggers reconsolidation — an LLM-driven revision of the
procedure content.  After revision the counters are reset and a new
version is created with an archived copy of the previous one.

Pipeline:
  1. Scan procedure frontmatter for reconsolidation targets
  2. LLM-based procedure revision
  3. Version-controlled update with counter reset
  4. Activity log event recording
"""

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import load_prompt

logger = logging.getLogger("animaworks.reconsolidation")


# ── ReconsolidationEngine ──────────────────────────────────────


class ReconsolidationEngine:
    """Failure-count-based memory reconsolidation engine.

    Scans procedure files for those with high failure counts and low
    confidence, then uses an LLM to revise the procedure content.
    Each revision is version-controlled with an archived copy.
    """

    def __init__(
        self,
        anima_dir: Path,
        anima_name: str,
        *,
        memory_manager: Any | None = None,
        activity_logger: Any | None = None,
    ) -> None:
        """Initialize the reconsolidation engine.

        Args:
            anima_dir: Path to anima's directory (~/.animaworks/animas/{name}).
            anima_name: Name of the anima for logging.
            memory_manager: Optional MemoryManager instance. Created if None.
            activity_logger: Optional ActivityLogger instance. Created if None.
        """
        self.anima_dir = anima_dir
        self.anima_name = anima_name

        if memory_manager is not None:
            self.memory_manager = memory_manager
        else:
            from core.memory.manager import MemoryManager
            self.memory_manager = MemoryManager(anima_dir)

        if activity_logger is not None:
            self.activity_logger = activity_logger
        else:
            from core.memory.activity import ActivityLogger
            self.activity_logger = ActivityLogger(anima_dir)

    # ── Target Detection ───────────────────────────────────────

    async def find_reconsolidation_targets(self) -> list[Path]:
        """Find procedures with failure_count >= 2 and confidence < 0.6.

        Scans all ``*.md`` files in the anima's ``procedures/`` directory,
        reading YAML frontmatter to check the trigger conditions.

        Returns:
            List of paths to procedure files that need reconsolidation.
        """
        targets: list[Path] = []
        procedures_dir = self.anima_dir / "procedures"
        if not procedures_dir.exists():
            return targets
        for md_file in procedures_dir.glob("*.md"):
            meta = self.memory_manager.read_procedure_metadata(md_file)
            failure_count = meta.get("failure_count", 0)
            confidence = meta.get("confidence", 1.0)
            if failure_count >= 2 and confidence < 0.6:
                targets.append(md_file)
        return targets

    # ── Reconsolidation Application ────────────────────────────

    async def apply_reconsolidation(
        self,
        targets: list[Path],
        model: str = "",
    ) -> dict[str, int]:
        """Revise procedures using LLM and reset counters.

        For each target procedure:
          1. Read current metadata and content
          2. Use LLM to generate a revised procedure
          3. Archive the current version
          4. Write revised content with reset metadata
          5. Record activity log event

        Args:
            targets: List of procedure file paths to reconsolidate.
            model: LLM model identifier (LiteLLM format) for revision.

        Returns:
            Dict with counts: "updated", "skipped", "errors".
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model
        results: dict[str, int] = {"updated": 0, "skipped": 0, "errors": 0}

        for proc_path in targets:
            try:
                meta = self.memory_manager.read_procedure_metadata(proc_path)
                content = self.memory_manager.read_procedure_content(proc_path)

                # Use LLM to revise the procedure
                revised = await self._revise_procedure(content, meta, model)
                if revised:
                    version = meta.get("version", 1)
                    self._archive_version(proc_path, content, version)

                    meta["version"] = version + 1
                    meta["failure_count"] = 0
                    meta["success_count"] = 0
                    meta["confidence"] = 0.5
                    meta["previous_version"] = f"v{version}"
                    meta["reconsolidated_at"] = datetime.now(UTC).isoformat()

                    self.memory_manager.write_procedure_with_meta(
                        proc_path, revised, meta,
                    )

                    # Activity log event
                    await self._log_reconsolidation(proc_path, version)
                    results["updated"] += 1

                    logger.info(
                        "Reconsolidated %s: v%d -> v%d",
                        proc_path.name, version, version + 1,
                    )
                else:
                    results["skipped"] += 1
                    logger.debug(
                        "Skipping reconsolidation (LLM returned no revision) "
                        "for %s",
                        proc_path.name,
                    )

            except Exception as e:
                logger.warning(
                    "Reconsolidation failed for %s: %s",
                    proc_path, e,
                )
                results["errors"] += 1

        logger.info(
            "Reconsolidation complete for anima=%s: "
            "updated=%d skipped=%d errors=%d",
            self.anima_name,
            results["updated"],
            results["skipped"],
            results["errors"],
        )
        return results

    # ── LLM Procedure Revision ─────────────────────────────────

    async def _revise_procedure(
        self,
        content: str,
        meta: dict[str, Any],
        model: str,
    ) -> str | None:
        """Use LLM to revise a procedure that has been failing.

        Args:
            content: Current procedure body text.
            meta: Current procedure metadata (failure_count, confidence, etc.).
            model: LLM model identifier (LiteLLM format).

        Returns:
            Revised procedure text, or None if revision was not possible.
        """
        failure_count = meta.get("failure_count", 0)
        confidence = meta.get("confidence", 0.5)
        description = meta.get("description", "")

        prompt = load_prompt(
            "memory/procedure_revision",
            description=description,
            content=content[:3000],
            failure_count=failure_count,
            confidence=confidence,
        )

        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )

            text = response.choices[0].message.content or ""

            # Sanitize LLM output
            from core.memory.consolidation import ConsolidationEngine
            text = ConsolidationEngine._sanitize_llm_output(text)

            if text.strip():
                return text.strip()
            return None

        except Exception as e:
            logger.warning(
                "LLM procedure revision failed: %s", e,
            )
            return None

    # ── Version Management ─────────────────────────────────────

    def _archive_version(
        self,
        file_path: Path,
        content: str,
        version: int,
    ) -> None:
        """Archive the current version of a procedure before reconsolidation.

        Saves a timestamped copy to archive/versions/ for audit trail.

        Args:
            file_path: Path to the procedure file to archive.
            content: Current content of the file (body text without frontmatter).
            version: Current version number from metadata.
        """
        if not file_path.exists():
            return

        archive_dir = self.anima_dir / "archive" / "versions"
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dest = archive_dir / f"{file_path.stem}_v{version}_{timestamp}{file_path.suffix}"
        shutil.copy2(str(file_path), str(dest))

        logger.debug(
            "Archived version %d of %s to %s",
            version, file_path.name, dest.name,
        )

    # ── Knowledge Reconsolidation ─────────────────────────────

    async def find_knowledge_reconsolidation_targets(self) -> list[Path]:
        """Find knowledge files with failure_count >= 2 and confidence < 0.6.

        Scans all ``*.md`` files in the anima's ``knowledge/`` directory,
        reading YAML frontmatter to check the trigger conditions.

        Returns:
            List of paths to knowledge files that need reconsolidation.
        """
        targets: list[Path] = []
        knowledge_dir = self.anima_dir / "knowledge"
        if not knowledge_dir.exists():
            return targets
        for md_file in knowledge_dir.glob("*.md"):
            meta = self.memory_manager.read_knowledge_metadata(md_file)
            failure_count = meta.get("failure_count", 0)
            confidence = meta.get("confidence", 1.0)
            if failure_count >= 2 and confidence < 0.6:
                targets.append(md_file)
        return targets

    async def reconsolidate_knowledge(
        self,
        model: str = "",
    ) -> dict[str, int]:
        """Revise knowledge files using LLM and reset counters.

        Scans knowledge/*.md files for those with failure_count >= 2
        and confidence < 0.6, then uses an LLM to revise the content.
        Each revision is version-controlled with an archived copy.

        Args:
            model: LLM model identifier (LiteLLM format) for revision.

        Returns:
            Dict with counts: "targets_found", "updated", "skipped", "errors".
        """
        if not model:
            from core.config.models import ConsolidationConfig
            model = ConsolidationConfig().llm_model
        targets = await self.find_knowledge_reconsolidation_targets()
        results: dict[str, Any] = {
            "targets_found": len(targets),
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "updated_files": [],
        }

        if not targets:
            return results

        for know_path in targets:
            try:
                meta = self.memory_manager.read_knowledge_metadata(know_path)
                content = self.memory_manager.read_knowledge_content(know_path)

                # Use LLM to revise the knowledge
                revised = await self._revise_knowledge(content, meta, model)
                if revised:
                    version = meta.get("version", 1)
                    self._archive_version(know_path, content, version)

                    meta["version"] = version + 1
                    meta["failure_count"] = 0
                    meta["success_count"] = 0
                    meta["confidence"] = 0.5
                    meta["previous_version"] = f"v{version}"
                    meta["reconsolidated_at"] = datetime.now(UTC).isoformat()

                    self.memory_manager.write_knowledge_with_meta(
                        know_path, revised, meta,
                    )

                    # Activity log event
                    self.activity_logger.log(
                        "knowledge_reconsolidated",
                        summary=(
                            f"Reconsolidated {know_path.name}: "
                            f"v{version} -> v{version + 1}"
                        ),
                        meta={
                            "knowledge": know_path.name,
                            "old_version": version,
                            "new_version": version + 1,
                        },
                    )
                    results["updated"] += 1
                    results["updated_files"].append(know_path.name)

                    logger.info(
                        "Reconsolidated knowledge %s: v%d -> v%d",
                        know_path.name, version, version + 1,
                    )
                else:
                    results["skipped"] += 1
                    logger.debug(
                        "Skipping knowledge reconsolidation (LLM returned "
                        "no revision) for %s",
                        know_path.name,
                    )

            except Exception as e:
                logger.warning(
                    "Knowledge reconsolidation failed for %s: %s",
                    know_path, e,
                )
                results["errors"] += 1

        logger.info(
            "Knowledge reconsolidation complete for anima=%s: "
            "targets=%d updated=%d skipped=%d errors=%d",
            self.anima_name,
            results["targets_found"],
            results["updated"],
            results["skipped"],
            results["errors"],
        )
        return results

    async def _revise_knowledge(
        self,
        content: str,
        meta: dict[str, Any],
        model: str,
    ) -> str | None:
        """Use LLM to revise knowledge that has been reported as inaccurate.

        Args:
            content: Current knowledge body text.
            meta: Current knowledge metadata (failure_count, confidence, etc.).
            model: LLM model identifier (LiteLLM format).

        Returns:
            Revised knowledge text, or None if revision was not possible.
        """
        failure_count = meta.get("failure_count", 0)
        confidence = meta.get("confidence", 0.5)

        prompt = f"""以下の知識ファイルが繰り返し不正確と報告されています。改善してください。

【現在の知識内容】
{content[:3000]}

【メタデータ】
- 失敗回数: {failure_count}
- 信頼度: {confidence}

タスク:
1. この知識がなぜ不正確と判断されたか考察してください
2. 改善された知識を出力してください

出力形式:
改善後の知識テキストのみを出力してください（説明やコメントは不要）。
コードフェンス（```）で囲まないでください。"""

        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )

            text = response.choices[0].message.content or ""

            # Sanitize LLM output
            from core.memory.consolidation import ConsolidationEngine
            text = ConsolidationEngine._sanitize_llm_output(text)

            if text.strip():
                return text.strip()
            return None

        except Exception as e:
            logger.warning(
                "LLM knowledge revision failed: %s", e,
            )
            return None

    # ── Activity Logging ───────────────────────────────────────

    async def _log_reconsolidation(
        self,
        proc_path: Path,
        old_version: int,
    ) -> None:
        """Record a procedure_reconsolidated event in the activity log.

        Args:
            proc_path: Path to the reconsolidated procedure file.
            old_version: Version number before reconsolidation.
        """
        self.activity_logger.log(
            "procedure_reconsolidated",
            summary=(
                f"Reconsolidated {proc_path.name}: "
                f"v{old_version} -> v{old_version + 1}"
            ),
            meta={
                "procedure": proc_path.name,
                "old_version": old_version,
                "new_version": old_version + 1,
            },
        )
