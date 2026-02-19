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
        model: str = "anthropic/claude-sonnet-4-20250514",
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

        prompt = f"""以下の手順書が繰り返し失敗しています。改善してください。

【手順書の説明】
{description}

【現在の手順書】
{content[:3000]}

【メタデータ】
- 失敗回数: {failure_count}
- 信頼度: {confidence}

タスク:
1. なぜこの手順書が失敗しているか考察してください
2. 改善された手順書を出力してください

出力形式:
改善後の手順書テキストのみを出力してください（説明やコメントは不要）。
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
