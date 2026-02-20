from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from datetime import datetime
from pathlib import Path

from core.memory._io import atomic_write_text
from core.schemas import SkillMeta

logger = logging.getLogger("animaworks.memory")

# ── FrontmatterService ────────────────────────────────────


class FrontmatterService:
    """YAML frontmatter read/write for knowledge and procedure files."""

    def __init__(
        self,
        anima_dir: Path,
        knowledge_dir: Path,
        procedures_dir: Path,
    ) -> None:
        self._anima_dir = anima_dir
        self._knowledge_dir = knowledge_dir
        self._procedures_dir = procedures_dir

    # ── Knowledge frontmatter ─────────────────────────────

    def write_knowledge_with_meta(self, path: Path, content: str, metadata: dict) -> None:
        """Write knowledge file with YAML frontmatter metadata."""
        import yaml

        frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        atomic_write_text(path, f"---\n{frontmatter}---\n\n{content}")
        logger.debug("Knowledge written with metadata path='%s' length=%d", path, len(content))

    def read_knowledge_content(self, path: Path) -> str:
        """Read knowledge file body, stripping YAML frontmatter if present."""
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text

    def read_knowledge_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a knowledge file.

        Applies legacy migration: renames ``superseded_at`` to
        ``valid_until`` when encountered.
        """
        import yaml

        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    logger.warning("Failed to parse YAML frontmatter in %s", path)
                    return {}
                # Legacy migration: superseded_at -> valid_until
                if "superseded_at" in meta and "valid_until" not in meta:
                    meta["valid_until"] = meta.pop("superseded_at")
                return meta
        return {}

    def update_knowledge_metadata(self, path: Path, updates: dict) -> None:
        """Partially update YAML frontmatter metadata of a knowledge file."""
        target = path if path.is_absolute() else self._knowledge_dir / path
        current = self.read_knowledge_metadata(target)
        current.update(updates)
        content = self.read_knowledge_content(target)
        self.write_knowledge_with_meta(target, content, current)

    # ── Procedure frontmatter ─────────────────────────────

    def write_procedure_with_meta(
        self, path: Path, content: str, metadata: dict,
    ) -> None:
        """Write a procedure file with YAML frontmatter metadata."""
        import yaml

        target = path if path.is_absolute() else self._procedures_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)

        fm_str = yaml.dump(metadata, default_flow_style=False, allow_unicode=True).rstrip()
        full = f"---\n{fm_str}\n---\n\n{content}"
        target.write_text(full, encoding="utf-8")
        logger.debug("Procedure written with metadata: %s", target.name)

    def read_procedure_content(self, path: Path) -> str:
        """Read procedure file body, stripping YAML frontmatter."""
        target = path if path.is_absolute() else self._procedures_dir / path
        if not target.exists():
            return ""
        text = target.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()

    def read_procedure_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a procedure file."""
        import yaml

        target = path if path.is_absolute() else self._procedures_dir / path
        if not target.exists():
            return {}
        text = target.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        try:
            fm = yaml.safe_load(parts[1])
            return fm if isinstance(fm, dict) else {}
        except Exception:
            return {}

    def list_procedure_metas(self, extract_skill_meta_fn) -> list[SkillMeta]:
        """Return SkillMeta for each procedure file."""
        return [
            extract_skill_meta_fn(f, is_common=False)
            for f in sorted(self._procedures_dir.glob("*.md"))
        ]

    def migrate_legacy_procedures(self) -> int:
        """Add YAML frontmatter to procedure files that lack it.

        Idempotent: uses ``{procedures_dir}/.migrated`` as a marker.
        Backs up originals to ``archive/pre_migration_procedures/``.

        Returns:
            Number of files migrated.
        """
        marker = self._procedures_dir / ".migrated"
        if marker.exists():
            logger.debug("Procedures already migrated (marker exists)")
            return 0

        md_files = sorted(self._procedures_dir.glob("*.md"))
        if not md_files:
            marker.write_text(datetime.now().isoformat(), encoding="utf-8")
            return 0

        backup_dir = self._anima_dir / "archive" / "pre_migration_procedures"
        backup_dir.mkdir(parents=True, exist_ok=True)

        migrated = 0
        for f in md_files:
            text = f.read_text(encoding="utf-8")
            if text.startswith("---"):
                continue  # already has frontmatter

            # Backup original
            shutil.copy2(f, backup_dir / f.name)

            # Derive description from filename
            desc = f.stem.replace("_", " ").replace("-", " ")

            metadata = {
                "description": desc,
                "tags": [],
                "success_count": 0,
                "failure_count": 0,
                "last_used": None,
                "confidence": 0.5,
                "version": 1,
                "created_at": datetime.now().isoformat(),
            }
            self.write_procedure_with_meta(f, text, metadata)
            migrated += 1
            logger.info("Migrated procedure: %s", f.name)

        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        logger.info("Migrated %d legacy procedures", migrated)
        return migrated
