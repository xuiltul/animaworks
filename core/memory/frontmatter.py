from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.schemas import SkillMeta

logger = logging.getLogger("animaworks.memory")


# ── Robust Frontmatter Parser ─────────────────────────────
#
# The old ``text.split("---", 2)`` approach splits on the *substring*
# ``---`` anywhere in the file, which breaks when YAML values contain
# ``---`` (e.g. ``description: "Before---After"``) or when LLM output
# produces double-frontmatter.  The line-based parser below only
# recognises ``---`` that appears as a **standalone line** (with
# optional trailing whitespace).

_FM_FENCE = re.compile(r"^---\s*$", re.MULTILINE)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split text into (frontmatter_yaml, body) using line-based parsing.

    Looks for an opening ``---`` line at position 0, then scans for the
    next standalone ``---`` line to close the block.  This avoids false
    splits when YAML values or body content contain the ``---`` substring.

    Returns:
        ``(yaml_str, body)`` where *yaml_str* is the raw YAML between
        delimiters (empty string if no frontmatter) and *body* is the
        remaining content (stripped of leading blank lines).
    """
    if not text.startswith("---"):
        return "", text

    # Skip the opening ``---`` line
    first_newline = text.index("\n") if "\n" in text else len(text)
    rest = text[first_newline + 1:]

    m = _FM_FENCE.search(rest)
    if m is None:
        return "", text

    yaml_str = rest[:m.start()]
    body = rest[m.end():]
    # Strip at most two leading newlines (the blank line after ``---``)
    if body.startswith("\n\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    return yaml_str, body


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and return ``(metadata_dict, body)``.

    Wraps :func:`split_frontmatter` with ``yaml.safe_load``.  Returns
    an empty dict when no frontmatter is present or parsing fails.
    """
    yaml_str, body = split_frontmatter(text)
    if not yaml_str:
        return {}, body

    import yaml
    try:
        meta = yaml.safe_load(yaml_str)
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        logger.debug("Failed to parse YAML frontmatter", exc_info=True)
        meta = {}
    return meta, body


def strip_frontmatter(text: str) -> str:
    """Return *text* with YAML frontmatter removed (body only)."""
    _, body = split_frontmatter(text)
    return body


def strip_content_frontmatter(content: str) -> str:
    """Strip accidental frontmatter from *content* before wrapping.

    Used by write helpers to prevent double-frontmatter when LLM output
    already contains ``---`` delimiters.
    """
    if content.lstrip().startswith("---"):
        _, body = split_frontmatter(content.lstrip())
        return body
    return content


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

        content = strip_content_frontmatter(content)
        frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        atomic_write_text(path, f"---\n{frontmatter}---\n\n{content}")
        logger.debug("Knowledge written with metadata path='%s' length=%d", path, len(content))

    def read_knowledge_content(self, path: Path) -> str:
        """Read knowledge file body, stripping YAML frontmatter if present."""
        text = path.read_text(encoding="utf-8")
        _, body = split_frontmatter(text)
        return body.strip()

    def read_knowledge_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a knowledge file.

        Applies legacy migration: renames ``superseded_at`` to
        ``valid_until`` when encountered.
        """
        text = path.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(text)
        if not meta:
            return {}
        if "superseded_at" in meta and "valid_until" not in meta:
            meta["valid_until"] = meta.pop("superseded_at")
        return meta

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

        content = strip_content_frontmatter(content)
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
        _, body = split_frontmatter(text)
        return body.strip()

    def read_procedure_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a procedure file."""
        target = path if path.is_absolute() else self._procedures_dir / path
        if not target.exists():
            return {}
        text = target.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(text)
        return meta

    def list_procedure_metas(self, extract_skill_meta_fn) -> list[SkillMeta]:
        """Return SkillMeta for each procedure file."""
        return [
            extract_skill_meta_fn(f, is_common=False)
            for f in sorted(self._procedures_dir.glob("*.md"))
        ]

    @staticmethod
    def _extract_description(text: str, fallback_name: str) -> str:
        """Extract description from the first ``# `` heading in *text*.

        Falls back to *fallback_name* with ``_`` and ``-`` replaced by
        spaces when no heading is found.
        """
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped.lstrip("# ").strip()
        return fallback_name.replace("_", " ").replace("-", " ")

    def ensure_procedure_frontmatter(self) -> int:
        """Ensure all procedure files have YAML frontmatter with description.

        Scans every .md file in procedures/. Files without frontmatter
        get auto-generated metadata with description extracted from the
        first Markdown heading. Idempotent per-file (not marker-based).

        Returns:
            Number of files that had frontmatter added.
        """
        if not self._procedures_dir.exists():
            return 0

        md_files = sorted(self._procedures_dir.glob("*.md"))
        migrated = 0
        for f in md_files:
            text = f.read_text(encoding="utf-8")
            if text.lstrip().startswith("---"):
                continue  # already has frontmatter

            desc = self._extract_description(text, f.stem)

            metadata = {
                "description": desc,
                "success_count": 0,
                "failure_count": 0,
                "confidence": 0.5,
            }
            self.write_procedure_with_meta(f, text, metadata)
            migrated += 1
            logger.info("Added frontmatter to procedure: %s", f.name)

        if migrated:
            logger.info("Added frontmatter to %d procedures", migrated)
        return migrated

    def ensure_knowledge_frontmatter(self) -> int:
        """Ensure all knowledge files have YAML frontmatter.

        Scans every .md file in knowledge/. Files without frontmatter
        get auto-generated metadata with timestamps derived from the
        file's mtime. Idempotent per-file.

        Returns:
            Number of files that had frontmatter added.
        """
        from datetime import datetime, timedelta, timezone

        if not self._knowledge_dir.exists():
            return 0

        _JST = timezone(timedelta(hours=9))
        md_files = sorted(self._knowledge_dir.glob("*.md"))
        migrated = 0
        for f in md_files:
            text = f.read_text(encoding="utf-8")
            if text.lstrip().startswith("---"):
                continue

            ts = datetime.fromtimestamp(f.stat().st_mtime, tz=_JST).isoformat()
            metadata = {
                "confidence": 0.5,
                "created_at": ts,
                "updated_at": ts,
                "source_episodes": 0,
                "auto_consolidated": False,
                "version": 1,
            }
            self.write_knowledge_with_meta(f, text, metadata)
            migrated += 1
            logger.info("Added frontmatter to knowledge: %s", f.name)

        if migrated:
            logger.info("Added frontmatter to %d knowledge files", migrated)
        return migrated
