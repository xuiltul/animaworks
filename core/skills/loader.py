from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Load SKILL.md files: YAML frontmatter plus Markdown body."""

import logging
from pathlib import Path
from typing import Any

from core.memory.frontmatter import parse_frontmatter, split_frontmatter
from core.skills.models import SkillMetadata

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────


def _infer_default_name(path: Path) -> str:
    """Derive skill name from path when frontmatter omits it."""
    return path.parent.name if path.name == "SKILL.md" else path.stem


def _overview_first_line(body: str) -> str | None:
    """Return the first non-empty line under a ``## 概要`` section, if any."""
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "## 概要":
            i += 1
            while i < len(lines):
                raw = lines[i]
                stripped_outer = raw.strip()
                if raw.startswith("## ") and stripped_outer != "## 概要":
                    return None
                if stripped_outer:
                    return stripped_outer
                i += 1
            return None
        i += 1
    return None


def _normalize_name(raw: Any, path: Path) -> str:
    """Resolve skill ``name`` from frontmatter or path defaults."""
    if raw is None:
        return _infer_default_name(path)
    if isinstance(raw, str) and not raw.strip():
        return _infer_default_name(path)
    return str(raw).strip()


def _normalize_description(raw: Any, body: str) -> str:
    """Resolve ``description`` from frontmatter, then ``## 概要``, then empty."""
    if raw is None:
        desc = ""
    elif isinstance(raw, str):
        desc = raw.strip()
    else:
        desc = str(raw).strip()
    if desc:
        return desc
    from_heading = _overview_first_line(body)
    return from_heading if from_heading else ""


def _parse_skill_file(path: Path) -> tuple[dict[str, Any], str]:
    """Read *path*, parse frontmatter, and warn when YAML is present but unusable."""
    text = path.read_text(encoding="utf-8")
    yaml_str, _ = split_frontmatter(text)
    meta, body = parse_frontmatter(text)
    if yaml_str.strip() and not meta:
        logger.warning(
            "Failed to parse YAML frontmatter in skill file %s; using defaults",
            path,
        )
    return meta, body


def _skill_metadata_from_dict(path: Path, meta: dict[str, Any], body: str) -> SkillMetadata:
    """Build :class:`SkillMetadata` from parsed YAML plus inferred fields."""
    name = _normalize_name(meta.get("name"), path)
    description = _normalize_description(meta.get("description"), body)
    passthrough = {k: v for k, v in meta.items() if k not in ("name", "description", "path")}
    source = passthrough.get("source")
    if isinstance(source, str):
        source_type = source.strip()
        passthrough["source"] = {"type": source_type or "local"}
    if "version" in passthrough:
        try:
            passthrough["version"] = int(passthrough["version"])
        except (TypeError, ValueError):
            del passthrough["version"]
    return SkillMetadata(name=name, description=description, path=path, **passthrough)


# ── Public API ─────────────────────────────────────────────


def load_skill_metadata(path: Path) -> SkillMetadata:
    """Parse YAML frontmatter from a skill file and return :class:`SkillMetadata`.

    When no frontmatter is present, ``name`` is inferred from *path*
    (parent directory name for ``SKILL.md``, otherwise the file stem) and
    ``trust_level`` defaults to ``trusted``.

    Args:
        path: Filesystem path to the skill Markdown file.

    Returns:
        Parsed metadata with ``path`` set to *path*.
    """
    meta, body = _parse_skill_file(path)
    return _skill_metadata_from_dict(path, meta, body)


def is_skill_blocked(metadata: SkillMetadata) -> bool:
    """Check if a skill should be blocked from loading.

    A skill is blocked if its trust_level is ``blocked`` or its security
    scan verdict is ``dangerous``.

    Args:
        metadata: Parsed skill metadata.

    Returns:
        True if the skill should not be loaded.
    """
    from core.skills.models import SkillScanVerdict, SkillTrustLevel

    if metadata.trust_level == SkillTrustLevel.blocked:
        return True
    return metadata.security.verdict == SkillScanVerdict.dangerous


def skill_access_decision(metadata: SkillMetadata, *, anima_dir: Path | None = None) -> tuple[bool, str]:
    """Return load/access decision for a skill metadata record."""
    from core.skills.curator import curator_allows_access

    return curator_allows_access(metadata, anima_dir=anima_dir)


def is_skill_loadable(metadata: SkillMetadata, *, anima_dir: Path | None = None) -> bool:
    """Return True when the skill is not blocked by trust, security, or curator state."""
    allowed, _reason = skill_access_decision(metadata, anima_dir=anima_dir)
    return allowed


def load_skill_body(path: Path) -> str:
    """Return the Markdown body of *path* with YAML frontmatter removed.

    Args:
        path: Filesystem path to the skill Markdown file.

    Returns:
        Body text after the closing frontmatter delimiter.
    """
    text = path.read_text(encoding="utf-8")
    _, body = parse_frontmatter(text)
    return body


def load_skill_document(path: Path) -> tuple[SkillMetadata, str]:
    """Load both metadata and body from a skill file.

    Args:
        path: Filesystem path to the skill Markdown file.

    Returns:
        Tuple of (:class:`SkillMetadata`, body string).
    """
    meta, body = _parse_skill_file(path)
    return _skill_metadata_from_dict(path, meta, body), body
