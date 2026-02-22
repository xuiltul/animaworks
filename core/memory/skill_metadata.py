from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import unicodedata
from pathlib import Path

from core.schemas import SkillMeta

logger = logging.getLogger("animaworks.memory")

# ── Skill matching (module-level functions) ──────────────


def _normalize_text(text: str) -> str:
    """NFKC normalization + lowercase for keyword matching."""
    return unicodedata.normalize("NFKC", text).lower()


def _extract_bracket_keywords(desc_norm: str) -> list[str]:
    """Extract keywords from bracket-delimited tokens."""
    return re.findall(r"\u300c(.+?)\u300d", desc_norm)


def _extract_comma_keywords(desc_norm: str) -> list[str]:
    """Extract keywords by splitting on commas, periods, and newlines.

    Used as fallback when no brackets are present.
    Returns short phrases (2-20 chars) that are likely meaningful keywords.
    """
    segments = re.split(r"[\u3001,\u3002.\n]", desc_norm)
    return [s.strip() for s in segments if 2 <= len(s.strip()) <= 20]


def _match_tier1(desc_norm: str, message_norm: str) -> bool:
    """Tier 1: Bracket keyword match + comma/delimiter keyword match.

    Returns True if any keyword extracted from the description is a
    substring of the message.
    """
    # Primary: bracket keywords
    keywords = _extract_bracket_keywords(desc_norm)
    if keywords:
        return any(kw in message_norm for kw in keywords)
    # Fallback: comma/delimiter separated keywords
    keywords = _extract_comma_keywords(desc_norm)
    if keywords:
        return any(kw in message_norm for kw in keywords)
    return False


# Common English stop words to exclude from Tier 2 vocabulary matching.
_TIER2_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "use", "used",
    "when", "into", "also", "can", "are", "was", "has", "have", "had",
    "not", "but", "its", "any", "all", "each", "more", "such", "than",
    "tool", "file", "new", "via", "etc", "using", "other",
})


def _match_tier2(desc_norm: str, message_norm: str) -> bool:
    """Tier 2: Description vocabulary match.

    Extracts words (>=3 chars) from description and checks if >=2 appear
    in the message. Stop words are filtered to avoid false positives.
    Word boundary matching is used for ASCII words to prevent substring
    collisions (e.g. 'git' matching inside 'digital').
    """
    raw_words = re.findall(r"[\w]{3,}", desc_norm)
    words = [w for w in raw_words if w not in _TIER2_STOP_WORDS]
    if not words:
        return False
    match_count = 0
    for w in words:
        if w.isascii():
            if re.search(rf"\b{re.escape(w)}\b", message_norm):
                match_count += 1
        else:
            if w in message_norm:
                match_count += 1
    return match_count >= 2


def match_skills_by_description(
    message: str,
    skills: list[SkillMeta],
    *,
    retriever: object | None = None,
    anima_name: str = "",
) -> list[SkillMeta]:
    """Return skills whose description matches the message (3-tier).

    Tier 1: bracket-delimited and comma/delimiter keyword substring match.
    Tier 2: Description vocabulary match (>=2 words overlap).
    Tier 3: Vector search via RAG retriever (semantic similarity).

    Each tier is applied only to skills not yet matched by prior tiers.
    Results are deduplicated and returned in tier priority order.
    """
    if not message:
        return []
    message_norm = _normalize_text(message)
    matched: list[SkillMeta] = []
    matched_names: set[str] = set()
    remaining: list[SkillMeta] = []

    # -- Tier 1: Bracket / comma keyword match --
    for skill in skills:
        if not skill.description:
            remaining.append(skill)
            continue
        desc_norm = _normalize_text(skill.description)
        if _match_tier1(desc_norm, message_norm):
            matched.append(skill)
            matched_names.add(skill.name)
        else:
            remaining.append(skill)

    # -- Tier 2: Description vocabulary match --
    still_remaining: list[SkillMeta] = []
    for skill in remaining:
        if not skill.description:
            still_remaining.append(skill)
            continue
        desc_norm = _normalize_text(skill.description)
        if _match_tier2(desc_norm, message_norm):
            if skill.name not in matched_names:
                matched.append(skill)
                matched_names.add(skill.name)
        else:
            still_remaining.append(skill)

    # -- Tier 3: Vector search (semantic match) --
    if retriever is not None and anima_name and still_remaining:
        try:
            vector_matched = _match_tier3_vector(
                message, still_remaining, retriever, anima_name,
            )
            for skill in vector_matched:
                if skill.name not in matched_names:
                    matched.append(skill)
                    matched_names.add(skill.name)
        except Exception as e:
            logger.warning("Tier 3 vector search failed: %s", e)

    return matched


def _match_tier3_vector(
    message: str,
    candidates: list[SkillMeta],
    retriever: object,
    anima_name: str,
    top_k: int = 3,
    min_score: float = 0.88,
) -> list[SkillMeta]:
    """Tier 3: Use RAG vector search to find semantically matching skills."""
    from core.memory.rag.retriever import MemoryRetriever

    if not isinstance(retriever, MemoryRetriever):
        return []

    results = retriever.search(
        query=message,
        anima_name=anima_name,
        memory_type="skills",
        top_k=top_k,
        include_shared=True,
    )

    # Build path-to-skill lookup from candidates
    candidate_by_path: dict[str, SkillMeta] = {}
    for skill in candidates:
        candidate_by_path[str(skill.path)] = skill
        candidate_by_path[skill.path.stem] = skill
        candidate_by_path[skill.name] = skill

    matched: list[SkillMeta] = []
    seen: set[str] = set()
    for r in results:
        if r.score < min_score:
            continue
        file_path = r.metadata.get("file_path", "") or r.metadata.get("source_file", "")
        skill = candidate_by_path.get(str(file_path))
        if skill is None and file_path:
            stem = Path(file_path).stem
            skill = candidate_by_path.get(stem)
        if skill and skill.name not in seen:
            matched.append(skill)
            seen.add(skill.name)
    return matched


# ── SkillMetadataService ─────────────────────────────────


class SkillMetadataService:
    """Skill YAML frontmatter parsing, listing, and matching."""

    def __init__(self, skills_dir: Path, common_skills_dir: Path) -> None:
        self._skills_dir = skills_dir
        self._common_skills_dir = common_skills_dir

    @staticmethod
    def extract_skill_meta(path: Path, *, is_common: bool = False) -> SkillMeta:
        """Extract SkillMeta from a skill file's YAML frontmatter.

        Supports Claude Code format (name + description frontmatter only).
        Falls back to filename stem and empty description if no frontmatter.
        """
        text = path.read_text(encoding="utf-8")
        name = path.stem
        description = ""
        allowed_tools: list[str] = []

        # Parse YAML frontmatter (--- delimited)
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    fm = yaml.safe_load(parts[1])
                    if isinstance(fm, dict):
                        name = fm.get("name", name)
                        description = fm.get("description", "")
                        if description:
                            description = str(description).strip()
                        allowed_tools = fm.get("allowed_tools", [])
                        if not isinstance(allowed_tools, list):
                            allowed_tools = []
                except Exception:
                    logger.debug("Failed to parse YAML frontmatter in %s", path, exc_info=True)

        # Fallback: extract from ## 概要 section (legacy format)
        if not description:
            in_overview = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped == "## \u6982\u8981":
                    in_overview = True
                    continue
                if in_overview:
                    if stripped.startswith("#"):
                        break
                    if stripped:
                        description = stripped
                        break

        return SkillMeta(
            name=name,
            description=description,
            path=path,
            is_common=is_common,
            allowed_tools=allowed_tools,
        )

    def list_skill_metas(self) -> list[SkillMeta]:
        """Return SkillMeta for each personal skill."""
        return [
            self.extract_skill_meta(f, is_common=False)
            for f in sorted(self._skills_dir.glob("*.md"))
        ]

    def list_common_skill_metas(self) -> list[SkillMeta]:
        """Return SkillMeta for each common skill."""
        if not self._common_skills_dir.is_dir():
            return []
        return [
            self.extract_skill_meta(f, is_common=True)
            for f in sorted(self._common_skills_dir.glob("*.md"))
        ]

    def list_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (name, description) for each personal skill."""
        return [(m.name, m.description) for m in self.list_skill_metas()]

    def list_common_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (name, description) for each common skill."""
        return [(m.name, m.description) for m in self.list_common_skill_metas()]
