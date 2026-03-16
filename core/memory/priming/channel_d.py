from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel D: Skill matching via description-based 3-tier search."""

import logging
from pathlib import Path

from core.tools._async_compat import run_sync

logger = logging.getLogger("animaworks.priming")

_MAX_SKILL_MATCHES = 5


async def channel_d_skill_match(
    anima_dir: Path,
    skills_dir: Path,
    get_retriever: callable,
    message: str,
    keywords: list[str],
    channel: str = "chat",
) -> list[str]:
    """Channel D: Skill matching via description-based 3-tier search.

    Uses ``match_skills_by_description()`` from ``core.memory.manager``
    which applies Tier 1 (keyword), Tier 2 (vocabulary), and Tier 3
    (vector search) matching.  The MemoryRetriever instance is shared
    with Channel C via get_retriever.

    Returns list of skill/procedure names (not full content, max 5).
    Searches personal skills/, common_skills/, and procedures/.
    """
    if not message and not keywords:
        return []

    from core.memory.manager import MemoryManager, match_skills_by_description
    from core.paths import get_common_skills_dir

    def _collect_metas() -> list:
        metas: list = []
        names: set[str] = set()

        if skills_dir.is_dir():
            personal_skill_files = sorted(skills_dir.glob("*/SKILL.md"))
            personal_skill_files.extend(sorted(skills_dir.glob("*.md")))
            for f in personal_skill_files:
                try:
                    meta = MemoryManager._extract_skill_meta(f, is_common=False)
                    if meta.name not in names:
                        metas.append(meta)
                        names.add(meta.name)
                except Exception:
                    logger.debug("Failed to extract skill meta from %s", f, exc_info=True)

        common_dir = get_common_skills_dir()
        if common_dir.is_dir():
            common_skill_files = sorted(common_dir.glob("*/SKILL.md"))
            common_skill_files.extend(sorted(common_dir.glob("*.md")))
            for f in common_skill_files:
                try:
                    meta = MemoryManager._extract_skill_meta(f, is_common=True)
                    if meta.name not in names:
                        metas.append(meta)
                        names.add(meta.name)
                except Exception:
                    logger.debug("Failed to extract common skill meta from %s", f, exc_info=True)

        procedures_dir = anima_dir / "procedures"
        if procedures_dir.is_dir():
            for f in sorted(procedures_dir.glob("*.md")):
                try:
                    meta = MemoryManager._extract_skill_meta(f, is_common=False)
                    if meta.name not in names:
                        metas.append(meta)
                        names.add(meta.name)
                except Exception:
                    logger.debug("Failed to extract procedure meta from %s", f, exc_info=True)

        return metas

    all_metas = await run_sync(_collect_metas)

    if not all_metas:
        return []

    try:
        retriever = get_retriever()
        anima_name = anima_dir.name

        matched = match_skills_by_description(
            message,
            all_metas,
            retriever=retriever,
            anima_name=anima_name,
        )

        result = [m.name for m in matched[:_MAX_SKILL_MATCHES]]

        if result:
            logger.debug(
                "Channel D: Matched %d skills: %s",
                len(result),
                result,
            )

        return result

    except Exception as e:
        logger.warning(
            "Channel D: Full skill matching failed, trying Tier 1/2 only: %s",
            e,
        )
        try:
            matched = match_skills_by_description(
                message,
                all_metas,
                retriever=None,
                anima_name="",
            )
            result = [m.name for m in matched[:_MAX_SKILL_MATCHES]]
            if result:
                logger.debug(
                    "Channel D: Tier 1/2 fallback matched %d skills: %s",
                    len(result),
                    result,
                )
            return result
        except Exception as e2:
            logger.warning("Channel D: Tier 1/2 fallback also failed: %s", e2)
            return []
