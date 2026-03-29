from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel D: Skill matching — DEPRECATED.

Skill discovery is now handled by the skill catalog in system prompt Group 4.
This channel is retained as a no-op stub for backward compatibility.
"""

from pathlib import Path
from typing import Any

_MAX_SKILL_MATCHES = 5  # Retained for backward compat


async def channel_d_skill_match(
    anima_dir: Path,
    skills_dir: Path,
    get_retriever: Any,
    message: str,
    keywords: list[str],
    channel: str = "chat",
) -> list[str]:
    """Channel D: No-op stub (deprecated).

    Skill matching is now handled by the skill catalog in the system prompt.
    Always returns an empty list.
    """
    return []
