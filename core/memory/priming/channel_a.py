from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel A: Sender profile lookup."""

import logging
from pathlib import Path

from core.tools._async_compat import run_sync

logger = logging.getLogger("animaworks.priming")


async def channel_a_sender_profile(anima_dir: Path, sender_name: str) -> str:
    """Channel A: Direct sender profile lookup.

    Reads shared/users/{sender_name}/index.md if it exists.
    """
    from core.paths import get_shared_dir

    shared_users_dir = get_shared_dir() / "users"
    profile_path = (shared_users_dir / sender_name / "index.md").resolve()
    if not profile_path.is_relative_to(shared_users_dir.resolve()):
        logger.warning("Channel A: path traversal in sender_name=%s", sender_name)
        return ""

    if not profile_path.exists():
        logger.debug("Channel A: No profile found for sender=%s", sender_name)
        return ""

    try:
        content = await run_sync(profile_path.read_text, encoding="utf-8")
        logger.debug(
            "Channel A: Loaded sender profile for %s (%d chars)",
            sender_name,
            len(content),
        )
        return content
    except Exception as e:
        logger.warning("Channel A: Failed to read profile for %s: %s", sender_name, e)
        return ""
