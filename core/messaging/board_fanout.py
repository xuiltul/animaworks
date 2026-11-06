from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Mention delivery for committed Board posts."""

import logging
from pathlib import Path

from core.i18n import t
from core.messaging.board import BoardPost, anima_principal, board_access
from core.messaging.mentions import extract_mentions
from core.platform.status_store import read_status_checked

logger = logging.getLogger("animaworks.messaging.board_fanout")


def _read_enabled(anima_dir: Path) -> tuple[bool, str]:
    """Return the strict status.json enabled state and a skip reason."""
    readable, data = read_status_checked(anima_dir)
    if not readable:
        return False, "status_unreadable"
    if data.get("enabled") is not True:
        return False, "disabled"
    return True, ""


def fanout_board_mentions(shared_dir: Path, post: BoardPost) -> tuple[list[str], dict[str, str]]:
    """Send board_mention messages to mentioned, enabled, authorized Animas."""
    animas_dir = shared_dir.parent / "animas"
    if not animas_dir.is_dir():
        return [], {}
    anima_names = sorted(path.name for path in animas_dir.iterdir() if path.is_dir())
    mentions = extract_mentions(post.text, anima_names)
    if not mentions:
        return [], {}

    if "all" in mentions:
        targets = set(anima_names)
    else:
        targets = mentions - {"all"}
    if post.principal.kind == "anima":
        targets.discard(post.from_person)

    notified: list[str] = []
    not_notified: dict[str, str] = {}
    from core.messaging.messenger import Messenger

    fanout_content = f"[board_reply:channel={post.channel},from={post.from_person}]\n" + t(
        "handler.board_mention_content",
        from_name=post.from_person,
        channel=post.channel,
        text=post.text,
    )
    messenger = Messenger(shared_dir, post.from_person)
    for target in sorted(targets):
        target_dir = animas_dir / target
        enabled, reason = _read_enabled(target_dir)
        if not enabled:
            not_notified[target] = reason
            continue

        # Mention targets are always Animas. Build their company-aware principal
        # only after the status file has passed the fail-closed enabled check.
        target_principal = anima_principal(target, animas_dir=animas_dir)
        access = board_access(shared_dir, post.channel, target_principal, "mention_target")
        if not access.allowed:
            not_notified[target] = access.reason
            continue

        if post.principal.kind == "anima":
            try:
                from core.org.company import check_company_boundary

                boundary = check_company_boundary(post.from_person, target, animas_dir=animas_dir)
            except Exception:
                logger.warning("Could not verify mention company boundary for %s", target, exc_info=True)
                not_notified[target] = "company_unreadable"
                continue
            if boundary.cross_company:
                not_notified[target] = (
                    "company_unreadable" if boundary.resolved_via == "fail_closed" else "cross_company"
                )
                continue

        try:
            messenger.send(
                to=target,
                content=fanout_content,
                msg_type="board_mention",
                meta={"board_channel": post.channel, "board_post_id": post.id},
                from_person=post.from_person,
                source=post.source,
                skip_logging=True,
            )
            notified.append(target)
        except Exception:
            logger.warning("Failed to fanout board_mention to %s", target, exc_info=True)
            not_notified[target] = "delivery_failed"
    return notified, not_notified
