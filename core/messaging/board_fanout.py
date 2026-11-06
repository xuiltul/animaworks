from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared @mention fan-out for Board posts."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t

if TYPE_CHECKING:
    from core.messaging.messenger import Messenger

logger = logging.getLogger("animaworks.messaging.board_fanout")


def fanout_board_mentions(
    messenger: Messenger,
    from_anima: str,
    channel: str,
    text: str,
    *,
    origin_chain: list[str] | None = None,
    animas_dir: Path | None = None,
) -> None:
    """Send ``board_mention`` DMs to running, authorized Animas mentioned in *text*.

    ToolHandler and the human/operator CLI path both call this function so the
    mention syntax, running-process check, channel ACL, company boundary, and
    delivery behavior cannot drift between entry points.
    """
    from core.paths import get_data_dir

    sockets_dir = get_data_dir() / "run" / "sockets"
    running = {path.stem for path in sockets_dir.glob("*.sock")} if sockets_dir.exists() else set()
    from core.messaging.mentions import extract_mentions

    mentions = extract_mentions(text, running)
    if not mentions:
        return

    if "all" in mentions:
        targets = running - {from_anima}
    else:
        named = {mention for mention in mentions if mention != "all"}
        targets = (named & running) - {from_anima}

    from core.messaging.messenger import is_channel_member

    targets = {target for target in targets if is_channel_member(messenger.shared_dir, channel, target)}

    # A human operator has no Anima company membership. For Anima posts, use
    # the same fail-closed company check as send_message and the channel tools.
    if from_anima != "human" and targets:
        from core.org.company import check_company_boundary
        from core.paths import get_animas_dir

        company_animas_dir = animas_dir or get_animas_dir()
        targets = {
            target
            for target in targets
            if not check_company_boundary(from_anima, target, animas_dir=company_animas_dir).cross_company
        }

    if not targets:
        return

    fanout_content = f"[board_reply:channel={channel},from={from_anima}]\n" + t(
        "handler.board_mention_content",
        from_name=from_anima,
        channel=channel,
        text=text,
    )

    for target in sorted(targets):
        try:
            kwargs: dict[str, Any] = {
                "to": target,
                "content": fanout_content,
                "msg_type": "board_mention",
            }
            if origin_chain is not None:
                kwargs["origin_chain"] = origin_chain
            messenger.send(**kwargs)
            logger.info(
                "board_mention fanout: %s -> %s (channel=%s)",
                from_anima,
                target,
                channel,
            )
        except Exception:
            logger.warning("Failed to fanout board_mention to %s", target, exc_info=True)
