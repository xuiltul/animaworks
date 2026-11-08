from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Mention delivery for committed Board posts."""

import logging
from pathlib import Path

from core.config.anima_registry import read_anima_company_checked
from core.hub.members import HubMember, load_members
from core.hub.wake import (
    WebhookSender,
    alert_loop_limit_once,
    count_round_trips,
    schedule_wake,
)
from core.i18n import t
from core.messaging.board import BoardPost, anima_principal, board_access, hub_principal
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


def _hub_settings() -> tuple[str, int]:
    """Return the configured hub board and round-trip limit."""
    try:
        from core.config.models import load_config

        hub = load_config().hub
    except Exception:
        logger.warning("Could not read hub settings for Board fanout", exc_info=True)
        return "hub", 4
    board = str(getattr(hub, "board", "hub") or "hub")
    try:
        max_round_trips = int(getattr(hub, "max_round_trips", 4))
    except (TypeError, ValueError):
        max_round_trips = 4
    return board, max_round_trips


def _hub_members(shared_dir: Path, post: BoardPost, hub_board: str) -> dict[str, HubMember]:
    if post.channel != hub_board:
        return {}
    try:
        return {member.id: member for member in load_members(shared_dir).values() if member.enabled}
    except Exception:
        logger.warning("Could not read hub members for Board fanout", exc_info=True)
        return {}


def fanout_board_mentions(
    shared_dir: Path,
    post: BoardPost,
    *,
    wake_sender: WebhookSender | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Send board_mention messages to mentioned, enabled, authorized targets.

    On the hub board, registered external members are wake targets as well.
    External members are only woken by an explicit ``@id``; ``@all`` from an
    external member wakes nobody, and an anima's ``@all`` never wakes external
    members.
    """
    animas_dir = shared_dir.parent / "animas"
    anima_names = sorted(path.name for path in animas_dir.iterdir() if path.is_dir()) if animas_dir.is_dir() else []

    hub_board, max_round_trips = _hub_settings()
    on_hub_board = post.channel == hub_board
    hub_members = _hub_members(shared_dir, post, hub_board)
    known_names = set(anima_names) | set(hub_members)
    mentions = extract_mentions(post.text, known_names)
    if not mentions:
        return [], {}

    if on_hub_board and "all" in mentions and post.principal.kind == "hub":
        return [], {}
    if "all" in mentions:
        targets = set(anima_names)
        if on_hub_board and post.principal.kind == "anima":
            # Anima @all never expands to external members, but an explicit
            # @id in the same post is still a deliberate wake.
            targets |= set(hub_members) & (mentions - {"all"})
    else:
        targets = mentions - {"all"}
    if on_hub_board or post.principal.kind == "anima":
        targets.discard(post.from_person)
    if not targets:
        return [], {}

    loop_limit = False
    if on_hub_board:
        transitions, top_id = count_round_trips(shared_dir, post.channel, post)
        if transitions > 2 * max_round_trips:
            loop_limit = True
            try:
                alert_loop_limit_once(shared_dir, post.channel, post, top_id)
            except Exception:
                logger.warning("Could not record the hub loop-limit alert", exc_info=True)

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
        if loop_limit:
            not_notified[target] = "loop_limit"
            continue

        if target in anima_names:
            target_dir = animas_dir / target
            enabled, reason = _read_enabled(target_dir)
            if not enabled:
                not_notified[target] = reason
                continue

            target_principal = anima_principal(target, animas_dir=animas_dir)
            access = board_access(shared_dir, post.channel, target_principal, "mention_target")
            if not access.allowed:
                reason = access.reason
                if post.principal.kind == "hub" and reason in {"company_mismatch", "company_assigned_legacy"}:
                    reason = "cross_company"
                not_notified[target] = reason
                continue

            if post.principal.kind == "hub":
                readable, target_company = read_anima_company_checked(target_dir)
                if not readable:
                    not_notified[target] = "company_unreadable"
                    continue
                if target_company != post.principal.company:
                    not_notified[target] = "cross_company"
                    continue
            elif post.principal.kind == "anima":
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
            continue

        member = hub_members.get(target)
        if member is None:
            continue
        access = board_access(shared_dir, post.channel, hub_principal(member), "mention_target")
        if not access.allowed:
            not_notified[target] = access.reason
            continue

        route_type = (member.message_route or {}).get("type")
        if route_type == "poll":
            not_notified[target] = "polls"
        elif route_type == "webhook":
            schedule_wake(shared_dir, post.channel, post, member, sender=wake_sender)
        else:
            not_notified[target] = "bad_route"

    return notified, not_notified
