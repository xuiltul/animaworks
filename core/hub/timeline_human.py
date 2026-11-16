from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect humans and their replies for the company timeline.

The company timeline keeps three human-facing kinds:

* ``human`` — something a human said to an Anima.
* ``answered`` — an Anima's answer to one of those human messages.
* ``notified`` — the notification the human replied to (written elsewhere).

This module is the single place that decides whether an actor is the human
principal.  Every entry point records best-effort: a timeline problem must
never stop message delivery, replies, or approvals.
"""

import logging
from typing import Any

from core.config import load_config
from core.config.anima_registry import read_anima_company_checked
from core.hub.members import load_members
from core.hub.timeline import append_item, find_latest
from core.paths import get_animas_dir, get_shared_dir

logger = logging.getLogger("animaworks.hub.timeline_human")

_TUI_CLIENT = "tui"


def aliased_slack_users() -> dict[str, str]:
    """Return Slack user id -> configured alias."""
    try:
        aliases = load_config().external_messaging.user_aliases
        return {alias.slack_user_id: name for name, alias in aliases.items() if getattr(alias, "slack_user_id", "")}
    except Exception:
        logger.warning("Failed to load Slack user aliases", exc_info=True)
        return {}


def aliased_discord_users() -> dict[str, str]:
    """Return Discord user id -> configured alias."""
    try:
        aliases = load_config().external_messaging.user_aliases
        return {alias.discord_user_id: name for name, alias in aliases.items() if getattr(alias, "discord_user_id", "")}
    except Exception:
        logger.warning("Failed to load Discord user aliases", exc_info=True)
        return {}


def resolve_person(
    *,
    via: str,
    auth_via: str | None = None,
    username: str | None = None,
    client_header: str | None = None,
    external_user_id: str | None = None,
    actor: str | None = None,
    anima: str | None = None,
) -> str | None:
    """Return the human principal for an entry point, or ``None``.

    * ``via="web"``/``"tui"``: only an authenticated web session, or a
      localhost request carrying the TUI header, is the human.
    * ``via="slack"``/``"discord"``: only an id present in ``user_aliases``.
    * ``via="approval"``/``"notification_reply"``/``"board"``: the actor is
      accepted unless it is an Anima or a hub member.
    """
    if via in {"web", "tui"}:
        if auth_via == "session" and username:
            return username
        if auth_via == "localhost" and client_header == _TUI_CLIENT and username:
            return username
        return None

    if via == "slack":
        if not external_user_id:
            return None
        return aliased_slack_users().get(str(external_user_id))

    if via == "discord":
        if not external_user_id:
            return None
        return aliased_discord_users().get(str(external_user_id))

    if via in {"approval", "notification_reply", "board", "actor"}:
        return _person_from_actor(actor, anima=anima)

    return None


def _person_from_actor(actor: str | None, *, anima: str | None = None) -> str | None:
    name = str(actor or "").strip()
    if not name:
        return None
    if name.lower() in _non_human_names():
        return None
    if anima and name.lower() == anima.lower():
        return None
    return name


def _non_human_names() -> set[str]:
    names: set[str] = {"system", "anima", "hub"}
    try:
        animas_dir = get_animas_dir()
        if animas_dir.is_dir():
            names.update(path.name.lower() for path in animas_dir.iterdir() if path.is_dir())
    except Exception:
        logger.warning("Failed to list Animas while resolving the human principal", exc_info=True)
    try:
        names.update(member.id.lower() for member in load_members(get_shared_dir()).values())
    except Exception:
        logger.warning("Failed to list hub members while resolving the human principal", exc_info=True)
    return names


def _hub_company() -> str | None:
    try:
        hub = load_config().hub
        if not hub.timeline_enabled:
            return None
        return str(hub.company or "").strip() or None
    except Exception:
        logger.warning("Failed to read hub timeline configuration", exc_info=True)
        return None


def _company_for_anima(anima: str) -> str | None:
    company = _hub_company()
    if company is None:
        return None
    if not anima:
        return company
    try:
        readable, anima_company = read_anima_company_checked(get_animas_dir() / anima)
    except Exception:
        logger.warning("Failed to read company for Anima %s", anima, exc_info=True)
        return None
    if not readable or anima_company != company:
        return None
    return company


def record_human(
    *,
    person: str | None,
    anima: str = "",
    text: str = "",
    via: str = "board",
    about: str | None = None,
    ref: str | None = None,
    label: str | None = None,
    to: str | None = None,
) -> str | None:
    """Append a ``human`` timeline item and return its id (best effort)."""
    try:
        if not person or not str(text or "").strip():
            return None
        company = _company_for_anima(anima)
        if company is None:
            return None
        item = append_item(
            get_shared_dir(),
            from_name=person,
            company=company,
            kind="human",
            text=text,
            source="human",
            about=about,
            ref=ref,
            label=label,
            to=to if to is not None else (anima or None),
            via=via,
        )
        return str(item.get("id") or "") or None
    except Exception:
        logger.warning(
            "Failed to record human timeline item (anima=%s person=%s via=%s)",
            anima,
            person,
            via,
            exc_info=True,
        )
        return None


def record_answered(
    *,
    anima: str,
    person: str | None,
    text: str,
    via: str | None,
    about: str | None,
) -> str | None:
    """Append an ``answered`` timeline item tied to the human item (best effort)."""
    try:
        if not person or not str(text or "").strip():
            return None
        company = _company_for_anima(anima)
        if company is None:
            return None
        item = append_item(
            get_shared_dir(),
            from_name=anima,
            company=company,
            kind="answered",
            text=text,
            source="anima",
            about=about,
            to=person,
            via=via,
        )
        return str(item.get("id") or "") or None
    except Exception:
        logger.warning(
            "Failed to record answered timeline item (anima=%s about=%s via=%s)",
            anima,
            about,
            via,
            exc_info=True,
        )
        return None


def find_notified(*, key: str | None = None, callback_id: str | None = None) -> str | None:
    """Return the matching ``notified`` item id, if any."""
    if not key and not callback_id:
        return None
    try:
        company = _hub_company()
        if company is None:
            return None
        item = find_latest(
            get_shared_dir(),
            company=company,
            kind="notified",
            key=key or None,
            callback_id=callback_id or None,
        )
        return str(item.get("id") or "") if item else None
    except Exception:
        logger.warning(
            "Failed to find notified timeline item (key=%s callback_id=%s)",
            key,
            callback_id,
            exc_info=True,
        )
        return None


def find_human_item(ref: str | None) -> dict[str, Any] | None:
    """Return the ``human`` item whose ``ref`` matches, if any."""
    if not ref:
        return None
    try:
        company = _hub_company()
        if company is None:
            return None
        return find_latest(
            get_shared_dir(),
            company=company,
            kind="human",
            ref=ref,
        )
    except Exception:
        logger.warning("Failed to find human timeline item for ref=%s", ref, exc_info=True)
        return None


def mentioned_animas(text: str) -> str | None:
    """Return comma-separated Anima names mentioned in *text*, or ``None``."""
    try:
        from core.messaging.mentions import extract_mentions

        animas_dir = get_animas_dir()
        names = sorted(path.name for path in animas_dir.iterdir() if path.is_dir()) if animas_dir.is_dir() else []
        mentions = extract_mentions(text or "", names) - {"all"}
        return ", ".join(sorted(mentions)) or None
    except Exception:
        logger.warning("Failed to extract Anima mentions from board post", exc_info=True)
        return None


__all__ = [
    "aliased_discord_users",
    "aliased_slack_users",
    "find_human_item",
    "find_notified",
    "mentioned_animas",
    "record_answered",
    "record_human",
    "resolve_person",
]
