from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the setup instructions for one external hub AI."""

from pathlib import Path
from typing import Any

from core.config.anima_registry import read_anima_company_checked, read_anima_supervisor
from core.config.models import load_config
from core.hub.mcp_app import MAX_POST_CHARS
from core.hub.members import HubMember, load_members
from core.i18n import t
from core.messaging.board import anima_principal, board_access
from core.org.company import get_company_display_name
from core.paths import load_prompt
from core.platform.status_store import read_status_checked


class SetupPromptError(ValueError):
    """A setup prompt could not be built from the roster or configuration."""


def _interval_label(expect_every: str | None) -> str:
    if not expect_every:
        return t("hub.setup_interval_default")
    number = int(expect_every[:-1])
    unit = expect_every[-1]
    if unit == "m":
        return t("hub.setup_interval_minutes", n=number)
    if unit == "h":
        return t("hub.setup_interval_hours", n=number)
    return t("hub.setup_interval_days", n=number)


def _roster(shared_dir: Path, company: str, board: str) -> list[tuple[str, str, str | None]]:
    """Return enabled, authorized animas as (name, role, supervisor)."""
    animas_dir = Path(shared_dir).parent / "animas"
    if not animas_dir.is_dir():
        return []
    company = str(company or "").strip()
    entries: list[tuple[str, str, str | None]] = []
    for anima_dir in sorted(path for path in animas_dir.iterdir() if path.is_dir()):
        readable, status = read_status_checked(anima_dir)
        if not readable or status.get("enabled") is not True:
            continue
        company_known, anima_company = read_anima_company_checked(anima_dir)
        if not company_known or anima_company != company:
            continue
        if not board_access(
            shared_dir, board, anima_principal(anima_dir.name, animas_dir=animas_dir), "mention_target"
        ).allowed:
            continue
        role = str(status.get("speciality") or status.get("role") or "-").strip() or "-"
        entries.append((anima_dir.name, role, read_anima_supervisor(anima_dir)))
    return entries


def _first_contact(entries: list[tuple[str, str, str | None]]) -> str:
    heads = [name for name, _role, supervisor in entries if not supervisor]
    if heads:
        return heads[0]
    if entries:
        return entries[0][0]
    return "human"


def _connection_url(hub_config: Any, url: str | None) -> str:
    if url is not None and url.strip():
        return url.strip()
    public_hosts = [str(host).strip() for host in getattr(hub_config, "public_hosts", []) or [] if str(host).strip()]
    if public_hosts:
        return f"https://{public_hosts[0]}/hub/mcp"
    raise SetupPromptError(t("cli.hub_setup_url_required"))


def build_setup_prompt(shared_dir: Path, member_id: str, *, url: str | None = None) -> str:
    """Return the Markdown setup document for *member_id*.

    The document never contains the member's token or token digest; only the
    operator holds the token and pastes it into the connector.
    """
    members = load_members(shared_dir)
    member: HubMember | None = members.get(member_id)
    if member is None:
        raise SetupPromptError(t("cli.hub_member_not_found", id=member_id))
    if not member.enabled:
        raise SetupPromptError(t("cli.hub_setup_member_disabled", id=member_id))

    try:
        hub_config = load_config().hub
    except Exception as exc:
        raise SetupPromptError(t("cli.hub_setup_config_error")) from exc

    board = str(getattr(hub_config, "board", "hub") or "hub")
    company = str(getattr(hub_config, "company", "") or "")
    company_name = get_company_display_name(company, data_dir=Path(shared_dir).parent) if company else ""
    connection_url = _connection_url(hub_config, url)

    entries = _roster(shared_dir, company, board)
    roster_lines = "\n".join(f"- {name} … {role}" for name, role, _supervisor in entries)
    if not roster_lines:
        roster_lines = t("hub.setup_roster_empty")
    first_contact = _first_contact(entries)

    max_posts = int(getattr(hub_config, "max_posts_per_hour", 20))
    max_round_trips = int(getattr(hub_config, "max_round_trips", 4))
    values = {
        "member_id": member_id,
        "display_name": member.display_name,
        "company_name": company_name or company,
        "url": connection_url,
        "board": board,
        "roster": roster_lines,
        "first_contact": first_contact,
        "max_posts_per_hour": max_posts,
        "max_post_chars": MAX_POST_CHARS,
        "max_round_trips": max_round_trips,
    }

    common = load_prompt("hub/setup_common", **values).strip()
    connector = load_prompt("hub/setup_connector", **values).strip()
    check = load_prompt("hub/setup_check", **values).strip()
    avatar = load_prompt("hub/setup_avatar", **values).strip()

    route = member.message_route or {}
    route_type = route.get("type", "poll")
    if route_type == "webhook":
        routine = load_prompt("hub/setup_webhook_routine", **values).strip()
        wake = load_prompt(
            "hub/setup_webhook",
            **values,
            url_ref=str(route.get("url_ref", "")),
            key_ref=str(route.get("key_ref", "")),
            routine=routine,
        ).strip()
    else:
        interval_label = _interval_label(member.expect_every)
        routine = load_prompt("hub/setup_poll_routine", **values, interval_label=interval_label).strip()
        wake = load_prompt("hub/setup_poll", **values, interval_label=interval_label, routine=routine).strip()

    text = load_prompt(
        "hub/setup_prompt",
        **values,
        common=common,
        connector=connector,
        wake=wake,
        check=check,
        avatar=avatar,
    )
    return text.strip() + "\n"
