# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from core.hub.members import (
    HubMember,
    last_seen,
    load_members,
    save_member,
    token_digest,
    token_from_env,
)
from core.paths import get_shared_dir


def _shared_dir():
    return get_shared_dir()


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _load_or_fail(member_id: str) -> tuple[HubMember, dict[str, HubMember]]:
    from core.i18n import t

    members = load_members(_shared_dir())
    member = members.get(member_id)
    if member is None:
        _fail(t("cli.hub_member_not_found", id=member_id))
    return member, members


def cmd_hub_member_add(args) -> None:
    """Add an external AI to the hub roster."""
    from core.i18n import t

    shared_dir = _shared_dir()
    members = load_members(shared_dir)
    if args.member_id in members:
        _fail(t("cli.hub_member_exists", id=args.member_id))

    route_type = args.route
    message_route: dict[str, str] | None = None
    if route_type == "webhook":
        if not args.url_ref or not args.key_ref:
            _fail(t("cli.hub_member_webhook_refs"))
        message_route = {
            "type": "webhook",
            "url_ref": args.url_ref,
            "key_ref": args.key_ref,
        }
    else:
        message_route = {"type": "poll"}

    member = HubMember(
        id=args.member_id,
        display_name=args.display_name or args.member_id,
        kind=args.kind,
        company=args.company,
        enabled=True,
        token_sha256="",
        message_route=message_route,
        expect_every=args.expect_every,
    )
    try:
        save_member(shared_dir, member)
    except ValueError as exc:
        _fail(t("cli.hub_member_invalid", detail=str(exc)))
    print(t("cli.hub_member_added", id=member.id))


def cmd_hub_member_list(args) -> None:
    """Print the hub roster without exposing token material."""
    from core.i18n import t

    shared_dir = _shared_dir()
    members = load_members(shared_dir)
    if not members:
        print(t("cli.hub_member_list_empty"))
        return
    for member_id in sorted(members):
        member = members[member_id]
        route = member.message_route.get("type", "-") if member.message_route else "-"
        seen = last_seen(shared_dir, member_id) or "-"
        status = "enabled" if member.enabled else "disabled"
        print(
            "\t".join(
                (
                    member.id,
                    member.display_name,
                    member.company,
                    member.kind,
                    status,
                    route,
                    member.expect_every or "-",
                    seen,
                )
            )
        )


def cmd_hub_member_set_token(args) -> None:
    """Store the SHA-256 digest of a bearer token read from the environment."""
    from core.i18n import t

    token = token_from_env(args.from_env)
    if token is None:
        _fail(t("cli.hub_member_token_env_missing", var=args.from_env))
    member, _members = _load_or_fail(args.member_id)
    member.token_sha256 = token_digest(token)
    try:
        save_member(_shared_dir(), member)
    except ValueError as exc:
        _fail(t("cli.hub_member_invalid", detail=str(exc)))
    print(t("cli.hub_member_token_set", id=member.id))


def cmd_hub_member_enable(args) -> None:
    """Enable a member so its token can authenticate."""
    from core.i18n import t

    member, _members = _load_or_fail(args.member_id)
    member.enabled = True
    save_member(_shared_dir(), member)
    print(t("cli.hub_member_enabled", id=member.id))


def cmd_hub_member_disable(args) -> None:
    """Disable a member without deleting its token digest."""
    from core.i18n import t

    member, _members = _load_or_fail(args.member_id)
    member.enabled = False
    save_member(_shared_dir(), member)
    print(t("cli.hub_member_disabled", id=member.id))
