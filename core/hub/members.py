from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hub member registry and per-member authentication state.

Members live in ``shared/hub/members/<id>.json``.  The registry only stores a
SHA-256 digest of each member's bearer token; the token itself is supplied by
the operator (``animaworks hub member set-token``) and never persisted.
"""

import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.platform.atomic_io import atomic_write_json
from core.platform.locks import locked_path
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

MEMBER_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
EXPECT_EVERY_RE = re.compile(r"^[1-9][0-9]*[mhd]$")
TOKEN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESERVED_IDS = frozenset({"all", "human"})
ROUTE_TYPES = frozenset({"webhook", "poll"})


@dataclass
class HubMember:
    """One external AI registered in the hub roster."""

    id: str
    display_name: str
    kind: str = "vendor_bot"
    company: str = ""
    enabled: bool = True
    token_sha256: str = ""
    message_route: dict[str, str] | None = None
    expect_every: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "company": self.company,
            "enabled": self.enabled,
            "token_sha256": self.token_sha256,
            "message_route": dict(self.message_route) if self.message_route is not None else None,
            "expect_every": self.expect_every,
        }


def members_dir(shared_dir: Path) -> Path:
    """Return the member registry directory."""
    return Path(shared_dir) / "hub" / "members"


def state_dir(shared_dir: Path) -> Path:
    """Return the per-member state directory."""
    return Path(shared_dir) / "hub" / "state"


def member_path(shared_dir: Path, member_id: str) -> Path:
    return members_dir(shared_dir) / f"{member_id}.json"


def state_path(shared_dir: Path, member_id: str) -> Path:
    return state_dir(shared_dir) / f"{member_id}.json"


def _as_non_empty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _validate_message_route(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("message_route must be an object or null")
    route_type = _as_non_empty_str(value.get("type"))
    if route_type not in ROUTE_TYPES:
        raise ValueError("message_route.type must be webhook or poll")
    route: dict[str, str] = {"type": route_type}
    if route_type == "webhook":
        url_ref = _as_non_empty_str(value.get("url_ref"))
        key_ref = _as_non_empty_str(value.get("key_ref"))
        if url_ref is None or key_ref is None:
            raise ValueError("webhook route requires url_ref and key_ref")
        route["url_ref"] = url_ref
        route["key_ref"] = key_ref
    else:
        for name in ("url_ref", "key_ref"):
            optional = _as_non_empty_str(value.get(name))
            if optional is not None:
                route[name] = optional
    return route


def validate_member(member: HubMember) -> None:
    """Validate one member record, raising ``ValueError`` when invalid."""
    if not MEMBER_ID_RE.fullmatch(member.id):
        raise ValueError(f"invalid member id: {member.id!r}")
    if not member.company or not member.company.strip():
        raise ValueError("member company must not be empty")
    if member.expect_every is not None and not EXPECT_EVERY_RE.fullmatch(member.expect_every):
        raise ValueError(f"invalid expect_every: {member.expect_every!r}")
    if member.token_sha256 and not TOKEN_SHA256_RE.fullmatch(member.token_sha256):
        raise ValueError("token_sha256 must be a lowercase hex SHA-256 digest")
    _validate_message_route(member.message_route)


def _parse_member(data: object) -> HubMember | None:
    if not isinstance(data, dict):
        raise ValueError("member record must be an object")
    member_id = _as_non_empty_str(data.get("id"))
    if member_id is None:
        raise ValueError("member id is required")
    display_name = _as_non_empty_str(data.get("display_name")) or member_id
    kind = _as_non_empty_str(data.get("kind")) or "vendor_bot"
    company = _as_non_empty_str(data.get("company")) or ""
    token_sha256 = _as_non_empty_str(data.get("token_sha256")) or ""
    expect_every = _as_non_empty_str(data.get("expect_every"))
    member = HubMember(
        id=member_id,
        display_name=display_name,
        kind=kind,
        company=company,
        enabled=bool(data.get("enabled", True)),
        token_sha256=token_sha256,
        message_route=_validate_message_route(data.get("message_route")),
        expect_every=expect_every,
    )
    validate_member(member)
    return member


def _anima_names(shared_dir: Path) -> set[str]:
    animas_dir = Path(shared_dir).parent / "animas"
    try:
        return {path.name for path in animas_dir.iterdir() if path.is_dir()}
    except OSError:
        return set()


def load_members(shared_dir: Path) -> dict[str, HubMember]:
    """Load every valid member, warning about skipped records."""
    directory = members_dir(shared_dir)
    if not directory.is_dir():
        return {}

    anima_names = _anima_names(shared_dir)
    members: dict[str, HubMember] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            member = _parse_member(data)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipping invalid hub member file %s: %s", path, exc)
            continue
        if member is None:  # pragma: no cover - defensive
            continue
        if member.id != path.stem:
            logger.warning("Skipping hub member file %s: id does not match filename", path)
            continue
        if member.id in RESERVED_IDS or member.id in anima_names:
            logger.warning("Skipping hub member %s: id is reserved or an anima name", member.id)
            continue
        members[member.id] = member
    return members


def find_member_by_token(shared_dir: Path, token: str) -> HubMember | None:
    """Return the enabled member whose token digest matches *token*."""
    if not isinstance(token, str) or not token:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    matched: HubMember | None = None
    for member in load_members(shared_dir).values():
        if not member.enabled or not member.token_sha256:
            continue
        if hmac.compare_digest(member.token_sha256, digest):
            matched = member
    return matched


def save_member(shared_dir: Path, member: HubMember) -> None:
    """Persist one member record atomically."""
    validate_member(member)
    atomic_write_json(member_path(shared_dir, member.id), member.to_dict())


def _read_state(shared_dir: Path, member_id: str) -> dict[str, Any]:
    path = state_path(shared_dir, member_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def member_state(shared_dir: Path, member_id: str) -> dict[str, Any]:
    """Return the member's persisted state object (never raises on bad files)."""
    return _read_state(shared_dir, member_id)


def update_member_state(shared_dir: Path, member_id: str, **updates: Any) -> dict[str, Any]:
    """Merge *updates* into a member's state file under a lock.

    The MCP surface calls :func:`touch_last_seen` on every request, so this
    keeps unrelated keys (such as ``silence_alerted_for``) intact while doing a
    single short read/modify/write.
    """
    path = state_path(shared_dir, member_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with locked_path(lock_path, exclusive=True, thread_lock=True):
        state = _read_state(shared_dir, member_id)
        state.update(updates)
        atomic_write_json(path, state, indent=2)
    return state


def touch_last_seen(shared_dir: Path, member_id: str) -> None:
    """Record that *member_id* reached the MCP surface just now."""
    update_member_state(shared_dir, member_id, last_seen=now_iso())


def last_seen(shared_dir: Path, member_id: str) -> str | None:
    """Return the last-seen timestamp for one member, if recorded."""
    value = _read_state(shared_dir, member_id).get("last_seen")
    return value if isinstance(value, str) and value else None


def token_digest(token: str) -> str:
    """Return the SHA-256 hex digest stored for a bearer token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_from_env(var_name: str) -> str | None:
    """Read a token from an environment variable without logging its value."""
    value = os.environ.get(var_name)
    if value is None or value == "":
        return None
    return value


def expect_every_seconds(value: str | None) -> int | None:
    """Convert an ``expect_every`` value (``15m`` / ``1h`` / ``2d``) to seconds."""
    if not isinstance(value, str) or not EXPECT_EVERY_RE.fullmatch(value):
        return None
    amount = int(value[:-1])
    unit = value[-1]
    multiplier = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60}[unit]
    return amount * multiplier


def parse_iso(value: str) -> datetime | None:
    """Parse an ISO timestamp, returning an aware UTC datetime when possible."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
