from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""The public MCP surface for external hub AIs.

The surface exposes exactly one tool (``hub``) over a stateless Streamable
HTTP endpoint at ``/hub/mcp``.  Authentication is a member bearer token, not
the web session cookie, so a mounted MCP app has its own ASGI middleware.
"""

import asyncio
import contextvars
import json
import logging
import shlex
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from core.config.models import load_config
from core.hub import timeline
from core.hub.avatar import AvatarError, fetch_avatar_url, normalize_avatar, save_avatar
from core.hub.members import (
    HubMember,
    find_member_by_token,
    member_state,
    parse_iso,
    touch_last_seen,
    update_member_state,
)
from core.i18n import t
from core.messaging.board import hub_principal, post_to_board
from core.messaging.board_reads import (
    advance_upto,
    entries_after_ts,
    is_after_ts,
    load_reads,
    read_board_entries,
)
from core.messaging.mentions import extract_mentions
from core.time_utils import now_iso, now_local

logger = logging.getLogger(__name__)

TOOL_DESCRIPTION = (
    "One tool for the AnimaWorks hub. Commands: `help`, `timeline`, `read`, "
    "`post [--reply-to ID] [--request-id ID]` (put the message body in `text`), "
    "`avatar <https image URL>` (register your avatar). "
    "Talk to animas only by posting in #hub. Learn the current situation from "
    "`timeline`. Before notifying the human directly, check `timeline`; do not "
    "send what is already there (especially items the human has already read). "
    "Except for urgent matters, do not message the human directly; post in #hub "
    "instead."
)

HELP_SUMMARY = "commands: help | timeline | read | post [--reply-to ID] [--request-id ID] | avatar <https image URL>"

HELP_TEXT = (
    "AnimaWorks hub tool\n"
    f"{HELP_SUMMARY}\n"
    "- timeline: the last 24 hours of the company timeline (read-only, newest first).\n"
    "- read: new #hub posts since your last read (oldest first, mentions first).\n"
    "- post: post to #hub; put the message body in `text`.\n"
    "- avatar: register a square avatar from an https image URL (PNG/JPEG/WebP/GIF, 5MB).\n"
    "Rules: talk to animas only in #hub. Check timeline before notifying the human and do not repeat "
    "what is already there. Post in #hub for non-urgent matters.\n" + t("hub.tool.help_bilingual_ja")
)

_MAX_TIMELINE_LINES = 50
_MAX_READ_POSTS = 30
MAX_POST_CHARS = 4000
_TIMELINE_WINDOW = timedelta(hours=24)
_AVATAR_UPDATES_PER_DAY = 5
_AVATAR_UPDATE_WINDOW = timedelta(hours=24)

_current_member: contextvars.ContextVar[HubMember | None] = contextvars.ContextVar(
    "hub_mcp_member",
    default=None,
)


def _error(message: str) -> str:
    return f"error: {message}\n{HELP_SUMMARY}"


def _one_line(value: object) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ")


def _timeline_line(item: dict[str, Any], *, is_new: bool, human_read: bool) -> str:
    prefix = "[新] " if is_new else ""
    about = f" about={item['about']}" if item.get("about") else ""
    return (
        f"{prefix}{item.get('id', '?')} {item.get('ts', '?')} "
        f"from={item.get('from', '?')} kind={item.get('kind', '?')} "
        f"human_read={'yes' if human_read else 'no'}{about}: {_one_line(item.get('text'))}"
    )


def _read_line(entry: dict[str, Any], *, to_me: bool) -> str:
    prefix = "[to you] " if to_me else ""
    return (
        f"{prefix}{entry.get('id', '?')} {entry.get('ts', '?')} "
        f"from={entry.get('from', '?')} kind={entry.get('kind', '?')}: {_one_line(entry.get('text'))}"
    )


def _run_timeline(shared_dir: Path, member: HubMember) -> tuple[str, str]:
    config = load_config().hub
    if not config.timeline_enabled:
        return t("hub.tool.timeline_disabled"), "ok"
    since = (now_local() - _TIMELINE_WINDOW).isoformat()
    items = timeline.list_items(
        shared_dir,
        company=str(config.company or ""),
        since=since,
        limit=_MAX_TIMELINE_LINES,
    )
    if not items:
        return t("hub.tool.timeline_empty"), "ok"

    reads = timeline.load_timeline_reads(shared_dir)
    reader = f"hub:{member.id}"
    record = reads.get("readers", {}).get(reader)
    previous_ts = str(record.get("upto_ts") or "") if isinstance(record, dict) else ""
    lines = []
    for item in items:
        is_new = is_after_ts(item.get("ts", ""), previous_ts)
        human_read = timeline.is_read_by_human(item, reads)
        lines.append(_timeline_line(item, is_new=is_new, human_read=human_read))

    newest = items[0]
    if isinstance(newest.get("id"), str):
        timeline.advance_timeline_read(shared_dir, reader, newest["id"])
    return "\n".join(lines), "ok"


def _run_read(shared_dir: Path, member: HubMember) -> tuple[str, str]:
    config = load_config().hub
    board = str(config.board or "hub")
    entries = read_board_entries(shared_dir, board)
    reads = load_reads(shared_dir, board)
    reader = f"hub:{member.id}"
    record = reads.get("readers", {}).get(reader)
    previous_ts = str(record.get("upto_ts") or "") if isinstance(record, dict) else ""

    fresh = [entry for entry in entries_after_ts(entries, previous_ts) if entry.get("from") != member.id]
    fresh = fresh[:_MAX_READ_POSTS]
    if not fresh:
        return t("hub.tool.no_new_posts", now=now_iso()), "ok"

    mentions = [entry for entry in fresh if extract_mentions(str(entry.get("text") or ""), {member.id})]
    mention_ids = {id(entry) for entry in mentions}
    ordered = mentions + [entry for entry in fresh if id(entry) not in mention_ids]
    lines = [_read_line(entry, to_me=id(entry) in mention_ids) for entry in ordered]

    last = fresh[-1]
    last_id = last.get("id")
    last_ts = last.get("ts")
    if isinstance(last_id, str) and isinstance(last_ts, str):
        advance_upto(shared_dir, board, reader, post_id=last_id, post_ts=last_ts)
    return "\n".join(lines), "ok"


def _parse_post_args(tokens: list[str]) -> tuple[str | None, str | None]:
    reply_to: str | None = None
    request_id: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in {"--reply-to", "--request-id"}:
            raise ValueError(f"unexpected argument: {token}")
        if index + 1 >= len(tokens):
            raise ValueError(f"{token} requires a value")
        value = tokens[index + 1]
        if not value:
            raise ValueError(f"{token} requires a value")
        if token == "--reply-to":
            reply_to = value
        else:
            request_id = value
        index += 2
    return reply_to, request_id


def _post_limit_reached(shared_dir: Path, board: str, member: HubMember, limit: int) -> bool:
    if limit <= 0:
        return False
    cutoff = (now_local() - timedelta(hours=1)).isoformat()
    count = 0
    for entry in read_board_entries(shared_dir, board):
        if entry.get("from") != member.id:
            continue
        if is_after_ts(entry.get("ts", ""), cutoff):
            count += 1
    return count >= limit


def _run_post(shared_dir: Path, member: HubMember, tokens: list[str], text: str) -> tuple[str, str]:
    try:
        reply_to, request_id = _parse_post_args(tokens)
    except ValueError as exc:
        return _error(str(exc)), "invalid_argument"

    body = str(text or "")
    if not body.strip():
        return _error("post requires text"), "invalid_argument"
    if len(body) > MAX_POST_CHARS:
        return _error(f"post text must be at most {MAX_POST_CHARS} characters"), "invalid_argument"

    config = load_config().hub
    board = str(config.board or "hub")
    if reply_to is not None:
        if not any(entry.get("id") == reply_to for entry in read_board_entries(shared_dir, board)):
            return _error(f"reply target not found in #{board}: {reply_to}"), "invalid_argument"

    if _post_limit_reached(shared_dir, board, member, int(config.max_posts_per_hour)):
        return _error(f"post limit reached ({config.max_posts_per_hour} per hour)"), "rate_limited"

    result = post_to_board(
        board,
        body,
        hub_principal(member),
        request_id=request_id,
        origin="hub",
        reply_to=reply_to,
        shared_dir=shared_dir,
    )
    if result.error:
        return _error(result.error), "post_failed"
    if result.duplicate:
        return f"duplicate post: {result.id}", "ok"

    lines = [f"posted {result.id}"]
    if result.notified:
        lines.append("notified: " + ", ".join(result.notified))
    if result.not_notified:
        reasons = ", ".join(f"{name} ({reason})" for name, reason in sorted(result.not_notified.items()))
        lines.append("not_notified: " + reasons)
    return "\n".join(lines), "ok"


def _avatar_updates(shared_dir: Path, member_id: str) -> list[str]:
    state = member_state(shared_dir, member_id)
    values = state.get("avatar_updates")
    if not isinstance(values, list):
        return []
    cutoff = now_local() - _AVATAR_UPDATE_WINDOW
    recent: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        parsed = parse_iso(value)
        if parsed is not None and parsed >= cutoff:
            recent.append(value)
    return recent


def _run_avatar(shared_dir: Path, member: HubMember, tokens: list[str], text: str) -> tuple[str, str]:
    if len(tokens) > 1:
        return _error("avatar takes one URL"), "invalid_argument"
    url = tokens[0] if tokens else str(text or "").strip()
    if not url:
        return _error("avatar requires a URL"), "invalid_argument"

    recent = _avatar_updates(shared_dir, member.id)
    if len(recent) >= _AVATAR_UPDATES_PER_DAY:
        return _error(f"avatar update limit reached ({_AVATAR_UPDATES_PER_DAY} per day)"), "rate_limited"

    try:
        png = normalize_avatar(fetch_avatar_url(url))
    except AvatarError as exc:
        return _error(str(exc)), "rejected"
    except Exception:
        logger.warning("Failed to fetch hub avatar for %s", member.id, exc_info=True)
        return _error("fetch failed"), "rejected"

    try:
        save_avatar(shared_dir, member.id, png)
        update_member_state(shared_dir, member.id, avatar_updates=[*recent, now_iso()])
    except (OSError, ValueError):
        logger.warning("Failed to save hub avatar for %s", member.id, exc_info=True)
        return _error("save failed"), "rejected"
    return "avatar updated (256x256)", "ok"


def run_hub_command(shared_dir: Path, member: HubMember, command: str, text: str = "") -> str:
    """Execute one ``hub`` command for *member* and return the text response."""
    try:
        tokens = shlex.split(command or "")
    except ValueError as exc:
        return _error(f"invalid command: {exc}")

    if not tokens:
        return _error("missing command")
    name = tokens[0]
    args = tokens[1:]
    if name not in {"help", "timeline", "read", "post", "avatar"}:
        return _error(f"unknown command: {name}")
    if text and name not in {"post", "avatar"}:
        return _error("text is only accepted by post or avatar")

    if name == "help":
        if args:
            return _error("help takes no arguments")
        return HELP_TEXT

    # Every data command is scoped to the hub's own company and board: a member
    # of another company sees neither this company's timeline nor its #hub.
    hub_config = load_config().hub
    if not hub_config.company or member.company != hub_config.company:
        return _error("this hub belongs to another company")
    if name == "read":
        from core.messaging.board import board_access

        board = str(hub_config.board or "hub")
        decision = board_access(shared_dir, board, hub_principal(member), "read")
        if not decision.allowed:
            return _error(f"cannot read #{board} ({decision.reason})")
    if name == "timeline":
        if args:
            return _error("timeline takes no arguments")
        return _run_timeline(shared_dir, member)[0]
    if name == "read":
        if args:
            return _error("read takes no arguments")
        return _run_read(shared_dir, member)[0]
    if name == "avatar":
        return _run_avatar(shared_dir, member, args, text)[0]
    return _run_post(shared_dir, member, args, text)[0]


def _status_of(result: str) -> str:
    if not result.startswith("error: "):
        return "ok"
    heading = result.splitlines()[0]
    if "post limit reached" in heading:
        return "rate_limited"
    return "rejected"


def _build_security(config: Any) -> TransportSecuritySettings:
    public_hosts = [str(host).strip() for host in getattr(config, "public_hosts", []) or [] if str(host).strip()]
    allowed_hosts = [*public_hosts, "127.0.0.1:*", "localhost:*", "127.0.0.1", "localhost"]
    allowed_origins: list[str] = []
    for host in public_hosts:
        allowed_origins.extend([f"https://{host}", f"http://{host}"])
    for host in ("127.0.0.1", "localhost"):
        allowed_origins.extend([f"https://{host}:*", f"http://{host}:*", f"https://{host}", f"http://{host}"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


async def _send_json(send: Callable[[dict[str, Any]], Any], status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _bearer_token(scope: dict[str, Any]) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() != b"authorization":
            continue
        try:
            header = value.decode("latin-1").strip()
        except (UnicodeDecodeError, AttributeError):
            return None
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        return token.strip()
    return None


class HubBearerAuthMiddleware:
    """Require a member bearer token (even from localhost) before FastMCP."""

    def __init__(self, app: Any, shared_dir_provider: Callable[[], Path]) -> None:
        self.app = app
        self.shared_dir_provider = shared_dir_provider

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        try:
            enabled = load_config().hub.enabled
        except Exception:
            logger.warning("Could not read hub settings for MCP request", exc_info=True)
            enabled = False
        if not enabled:
            await _send_json(send, 404, {"error": "not_found"})
            return

        token = _bearer_token(scope)
        shared_dir = Path(self.shared_dir_provider())
        member = find_member_by_token(shared_dir, token) if token else None
        if member is None:
            await _send_json(send, 401, {"error": "unauthorized"})
            return

        try:
            touch_last_seen(shared_dir, member.id)
        except OSError:
            logger.warning("Could not record hub last_seen for %s", member.id, exc_info=True)

        token_context = _current_member.set(member)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_member.reset(token_context)


def build_hub_mcp_app(
    shared_dir_provider: Callable[[], Path],
) -> tuple[Any, Any]:
    """Build the hub MCP ASGI app and its Streamable HTTP session manager."""
    config = load_config().hub
    mcp = FastMCP(
        "AnimaWorks Hub",
        instructions="Use the one `hub` tool. Talk to animas only in #hub.",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
        transport_security=_build_security(config),
    )

    @mcp.tool(name="hub", description=TOOL_DESCRIPTION)
    async def hub_tool(command: str, text: str = "") -> str:
        """Run one hub command (help, timeline, read, post, avatar)."""
        member = _current_member.get()
        if member is None:
            return _error("unauthorized")
        shared_dir = Path(shared_dir_provider())
        try:
            # File I/O, Board fan-out and avatar fetches block; keep them off the
            # server event loop that also serves the Web UI and the animas.
            result = await asyncio.to_thread(run_hub_command, shared_dir, member, command, text)
        except Exception:
            logger.exception("hub command failed member=%s", member.id)
            return "error: internal\n" + HELP_SUMMARY
        command_name = (command or "").split(" ")[0] or "?"
        logger.info("hub tool member=%s command=%s status=%s", member.id, command_name, _status_of(result))
        return result

    app = mcp.streamable_http_app()
    session_manager = mcp.session_manager
    return HubBearerAuthMiddleware(app, shared_dir_provider), session_manager
