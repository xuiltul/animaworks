from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Board (shared channel) and DM history API routes."""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.activity.logger import ActivityLogger
from core.messaging.board import board_access, human_principal
from core.messaging.board_deliveries import attach_deliveries
from core.messaging.messenger import Messenger

logger = logging.getLogger("animaworks.routes.channels")

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")

# ── Caches ────────────────────────────────────────
# Channel metadata cache: keyed by file path, invalidated on mtime/size change
_channel_meta_cache: dict[str, dict] = {}

# DM pairs cache: TTL-based, keyed by shared_dir to avoid stale data
_dm_cache: dict[str, object] = {"data": None, "ts": 0.0, "key": ""}
_DM_CACHE_TTL = 60.0


def _get_channel_meta(f: Path) -> dict:
    """Return {count, last_ts} for a channel file, cached by mtime+size."""
    key = str(f)
    try:
        stat = f.stat()
    except OSError:
        return {"count": 0, "last_ts": ""}

    cached = _channel_meta_cache.get(key)
    if cached and cached["_mtime"] == stat.st_mtime and cached["_size"] == stat.st_size:
        return cached

    try:
        raw = f.read_bytes().strip()
    except OSError:
        raw = b""

    if not raw:
        entry = {"count": 0, "last_ts": "", "_mtime": stat.st_mtime, "_size": stat.st_size}
        _channel_meta_cache[key] = entry
        return entry

    count = raw.count(b"\n") + 1

    last_ts = ""
    try:
        last_line = raw.rsplit(b"\n", 1)[-1]
        last_ts = json.loads(last_line).get("ts", "")
    except (json.JSONDecodeError, IndexError):
        pass

    entry = {"count": count, "last_ts": last_ts, "_mtime": stat.st_mtime, "_size": stat.st_size}
    _channel_meta_cache[key] = entry
    return entry


def _validate_name(name: str) -> JSONResponse | None:
    """Return a 400 response if name contains unsafe characters, else None."""
    if not _SAFE_NAME_RE.match(name):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid name: {name!r}"},
        )
    return None


class ChannelPostRequest(BaseModel):
    text: str = Field(max_length=10000)
    from_name: str = Field(default="human", min_length=1)
    reply_to: str | None = None


def _channel_has_post(channel_file: Path, post_id: str) -> bool:
    """Return whether *post_id* exists in the channel log."""
    try:
        lines = channel_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id") == post_id:
            return True
    return False


def _get_messenger(shared_dir: Path) -> Messenger:
    """Create a Messenger instance for human-originated operations."""
    return Messenger(shared_dir, anima_name="human")


def create_channels_router() -> APIRouter:
    router = APIRouter()

    # ── Channel endpoints ────────────────────────────────────

    @router.get("/channels")
    def list_channels(request: Request):
        """List all shared channels with metadata including ACL info."""
        shared_dir: Path = request.app.state.shared_dir
        channels_dir = shared_dir / "channels"
        if not channels_dir.exists():
            return []

        from core.messaging.messenger import load_channel_meta

        principal = human_principal("human")
        channels: list[dict] = []
        for f in sorted(channels_dir.glob("*.jsonl")):
            name = f.stem
            if not board_access(shared_dir, name, principal, "read").allowed:
                continue
            meta_info = _get_channel_meta(f)

            channel_info: dict = {
                "name": name,
                "message_count": meta_info["count"],
                "last_post_ts": meta_info["last_ts"],
            }

            meta = load_channel_meta(shared_dir, name)
            if meta is not None and meta.members:
                channel_info["members"] = meta.members
                channel_info["description"] = meta.description

            channels.append(channel_info)

        return channels

    @router.get("/channels/{name}")
    def get_channel_messages(
        request: Request,
        name: str,
        limit: int = 50,
        offset: int = 0,
        since: str = "",
    ):
        """Get messages from a specific channel.

        Web UI requests are treated as human principals and follow ``board_access``.

        When *since* is provided (ISO timestamp), only messages strictly
        newer than that timestamp are returned (for incremental polling).
        """
        if err := _validate_name(name):
            return err
        shared_dir: Path = request.app.state.shared_dir
        channel_file = shared_dir / "channels" / f"{name}.jsonl"

        if not channel_file.exists():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Channel '{name}' not found"},
            )

        if not board_access(shared_dir, name, human_principal("human"), "read").allowed:
            return JSONResponse(
                status_code=403,
                content={"detail": f"Access denied to channel '{name}'"},
            )

        try:
            all_lines = channel_file.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            all_lines = []
        all_lines = [line for line in all_lines if line.strip()]

        total = len(all_lines)

        if since:
            # Incremental mode: scan from the tail for new messages only
            new_msgs: list[dict] = []
            for line in reversed(all_lines):
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("ts", "") <= since:
                    break
                new_msgs.append(msg)
            new_msgs.reverse()
            new_msgs = attach_deliveries(shared_dir, name, new_msgs)
            return {
                "channel": name,
                "messages": new_msgs,
                "total": total,
                "offset": 0,
                "limit": limit,
                "has_more": False,
                "incremental": True,
            }

        # Reverse pagination: offset=0 → newest N messages
        start = max(0, total - offset - limit)
        end = max(0, total - offset)
        sliced_lines = all_lines[start:end]

        paginated: list[dict] = []
        for line in sliced_lines:
            try:
                paginated.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        has_more = start > 0
        paginated = attach_deliveries(shared_dir, name, paginated)

        return {
            "channel": name,
            "messages": paginated,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
        }

    @router.post("/channels/{name}")
    async def post_to_channel(
        request: Request,
        name: str,
        body: ChannelPostRequest,
    ):
        """Post a message to a channel (human-originated)."""
        if err := _validate_name(name):
            return err
        # Override from_name with authenticated user; force "human" when unauthenticated
        if hasattr(request.state, "user"):
            body.from_name = request.state.user.username
        else:
            body.from_name = "human"
        shared_dir: Path = request.app.state.shared_dir

        from core.messaging.board import human_principal, post_to_board

        channel_file = shared_dir / "channels" / f"{name}.jsonl"
        if body.reply_to is not None and not _channel_has_post(channel_file, body.reply_to):
            from core.i18n import t

            return JSONResponse(
                status_code=400,
                content={"detail": t("server.channel_reply_to_not_found", post_id=body.reply_to)},
            )

        post = await asyncio.to_thread(
            post_to_board,
            name,
            body.text,
            human_principal(body.from_name),
            origin="http",
            shared_dir=shared_dir,
            reply_to=body.reply_to,
        )
        if post.error == "not_found":
            return JSONResponse(
                status_code=404,
                content={"detail": f"Channel '{name}' not found"},
            )
        if post.error and post.error.startswith("denied:"):
            return JSONResponse(
                status_code=403,
                content={"detail": f"Access denied to channel '{name}'"},
            )
        if post.error == "invalid_name":
            return JSONResponse(status_code=400, content={"detail": f"Invalid name: {name!r}"})
        if post.error:
            return JSONResponse(status_code=500, content={"detail": post.error})

        logger.info("Human posted to #%s: %s", name, body.text[:80])
        return {"status": "ok", "channel": name, "post": post.to_dict()}

    @router.get("/channels/{name}/mentions/{anima}")
    async def get_channel_mentions(
        request: Request,
        name: str,
        anima: str,
        limit: int = 10,
    ):
        """Get messages mentioning a specific anima in a channel."""
        if err := _validate_name(name):
            return err
        shared_dir: Path = request.app.state.shared_dir
        messenger = _get_messenger(shared_dir)
        mentions = messenger.read_channel_mentions(name, name=anima, limit=limit, source="human")
        return {
            "channel": name,
            "anima": anima,
            "mentions": mentions,
            "count": len(mentions),
        }

    # ── DM endpoints ─────────────────────────────────────────

    @router.get("/dm")
    def list_dm_pairs(request: Request):
        """List all DM conversation pairs with metadata.

        Source: per-Anima activity_log/ (message_sent/message_received).
        Uses a TTL cache to avoid scanning all activity logs on every request.
        """
        shared_dir: Path = request.app.state.shared_dir
        cache_key = str(shared_dir)
        now = time.monotonic()
        if _dm_cache["data"] is not None and _dm_cache["key"] == cache_key and now - _dm_cache["ts"] < _DM_CACHE_TTL:
            return _dm_cache["data"]
        animas_dir = shared_dir.parent / "animas"

        pair_map: dict[str, dict] = {}

        valid_names: set[str] = set()
        try:
            from core.config.models import load_config

            valid_names = set(load_config().animas.keys())
        except Exception:
            logger.debug("load_config unavailable for DM pair filter", exc_info=True)

        # ── Primary: scan activity_log from all Animas ────────
        if animas_dir.exists():
            for anima_dir in animas_dir.iterdir():
                if not anima_dir.is_dir():
                    continue
                try:
                    activity = ActivityLogger(anima_dir)
                    entries = activity.recent(
                        days=30,
                        limit=500,
                        types=["message_sent", "message_received"],
                    )
                    for e in entries:
                        sender = e.from_person or anima_dir.name
                        receiver = e.to_person
                        # Self-addressed entries (e.g. background task failure
                        # notices) are not a conversation between two parties.
                        if not receiver or receiver == sender:
                            continue
                        pair_key = "-".join(sorted([sender, receiver]))
                        if pair_key not in pair_map:
                            pair_map[pair_key] = {"count": 0, "last_ts": ""}
                        pair_map[pair_key]["count"] += 1
                        if e.ts > pair_map[pair_key]["last_ts"]:
                            pair_map[pair_key]["last_ts"] = e.ts
                except Exception:
                    logger.debug(
                        "Failed to read activity_log for %s",
                        anima_dir.name,
                        exc_info=True,
                    )

        # ── Build response: filter + sort by last_message_ts desc ─
        pairs: list[dict] = []
        for pair_key, info in pair_map.items():
            if info["count"] == 0:
                continue
            parts = pair_key.split("-", 1)
            if len(parts) != 2 or parts[0] == parts[1]:
                continue
            if valid_names and not (parts[0] in valid_names and parts[1] in valid_names):
                continue
            pairs.append(
                {
                    "pair": pair_key,
                    "participants": parts,
                    "message_count": info["count"],
                    "last_message_ts": info["last_ts"],
                }
            )

        pairs.sort(key=lambda p: p["last_message_ts"], reverse=True)
        _dm_cache.update({"data": pairs, "ts": now, "key": cache_key})
        return pairs

    @router.get("/dm/{pair}")
    def get_dm_history(
        request: Request,
        pair: str,
        limit: int = 50,
    ):
        """Get DM history for a specific pair.

        Reads only ``message_sent`` (+ alias ``dm_sent``) from each
        participant's activity_log. Because each DM is recorded once on
        the sender side, duplicates are structurally impossible.
        """
        if err := _validate_name(pair):
            return err
        shared_dir: Path = request.app.state.shared_dir
        animas_dir = shared_dir.parent / "animas"
        participants = pair.split("-", 1)
        if len(participants) != 2:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid pair format: {pair!r}"},
            )

        messages: list[dict] = []
        # ── Primary: read message_sent from each participant ──
        for name in participants:
            other = participants[1] if name == participants[0] else participants[0]
            anima_dir = animas_dir / name
            if not anima_dir.exists():
                continue
            try:
                activity = ActivityLogger(anima_dir)
                entries = activity.recent(
                    days=30,
                    limit=limit * 2,
                    types=["message_sent"],
                    involving=other,
                )
                for e in entries:
                    messages.append(
                        {
                            "ts": e.ts,
                            "from": e.from_person or name,
                            "text": e.content,
                            "source": "activity_log",
                        }
                    )
            except Exception:
                logger.debug(
                    "Failed to read DM activity_log for %s",
                    name,
                    exc_info=True,
                )

        if not messages:
            return JSONResponse(
                status_code=404,
                content={"detail": f"DM history '{pair}' not found"},
            )

        # Sort by timestamp, return last N
        messages.sort(key=lambda m: m.get("ts", ""))
        paginated = messages[-limit:] if len(messages) > limit else messages

        return {
            "pair": pair,
            "participants": participants,
            "messages": paginated,
            "total": len(messages),
            "limit": limit,
        }

    return router
