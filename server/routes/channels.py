from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Board (shared channel) and DM history API routes."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.memory.activity import ActivityLogger
from core.messenger import Messenger
from server.events import emit

logger = logging.getLogger("animaworks.routes.channels")

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


def _validate_name(name: str) -> JSONResponse | None:
    """Return a 400 response if name contains unsafe characters, else None."""
    if not _SAFE_NAME_RE.match(name):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid name: {name!r}"},
        )
    return None


class ChannelPostRequest(BaseModel):
    text: str
    from_name: str = "human"


def _get_messenger(shared_dir: Path) -> Messenger:
    """Create a Messenger instance for human-originated operations."""
    return Messenger(shared_dir, anima_name="human")


def create_channels_router() -> APIRouter:
    router = APIRouter()

    # ── Channel endpoints ────────────────────────────────────

    @router.get("/channels")
    async def list_channels(request: Request):
        """List all shared channels with metadata."""
        shared_dir: Path = request.app.state.shared_dir
        channels_dir = shared_dir / "channels"
        if not channels_dir.exists():
            return []

        channels: list[dict] = []
        for f in sorted(channels_dir.glob("*.jsonl")):
            name = f.stem
            try:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
            except OSError:
                lines = []

            count = len(lines)
            last_ts = ""
            if lines:
                try:
                    last_entry = json.loads(lines[-1])
                    last_ts = last_entry.get("ts", "")
                except (json.JSONDecodeError, IndexError):
                    pass

            channels.append({
                "name": name,
                "message_count": count,
                "last_post_ts": last_ts,
            })

        return channels

    @router.get("/channels/{name}")
    async def get_channel_messages(
        request: Request,
        name: str,
        limit: int = 50,
        offset: int = 0,
    ):
        """Get messages from a specific channel."""
        if err := _validate_name(name):
            return err
        shared_dir: Path = request.app.state.shared_dir
        channel_file = shared_dir / "channels" / f"{name}.jsonl"

        if not channel_file.exists():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Channel '{name}' not found"},
            )

        # Read all lines for accurate total count
        try:
            all_lines = channel_file.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            all_lines = []
        all_lines = [l for l in all_lines if l.strip()]

        # Parse messages in range
        all_messages: list[dict] = []
        for line in all_lines:
            try:
                all_messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        total = len(all_messages)
        paginated = all_messages[offset:offset + limit]

        return {
            "channel": name,
            "messages": paginated,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total,
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
        # Override from_name with authenticated user
        if hasattr(request.state, "user"):
            body.from_name = request.state.user.username
        shared_dir: Path = request.app.state.shared_dir
        channels_dir = shared_dir / "channels"

        # Check channel exists
        channel_file = channels_dir / f"{name}.jsonl"
        if not channel_file.exists():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Channel '{name}' not found"},
            )

        messenger = _get_messenger(shared_dir)
        messenger.post_channel(
            name, body.text, source="human", from_name=body.from_name,
        )

        # Broadcast WebSocket event
        event_data = {
            "channel": name,
            "from": body.from_name,
            "text": body.text,
            "source": "human",
            "ts": datetime.now().isoformat(),
        }
        await emit(request, "board.post", event_data)

        logger.info("Human posted to #%s: %s", name, body.text[:80])
        return {"status": "ok", "channel": name}

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
        mentions = messenger.read_channel_mentions(name, name=anima, limit=limit)
        return {
            "channel": name,
            "anima": anima,
            "mentions": mentions,
            "count": len(mentions),
        }

    # ── DM endpoints ─────────────────────────────────────────

    @router.get("/dm")
    async def list_dm_pairs(request: Request):
        """List all DM conversation pairs with metadata.

        Primary source: per-Anima activity_log/ (dm_sent/dm_received).
        Fallback: legacy shared/dm_logs/ for historical data.
        """
        shared_dir: Path = request.app.state.shared_dir
        animas_dir = shared_dir.parent / "animas"

        # pair_key (sorted "alice-bob") -> {count, last_ts}
        pair_map: dict[str, dict] = {}

        # ── Primary: scan activity_log from all Animas ────────
        if animas_dir.exists():
            for anima_dir in animas_dir.iterdir():
                if not anima_dir.is_dir():
                    continue
                try:
                    activity = ActivityLogger(anima_dir)
                    entries = activity.recent(
                        days=30, limit=500,
                        types=["dm_sent", "dm_received"],
                    )
                    for e in entries:
                        # Determine the two participants
                        sender = e.from_person or anima_dir.name
                        receiver = e.to_person
                        if not receiver:
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
                        anima_dir.name, exc_info=True,
                    )

        # ── Fallback: legacy dm_logs/ ─────────────────────────
        dm_logs_dir = shared_dir / "dm_logs"
        if dm_logs_dir.exists():
            for f in dm_logs_dir.glob("*.jsonl"):
                pair_name = f.stem
                try:
                    lines = f.read_text(encoding="utf-8").strip().splitlines()
                except OSError:
                    lines = []

                count = len(lines)
                last_ts = ""
                if lines:
                    try:
                        last_entry = json.loads(lines[-1])
                        last_ts = last_entry.get("ts", "")
                    except (json.JSONDecodeError, IndexError):
                        pass

                if pair_name in pair_map:
                    # Merge: add legacy count, keep latest ts
                    pair_map[pair_name]["count"] += count
                    if last_ts > pair_map[pair_name]["last_ts"]:
                        pair_map[pair_name]["last_ts"] = last_ts
                else:
                    pair_map[pair_name] = {"count": count, "last_ts": last_ts}

        # ── Build response sorted by last_message_ts desc ─────
        pairs: list[dict] = []
        for pair_key, info in pair_map.items():
            pairs.append({
                "pair": pair_key,
                "participants": pair_key.split("-", 1),
                "message_count": info["count"],
                "last_message_ts": info["last_ts"],
            })

        pairs.sort(key=lambda p: p["last_message_ts"], reverse=True)
        return pairs

    @router.get("/dm/{pair}")
    async def get_dm_history(
        request: Request,
        pair: str,
        limit: int = 50,
    ):
        """Get DM history for a specific pair.

        Primary source: per-Anima activity_log/ (dm_sent/dm_received).
        Fallback: legacy shared/dm_logs/ for historical data.
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
        seen: set[str] = set()  # dedup key: "ts|content"

        # ── Primary: read from each participant's activity_log ─
        for name in participants:
            other = participants[1] if name == participants[0] else participants[0]
            anima_dir = animas_dir / name
            if not anima_dir.exists():
                continue
            try:
                activity = ActivityLogger(anima_dir)
                entries = activity.recent(
                    days=30, limit=limit * 2,
                    types=["dm_sent", "dm_received"],
                    involving=other,
                )
                for e in entries:
                    dedup_key = f"{e.ts}|{e.content}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    messages.append({
                        "ts": e.ts,
                        "from": e.from_person or name,
                        "text": e.content,
                        "source": "activity_log",
                    })
            except Exception:
                logger.debug(
                    "Failed to read DM activity_log for %s",
                    name, exc_info=True,
                )

        # ── Fallback: legacy dm_logs/ ─────────────────────────
        dm_file = shared_dir / "dm_logs" / f"{pair}.jsonl"
        if dm_file.exists():
            try:
                lines = dm_file.read_text(encoding="utf-8").strip().splitlines()
            except OSError:
                lines = []

            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dedup_key = f"{entry.get('ts', '')}|{entry.get('text', entry.get('content', ''))}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                messages.append(entry)

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
