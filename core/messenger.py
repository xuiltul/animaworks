from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.i18n import t
from core.time_utils import now_iso

from core.exceptions import MessagingError, DeliveryError, RecipientNotFoundError  # noqa: F401
from core.schemas import Message

logger = logging.getLogger("animaworks.messenger")

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


def _validate_name(name: str, kind: str = "name") -> None:
    """Validate a channel or peer name to prevent path traversal."""
    if not _SAFE_NAME_RE.match(name):
        raise RecipientNotFoundError(f"Invalid {kind}: {name!r}")


# ── Channel ACL helpers ──────────────────────────────────


@dataclass
class ChannelMeta:
    """Channel metadata including ACL membership list.

    If ``members`` is empty, the channel is open (all Animas can access).
    """
    members: list[str]
    created_by: str = ""
    created_at: str = ""
    description: str = ""


def load_channel_meta(shared_dir: Path, channel: str) -> ChannelMeta | None:
    """Load channel metadata from ``shared/channels/{channel}.meta.json``.

    Returns ``None`` when no meta file exists (channel is open / legacy).
    """
    meta_path = shared_dir / "channels" / f"{channel}.meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return ChannelMeta(
            members=data.get("members", []),
            created_by=data.get("created_by", ""),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load channel meta for %s: %s", channel, exc)
        return None


def save_channel_meta(shared_dir: Path, channel: str, meta: ChannelMeta) -> None:
    """Persist channel metadata to ``shared/channels/{channel}.meta.json``."""
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    meta_path = channels_dir / f"{channel}.meta.json"
    data = {
        "members": meta.members,
        "created_by": meta.created_by,
        "created_at": meta.created_at,
        "description": meta.description,
    }
    meta_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_channel_member(
    shared_dir: Path,
    channel: str,
    anima_name: str,
    *,
    source: str = "anima",
) -> bool:
    """Check whether *anima_name* may access *channel*.

    Rules:
    - ``human`` source always has access (Web UI bypass).
    - If no ``.meta.json`` exists the channel is open — everyone has access.
    - If ``members`` list is empty the channel is open.
    - Otherwise the anima must appear in the ``members`` list.
    """
    if source == "human":
        return True
    meta = load_channel_meta(shared_dir, channel)
    if meta is None or not meta.members:
        return True
    return anima_name in meta.members


@dataclass
class InboxItem:
    """Message paired with its file path for selective archiving."""
    msg: Message
    path: Path


class Messenger:
    """File-system based messaging.

    Messages are JSON files in shared/inbox/{name}/.
    """

    def __init__(
        self,
        shared_dir: Path,
        anima_name: str,
    ) -> None:
        self.shared_dir = shared_dir
        self.anima_name = anima_name
        self.inbox_dir = shared_dir / "inbox" / anima_name
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.inbox_dir.chmod(0o700)
        except OSError:
            pass

    def send(
        self,
        to: str,
        content: str,
        msg_type: str = "message",
        thread_id: str = "",
        reply_to: str = "",
        skip_logging: bool = False,
        intent: str = "",
        origin_chain: list[str] | None = None,
    ) -> Message:
        # ── Conversation depth check (internal Anima only) ──
        if msg_type not in ("ack", "error", "system_alert"):
            from core.paths import get_animas_dir
            animas_dir = get_animas_dir()
            is_internal = (animas_dir / to).is_dir() if animas_dir.exists() else False
            if is_internal:
                from core.cascade_limiter import depth_limiter
                sender_dir = animas_dir / self.anima_name
                outbound_check = depth_limiter.check_global_outbound(self.anima_name, sender_dir)
                if outbound_check is not True:
                    logger.warning(
                        "Global outbound limit exceeded: %s. Message not sent.",
                        self.anima_name,
                    )
                    return Message(
                        from_person="system",
                        to_person=self.anima_name,
                        type="error",
                        content=str(outbound_check),
                    )
                if not depth_limiter.check_depth(self.anima_name, to, sender_dir):
                    logger.warning(
                        "Depth limit exceeded: %s -> %s. Message not sent.",
                        self.anima_name, to,
                    )
                    return Message(
                        from_person="system",
                        to_person=self.anima_name,
                        type="error",
                        content=t("messenger.depth_exceeded", to=to),
                    )

        msg = Message(
            from_person=self.anima_name,
            to_person=to,
            type=msg_type,
            content=content,
            thread_id=thread_id,
            reply_to=reply_to,
            intent=intent,
            origin_chain=origin_chain or [],
        )
        # New thread: use message id as thread_id
        if not msg.thread_id:
            msg.thread_id = msg.id
        target_dir = self.shared_dir / "inbox" / to
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{msg.id}.json"
        filepath.write_text(msg.model_dump_json(indent=2), encoding="utf-8")

        if not filepath.exists():
            raise DeliveryError(
                f"Message delivery failed: file not created at {filepath} "
                f"({self.anima_name} -> {to})"
            )

        logger.info("Message sent: %s -> %s (%s)", self.anima_name, to, msg.id)

        # Activity Log: record message_sent for all send paths (S/A/B/CLI)
        if not skip_logging:
            try:
                from core.memory.activity import ActivityLogger
                anima_dir = self.shared_dir.parent / "animas" / self.anima_name
                if anima_dir.exists():
                    activity = ActivityLogger(anima_dir)
                    meta: dict[str, Any] = {"from_type": "anima"}
                    if intent:
                        meta["intent"] = intent
                    activity.log("message_sent", content=content, to_person=to, meta=meta)
            except Exception as e:
                logger.warning(
                    "Activity logging failed for message_sent (%s -> %s): %s",
                    self.anima_name, to, e,
                )

        return msg

    def reply(self, original: Message, content: str, *, intent: str = "") -> Message:
        """Reply to a message, inheriting thread_id."""
        return self.send(
            to=original.from_person,
            content=content,
            thread_id=original.thread_id or original.id,
            reply_to=original.id,
            intent=intent,
        )

    # ── Channel operations ──────────────────────────────────

    def post_channel(
        self, channel: str, text: str, source: str = "anima",
        from_name: str | None = None,
    ) -> None:
        """Post a message to a shared channel (append-only JSONL)."""
        _validate_name(channel, "channel name")

        poster = from_name or self.anima_name

        # ── ACL check ──
        if not is_channel_member(self.shared_dir, channel, self.anima_name, source=source):
            logger.warning(
                "ACL denied: %s cannot post to #%s", self.anima_name, channel,
            )
            return

        # Validate from_name: must be a known anima or "human"
        if poster != "human":
            try:
                from core.config.models import load_config
                known_animas = set(load_config().animas.keys())
                if known_animas and self.anima_name in known_animas and poster not in known_animas:
                    logger.warning(
                        "Rejecting channel post with unknown from_name=%r to #%s",
                        poster, channel,
                    )
                    return
            except Exception:
                pass
        channels_dir = self.shared_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        filepath = channels_dir / f"{channel}.jsonl"
        entry = json.dumps({
            "ts": now_iso(),
            "from": poster,
            "text": text,
            "source": source,
        }, ensure_ascii=False)
        try:
            with filepath.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
            logger.info("Channel post: %s -> #%s", poster, channel)
        except OSError:
            logger.warning("Failed to post to channel: %s", channel)

    def last_post_by(self, anima_name: str, channel: str) -> dict | None:
        """Return the last post by *anima_name* in *channel*, or None.

        Scans the channel JSONL file from the tail for efficiency.
        Used by ToolHandler for cross-run cooldown checks.
        """
        filepath = self.shared_dir / "channels" / f"{channel}.jsonl"
        if not filepath.exists():
            return None
        try:
            lines = filepath.read_text(encoding="utf-8").strip().splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("from") == anima_name:
                        return entry
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return None

    def read_channel(
        self, channel: str, limit: int = 20, human_only: bool = False,
        *, source: str = "anima",
    ) -> list[dict]:
        """Read recent messages from a shared channel."""
        _validate_name(channel, "channel name")

        # ── ACL check ──
        if not is_channel_member(self.shared_dir, channel, self.anima_name, source=source):
            logger.warning(
                "ACL denied: %s cannot read #%s", self.anima_name, channel,
            )
            return []

        filepath = self.shared_dir / "channels" / f"{channel}.jsonl"
        if not filepath.exists():
            return []
        try:
            lines = filepath.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            logger.warning("Failed to read channel: %s", channel)
            return []
        entries: list[dict] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if human_only and entry.get("source") != "human":
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        entries.reverse()  # Restore chronological order
        return entries

    def read_channel_mentions(
        self, channel: str, name: str | None = None, limit: int = 10,
        *, source: str = "anima",
    ) -> list[dict]:
        """Read messages mentioning @name from a shared channel."""
        _validate_name(channel, "channel name")
        target = name or self.anima_name
        mention_tag = f"@{target}"
        all_msgs = self.read_channel(channel, limit=1000, source=source)
        mentions = [m for m in all_msgs if mention_tag in m.get("text", "")]
        return mentions[-limit:]

    def read_dm_history(self, peer: str, limit: int = 20) -> list[dict]:
        """Read DM history with a specific peer.

        Reads from unified activity log first, falls back to legacy dm_logs/.
        """
        _validate_name(peer, "peer name")
        entries: list[dict] = []

        # New source: unified activity log
        try:
            from core.memory.activity import ActivityLogger
            # Determine anima_dir from shared_dir (shared_dir is {data}/shared,
            # anima_dir is {data}/animas/{name})
            anima_dir = self.shared_dir.parent / "animas" / self.anima_name
            if anima_dir.exists():
                activity = ActivityLogger(anima_dir)
                recent = activity.recent(
                    days=30, limit=limit * 2,
                    types=["message_sent", "message_received"],
                    involving=peer,
                )
                for e in recent:
                    # Exclude chat message_received (from_type=human)
                    if e.type == "message_received" and e.meta.get("from_type") != "anima":
                        continue
                    entries.append({
                        "ts": e.ts,
                        "from": e.from_person or self.anima_name,
                        "text": e.content,
                        "source": "activity_log",
                    })
        except Exception:
            logger.debug("Failed to read DM history from activity log", exc_info=True)

        # Fallback: legacy dm_logs/
        if len(entries) < limit:
            filepath = self._get_dm_log_path(peer)
            if filepath.exists():
                try:
                    lines = filepath.read_text(encoding="utf-8").strip().splitlines()
                except OSError:
                    logger.warning("Failed to read DM history with: %s", peer)
                    lines = []
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(entries) >= limit * 2:
                        break

        # Sort by timestamp and return most recent
        entries.sort(key=lambda e: e.get("ts", ""))
        return entries[-limit:]

    def _get_dm_log_path(self, peer: str) -> Path:
        """Get DM log file path (pair names sorted alphabetically)."""
        pair = sorted([self.anima_name, peer])
        return self.shared_dir / "dm_logs" / f"{pair[0]}-{pair[1]}.jsonl"

    def _append_dm_log(self, peer: str, content: str, *, intent: str = "") -> None:
        """Append a DM entry to the legacy dm_logs/ file."""
        filepath = self._get_dm_log_path(peer)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        entry_dict: dict[str, Any] = {
            "ts": now_iso(),
            "from": self.anima_name,
            "to": peer,
            "text": content,
        }
        if intent:
            entry_dict["intent"] = intent
        entry = json.dumps(entry_dict, ensure_ascii=False)
        with filepath.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def receive(self) -> list[Message]:
        known_animas: set[str] | None = None
        try:
            from core.config.models import load_config
            cfg_animas = set(load_config().animas.keys())
            if self.anima_name in cfg_animas:
                known_animas = cfg_animas
        except Exception:
            logger.debug("load_config unavailable for inbox validation", exc_info=True)

        messages: list[Message] = []
        for f in sorted(self.inbox_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                msg = Message(**data)
                if known_animas is not None and msg.from_person not in known_animas:
                    logger.warning(
                        "Ignoring inbox message with unknown from_person=%r in %s",
                        msg.from_person, f,
                    )
                    continue
                messages.append(msg)
            except Exception as e:
                logger.error("Failed to parse message %s: %s", f, e)
        return messages

    def receive_with_paths(self) -> list[InboxItem]:
        """Read all unread messages, returning Message + file path pairs.

        Unlike receive(), the caller is responsible for archiving
        via archive_paths() after processing.  Messages that arrive
        while the caller is working will remain in the inbox.
        """
        known_animas: set[str] | None = None
        try:
            from core.config.models import load_config
            cfg_animas = set(load_config().animas.keys())
            if self.anima_name in cfg_animas:
                known_animas = cfg_animas
        except Exception:
            logger.debug("load_config unavailable for inbox validation", exc_info=True)

        items: list[InboxItem] = []
        for f in sorted(self.inbox_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                msg = Message(**data)
                if known_animas is not None and msg.from_person not in known_animas:
                    logger.warning(
                        "Ignoring inbox message with unknown from_person=%r in %s",
                        msg.from_person, f,
                    )
                    continue
                items.append(InboxItem(msg=msg, path=f))
            except Exception:
                logger.warning("Failed to read inbox file: %s", f, exc_info=True)
        return items

    def archive_paths(self, items: list[InboxItem]) -> int:
        """Archive only the specified inbox items to processed/.

        Files that no longer exist (e.g. already archived) are silently
        skipped.
        """
        processed_dir = self.inbox_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        count = 0
        for item in items:
            if item.path.exists():
                item.path.rename(processed_dir / item.path.name)
                count += 1
        return count

    def receive_and_archive(self) -> list[Message]:
        messages = self.receive()
        if messages:
            # E: Send read ACK to senders (disabled by default via heartbeat.enable_read_ack)
            try:
                from core.config.models import load_config
                _send_ack = load_config().heartbeat.enable_read_ack
            except Exception:
                _send_ack = False

            if _send_ack:
                non_ack_messages = [m for m in messages if m.type not in ("ack", "board_mention")]
                if non_ack_messages:
                    senders: dict[str, list[Message]] = {}
                    for m in non_ack_messages:
                        senders.setdefault(m.from_person, []).append(m)
                    for sender, sender_msgs in senders.items():
                        summary = ", ".join(m.content[:50] for m in sender_msgs[:3])
                        if len(sender_msgs) > 3:
                            summary += " " + t("messenger.more_count", count=len(sender_msgs) - 3)
                        try:
                            self.send(
                                to=sender,
                                content=t("messenger.read_receipt", count=len(sender_msgs), summary=summary),
                                msg_type="ack",
                                skip_logging=True,
                            )
                        except Exception:
                            logger.debug(
                                "Failed to send read ACK to %s", sender, exc_info=True,
                            )
            self.archive_all()
        return messages

    def archive_all(self) -> int:
        """Move all unread messages in inbox to processed/."""
        processed_dir = self.inbox_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        count = 0
        for f in self.inbox_dir.glob("*.json"):
            f.rename(processed_dir / f.name)
            count += 1
        # Clean up non-JSON files that may have been left by Agent SDK
        for f in self.inbox_dir.iterdir():
            if f.is_file() and f.suffix != ".json" and not f.name.startswith("."):
                logger.warning("Cleaning up non-JSON file in inbox: %s", f.name)
                f.rename(processed_dir / f.name)
        return count

    def archive_from(self, sender: str) -> int:
        """Move messages from a specific sender to processed/.

        Only archives messages where ``from_person`` matches *sender*.
        Returns the number of archived messages.
        """
        processed_dir = self.inbox_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        count = 0
        for f in self.inbox_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("from_person") == sender:
                    f.rename(processed_dir / f.name)
                    count += 1
            except Exception as e:
                logger.error("Failed to check message %s: %s", f, e)
        return count

    def has_unread(self) -> bool:
        return any(self.inbox_dir.glob("*.json"))

    def unread_count(self) -> int:
        return len(list(self.inbox_dir.glob("*.json")))

    def receive_external(
        self,
        content: str,
        source: str,
        source_message_id: str = "",
        external_user_id: str = "",
        external_channel_id: str = "",
    ) -> Message:
        """Receive a message from an external platform and place it in inbox.

        Creates a Message with external source metadata and writes it to the
        anima's inbox directory.
        """
        from core.execution._sanitize import ORIGIN_EXTERNAL_PLATFORM
        msg = Message(
            from_person=f"{source}:{external_user_id}" if external_user_id else source,
            to_person=self.anima_name,
            content=content,
            source=source,
            source_message_id=source_message_id,
            external_user_id=external_user_id,
            external_channel_id=external_channel_id,
            origin_chain=[ORIGIN_EXTERNAL_PLATFORM],
        )
        if not msg.thread_id:
            msg.thread_id = msg.id
        filepath = self.inbox_dir / f"{msg.id}.json"
        filepath.write_text(msg.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "External message received: %s -> %s (source=%s, id=%s)",
            msg.from_person, self.anima_name, source, msg.id,
        )
        # Mirror to general channel if human uses @all
        if source == "human" and "@all" in content:
            human_name = external_user_id or "human"
            self.post_channel("general", content, source="human", from_name=human_name)
        return msg

    async def send_async(
        self,
        to: str,
        content: str,
        msg_type: str = "message",
        thread_id: str = "",
        reply_to: str = "",
        intent: str = "",
        origin_chain: list[str] | None = None,
    ) -> Message:
        """Async wrapper for filesystem-based send."""
        return self.send(
            to=to,
            content=content,
            msg_type=msg_type,
            thread_id=thread_id,
            reply_to=reply_to,
            intent=intent,
            origin_chain=origin_chain,
        )
