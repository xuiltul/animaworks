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
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.schemas import Message

logger = logging.getLogger("animaworks.messenger")

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


def _validate_name(name: str, kind: str = "name") -> None:
    """Validate a channel or peer name to prevent path traversal."""
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")


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

    def send(
        self,
        to: str,
        content: str,
        msg_type: str = "message",
        thread_id: str = "",
        reply_to: str = "",
        skip_logging: bool = False,
        intent: str = "",
    ) -> Message:
        # ── Conversation depth check (internal Anima only) ──
        if msg_type not in ("ack", "error", "system_alert", "board_mention"):
            from core.paths import get_animas_dir
            animas_dir = get_animas_dir()
            is_internal = (animas_dir / to).is_dir() if animas_dir.exists() else False
            if is_internal:
                from core.cascade_limiter import depth_limiter
                if not depth_limiter.check_and_record(self.anima_name, to):
                    logger.warning(
                        "Depth limit exceeded: %s -> %s. Message not sent.",
                        self.anima_name, to,
                    )
                    return Message(
                        from_person="system",
                        to_person=self.anima_name,
                        type="error",
                        content=(
                            f"ConversationDepthExceeded: {to}との会話が"
                            f"10分間に6ターンに達しました。"
                            f"次のハートビートサイクルまでお待ちください"
                        ),
                    )

        msg = Message(
            from_person=self.anima_name,
            to_person=to,
            type=msg_type,
            content=content,
            thread_id=thread_id,
            reply_to=reply_to,
            intent=intent,
        )
        # New thread: use message id as thread_id
        if not msg.thread_id:
            msg.thread_id = msg.id
        target_dir = self.shared_dir / "inbox" / to
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{msg.id}.json"
        filepath.write_text(msg.model_dump_json(indent=2), encoding="utf-8")

        # Delivery verification: confirm file was written
        if not filepath.exists():
            raise OSError(
                f"Message delivery failed: file not created at {filepath} "
                f"({self.anima_name} -> {to})"
            )

        logger.info("Message sent: %s -> %s (%s)", self.anima_name, to, msg.id)

        # Activity Log: record dm_sent for all send paths (A1/A2/B/CLI)
        if not skip_logging:
            try:
                from core.memory.activity import ActivityLogger
                anima_dir = self.shared_dir.parent / "animas" / self.anima_name
                if anima_dir.exists():
                    activity = ActivityLogger(anima_dir)
                    log_kwargs: dict[str, Any] = {"content": content, "to_person": to}
                    if intent:
                        log_kwargs["meta"] = {"intent": intent}
                    activity.log("dm_sent", **log_kwargs)
            except Exception as e:
                logger.warning(
                    "Activity logging failed for dm_sent (%s -> %s): %s",
                    self.anima_name, to, e,
                )

        # Parallel write to legacy dm_logs/ (fallback data source)
        try:
            self._append_dm_log(to, content, intent=intent)
        except Exception:
            pass  # Never fail the send itself

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
        channels_dir = self.shared_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        filepath = channels_dir / f"{channel}.jsonl"
        entry = json.dumps({
            "ts": datetime.now().isoformat(),
            "from": from_name or self.anima_name,
            "text": text,
            "source": source,
        }, ensure_ascii=False)
        try:
            with filepath.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
            logger.info("Channel post: %s -> #%s", self.anima_name, channel)
        except OSError:
            logger.warning("Failed to post to channel: %s", channel)

    def read_channel(
        self, channel: str, limit: int = 20, human_only: bool = False,
    ) -> list[dict]:
        """Read recent messages from a shared channel."""
        _validate_name(channel, "channel name")
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
    ) -> list[dict]:
        """Read messages mentioning @name from a shared channel."""
        _validate_name(channel, "channel name")
        target = name or self.anima_name
        mention_tag = f"@{target}"
        all_msgs = self.read_channel(channel, limit=1000)
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
                    types=["dm_sent", "dm_received"],
                    involving=peer,
                )
                for e in recent:
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
            "ts": datetime.now().isoformat(),
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
        messages: list[Message] = []
        for f in sorted(self.inbox_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                messages.append(Message(**data))
            except Exception as e:
                logger.error("Failed to parse message %s: %s", f, e)
        return messages

    def receive_with_paths(self) -> list[InboxItem]:
        """Read all unread messages, returning Message + file path pairs.

        Unlike receive(), the caller is responsible for archiving
        via archive_paths() after processing.  Messages that arrive
        while the caller is working will remain in the inbox.
        """
        items: list[InboxItem] = []
        for f in sorted(self.inbox_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                items.append(InboxItem(msg=Message(**data), path=f))
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
            # E: Send read ACK to senders (skip ACK for ack/board_mention to prevent loops)
            non_ack_messages = [m for m in messages if m.type not in ("ack", "board_mention")]
            if non_ack_messages:
                senders: dict[str, list[Message]] = {}
                for m in non_ack_messages:
                    senders.setdefault(m.from_person, []).append(m)
                for sender, sender_msgs in senders.items():
                    summary = ", ".join(m.content[:50] for m in sender_msgs[:3])
                    if len(sender_msgs) > 3:
                        summary += f" (+{len(sender_msgs) - 3}件)"
                    try:
                        self.send(
                            to=sender,
                            content=f"[既読通知] {len(sender_msgs)}件のメッセージを受信しました: {summary}",
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
        msg = Message(
            from_person=f"{source}:{external_user_id}" if external_user_id else source,
            to_person=self.anima_name,
            content=content,
            source=source,
            source_message_id=source_message_id,
            external_user_id=external_user_id,
            external_channel_id=external_channel_id,
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
    ) -> Message:
        """Async wrapper for filesystem-based send."""
        return self.send(
            to=to,
            content=content,
            msg_type=msg_type,
            thread_id=thread_id,
            reply_to=reply_to,
            intent=intent,
        )
