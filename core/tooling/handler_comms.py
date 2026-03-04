from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CommsToolsMixin — messaging, channel, DM history, and human notification handlers."""

import json as _json
import logging
import re
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.time_utils import now_iso

from core.tooling.handler_base import (
    OnMessageSentFn,
    _error_result,
    active_session_type,
    build_outgoing_origin_chain,
    suppress_board_fanout,
)

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger
    from core.messenger import Messenger
    from core.notification.notifier import HumanNotifier

logger = logging.getLogger("animaworks.tool_handler")


class CommsToolsMixin:
    """Message sending, channel posting/reading, DM history, and human notification."""

    # Declared for type-checker visibility
    _messenger: Messenger | None
    _anima_name: str
    _activity: ActivityLogger
    _on_message_sent: OnMessageSentFn | None
    _replied_to: dict[str, set[str]]
    _posted_channels: dict[str, set[str]]
    _human_notifier: HumanNotifier | None
    _pending_notifications: list[dict[str, Any]]
    _session_origin: str
    _session_origin_chain: list[str]

    def _handle_send_message(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"

        to = args["to"]
        content = args["content"]
        intent = args.get("intent", "")

        # ── Per-run DM limits ──
        if intent not in ("report", "delegation", "question"):
            return t("handler.dm_intent_error")

        current_replied = self.replied_to_for(active_session_type.get())
        if to in current_replied:
            return t("handler.dm_already_sent", to=to)

        if len(current_replied) >= 2 and to not in current_replied:
            return t("handler.dm_max_recipients")

        # ── Resolve recipient ──
        try:
            from core.outbound import resolve_recipient, send_external
            from core.config.models import load_config
            from core.paths import get_animas_dir

            config = load_config()
            animas_dir = get_animas_dir()
            known_animas = {
                d.name for d in animas_dir.iterdir() if d.is_dir()
            } if animas_dir.exists() else set()

            resolved = resolve_recipient(
                to, known_animas, config.external_messaging,
            )
        except (ValueError, Exception) as e:
            from core.exceptions import RecipientNotFoundError
            if isinstance(e, (ValueError, RecipientNotFoundError)):
                session = active_session_type.get()
                if session == "chat":
                    return (
                        f"宛先 '{to}' には send_message で送信できません。"
                        "チャット中は直接テキストで返答すれば人間ユーザーに届きます。"
                        "send_message は他のAnima宛てにのみ使用してください。"
                    )
                return (
                    f"宛先 '{to}' には send_message で送信できません。"
                    "人間への連絡は call_human を使用してください。"
                    "send_message は他のAnima宛てにのみ使用してください。"
                )
            logger.warning(
                "Recipient resolution failed for '%s': %s",
                to, e, exc_info=True,
            )
            return _error_result(
                "RecipientResolutionError",
                f"Failed to resolve recipient '{to}': {e}",
                suggestion="Check config.json external_messaging settings",
            )

        # ── Build outgoing origin_chain (provenance Phase 3) ──
        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin, self._session_origin_chain,
        )

        # ── External routing ──
        if resolved is not None and not resolved.is_internal:
            logger.info(
                "send_message routed externally: to=%s channel=%s",
                to, resolved.channel,
            )
            self._replied_to.setdefault(active_session_type.get(), set()).add(to)
            self._persist_replied_to(to, success=True)

            msg = self._messenger.send(
                to=to,
                content=content,
                thread_id=args.get("thread_id", ""),
                reply_to=args.get("reply_to", ""),
                intent=intent,
                origin_chain=outgoing_chain,
            )

            if self._on_message_sent:
                try:
                    self._on_message_sent(
                        self._messenger.anima_name, to, content,
                    )
                except Exception:
                    logger.exception("on_message_sent callback failed")

            from core.outbound import send_external
            result = send_external(
                resolved, content, sender_name=self._anima_name,
            )
            return result

        # ── Internal messaging ──
        internal_to = resolved.name if resolved else to
        msg = self._messenger.send(
            to=internal_to,
            content=content,
            thread_id=args.get("thread_id", ""),
            reply_to=args.get("reply_to", ""),
            intent=intent,
            origin_chain=outgoing_chain,
        )

        if msg.type == "error":
            return f"Error: {msg.content}"

        logger.info("send_message to=%s thread=%s", internal_to, msg.thread_id)
        self._replied_to.setdefault(active_session_type.get(), set()).add(internal_to)
        self._persist_replied_to(internal_to, success=True)

        if self._on_message_sent:
            try:
                self._on_message_sent(
                    self._messenger.anima_name, internal_to, content,
                )
            except Exception:
                logger.exception("on_message_sent callback failed")

        return f"Message sent to {internal_to} (id: {msg.id}, thread: {msg.thread_id})"

    # ── Channel tool handlers ────────────────────────────────

    def _handle_post_channel(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        text = args.get("text", "")
        if not channel or not text:
            return _error_result("InvalidArguments", "channel and text are required")

        # ── ACL gate ──
        from core.messenger import is_channel_member
        if not is_channel_member(self._messenger.shared_dir, channel, self._anima_name):
            return t("handler.channel_acl_denied", channel=channel)

        current_posted = self.posted_channels_for(active_session_type.get())
        if channel in current_posted:
            alt_channels = {"general", "ops"} - {channel} - current_posted
            alt_hint = ""
            if alt_channels:
                alt_hint = t(
                    "handler.post_alt_hint",
                    channels=", ".join(f"#{c}" for c in sorted(alt_channels)),
                )
            return t(
                "handler.post_already_posted",
                channel=channel,
                alt_hint=alt_hint,
            )

        # ── Cross-run guard: file-based cooldown check ──
        try:
            from core.config.models import load_config
            cooldown = load_config().heartbeat.channel_post_cooldown_s
        except Exception:
            cooldown = 300
        if cooldown > 0:
            last = self._messenger.last_post_by(self._anima_name, channel)
            if last:
                from datetime import datetime
                from core.time_utils import ensure_aware, now_jst
                try:
                    ts = ensure_aware(datetime.fromisoformat(last["ts"]))
                    elapsed = (now_jst() - ts).total_seconds()
                    if elapsed < cooldown:
                        return t(
                            "handler.post_cooldown",
                            channel=channel,
                            ts=last["ts"][11:16],
                            elapsed=int(elapsed),
                            cooldown=cooldown,
                        )
                except (ValueError, TypeError):
                    pass

        self._messenger.post_channel(channel, text)
        self._posted_channels.setdefault(active_session_type.get(), set()).add(channel)
        logger.info("post_channel channel=%s anima=%s", channel, self._anima_name)

        if not suppress_board_fanout.get():
            self._fanout_board_mentions(channel, text)
        else:
            logger.info(
                "Suppressed board fanout for board_mention reply: channel=%s anima=%s",
                channel, self._anima_name,
            )

        return f"Posted to #{channel}"

    def _fanout_board_mentions(self, channel: str, text: str) -> None:
        """Send DM notifications to mentioned Animas when posting to a board channel."""
        if not self._messenger:
            return

        mentions = re.findall(r"@(\w+)", text)
        if not mentions:
            return

        is_all = "all" in mentions

        from core.paths import get_data_dir
        sockets_dir = get_data_dir() / "run" / "sockets"
        if sockets_dir.exists():
            running = {p.stem for p in sockets_dir.glob("*.sock")}
        else:
            running = set()

        if is_all:
            targets = running - {self._anima_name}
        else:
            named = {m for m in mentions if m != "all"}
            targets = (named & running) - {self._anima_name}

        # ── ACL filter: only notify channel members ──
        from core.messenger import is_channel_member
        targets = {
            t for t in targets
            if is_channel_member(self._messenger.shared_dir, channel, t)
        }

        if not targets:
            return

        from_name = self._anima_name
        fanout_content = (
            f"[board_reply:channel={channel},from={from_name}]\n"
            + t("handler.board_mention_content", from_name=from_name, channel=channel, text=text)
        )

        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin, self._session_origin_chain,
        )

        for target in sorted(targets):
            try:
                self._messenger.send(
                    to=target,
                    content=fanout_content,
                    msg_type="board_mention",
                    origin_chain=outgoing_chain,
                )
                logger.info(
                    "board_mention fanout: %s -> %s (channel=%s)",
                    from_name, target, channel,
                )
            except Exception:
                logger.warning(
                    "Failed to fanout board_mention to %s", target, exc_info=True,
                )

    def _handle_read_channel(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        if not channel:
            return _error_result("InvalidArguments", "channel is required")

        # ── ACL gate ──
        from core.messenger import is_channel_member
        if not is_channel_member(self._messenger.shared_dir, channel, self._anima_name):
            return t("handler.channel_acl_denied", channel=channel)

        limit = args.get("limit", 20)
        human_only = args.get("human_only", False)
        messages = self._messenger.read_channel(channel, limit=limit, human_only=human_only)
        if not messages:
            return f"No messages in #{channel}"
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    def _handle_read_dm_history(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        peer = args.get("peer", "")
        if not peer:
            return _error_result("InvalidArguments", "peer is required")
        limit = args.get("limit", 20)
        messages = self._messenger.read_dm_history(peer, limit=limit)
        if not messages:
            return f"No DM history with {peer}"
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    # ── Channel management handler ────────────────────────────

    def _handle_manage_channel(self, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"

        action = args.get("action", "")
        channel = args.get("channel", "")
        if not action or not channel:
            return _error_result("InvalidArguments", "action and channel are required")

        from core.messenger import (
            ChannelMeta,
            load_channel_meta,
            save_channel_meta,
            is_channel_member,
            _validate_name,
        )
        from core.exceptions import RecipientNotFoundError

        try:
            _validate_name(channel, "channel name")
        except RecipientNotFoundError:
            return _error_result("InvalidArguments", f"Invalid channel name: {channel!r}")

        shared_dir = self._messenger.shared_dir

        if action == "create":
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            if channel_file.exists():
                return t("handler.channel_already_exists", channel=channel)
            members = args.get("members", [])
            if self._anima_name not in members:
                members = [self._anima_name] + members
            meta = ChannelMeta(
                members=members,
                created_by=self._anima_name,
                created_at=now_iso(),
                description=args.get("description", ""),
            )
            channels_dir = shared_dir / "channels"
            channels_dir.mkdir(parents=True, exist_ok=True)
            channel_file.write_text("", encoding="utf-8")
            save_channel_meta(shared_dir, channel, meta)
            members_str = ", ".join(members) if members else "open"
            logger.info("manage_channel create: #%s by %s", channel, self._anima_name)
            return t("handler.channel_created", channel=channel, members=members_str)

        elif action == "add_member":
            meta = load_channel_meta(shared_dir, channel)
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            if not channel_file.exists():
                return t("handler.channel_not_found", channel=channel)
            new_members = args.get("members", [])
            if not new_members:
                return _error_result("InvalidArguments", "members list is required for add_member")
            # Reject add_member on open/legacy channels to prevent accidental restriction
            if meta is None:
                return t("handler.channel_add_member_open_denied", channel=channel)
            # Caller must be a member of the channel
            if not is_channel_member(shared_dir, channel, self._anima_name):
                return t("handler.channel_acl_not_member", channel=channel)
            for m in new_members:
                if m not in meta.members:
                    meta.members.append(m)
            save_channel_meta(shared_dir, channel, meta)
            logger.info("manage_channel add_member: #%s += %s", channel, new_members)
            return t("handler.channel_members_added", channel=channel, members=", ".join(new_members))

        elif action == "remove_member":
            meta = load_channel_meta(shared_dir, channel)
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            if not channel_file.exists():
                return t("handler.channel_not_found", channel=channel)
            if meta is None:
                return t("handler.channel_open", channel=channel)
            # Caller must be a member of the channel
            if not is_channel_member(shared_dir, channel, self._anima_name):
                return t("handler.channel_acl_not_member", channel=channel)
            remove_members = args.get("members", [])
            if not remove_members:
                return _error_result("InvalidArguments", "members list is required for remove_member")
            meta.members = [m for m in meta.members if m not in remove_members]
            save_channel_meta(shared_dir, channel, meta)
            logger.info("manage_channel remove_member: #%s -= %s", channel, remove_members)
            return t("handler.channel_members_removed", channel=channel, members=", ".join(remove_members))

        elif action == "info":
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            if not channel_file.exists():
                return t("handler.channel_not_found", channel=channel)
            meta = load_channel_meta(shared_dir, channel)
            if meta is None or not meta.members:
                return t("handler.channel_open", channel=channel)
            info = {
                "channel": channel,
                "members": meta.members,
                "created_by": meta.created_by,
                "created_at": meta.created_at,
                "description": meta.description,
            }
            return _json.dumps(info, ensure_ascii=False, indent=2)

        else:
            return _error_result(
                "InvalidArguments",
                f"Unknown action: {action!r}. Use create, add_member, remove_member, or info.",
            )

    # ── Human notification handler ────────────────────────────

    def _handle_call_human(self, args: dict[str, Any]) -> str:
        if not self._human_notifier:
            return _error_result(
                "NotConfigured",
                "Human notification is not configured",
                suggestion="Enable human_notification in config.json",
            )
        if self._human_notifier.channel_count == 0:
            return _error_result(
                "NotConfigured",
                "No notification channels configured",
                suggestion="Add channels to human_notification.channels in config.json",
            )

        import asyncio

        subject = args.get("subject", "")
        body = args.get("body", "")
        priority = args.get("priority", "normal")

        if not subject or not body:
            return _error_result(
                "InvalidArguments",
                "subject and body are required",
            )

        try:
            coro = self._human_notifier.notify(
                subject, body, priority,
                anima_name=self._anima_name,
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(asyncio.run, coro).result(timeout=60)
            else:
                results = asyncio.run(coro)
        except Exception as e:
            return _error_result("NotificationError", f"Failed to send notification: {e}")

        notif_data = {
            "anima": self._anima_name,
            "subject": subject,
            "body": body,
            "priority": priority,
            "timestamp": now_iso(),
        }
        self._pending_notifications.append(notif_data)

        return _json.dumps(
            {"status": "sent", "results": results},
            ensure_ascii=False,
        )
