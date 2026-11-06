from __future__ import annotations

from core.tooling._handler_protocols import _CommsToolsHost

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CommsToolsMixin — messaging, channel, DM history, and human notification handlers."""

import inspect
import json as _json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.i18n import t
from core.time_utils import now_iso
from core.tooling.handler_base import (
    OnMessageSentFn,
    _error_result,
    active_session_type,
    build_outgoing_origin_chain,
    meeting_context,
    meeting_mode,
    record_meeting_redirect,
    suppress_board_fanout,
)

if TYPE_CHECKING:
    from core.activity.logger import ActivityLogger
    from core.messaging.messenger import Messenger
    from core.notification.notifier import HumanNotifier

logger = logging.getLogger("animaworks.tool_handler")


def _notify_message_sent(
    callback: OnMessageSentFn | None,
    from_person: str,
    to_person: str,
    content: str,
    *,
    message_id: str = "",
) -> None:
    """Invoke message callbacks, passing an ID when their signature accepts it."""
    if callback is None:
        return
    try:
        parameters = inspect.signature(callback).parameters.values()
        accepts_message_id = any(
            parameter.name == "message_id" or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_message_id = False
    if accepts_message_id:
        cast(Any, callback)(from_person, to_person, content, message_id=message_id)
    else:
        callback(from_person, to_person, content)


def _company_boundary_error(
    from_anima: str,
    to_anima: str,
    *,
    animas_dir: Any,
) -> str | None:
    """Return a stable user-facing error when a company boundary blocks access."""
    from core.org.company import check_company_boundary

    boundary = check_company_boundary(from_anima, to_anima, animas_dir=animas_dir)
    if not boundary.cross_company:
        return None
    if boundary.resolved_via == "fail_closed":
        return t("handler.company_boundary_unverifiable")
    return t(
        "handler.cross_company_message_blocked",
        display_name=boundary.display_name,
    )


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

    @staticmethod
    def _resolve_meeting_participant(to: str, participants: list[Any]) -> str:
        participant_map = {str(name).lower(): str(name) for name in participants if str(name)}
        return participant_map.get(to.lower(), "")

    def _handle_send_message(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"

        from core.tooling.org_helpers import resolve_anima_name

        to = resolve_anima_name(args["to"]) if args.get("to") else args.get("to", "")
        content = args["content"]
        intent = args.get("intent", "")

        if to == self._anima_name:
            return t("handler.dm_self_addressed_error")

        # ── Per-run DM limits ──
        if intent == "delegation":
            return t("handler.delegation_intent_deprecated")
        if intent not in ("report", "question"):
            return t("handler.dm_intent_error")

        current_replied = self.replied_to_for(active_session_type.get())
        meeting_target = ""
        if meeting_mode.get():
            ctx = meeting_context.get() or {}
            participants = ctx.get("participants", []) if isinstance(ctx, dict) else []
            meeting_target = self._resolve_meeting_participant(to, participants)
            if not meeting_target:
                return t("handler.meeting_tool_blocked", tool="send_message")

        effective_to = meeting_target or to
        if effective_to in current_replied:
            return t("handler.dm_already_sent", to=effective_to)

        if meeting_target:
            from core.paths import get_animas_dir

            animas_dir = get_animas_dir()
            company_error = _company_boundary_error(
                self._anima_name,
                meeting_target,
                animas_dir=animas_dir,
            )
            if company_error is not None:
                return company_error
            try:
                record_meeting_redirect(
                    from_name=self._anima_name,
                    to_name=meeting_target,
                    content=content,
                    intent=intent,
                )
            except Exception as e:
                logger.warning("Meeting redirect delivery failed: %s -> %s: %s", self._anima_name, meeting_target, e)
                return _error_result("DeliveryFailed", f"Failed to deliver meeting redirect: {e}")
            self._replied_to.setdefault(active_session_type.get(), set()).add(meeting_target)
            self._persist_replied_to(meeting_target, success=True)
            try:
                self._activity.log(
                    "meeting_redirect_sent",
                    content=content,
                    to_person=meeting_target,
                    summary=f"meeting → {meeting_target}: {content[:80]}",
                    meta={"intent": intent, "delivery": "meeting"},
                )
            except Exception:
                logger.warning("Activity logging failed for meeting redirect to %s", meeting_target)
            return t("handler.meeting_dm_redirected", to=meeting_target)

        # ── Resolve recipient ──
        try:
            from core.config.models import load_config
            from core.messaging.outbound import resolve_recipient, send_external
            from core.paths import get_animas_dir

            config = load_config()
            animas_dir = get_animas_dir()
            known_animas = {d.name for d in animas_dir.iterdir() if d.is_dir()} if animas_dir.exists() else set()

            resolved = resolve_recipient(
                to,
                known_animas,
                config.external_messaging,
            )
        except (ValueError, Exception) as e:
            from core.exceptions import RecipientNotFoundError

            if isinstance(e, (ValueError, RecipientNotFoundError)):
                session = active_session_type.get()
                if session == "chat":
                    return t("handler.send_msg_chat_hint", to=to)
                return t("handler.send_msg_non_chat_hint", to=to)
            logger.warning(
                "Recipient resolution failed for '%s': %s",
                to,
                e,
                exc_info=True,
            )
            return _error_result(
                "RecipientResolutionError",
                f"Failed to resolve recipient '{to}': {e}",
                suggestion="Check config.json external_messaging settings",
            )

        if resolved is None or resolved.is_internal:
            internal_to = resolved.name if resolved is not None else to
            company_error = _company_boundary_error(
                self._anima_name,
                internal_to,
                animas_dir=animas_dir,
            )
            if company_error is not None:
                return company_error

        # ── Build outgoing origin_chain (provenance Phase 3) ──
        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin,
            self._session_origin_chain,
        )

        # ── External routing ──
        if resolved is not None and not resolved.is_internal:
            logger.info(
                "send_message routed externally: to=%s channel=%s",
                to,
                resolved.channel,
            )
            self._replied_to.setdefault(active_session_type.get(), set()).add(to)
            self._persist_replied_to(to, success=True)

            try:
                meta: dict[str, Any] = {"from_type": "external", "channel": resolved.channel}
                if intent:
                    meta["intent"] = intent
                self._activity.log(
                    "message_sent",
                    content=content,
                    to_person=to,
                    summary=f"→ {to}: {content[:80]}",
                    meta=meta,
                )
            except Exception:
                logger.warning("Activity logging failed for external send to %s", to)

            try:
                _notify_message_sent(
                    self._on_message_sent,
                    self._messenger.anima_name,
                    to,
                    content,
                )
            except Exception:
                logger.exception("on_message_sent callback failed")

            from core.messaging.outbound import send_external

            result = send_external(
                resolved,
                content,
                sender_name=self._anima_name,
                anima_name=self._anima_name,
            )
            feedback = self._build_send_feedback(to)
            if feedback:
                return f"{result}\n{feedback}"
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

        try:
            _notify_message_sent(
                self._on_message_sent,
                self._messenger.anima_name,
                internal_to,
                content,
                message_id=msg.id,
            )
        except Exception:
            logger.exception("on_message_sent callback failed")

        base = f"Message sent to {internal_to} (id: {msg.id}, thread: {msg.thread_id})"
        extras = [
            line
            for line in (self._build_recipient_busy_notice(internal_to), self._build_send_feedback(internal_to))
            if line
        ]
        if extras:
            return "\n".join([base, *extras])
        return base

    def _build_recipient_busy_notice(self: _CommsToolsHost, to: str) -> str:
        """Tell the sender when *to* is stuck in a long inbox run and will read later."""
        try:
            from core.messaging.recipient_status import read_recipient_inbox_busy
            from core.time_utils import now_local

            busy = read_recipient_inbox_busy(Path(self._messenger.shared_dir), to)
            if busy is None:
                return ""
            minutes = int((now_local() - busy.since).total_seconds() // 60)
            notice = t(
                "handler.send_recipient_inbox_busy",
                to=to,
                since=busy.since.strftime("%H:%M"),
                elapsed=f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m",
                count=busy.unread_count,
            )
            if busy.oldest_unread_at is not None:
                notice += t(
                    "handler.send_recipient_inbox_oldest",
                    oldest=busy.oldest_unread_at.strftime("%H:%M"),
                )
            return notice
        except Exception:
            logger.debug("Recipient busy notice failed for %s", to, exc_info=True)
            return ""

    def _build_send_feedback(self: _CommsToolsHost, to: str) -> str:
        """Build behavioral feedback showing recent send history to the same recipient.

        Returns a short summary of how many messages were sent to *to* in the
        last 24 hours (rolling window) plus the most recent 3 summaries.
        Returns empty string on any failure or when count is 0.
        """
        try:
            from datetime import datetime, timedelta

            from core.activity.logger import ActivityLogger
            from core.time_utils import ensure_aware, now_local

            activity = ActivityLogger(self._anima_dir)
            entries = activity.recent(
                days=2,
                limit=500,
                types=["message_sent"],
                involving=to,
            )
            cutoff = now_local() - timedelta(hours=24)
            sent_entries = []
            for e in entries:
                if (getattr(e, "to_person", None) or "") != to:
                    continue
                try:
                    ts_dt = ensure_aware(datetime.fromisoformat(e.ts))
                    if ts_dt >= cutoff:
                        sent_entries.append(e)
                except (ValueError, TypeError):
                    continue
            count = len(sent_entries)
            if count == 0:
                return ""

            recent_summaries = sent_entries[-3:]
            lines = [t("handler.send_feedback_header", to=to, count=count)]
            for e in reversed(recent_summaries):
                ts_short = e.ts[11:16] if len(e.ts) >= 16 else e.ts
                preview = (e.summary or e.content or "")[:80]
                if preview.startswith(f"→ {to}: "):
                    preview = preview[len(f"→ {to}: ") :]
                lines.append(f"  {ts_short} {preview}")
            return "\n".join(lines)
        except Exception:
            logger.debug("Failed to build send feedback for %s", to, exc_info=True)
            return ""

    def _cross_company_communication_error(self: _CommsToolsHost, peers: list[str]) -> str | None:
        """Return the standard DM boundary error for the first cross-company peer."""
        animas_dir = self._anima_dir.parent
        for peer in sorted(set(peers)):
            company_error = _company_boundary_error(
                self._anima_name,
                peer,
                animas_dir=animas_dir,
            )
            if company_error is not None:
                return company_error
        return None

    def _channel_company_boundary_error(self: _CommsToolsHost, channel: str) -> str | None:
        """Enforce company scope for channel post/read access.

        Decision order:
        1. Restricted channels (non-empty members) → pair-check against members
        2. Company-scoped open channels (meta.company set) → same-company only
        3. Legacy open channels (no meta / empty members & company) → deny
           company-assigned animas until company is configured; unassigned
           animas keep legacy unrestricted access
        """
        if not self._messenger:
            return None

        from core.messaging.messenger import load_channel_meta
        from core.org.company import get_company, get_company_display_name

        meta = load_channel_meta(self._messenger.shared_dir, channel)
        animas_dir = self._anima_dir.parent
        data_dir = animas_dir.parent

        # Restricted channel: keep existing member pair-check.
        if meta is not None and meta.members:
            return self._cross_company_communication_error(meta.members)

        # Company-scoped open channel.
        company = (meta.company if meta is not None else "") or ""
        if company:
            poster_company = get_company(self._anima_name, animas_dir=animas_dir)
            if poster_company is None:
                return None  # unassigned anima: legacy unrestricted
            if poster_company == company:
                return None
            display_name = get_company_display_name(company, data_dir=data_dir)
            return t(
                "handler.cross_company_message_blocked",
                display_name=display_name,
            )

        # Legacy open channel without company attribution.
        # closed channels are already denied by ACL; leave boundary alone.
        poster_company = get_company(self._anima_name, animas_dir=animas_dir)
        if poster_company is None:
            return None  # unassigned anima: legacy unrestricted
        return t("handler.channel_company_unset", channel=channel)

    # ── Channel tool handlers ────────────────────────────────

    def _handle_post_channel(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        self._last_post_channel_succeeded = False
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        text = args.get("text", "")
        if not channel or not text:
            return _error_result("InvalidArguments", "channel and text are required")

        # ── ACL gate ──
        from core.messaging.messenger import is_channel_member

        if not is_channel_member(self._messenger.shared_dir, channel, self._anima_name):
            return t("handler.channel_acl_denied", channel=channel)
        company_error = self._channel_company_boundary_error(channel)
        if company_error is not None:
            return company_error

        session_type = active_session_type.get()
        current_posted = self.posted_channels_for(session_type)
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

        from core.exceptions import ChannelAccessDeniedError, ChannelNotFoundError

        try:
            self._messenger.post_channel(channel, text)
        except ChannelNotFoundError:
            return t("handler.channel_not_found", channel=channel)
        except ChannelAccessDeniedError:
            return t("handler.channel_acl_denied", channel=channel)
        except ValueError as exc:
            return _error_result("InvalidArguments", str(exc))

        self._last_post_channel_succeeded = True
        self._posted_channels.setdefault(session_type, set()).add(channel)
        logger.info("post_channel channel=%s anima=%s", channel, self._anima_name)

        if not suppress_board_fanout.get():
            self._fanout_board_mentions(channel, text)
        else:
            logger.info(
                "Suppressed board fanout for board_mention reply: channel=%s anima=%s",
                channel,
                self._anima_name,
            )

        # Sync board post to mapped Slack channel (fire-and-forget)
        self._fire_board_slack_sync(channel, text)

        try:
            _notify_message_sent(
                self._on_message_sent,
                self._messenger.anima_name,
                f"#channel:{channel}",
                text,
            )
        except Exception:
            logger.exception("on_message_sent callback failed for board post")

        return f"Posted to #{channel}"

    def _fanout_board_mentions(self: _CommsToolsHost, channel: str, text: str) -> None:
        """Delegate mention delivery to the shared Board fan-out implementation."""
        if not self._messenger:
            return

        from core.messaging.board_fanout import fanout_board_mentions

        fanout_board_mentions(
            self._messenger,
            self._anima_name,
            channel,
            text,
            origin_chain=build_outgoing_origin_chain(
                self._session_origin,
                self._session_origin_chain,
            ),
            animas_dir=Path(self._anima_dir).parent,
        )

    def _fire_board_slack_sync(self: _CommsToolsHost, channel: str, text: str) -> None:
        """Sync a board post to the mapped Slack channel.

        The MCP tool handler dispatches ``handle()`` via
        ``asyncio.to_thread()``, so this method runs in a thread-pool
        worker with no running event loop.  We therefore create a
        short-lived loop via ``asyncio.run()`` to execute the async
        HTTP call.  The ~1 s blocking is acceptable because we are
        already off the main event loop.
        """
        from core.platform.env import get_env

        if get_env("ANIMAWORKS_DISABLE_EXTERNAL_SYNC") == "1":
            return

        try:
            import asyncio

            from core.messaging.outbound_auto import BoardSlackSync

            sync = BoardSlackSync()
            coro = sync.sync_board_post(
                board_name=channel,
                text=text,
                from_person=self._anima_name,
                source="anima",
            )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Inside an async context — schedule as task. Cycle-context
                # inheritance is intentional: this is the outbound delivery of a
                # board post the agent just made this cycle, so the sync (and its
                # failure logs) belong to that cycle. It is a short-lived one-shot,
                # not detached later maintenance.
                loop.create_task(coro)
            else:
                # Sync context (MCP tool handler thread pool) —
                # create a short-lived event loop to run the coroutine.
                asyncio.run(coro)
        except Exception:
            logger.warning("Board→Slack sync failed for #%s", channel, exc_info=True)

    def _handle_read_channel(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        channel = args.get("channel", "")
        if not channel:
            return _error_result("InvalidArguments", "channel is required")
        _ch_lower = channel.strip().lower()
        if _ch_lower == "inbox" or _ch_lower.startswith("inbox/") or _ch_lower.startswith("inbox\\"):
            return _error_result(
                "InvalidArguments",
                f"'{channel}' is not a channel. Inbox messages are processed automatically; "
                "do not use read_channel for inbox.",
            )

        # ── ACL gate ──
        from core.exceptions import RecipientNotFoundError
        from core.messaging.messenger import _validate_name, is_channel_member

        try:
            _validate_name(channel, "channel name")
        except RecipientNotFoundError:
            return _error_result("InvalidArguments", f"Invalid channel name: {channel!r}")
        if not is_channel_member(self._messenger.shared_dir, channel, self._anima_name):
            return t("handler.channel_acl_denied", channel=channel)
        company_error = self._channel_company_boundary_error(channel)
        if company_error is not None:
            return company_error

        limit = args.get("limit", 20)
        human_only = args.get("human_only", False)
        messages = self._messenger.read_channel(channel, limit=limit, human_only=human_only)
        if not messages:
            return f"No messages in #{channel}"
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    def _handle_read_dm_history(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"
        peer = args.get("peer", "")
        if not peer:
            return _error_result("InvalidArguments", "peer is required")
        company_error = self._cross_company_communication_error([peer])
        if company_error is not None:
            return company_error
        limit = args.get("limit", 20)
        direction = args.get("direction", "both")
        hours = args.get("hours")
        keyword = args.get("keyword")
        messages = self._messenger.read_dm_history(
            peer,
            limit=limit,
            direction=direction,
            hours=hours,
            keyword=keyword,
        )
        if not messages:
            return f"No DM history with {peer}"
        return _json.dumps(messages, ensure_ascii=False, indent=2)

    # ── Channel management handler ────────────────────────────

    def _handle_manage_channel(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        if not self._messenger:
            return "Error: messenger not configured"

        action = args.get("action", "")
        channel = args.get("channel", "")
        if not action or not channel:
            return _error_result("InvalidArguments", "action and channel are required")

        from core.exceptions import RecipientNotFoundError
        from core.messaging.messenger import (
            ChannelMeta,
            _validate_name,
            is_channel_member,
            load_channel_meta,
            save_channel_meta,
            update_channel_meta,
        )

        try:
            _validate_name(channel, "channel name")
        except RecipientNotFoundError:
            return _error_result("InvalidArguments", f"Invalid channel name: {channel!r}")

        shared_dir = self._messenger.shared_dir

        if action == "create":
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            meta = load_channel_meta(shared_dir, channel)
            if meta is not None and meta.closed:
                return t("handler.channel_archived_name_reserved", channel=channel)
            if channel_file.exists():
                return t("handler.channel_already_exists", channel=channel)
            members = args.get("members", [])
            if self._anima_name not in members:
                members = [self._anima_name] + members
            company_error = self._cross_company_communication_error(members)
            if company_error is not None:
                return company_error
            from core.org.company import get_company

            creator_company = get_company(self._anima_name, animas_dir=self._anima_dir.parent) or ""
            meta = ChannelMeta(
                members=members,
                created_by=self._anima_name,
                created_at=now_iso(),
                description=args.get("description", ""),
                company=creator_company,
            )
            channels_dir = shared_dir / "channels"
            channels_dir.mkdir(parents=True, exist_ok=True)
            channel_file.touch(exist_ok=True)
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
            company_error = self._cross_company_communication_error(new_members)
            if company_error is not None:
                return company_error

            def _add_members(current: ChannelMeta | None) -> ChannelMeta | None:
                if current is None:
                    return None
                current.members.extend(member for member in new_members if member not in current.members)
                return current

            updated_meta = update_channel_meta(shared_dir, channel, _add_members)
            if updated_meta is None:
                return t("handler.channel_not_found", channel=channel)
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

            def _remove_members(current: ChannelMeta | None) -> ChannelMeta | None:
                if current is None:
                    return None
                current.members = [member for member in current.members if member not in remove_members]
                return current

            updated_meta = update_channel_meta(shared_dir, channel, _remove_members)
            if updated_meta is None:
                return t("handler.channel_not_found", channel=channel)
            logger.info("manage_channel remove_member: #%s -= %s", channel, remove_members)
            return t("handler.channel_members_removed", channel=channel, members=", ".join(remove_members))

        elif action == "archive":
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            from core.platform.locks import locked_path

            lock_path = channel_file.with_name(f"{channel_file.name}.lock")
            try:
                with locked_path(lock_path, exclusive=True, thread_lock=True):
                    meta = load_channel_meta(shared_dir, channel)
                    if not channel_file.exists() and meta is None:
                        return t("handler.channel_not_found", channel=channel)
                    company_error = self._channel_company_boundary_error(channel)
                    if company_error is not None:
                        return company_error
                    if meta is not None and meta.members and self._anima_name not in meta.members:
                        return t("handler.channel_acl_not_member", channel=channel)
                    if meta is not None and meta.closed and not channel_file.exists():
                        return _json.dumps(
                            {
                                "action": "archive",
                                "channel": channel,
                                "archived": False,
                                "already_archived": True,
                                "closed": True,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )

                    def _close_channel(current: ChannelMeta | None) -> ChannelMeta:
                        if current is None:
                            from core.org.company import get_company

                            return ChannelMeta(
                                members=[],
                                created_by=self._anima_name,
                                created_at=now_iso(),
                                closed=True,
                                company=get_company(self._anima_name, animas_dir=self._anima_dir.parent) or "",
                            )
                        current.closed = True
                        return current

                    closed_meta = update_channel_meta(shared_dir, channel, _close_channel, create_if_missing=True)
                    if closed_meta is None:
                        return _error_result("ArchiveFailed", f"Could not write channel tombstone for #{channel}")

                    archived_path = None
                    if channel_file.exists():
                        archive_dir = shared_dir / "channels" / "archive"
                        archive_dir.mkdir(parents=True, exist_ok=True)
                        archived_path = archive_dir / channel_file.name
                        if archived_path.exists():
                            stem, suffix = channel_file.stem, channel_file.suffix
                            counter = 1
                            while archived_path.exists():
                                archived_path = archive_dir / f"{stem}_{counter}{suffix}"
                                counter += 1
                        channel_file.rename(archived_path)
            except OSError as exc:
                logger.warning("manage_channel archive failed: #%s: %s", channel, exc)
                return _error_result("ArchiveFailed", f"Failed to archive channel #{channel}: {exc}")

            result: dict[str, Any] = {
                "action": "archive",
                "channel": channel,
                "archived": True,
                "closed": True,
            }
            if archived_path is not None:
                result["to"] = str(archived_path.relative_to(shared_dir))
            logger.info("manage_channel archive: #%s by %s", channel, self._anima_name)
            return _json.dumps(result, ensure_ascii=False, indent=2)

        elif action == "info":
            channel_file = shared_dir / "channels" / f"{channel}.jsonl"
            if not channel_file.exists():
                return t("handler.channel_not_found", channel=channel)
            meta = load_channel_meta(shared_dir, channel)
            if meta is None or (not meta.members and not meta.closed):
                return t("handler.channel_open", channel=channel)
            info = {
                "channel": channel,
                "members": meta.members,
                "closed": meta.closed,
                "created_by": meta.created_by,
                "created_at": meta.created_at,
                "description": meta.description,
            }
            return _json.dumps(info, ensure_ascii=False, indent=2)

        else:
            return _error_result(
                "InvalidArguments",
                f"Unknown action: {action!r}. Use create, archive, add_member, remove_member, or info.",
            )

    # ── Human notification handler ────────────────────────────

    def _urgent_phone_context(self: _CommsToolsHost, args: dict[str, Any]) -> dict[str, Any] | None:
        """Still ring the phone for urgent calls when the other channels fail."""
        subject = args.get("subject", "")
        body = args.get("body", "")
        if args.get("priority") != "urgent" or not subject or not body:
            return None
        from core.phone.urgent import request_phone_alert

        return {"phone": request_phone_alert(subject, body, self._anima_name)}

    def _handle_call_human(self: _CommsToolsHost, args: dict[str, Any]) -> str:
        self._last_call_human_denied = False
        if not self._human_notifier:
            return _error_result(
                "NotConfigured",
                "Human notification is not configured",
                context=self._urgent_phone_context(args),
                suggestion="Enable human_notification in config.json",
            )
        if self._human_notifier.channel_count == 0:
            return _error_result(
                "NotConfigured",
                "No notification channels configured",
                context=self._urgent_phone_context(args),
                suggestion="Add channels to human_notification.channels in config.json",
            )

        import asyncio

        subject = args.get("subject", "")
        body = args.get("body", "")
        priority = args.get("priority", "normal")
        interactive = bool(args.get("interactive", False))
        category = str(args.get("category") or "approval")
        raw_opts = args.get("options", "approve,reject,comment")

        raw_allowed = args.get("allowed_users")
        allowed_list: list[str]
        if raw_allowed is None:
            allowed_list = []
        elif isinstance(raw_allowed, list):
            allowed_list = [str(x).strip() for x in raw_allowed if str(x).strip()]
        else:
            s = str(raw_allowed).strip()
            allowed_list = [s] if s else []

        if not subject or not body:
            return _error_result(
                "InvalidArguments",
                "subject and body are required",
            )

        # The confirm step asks the Anima to check Slack/Chatwork first; with only
        # the built-in Web UI channel there is nothing outside to check. CLI
        # subprocesses share the server's per-session key store so confirmation
        # remains usable across separate invocations.
        if getattr(self._human_notifier, "has_external_channels", True) is False:
            issued_key = None
        else:
            from core.platform.env import get_env

            cli_session_id = get_env("ANIMAWORKS_TOOL_SESSION_ID", "").strip()
            if cli_session_id:
                from core.internal_api import host_api

                try:
                    response = host_api.post(
                        "/api/internal/call-human/confirm",
                        json={
                            "anima_name": self._anima_name,
                            "session_id": cli_session_id,
                            "sha": args.get("sha", ""),
                        },
                        timeout=10.0,
                    )
                    response.raise_for_status()
                    confirmation = response.json()
                    if (
                        not isinstance(confirmation, dict)
                        or not isinstance(confirmation.get("ok"), bool)
                        or not isinstance(confirmation.get("sha"), str)
                        or (not confirmation["ok"] and not confirmation["sha"])
                    ):
                        raise ValueError("Invalid call_human confirmation response")
                    issued_key = confirmation["sha"] or None
                except Exception:
                    logger.warning("Could not verify call_human confirmation with the server", exc_info=True)
                    self._last_call_human_denied = True
                    return _error_result(
                        "ConfirmationUnavailable",
                        t("handler.call_human_confirm_unavailable"),
                    )
            else:
                issued_key = self._call_human_keys.check(self._anima_name, self.session_id, args.get("sha", ""))
        self._last_call_human_denied = issued_key is not None
        if issued_key is not None:
            return _error_result(
                "ConfirmationRequired",
                t("handler.call_human_confirm_required", sha=issued_key),
            )

        interaction_req = None
        self._last_call_human_callback_id = None
        if interactive:
            from core.config.models import load_config
            from core.notification.interactive import create_interaction_resilient

            cfg = load_config()
            defaults = list(cfg.interaction.default_approver_ids)
            merged = list(dict.fromkeys(allowed_list + defaults))
            if isinstance(raw_opts, list):
                opts_list = [str(x).strip() for x in raw_opts if str(x).strip()]
            else:
                opts_list = [p.strip() for p in str(raw_opts).split(",") if p.strip()]
            if not opts_list:
                opts_list = ["approve", "reject", "comment"]
            aud: dict[str, list[str]] = {"slack": merged} if merged else {}

            # Server-API-first: this handler runs inside sandboxed MCP servers
            # where {data_dir}/run/ is read-only. A local-only write would be
            # lost and every button would report "expired" on first click.
            try:
                interaction_req = create_interaction_resilient(
                    self._anima_name,
                    category,
                    opts_list,
                    allowed_users=aud or None,
                    callback_id=str(args.get("callback_id") or ""),
                )
            except ValueError as ve:
                return _error_result("InvalidArguments", str(ve))
            except OSError as oe:
                return _error_result(
                    "InteractionPersistError",
                    f"Failed to persist interactive request: {oe}",
                    suggestion="Check that the AnimaWorks server is reachable from this process",
                )
            if interaction_req is not None:
                # Consumed by _log_tool_activity (runs after this handler returns).
                self._last_call_human_callback_id = interaction_req.callback_id

        try:
            coro = self._human_notifier.notify(
                subject,
                body,
                priority,
                anima_name=self._anima_name,
                interaction=interaction_req,
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
            return _error_result(
                "NotificationError",
                f"Failed to send notification: {e}",
                context=self._urgent_phone_context(args),
            )

        if priority == "urgent":
            from core.phone.urgent import request_phone_alert

            results = [*results, request_phone_alert(subject, body, self._anima_name)]

        # The Web UI channel already pushed it to the browser; queuing it for the
        # chat stream as well would show the same notification twice.
        if getattr(self._human_notifier, "delivers_to_web", False) is not True:
            notif_data = {
                "anima": self._anima_name,
                "subject": subject,
                "body": body,
                "priority": priority,
                "timestamp": now_iso(),
            }
            self._pending_notifications.append(notif_data)

        payload: dict[str, Any] = {"status": "sent", "results": results}
        if interactive and interaction_req is not None:
            payload["interactive"] = True
            payload["callback_id"] = interaction_req.callback_id

        return _json.dumps(payload, ensure_ascii=False)
