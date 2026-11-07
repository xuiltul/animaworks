from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Centralized write path for shared Board channels."""

import json
import logging
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.config.anima_registry import read_anima_company_checked
from core.exceptions import RecipientNotFoundError
from core.messaging.messenger import _validate_name, load_channel_meta_state
from core.platform.atomic_io import append_jsonl_locked
from core.platform.locks import locked_path
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.messaging.board")

PrincipalKind = Literal["anima", "human", "operator", "external"]
Action = Literal["read", "write", "mention_target", "outbound"]
ExternalKind = Literal["user", "bot"]
HookStatus = Literal["sent", "skipped", "failed"]


@dataclass(frozen=True)
class Principal:
    kind: PrincipalKind
    name: str
    company: str | None = None
    company_known: bool = True
    external_kind: ExternalKind | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


@dataclass
class PostResult:
    committed: bool = False
    duplicate: bool = False
    id: str | None = None
    error: str | None = None
    notified: list[str] = field(default_factory=list)
    not_notified: dict[str, str] = field(default_factory=dict)
    hooks: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "duplicate": self.duplicate,
            "id": self.id,
            "error": self.error,
            "notified": list(self.notified),
            "not_notified": dict(self.not_notified),
            "hooks": dict(self.hooks),
        }


@dataclass(frozen=True)
class BoardPost:
    channel: str
    ts: str
    from_person: str
    text: str
    source: str
    id: str
    kind: str
    origin: str
    principal: Principal
    reply_to: str | None = None


PostHook = Callable[[BoardPost, PostResult], HookStatus]
_post_hooks: dict[str, PostHook] = {}
_post_hooks_lock = threading.RLock()


def reply_metadata_for_post(
    channel: str,
    triggers: Sequence[Mapping[str, Any]] | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Derive the ``reply_to`` and ``trigger`` fields for a Board post.

    ``reply_to`` points at the last ``board_mention`` that came from the same
    channel; ``trigger`` points at the last inbox message of the run.
    """
    items = [
        trigger
        for trigger in (triggers or ())
        if isinstance(trigger, Mapping) and isinstance(trigger.get("id"), str) and trigger.get("id")
    ]
    if not items:
        return None, None

    reply_to: str | None = None
    for trigger in reversed(items):
        if trigger.get("type") == "board_mention" and trigger.get("board_channel") == channel:
            board_post_id = trigger.get("board_post_id")
            if isinstance(board_post_id, str) and board_post_id:
                reply_to = board_post_id
                break

    last = items[-1]
    trigger_meta: dict[str, str] | None = None
    if last.get("type") == "board_mention":
        board_post_id = last.get("board_post_id")
        if isinstance(board_post_id, str) and board_post_id:
            trigger_meta = {"type": "board_mention", "id": board_post_id}
    else:
        message_id = last.get("id")
        if isinstance(message_id, str) and message_id:
            trigger_meta = {"type": "message", "id": message_id}
    return reply_to, trigger_meta


def anima_principal(name: str, *, animas_dir: Path) -> Principal:
    """Build an Anima identity while preserving unreadable company state."""
    known, company = read_anima_company_checked(animas_dir / name)
    return Principal(kind="anima", name=name, company=company, company_known=known)


def human_principal(name: str = "human") -> Principal:
    return Principal(kind="human", name=name)


def operator_principal(name: str = "operator") -> Principal:
    return Principal(kind="operator", name=name)


def external_principal(name: str, *, external_kind: ExternalKind = "user") -> Principal:
    return Principal(kind="external", name=name or "external", external_kind=external_kind)


def _decision(allowed: bool, reason: str = "allowed") -> Decision:
    return Decision(allowed=allowed, reason=reason)


def board_access(shared_dir: Path, channel: str, principal: Principal, action: Action) -> Decision:
    """Apply the centralized Board ACL and company-scope rules."""
    state, meta = load_channel_meta_state(shared_dir, channel)
    if state == "unreadable":
        return _decision(False, "meta_unreadable")

    if meta is not None and meta.closed:
        if action == "read" and principal.kind in {"human", "operator"}:
            return _decision(True)
        return _decision(False, "closed")

    if action not in {"read", "write", "mention_target", "outbound"}:
        return _decision(False, "action_not_allowed")

    if principal.kind in {"human", "operator"}:
        return _decision(True)

    if principal.kind == "external":
        if action != "write":
            return _decision(False, "action_not_allowed")
        try:
            from core.config.models import load_config

            config = load_config()
            external = config.external_messaging
            mapped_boards: set[str] = set()
            for platform_name in ("slack", "discord"):
                mapping = getattr(getattr(external, platform_name), "board_mapping", {})
                mapped_boards.update(value for value in mapping.values() if isinstance(value, str))
        except Exception:
            logger.warning("Could not read external Board mappings", exc_info=True)
            return _decision(False, "not_mapped")
        return _decision(channel in mapped_boards, "allowed" if channel in mapped_boards else "not_mapped")

    if principal.kind != "anima":
        return _decision(False, "action_not_allowed")
    members = list(meta.members or []) if meta is not None else []
    if members:
        if principal.name not in members:
            return _decision(False, "not_member")
        animas_dir = shared_dir.parent / "animas"
        try:
            from core.org.company import check_company_boundary

            for member in sorted(set(members)):
                boundary = check_company_boundary(principal.name, member, animas_dir=animas_dir)
                if boundary.cross_company:
                    reason = "company_unreadable" if boundary.resolved_via == "fail_closed" else "cross_company"
                    return _decision(False, reason)
        except Exception:
            logger.warning("Could not verify member company boundary for #%s", channel, exc_info=True)
            return _decision(False, "company_unreadable")
        return _decision(True)

    company = (meta.company if meta is not None else "") or ""
    if company:
        if not principal.company_known:
            return _decision(False, "company_unreadable")
        if principal.company is None:
            return _decision(True)
        if principal.company == company:
            return _decision(True)
        return _decision(False, "company_mismatch")

    if not principal.company_known:
        return _decision(False, "company_unreadable")
    if principal.company is not None:
        return _decision(False, "company_assigned_legacy")
    return _decision(True)


def register_post_hook(name: str, fn: PostHook) -> None:
    """Register or replace a synchronous post-commit hook."""
    with _post_hooks_lock:
        _post_hooks[name] = fn


def unregister_post_hook(name: str) -> None:
    with _post_hooks_lock:
        _post_hooks.pop(name, None)


def clear_post_hooks() -> None:
    """Remove registered hooks; intended for isolated tests."""
    with _post_hooks_lock:
        _post_hooks.clear()


def _tail_jsonl_records(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read at most the last *limit* JSON objects without loading the full log."""
    if not path.exists() or limit <= 0:
        return []
    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        while position > 0 and newline_count <= limit:
            read_size = min(8192, position)
            position -= read_size
            stream.seek(position)
            chunk = stream.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    raw = b"".join(reversed(chunks))
    lines = raw.splitlines()[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _source_for(principal: Principal, origin: str) -> str:
    if principal.kind == "anima":
        return "anima"
    if principal.kind in {"human", "operator"}:
        return "human"
    if origin in {"slack", "discord"}:
        return origin
    return "human"


def _kind_for(principal: Principal) -> str:
    if principal.kind == "external":
        return f"external_{principal.external_kind or 'user'}"
    return principal.kind


def _run_post_hooks(post: BoardPost, result: PostResult) -> None:
    with _post_hooks_lock:
        hooks = list(_post_hooks.items())
    for name, hook in hooks:
        try:
            status = hook(post, result)
            result.hooks[name] = status if status in {"sent", "skipped", "failed"} else "failed"
        except Exception:
            logger.exception("Board post hook %s failed for #%s", name, post.channel)
            result.hooks[name] = "failed"


def post_to_board(
    channel: str,
    text: str,
    principal: Principal,
    *,
    request_id: str | None = None,
    origin: str,
    notify: bool = True,
    shared_dir: Path | None = None,
    reply_to: str | None = None,
    trigger: dict[str, str] | None = None,
) -> PostResult:
    """Authorize and append one Board post, then run best-effort side effects."""
    from core.paths import get_shared_dir

    return post_to_board_in_shared_dir(
        shared_dir or get_shared_dir(),
        channel,
        text,
        principal,
        request_id=request_id,
        origin=origin,
        notify=notify,
        reply_to=reply_to,
        trigger=trigger,
    )


def post_to_board_in_shared_dir(
    shared_dir: Path,
    channel: str,
    text: str,
    principal: Principal,
    *,
    request_id: str | None = None,
    origin: str,
    notify: bool = True,
    reply_to: str | None = None,
    trigger: dict[str, str] | None = None,
) -> PostResult:
    """Path-explicit implementation used by the runtime entry points and tests."""
    try:
        _validate_name(channel, "channel name")
    except (RecipientNotFoundError, TypeError):
        return PostResult(error="invalid_name")

    channels_dir = shared_dir / "channels"
    filepath = channels_dir / f"{channel}.jsonl"
    meta_path = channels_dir / f"{channel}.meta.json"
    if not filepath.exists() and not meta_path.exists():
        return PostResult(error="not_found")

    ts = now_iso()
    post_id = uuid.uuid4().hex[:16]
    source = _source_for(principal, origin)
    kind = _kind_for(principal)
    record: dict[str, Any] = {
        "ts": ts,
        "from": principal.name,
        "text": text,
        "source": source,
        "id": post_id,
        "kind": kind,
    }
    if request_id is not None:
        record["request_id"] = request_id
    if reply_to is not None:
        record["reply_to"] = reply_to
    if trigger is not None:
        record["trigger"] = trigger

    channels_dir.mkdir(parents=True, exist_ok=True)
    lock_path = filepath.with_name(f"{filepath.name}.lock")
    try:
        with locked_path(lock_path, exclusive=True, thread_lock=True):
            # Re-check under the append lock: archive closes the meta and moves
            # the JSONL while holding this same lock.
            if not filepath.exists() and not meta_path.exists():
                return PostResult(error="not_found")
            decision = board_access(shared_dir, channel, principal, "write")
            if not decision.allowed:
                return PostResult(error=f"denied:{decision.reason}")
            if request_id is not None:
                for previous in reversed(_tail_jsonl_records(filepath, 200)):
                    if previous.get("request_id") == request_id:
                        return PostResult(duplicate=True, id=previous.get("id"))
            append_jsonl_locked(filepath, record, lock_held=True)
    except OSError:
        logger.exception("Failed to append Board post to #%s", channel)
        return PostResult(error="write_failed")

    result = PostResult(committed=True, id=post_id)
    post = BoardPost(
        channel=channel,
        ts=ts,
        from_person=principal.name,
        text=text,
        source=source,
        id=post_id,
        kind=kind,
        origin=origin,
        principal=principal,
        reply_to=reply_to,
    )

    if notify:
        try:
            from core.messaging.board_fanout import fanout_board_mentions

            result.notified, result.not_notified = fanout_board_mentions(shared_dir, post)
        except Exception:
            logger.exception("Board mention fan-out failed for #%s", channel)

    if result.notified or result.not_notified:
        try:
            from core.messaging.board_deliveries import append_delivery

            append_delivery(
                shared_dir,
                channel,
                post_id=post_id,
                ts=ts,
                notified=result.notified,
                not_notified=result.not_notified,
            )
        except Exception:
            logger.warning("Failed to record Board delivery for #%s", channel, exc_info=True)

    if principal.kind == "anima":
        try:
            from core.activity.logger import ActivityLogger

            anima_dir = shared_dir.parent / "animas" / principal.name
            if anima_dir.is_dir():
                ActivityLogger(anima_dir).log(
                    "channel_post",
                    content=text[:200],
                    channel=channel,
                    summary=f"#{channel}: {text[:80]}",
                )
        except Exception:
            logger.warning("Failed to log channel_post for %s", principal.name, exc_info=True)

    _run_post_hooks(post, result)
    return result
