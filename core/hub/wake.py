from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Wake external hub members and enforce the hub conversation rules.

Webhook wakes are best effort and never block the Board write: the request is
sent on a background thread and the outcome is appended to the delivery log
afterwards.  The loop-limit rule and the "member went silent" alert share this
module's small state helpers.
"""

import asyncio
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from core.config.vault import VaultManager
from core.hub.members import HubMember, state_dir
from core.i18n import t
from core.messaging.board import BoardPost
from core.messaging.board_deliveries import append_delivery
from core.platform.atomic_io import atomic_write_json
from core.platform.locks import locked_path
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.hub.wake")

WEBHOOK_TIMEOUT_S = 10.0
MAX_PAYLOAD_TEXT = 500
MAX_REPLY_DEPTH = 20
BOT_KINDS = frozenset({"anima", "external_agent"})
_WAKE_THREADS: set[threading.Thread] = set()
_WAKE_THREADS_LOCK = threading.Lock()

WebhookSender = Callable[[str, str, dict[str, Any]], tuple[bool, str]]


def build_task_text(member_id: str, post_id: str) -> str:
    """Return the fixed instruction sent to a woken hub member."""
    return t("hub.wake_task", member_id=member_id, post_id=post_id)


def build_payload(board: str, post: BoardPost, member_id: str) -> dict[str, Any]:
    """Build the webhook JSON body for one wake request."""
    return {
        "task": build_task_text(member_id, post.id),
        "board": board,
        "post_id": post.id,
        "from": post.from_person,
        "text": post.text[:MAX_PAYLOAD_TEXT],
        "request_id": post.id,
    }


def send_webhook(url: str, key: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """POST *payload* to *url* with a bearer token, following no redirects.

    Returns ``(True, "")`` for a 2xx response and ``(False, reason)`` otherwise.
    The URL and token are never logged; only the host name may appear.
    """
    try:
        with httpx.Client(timeout=WEBHOOK_TIMEOUT_S, follow_redirects=False) as client:
            response = client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
    except (httpx.TimeoutException, TimeoutError):
        logger.info("Hub wake timed out for host %s", _host_of(url))
        return False, "timeout"
    except Exception as exc:
        logger.warning("Hub wake failed for host %s: %s", _host_of(url), exc.__class__.__name__)
        return False, "error"
    if 200 <= response.status_code < 300:
        return True, ""
    return False, f"http_{response.status_code}"


def _host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or "unknown"
    except ValueError:
        return "unknown"


def _vault_secret(shared_dir: Path, ref: object) -> str | None:
    if not isinstance(ref, str) or not ref:
        return None
    try:
        value = VaultManager(Path(shared_dir).parent).get("shared", ref)
    except Exception:
        logger.warning("Could not read hub webhook secret from the vault", exc_info=True)
        return None
    if not isinstance(value, str) or not value:
        return None
    return value


def _is_https(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.hostname)


def _send_webhook(
    shared_dir: Path,
    board: str,
    post: BoardPost,
    member: HubMember,
    *,
    sender: WebhookSender | None = None,
) -> tuple[list[str], dict[str, str]]:
    route = member.message_route or {}
    url_ref = route.get("url_ref")
    key_ref = route.get("key_ref")
    if not url_ref or not key_ref:
        return [], {member.id: "bad_route"}

    url = _vault_secret(shared_dir, url_ref)
    key = _vault_secret(shared_dir, key_ref)
    if url is None or key is None:
        return [], {member.id: "no_secret"}
    if not _is_https(url):
        return [], {member.id: "bad_route"}

    payload = build_payload(board, post, member.id)
    try:
        if sender is not None:
            ok, reason = sender(url, key, payload)
        else:
            ok, reason = send_webhook(url, key, payload)
    except (httpx.TimeoutException, TimeoutError):
        return [], {member.id: "timeout"}
    except Exception:
        logger.warning("Hub wake sender failed for %s", member.id, exc_info=True)
        return [], {member.id: "error"}

    if ok:
        logger.info("Hub wake delivered to %s via host %s", member.id, _host_of(url))
        return [member.id], {}
    return [], {member.id: str(reason or "error")}


def wake_member(
    shared_dir: Path,
    board: str,
    post: BoardPost,
    member: HubMember,
    *,
    sender: WebhookSender | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Wake one member synchronously and return the delivery outcome.

    Poll members are not contacted here; they see the post the next time they
    poll the hub.
    """
    route_type = (member.message_route or {}).get("type")
    if route_type == "poll":
        return [], {member.id: "polls"}
    if route_type != "webhook":
        return [], {member.id: "bad_route"}
    return _send_webhook(shared_dir, board, post, member, sender=sender)


def deliver_wake(
    shared_dir: Path,
    channel: str,
    post: BoardPost,
    member: HubMember,
    *,
    sender: WebhookSender | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Wake one member and append its own delivery record."""
    notified, not_notified = wake_member(shared_dir, channel, post, member, sender=sender)
    append_delivery(
        shared_dir,
        channel,
        post_id=post.id,
        ts=now_iso(),
        notified=notified,
        not_notified=not_notified,
    )
    return notified, not_notified


def schedule_wake(
    shared_dir: Path,
    channel: str,
    post: BoardPost,
    member: HubMember,
    *,
    sender: WebhookSender | None = None,
) -> threading.Thread:
    """Send a webhook wake on a background thread.

    The thread appends the result to the delivery log when it finishes, so the
    Board write path never waits for an external HTTP request.
    """

    def run() -> None:
        try:
            deliver_wake(shared_dir, channel, post, member, sender=sender)
        except Exception:
            logger.warning("Hub wake recording failed for %s", member.id, exc_info=True)
        finally:
            with _WAKE_THREADS_LOCK:
                _WAKE_THREADS.discard(thread)

    thread = threading.Thread(target=run, name=f"hub-wake-{member.id}", daemon=True)
    with _WAKE_THREADS_LOCK:
        _WAKE_THREADS.add(thread)
    thread.start()
    return thread


def wait_for_pending_wakes(timeout: float | None = None) -> None:
    """Join wake threads; useful for tests and orderly shutdown."""
    deadline = None if timeout is None else time.monotonic() + timeout
    with _WAKE_THREADS_LOCK:
        threads = list(_WAKE_THREADS)
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(remaining)


def _channel_records(shared_dir: Path, channel: str) -> dict[str, dict[str, Any]]:
    path = Path(shared_dir) / "channels" / f"{channel}.jsonl"
    records: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                post_id = record.get("id")
                if isinstance(post_id, str) and post_id:
                    records[post_id] = record
    except (OSError, UnicodeDecodeError):
        return {}
    return records


def count_round_trips(
    shared_dir: Path,
    channel: str,
    post: BoardPost,
    *,
    max_depth: int = MAX_REPLY_DEPTH,
) -> tuple[int, str]:
    """Count anima/external alternations above *post* and return the thread top.

    Only ancestors in *channel* are followed, and at most *max_depth* hops are
    inspected so a corrupt reply chain cannot loop forever.
    """
    records = _channel_records(shared_dir, channel)
    top_id = post.id
    current_kind: object = post.kind
    reply_to = post.reply_to
    transitions = 0
    seen: set[str] = {post.id}
    depth = 0
    while isinstance(reply_to, str) and reply_to and depth < max_depth:
        record = records.get(reply_to)
        if record is None:
            break
        top_id = reply_to
        seen.add(reply_to)
        parent_kind = record.get("kind")
        if isinstance(parent_kind, str) and parent_kind in BOT_KINDS:
            if isinstance(current_kind, str) and current_kind in BOT_KINDS and parent_kind != current_kind:
                transitions += 1
            current_kind = parent_kind
        else:
            current_kind = parent_kind
        next_reply = record.get("reply_to")
        reply_to = next_reply if isinstance(next_reply, str) and next_reply not in seen else None
        depth += 1
    return transitions, top_id


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _loop_alerts_path(shared_dir: Path) -> Path:
    return state_dir(shared_dir) / "loop_alerts.json"


def alert_loop_limit_once(
    shared_dir: Path,
    channel: str,
    post: BoardPost,
    top_id: str,
    *,
    notifier: Any | None = None,
) -> bool:
    """Notify a human once per reply thread that hit the round-trip limit."""
    path = _loop_alerts_path(shared_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with locked_path(lock_path, exclusive=True, thread_lock=True):
        alerts = _read_json_object(path)
        if top_id in alerts:
            return False
        alerts[top_id] = now_iso()
        atomic_write_json(path, alerts, indent=2)

    notify_human(
        t("hub.loop_limit_subject"),
        t("hub.loop_limit_body", board=channel, post_id=post.id),
        "high",
        notifier=notifier,
    )
    return True


def notify_human(subject: str, body: str, priority: str = "normal", *, notifier: Any | None = None) -> None:
    """Run the async human notifier from synchronous hub code."""
    try:
        if notifier is None:
            from core.config.models import load_config
            from core.notification.notifier import HumanNotifier

            notifier = HumanNotifier.from_config(load_config().human_notification)
        result = notifier.notify(subject, body, priority)
        if inspect.isawaitable(result):
            _run_coroutine_sync(result)
    except Exception:
        logger.warning("Failed to notify a human about a hub event", exc_info=True)


def _run_coroutine_sync(coro: Any) -> Any:
    """Run *coro* to completion from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    errors: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    thread = threading.Thread(target=runner, name="hub-notifier", daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return None
