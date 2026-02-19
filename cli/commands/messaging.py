# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("animaworks")


# ── Send ───────────────────────────────────────────────────


def cmd_send(args: argparse.Namespace) -> None:
    """Send a message from one anima to another (filesystem based)."""
    from core.init import ensure_runtime_dir
    from core.messenger import Messenger
    from core.paths import get_shared_dir

    ensure_runtime_dir()
    messenger = Messenger(get_shared_dir(), args.from_person)
    msg = messenger.send(
        to=args.to_person,
        content=args.message,
        thread_id=args.thread_id or "",
        reply_to=args.reply_to or "",
    )
    print(f"Sent: {msg.from_person} -> {msg.to_person} (id: {msg.id}, thread: {msg.thread_id})")
    _notify_server_message_sent(args.from_person, args.to_person, args.message, msg.id)


def _notify_server_message_sent(
    from_anima: str, to_anima: str, content: str, message_id: str = "",
) -> None:
    """Notify the running server about a CLI-sent message.

    Triggers WebSocket broadcast and reply tracking.
    Fails silently if the server is not running.
    """
    from cli.commands.server import _is_process_alive, _read_pid

    pid = _read_pid()
    if pid is None or not _is_process_alive(pid):
        return

    server_url = os.environ.get("ANIMAWORKS_SERVER_URL", "http://localhost:18500")
    try:
        import httpx

        resp = httpx.post(
            f"{server_url}/api/internal/message-sent",
            json={
                "from_person": from_anima,
                "to_person": to_anima,
                "content": content[:200],
                "message_id": message_id,
            },
            timeout=5.0,
        )
        if resp.status_code == 200:
            logger.debug("Server notified of CLI send: %s -> %s", from_anima, to_anima)
        else:
            logger.debug("Server notification failed: %s", resp.status_code)
    except Exception:
        logger.debug("Could not notify server of CLI message send", exc_info=True)


# ── List ───────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    """List all animas (from gateway or filesystem)."""
    if args.local:
        _list_local()
    else:
        from cli._gateway import gateway_request_or_none

        data = gateway_request_or_none(
            args, "GET", "/api/animas", timeout=10.0
        )
        if data is None:
            print("Gateway not reachable, falling back to filesystem...")
            _list_local()
        elif isinstance(data, list):
            for p in data:
                name = p.get("name", "unknown")
                status = p.get("status", "unknown")
                print(f"  {name} ({status})")
        else:
            print(data)


def _list_local() -> None:
    from core.init import ensure_runtime_dir
    from core.paths import get_animas_dir

    ensure_runtime_dir()
    animas_dir = get_animas_dir()
    if not animas_dir.exists():
        print("No animas directory found.")
        return
    for d in sorted(animas_dir.iterdir()):
        if d.is_dir() and (d / "identity.md").exists():
            print(f"  {d.name}")


# ── Status ─────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    from cli._gateway import gateway_request

    data = gateway_request(args, "GET", "/api/system/status", timeout=10.0)
    if isinstance(data, dict):
        print(f"Animas: {data.get('animas', 0)}")
        print(f"Scheduler: {'running' if data.get('scheduler_running') else 'stopped'}")
        for j in data.get("jobs", []):
            print(f"  [{j['id']}] {j['name']} -> next: {j['next_run']}")
    else:
        print(data)
