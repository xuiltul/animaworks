# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Standalone CLI entry point for the Chatwork tool."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from core.tools._base import ToolConfigError, logger
from core.tools._chatwork_cache import (
    DEFAULT_CACHE_DIR,
    MessageCache,
    _format_timestamp,
    resolve_cache_db_path,
)
from core.tools._chatwork_client import ChatworkClient
from core.tools._chatwork_identity import check_write_allowed, resolve_identity
from core.tools._chatwork_markdown import md_to_chatwork
from core.tools._comm_cli import run_cli_safely

# ── Config ──────────────────────────────────────────────────


def _load_chatwork_tool_config() -> dict:
    """Load tool-local config from DEFAULT_CACHE_DIR / config.json."""
    config_path = DEFAULT_CACHE_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


# ── Sync ──────────────────────────────────────────────────────


def _sync_rooms(
    client: ChatworkClient,
    cache: MessageCache,
    sync_limit: int = 30,
    sleep_interval: float = 0.3,
) -> dict[str, int]:
    """Fetch all room metadata + sync messages for top N rooms.

    Returns dict with ``rooms`` (total room count) and ``messages``
    (total messages fetched).
    """
    rooms_data = client.rooms()
    for room in rooms_data:
        cache.upsert_room(room)

    rooms_data.sort(key=lambda r: r.get("last_update_time", 0), reverse=True)
    total_msgs = 0
    for room in rooms_data[:sync_limit]:
        rid = str(room["room_id"])
        try:
            msgs = client.get_messages(rid, force=True)
            if msgs:
                cache.upsert_messages(rid, msgs)
                cache.update_sync_state(rid)
                total_msgs += len(msgs)
        except Exception:
            logger.warning("sync failed for room %s", rid)
        time.sleep(sleep_interval)
    return {"rooms": len(rooms_data), "messages": total_msgs}


# ── CLI Guide ────────────────────────────────────────────────


def get_cli_guide() -> str:
    """Return CLI usage guide for Chatwork tools."""
    return """\
### Chatwork
```bash
animaworks-tool chatwork sync [--limit N]
animaworks-tool chatwork rooms
animaworks-tool chatwork messages <ルーム名またはID> [-n 20]
animaworks-tool chatwork send <ルーム名またはID> "メッセージ本文"
animaworks-tool chatwork delete <ルーム名またはID> <message_id>  # 自分の投稿のみ削除可
animaworks-tool chatwork search "キーワード" [-r ルーム] [-n 50]
animaworks-tool chatwork unreplied [--sync] [--sync-limit 50] [--json]
animaworks-tool chatwork mentions [--sync] [--sync-limit 50] [-n 200] [--json]
animaworks-tool chatwork me
animaworks-tool chatwork members <ルーム名またはID>
animaworks-tool chatwork contacts
animaworks-tool chatwork task <ルーム名またはID> "タスク本文" "担当者ID(カンマ区切り)"
animaworks-tool chatwork mytasks [--done]
animaworks-tool chatwork tasks <ルーム名またはID> [--done]
animaworks-tool chatwork files <ルーム名またはID> [--account-id ID]
animaworks-tool chatwork download <ルーム名またはID> <file_id> [-o 保存先パス]
animaworks-tool chatwork upload <ルーム名またはID> <ファイルパス> [-m "添付メッセージ"]  # 5MBまで
animaworks-tool chatwork stats
```
全サブコマンドで `--as <identity>` を指定すると、委任されたidentityとして実行できます。"""


# ── CLI Main ─────────────────────────────────────────────────


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI entry point for the Chatwork tool."""

    def _run() -> None:
        parser = argparse.ArgumentParser(
            prog="animaworks-chatwork",
            description="Chatwork CLI (AnimaWorks integration)",
        )
        sub = parser.add_subparsers(dest="command", help="Command")

        # send
        p = sub.add_parser("send", help="Send a message")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("message", nargs="+", help="Message body")

        # delete
        p = sub.add_parser("delete", help="Delete your own message")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("message_id", help="Message ID to delete")

        # messages
        p = sub.add_parser("messages", help="Get recent messages")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("-n", "--num", type=int, default=20, help="Number of messages (default 20)")

        # search
        p = sub.add_parser("search", help="Search cached messages")
        p.add_argument("keyword", nargs="+", help="Search keyword")
        p.add_argument("-r", "--room", help="Filter by room name or ID")
        p.add_argument("-n", "--num", type=int, default=50, help="Max results (default 50)")

        # unreplied
        p = sub.add_parser("unreplied", help="Show unreplied messages addressed to me")
        p.add_argument("--sync", action="store_true", help="Sync before checking")
        p.add_argument("--sync-limit", type=int, default=50, help="Rooms to sync (default 50)")
        p.add_argument("--include-toall", action="store_true", help="Include @all mentions")
        p.add_argument("--json", action="store_true", help="Output as JSON")

        # rooms
        sub.add_parser("rooms", help="List accessible rooms")

        # sync
        p = sub.add_parser("sync", help="Sync room metadata and messages to local cache")
        p.add_argument("room", nargs="?", help="Specific room name or ID to sync")
        p.add_argument(
            "-l",
            "--limit",
            type=int,
            default=30,
            help="Number of rooms to sync messages for (default 30)",
        )

        # me
        sub.add_parser("me", help="Show own account info")

        # members
        p = sub.add_parser("members", help="List room members")
        p.add_argument("room", help="Room name or ID")

        # contacts
        sub.add_parser("contacts", help="List contacts")

        # task
        p = sub.add_parser("task", help="Create a task in a room")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("body", help="Task body text")
        p.add_argument("to_ids", help="Comma-separated account IDs for assignees")

        # mytasks
        p = sub.add_parser("mytasks", help="List my tasks")
        p.add_argument("--done", action="store_true", help="Show completed tasks")

        # tasks
        p = sub.add_parser("tasks", help="List tasks in a room")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("--done", action="store_true", help="Show completed tasks")

        # mentions
        p = sub.add_parser("mentions", help="Show messages addressed to me")
        p.add_argument("--sync", action="store_true", help="Sync before checking")
        p.add_argument("--sync-limit", type=int, default=50, help="Rooms to sync (default 50)")
        p.add_argument("--include-toall", action="store_true", help="Include @all mentions")
        p.add_argument("-n", "--num", type=int, default=200, help="Max mentions (default 200)")
        p.add_argument("--json", action="store_true", help="Output as JSON")

        # files
        p = sub.add_parser("files", help="List files in a room")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("--account-id", help="Filter by uploader account ID")

        # download
        p = sub.add_parser("download", help="Download a file from a room")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("file_id", help="File ID")
        p.add_argument(
            "-o",
            "--output",
            help="Output file path (default: original filename in current directory)",
        )

        # upload
        p = sub.add_parser("upload", help="Upload a file to a room")
        p.add_argument("room", help="Room name or ID")
        p.add_argument("file", help="File path to upload")
        p.add_argument("-m", "--message", help="Attachment message (optional)")

        # stats
        sub.add_parser("stats", help="Show cache statistics")

        for command_parser in sub.choices.values():
            command_parser.add_argument(
                "--as",
                dest="as_identity",
                metavar="IDENTITY",
                help="Use a delegated Chatwork identity",
            )

        args = parser.parse_args(argv)

        if not args.command:
            parser.print_help()
            sys.exit(0)

        try:
            identity = resolve_identity(args.as_identity)
            if args.command in {"send", "upload", "delete", "task"}:
                check_write_allowed(args.as_identity)
            client = ChatworkClient(api_token=identity.token)
        except ToolConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.command == "send":
            room_id = client.resolve_room_id(args.room)
            message = md_to_chatwork(" ".join(args.message))
            result = client.post_message(room_id, message)
            if result and "message_id" in result:
                print(f"Sent (message_id: {result['message_id']})")
            else:
                print(f"Result: {result}")

        elif args.command == "delete":
            room_id = client.resolve_room_id(args.room)
            message_id = args.message_id
            # Ownership check
            my_info = client.me()
            my_account_id = str(my_info["account_id"])
            msg = client.get_message(room_id, message_id)
            msg_account_id = str(msg["account"]["account_id"])
            if msg_account_id != my_account_id:
                print(
                    f"Error: Cannot delete message {message_id}. "
                    f"It was posted by '{msg['account']['name']}' "
                    f"(account_id={msg_account_id}), not by you "
                    f"(account_id={my_account_id}).",
                    file=sys.stderr,
                )
                sys.exit(1)
            client.delete_message(room_id, message_id)
            print(f"Deleted message {message_id} from room {room_id}")

        elif args.command == "messages":
            room_id = client.resolve_room_id(args.room)
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                msgs = client.get_messages(room_id, force=True)
                if msgs:
                    cache.upsert_messages(room_id, msgs)
                    cache.update_sync_state(room_id)
                    print(f"({len(msgs)} fetched & cached)\n", file=sys.stderr)
                cached = cache.get_recent(room_id, limit=args.num)
                if cached:
                    for m in reversed(cached):
                        ts = m.get("send_time_jst", "")
                        name = m.get("account_name", "?")
                        room_name = m.get("room_name", "")
                        room_tag = f"[{room_name}] " if room_name else ""
                        body = m.get("body", "").strip()
                        print(f"{ts} {room_tag}{name}")
                        for line in body.split("\n"):
                            print(f"  {line}")
                        print()
                else:
                    print("No messages found. Run 'sync' first.")
            finally:
                cache.close()

        elif args.command == "search":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                keyword = " ".join(args.keyword)
                room_id = None
                if args.room:
                    room_id = client.resolve_room_id(args.room)
                results = cache.search(keyword, room_id=room_id, limit=args.num)
                if not results:
                    print(f"No messages matching '{keyword}'.")
                else:
                    print(f"Results: {len(results)} (keyword: '{keyword}')\n")
                    for m in reversed(results):
                        ts = m.get("send_time_jst", "")
                        name = m.get("account_name", "?")
                        room_name = m.get("room_name", "")
                        room_tag = f"[{room_name}] " if room_name else ""
                        body = m.get("body", "").strip()
                        print(f"{ts} {room_tag}{name}")
                        for line in body.split("\n"):
                            print(f"  {line}")
                        print()
            finally:
                cache.close()

        elif args.command == "unreplied":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                my_info = client.me()
                my_id = str(my_info["account_id"])
                my_name = my_info["name"]
                if args.sync:
                    print(
                        f"Syncing (top {args.sync_limit} rooms)...",
                        file=sys.stderr,
                    )
                    _sync_rooms(client, cache, args.sync_limit)
                    print("Sync complete.\n", file=sys.stderr)
                cli_config = _load_chatwork_tool_config()
                unreplied = cache.find_unreplied(
                    my_id,
                    exclude_toall=(not args.include_toall),
                    config=cli_config,
                )
                if getattr(args, "json", False):
                    output = []
                    for m in unreplied:
                        output.append(
                            {
                                "message_id": m.get("message_id", ""),
                                "room_id": m.get("room_id", ""),
                                "room_name": m.get("room_name", m.get("room_id", "")),
                                "account_id": m.get("account_id", ""),
                                "account_name": m.get("account_name", ""),
                                "body": m.get("body", "").strip(),
                                "send_time": m.get("send_time", 0),
                                "send_time_jst": m.get("send_time_jst", ""),
                            }
                        )
                    print(json.dumps(output, ensure_ascii=False, indent=2))
                elif not unreplied:
                    print(f"No unreplied messages ({my_name} / ID:{my_id})")
                else:
                    print(f"=== Unreplied: {len(unreplied)} ({my_name}) ===\n")
                    for m in unreplied:
                        ts = m.get("send_time_jst", "")
                        name = m.get("account_name", "?")
                        room_name = m.get("room_name", m.get("room_id", ""))
                        body = m.get("body", "").strip()
                        body_clean = re.sub(r"\[To:\d+\][^\n]*\n?", "", body).strip()
                        body_preview = body_clean.replace("\n", " ")[:120]
                        if len(body_clean) > 120:
                            body_preview += "..."
                        print(f"{ts} [{room_name}]")
                        print(f"  From: {name}")
                        print(f"  {body_preview}")
                        print()
            finally:
                cache.close()

        elif args.command == "rooms":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                rooms_data = client.rooms()
                for room in rooms_data:
                    cache.upsert_room(room)
                rooms_data.sort(key=lambda r: r.get("last_update_time", 0), reverse=True)
                print(f"{'ID':>12}  {'Updated':19}  {'Name'}")
                print("-" * 70)
                for r in rooms_data:
                    ts = _format_timestamp(r.get("last_update_time", 0))
                    print(f"{r['room_id']:>12}  {ts}  {r['name']}")
            finally:
                cache.close()

        elif args.command == "sync":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                if args.room:
                    room_id = client.resolve_room_id(args.room)
                    room_obj = {"room_id": room_id, "name": args.room}
                    cache.upsert_room(room_obj)
                    print(
                        f"Syncing room {args.room} (ID:{room_id})...",
                        end=" ",
                        flush=True,
                    )
                    msgs = client.get_messages(room_id, force=True)
                    if msgs:
                        cache.upsert_messages(room_id, msgs)
                        cache.update_sync_state(room_id)
                        print(f"{len(msgs)} messages")
                    else:
                        print("0 messages")
                else:
                    rooms_data = client.rooms()
                    for room in rooms_data:
                        cache.upsert_room(room)
                    rooms_data.sort(key=lambda r: r.get("last_update_time", 0), reverse=True)
                    rooms_to_sync = rooms_data[: args.limit]
                    print(f"Syncing {len(rooms_to_sync)} rooms (metadata: {len(rooms_data)} rooms saved)...\n")
                    total_msgs = 0
                    for i, room in enumerate(rooms_to_sync, 1):
                        rid = str(room["room_id"])
                        name = room.get("name", rid)
                        print(
                            f"[{i}/{len(rooms_to_sync)}] {name} (ID:{rid})...",
                            end=" ",
                            flush=True,
                        )
                        try:
                            msgs = client.get_messages(rid, force=True)
                            if msgs:
                                cache.upsert_messages(rid, msgs)
                                cache.update_sync_state(rid)
                                print(f"{len(msgs)} messages")
                                total_msgs += len(msgs)
                            else:
                                print("0 messages")
                        except Exception as exc:
                            print(f"Error: {exc}")
                        time.sleep(0.5)
                    stats = cache.get_stats()
                    print(f"\nSync complete: {total_msgs} messages fetched")
                    print(f"Cache total: {stats['rooms']} rooms / {stats['messages']} messages")
            finally:
                cache.close()

        elif args.command == "me":
            info = client.me()
            print(f"Account ID: {info['account_id']}")
            print(f"Name: {info['name']}")
            print(f"Email: {info.get('mail', 'N/A')}")
            print(f"Organization: {info.get('organization_name', 'N/A')}")

        elif args.command == "members":
            room_id = client.resolve_room_id(args.room)
            members = client.room_members(room_id)
            print(f"{'Account ID':>12}  {'Role':10}  {'Name'}")
            print("-" * 50)
            for m in members:
                print(f"{m['account_id']:>12}  {m.get('role', ''):10}  {m['name']}")

        elif args.command == "contacts":
            contacts = client.contacts()
            print(f"{'Account ID':>12}  {'Name'}")
            print("-" * 40)
            for c in contacts:
                print(f"{c['account_id']:>12}  {c['name']}")

        elif args.command == "task":
            room_id = client.resolve_room_id(args.room)
            result = client.add_task(room_id, args.body, args.to_ids)
            print(f"Task created: {result}")

        elif args.command == "mytasks":
            status = "done" if args.done else "open"
            tasks = client.my_tasks(status=status)
            if not tasks:
                print(f"No {status} tasks.")
                return
            print(f"=== My Tasks ({status}): {len(tasks)} ===\n")
            for t in tasks:
                room_name = t.get("room", {}).get("name", "?")
                body = t.get("body", "").strip()
                body_clean = re.sub(r"\[.*?\]", "", body).strip()
                body_preview = body_clean.replace("\n", " ")[:120]
                if len(body_clean) > 120:
                    body_preview += "..."
                limit_time = t.get("limit_time", 0)
                deadline = _format_timestamp(limit_time) if limit_time else "No deadline"
                assigned_by = t.get("assigned_by_account", {}).get("name", "?")
                print(f"[{room_name}]  Deadline: {deadline}  By: {assigned_by}")
                print(f"  {body_preview}")
                print()

        elif args.command == "tasks":
            room_id = client.resolve_room_id(args.room)
            status = "done" if args.done else "open"
            tasks = client.room_tasks(room_id, status=status)
            if not tasks:
                print(f"No {status} tasks.")
                return
            print(f"=== Room Tasks ({status}): {len(tasks)} ===\n")
            for t in tasks:
                body = t.get("body", "").strip()
                body_clean = re.sub(r"\[.*?\]", "", body).strip()
                body_preview = body_clean.replace("\n", " ")[:120]
                if len(body_clean) > 120:
                    body_preview += "..."
                limit_time = t.get("limit_time", 0)
                deadline = _format_timestamp(limit_time) if limit_time else "No deadline"
                assignee = t.get("account", {}).get("name", "?")
                assigned_by = t.get("assigned_by_account", {}).get("name", "?")
                print(f"  Assignee: {assignee}  Deadline: {deadline}  By: {assigned_by}")
                print(f"  {body_preview}")
                print()

        elif args.command == "mentions":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                my_info = client.me()
                my_id = str(my_info["account_id"])
                if args.sync:
                    print(
                        f"Syncing (top {args.sync_limit} rooms)...",
                        file=sys.stderr,
                    )
                    _sync_rooms(client, cache, args.sync_limit)
                    print("Sync complete.\n", file=sys.stderr)
                cli_config = _load_chatwork_tool_config()
                mentions = cache.find_mentions(
                    my_id,
                    exclude_toall=(not args.include_toall),
                    limit=args.num,
                    config=cli_config,
                )
                unreplied_set: set[tuple[str, str]] = set()
                if mentions:
                    unreplied_list = cache.find_unreplied(
                        my_id,
                        exclude_toall=(not args.include_toall),
                        config=cli_config,
                    )
                    unreplied_set = {(m["room_id"], m["message_id"]) for m in unreplied_list}
                if getattr(args, "json", False):
                    output = []
                    for m in mentions:
                        is_unreplied = (m["room_id"], m["message_id"]) in unreplied_set
                        output.append(
                            {
                                "message_id": m.get("message_id", ""),
                                "room_id": m.get("room_id", ""),
                                "room_name": m.get("room_name", m.get("room_id", "")),
                                "account_id": m.get("account_id", ""),
                                "account_name": m.get("account_name", ""),
                                "body": m.get("body", "").strip(),
                                "send_time": m.get("send_time", 0),
                                "send_time_jst": m.get("send_time_jst", ""),
                                "unreplied": is_unreplied,
                            }
                        )
                    print(json.dumps(output, ensure_ascii=False, indent=2))
                elif not mentions:
                    print("No mentions found.")
                    print("Hint: run 'chatwork mentions --sync' to sync first.")
                else:
                    print(f"=== Mentions: {len(mentions)} ===\n")
                    for m in mentions:
                        ts = m.get("send_time_jst", "")
                        name = m.get("account_name", "?")
                        room_name = m.get("room_name", m.get("room_id", ""))
                        is_unreplied = (m["room_id"], m["message_id"]) in unreplied_set
                        status = "[UNREPLIED] " if is_unreplied else "[replied]   "
                        body = m.get("body", "").strip()
                        body_clean = re.sub(r"\[To:\d+\][^\n]*\n?", "", body).strip()
                        body_preview = body_clean.replace("\n", " ")[:100]
                        if len(body_clean) > 100:
                            body_preview += "..."
                        print(f"{status}{ts} [{room_name}] {name}")
                        print(f"  {body_preview}")
                        print()
            finally:
                cache.close()

        elif args.command == "files":
            room_id = client.resolve_room_id(args.room)
            files = client.get_files(room_id, account_id=getattr(args, "account_id", None))
            if not files:
                print("No files found.")
            else:
                print(f"{'File ID':>12}  {'Size':>10}  {'Uploaded':19}  {'Uploader':20}  Name")
                print("-" * 90)
                for f in files:
                    ts = _format_timestamp(f.get("upload_time", 0))
                    uploader = f.get("account", {}).get("name", "?")
                    size = f.get("filesize", 0)
                    name = f.get("filename", "?")
                    print(f"{f['file_id']:>12}  {size:>10}  {ts}  {uploader:20}  {name}")

        elif args.command == "upload":
            try:
                room_id = client.resolve_room_id(args.room)
                message = md_to_chatwork(args.message) if args.message else None
                result = client.upload_file(room_id, Path(args.file).expanduser(), message)
                name = Path(args.file).name
                print(f"Uploaded '{name}' (file_id: {result['file_id']})")
            except (FileNotFoundError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "download":
            room_id = client.resolve_room_id(args.room)
            file_info = client.get_file(room_id, args.file_id, create_download_url=True)
            download_url = file_info.get("download_url")
            if not download_url:
                print("Error: download URL not available.", file=sys.stderr)
                sys.exit(1)
            filename = file_info.get("filename", args.file_id)
            output_path = Path(args.output) if args.output else Path(filename)
            print(f"Downloading '{filename}' → {output_path} ...", end=" ", flush=True)
            size = client.download_file(download_url, output_path)
            print(f"done ({size:,} bytes)")

        elif args.command == "stats":
            cache = MessageCache(db_path=resolve_cache_db_path(client))
            try:
                stats = cache.get_stats()
                print("Cache statistics:")
                print(f"  Rooms: {stats['rooms']}")
                print(f"  Messages: {stats['messages']}")
                print(f"  DB file: {cache.db_path}")
            finally:
                cache.close()

    run_cli_safely(_run)


if __name__ == "__main__":
    cli_main()
