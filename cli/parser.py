# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os


def cli_main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from core.logging_config import setup_logging
    from core.paths import get_data_dir

    setup_logging(
        level=os.environ.get("ANIMAWORKS_LOG_LEVEL", "INFO"),
        log_dir=get_data_dir() / "logs",
    )

    parser = argparse.ArgumentParser(
        description="AnimaWorks - Digital Anima Framework"
    )
    parser.add_argument("--gateway-url", default=None, help="Gateway URL")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override runtime data directory (default: ~/.animaworks or ANIMAWORKS_DATA_DIR)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── Init ──────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Initialize runtime directory from templates")
    init_mode = p_init.add_mutually_exclusive_group()
    init_mode.add_argument(
        "--force", action="store_true",
        help="Merge missing template files into existing runtime",
    )
    init_mode.add_argument(
        "--template", metavar="NAME",
        help="Non-interactive: create anima from named template",
    )
    init_mode.add_argument(
        "--from-md", metavar="PATH",
        help="Non-interactive: create anima from MD file",
    )
    init_mode.add_argument(
        "--blank", metavar="NAME",
        help="Non-interactive: create blank anima with given name",
    )
    init_mode.add_argument(
        "--skip-anima", action="store_true",
        help="Initialize infrastructure only, skip anima creation",
    )
    p_init.add_argument(
        "--name", default=None,
        help="Override anima name (used with --from-md)",
    )
    p_init.set_defaults(func=_lazy_init)

    # ── Create Anima ─────────────────────────────────────
    p_create = sub.add_parser(
        "create-anima", help="Create a new Digital Anima"
    )
    p_create.add_argument(
        "--name", default=None,
        help="Anima name (required for blank, optional for template/md)",
    )
    p_create.add_argument(
        "--template", default=None,
        help="Create from a named template",
    )
    p_create.add_argument(
        "--from-md", default=None, metavar="PATH",
        help="Create from an MD file",
    )
    p_create.add_argument(
        "--supervisor", default=None,
        help="Supervisor anima name (overrides character sheet)",
    )
    p_create.add_argument(
        "--role", default=None,
        choices=["engineer", "researcher", "manager", "writer", "ops", "general"],
        help="Role template to apply (default: general)",
    )
    p_create.set_defaults(func=_lazy_create_anima)

    # ── Start ─────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Start the AnimaWorks server")
    p_start.add_argument("--host", default="0.0.0.0")
    p_start.add_argument("--port", type=int, default=18500)
    p_start.set_defaults(func=_lazy_start)

    # ── Serve (alias) ─────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start the server (alias for start)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=18500)
    p_serve.set_defaults(func=_lazy_serve)

    # ── Stop ──────────────────────────────────────────────
    p_stop = sub.add_parser("stop", help="Stop the running server")
    p_stop.set_defaults(func=_lazy_stop)

    # ── Reset ─────────────────────────────────────────────
    p_reset = sub.add_parser(
        "reset", help="Stop server, delete runtime directory, and re-initialize",
    )
    p_reset.add_argument(
        "--restart", action="store_true",
        help="Start the server after reset",
    )
    p_reset.set_defaults(func=_lazy_reset)

    # ── Restart ───────────────────────────────────────────
    p_restart = sub.add_parser("restart", help="Restart the server (stop then start)")
    p_restart.add_argument("--host", default="0.0.0.0")
    p_restart.add_argument("--port", type=int, default=18500)
    p_restart.set_defaults(func=_lazy_restart)

    # ── Gateway (deprecated) ──────────────────────────────
    p_gw = sub.add_parser("gateway", help=argparse.SUPPRESS)
    p_gw.set_defaults(func=_lazy_gateway)

    # ── Worker (deprecated) ───────────────────────────────
    p_wk = sub.add_parser("worker", help=argparse.SUPPRESS)
    p_wk.set_defaults(func=_lazy_worker)

    # ── Chat ──────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Chat with an anima")
    p_chat.add_argument("anima", help="Anima name")
    p_chat.add_argument("message", help="Message to send")
    p_chat.add_argument(
        "--local", action="store_true", help="Direct mode (no gateway)"
    )
    p_chat.add_argument(
        "--from", dest="from_person", default="human",
        help="Sender name (default: human)",
    )
    p_chat.set_defaults(func=_lazy_chat)

    # ── Heartbeat ─────────────────────────────────────────
    p_hb = sub.add_parser("heartbeat", help="Trigger heartbeat")
    p_hb.add_argument("anima", help="Anima name")
    p_hb.add_argument(
        "--local", action="store_true", help="Direct mode (no gateway)"
    )
    p_hb.set_defaults(func=_lazy_heartbeat)

    # ── Send ──────────────────────────────────────────────
    p_send = sub.add_parser("send", help="Send message between animas")
    p_send.add_argument("from_person", help="Sender name")
    p_send.add_argument("to_person", help="Recipient name")
    p_send.add_argument("message", help="Message content")
    p_send.add_argument("--thread-id", default=None, help="Thread ID")
    p_send.add_argument("--reply-to", default=None, help="Reply to message ID")
    p_send.set_defaults(func=_lazy_send)

    # ── Board ─────────────────────────────────────────────
    p_board = sub.add_parser("board", help="Board shared channel operations")
    board_sub = p_board.add_subparsers(dest="board_command")

    # board read
    p_board_read = board_sub.add_parser("read", help="Read channel messages")
    p_board_read.add_argument("channel", help="Channel name (e.g. general, ops)")
    p_board_read.add_argument("--limit", type=int, default=20, help="Max messages")
    p_board_read.add_argument(
        "--human-only", action="store_true", help="Show human messages only",
    )
    p_board_read.set_defaults(func=_lazy_board_read)

    # board post
    p_board_post = board_sub.add_parser("post", help="Post to channel")
    p_board_post.add_argument("from_anima", help="Sender anima name")
    p_board_post.add_argument("channel", help="Channel name")
    p_board_post.add_argument("text", help="Message text")
    p_board_post.set_defaults(func=_lazy_board_post)

    # board dm-history
    p_board_dm = board_sub.add_parser("dm-history", help="Read DM history with peer")
    p_board_dm.add_argument("from_anima", help="Self anima name")
    p_board_dm.add_argument("peer", help="Peer anima name")
    p_board_dm.add_argument("--limit", type=int, default=20, help="Max messages")
    p_board_dm.set_defaults(func=_lazy_board_dm_history)

    # ── List ──────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List all animas")
    p_list.add_argument(
        "--local", action="store_true", help="Scan filesystem directly"
    )
    p_list.set_defaults(func=_lazy_list)

    # ── Status ────────────────────────────────────────────
    p_status = sub.add_parser("status", help="Show system status from gateway")
    p_status.set_defaults(func=_lazy_status)

    # ── Index ─────────────────────────────────────────────
    from cli.commands.index_cmd import setup_index_command

    setup_index_command(sub)

    # ── Config ────────────────────────────────────────────
    from core.config.cli import (
        cmd_config_dispatch,
        cmd_config_get,
        cmd_config_list,
        cmd_config_set,
    )

    p_config = sub.add_parser("config", help="Manage configuration")
    p_config.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive setup wizard",
    )
    p_config.set_defaults(func=cmd_config_dispatch, config_parser=p_config)
    config_sub = p_config.add_subparsers(dest="config_command")

    p_cfg_get = config_sub.add_parser("get", help="Get a config value")
    p_cfg_get.add_argument("key", help="Dot-notation key (e.g. system.gateway.port)")
    p_cfg_get.add_argument(
        "--show-secrets", action="store_true", help="Show API key values",
    )
    p_cfg_get.set_defaults(func=cmd_config_get)

    p_cfg_set = config_sub.add_parser("set", help="Set a config value")
    p_cfg_set.add_argument("key", help="Dot-notation key")
    p_cfg_set.add_argument("value", help="Value to set")
    p_cfg_set.set_defaults(func=cmd_config_set)

    p_cfg_list = config_sub.add_parser("list", help="List all config values")
    p_cfg_list.add_argument("--section", default=None, help="Filter by section")
    p_cfg_list.add_argument(
        "--show-secrets", action="store_true", help="Show API key values",
    )
    p_cfg_list.set_defaults(func=cmd_config_list)

    # ── Anima Management ─────────────────────────────────────
    p_anima = sub.add_parser("anima", help="Manage anima processes")
    anima_sub = p_anima.add_subparsers(dest="anima_command")

    # anima restart
    p_anima_restart = anima_sub.add_parser("restart", help="Restart an anima process")
    p_anima_restart.add_argument("anima", help="Anima name")
    p_anima_restart.set_defaults(func=_lazy_anima_restart)

    # anima status
    p_anima_status = anima_sub.add_parser("status", help="Show anima process status")
    p_anima_status.add_argument(
        "anima", nargs="?", default=None,
        help="Anima name (omit for all animas)"
    )
    p_anima_status.set_defaults(func=_lazy_anima_status)

    # anima create
    p_anima_create = anima_sub.add_parser("create", help="Create a new anima")
    p_anima_create.add_argument(
        "--name", default=None,
        help="Anima name (required for blank, optional for template/md)",
    )
    p_anima_create.add_argument(
        "--template", default=None,
        help="Create from a named template",
    )
    p_anima_create.add_argument(
        "--from-md", default=None, metavar="PATH",
        help="Create from an MD file",
    )
    p_anima_create.add_argument(
        "--supervisor", default=None,
        help="Supervisor anima name (overrides character sheet)",
    )
    p_anima_create.add_argument(
        "--role", default=None,
        choices=["engineer", "researcher", "manager", "writer", "ops", "general"],
        help="Role template to apply (default: general)",
    )
    p_anima_create.set_defaults(func=_lazy_anima_create)

    # anima delete
    p_anima_delete = anima_sub.add_parser("delete", help="Delete an anima (with optional archive)")
    p_anima_delete.add_argument("anima", help="Anima name to delete")
    p_anima_delete.add_argument(
        "--no-archive", action="store_true",
        help="Skip creating a ZIP archive before deletion",
    )
    p_anima_delete.add_argument(
        "--force", action="store_true",
        help="Skip confirmation prompt",
    )
    p_anima_delete.set_defaults(func=_lazy_anima_delete)

    # anima disable
    p_anima_disable = anima_sub.add_parser("disable", help="Disable (休養) an anima")
    p_anima_disable.add_argument("anima", help="Anima name to disable")
    p_anima_disable.set_defaults(func=_lazy_anima_disable)

    # anima enable
    p_anima_enable = anima_sub.add_parser("enable", help="Enable (復帰) an anima")
    p_anima_enable.add_argument("anima", help="Anima name to enable")
    p_anima_enable.set_defaults(func=_lazy_anima_enable)

    # anima list
    p_anima_list = anima_sub.add_parser("list", help="List all animas with status")
    p_anima_list.add_argument(
        "--local", action="store_true", help="Scan filesystem directly"
    )
    p_anima_list.set_defaults(func=_lazy_anima_list)

    # ── Logs ──────────────────────────────────────────────────
    p_logs = sub.add_parser("logs", help="View anima logs")
    p_logs.add_argument(
        "anima", nargs="?", default=None,
        help="Anima name (required unless --all)"
    )
    p_logs.add_argument(
        "--all", action="store_true",
        help="Show all logs (server + all animas)"
    )
    p_logs.add_argument(
        "--lines", type=int, default=50,
        help="Number of lines to show (default: 50)"
    )
    p_logs.add_argument(
        "--date", default=None,
        help="Specific date (YYYYMMDD format)"
    )
    p_logs.set_defaults(func=_lazy_logs)

    # ── Migrate Cron ─────────────────────────────────────────
    p_migrate_cron = sub.add_parser(
        "migrate-cron",
        help="Migrate cron.md files from Japanese format to standard cron expressions",
    )
    p_migrate_cron.set_defaults(func=_lazy_migrate_cron)

    # ── Optimize Assets ──────────────────────────────────────
    from cli.commands import optimize_assets

    optimize_assets.register(sub)

    # ── Remake Assets ─────────────────────────────────────────
    from cli.commands import remake_cmd

    remake_cmd.register(sub)

    args = parser.parse_args()

    # Apply --data-dir override before any command
    if args.data_dir:
        os.environ["ANIMAWORKS_DATA_DIR"] = args.data_dir

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


# ── Lazy import wrappers ──────────────────────────────────


def _lazy_init(args: argparse.Namespace) -> None:
    from cli.commands.init_cmd import cmd_init

    cmd_init(args)


def _lazy_create_anima(args: argparse.Namespace) -> None:
    import sys

    print("Warning: 'create-anima' is deprecated. Use 'anima create' instead.", file=sys.stderr)
    from cli.commands.anima import cmd_create_anima

    cmd_create_anima(args)


def _lazy_start(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_start

    cmd_start(args)


def _lazy_serve(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_serve

    cmd_serve(args)


def _lazy_stop(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_stop

    cmd_stop(args)


def _lazy_restart(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_restart

    cmd_restart(args)


def _lazy_reset(args: argparse.Namespace) -> None:
    from cli.commands.init_cmd import cmd_reset

    cmd_reset(args)


def _lazy_gateway(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_gateway

    cmd_gateway(args)


def _lazy_worker(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_worker

    cmd_worker(args)


def _lazy_chat(args: argparse.Namespace) -> None:
    from cli.commands.anima import cmd_chat

    cmd_chat(args)


def _lazy_heartbeat(args: argparse.Namespace) -> None:
    from cli.commands.anima import cmd_heartbeat

    cmd_heartbeat(args)


def _lazy_send(args: argparse.Namespace) -> None:
    from cli.commands.messaging import cmd_send

    cmd_send(args)


def _lazy_list(args: argparse.Namespace) -> None:
    import sys

    print("Warning: 'list' is deprecated. Use 'anima list' instead.", file=sys.stderr)
    from cli.commands.messaging import cmd_list

    cmd_list(args)


def _lazy_anima_restart(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_restart

    cmd_anima_restart(args)


def _lazy_anima_status(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_status

    cmd_anima_status(args)


def _lazy_logs(args: argparse.Namespace) -> None:
    from cli.commands.logs import cmd_logs

    cmd_logs(args)


def _lazy_status(args: argparse.Namespace) -> None:
    from cli.commands.messaging import cmd_status

    cmd_status(args)


def _lazy_migrate_cron(args: argparse.Namespace) -> None:
    from core.config.migrate import migrate_all_cron
    from core.paths import get_data_dir

    animas_dir = get_data_dir() / "animas"
    count = migrate_all_cron(animas_dir)
    if count:
        print(f"Migrated {count} anima(s) to standard cron format.")
    else:
        print("No migration needed — all cron.md files are already in standard format.")


def _lazy_anima_create(args: argparse.Namespace) -> None:
    from cli.commands.anima import cmd_create_anima

    cmd_create_anima(args)


def _lazy_anima_delete(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_delete

    cmd_anima_delete(args)


def _lazy_anima_disable(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_disable

    cmd_anima_disable(args)


def _lazy_anima_enable(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_enable

    cmd_anima_enable(args)


def _lazy_anima_list(args: argparse.Namespace) -> None:
    from cli.commands.anima_mgmt import cmd_anima_list

    cmd_anima_list(args)


def _lazy_board_read(args: argparse.Namespace) -> None:
    from cli.commands.board import cmd_board_read

    cmd_board_read(args)


def _lazy_board_post(args: argparse.Namespace) -> None:
    from cli.commands.board import cmd_board_post

    cmd_board_post(args)


def _lazy_board_dm_history(args: argparse.Namespace) -> None:
    from cli.commands.board import cmd_board_dm_history

    cmd_board_dm_history(args)
