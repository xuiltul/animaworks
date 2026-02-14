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
        description="AnimaWorks - Digital Person Framework"
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
        "--reset", action="store_true",
        help="DELETE runtime directory and re-initialize (dangerous)",
    )
    init_mode.add_argument(
        "--template", metavar="NAME",
        help="Non-interactive: create person from named template",
    )
    init_mode.add_argument(
        "--from-md", metavar="PATH",
        help="Non-interactive: create person from MD file",
    )
    init_mode.add_argument(
        "--blank", metavar="NAME",
        help="Non-interactive: create blank person with given name",
    )
    init_mode.add_argument(
        "--skip-person", action="store_true",
        help="Initialize infrastructure only, skip person creation",
    )
    p_init.add_argument(
        "--name", default=None,
        help="Override person name (used with --from-md)",
    )
    p_init.set_defaults(func=_lazy_init)

    # ── Create Person ─────────────────────────────────────
    p_create = sub.add_parser(
        "create-person", help="Create a new Digital Person"
    )
    p_create.add_argument(
        "--name", default=None,
        help="Person name (required for blank, optional for template/md)",
    )
    p_create.add_argument(
        "--template", default=None,
        help="Create from a named template",
    )
    p_create.add_argument(
        "--from-md", default=None, metavar="PATH",
        help="Create from an MD file",
    )
    p_create.set_defaults(func=_lazy_create_person)

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
    p_chat = sub.add_parser("chat", help="Chat with a person")
    p_chat.add_argument("person", help="Person name")
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
    p_hb.add_argument("person", help="Person name")
    p_hb.add_argument(
        "--local", action="store_true", help="Direct mode (no gateway)"
    )
    p_hb.set_defaults(func=_lazy_heartbeat)

    # ── Send ──────────────────────────────────────────────
    p_send = sub.add_parser("send", help="Send message between persons")
    p_send.add_argument("from_person", help="Sender name")
    p_send.add_argument("to_person", help="Recipient name")
    p_send.add_argument("message", help="Message content")
    p_send.add_argument("--thread-id", default=None, help="Thread ID")
    p_send.add_argument("--reply-to", default=None, help="Reply to message ID")
    p_send.set_defaults(func=_lazy_send)

    # ── List ──────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List all persons")
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


def _lazy_create_person(args: argparse.Namespace) -> None:
    from cli.commands.person import cmd_create_person

    cmd_create_person(args)


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


def _lazy_gateway(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_gateway

    cmd_gateway(args)


def _lazy_worker(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_worker

    cmd_worker(args)


def _lazy_chat(args: argparse.Namespace) -> None:
    from cli.commands.person import cmd_chat

    cmd_chat(args)


def _lazy_heartbeat(args: argparse.Namespace) -> None:
    from cli.commands.person import cmd_heartbeat

    cmd_heartbeat(args)


def _lazy_send(args: argparse.Namespace) -> None:
    from cli.commands.messaging import cmd_send

    cmd_send(args)


def _lazy_list(args: argparse.Namespace) -> None:
    from cli.commands.messaging import cmd_list

    cmd_list(args)


def _lazy_status(args: argparse.Namespace) -> None:
    from cli.commands.messaging import cmd_status

    cmd_status(args)
