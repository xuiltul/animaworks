from __future__ import annotations

import atexit
import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.init import ensure_runtime_dir
from core.paths import get_data_dir, get_persons_dir, get_shared_dir

from core.logging_config import setup_logging

setup_logging(
    level=os.environ.get("ANIMAWORKS_LOG_LEVEL", "INFO"),
    log_dir=get_data_dir() / "logs",
)
logger = logging.getLogger("animaworks")


# ── Init ──────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize the runtime data directory from templates."""
    from pathlib import Path

    from core.init import merge_templates, reset_runtime_dir
    from core.person_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        validate_person_name,
    )

    data_dir = get_data_dir()

    # --reset: complete deletion + re-initialization (interactive)
    if getattr(args, "reset", False):
        if data_dir.exists():
            answer = input(
                f"WARNING: This will DELETE all data in {data_dir}\n"
                f"  (episodes, knowledge, state, config — all will be lost)\n"
                f"Continue? [yes/no]: "
            )
            if answer.strip().lower() != "yes":
                print("Aborted.")
                return
        reset_runtime_dir(data_dir, skip_persons=True)
        print(f"Runtime directory reset: {data_dir}")
        _interactive_person_setup(data_dir)
        _interactive_user_setup(data_dir)
        return

    # --force: safe merge (add missing files only)
    if getattr(args, "force", False):
        if not data_dir.exists():
            ensure_runtime_dir(skip_persons=True)
            print(f"Runtime directory initialized: {data_dir}")
            _interactive_person_setup(data_dir)
            _interactive_user_setup(data_dir)
            return
        added = merge_templates(data_dir)
        if added:
            print(f"Merged {len(added)} new file(s) from templates:")
            for f in added:
                print(f"  + {f}")
        else:
            print("Already up to date — no new template files to add.")
        return

    # Non-interactive shortcuts
    # Always call ensure_runtime_dir — it's idempotent (checks config.json).
    persons_dir = data_dir / "persons"

    if getattr(args, "template", None):
        ensure_runtime_dir(skip_persons=True)
        persons_dir.mkdir(parents=True, exist_ok=True)
        person_dir = create_from_template(persons_dir, args.template)
        _register_person_in_config(data_dir, person_dir.name, role="commander")
        print(f"Created person '{person_dir.name}' from template '{args.template}'")
        return

    if getattr(args, "from_md", None):
        ensure_runtime_dir(skip_persons=True)
        persons_dir.mkdir(parents=True, exist_ok=True)
        md_path = Path(args.from_md).resolve()
        person_dir = create_from_md(
            persons_dir, md_path, name=getattr(args, "name", None)
        )
        _register_person_in_config(data_dir, person_dir.name, role="commander")
        print(f"Created person '{person_dir.name}' from {md_path.name}")
        return

    if getattr(args, "blank", None):
        ensure_runtime_dir(skip_persons=True)
        persons_dir.mkdir(parents=True, exist_ok=True)
        err = validate_person_name(args.blank)
        if err:
            print(f"Error: {err}")
            sys.exit(1)
        person_dir = create_blank(persons_dir, args.blank)
        _register_person_in_config(data_dir, person_dir.name, role="commander")
        print(f"Created blank person '{person_dir.name}'")
        return

    if getattr(args, "skip_person", False):
        ensure_runtime_dir(skip_persons=True)
        print(f"Runtime directory initialized (no persons): {data_dir}")
        return

    # Default: interactive first-time setup
    # Use config.json as the proper initialization marker
    config_json = data_dir / "config.json"
    if config_json.exists():
        print(f"Runtime directory already exists: {data_dir}")
        print("Use --force to merge new template files, or --reset to re-initialize.")
        return

    ensure_runtime_dir(skip_persons=True)
    print(f"Runtime directory initialized: {data_dir}")
    _interactive_person_setup(data_dir)
    _interactive_user_setup(data_dir)


def _interactive_person_setup(data_dir) -> None:
    """Interactive person creation during init."""
    from pathlib import Path

    from core.person_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        list_person_templates,
        validate_person_name,
    )

    persons_dir = data_dir / "persons"
    persons_dir.mkdir(parents=True, exist_ok=True)

    templates = list_person_templates()
    template_list = ", ".join(templates) if templates else "none"

    print()
    print("最初のDigital Personをどのように作成しますか？")
    print(f"  1. テンプレートから作成 ({template_list})")
    print("  2. MDファイルから作成")
    print("  3. ブランクで作成（名前のみ指定）")
    print("  4. スキップ（後で作成）")

    choice = input("\n選択 [1]: ").strip() or "1"

    if choice == "1":
        if not templates:
            print("テンプレートが見つかりません。ブランクで作成します。")
            choice = "3"
        elif len(templates) == 1:
            tpl = templates[0]
            person_dir = create_from_template(persons_dir, tpl)
            _register_person_in_config(data_dir, person_dir.name, role="commander")
            print(f"\n{person_dir.name} を作成しました（commander）。")
            return
        else:
            print(f"\n利用可能なテンプレート:")
            for i, t in enumerate(templates, 1):
                print(f"  {i}. {t}")
            idx = input(f"番号 [1]: ").strip() or "1"
            try:
                tpl = templates[int(idx) - 1]
            except (ValueError, IndexError):
                tpl = templates[0]
            person_dir = create_from_template(persons_dir, tpl)
            _register_person_in_config(data_dir, person_dir.name, role="commander")
            print(f"\n{person_dir.name} を作成しました（commander）。")
            return

    if choice == "2":
        md_path_str = input("MDファイルのパス: ").strip()
        if not md_path_str:
            print("スキップしました。")
            return
        md_path = Path(md_path_str).expanduser().resolve()
        if not md_path.exists():
            print(f"ファイルが見つかりません: {md_path}")
            return
        name = input("パーソン名（英小文字、空欄で自動検出）: ").strip() or None
        if name:
            err = validate_person_name(name)
            if err:
                print(f"Error: {err}")
                return
        try:
            person_dir = create_from_md(persons_dir, md_path, name=name)
            _register_person_in_config(data_dir, person_dir.name, role="commander")
            print(f"\n{person_dir.name} を作成しました（commander）。")
        except ValueError as e:
            print(f"Error: {e}")
        return

    if choice == "3":
        name = input("パーソン名（英小文字）: ").strip()
        if not name:
            print("スキップしました。")
            return
        err = validate_person_name(name)
        if err:
            print(f"Error: {err}")
            return
        person_dir = create_blank(persons_dir, name)
        _register_person_in_config(data_dir, person_dir.name, role="commander")
        print(f"\n{person_dir.name} を作成しました（commander）。")
        return

    # choice == "4" or anything else
    print("パーソンの作成をスキップしました。")


def _interactive_user_setup(data_dir) -> None:
    """Optionally collect user info during init."""
    print()
    answer = input("あなたの情報を登録しますか？ (パーソンがあなたを覚えます) [Y/n]: ").strip()
    if answer.lower() in ("n", "no"):
        return

    user_name = input("  お名前: ").strip()
    if not user_name:
        return

    timezone = input("  タイムゾーン [Asia/Tokyo]: ").strip() or "Asia/Tokyo"
    notes = input("  メモ（任意）: ").strip()

    # Create user directory following the behavior_rules.md structure
    user_dir = data_dir / "shared" / "users" / user_name
    user_dir.mkdir(parents=True, exist_ok=True)

    index_content = (
        f"# {user_name}\n\n"
        f"## 基本情報\n"
        f"- 名前: {user_name}\n"
        f"- タイムゾーン: {timezone}\n"
    )
    if notes:
        index_content += f"\n## 重要な好み・傾向\n{notes}\n"
    else:
        index_content += "\n## 重要な好み・傾向\n\n"
    index_content += "\n## 注意事項\n\n"

    (user_dir / "index.md").write_text(index_content, encoding="utf-8")
    (user_dir / "log.md").write_text("", encoding="utf-8")

    print(f"\nユーザー情報を保存しました: {user_dir}/index.md")


def _register_person_in_config(
    data_dir, person_name: str, *, role: str | None = None
) -> None:
    """Register a newly created person in config.json."""
    from core.config import PersonModelConfig, load_config, save_config

    config_path = data_dir / "config.json"
    if not config_path.exists():
        return
    config = load_config(config_path)
    if person_name not in config.persons:
        config.persons[person_name] = PersonModelConfig(role=role)
        save_config(config, config_path)


# ── Create Person ─────────────────────────────────────────


def cmd_create_person(args: argparse.Namespace) -> None:
    """Create a new Digital Person."""
    from pathlib import Path

    from core.person_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        validate_person_name,
    )

    ensure_runtime_dir(skip_persons=True)
    data_dir = get_data_dir()
    persons_dir = get_persons_dir()
    persons_dir.mkdir(parents=True, exist_ok=True)

    if args.from_md:
        md_path = Path(args.from_md).resolve()
        person_dir = create_from_md(persons_dir, md_path, name=args.name)
        _register_person_in_config(data_dir, person_dir.name)
        print(f"Created person '{person_dir.name}' from {md_path.name}")
        return

    if args.template:
        person_dir = create_from_template(
            persons_dir, args.template, person_name=args.name
        )
        _register_person_in_config(data_dir, person_dir.name)
        print(f"Created person '{person_dir.name}' from template '{args.template}'")
        return

    # Default: blank creation
    name = args.name
    if not name:
        print("Error: --name is required for blank person creation")
        sys.exit(1)
    err = validate_person_name(name)
    if err:
        print(f"Error: {err}")
        sys.exit(1)
    person_dir = create_blank(persons_dir, name)
    _register_person_in_config(data_dir, person_dir.name)
    print(f"Created blank person '{person_dir.name}'")


# ── Server ────────────────────────────────────────────────


def _get_pid_file() -> Path:
    """Return the path to the server PID file."""
    return get_data_dir() / "server.pid"


def _write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    pid_file = _get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    logger.info("PID file written: %s (pid=%d)", pid_file, os.getpid())


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_file = _get_pid_file()
    try:
        pid_file.unlink(missing_ok=True)
        logger.debug("PID file removed: %s", pid_file)
    except OSError as exc:
        logger.warning("Failed to remove PID file %s: %s", pid_file, exc)


def _read_pid() -> int | None:
    """Read and validate the PID from the PID file.

    Returns the PID if the file exists and contains a valid integer,
    or None if the file is missing or contains invalid data.
    """
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
        return int(text)
    except (ValueError, OSError) as exc:
        logger.warning("Invalid PID file %s: %s", pid_file, exc)
        return None


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_server(timeout: int = 10) -> bool:
    """Send SIGTERM to the running server and wait for it to exit.

    Args:
        timeout: Maximum seconds to wait before reporting failure.

    Returns:
        True if the server was stopped (or was not running), False if
        it failed to stop within the timeout.
    """
    pid = _read_pid()
    if pid is None:
        print("No PID file found. Server is not running.")
        return True

    if not _is_process_alive(pid):
        print(f"Stale PID file (pid={pid}). Server is not running. Cleaning up.")
        _remove_pid_file()
        return True

    print(f"Stopping server (pid={pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("Server already exited.")
        _remove_pid_file()
        return True
    except PermissionError:
        print(f"Error: Permission denied sending signal to pid={pid}.")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            print("Server stopped.")
            _remove_pid_file()
            return True
        time.sleep(0.2)

    print(f"Error: Server (pid={pid}) did not stop within {timeout}s.")
    return False


def cmd_start(args: argparse.Namespace) -> None:
    """Start the AnimaWorks server."""
    import uvicorn
    from server.app import create_app

    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_alive(existing_pid):
        print(f"Error: Server is already running (pid={existing_pid}).")
        print("Use 'animaworks stop' first, or 'animaworks restart'.")
        sys.exit(1)
    elif existing_pid is not None:
        logger.info("Stale PID file found (pid=%d). Cleaning up.", existing_pid)
        _remove_pid_file()

    ensure_runtime_dir()
    _write_pid_file()
    atexit.register(_remove_pid_file)

    try:
        app = create_app(get_persons_dir(), get_shared_dir())
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        _remove_pid_file()


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the server (alias for 'start')."""
    cmd_start(args)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running AnimaWorks server."""
    if not _stop_server():
        sys.exit(1)


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart the AnimaWorks server (stop then start)."""
    if not _stop_server():
        print("Error: Cannot restart — failed to stop the running server.")
        sys.exit(1)
    time.sleep(0.5)
    cmd_start(args)


# ── Deprecated modes ──────────────────────────────────────


def cmd_gateway(args: argparse.Namespace) -> None:
    print("Error: 'gateway' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)


def cmd_worker(args: argparse.Namespace) -> None:
    print("Error: 'worker' mode has been deprecated. Use 'animaworks start' instead.")
    sys.exit(1)


# ── Chat ───────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    """Chat with a person (via gateway or direct)."""
    if args.local:
        from core.person import DigitalPerson

        ensure_runtime_dir()
        person_dir = get_persons_dir() / args.person
        if not person_dir.exists():
            print(f"Person not found: {args.person}")
            sys.exit(1)

        person = DigitalPerson(person_dir, get_shared_dir())
        response = asyncio.run(
            person.process_message(args.message, from_person=args.from_person)
        )
        print(response)
    else:
        import httpx

        gateway = args.gateway_url or os.environ.get(
            "ANIMAWORKS_GATEWAY_URL", "http://localhost:18500"
        )
        try:
            resp = httpx.post(
                f"{gateway}/api/persons/{args.person}/chat",
                json={"message": args.message, "from_person": args.from_person},
                timeout=300.0,
            )
            data = resp.json()
            print(data.get("response", data.get("error", "Unknown error")))
        except httpx.ConnectError:
            print(f"Cannot connect to gateway at {gateway}. Use --local for direct mode.")
            sys.exit(1)


# ── Heartbeat ──────────────────────────────────────────────


def cmd_heartbeat(args: argparse.Namespace) -> None:
    """Trigger heartbeat (via gateway or direct)."""
    if args.local:
        from core.person import DigitalPerson

        ensure_runtime_dir()
        person_dir = get_persons_dir() / args.person
        if not person_dir.exists():
            print(f"Person not found: {args.person}")
            sys.exit(1)

        person = DigitalPerson(person_dir, get_shared_dir())
        result = asyncio.run(person.run_heartbeat())
        print(f"[{result.action}] {result.summary[:500]}")
    else:
        import httpx

        gateway = args.gateway_url or os.environ.get(
            "ANIMAWORKS_GATEWAY_URL", "http://localhost:18500"
        )
        try:
            resp = httpx.post(
                f"{gateway}/api/persons/{args.person}/trigger",
                timeout=120.0,
            )
            print(resp.json())
        except httpx.ConnectError:
            print(f"Cannot connect to gateway at {gateway}. Use --local for direct mode.")
            sys.exit(1)


# ── Send ───────────────────────────────────────────────────


def cmd_send(args: argparse.Namespace) -> None:
    """Send a message from one person to another (filesystem based)."""
    from core.messenger import Messenger

    ensure_runtime_dir()
    messenger = Messenger(get_shared_dir(), args.from_person)
    msg = messenger.send(
        to=args.to_person,
        content=args.message,
        thread_id=args.thread_id or "",
        reply_to=args.reply_to or "",
    )
    print(f"Sent: {msg.from_person} -> {msg.to_person} (id: {msg.id}, thread: {msg.thread_id})")


# ── List ───────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    """List all persons (from gateway or filesystem)."""
    if args.local:
        _list_local()
    else:
        import httpx

        gateway = args.gateway_url or os.environ.get(
            "ANIMAWORKS_GATEWAY_URL", "http://localhost:18500"
        )
        try:
            resp = httpx.get(f"{gateway}/api/persons", timeout=10.0)
            for p in resp.json():
                name = p.get("name", "unknown")
                status = p.get("status", "unknown")
                print(f"  {name} ({status})")
        except httpx.ConnectError:
            print("Gateway not reachable, falling back to filesystem...")
            _list_local()


def _list_local() -> None:
    ensure_runtime_dir()
    persons_dir = get_persons_dir()
    if not persons_dir.exists():
        print("No persons directory found.")
        return
    for d in sorted(persons_dir.iterdir()):
        if d.is_dir() and (d / "identity.md").exists():
            print(f"  {d.name}")


# ── Status ─────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    import httpx
    url = args.gateway_url or os.environ.get(
        "ANIMAWORKS_GATEWAY_URL", "http://localhost:18500"
    )
    try:
        resp = httpx.get(f"{url}/api/system/status", timeout=10.0)
        data = resp.json()
        print(f"Persons: {data.get('persons', 0)}")
        print(f"Scheduler: {'running' if data.get('scheduler_running') else 'stopped'}")
        for j in data.get("jobs", []):
            print(f"  [{j['id']}] {j['name']} -> next: {j['next_run']}")
    except httpx.ConnectError:
        print(f"Cannot connect to server at {url}.")
        sys.exit(1)


# ── CLI Parser ─────────────────────────────────────────────


def cli_main() -> None:
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

    # Init
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
    p_init.set_defaults(func=cmd_init)

    # Create Person
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
    p_create.set_defaults(func=cmd_create_person)

    # Start
    p_start = sub.add_parser("start", help="Start the AnimaWorks server")
    p_start.add_argument("--host", default="0.0.0.0")
    p_start.add_argument("--port", type=int, default=18500)
    p_start.set_defaults(func=cmd_start)

    # Serve (alias)
    p_serve = sub.add_parser("serve", help="Start the server (alias for start)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=18500)
    p_serve.set_defaults(func=cmd_serve)

    # Stop
    p_stop = sub.add_parser("stop", help="Stop the running server")
    p_stop.set_defaults(func=cmd_stop)

    # Restart
    p_restart = sub.add_parser("restart", help="Restart the server (stop then start)")
    p_restart.add_argument("--host", default="0.0.0.0")
    p_restart.add_argument("--port", type=int, default=18500)
    p_restart.set_defaults(func=cmd_restart)

    # Gateway (deprecated)
    p_gw = sub.add_parser("gateway", help=argparse.SUPPRESS)
    p_gw.set_defaults(func=cmd_gateway)

    # Worker (deprecated)
    p_wk = sub.add_parser("worker", help=argparse.SUPPRESS)
    p_wk.set_defaults(func=cmd_worker)

    # Chat
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
    p_chat.set_defaults(func=cmd_chat)

    # Heartbeat
    p_hb = sub.add_parser("heartbeat", help="Trigger heartbeat")
    p_hb.add_argument("person", help="Person name")
    p_hb.add_argument(
        "--local", action="store_true", help="Direct mode (no gateway)"
    )
    p_hb.set_defaults(func=cmd_heartbeat)

    # Send
    p_send = sub.add_parser("send", help="Send message between persons")
    p_send.add_argument("from_person", help="Sender name")
    p_send.add_argument("to_person", help="Recipient name")
    p_send.add_argument("message", help="Message content")
    p_send.add_argument("--thread-id", default=None, help="Thread ID")
    p_send.add_argument("--reply-to", default=None, help="Reply to message ID")
    p_send.set_defaults(func=cmd_send)

    # List
    p_list = sub.add_parser("list", help="List all persons")
    p_list.add_argument(
        "--local", action="store_true", help="Scan filesystem directly"
    )
    p_list.set_defaults(func=cmd_list)

    # Status
    p_status = sub.add_parser("status", help="Show system status from gateway")
    p_status.set_defaults(func=cmd_status)

    # Config management
    from core.config_cli import (
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


if __name__ == "__main__":
    cli_main()
