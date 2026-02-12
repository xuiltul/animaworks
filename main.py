from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PERSONS_DIR = BASE_DIR / "persons"
SHARED_DIR = BASE_DIR / "shared"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("animaworks")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the daemon (FastAPI + APScheduler)."""
    import uvicorn

    from server.app import create_app

    app = create_app(PERSONS_DIR, SHARED_DIR)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_chat(args: argparse.Namespace) -> None:
    """One-shot chat with a person from CLI."""
    from core.person import DigitalPerson

    person_dir = PERSONS_DIR / args.person
    if not person_dir.exists():
        print(f"Person not found: {args.person}")
        sys.exit(1)

    person = DigitalPerson(person_dir, SHARED_DIR)
    response = asyncio.run(person.process_message(args.message))
    print(response)


def cmd_heartbeat(args: argparse.Namespace) -> None:
    """Manually trigger heartbeat."""
    from core.person import DigitalPerson

    person_dir = PERSONS_DIR / args.person
    if not person_dir.exists():
        print(f"Person not found: {args.person}")
        sys.exit(1)

    person = DigitalPerson(person_dir, SHARED_DIR)
    result = asyncio.run(person.run_heartbeat())
    print(f"[{result.action}] {result.summary[:500]}")


def cmd_send(args: argparse.Namespace) -> None:
    """Send a message from one person to another."""
    from core.messenger import Messenger

    messenger = Messenger(SHARED_DIR, args.from_person)
    msg = messenger.send(
        to=args.to_person,
        content=args.message,
        thread_id=args.thread_id or "",
        reply_to=args.reply_to or "",
    )
    print(f"Sent: {msg.from_person} -> {msg.to_person} (id: {msg.id}, thread: {msg.thread_id})")


def cmd_list(args: argparse.Namespace) -> None:
    """List all persons."""
    if not PERSONS_DIR.exists():
        print("No persons directory found.")
        return
    for d in sorted(PERSONS_DIR.iterdir()):
        if d.is_dir() and (d / "identity.md").exists():
            print(f"  {d.name}")


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="AnimaWorks - Digital Person Framework"
    )
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start daemon (web + scheduler)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=18500)
    p_serve.set_defaults(func=cmd_serve)

    p_chat = sub.add_parser("chat", help="Chat with a person")
    p_chat.add_argument("person", help="Person name")
    p_chat.add_argument("message", help="Message to send")
    p_chat.set_defaults(func=cmd_chat)

    p_hb = sub.add_parser("heartbeat", help="Trigger heartbeat")
    p_hb.add_argument("person", help="Person name")
    p_hb.set_defaults(func=cmd_heartbeat)

    p_send = sub.add_parser("send", help="Send message between persons")
    p_send.add_argument("from_person", help="Sender name")
    p_send.add_argument("to_person", help="Recipient name")
    p_send.add_argument("message", help="Message content")
    p_send.add_argument("--thread-id", default=None, help="Thread ID")
    p_send.add_argument("--reply-to", default=None, help="Reply to message ID")
    p_send.set_defaults(func=cmd_send)

    p_list = sub.add_parser("list", help="List all persons")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
