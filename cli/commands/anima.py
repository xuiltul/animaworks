# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import sys

# ── Create Anima ─────────────────────────────────────────


def cmd_create_anima(args: argparse.Namespace) -> None:
    """Create a new Digital Anima."""
    from pathlib import Path

    from cli.commands.init_cmd import _register_anima_in_config
    from core.anima_factory import (
        create_blank,
        create_from_md,
        create_from_template,
        validate_anima_name,
    )
    from core.init import ensure_runtime_dir
    from core.paths import get_animas_dir, get_data_dir

    ensure_runtime_dir(skip_animas=True)
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    animas_dir.mkdir(parents=True, exist_ok=True)

    supervisor = getattr(args, "supervisor", None)

    if args.from_md:
        md_path = Path(args.from_md).resolve()
        role = getattr(args, "role", None)
        anima_dir = create_from_md(
            animas_dir,
            md_path,
            name=args.name,
            supervisor=supervisor,
            role=role,
        )
        _register_anima_in_config(data_dir, anima_dir.name)
        print(f"Created anima '{anima_dir.name}' from {md_path.name}")
        return

    if args.template:
        anima_dir = create_from_template(animas_dir, args.template, anima_name=args.name)
        _register_anima_in_config(data_dir, anima_dir.name)
        print(f"Created anima '{anima_dir.name}' from template '{args.template}'")
        return

    # Default: blank creation
    name = args.name
    if not name:
        print("Error: --name is required for blank anima creation")
        sys.exit(1)
    err = validate_anima_name(name)
    if err:
        print(f"Error: {err}")
        sys.exit(1)
    anima_dir = create_blank(animas_dir, name)
    _register_anima_in_config(data_dir, anima_dir.name)
    print(f"Created blank anima '{anima_dir.name}'")


# ── Chat ───────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    """Chat with an anima (via gateway or direct)."""
    thread_id = getattr(args, "thread_id", "default") or "default"
    resume = getattr(args, "resume", None)

    if getattr(args, "sessions", False):
        # List saved TUI sessions and exit without opening the TUI.
        from cli.tui.session import list_sessions, sessions_table

        infos = list_sessions()
        if not infos:
            print("no sessions")
            return
        print(sessions_table(infos))
        return

    if args.anima is None and resume is None:
        print("an anima name or --resume is required", file=sys.stderr)
        sys.exit(2)

    if resume and not args.local:
        from cli.tui.session import latest_session, load_session

        sid = None if resume == "latest" else resume
        if (latest_session() if sid is None else load_session(sid)) is None:
            print("no session to resume", file=sys.stderr)
            sys.exit(1)

    if args.local:
        import warnings

        warnings.warn(
            "--local is deprecated and bypasses ProcessSupervisor. "
            "Use the server (animaworks server start) and omit --local.",
            DeprecationWarning,
            stacklevel=2,
        )
        from core.anima import DigitalAnima
        from core.init import ensure_runtime_dir
        from core.paths import get_animas_dir, get_shared_dir

        ensure_runtime_dir()
        anima_dir = get_animas_dir() / args.anima
        if not anima_dir.exists():
            print(f"Anima not found: {args.anima}")
            sys.exit(1)

        anima = DigitalAnima(anima_dir, get_shared_dir())
        message = args.message if args.message is not None else sys.stdin.read()
        response = asyncio.run(anima.process_message(message, from_person=args.from_person))
        print(response)
    else:
        message = args.message
        if message is not None:
            # One-shot reply (existing behaviour).
            from cli._gateway import gateway_request

            data = gateway_request(
                args,
                "POST",
                f"/api/animas/{args.anima}/chat",
                json={
                    "message": message,
                    "from_person": args.from_person,
                    "thread_id": thread_id,
                },
                timeout=300.0,
            )
            print(data.get("response", data.get("error", "Unknown error")))
            return

        # No message given.
        no_tui = getattr(args, "no_tui", False) or False
        if no_tui:
            # Read the whole stdin as the message, one-shot reply.
            from cli._gateway import gateway_request

            stdin_text = sys.stdin.read()
            data = gateway_request(
                args,
                "POST",
                f"/api/animas/{args.anima}/chat",
                json={
                    "message": stdin_text,
                    "from_person": args.from_person,
                    "thread_id": thread_id,
                },
                timeout=300.0,
            )
            print(data.get("response", data.get("error", "Unknown error")))
            return

        if sys.stdin.isatty() and sys.stdout.isatty():
            from cli.tui import run_tui

            run_tui(args)
            return

        print(
            f"TUI requires a terminal for '{args.anima}' (no message given). "
            "Provide a message or use --no-tui to read from stdin.",
            file=sys.stderr,
        )
        sys.exit(2)


# ── Heartbeat ──────────────────────────────────────────────


def cmd_heartbeat(args: argparse.Namespace) -> None:
    """Trigger heartbeat (via gateway or direct)."""
    if args.local:
        import warnings

        warnings.warn(
            "--local is deprecated and bypasses ProcessSupervisor. "
            "Use the server (animaworks server start) and omit --local.",
            DeprecationWarning,
            stacklevel=2,
        )
        from core.anima import DigitalAnima
        from core.init import ensure_runtime_dir
        from core.paths import get_animas_dir, get_shared_dir

        ensure_runtime_dir()
        anima_dir = get_animas_dir() / args.anima
        if not anima_dir.exists():
            print(f"Anima not found: {args.anima}")
            sys.exit(1)

        anima = DigitalAnima(anima_dir, get_shared_dir())
        result = asyncio.run(anima.run_heartbeat())
        print(f"[{result.action}] {result.summary[:500]}")
    else:
        from cli._gateway import gateway_request

        data = gateway_request(
            args,
            "POST",
            f"/api/animas/{args.anima}/trigger",
            timeout=120.0,
        )
        print(data)
