# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse


def run_tui(args: argparse.Namespace) -> None:
    """Launch the interactive terminal chat UI.

    Resolves the gateway URL, builds an ``AnimaWorksClient`` and runs the
    Textual app. Exits with a helpful message when Textual (or one of its
    peer dependencies) is not installed.
    """
    from cli.tui.app import AnimaChatApp
    from cli.tui.client import AnimaWorksClient

    try:
        import textual  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        print(
            "The TUI requires the optional 'tui' extra (textual, websockets).\n"
            'Install it with:  pip install "animaworks[tui]"   (or)   uv sync --extra tui',
            file=__import__("sys").stderr,
        )
        __import__("sys").exit(1)

    from cli._gateway import resolve_gateway_url

    base_url = resolve_gateway_url(args)
    thread_id = getattr(args, "thread_id", "default") or "default"
    from_person = getattr(args, "from_person", "human") or "human"
    resume = getattr(args, "resume", None)

    client = AnimaWorksClient(base_url, from_person=from_person)

    # Resolve the session (resume vs. new).
    from cli.tui.session import (
        SessionInfo,
        latest_session,
        load_session,
        new_session,
    )

    session: SessionInfo | None = None
    if resume:
        sid = None if resume == "latest" else resume
        session = latest_session() if sid is None else load_session(sid)
        if session is None:
            print("no session to resume", file=__import__("sys").stderr)
            __import__("sys").exit(1)
        if getattr(args, "anima", None):
            session.anima = args.anima  # positional anima wins

    if session is None:
        session = new_session(
            anima=args.anima,
            thread_id=thread_id,
            gateway_url=base_url,
            from_person=from_person,
        )

    # Authenticate before starting the TUI (only when required).
    if not _maybe_login(client, args):
        __import__("sys").exit(1)

    app = AnimaChatApp(
        client=client,
        anima_name=session.anima,
        thread_id=session.thread_id,
        session=session,
        no_reattach=getattr(args, "no_reattach", False),
    )
    _install_stack_dump(app)
    app.run()


def _install_stack_dump(app=None) -> None:
    """Diagnostics for a frozen UI, triggered from another terminal.

    - ``kill -USR1 <pid>``: dump all thread stacks (faulthandler, works even
      when the event loop is stuck).
    - ``kill -USR2 <pid>``: dump app internals (scroll position, focus,
      message-queue sizes, asyncio tasks); needs a live event loop.

    Both append to ``~/.animaworks/tui/stackdump.log``.
    """
    import faulthandler
    import signal

    from cli.tui.session import tui_base_dir

    try:
        path = tui_base_dir() / "stackdump.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a")  # noqa: SIM115 - kept open for the process lifetime
        faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True, chain=False)
    except Exception:  # pragma: no cover - best effort diagnostics
        return

    def _dump_app(_signum, _frame) -> None:  # pragma: no cover - manual diagnostics
        import asyncio
        import datetime as _dt

        lines = [f"=== app dump {_dt.datetime.now().isoformat(timespec='seconds')}"]
        try:
            if app is not None:
                tr = getattr(app, "transcript", None)
                if tr is not None:
                    lines.append(
                        f"transcript: scroll_y={tr.scroll_y} max={tr.max_scroll_y} "
                        f"children={len(tr.children)} size={tr.size}"
                    )
                    sb = getattr(tr, "vertical_scrollbar", None)
                    if sb is not None:
                        lines.append(
                            f"transcript scrollbar: region={sb.region} display={sb.display} "
                            f"position={sb.position} window={sb.window_size}/{sb.window_virtual_size}"
                        )
                lines.append(f"focused={app.focused!r} busy={getattr(app, 'busy', None)}")
                mouse = getattr(app, "mouse_position", None)
                lines.append(f"mouse={mouse} over={getattr(app, 'mouse_over', None)!r} captured={app.mouse_captured!r}")
                # Text selection: everything the drag path depends on. A
                # drag that highlights nothing while the mouse is clearly
                # over a selectable widget shows up here as selecting=False
                # or an allow_select that has gone False.
                screen = app.screen
                lines.append(
                    f"selection: n={len(getattr(screen, 'selections', {}) or {})} "
                    f"selecting={getattr(screen, '_selecting', None)} "
                    f"state={'set' if getattr(screen, '_select_state', None) else None} "
                    f"allow: app={getattr(app, 'ALLOW_SELECT', None)} screen={getattr(screen, 'allow_select', None)}"
                )
                lines.append(f"queues: app={app._message_queue.qsize()} screen={app.screen._message_queue.qsize()}")
                lines.append(
                    f"history: loading={getattr(app, '_history_loading', None)} "
                    f"end={getattr(app, '_history_end', None)} "
                    f"cursor={getattr(app, '_history_cursor', None)}"
                )
            try:
                tasks = asyncio.all_tasks()
            except RuntimeError:
                tasks = set()
            lines.append(f"asyncio tasks={len(tasks)}")
            for t in tasks:
                name = t.get_coro().__qualname__
                if name.startswith("MessagePump"):
                    continue
                fr = t.get_stack(limit=1)
                where = f"{fr[0].f_code.co_filename.rsplit('/', 1)[-1]}:{fr[0].f_lineno}" if fr else "?"
                lines.append(f"  task {name} @ {where}")
        except Exception as exc:
            lines.append(f"dump error: {exc!r}")
        fh.write("\n".join(lines) + "\n")
        fh.flush()

    try:
        signal.signal(signal.SIGUSR2, _dump_app)
    except Exception:  # pragma: no cover
        pass


def _maybe_login(client, args) -> bool:
    """Authenticate the client if the gateway requires it.

    Returns True when we are ready to proceed (already authenticated,
    login succeeded, or auth is disabled).
    """
    import getpass as _getpass
    import os
    import sys

    try:
        me = client.auth_me()
    except Exception:
        # Connection issues surface later in the TUI; proceed unauthenticated.
        return True
    if me is not None:
        return True  # already authenticated (or auth disabled)

    username = getattr(args, "user", None)
    if not username:
        try:
            username = input("Username: ")
        except EOFError:
            print("Login required but no username given", file=sys.stderr)
            return False
    password = getattr(args, "password", None) or os.environ.get("ANIMAWORKS_TUI_PASSWORD")
    if password is None:
        try:
            password = _getpass.getpass("Password: ")
        except EOFError:
            print("Login required but no password given", file=sys.stderr)
            return False
    try:
        client.login(username, password)
    except Exception as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return False
    return True
