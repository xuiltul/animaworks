# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Handler signature: called with ``(args, app)`` where ``app`` is the
# running ``AnimaChatApp`` instance (duck-typed, so no import needed).
CommandHandler = Callable[[list[str], Any], None]


@dataclass
class SlashCommand:
    """A single slash command registered for the chat input.

    ``name`` is the command without the leading ``/`` (e.g. ``"help"``).
    ``handler`` receives the list of arguments and the running app.
    ``takes_args`` marks commands that need further input after being
    picked from the palette (the palette inserts the text instead of
    running instantly). ``usage`` is shown in ``/help``.

    Commands can be added later by appending to :func:`register`.
    """

    name: str
    description: str
    handler: CommandHandler
    takes_args: bool = False
    usage: str = ""

    @property
    def signature(self) -> str:
        return f"/{self.name}"


def _cmd_help(_args: list[str], app: Any) -> None:
    lines = ["Available commands:"]
    for cmd in list_commands():
        usage = f" {cmd.usage}" if cmd.usage else ""
        lines.append(f"  {cmd.signature}{usage:<10} {cmd.description}")
    lines.append("  /<skill> [text]  Activate a skill for this thread, then send text (if any)")
    lines.append("  Esc            interrupt the current response")
    lines.append("  Ctrl+B         toggle the sidebar")
    lines.append("  Ctrl+C (x2)    quit")
    app.show_transient("\n".join(lines))


def _cmd_quit(_args: list[str], app: Any) -> None:
    app.exit()


def _cmd_clear(_args: list[str], app: Any) -> None:
    app.new_thread()


def _cmd_thread(args: list[str], app: Any) -> None:
    if args:
        app.switch_thread(args[0])
    else:
        app.new_thread()


def _cmd_threads(_args: list[str], app: Any) -> None:
    app.show_threads()


def _cmd_thinking(_args: list[str], app: Any) -> None:
    app.toggle_thinking()


def _cmd_model(args: list[str], app: Any) -> None:
    app.choose_model(args)


def _cmd_interrupt(_args: list[str], app: Any) -> None:
    app.request_interrupt()


def _cmd_history(_args: list[str], app: Any) -> None:
    # Phase 3: load 50 more (older) messages instead of reloading everything.
    app.load_more_history()


# ── Phase 2 commands ─────────────────────────────────────


def _cmd_anima(args: list[str], app: Any) -> None:
    name = args[0] if args else ""
    if not name:
        app.focus_sidebar()
        return
    app.switch_anima(name)


def _cmd_animas(_args: list[str], app: Any) -> None:
    app.show_animas()


def _cmd_skills(_args: list[str], app: Any) -> None:
    app.show_skills()


def _cmd_skill(args: list[str], app: Any) -> None:
    app.activate_skill(args)


def _cmd_board(args: list[str], app: Any) -> None:
    app.show_board(args)


def _cmd_post(args: list[str], app: Any) -> None:
    app.post_to_channel(args)


def _cmd_tasks(args: list[str], app: Any) -> None:
    app.show_tasks(args)


def _cmd_sidebar(_args: list[str], app: Any) -> None:
    app.toggle_sidebar()


def _cmd_approve(args: list[str], app: Any) -> None:
    app.resolve_interaction(args)


def _cmd_sessions(_args: list[str], app: Any) -> None:
    app.show_sessions()


def _cmd_keys(_args: list[str], app: Any) -> None:
    app.show_keys()


def _cmd_compact(_args: list[str], app: Any) -> None:
    app.compact_session()


def _cmd_reject(args: list[str], app: Any) -> None:
    # /reject <callback_id> → approve with "reject" option if it exists
    if args:
        app.resolve_interaction([args[0], "reject"])


_registry: dict[str, SlashCommand] = {}
_registered = False


def register_command(cmd: SlashCommand) -> None:
    """Register a single slash command (overwrites any same-name entry)."""
    _registry[cmd.name] = cmd


def register_default_commands() -> None:
    """Register the built-in slash commands (idempotent)."""
    global _registered
    if _registered:
        return
    for cmd in _default_commands():
        _registry[cmd.name] = cmd
    _registered = True


def _default_commands() -> list[SlashCommand]:
    return [
        SlashCommand("help", "Show available commands", _cmd_help),
        SlashCommand("quit", "Quit the TUI", _cmd_quit),
        SlashCommand("clear", "Start a new thread (same as /thread)", _cmd_clear),
        SlashCommand(
            "thread",
            "Start a new thread, or switch to <id>",
            _cmd_thread,
            takes_args=True,
            usage="[id]",
        ),
        SlashCommand("threads", "List this anima's threads", _cmd_threads),
        SlashCommand("thinking", "Toggle thinking display", _cmd_thinking),
        SlashCommand(
            "model",
            "Choose the model for this thread (no args: pick from a list)",
            _cmd_model,
            takes_args=True,
            usage="[id|default]",
        ),
        SlashCommand("interrupt", "Stop the current response", _cmd_interrupt),
        SlashCommand("history", "Reload conversation history", _cmd_history, takes_args=True, usage="[n]"),
        SlashCommand("anima", "Switch to another anima", _cmd_anima, takes_args=True, usage="<name>"),
        SlashCommand("animas", "List all animas", _cmd_animas),
        SlashCommand("skills", "List the current anima's skills", _cmd_skills),
        SlashCommand(
            "skill",
            "Activate/deactivate a skill for this thread",
            _cmd_skill,
            takes_args=True,
            usage="<name> [--confirm] [--off]",
        ),
        SlashCommand("board", "List channels or read a channel", _cmd_board, takes_args=True, usage="[channel] [n]"),
        SlashCommand("post", "Post to a channel", _cmd_post, takes_args=True, usage="<channel> <text>"),
        SlashCommand(
            "tasks",
            "Show the task board for an anima",
            _cmd_tasks,
            takes_args=True,
            usage="[anima]",
        ),
        SlashCommand("sidebar", "Toggle the sidebar", _cmd_sidebar),
        SlashCommand("sessions", "List saved TUI sessions", _cmd_sessions),
        SlashCommand("keys", "Show current key bindings", _cmd_keys),
        SlashCommand(
            "compact",
            "Compact this thread's context (save a summary, start a fresh session)",
            _cmd_compact,
        ),
        SlashCommand(
            "approve",
            "Resolve a call_human card from the keyboard",
            _cmd_approve,
            takes_args=True,
            usage="<callback_id> [option]",
        ),
        SlashCommand("reject", "Reject a call_human card", _cmd_reject, takes_args=True, usage="<callback_id>"),
    ]


def get_command(name: str) -> SlashCommand | None:
    register_default_commands()
    return _registry.get(name)


def iter_commands() -> list[SlashCommand]:
    """Return the full (ordered) command registry.

    This is the single source of truth used by both ``/help`` and the
    slash-command palette.
    """
    register_default_commands()
    return list(_registry.values())


def list_commands() -> list[SlashCommand]:
    """Backwards-compatible alias for :func:`iter_commands`."""
    return iter_commands()


def is_command(text: str) -> bool:
    """Return True if the trimmed input looks like a slash command."""
    return text.startswith("/")
