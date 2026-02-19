from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI subcommand for task queue management (A1 mode).

Usage via animaworks-tool:
    animaworks-tool task add --source human --instruction "..." --assignee rin
    animaworks-tool task update --task-id abc123 --status in_progress
    animaworks-tool task list [--status pending]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_task(args: argparse.Namespace) -> None:
    """Dispatch task subcommand."""
    anima_dir_str = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    if not anima_dir_str:
        print("Error: ANIMAWORKS_ANIMA_DIR not set", file=sys.stderr)
        sys.exit(1)

    anima_dir = Path(anima_dir_str)
    if not anima_dir.is_dir():
        print(f"Error: anima_dir not found: {anima_dir}", file=sys.stderr)
        sys.exit(1)

    from core.memory.task_queue import TaskQueueManager

    manager = TaskQueueManager(anima_dir)

    sub = getattr(args, "task_command", None)
    if sub == "add":
        _cmd_add(args, manager)
    elif sub == "update":
        _cmd_update(args, manager)
    elif sub == "list":
        _cmd_list(args, manager)
    else:
        print("Usage: animaworks-tool task {add|update|list}", file=sys.stderr)
        sys.exit(1)


def _cmd_add(args: argparse.Namespace, manager) -> None:
    source = getattr(args, "source", "anima")
    instruction = getattr(args, "instruction", "")
    assignee = getattr(args, "assignee", "")
    summary = getattr(args, "summary", "") or instruction[:100]
    deadline = getattr(args, "deadline", None)
    relay_chain_raw = getattr(args, "relay_chain", None)
    relay_chain = relay_chain_raw.split(",") if relay_chain_raw else []

    if not instruction:
        print("Error: --instruction is required", file=sys.stderr)
        sys.exit(1)
    if not assignee:
        print("Error: --assignee is required", file=sys.stderr)
        sys.exit(1)

    entry = manager.add_task(
        source=source,
        original_instruction=instruction,
        assignee=assignee,
        summary=summary,
        deadline=deadline,
        relay_chain=relay_chain,
    )
    result = entry.model_dump()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_update(args: argparse.Namespace, manager) -> None:
    task_id = getattr(args, "task_id", "")
    status = getattr(args, "status", "")
    summary = getattr(args, "summary", None)

    if not task_id:
        print("Error: --task-id is required", file=sys.stderr)
        sys.exit(1)
    if not status:
        print("Error: --status is required", file=sys.stderr)
        sys.exit(1)

    entry = manager.update_status(task_id, status, summary=summary)
    if entry is None:
        print(f"Error: task not found or invalid status: {task_id}", file=sys.stderr)
        sys.exit(1)

    result = entry.model_dump()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_list(args: argparse.Namespace, manager) -> None:
    status_filter = getattr(args, "status", None)
    tasks = manager.list_tasks(status=status_filter)
    result = [t.model_dump() for t in tasks]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def register_task_command(subparsers) -> None:
    """Register the task subcommand under animaworks-tool."""
    p_task = subparsers.add_parser("task", help="Manage persistent task queue")
    task_sub = p_task.add_subparsers(dest="task_command")

    # task add
    p_add = task_sub.add_parser("add", help="Add a new task")
    p_add.add_argument("--source", default="anima", choices=["human", "anima"])
    p_add.add_argument("--instruction", required=True, help="Original instruction text")
    p_add.add_argument("--assignee", required=True, help="Assignee anima name")
    p_add.add_argument("--summary", default=None, help="1-line summary (default: instruction[:100])")
    p_add.add_argument("--deadline", default=None, help="ISO8601 deadline")
    p_add.add_argument("--relay-chain", default=None, help="Comma-separated relay chain")

    # task update
    p_update = task_sub.add_parser("update", help="Update task status")
    p_update.add_argument("--task-id", required=True, help="Task ID")
    p_update.add_argument("--status", required=True, choices=["pending", "in_progress", "done", "cancelled", "blocked"])
    p_update.add_argument("--summary", default=None, help="Updated summary")

    # task list
    p_list = task_sub.add_parser("list", help="List tasks")
    p_list.add_argument("--status", default=None, choices=["pending", "in_progress", "done", "cancelled", "blocked"])

    p_task.set_defaults(func=cmd_task)
