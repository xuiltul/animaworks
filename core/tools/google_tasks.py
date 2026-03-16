# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks Google Tasks tool -- Google Tasks API access.

Provides task list and task listing, task/tasklist creation and update via Google Tasks API.
Uses the same OAuth2 credential pattern as Gmail and Google Calendar.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "list_tasklists": {"expected_seconds": 10, "background_eligible": False},
    "list_tasks": {"expected_seconds": 10, "background_eligible": False},
    "insert_task": {"expected_seconds": 10, "background_eligible": False},
    "insert_tasklist": {"expected_seconds": 10, "background_eligible": False},
    "update_task": {"expected_seconds": 10, "background_eligible": False},
    "update_tasklist": {"expected_seconds": 10, "background_eligible": False},
}

TOOL_DESCRIPTION = "Google Tasks task lists and tasks (list, create, update)"

SCOPES = ["https://www.googleapis.com/auth/tasks"]

_DEFAULT_CREDENTIALS_DIR = Path.home() / ".animaworks" / "credentials" / "google_tasks"


# ── Client ────────────────────────────────────────────────


class GoogleTasksClient:
    """Google Tasks API client with OAuth2 authentication."""

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.credentials_path = credentials_path or (_DEFAULT_CREDENTIALS_DIR / "credentials.json")
        self.token_path = token_path or (_DEFAULT_CREDENTIALS_DIR / "token.json")
        self.client_id = client_id or os.environ.get("GOOGLE_TASKS_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("GOOGLE_TASKS_CLIENT_SECRET")
        self._service = None

    def _get_credentials(self) -> Any:
        """Obtain valid credentials via OAuth2."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            raise ImportError(
                "google_tasks tool requires google-api packages. "
                "Install with: pip install animaworks[gmail] or google-api-python-client google-auth-oauthlib"
            ) from None

        creds = None

        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                self.token_path.write_text(creds.to_json())
            else:
                if self.credentials_path.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path),
                        SCOPES,
                    )
                elif self.client_id and self.client_secret:
                    client_config = {
                        "installed": {
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": ["http://localhost"],
                        }
                    }
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                else:
                    raise FileNotFoundError(
                        f"No credentials found. Place credentials.json at "
                        f"{self.credentials_path} or set GOOGLE_TASKS_CLIENT_ID "
                        f"and GOOGLE_TASKS_CLIENT_SECRET environment variables."
                    )
                creds = flow.run_local_server(port=0)
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                self.token_path.write_text(creds.to_json())

        return creds

    def _build_service(self) -> Any:
        """Build the Tasks API service."""
        if self._service is None:
            from googleapiclient.discovery import build

            creds = self._get_credentials()
            self._service = build("tasks", "v1", credentials=creds)
        return self._service

    def list_tasklists(self, *, max_results: int = 50) -> list[dict[str, Any]]:
        """List the user's task lists."""
        service = self._build_service()
        result = service.tasklists().list(maxResults=max_results).execute()
        items = result.get("items", [])
        return [{"id": i.get("id", ""), "title": i.get("title", ""), "updated": i.get("updated", "")} for i in items]

    def list_tasks(
        self,
        *,
        tasklist_id: str,
        max_results: int = 50,
        show_completed: bool = True,
    ) -> list[dict[str, Any]]:
        """List tasks in a task list."""
        service = self._build_service()
        result = (
            service.tasks()
            .list(
                tasklist=tasklist_id,
                maxResults=max_results,
                showCompleted=show_completed,
            )
            .execute()
        )
        items = result.get("items", [])
        return [
            {
                "id": i.get("id", ""),
                "title": i.get("title", ""),
                "status": i.get("status", ""),
                "due": i.get("due", ""),
                "updated": i.get("updated", ""),
                "notes": (i.get("notes") or "")[:200],
            }
            for i in items
        ]

    def insert_task(
        self,
        *,
        tasklist_id: str,
        title: str,
        notes: str = "",
        due: str | None = None,
    ) -> dict[str, Any]:
        """Create a task in a task list."""
        service = self._build_service()
        body: dict[str, Any] = {"title": title}
        if notes:
            body["notes"] = notes
        if due:
            body["due"] = due
        created = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return {
            "id": created.get("id", ""),
            "title": created.get("title", ""),
            "status": created.get("status", ""),
            "due": created.get("due", ""),
        }

    def insert_tasklist(self, *, title: str) -> dict[str, Any]:
        """Create a new task list."""
        service = self._build_service()
        created = service.tasklists().insert(body={"title": title}).execute()
        return {
            "id": created.get("id", ""),
            "title": created.get("title", ""),
            "updated": created.get("updated", ""),
        }

    def update_task(
        self,
        *,
        tasklist_id: str,
        task_id: str,
        title: str | None = None,
        notes: str | None = None,
        due: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Update a task (patch: only provided fields are updated)."""
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if notes is not None:
            body["notes"] = notes
        if due is not None:
            body["due"] = due
        if status is not None:
            if status not in ("needsAction", "completed"):
                raise ValueError("status must be 'needsAction' or 'completed'")
            body["status"] = status
        if not body:
            return {"error": "At least one of title, notes, due, status is required"}
        service = self._build_service()
        updated = service.tasks().patch(tasklist=tasklist_id, task=task_id, body=body).execute()
        return {
            "id": updated.get("id", ""),
            "title": updated.get("title", ""),
            "status": updated.get("status", ""),
            "due": updated.get("due", ""),
            "notes": (updated.get("notes") or "")[:200],
        }

    def update_tasklist(self, *, tasklist_id: str, title: str) -> dict[str, Any]:
        """Update a task list's title."""
        service = self._build_service()
        updated = service.tasklists().patch(tasklist=tasklist_id, body={"title": title}).execute()
        return {
            "id": updated.get("id", ""),
            "title": updated.get("title", ""),
            "updated": updated.get("updated", ""),
        }


# ── Tool schemas ──────────────────────────────────────────


def get_tool_schemas() -> list[dict]:
    """Return tool schemas (empty — use skill-based documentation)."""
    return []


# ── Dispatch ──────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    _args = {k: v for k, v in args.items() if k != "anima_dir"}
    client = GoogleTasksClient()

    if name == "google_tasks_list_tasklists":
        return client.list_tasklists(max_results=int(_args.get("max_results", 50)))

    if name == "google_tasks_list_tasks":
        tasklist_id = _args.get("tasklist_id", "")
        if not tasklist_id:
            return {"error": "tasklist_id is required"}
        return client.list_tasks(
            tasklist_id=tasklist_id,
            max_results=int(_args.get("max_results", 50)),
            show_completed=_args.get("show_completed", True),
        )

    if name == "google_tasks_insert_task":
        tasklist_id = _args.get("tasklist_id", "")
        title = _args.get("title", "")
        if not tasklist_id or not title:
            return {"error": "tasklist_id and title are required"}
        return client.insert_task(
            tasklist_id=tasklist_id,
            title=title,
            notes=_args.get("notes", ""),
            due=_args.get("due") or None,
        )

    if name == "google_tasks_insert_tasklist":
        title = _args.get("title", "")
        if not title:
            return {"error": "title is required"}
        return client.insert_tasklist(title=title)

    if name == "google_tasks_update_task":
        tasklist_id = _args.get("tasklist_id", "")
        task_id = _args.get("task_id", "")
        if not tasklist_id or not task_id:
            return {"error": "tasklist_id and task_id are required"}
        try:
            return client.update_task(
                tasklist_id=tasklist_id,
                task_id=task_id,
                title=_args.get("title") or None,
                notes=_args.get("notes") if "notes" in _args else None,
                due=_args.get("due") if "due" in _args else None,
                status=_args.get("status") if "status" in _args else None,
            )
        except ValueError as e:
            return {"error": str(e)}

    if name == "google_tasks_update_tasklist":
        tasklist_id = _args.get("tasklist_id", "")
        title = _args.get("title", "")
        if not tasklist_id or not title:
            return {"error": "tasklist_id and title are required"}
        return client.update_tasklist(tasklist_id=tasklist_id, title=title)

    return {"error": f"Unknown action: {name}"}


# ── CLI ───────────────────────────────────────────────────


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Google Tasks tool."""
    parser = argparse.ArgumentParser(
        prog="animaworks-tool google_tasks",
        description="Google Tasks operations",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_lists = subparsers.add_parser("tasklists", help="List task lists")
    p_lists.add_argument("-n", "--max-results", type=int, default=50, help="Max results")
    p_lists.add_argument("-j", "--json", action="store_true", help="JSON output")

    p_tasks = subparsers.add_parser("list", help="List tasks in a task list")
    p_tasks.add_argument("tasklist_id", help="Task list ID")
    p_tasks.add_argument("-n", "--max-results", type=int, default=50, help="Max results")
    p_tasks.add_argument("--no-completed", action="store_true", help="Hide completed tasks")
    p_tasks.add_argument("-j", "--json", action="store_true", help="JSON output")

    p_add = subparsers.add_parser("add", help="Add a task to a task list")
    p_add.add_argument("tasklist_id", help="Task list ID")
    p_add.add_argument("title", help="Task title")
    p_add.add_argument("--notes", default="", help="Task notes")
    p_add.add_argument("--due", default="", help="Due date (RFC 3339)")
    p_add.add_argument("-j", "--json", action="store_true", help="JSON output")

    p_newlist = subparsers.add_parser("new-list", help="Create a new task list")
    p_newlist.add_argument("title", help="Task list title")
    p_newlist.add_argument("-j", "--json", action="store_true", help="JSON output")

    p_update = subparsers.add_parser("update", help="Update a task")
    p_update.add_argument("tasklist_id", help="Task list ID")
    p_update.add_argument("task_id", help="Task ID")
    p_update.add_argument("--title", default="", help="New task title")
    p_update.add_argument("--notes", default="", help="New notes")
    p_update.add_argument("--due", default="", help="Due date (RFC 3339)")
    p_update.add_argument("--status", choices=("needsAction", "completed"), help="Status")
    p_update.add_argument("-j", "--json", action="store_true", help="JSON output")

    p_updatelist = subparsers.add_parser("update-list", help="Update a task list title")
    p_updatelist.add_argument("tasklist_id", help="Task list ID")
    p_updatelist.add_argument("title", help="New list title")
    p_updatelist.add_argument("-j", "--json", action="store_true", help="JSON output")

    args = parser.parse_args(argv)
    client = GoogleTasksClient()

    try:
        if args.command == "tasklists":
            out = client.list_tasklists(max_results=args.max_results)
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                for i in out:
                    print(f"  {i.get('id', '')}  {i.get('title', '')}")

        elif args.command == "list":
            out = client.list_tasks(
                tasklist_id=args.tasklist_id,
                max_results=args.max_results,
                show_completed=not getattr(args, "no_completed", False),
            )
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                for i in out:
                    print(f"  [{i.get('status', '')}] {i.get('title', '')}  {i.get('due', '')}")

        elif args.command == "add":
            out = client.insert_task(
                tasklist_id=args.tasklist_id,
                title=args.title,
                notes=getattr(args, "notes", "") or "",
                due=args.due or None,
            )
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"Created: {out.get('title', '')} (id={out.get('id', '')})")

        elif args.command == "new-list":
            out = client.insert_tasklist(title=args.title)
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"Created list: {out.get('title', '')} (id={out.get('id', '')})")

        elif args.command == "update":
            body = {}
            if getattr(args, "title", ""):
                body["title"] = args.title
            if getattr(args, "notes", ""):
                body["notes"] = args.notes
            if getattr(args, "due", ""):
                body["due"] = args.due
            if getattr(args, "status", ""):
                body["status"] = args.status
            if not body:
                print(
                    "Error: provide at least one of --title, --notes, --due, --status",
                    file=sys.stderr,
                )
                sys.exit(1)
            out = client.update_task(
                tasklist_id=args.tasklist_id,
                task_id=args.task_id,
                title=body.get("title"),
                notes=body.get("notes"),
                due=body.get("due"),
                status=body.get("status"),
            )
            if "error" in out:
                print(f"Error: {out['error']}", file=sys.stderr)
                sys.exit(1)
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"Updated: {out.get('title', '')} (id={out.get('id', '')})")

        elif args.command == "update-list":
            out = client.update_tasklist(tasklist_id=args.tasklist_id, title=args.title)
            if getattr(args, "json", False):
                print(json.dumps(out, ensure_ascii=False, indent=2))
            else:
                print(f"Updated list: {out.get('title', '')} (id={out.get('id', '')})")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
