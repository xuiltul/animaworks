# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks Notion tool — Notion API client and tool interface.

Provides CRUD operations on Notion databases and pages via the
Notion API v2022-06-28.  Supports simple mode (task_name + summary)
for the default development task DB, and detailed mode (raw
properties + arbitrary database_id) for full customization.

Requires a Notion Integration token, resolved via:
  config.json → shared/credentials.json (key: ``notion_token``)
  → env ``$NOTION_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from core.tools._base import get_credential, logger
from core.tools._retry import retry_on_rate_limit

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "create": {"expected_seconds": 10, "background_eligible": False},
    "query":  {"expected_seconds": 15, "background_eligible": False},
    "update": {"expected_seconds": 10, "background_eligible": False},
    "batch":  {"expected_seconds": 120, "background_eligible": True},
}

# ── Constants ─────────────────────────────────────────────

DEFAULT_DATABASE_ID = "3c7c44cc2f4a4b168ebbafe0c78bb43b"

BATCH_INTERVAL = 0.35  # 350ms ≈ 2.8 req/s (safety margin for 3 req/s limit)

MAX_PAYLOAD_BYTES = 500_000  # 500 KB per Notion API spec

# Schema for the FB (feedback) child database created per task page
FB_DB_SCHEMA: dict[str, Any] = {
    "FBのID": {"title": {}},
    "問題の概要": {"rich_text": {}},
    "再現方法": {"rich_text": {}},
    "期待する動作": {"rich_text": {}},
    "タグ": {"multi_select": {}},
    "ステータス": {
        "select": {
            "options": [
                {"name": "未対応", "color": "red"},
                {"name": "対応中", "color": "yellow"},
                {"name": "解消済み", "color": "green"},
            ],
        },
    },
}


def build_page_url(page_id: str) -> str:
    """Convert a Notion page ID to a valid browser URL.

    Notion requires hyphens stripped and www. prefix.
    """
    return "https://www.notion.so/" + page_id.replace("-", "")

# Category → status mapping (task_status_flow.md §4.1)
STATUS_CATEGORIES: dict[str, list[str]] = {
    "未着手": ["新規", "営業要望", "やりたいこと"],
    "検討中": ["次回開発MTG確認", "仕様策定中", "仕様確認中", "デザイン作成中", "デザイン確認中"],
    "作業準備": ["着手予定"],
    "作業中": ["実装中"],
    "テスト": ["機能テスト中"],
    "リリース準備": ["SEEDマージ", "マニュアル確認中", "リリース待ち"],
    "完了": ["デプロイ済", "済"],
    "保留": ["pending"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────────────────────────────────────


class NotionAPIError(Exception):
    """Base class for Notion API errors."""


class RateLimitError(NotionAPIError):
    """429 Too Many Requests."""

    def __init__(self, retry_after: float, response: Any) -> None:
        self.retry_after = retry_after
        self.response = response
        super().__init__(f"Rate limited, retry after {retry_after}s")


class ServerError(NotionAPIError):
    """5xx server error."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"Server error {status_code}: {body[:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# Retry helper
# ──────────────────────────────────────────────────────────────────────────────


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from a RateLimitError."""
    if isinstance(exc, RateLimitError):
        return exc.retry_after
    return None


# ──────────────────────────────────────────────────────────────────────────────
# NotionClient
# ──────────────────────────────────────────────────────────────────────────────


class NotionClient:
    """Notion API v2022-06-28 client."""

    BASE_URL = "https://api.notion.com/v1"
    API_VERSION = "2022-06-28"

    def __init__(self, token: str | None = None) -> None:
        """Initialise the client.

        Args:
            token: Notion Integration token.  If *None*, resolved via
                ``get_credential`` cascade or shared/credentials.json
                nested key ``notion.integration_token``.
        """
        import httpx

        if not token:
            try:
                token = get_credential(
                    "notion", "integration_token", env_var="NOTION_TOKEN",
                )
            except Exception:
                from core.tools._base import _lookup_shared_credentials

                val = _lookup_shared_credentials("notion")
                if isinstance(val, dict) and val.get("integration_token"):
                    token = val["integration_token"]
                else:
                    raise
        self.token = token
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a page in the specified database.

        Args:
            database_id: Target database ID.
            properties: Notion API property format dict.

        Returns:
            Created page object.
        """
        body = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }
        self._validate_payload(body)
        return self._request("POST", "/pages", body)

    def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update page properties.

        Args:
            page_id: Target page ID.
            properties: Properties to update (partial update).

        Returns:
            Updated page object.
        """
        body = {"properties": properties}
        return self._request("PATCH", f"/pages/{page_id}", body)

    def query_database(
        self,
        database_id: str,
        filter: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None,
        page_size: int = 100,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Query a database.

        Args:
            database_id: Target database ID.
            filter: Notion filter object.
            sorts: Sort conditions list.
            page_size: Results per page (max 100).
            start_cursor: Pagination cursor.

        Returns:
            Dict with ``results``, ``has_more``, and ``next_cursor``.
        """
        body: dict[str, Any] = {"page_size": min(page_size, 100)}
        if filter:
            body["filter"] = filter
        if sorts:
            body["sorts"] = sorts
        if start_cursor:
            body["start_cursor"] = start_cursor
        return self._request("POST", f"/databases/{database_id}/query", body)

    def get_page(self, page_id: str) -> dict[str, Any]:
        """Retrieve a single page by ID.

        Args:
            page_id: Target page ID.

        Returns:
            Page object.
        """
        return self._request("GET", f"/pages/{page_id}")

    # ── Database operations ────────────────────────────────────────────────

    def create_database(
        self,
        parent_page_id: str,
        title: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a child database inside a Notion page.

        Args:
            parent_page_id: Parent page ID.
            title: Database title.
            properties: Property schema dict.

        Returns:
            Created database object.
        """
        body = {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "title": [{"type": "text", "text": {"content": title}}],
            "properties": properties,
        }
        return self._request("POST", "/databases", body)

    def find_child_database(
        self,
        parent_page_id: str,
        title: str,
    ) -> dict[str, Any] | None:
        """Check if a child database with given title already exists.

        Args:
            parent_page_id: Parent page ID.
            title: Database title to search for.

        Returns:
            Block object if found, else None.
        """
        result = self._request("GET", f"/blocks/{parent_page_id}/children")
        for block in result.get("results", []):
            if block.get("type") == "child_database":
                db_title = block.get("child_database", {}).get("title", "")
                if db_title == title:
                    return block
        return None

    # ── Batch ─────────────────────────────────────────────────────────────

    def batch_create_pages(
        self,
        database_id: str,
        items: list[dict[str, Any]],
        batch_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Create multiple pages sequentially with rate-limit throttling.

        Args:
            database_id: Target database ID.
            items: List of property dicts.
            batch_size: Progress log interval.

        Returns:
            List of result dicts (``success``/``error`` mixed).
        """
        results: list[dict[str, Any]] = []
        for i, props in enumerate(items):
            if i > 0:
                time.sleep(BATCH_INTERVAL)
            try:
                page = self.create_page(database_id, props)
                results.append({"success": True, "page": page})
            except NotionAPIError as e:
                logger.warning("Batch item %d failed: %s", i, e)
                results.append({"success": False, "error": str(e)})
            if (i + 1) % batch_size == 0:
                logger.info("Batch progress: %d/%d", i + 1, len(items))
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an API request with retry on rate-limit/server errors."""
        import httpx

        def _do_request() -> dict[str, Any]:
            response = self._client.request(method, path, json=body)
            if response.status_code == 429:
                retry_after = float(
                    response.headers.get("Retry-After", "1"),
                )
                raise RateLimitError(retry_after, response)
            if response.status_code >= 500:
                raise ServerError(response.status_code, response.text)
            response.raise_for_status()
            return response.json()

        return retry_on_rate_limit(
            _do_request,
            max_retries=5,
            default_wait=30.0,
            get_retry_after=_extract_retry_after,
            retry_on=(RateLimitError, ServerError, httpx.ConnectError),
        )

    def _validate_payload(self, body: dict[str, Any]) -> None:
        """Pre-check payload size against Notion's 500KB limit."""
        size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        if size > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"Payload size {size} bytes exceeds Notion limit "
                f"({MAX_PAYLOAD_BYTES} bytes)"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Property helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_default_properties(
    task_name: str,
    summary: str,
    status: str = "新規",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Convert simple-mode params to Notion API properties.

    Maps to the 4 properties confirmed in notion_schema_mapping.md:
      - Name (title) ← task_name
      - Status (select) ← status
      - Label (multi_select) ← labels
      - AIによる要約 (rich_text) ← summary (truncated to 2000 chars)
    """
    if labels is None:
        labels = ["AI"]
    return {
        "Name": {"title": [{"text": {"content": task_name}}]},
        "Status": {"select": {"name": status}},
        "Label": {"multi_select": [{"name": l} for l in labels]},
        "AIによる要約": {
            "rich_text": [{"text": {"content": summary[:2000]}}],
        },
    }


def _build_category_filter(category: str) -> dict[str, Any]:
    """Convert a category name to a Notion filter object.

    Based on task_status_flow.md §4.1 category definitions.

    Raises:
        ValueError: If the category name is not recognised.
    """
    statuses = STATUS_CATEGORIES.get(category)
    if not statuses:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Valid: {', '.join(STATUS_CATEGORIES.keys())}"
        )
    if len(statuses) == 1:
        return {"property": "Status", "select": {"equals": statuses[0]}}
    return {
        "or": [
            {"property": "Status", "select": {"equals": s}}
            for s in statuses
        ],
    }


def _resolve_database_id(database_id: str | None = None) -> str:
    """Resolve database ID, falling back to credentials.json then constant."""
    if database_id:
        return database_id
    from core.tools._base import _lookup_shared_credentials

    db_id = _lookup_shared_credentials("notion_database_id")
    if db_id:
        return db_id
    return DEFAULT_DATABASE_ID


def find_task_page_id(
    client: NotionClient,
    task_number: str,
    database_id: str | None = None,
) -> str | None:
    """Find a task page ID by task number (e.g. 'TASK-3296').

    Queries the default development task DB and searches the Name
    (title) property for a match containing *task_number*.

    Args:
        client: NotionClient instance.
        task_number: Task identifier (e.g. ``"TASK-3296"``).
        database_id: Override DB ID. Falls back to default.

    Returns:
        Page ID string, or *None* if not found.
    """
    db_id = _resolve_database_id(database_id)
    result = client.query_database(
        database_id=db_id,
        filter={
            "property": "Name",
            "title": {"contains": task_number},
        },
        page_size=1,
    )
    pages = result.get("results", [])
    if pages:
        return pages[0]["id"]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch helpers
# ──────────────────────────────────────────────────────────────────────────────


def _dispatch_create(
    client: NotionClient,
    db_id: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handle notion_create dispatch."""
    # Detailed mode: raw properties take precedence
    raw_props = args.get("properties")
    if raw_props:
        return client.create_page(db_id, raw_props)

    # Simple mode: task_name + summary required
    task_name = args.get("task_name")
    summary = args.get("summary")
    if not task_name or not summary:
        raise ValueError(
            "Either 'properties' (detailed mode) or both "
            "'task_name' and 'summary' (simple mode) are required."
        )
    properties = _build_default_properties(
        task_name=task_name,
        summary=summary,
        status=args.get("status", "新規"),
        labels=args.get("labels"),
    )
    return client.create_page(db_id, properties)


def _dispatch_query(
    client: NotionClient,
    db_id: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handle notion_query dispatch."""
    # Build filter: category_filter > status_filter > raw filter
    filter_obj = args.get("filter")
    category = args.get("category_filter")
    status = args.get("status_filter")

    if category:
        filter_obj = _build_category_filter(category)
    elif status:
        filter_obj = {"property": "Status", "select": {"equals": status}}

    return client.query_database(
        database_id=db_id,
        filter=filter_obj,
        sorts=args.get("sorts"),
        page_size=args.get("page_size", 100),
    )


def _dispatch_update(
    client: NotionClient,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handle notion_update dispatch."""
    page_id = args["page_id"]

    # Detailed mode: raw properties
    raw_props = args.get("properties")
    if raw_props:
        return client.update_page(page_id, raw_props)

    # Simple mode: status shorthand
    status = args.get("status")
    if status:
        properties = {"Status": {"select": {"name": status}}}
        return client.update_page(page_id, properties)

    raise ValueError(
        "Either 'properties' (detailed mode) or 'status' "
        "(simple mode) is required for update."
    )


def _dispatch_create_fb_db(
    client: NotionClient,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handle notion_create_fb_db dispatch.

    Finds the task page, checks for existing FB DB, creates if missing.
    """
    task_number = args["task_number"]
    page_id = find_task_page_id(client, task_number)
    if not page_id:
        raise ValueError(f"Task page not found for: {task_number}")

    existing = client.find_child_database(page_id, "FB一覧")
    if existing:
        db_id = existing["id"]
        return {
            "status": "already_exists",
            "database_id": db_id,
            "url": build_page_url(db_id),
        }

    db = client.create_database(page_id, "FB一覧", FB_DB_SCHEMA)
    return {
        "status": "created",
        "database_id": db["id"],
        "url": build_page_url(db["id"]),
    }


def _dispatch_add_fb(
    client: NotionClient,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Handle notion_add_fb dispatch.

    Adds a feedback record to an existing FB一覧 database.
    """
    db_id = args["db_id"]
    properties: dict[str, Any] = {
        "FBのID": {"title": [{"text": {"content": args["fb_id"]}}]},
    }
    if args.get("問題の概要"):
        properties["問題の概要"] = {
            "rich_text": [{"text": {"content": args["問題の概要"][:2000]}}],
        }
    if args.get("再現方法"):
        properties["再現方法"] = {
            "rich_text": [{"text": {"content": args["再現方法"][:2000]}}],
        }
    if args.get("期待する動作"):
        properties["期待する動作"] = {
            "rich_text": [{"text": {"content": args["期待する動作"][:2000]}}],
        }
    if args.get("タグ"):
        properties["タグ"] = {
            "multi_select": [{"name": t} for t in args["タグ"]],
        }
    status_name = args.get("ステータス", "未対応")
    properties["ステータス"] = {"select": {"name": status_name}}

    page = client.create_page(db_id, properties)
    return {
        "page_id": page["id"],
        "url": build_page_url(page["id"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic tool_use format)
# ──────────────────────────────────────────────────────────────────────────────


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return Anthropic-compatible tool schemas for Notion operations."""
    return [
        {
            "name": "notion_create",
            "description": (
                "Notion DBにページを作成する。"
                "簡易モード: task_name + summary で開発task元DBに登録。"
                "詳細モード: database_id + properties でフルカスタマイズ。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": (
                            "タスク名。簡易モード用"
                            "（database_id省略時に使用）"
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": "AI要約テキスト。簡易モード用",
                    },
                    "database_id": {
                        "type": "string",
                        "description": (
                            "対象DB ID。省略時はデフォルトDB"
                            "（開発task元DB）"
                        ),
                    },
                    "properties": {
                        "type": "object",
                        "description": (
                            "Notion APIプロパティ形式。"
                            "指定時はtask_name/summaryより優先"
                        ),
                    },
                    "status": {
                        "type": "string",
                        "description": "ステータス名。デフォルト: 新規",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ラベル名のリスト。デフォルト: ['AI']",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "notion_query",
            "description": (
                "Notion DBをクエリする。"
                "フィルタ・ソート・ページネーション対応。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "対象DB ID。省略時はデフォルトDB",
                    },
                    "filter": {
                        "type": "object",
                        "description": "Notion filter object",
                    },
                    "sorts": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "ソート条件のリスト",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "取得件数 (1-100, デフォルト 100)",
                    },
                    "status_filter": {
                        "type": "string",
                        "description": (
                            "簡易フィルタ: ステータス名で絞り込み"
                            "（例: '新規', '実装中'）"
                        ),
                    },
                    "category_filter": {
                        "type": "string",
                        "description": (
                            "カテゴリフィルタ: '未着手', '検討中', "
                            "'作業中' 等（task_status_flow.md §4.1準拠）"
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "notion_update",
            "description": "Notion DBのページプロパティを更新する。",
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "更新対象のページID",
                    },
                    "properties": {
                        "type": "object",
                        "description": (
                            "更新するプロパティ（Notion API形式）"
                        ),
                    },
                    "status": {
                        "type": "string",
                        "description": (
                            "簡易モード: ステータス名を直接指定"
                            "（例: '実装中'）"
                        ),
                    },
                },
                "required": ["page_id"],
            },
        },
        {
            "name": "notion_create_fb_db",
            "description": (
                "タスクのNotionページ内に「FB一覧」子DBを作成する。"
                "既に存在する場合はスキップしてURLを返す。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_number": {
                        "type": "string",
                        "description": (
                            "タスク番号（例: 'TASK-3296'）。"
                            "開発task元DBからページを検索する"
                        ),
                    },
                },
                "required": ["task_number"],
            },
        },
        {
            "name": "notion_add_fb",
            "description": (
                "FB一覧DBにフィードバックレコードを追加する。"
                "db_idはnotion_create_fb_dbの戻り値から取得する。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "db_id": {
                        "type": "string",
                        "description": "FB一覧DBのID",
                    },
                    "fb_id": {
                        "type": "string",
                        "description": "FBのID（タイトル）",
                    },
                    "問題の概要": {
                        "type": "string",
                        "description": "問題の概要テキスト",
                    },
                    "再現方法": {
                        "type": "string",
                        "description": "再現方法テキスト",
                    },
                    "期待する動作": {
                        "type": "string",
                        "description": "期待する動作テキスト",
                    },
                    "タグ": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "タグ名のリスト",
                    },
                    "ステータス": {
                        "type": "string",
                        "description": (
                            "ステータス（未対応/対応中/解消済み）。"
                            "デフォルト: 未対応"
                        ),
                    },
                },
                "required": ["db_id", "fb_id"],
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def get_cli_guide() -> str:
    """Return CLI usage guide for Notion tools."""
    return """\
### Notion
```bash
# Create (simple mode)
animaworks-tool notion create --task-name "タスク名" --summary "要約" -j

# Create (detailed mode)
animaworks-tool notion create --database-id "xxx" --properties '{"Name":...}' -j

# Query
animaworks-tool notion query -j
animaworks-tool notion query --status "新規" -j
animaworks-tool notion query --category "未着手" -j

# Update
animaworks-tool notion update --page-id "xxx" --status "実装中" -j
```"""


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for Notion operations.

    Sub-commands::

        create    Create a page (simple or detailed mode)
        query     Query a database
        update    Update a page's properties
        get       Retrieve a single page
    """
    parser = argparse.ArgumentParser(
        prog="animaworks-notion",
        description="AnimaWorks Notion CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a page")
    p_create.add_argument("--task-name", default=None, help="Task name (simple mode)")
    p_create.add_argument("--summary", default=None, help="Summary text (simple mode)")
    p_create.add_argument("--database-id", default=None, help="Database ID")
    p_create.add_argument("--properties", default=None, help="Properties JSON (detailed mode)")
    p_create.add_argument("--status", default="新規", help="Status name")
    p_create.add_argument("--labels", nargs="*", default=None, help="Label names")

    # query
    p_query = sub.add_parser("query", help="Query a database")
    p_query.add_argument("--database-id", default=None, help="Database ID")
    p_query.add_argument("--status", default=None, help="Status filter")
    p_query.add_argument("--category", default=None, help="Category filter")
    p_query.add_argument("--filter", default=None, help="Notion filter JSON")
    p_query.add_argument("--page-size", type=int, default=100, help="Results per page")

    # update
    p_update = sub.add_parser("update", help="Update a page")
    p_update.add_argument("--page-id", required=True, help="Page ID")
    p_update.add_argument("--status", default=None, help="New status name")
    p_update.add_argument("--properties", default=None, help="Properties JSON")

    # get
    p_get = sub.add_parser("get", help="Get a page")
    p_get.add_argument("page_id", help="Page ID")

    args = parser.parse_args(argv)
    client = NotionClient()

    if args.command == "create":
        db_id = _resolve_database_id(args.database_id)
        if args.properties:
            props = json.loads(args.properties)
            result = client.create_page(db_id, props)
        else:
            if not args.task_name or not args.summary:
                parser.error("--task-name and --summary are required in simple mode")
            props = _build_default_properties(
                task_name=args.task_name,
                summary=args.summary,
                status=args.status,
                labels=args.labels,
            )
            result = client.create_page(db_id, props)

    elif args.command == "query":
        db_id = _resolve_database_id(args.database_id)
        filter_obj = None
        if args.category:
            filter_obj = _build_category_filter(args.category)
        elif args.status:
            filter_obj = {"property": "Status", "select": {"equals": args.status}}
        elif args.filter:
            filter_obj = json.loads(args.filter)
        result = client.query_database(
            database_id=db_id,
            filter=filter_obj,
            page_size=args.page_size,
        )

    elif args.command == "update":
        if args.properties:
            props = json.loads(args.properties)
            result = client.update_page(args.page_id, props)
        elif args.status:
            props = {"Status": {"select": {"name": args.status}}}
            result = client.update_page(args.page_id, props)
        else:
            parser.error("--status or --properties required")

    elif args.command == "get":
        result = client.get_page(args.page_id)

    else:
        parser.print_help()
        sys.exit(1)

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()  # trailing newline


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    args.pop("anima_dir", None)
    client = NotionClient()
    db_id = _resolve_database_id(args.get("database_id"))

    if tool_name == "notion_create":
        return _dispatch_create(client, db_id, args)
    if tool_name == "notion_query":
        return _dispatch_query(client, db_id, args)
    if tool_name == "notion_update":
        return _dispatch_update(client, args)
    if tool_name == "notion_create_fb_db":
        return _dispatch_create_fb_db(client, args)
    if tool_name == "notion_add_fb":
        return _dispatch_add_fb(client, args)
    raise ValueError(f"Unknown tool: {tool_name}")
