from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Canonical tool schema definitions and format converters.

All tool schemas are defined once in a provider-neutral format and converted
to Anthropic or LiteLLM/OpenAI formats on demand.  This eliminates the
duplicate definitions that previously lived in ``_build_a2_tools()`` and
``_build_anthropic_tools()``.
"""

import logging
from typing import Any

logger = logging.getLogger("animaworks.tool_schemas")

# ── Canonical definitions ────────────────────────────────────
#
# Format: {"name", "description", "parameters"} where ``parameters`` is a
# standard JSON Schema object.  This is convertible to both Anthropic
# (``input_schema``) and OpenAI/LiteLLM (``function.parameters``) formats.

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_memory",
        "description": (
            "Search the anima's long-term memory "
            "(knowledge, episodes, procedures) by keyword."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
                "scope": {
                    "type": "string",
                    "enum": ["knowledge", "episodes", "procedures", "common_knowledge", "all"],
                    "description": "Memory category to search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_memory_file",
        "description": "Read a file from the anima's memory directory by relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within anima dir",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_memory_file",
        "description": "Write or append to a file in the anima's memory directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "send_message",
        "description": (
            "Send a message to another anima or a human user. "
            "For human users, the message is automatically routed "
            "to the configured preferred channel (e.g. Slack, Chatwork)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient name. Can be an anima name (e.g. 'sakura') "
                        "or a human alias (e.g. 'user', 'taka'). "
                        "Messages to human aliases are automatically delivered "
                        "via the configured external channel."
                    ),
                },
                "content": {"type": "string", "description": "Message content"},
                "reply_to": {"type": "string", "description": "Message ID to reply to"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "intent": {
                    "type": "string",
                    "description": (
                        "Message intent: 'delegation' (task assignment), "
                        "'report' (status/result report — use the report template), "
                        "'question' (question/confirmation), "
                        "or '' (default, casual/FYI)."
                    ),
                },
            },
            "required": ["to", "content"],
        },
    },
]

CHANNEL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "post_channel",
        "description": (
            "Boardの共有チャネルにメッセージを投稿する。"
            "チーム全体に共有すべき情報はgeneralチャネルに、"
            "運用・インフラ関連はopsチャネルに投稿する。"
            "全Animaが閲覧できるため、解決済み情報の共有や"
            "お知らせに使うこと。1対1の連絡にはsend_messageを使う。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "チャネル名 (general=全体共有, ops=運用系)",
                },
                "text": {
                    "type": "string",
                    "description": "投稿するメッセージ本文。@名前 でメンション可能（メンション先にDM通知される）。@all で起動中の全員にDM通知",
                },
            },
            "required": ["channel", "text"],
        },
    },
    {
        "name": "read_channel",
        "description": (
            "Boardの共有チャネルの直近メッセージを読む。"
            "他のAnimaやユーザーが共有した情報を確認できる。"
            "human_only=trueでユーザー発言のみフィルタリング可能。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "チャネル名 (general, ops)",
                },
                "limit": {
                    "type": "integer",
                    "description": "取得件数（デフォルト: 20）",
                },
                "human_only": {
                    "type": "boolean",
                    "description": "trueの場合、人間の発言のみ返す",
                },
            },
            "required": ["channel"],
        },
    },
    {
        "name": "read_dm_history",
        "description": (
            "特定の相手との過去のDM履歴を読む。"
            "send_messageで送受信したメッセージの履歴を時系列で確認できる。"
            "以前のやり取りの文脈を確認したいときに使う。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "peer": {
                    "type": "string",
                    "description": "DM相手の名前",
                },
                "limit": {
                    "type": "integer",
                    "description": "取得件数（デフォルト: 20）",
                },
            },
            "required": ["peer"],
        },
    },
]

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read an arbitrary file (subject to permissions).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (subject to permissions).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace a specific string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "old_string": {"type": "string", "description": "Text to find"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command (subject to permissions allow-list).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                },
            },
            "required": ["command"],
        },
    },
]

SEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_code",
        "description": (
            "Search for a text pattern in files using regex. "
            "Returns matching lines with file paths and line numbers. "
            "Use this instead of execute_command with grep."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in (default: anima_dir)",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.py', '*.md')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and directories at a given path. "
            "Supports glob patterns for filtering. "
            "Use this instead of execute_command with ls or find."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: anima_dir)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern filter (e.g. '**/*.py')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Include subdirectories (default: false)",
                },
            },
        },
    },
]

NOTIFICATION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "call_human",
        "description": (
            "人間の管理者に連絡します。"
            "重要な報告、問題のエスカレーション、判断が必要な事項がある場合に使用してください。"
            "チャット画面と外部通知チャネル（Slack等）の両方に届きます。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "通知の件名（簡潔に）",
                },
                "body": {
                    "type": "string",
                    "description": "通知の本文（詳細な報告内容）",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "通知の優先度（デフォルト: normal）",
                },
            },
            "required": ["subject", "body"],
        },
    },
]

DISCOVERY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "discover_tools",
        "description": (
            "Discover available external tools. "
            "Call without arguments to list available categories. "
            "Call with a category name to activate that category's tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Tool category to activate "
                        "(e.g. 'chatwork', 'slack', 'gmail'). "
                        "Omit to list all available categories."
                    ),
                },
            },
        },
    },
]

TOOL_MANAGEMENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "refresh_tools",
        "description": (
            "Re-scan personal and common tool directories to discover "
            "newly created tools. Call this after creating a new tool "
            "file to make it immediately available in the current session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "share_tool",
        "description": (
            "Copy a personal tool to common_tools/ so all animas can use it. "
            "The tool file is copied from your tools/ directory to the shared "
            "common_tools/ directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Tool file name without .py extension",
                },
            },
            "required": ["tool_name"],
        },
    },
]

ADMIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "create_anima",
        "description": (
            "Create a new Digital Anima from a character sheet. "
            "Pass the character sheet content directly via character_sheet_content, "
            "or specify a path via character_sheet_path. "
            "The factory creates the directory structure atomically, "
            "and the new anima self-configures via bootstrap on first startup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "character_sheet_content": {
                    "type": "string",
                    "description": (
                        "Character sheet markdown content as a string. "
                        "Preferred over character_sheet_path. "
                        "Must include required sections: 基本情報, 人格, 役割・行動方針."
                    ),
                },
                "character_sheet_path": {
                    "type": "string",
                    "description": (
                        "Path to the character_sheet.md file "
                        "(absolute or relative to anima_dir). "
                        "Ignored if character_sheet_content is provided."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Anima name (lowercase alphanumeric). If omitted, extracted from sheet.",
                },
                "supervisor": {
                    "type": "string",
                    "description": (
                        "Supervisor person name (lowercase). "
                        "Overrides the 上司 field in the character sheet. "
                        "If omitted, falls back to the sheet value, "
                        "then to the calling person."
                    ),
                },
                "role": {
                    "type": "string",
                    "enum": ["engineer", "researcher", "manager", "writer", "ops", "general"],
                    "description": (
                        "Role template to apply. Determines specialty prompt, "
                        "default model, and execution parameters. "
                        "Default: general."
                    ),
                },
            },
            "required": [],
        },
    },
]

PROCEDURE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "report_procedure_outcome",
        "description": (
            "Report the outcome of following a procedure or skill. "
            "Updates success/failure counts and confidence. "
            "Call this after completing a procedure to track its reliability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to the procedure or skill file "
                        "(e.g. 'procedures/deploy.md' or 'skills/git-flow.md')"
                    ),
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the procedure succeeded",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes on what worked or failed",
                },
            },
            "required": ["path", "success"],
        },
    },
]

TASK_TOOLS: list[dict[str, Any]] = [
    {
        "name": "add_task",
        "description": (
            "タスクキューに新しいタスクを追加する。"
            "人間からの指示は必ず source='human' で記録すること。"
            "Anima間の委任は source='anima' で記録する。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["human", "anima"],
                    "description": "タスクの発生源 (human=人間からの指示, anima=Anima間委任)",
                },
                "original_instruction": {
                    "type": "string",
                    "description": "元の指示文（委任時は原文引用を含める）",
                },
                "assignee": {
                    "type": "string",
                    "description": "担当者名（自分自身または委任先のAnima名）",
                },
                "summary": {
                    "type": "string",
                    "description": "タスクの1行要約",
                },
                "deadline": {
                    "type": "string",
                    "description": "期限（必須）。相対形式 '30m','2h','1d' またはISO8601。例: '1h' = 1時間後",
                },
                "relay_chain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "委任経路（例: ['taka', 'sakura', 'rin']）",
                },
            },
            "required": ["source", "original_instruction", "assignee", "summary", "deadline"],
        },
    },
    {
        "name": "update_task",
        "description": (
            "タスクのステータスを更新する。"
            "完了時は status='done'、中断時は status='cancelled' に設定する。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "タスクID（add_task時に返されたID）",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "cancelled", "blocked"],
                    "description": "新しいステータス",
                },
                "summary": {
                    "type": "string",
                    "description": "更新後の要約（任意）",
                },
            },
            "required": ["task_id", "status"],
        },
    },
    {
        "name": "list_tasks",
        "description": (
            "タスクキューの一覧を取得する。"
            "ステータスでフィルタリング可能。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "cancelled", "blocked"],
                    "description": "フィルタするステータス（省略時は全件）",
                },
            },
        },
    },
]

# ── Format converters ────────────────────────────────────────


def to_anthropic_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to Anthropic API format (``input_schema``)."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def to_litellm_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical schemas to LiteLLM/OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def to_text_format(schemas: list[dict[str, Any]]) -> str:
    """Convert canonical tool schemas to text specification for Mode B.

    Generates a markdown-formatted tool guide that instructs the LLM to
    output tool calls as JSON code blocks.  Used by ``AssistedExecutor``
    to inject tool specifications into the system prompt.
    """
    lines = [
        "## 利用可能なツール",
        "",
        "ツールを使いたい場合、以下の形式で ```json コードブロックとして出力してください:",
        "",
        "```json",
        '{"tool": "ツール名", "arguments": {"引数名": "値"}}',
        "```",
        "",
        "ツールの実行結果は次のメッセージで提供されます。",
        "ツールを使う必要がなければ、普通にテキストで返答してください。",
        "1回のメッセージでツール呼び出しは1つだけにしてください。",
        "",
    ]
    for schema in schemas:
        name = schema["name"]
        desc = schema.get("description", "")
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))
        args_parts = []
        for k, v in params.items():
            type_str = v.get("type", "?")
            req_str = " (必須)" if k in required else ""
            args_parts.append(f"{k}: {type_str}{req_str}")
        args_desc = ", ".join(args_parts)
        lines.append(f"- **{name}**: {desc}")
        if args_desc:
            lines.append(f"  - 引数: {args_desc}")
    return "\n".join(lines)


# ── Builder helpers ──────────────────────────────────────────


def build_tool_list(
    *,
    include_file_tools: bool = False,
    include_search_tools: bool = False,
    include_discovery_tools: bool = False,
    include_notification_tools: bool = False,
    include_admin_tools: bool = False,
    include_tool_management: bool = False,
    include_task_tools: bool = False,
    external_schemas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble a tool list from canonical definitions.

    Args:
        include_file_tools: Include file/command operation tools (for A2 mode).
        include_search_tools: Include search_code/list_directory tools.
        include_discovery_tools: Include discover_tools tool.
        include_notification_tools: Include call_human tool (for top-level Animas).
        include_admin_tools: Include admin tools (create_anima etc.).
        include_tool_management: Include refresh_tools/share_tool tools.
        include_task_tools: Include task queue tools (add_task, update_task, list_tasks).
        external_schemas: Additional tool schemas in canonical format.

    Returns:
        Combined list in canonical format.
    """
    tools: list[dict[str, Any]] = list(MEMORY_TOOLS)
    # Channel tools are always included (shared messaging)
    tools.extend(CHANNEL_TOOLS)
    # Procedure outcome reporting is always included
    tools.extend(PROCEDURE_TOOLS)
    if include_file_tools:
        tools.extend(FILE_TOOLS)
    if include_search_tools:
        tools.extend(SEARCH_TOOLS)
    if include_discovery_tools:
        tools.extend(DISCOVERY_TOOLS)
    if include_notification_tools:
        tools.extend(NOTIFICATION_TOOLS)
    if include_admin_tools:
        tools.extend(ADMIN_TOOLS)
    if include_tool_management:
        tools.extend(TOOL_MANAGEMENT_TOOLS)
    if include_task_tools:
        tools.extend(TASK_TOOLS)
    if external_schemas:
        tools.extend(external_schemas)
    return tools


# ── Schema loading ───────────────────────────────────────────


def _normalise_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single tool schema to canonical format."""
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "parameters": raw.get("input_schema", raw.get("parameters", {})),
    }


def load_external_schemas(tool_registry: list[str]) -> list[dict[str, Any]]:
    """Load schemas from external tool modules, normalised to canonical format."""
    if not tool_registry:
        return []

    import importlib

    from core.tools import TOOL_MODULES

    schemas: list[dict[str, Any]] = []
    for tool_name in tool_registry:
        if tool_name not in TOOL_MODULES:
            continue
        try:
            mod = importlib.import_module(TOOL_MODULES[tool_name])
            if not hasattr(mod, "get_tool_schemas"):
                continue
            for s in mod.get_tool_schemas():
                schemas.append(_normalise_schema(s))
        except Exception:
            logger.debug("Failed to load schemas for %s", tool_name, exc_info=True)
    return schemas


def load_personal_tool_schemas(
    personal_tools: dict[str, str],
) -> list[dict[str, Any]]:
    """Load schemas from personal tool modules, normalised to canonical format."""
    import importlib.util

    schemas: list[dict[str, Any]] = []
    for tool_name, file_path in personal_tools.items():
        try:
            spec = importlib.util.spec_from_file_location(
                f"animaworks_personal_tool_{tool_name}", file_path,
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if not hasattr(mod, "get_tool_schemas"):
                continue
            for s in mod.get_tool_schemas():
                schemas.append(_normalise_schema(s))
        except Exception:
            logger.debug(
                "Failed to load personal tool schemas: %s",
                tool_name, exc_info=True,
            )
    return schemas


def load_all_tool_schemas(
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load and normalise tool schemas from all enabled modules."""
    schemas = load_external_schemas(tool_registry or [])
    if personal_tools:
        schemas.extend(load_personal_tool_schemas(personal_tools))
    return schemas
