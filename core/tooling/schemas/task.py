from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Task queue and submit_tasks tool schemas."""

from typing import Any

from core.i18n import t as _t


def _submit_tasks_tools() -> list[dict[str, Any]]:
    """Return submit_tasks tool schema list."""
    return [
        {
            "name": "submit_tasks",
            "description": _t("schema.submit_tasks.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {
                        "type": "string",
                        "description": "Unique identifier for this batch of tasks",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "parallel": {"type": "boolean", "default": False},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                                "acceptance_criteria": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                                "constraints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                                "file_paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Background context for the task executor",
                                    "default": "",
                                },
                                "reply_to": {
                                    "type": "string",
                                    "description": "Name of the Anima to notify on completion (default: submitter)",
                                },
                                "model": {
                                    "type": "string",
                                    "description": _t("schema.submit_tasks.task_model"),
                                },
                                "workspace": {
                                    "type": "string",
                                    "description": "Workspace alias or alias#hash for the task's working directory",
                                    "default": "",
                                },
                            },
                            "required": ["task_id", "title", "description"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["batch_id", "tasks"],
            },
        },
    ]


SUBMIT_TASKS_TOOLS = _submit_tasks_tools


def _task_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "backlog_task",
            "description": _t("schema.backlog_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["human", "anima"],
                        "description": _t("schema.backlog_task.source"),
                    },
                    "original_instruction": {
                        "type": "string",
                        "description": _t("schema.backlog_task.original_instruction"),
                    },
                    "assignee": {
                        "type": "string",
                        "description": _t("schema.backlog_task.assignee"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.backlog_task.summary"),
                    },
                    "relay_chain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.backlog_task.relay_chain"),
                    },
                },
                "required": ["source", "original_instruction", "assignee", "summary"],
            },
        },
        {
            "name": "update_task",
            "description": _t("schema.update_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": _t("schema.update_task.task_id"),
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled"],
                        "description": _t("schema.update_task.status"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.update_task.summary"),
                    },
                    "result": {
                        "type": "string",
                        "description": _t("schema.update_task.result"),
                    },
                },
                "required": ["task_id", "status"],
            },
        },
        {
            "name": "list_tasks",
            "description": _t("schema.list_tasks.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled", "delegated"],
                        "description": _t("schema.list_tasks.status"),
                    },
                    "detail": {
                        "type": "boolean",
                        "description": _t("schema.list_tasks.detail"),
                    },
                },
            },
        },
    ]
