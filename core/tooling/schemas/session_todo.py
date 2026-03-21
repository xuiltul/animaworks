from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Session todo_write tool schema (Mode A planning aid)."""

from typing import Any

from core.i18n import t as _t


def _session_todo_tools() -> list[dict[str, Any]]:
    """Return todo_write tool schema list."""
    return [
        {
            "name": "todo_write",
            "description": _t("schema.todo_write.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": _t("schema.todo_write.todos"),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": _t("schema.todo_write.id"),
                                },
                                "content": {
                                    "type": "string",
                                    "description": _t("schema.todo_write.content"),
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": _t("schema.todo_write.status"),
                                },
                            },
                            "required": ["id", "content", "status"],
                        },
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "merge": {
                        "type": "boolean",
                        "description": _t("schema.todo_write.merge"),
                    },
                },
                "required": ["todos"],
            },
        },
    ]


SESSION_TODO_TOOLS = _session_todo_tools
