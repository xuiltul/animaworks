from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Skill, use_tool, and tool management schemas."""

from typing import Any

from core.i18n import t as _t

DISCOVERY_TOOLS: list[dict[str, Any]] = []

USE_TOOL: list[dict[str, Any]] = [
    {
        "name": "use_tool",
        "description": (
            "Execute an external tool action. "
            "Combines tool_name and action to dispatch to the appropriate "
            "external tool module (e.g. chatwork + send → chatwork_send). "
            "Available tools depend on permissions.md settings. "
            "Use read_memory_file on common_skills/<name>/SKILL.md to look up detailed usage for each tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "External tool module name "
                        "(e.g. 'chatwork', 'slack', 'gmail', 'web_search', "
                        "'github', 'aws_collector', 'image_gen', 'x_search', "
                        "'transcribe', 'local_llm', 'google_calendar', 'google_tasks')"
                    ),
                },
                "action": {
                    "type": "string",
                    "description": (
                        "Action/subcommand to execute within the tool (e.g. 'send', 'read', 'search', 'query', 'list')"
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Arguments to pass to the tool action. "
                        "Refer to the tool's skill documentation for "
                        "required and optional parameters."
                    ),
                },
            },
            "required": ["tool_name", "action"],
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


def _create_skill_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "create_skill",
            "description": _t("schema.create_skill.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": _t("schema.create_skill.skill_name"),
                    },
                    "description": {
                        "type": "string",
                        "description": _t("schema.create_skill.description"),
                    },
                    "body": {
                        "type": "string",
                        "description": _t("schema.create_skill.body"),
                    },
                    "location": {
                        "type": "string",
                        "enum": ["personal", "common"],
                        "description": _t("schema.create_skill.location"),
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                        "description": _t("schema.create_skill.references"),
                    },
                    "templates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                        "description": _t("schema.create_skill.templates"),
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.create_skill.allowed_tools"),
                    },
                },
                "required": ["skill_name", "description", "body"],
            },
        },
    ]
