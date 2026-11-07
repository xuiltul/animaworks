from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel (Board) tool schemas."""

from typing import Any

from core.i18n import t as _t


def _channel_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "post_channel",
            "description": _t("schema.post_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": _t("schema.post_channel.channel"),
                    },
                    "text": {
                        "type": "string",
                        "description": _t("schema.post_channel.text"),
                    },
                },
                "required": ["channel", "text"],
            },
        },
        {
            "name": "read_channel",
            "description": _t("schema.read_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": _t("schema.read_channel.channel"),
                    },
                    "limit": {
                        "type": "integer",
                        "description": _t("schema.read_channel.limit"),
                    },
                    "human_only": {
                        "type": "boolean",
                        "description": _t("schema.read_channel.human_only"),
                    },
                    "latest": {
                        "type": "boolean",
                        "description": _t("schema.read_channel.latest"),
                    },
                },
                "required": ["channel"],
            },
        },
        {
            "name": "read_dm_history",
            "description": _t("schema.read_dm_history.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "peer": {
                        "type": "string",
                        "description": _t("schema.read_dm_history.peer"),
                    },
                    "limit": {
                        "type": "integer",
                        "description": _t("schema.read_dm_history.limit"),
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["sent", "received", "both"],
                        "description": _t("schema.read_dm_history.direction"),
                    },
                    "hours": {
                        "type": "integer",
                        "description": _t("schema.read_dm_history.hours"),
                    },
                    "keyword": {
                        "type": "string",
                        "description": _t("schema.read_dm_history.keyword"),
                    },
                },
                "required": ["peer"],
            },
        },
        {
            "name": "manage_channel",
            "description": _t("schema.manage_channel.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "archive", "add_member", "remove_member", "info"],
                        "description": _t("schema.manage_channel.action"),
                    },
                    "channel": {
                        "type": "string",
                        "description": _t("schema.manage_channel.channel"),
                    },
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.manage_channel.members"),
                    },
                    "description": {
                        "type": "string",
                        "description": _t("schema.manage_channel.description"),
                    },
                },
                "required": ["action", "channel"],
            },
        },
    ]
