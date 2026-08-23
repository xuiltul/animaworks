from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Supervisor, vault, and background task tool schemas."""

from typing import Any

from core.i18n import t as _t


def _supervisor_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "disable_subordinate",
            "description": _t("schema.disable_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.disable_subordinate.name"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.disable_subordinate.reason"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "enable_subordinate",
            "description": _t("schema.enable_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.enable_subordinate.name"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "set_subordinate_model",
            "description": _t("schema.set_subordinate_model.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.name"),
                    },
                    "model": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.model"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_model.reason"),
                    },
                },
                "required": ["name", "model"],
            },
        },
        {
            "name": "set_subordinate_background_model",
            "description": _t("schema.set_subordinate_background_model.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.name"),
                    },
                    "model": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.model"),
                    },
                    "credential": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.credential"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.set_subordinate_background_model.reason"),
                    },
                },
                "required": ["name", "model"],
            },
        },
        {
            "name": "restart_subordinate",
            "description": _t("schema.restart_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.restart_subordinate.name"),
                    },
                    "reason": {
                        "type": "string",
                        "description": _t("schema.restart_subordinate.reason"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "org_dashboard",
            "description": _t("schema.org_dashboard.desc"),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ping_subordinate",
            "description": _t("schema.ping_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.ping_subordinate.name"),
                    },
                },
            },
        },
        {
            "name": "read_subordinate_state",
            "description": _t("schema.read_subordinate_state.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.read_subordinate_state.name"),
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "delegate_task",
            "description": _t("schema.delegate_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.delegate_task.name"),
                    },
                    "instruction": {
                        "type": "string",
                        "description": _t("schema.delegate_task.instruction"),
                    },
                    "summary": {
                        "type": "string",
                        "description": _t("schema.delegate_task.summary"),
                    },
                    "deadline": {
                        "type": "string",
                        "description": _t("schema.delegate_task.deadline"),
                    },
                    "workspace": {
                        "type": "string",
                        "description": _t("schema.delegate_task.workspace"),
                    },
                    "exclusive_key": {
                        "type": "string",
                        "description": _t("schema.delegate_task.exclusive_key"),
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _t("schema.delegate_task.acceptance_criteria"),
                        "default": [],
                    },
                    "model": {
                        "type": "string",
                        "description": _t("schema.delegate_task.model"),
                    },
                },
                "required": ["name", "instruction", "deadline"],
            },
        },
        {
            "name": "task_tracker",
            "description": _t("schema.task_tracker.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "active", "completed"],
                        "description": _t("schema.task_tracker.status"),
                    },
                },
            },
        },
        {
            "name": "audit_subordinate",
            "description": _t("schema.audit_subordinate.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": _t("schema.audit_subordinate.name"),
                    },
                    "hours": {
                        "type": "integer",
                        "description": _t("schema.audit_subordinate.hours"),
                    },
                    "direct_only": {
                        "type": "boolean",
                        "description": _t("schema.audit_subordinate.direct_only"),
                    },
                    "since": {
                        "type": "string",
                        "description": _t("schema.audit_subordinate.since"),
                    },
                },
            },
        },
    ]


def _check_permissions_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "check_permissions",
            "description": _t("schema.check_permissions.desc"),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def _vault_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "vault_get",
            "description": _t("schema.vault_get.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_get.section"),
                    },
                    "key": {
                        "type": "string",
                        "description": _t("schema.vault_get.key"),
                    },
                },
                "required": ["section", "key"],
            },
        },
        {
            "name": "vault_store",
            "description": _t("schema.vault_store.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_store.section"),
                    },
                    "key": {
                        "type": "string",
                        "description": _t("schema.vault_store.key"),
                    },
                    "value": {
                        "type": "string",
                        "description": _t("schema.vault_store.value"),
                    },
                },
                "required": ["section", "key", "value"],
            },
        },
        {
            "name": "vault_list",
            "description": _t("schema.vault_list.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": _t("schema.vault_list.section"),
                    },
                },
            },
        },
    ]


def _background_task_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "check_background_task",
            "description": _t("schema.check_background_task.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": _t("schema.check_background_task.task_id"),
                    },
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "list_background_tasks",
            "description": _t("schema.list_background_tasks.desc"),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "pending"],
                        "description": _t("schema.list_background_tasks.status"),
                    },
                },
            },
        },
    ]
