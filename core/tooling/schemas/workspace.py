from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Workspace access management tool schemas."""

from typing import Any

WORKSPACE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "grant_workspace_access",
        "description": (
            "Register a workspace and grant explicit write access to a top-level Anima "
            "or one of its descendant Animas. This tool is allowed only for human-origin "
            "instructions handled by a top-level Anima. It updates the global workspace "
            "registry, the target Anima's permissions.json file_roots, and optionally "
            "the target Anima's default_workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "alias": {
                    "type": "string",
                    "description": "Short workspace alias, such as finance-dashboard.",
                },
                "path": {
                    "type": "string",
                    "description": "Existing directory path to grant as a writable workspace.",
                },
                "target_anima": {
                    "type": "string",
                    "description": (
                        "Optional target Anima name or alias. Omit to grant access to the calling top-level Anima."
                    ),
                },
                "make_default": {
                    "type": "boolean",
                    "description": "Whether to set target status.json default_workspace to alias#hash. Default: true.",
                },
            },
            "required": ["alias", "path"],
        },
    }
]
