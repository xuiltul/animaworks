from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""completion_gate tool schema (Mode S pre-completion verification)."""

from typing import Any

from core.i18n import t as _t


def _completion_gate_tools() -> list[dict[str, Any]]:
    """Return ``completion_gate`` tool schema list."""
    return [
        {
            "name": "completion_gate",
            "description": _t("completion_gate.schema_description"),
            "parameters": {
                "type": "object",
                "properties": {
                    "applied_skill_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Skill refs actually applied in this run. Use "
                            "skills/{name}/SKILL.md or common_skills/{name}/SKILL.md."
                        ),
                    },
                    "applied_procedure_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Procedure refs actually followed, e.g. procedures/deploy.md.",
                    },
                    "skill_creation": {
                        "type": "object",
                        "description": "Report whether this run created or identified a reusable skill.",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["created", "candidate_only", "not_needed"],
                                "description": "Whether a skill was created, only identified, or not needed.",
                            },
                            "created_skill_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Created skill refs, e.g. skills/{name}/SKILL.md.",
                            },
                            "candidate_name": {
                                "type": "string",
                                "description": "Reusable skill candidate name when status is candidate_only.",
                            },
                            "candidate_reason": {
                                "type": "string",
                                "description": "Why this candidate may deserve a skill.",
                            },
                            "no_new_skill_reason": {
                                "type": "string",
                                "description": "Why no new skill was needed.",
                            },
                        },
                    },
                },
            },
        },
    ]
