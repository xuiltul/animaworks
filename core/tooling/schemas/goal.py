from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent goal loop tool schema."""

from typing import Any


def _goal_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "goal",
            "description": "Manage a persistent goal loop backed by TaskExec and task_queue metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "pause", "resume", "clear", "status", "judge"],
                    },
                    "goal_id": {"type": "string"},
                    "title": {"type": "string"},
                    "objective": {"type": "string"},
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "max_iterations": {"type": "integer", "minimum": 1, "default": 5},
                    "judge_model": {"type": "string"},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                    "task_id": {"type": "string"},
                    "result_summary": {"type": "string"},
                    "verification_output": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["done", "continue", "blocked"],
                    },
                    "continuation_prompt": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    ]
