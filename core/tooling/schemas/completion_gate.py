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
                "properties": {},
            },
        },
    ]
