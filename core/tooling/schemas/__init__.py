# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical tool schema definitions and format converters.

All tool schemas are defined once in a provider-neutral format and converted
to Anthropic or LiteLLM/OpenAI formats on demand.
"""

from __future__ import annotations

from core.tooling.schemas.admin import _AW_CORE_NAMES, ADMIN_TOOLS, CC_TOOLS
from core.tooling.schemas.builder import (
    _CONSOLIDATION_BLOCKED_TOOLS,
    build_tool_list,
    build_unified_tool_list,
)
from core.tooling.schemas.channel import _channel_tools
from core.tooling.schemas.completion_gate import _completion_gate_tools
from core.tooling.schemas.converters import (
    apply_db_descriptions,
    to_anthropic_format,
    to_litellm_format,
    to_text_format,
)
from core.tooling.schemas.loader import (
    _normalise_schema,
    load_all_tool_schemas,
    load_external_schemas,
    load_external_schemas_by_category,
    load_personal_tool_schemas,
)
from core.tooling.schemas.memory import (
    FILE_TOOLS,
    KNOWLEDGE_TOOLS,
    MEMORY_TOOLS,
    PROCEDURE_TOOLS,
    SEARCH_TOOLS,
)
from core.tooling.schemas.notification import _notification_tools
from core.tooling.schemas.skill import (
    DISCOVERY_TOOLS,
    TOOL_MANAGEMENT_TOOLS,
    USE_TOOL,
    _create_skill_schemas,
)
from core.tooling.schemas.supervisor import (
    _background_task_tools,
    _check_permissions_tools,
    _supervisor_tools,
    _vault_tools,
)
from core.tooling.schemas.task import SUBMIT_TASKS_TOOLS, _submit_tasks_tools, _task_tools

__all__ = [
    "ADMIN_TOOLS",
    "CC_TOOLS",
    "DISCOVERY_TOOLS",
    "FILE_TOOLS",
    "KNOWLEDGE_TOOLS",
    "MEMORY_TOOLS",
    "PROCEDURE_TOOLS",
    "SEARCH_TOOLS",
    "SUBMIT_TASKS_TOOLS",
    "_submit_tasks_tools",
    "TOOL_MANAGEMENT_TOOLS",
    "USE_TOOL",
    "_AW_CORE_NAMES",
    "_CONSOLIDATION_BLOCKED_TOOLS",
    "_background_task_tools",
    "_channel_tools",
    "_completion_gate_tools",
    "_check_permissions_tools",
    "_notification_tools",
    "_normalise_schema",
    "_create_skill_schemas",
    "_supervisor_tools",
    "_task_tools",
    "_vault_tools",
    "apply_db_descriptions",
    "build_tool_list",
    "build_unified_tool_list",
    "load_all_tool_schemas",
    "load_external_schemas",
    "load_external_schemas_by_category",
    "load_personal_tool_schemas",
    "to_anthropic_format",
    "to_litellm_format",
    "to_text_format",
]
