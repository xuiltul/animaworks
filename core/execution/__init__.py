from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Execution engines for AgentCore.

Each engine implements one execution mode:
  - ``AgentSDKExecutor``  (A1): Claude Agent SDK -- full tool access via subprocess
  - ``LiteLLMExecutor``   (A2): LiteLLM + tool_use loop -- any model with tool support
  - ``AssistedExecutor``  (B):  1-shot LLM call -- framework handles memory I/O
  - ``AnthropicFallbackExecutor``: Anthropic SDK direct -- fallback when Agent SDK unavailable
"""

# AgentSDKExecutor requires claude_agent_sdk which may not be installed.
# Import it lazily so the rest of the package works regardless.
try:
    from core.execution.agent_sdk import AgentSDKExecutor
except ImportError:  # pragma: no cover
    AgentSDKExecutor = None  # type: ignore[assignment,misc]

from core.execution.anthropic_fallback import AnthropicFallbackExecutor
from core.execution.assisted import AssistedExecutor
from core.execution.base import BaseExecutor, ExecutionResult
from core.execution.litellm_loop import LiteLLMExecutor

__all__ = [
    "AgentSDKExecutor",
    "AnthropicFallbackExecutor",
    "AssistedExecutor",
    "BaseExecutor",
    "ExecutionResult",
    "LiteLLMExecutor",
]
