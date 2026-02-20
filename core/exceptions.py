from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Unified exception hierarchy for AnimaWorks.

All domain-specific exceptions derive from :class:`AnimaWorksError`,
enabling callers to catch the entire family with a single clause::

    try:
        ...
    except AnimaWorksError as e:
        logger.error("Domain error: %s", e)
"""


class AnimaWorksError(Exception):
    """Base exception for all AnimaWorks errors."""


# ── Execution ────────────────────────────────────────────────


class ExecutionError(AnimaWorksError):
    """LLM execution errors."""


class LLMAPIError(ExecutionError):
    """LLM API call failure (network, auth, rate limit)."""


class LLMTimeoutError(ExecutionError):
    """LLM API timeout."""


class StreamDisconnectedError(ExecutionError):
    """Streaming session disconnected unexpectedly.

    Carries partial response text accumulated before the disconnect
    so AgentCore can build a checkpoint-based retry prompt.
    """

    def __init__(
        self,
        message: str = "Stream disconnected",
        *,
        partial_text: str = "",
    ) -> None:
        super().__init__(message)
        self.partial_text = partial_text


# ── Tool ─────────────────────────────────────────────────────


class ToolError(AnimaWorksError):
    """Tool execution errors."""


class ToolConfigError(ToolError):
    """Tool configuration incomplete (missing env var / credential)."""


class ToolExecutionError(ToolError):
    """Tool execution failure at runtime."""


class ToolNotFoundError(ToolError):
    """Requested tool not found or not available."""


# ── Memory I/O ───────────────────────────────────────────────


class MemoryIOError(AnimaWorksError):
    """Memory subsystem I/O errors."""


class MemoryReadError(MemoryIOError):
    """Failed to read from memory storage."""


class MemoryWriteError(MemoryIOError):
    """Failed to write to memory storage."""


class MemoryCorruptedError(MemoryIOError):
    """Memory data is corrupted (JSON decode failure, schema mismatch)."""


# ── Process / IPC ────────────────────────────────────────────


class ProcessError(AnimaWorksError):
    """Process and IPC errors."""


class AnimaNotFoundError(ProcessError):
    """Referenced Anima does not exist."""


class AnimaNotRunningError(ProcessError):
    """Anima process is not running."""


class IPCConnectionError(ProcessError):
    """IPC connection failure (socket, timeout)."""


# ── Configuration ────────────────────────────────────────────


class ConfigError(AnimaWorksError):
    """Configuration errors."""


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""


class ConfigValidationError(ConfigError):
    """Configuration validation failure."""


# ── Messaging ────────────────────────────────────────────────


class MessagingError(AnimaWorksError):
    """Messaging and routing errors."""


class RecipientNotFoundError(MessagingError):
    """Message recipient not found."""


class DeliveryError(MessagingError):
    """Message delivery failure."""
