from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Mode S session persistence, SDK input helpers, and cleanup utilities.

Leaf module in the dependency graph — no internal framework imports.
"""

import json
import logging
import shutil
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.schemas import ImageData

logger = logging.getLogger("animaworks.execution.agent_sdk")


# ── SDK constants ────────────────────────────────────────────

# Safety margin for Agent SDK JSON-RPC buffer.  The default (1 MB) is too
# small when system_prompt + conversation history grow large; 4 MB gives
# comfortable headroom while still catching genuinely broken messages.
_SDK_MAX_BUFFER_SIZE = 4 * 1024 * 1024  # 4 MB

# Linux MAX_ARG_STRLEN is 128 KiB (131072 bytes) per argument — a kernel
# compile-time constant that cannot be changed at runtime.  When the
# system prompt exceeds this limit, `execve` fails with E2BIG ([Errno 7]).
# We use a conservative threshold (100 KB) to leave headroom for encoding
# overhead and other arguments.  When exceeded, the prompt is written to a
# temp file and passed via the CLI's undocumented --system-prompt-file flag.
_PROMPT_FILE_THRESHOLD = 100_000  # 100 KB

# SDK Issue #387: invalid session ID causes SDK to hang for ~60s before
# raising an error.  We wrap the first-event receive in asyncio.wait_for
# so that a stale/invalid resume fails fast and falls back to a fresh session.
RESUME_TIMEOUT_SEC = 15.0

# When estimated context usage leaves fewer than max_tokens * this factor
# free, the PreToolUse hook triggers session termination for auto-compact.
_CONTEXT_AUTOCOMPACT_SAFETY = 2

# Timeout for client.interrupt() during graceful session interruption.
# The SDK default is 60s; we use a shorter timeout to avoid blocking the
# user's next message.  On timeout, the StreamEvent fallback captures the
# session_id so the session can still be resumed.
INTERRUPT_TIMEOUT_SEC: float = 5.0


# ── Debug helpers ────────────────────────────────────────────


def _is_debug_superuser(anima_dir: Path) -> bool:
    """Check if an anima has debug_superuser flag in status.json."""
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return bool(data.get("debug_superuser"))
    except (json.JSONDecodeError, OSError):
        return False


# ── Prompt file cleanup ──────────────────────────────────────


def _cleanup_prompt_files(files: list[Path]) -> None:
    """Remove temp prompt files created for --system-prompt-file."""
    for f in files:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove temp prompt file: %s", f)


# ── Session ID persistence ───────────────────────────────────

# ── Session type constants ────────────────────────────────────

SESSION_TYPE_CHAT = "chat"
SESSION_TYPE_HEARTBEAT = "heartbeat"
SESSION_TYPE_CRON = "cron"
SESSION_TYPE_TASK = "task"
SESSION_TYPE_INBOX = "inbox"

_RESUMABLE_SESSION_TYPES: frozenset[str] = frozenset({SESSION_TYPE_CHAT})
"""Only these session types persist and resume SDK sessions.
All other types start fresh each time (no resume, no save)."""


def _resolve_session_type(trigger: str) -> str:
    """Resolve SDK session type from execution trigger.

    Each trigger category gets its own session namespace to prevent
    cross-contamination of Claude CLI conversation histories.
    """
    if trigger == "heartbeat":
        return SESSION_TYPE_HEARTBEAT
    if trigger.startswith("cron:"):
        return SESSION_TYPE_CRON
    if trigger.startswith("task:"):
        return SESSION_TYPE_TASK
    if trigger.startswith("inbox:"):
        return SESSION_TYPE_INBOX
    return SESSION_TYPE_CHAT


def _session_file(session_type: str, thread_id: str = "default") -> str:
    """Return the session file name for the given session type."""
    if thread_id != "default":
        return f"current_session_{session_type}_{thread_id}.json"
    return f"current_session_{session_type}.json"


def _load_session_id(
    anima_dir: Path,
    session_type: str = "chat",
    thread_id: str = "default",
) -> str | None:
    """Load persisted session ID for SDK session resume.

    Sessions are immortal — no TTL.  The timestamp is retained for
    debug logging only and never used for expiry decisions.
    """
    path = anima_dir / "state" / _session_file(session_type, thread_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        session_id = data.get("session_id")
        if not session_id:
            return None

        ts_str = data.get("timestamp")
        if ts_str:
            saved = datetime.fromisoformat(ts_str)
            if saved.tzinfo is None:
                saved = saved.replace(tzinfo=UTC)
            elapsed_min = (datetime.now(UTC) - saved).total_seconds() / 60
            logger.debug(
                "Session loaded (%s/%s, thread=%s): age %.1f min",
                session_type,
                anima_dir.name,
                thread_id,
                elapsed_min,
            )

        return session_id
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Failed to load session ID from %s: %s", path, exc)
        return None


def _save_session_id(anima_dir: Path, session_id: str, session_type: str = "chat", thread_id: str = "default") -> None:
    """Persist session ID for future SDK session resume."""
    path = anima_dir / "state" / _session_file(session_type, thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _clear_session_id(anima_dir: Path, session_type: str = "chat", thread_id: str = "default") -> None:
    """Clear persisted session ID (e.g., after resume failure)."""
    path = anima_dir / "state" / _session_file(session_type, thread_id)
    if path.exists():
        logger.debug(
            "Clearing session ID (%s/%s, thread=%s)",
            session_type,
            anima_dir.name,
            thread_id,
        )
        path.unlink(missing_ok=True)


# ── SDK query input construction ─────────────────────────────


async def _image_prompt_messages(
    prompt: str,
    images: list[ImageData],
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield a single SDK user message with image content blocks.

    The Agent SDK ``query()`` accepts ``str | AsyncIterable[dict]``.
    When images are present we build an Anthropic-format multimodal
    content block list and wrap it in the SDK's message envelope.
    """
    content_blocks: list[dict[str, Any]] = []
    for img in images:
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"],
                },
            }
        )
    content_blocks.append({"type": "text", "text": prompt})
    yield {
        "type": "user",
        "message": {"role": "user", "content": content_blocks},
        "parent_tool_use_id": None,
    }


def _build_sdk_query_input(
    prompt: str,
    images: list[ImageData] | None,
) -> str | AsyncGenerator[dict[str, Any], None]:
    """Return the appropriate input for ``ClaudeSDKClient.query()``.

    Text-only prompts are passed as plain strings.  When images are
    present, an async generator of Anthropic-format message dicts is
    returned.  Each call produces a fresh generator (they are single-use).
    """
    if images:
        return _image_prompt_messages(prompt, images)
    return prompt


# ── Tool output cleanup ──────────────────────────────────────


def _cleanup_tool_outputs(anima_dir: Path) -> None:
    """Remove temporary tool output files created during the session."""
    tool_output_dir = anima_dir / "shortterm" / "tool_outputs"
    if tool_output_dir.exists():
        shutil.rmtree(tool_output_dir, ignore_errors=True)
        logger.debug("Cleaned up tool output directory: %s", tool_output_dir)
