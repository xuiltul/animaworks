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
from datetime import datetime
from pathlib import Path
from typing import Any

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

def _session_file(session_type: str) -> str:
    """Return the session file name for the given session type."""
    return f"current_session_{session_type}.json"


def _load_session_id(anima_dir: Path, session_type: str = "chat") -> str | None:
    """Load persisted session ID for SDK session resume."""
    path = anima_dir / "state" / _session_file(session_type)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("session_id")
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load session ID from %s", path)
        return None


def _save_session_id(anima_dir: Path, session_id: str, session_type: str = "chat") -> None:
    """Persist session ID for future SDK session resume."""
    path = anima_dir / "state" / _session_file(session_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False), encoding="utf-8")


def _clear_session_id(anima_dir: Path, session_type: str = "chat") -> None:
    """Clear persisted session ID (e.g., after resume failure)."""
    path = anima_dir / "state" / _session_file(session_type)
    if path.exists():
        path.unlink(missing_ok=True)


def clear_session_ids(anima_dir: Path) -> None:
    """Clear all session IDs for an anima (chat and heartbeat).

    Public wrapper for use by streaming_handler.py on done=False disconnection.
    """
    for session_type in ("chat", "heartbeat"):
        _clear_session_id(anima_dir, session_type)


# ── SDK query input construction ─────────────────────────────

async def _image_prompt_messages(
    prompt: str,
    images: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield a single SDK user message with image content blocks.

    The Agent SDK ``query()`` accepts ``str | AsyncIterable[dict]``.
    When images are present we build an Anthropic-format multimodal
    content block list and wrap it in the SDK's message envelope.
    """
    content_blocks: list[dict[str, Any]] = []
    for img in images:
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["data"],
            },
        })
    content_blocks.append({"type": "text", "text": prompt})
    yield {
        "type": "user",
        "message": {"role": "user", "content": content_blocks},
        "parent_tool_use_id": None,
    }


def _build_sdk_query_input(
    prompt: str,
    images: list[dict[str, Any]] | None,
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
