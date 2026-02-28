from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-log constants and helpers extracted from ``core.agent``.

Pure module-level functions (no class).  ``core.agent`` re-exports every
public symbol so existing ``from core.agent import _save_prompt_log``
continues to work.
"""

import json as _json
import logging
from datetime import timedelta
from pathlib import Path

from core.time_utils import now_iso, now_jst

logger = logging.getLogger("animaworks.agent")

# ── Prompt size guards ──────────────────────────────────────────
# Agent SDK uses JSON-RPC with a default 1 MB buffer (now raised to 4 MB via
# max_buffer_size).  These thresholds trigger defensive actions well before
# the hard limit is hit.  JSON framing + tool schemas add ~30-50% overhead
# on top of the raw text, so we use conservative byte limits.
_PROMPT_SOFT_LIMIT_BYTES = 600_000   # Force compression
_PROMPT_HARD_LIMIT_BYTES = 1_200_000  # Fall back to S Fallback


_PROMPT_LOG_RETENTION_DAYS = 3
_last_rotation_date: str = ""


def _rotate_prompt_logs(log_dir: Path) -> None:
    """Delete prompt_log files older than *_PROMPT_LOG_RETENTION_DAYS*.

    Uses the filename date (``YYYY-MM-DD.jsonl``) for comparison so no
    filesystem stat is required.  Runs at most once per calendar day
    (module-level ``_last_rotation_date`` cache).
    """
    global _last_rotation_date
    today = now_jst().strftime("%Y-%m-%d")
    if _last_rotation_date == today:
        return  # already rotated today
    _last_rotation_date = today

    cutoff = now_jst() - timedelta(days=_PROMPT_LOG_RETENTION_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    for f in log_dir.glob("*.jsonl"):
        # Filename expected format: YYYY-MM-DD.jsonl
        date_str = f.stem
        if date_str < cutoff_str:
            f.unlink(missing_ok=True)


def _save_prompt_log(
    anima_dir: Path,
    *,
    trigger: str,
    sender: str,
    model: str,
    mode: str,
    system_prompt: str,
    user_message: str,
    tools: list[str],
    session_id: str,
    context_window: int = 0,
    prior_messages: list | None = None,
    tool_schemas: list | None = None,
) -> None:
    """Persist the full prompt payload to a JSONL log for post-hoc debugging.

    Writes to ``{anima_dir}/prompt_logs/{date}.jsonl``.
    Failures are silently logged -- prompt logging must never break execution.
    """
    try:
        log_dir = anima_dir / "prompt_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Auto-rotate old log files (at most once per day)
        _rotate_prompt_logs(log_dir)

        today = now_iso()[:10]  # YYYY-MM-DD
        entry = {
            "ts": now_iso(),
            "type": "request_start",
            "trigger": trigger,
            "from": sender,
            "model": model,
            "mode": mode,
            "system_prompt_length": len(system_prompt),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "tools": tools,
            "session_id": session_id,
            "context_window": context_window,
            "prior_messages": prior_messages,
            "prior_messages_count": len(prior_messages) if prior_messages else 0,
            "tool_schemas": tool_schemas,
        }
        log_file = log_dir / f"{today}.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        logger.debug("Prompt log saved: %s (%d bytes)", log_file, len(system_prompt))
    except Exception:
        logger.warning("Failed to save prompt log", exc_info=True)


def _save_prompt_log_end(
    anima_dir: Path,
    session_id: str,
    final_messages: list[dict] | None = None,
    tool_call_count: int = 0,
    total_tokens_estimate: int = 0,
) -> None:
    """Persist post-execution metadata to the same JSONL log.

    Writes a ``request_end`` entry after the tool loop completes, capturing
    final message counts and token estimates.
    """
    try:
        log_dir = anima_dir / "prompt_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = now_iso()[:10]
        log_file = log_dir / f"{today}.jsonl"

        entry = {
            "ts": now_iso(),
            "type": "request_end",
            "session_id": session_id,
            "final_messages_count": len(final_messages) if final_messages else 0,
            "final_messages": final_messages,
            "tool_call_count": tool_call_count,
            "total_tokens_estimate": total_tokens_estimate,
        }
        with log_file.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.warning("Failed to save prompt log end", exc_info=True)
