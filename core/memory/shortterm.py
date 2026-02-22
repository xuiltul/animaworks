# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Short-term memory (短期記憶) management.

Handles writing and reading transient session state to the
``{anima_dir}/shortterm/`` folder.  This state bridges across
session restarts when the context window threshold is crossed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.time_utils import now_jst

logger = logging.getLogger("animaworks.shortterm_memory")

# Maximum characters for accumulated_response in the markdown dump.
_MAX_RESPONSE_CHARS = 8000


@dataclass
class SessionState:
    """State captured from a session for externalization."""

    session_id: str = ""
    timestamp: str = ""
    trigger: str = ""
    original_prompt: str = ""
    accumulated_response: str = ""
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    context_usage_ratio: float = 0.0
    turn_count: int = 0
    notes: str = ""


@dataclass
class StreamCheckpoint:
    """Checkpoint captured during streaming execution for retry on disconnect.

    Records completed tool calls and accumulated text so that a retry
    can resume from where the stream was interrupted.
    """

    timestamp: str = ""
    trigger: str = ""
    original_prompt: str = ""
    completed_tools: list[dict[str, Any]] = field(default_factory=list)
    accumulated_text: str = ""
    retry_count: int = 0


class ShortTermMemory:
    """Manages the short-term memory folder for a DigitalAnima.

    Folder layout::

        {anima_dir}/shortterm/
          ├── session_state.md    # Human-readable (fed to agent)
          ├── session_state.json  # Machine-readable (for programmatic restore)
          └── archive/            # Completed / superseded states
    """

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.shortterm_dir = anima_dir / "shortterm"
        self._archive_dir = self.shortterm_dir / "archive"

    # ── Query ───────────────────────────────────────────────

    def has_pending(self) -> bool:
        """Check if there is an unresolved short-term memory to restore."""
        return (self.shortterm_dir / "session_state.json").exists()

    # ── Save ────────────────────────────────────────────────

    def save(self, state: SessionState) -> Path:
        """Externalize session state to the shortterm folder.

        Returns the path to the saved JSON file.
        """
        self.shortterm_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive any existing state before overwriting
        self._archive_existing()

        # Write JSON
        json_path = self.shortterm_dir / "session_state.json"
        json_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write Markdown
        md_path = self.shortterm_dir / "session_state.md"
        md_path.write_text(self._render_markdown(state), encoding="utf-8")

        logger.info(
            "Short-term memory saved: %.1f%% context, %d turns",
            state.context_usage_ratio * 100,
            state.turn_count,
        )
        return json_path

    def save_if_not_exists(self, state: SessionState) -> Path | None:
        """Save only if the agent did not already write a state file.

        This acts as a framework-side fallback in case the agent
        ignored the hook's ``additionalContext`` instruction.
        """
        # Check if the agent already wrote session_state.md via its Write tool
        agent_wrote = (self.shortterm_dir / "session_state.md").exists()
        if agent_wrote:
            logger.info("Agent already wrote short-term memory; skipping fallback save")
            return None
        return self.save(state)

    # ── Load ────────────────────────────────────────────────

    def load(self) -> SessionState | None:
        """Load the current short-term memory state."""
        json_path = self.shortterm_dir / "session_state.json"
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return SessionState(**data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse short-term memory JSON")
            return None

    def load_markdown(self) -> str:
        """Load the markdown dump directly (for system prompt injection)."""
        md_path = self.shortterm_dir / "session_state.md"
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")
        return ""

    # ── Clear ───────────────────────────────────────────────

    def clear(self) -> None:
        """Archive and clear the current short-term memory."""
        if not self.shortterm_dir.exists():
            return
        self._archive_existing()
        logger.info("Short-term memory cleared")

    # ── Stream checkpoint ─────────────────────────────────

    _CHECKPOINT_FILE = "stream_checkpoint.json"

    def save_checkpoint(self, checkpoint: StreamCheckpoint) -> Path:
        """Persist a streaming checkpoint for retry-on-disconnect."""
        self.shortterm_dir.mkdir(parents=True, exist_ok=True)
        path = self.shortterm_dir / self._CHECKPOINT_FILE
        path.write_text(
            json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(
            "Stream checkpoint saved: %d completed tools, retry=%d",
            len(checkpoint.completed_tools),
            checkpoint.retry_count,
        )
        return path

    def load_checkpoint(self) -> StreamCheckpoint | None:
        """Load the current stream checkpoint, if any."""
        path = self.shortterm_dir / self._CHECKPOINT_FILE
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StreamCheckpoint(**data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse stream checkpoint JSON")
            return None

    def clear_checkpoint(self) -> None:
        """Remove the stream checkpoint file."""
        path = self.shortterm_dir / self._CHECKPOINT_FILE
        if path.exists():
            path.unlink()
            logger.debug("Stream checkpoint cleared")

    # ── Private ─────────────────────────────────────────────

    def _archive_existing(self) -> None:
        """Move existing session_state files to archive/."""
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        ts = now_jst().strftime("%Y%m%d_%H%M%S")
        for suffix in (".json", ".md"):
            src = self.shortterm_dir / f"session_state{suffix}"
            if src.exists():
                src.rename(self._archive_dir / f"{ts}{suffix}")
        self._prune_archive()

    def _prune_archive(self, max_files: int = 100) -> None:
        """Remove oldest archive files when count exceeds limit."""
        if not self._archive_dir.exists():
            return
        files = sorted(self._archive_dir.iterdir(), key=lambda f: f.name)
        excess = len(files) - max_files
        if excess > 0:
            for f in files[:excess]:
                f.unlink()
            logger.debug("Pruned %d old archive files", excess)

    def _render_markdown(self, state: SessionState) -> str:
        """Render a human-readable markdown dump of the session state."""
        # Truncate accumulated_response if too long
        response = state.accumulated_response
        if len(response) > _MAX_RESPONSE_CHARS:
            response = "...(前半省略)...\n" + response[-_MAX_RESPONSE_CHARS:]

        # Tool use summary (last 20)
        tool_lines = ""
        if state.tool_uses:
            entries = []
            for tu in state.tool_uses[-20:]:
                name = tu.get("name", "?")
                inp = str(tu.get("input", ""))[:500]
                entries.append(f"- {name}: {inp}")
                result = str(tu.get("result", ""))[:500]
                if result:
                    entries.append(f"  → {result}")
            tool_lines = "\n".join(entries)

        return f"""\
# 短期記憶（セッション引き継ぎ）

## メタ情報
- セッションID: {state.session_id}
- 時刻: {state.timestamp}
- トリガー: {state.trigger}
- コンテキスト使用率: {state.context_usage_ratio:.0%}
- ターン数: {state.turn_count}

## 元の依頼
{state.original_prompt}

## これまでの作業内容
{response}

## 使用したツール（直近）
{tool_lines or "(なし)"}

## 補足
{state.notes or "(なし)"}
"""