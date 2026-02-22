# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Conversation memory (会話記憶 / ワーキングメモリ) management.

Maintains a rolling history of chat turns per DigitalAnima.
When the accumulated history exceeds the configured threshold,
older turns are compressed into an LLM-generated summary while
recent turns are kept verbatim.

Storage: ``{anima_dir}/state/conversation.json``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from core.time_utils import ensure_aware, now_iso, now_jst
from typing import TYPE_CHECKING, Any

from core.memory._io import atomic_write_text
from core.paths import load_prompt
from core.schemas import ModelConfig

if TYPE_CHECKING:
    from core.memory.manager import MemoryManager

logger = logging.getLogger("animaworks.conversation_memory")

# ── Error detection patterns for auto-tracking ───────────────

_ERROR_PATTERN = re.compile(
    r"\b(error|failed)\b|エラー[がはをの]|失敗し[たてま]",
    re.IGNORECASE,
)
_RESOLVED_PATTERN = re.compile(
    r"(fixed|resolved|解決|修正済み|成功)",
    re.IGNORECASE,
)

# Truncate assistant responses to this length in the history display.
_MAX_RESPONSE_CHARS_IN_HISTORY = 2500

# Truncate human messages to this length in the history display.
# Human messages are typically short (questions/instructions); long
# command outputs pasted verbatim are truncated to prevent context bloat.
_MAX_HUMAN_CHARS_IN_HISTORY = 800

# Hard cap on content stored in conversation.json per turn.
# Prevents unbounded file growth even before compression kicks in.
_MAX_STORED_CONTENT_CHARS = 5000

# Rough characters-per-token for estimation (conservative for Japanese).
_CHARS_PER_TOKEN = 4

# Maximum number of turns to include in the chat prompt.
_MAX_DISPLAY_TURNS = 15

# Trigger compression when stored turns exceed this count,
# regardless of token estimate.  Prevents conversation.json bloat
# even when the token-budget heuristic underestimates.
_MAX_TURNS_BEFORE_COMPRESS = 50

# ── Tool record limits ─────────────────────────────────────
_MAX_TOOL_INPUT_SUMMARY = 500
_MAX_TOOL_RESULT_SUMMARY = 2000
_MAX_TOOL_RECORDS_PER_TURN = 10
_MAX_RENDERED_TOOL_RECORDS = 50


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolRecord:
    """A single tool call + result pair stored in conversation memory."""

    tool_name: str
    tool_id: str = ""
    input_summary: str = ""   # max _MAX_TOOL_INPUT_SUMMARY chars
    result_summary: str = ""  # max _MAX_TOOL_RESULT_SUMMARY chars
    is_error: bool = False

    def __post_init__(self) -> None:
        if len(self.input_summary) > _MAX_TOOL_INPUT_SUMMARY:
            self.input_summary = self.input_summary[:_MAX_TOOL_INPUT_SUMMARY] + "..."
        if len(self.result_summary) > _MAX_TOOL_RESULT_SUMMARY:
            self.result_summary = self.result_summary[:_MAX_TOOL_RESULT_SUMMARY] + "..."

    @classmethod
    def from_dict(cls, d: dict) -> "ToolRecord":
        """Create a ToolRecord from a dict (e.g., from CycleResult)."""
        return cls(
            tool_name=d.get("tool_name", ""),
            tool_id=d.get("tool_id", ""),
            input_summary=d.get("input_summary", ""),
            result_summary=d.get("result_summary", ""),
            is_error=d.get("is_error", False),
        )


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "human" or "assistant"
    content: str
    timestamp: str = ""
    token_estimate: int = 0
    attachments: list[str] = field(default_factory=list)
    tool_records: list[ToolRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_iso()
        if not self.token_estimate:
            self.token_estimate = len(self.content) // _CHARS_PER_TOKEN


@dataclass
class ConversationState:
    """Full conversation state including compressed summary."""

    anima_name: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    compressed_summary: str = ""
    compressed_turn_count: int = 0
    last_finalized_turn_index: int = 0

    @property
    def total_token_estimate(self) -> int:
        summary_tokens = len(self.compressed_summary) // _CHARS_PER_TOKEN
        turn_tokens = sum(t.token_estimate for t in self.turns)
        return summary_tokens + turn_tokens

    @property
    def total_turn_count(self) -> int:
        return len(self.turns) + self.compressed_turn_count


SESSION_GAP_MINUTES = 10


@dataclass
class ParsedSessionSummary:
    """Parsed result of LLM session summary with state changes."""

    title: str
    episode_body: str
    resolved_items: list[str]
    new_tasks: list[str]
    current_status: str
    has_state_changes: bool


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Manages per-anima conversation history with automatic compression."""

    def __init__(
        self,
        anima_dir: Path,
        model_config: ModelConfig,
    ) -> None:
        self.anima_dir = anima_dir
        self.anima_name = anima_dir.name
        self.model_config = model_config
        self._state_dir = anima_dir / "state"
        self._state_path = self._state_dir / "conversation.json"
        self._transcript_dir = anima_dir / "transcripts"
        self._state: ConversationState | None = None

    # ── Context window overrides ────────────────────────────

    def _load_context_window_overrides(self) -> dict[str, int] | None:
        """Load model_context_windows from config.json for context resolution."""
        try:
            from core.config.models import load_config
            config = load_config()
            return config.model_context_windows or None
        except Exception:
            return None

    # ── Load / Save ──────────────────────────────────────────

    def load(self) -> ConversationState:
        """Load conversation state from disk (cached after first read)."""
        if self._state is not None:
            return self._state

        if self._state_path.exists():
            try:
                data = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
                turns = []
                for t in data.get("turns", []):
                    raw_records = t.get("tool_records", [])
                    filtered = {k: v for k, v in t.items() if k != "tool_records"}
                    turn = ConversationTurn(**filtered)
                    turn.tool_records = [
                        ToolRecord(**r) for r in raw_records
                    ]
                    turns.append(turn)
                self._state = ConversationState(
                    anima_name=data.get("anima_name", self.anima_name),
                    turns=turns,
                    compressed_summary=data.get("compressed_summary", ""),
                    compressed_turn_count=data.get("compressed_turn_count", 0),
                    last_finalized_turn_index=data.get("last_finalized_turn_index", 0),
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse conversation state; starting fresh")
                self._state = ConversationState(anima_name=self.anima_name)
        else:
            self._state = ConversationState(anima_name=self.anima_name)

        return self._state

    def save(self) -> None:
        """Persist current conversation state to disk."""
        state = self.load()
        data = {
            "anima_name": state.anima_name,
            "turns": [asdict(t) for t in state.turns],
            "compressed_summary": state.compressed_summary,
            "compressed_turn_count": state.compressed_turn_count,
            "last_finalized_turn_index": state.last_finalized_turn_index,
        }
        atomic_write_text(
            self._state_path,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    # ── Pending injected procedures ─────────────────────────────

    @property
    def _pending_procedures_path(self) -> Path:
        return self._state_dir / "pending_procedures.json"

    def store_injected_procedures(
        self,
        procedures: list[Path],
        session_id: str = "",
    ) -> None:
        """Persist injected procedure paths for later finalization.

        Called by the agent after ``build_system_prompt()`` so that
        ``finalize_if_session_ended()`` (triggered by heartbeat) can
        pass them to ``finalize_session()``.
        """
        if not procedures:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "procedures": [str(p) for p in procedures],
            "session_id": session_id,
        }
        self._pending_procedures_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_pending_procedures(self) -> tuple[list[Path], str]:
        """Load and clear pending procedure info.

        Returns:
            Tuple of (procedure paths, session_id).  Empty if none pending.
        """
        path = self._pending_procedures_path
        if not path.exists():
            return [], ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            procedures = [Path(p) for p in data.get("procedures", [])]
            session_id = data.get("session_id", "")
            path.unlink(missing_ok=True)
            return procedures, session_id
        except (json.JSONDecodeError, TypeError):
            path.unlink(missing_ok=True)
            return [], ""

    # ── Transcript ────────────────────────────────────────────

    @staticmethod
    def _valid_date(date: str) -> bool:
        """Return True if *date* looks like YYYY-MM-DD."""
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date))

    def list_transcript_dates(self) -> list[str]:
        """Return sorted list of dates that have transcript files (newest first)."""
        if not self._transcript_dir.exists():
            return []
        return sorted(
            [f.stem for f in self._transcript_dir.glob("*.jsonl")],
            reverse=True,
        )

    def load_transcript(self, date: str) -> list[dict]:
        """Load all messages from a specific date's transcript."""
        if not self._valid_date(date):
            logger.warning("Invalid transcript date format: %s", date)
            return []
        path = self._transcript_dir / f"{date}.jsonl"
        if not path.exists():
            return []
        messages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed transcript line in %s", path)
        return messages

    # ── Mutation ─────────────────────────────────────────────

    def append_turn(
        self,
        role: str,
        content: str,
        attachments: list[str] | None = None,
        tool_records: list[ToolRecord] | None = None,
    ) -> None:
        """Record a conversation turn.

        Transcript recording has been replaced by the unified activity log
        (core.memory.activity.ActivityLogger).  The conversation.json state
        is still maintained for prompt building and compression.
        """
        state = self.load()
        # Truncate excessively long content at storage time to prevent
        # conversation.json bloat (70KB+ with 55 turns caused Agent SDK crash).
        if len(content) > _MAX_STORED_CONTENT_CHARS:
            logger.info(
                "Truncating %s turn content from %d to %d chars",
                role, len(content), _MAX_STORED_CONTENT_CHARS,
            )
            content = content[:_MAX_STORED_CONTENT_CHARS] + f"\n[...truncated, original {len(content)} chars]"
        # Cap tool records per turn
        records = tool_records or []
        if len(records) > _MAX_TOOL_RECORDS_PER_TURN:
            records = records[:_MAX_TOOL_RECORDS_PER_TURN]
        turn = ConversationTurn(
            role=role, content=content,
            attachments=attachments or [],
            tool_records=records,
        )
        state.turns.append(turn)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._state = ConversationState(anima_name=self.anima_name)
        if self._state_path.exists():
            self._state_path.unlink()
        logger.info("Conversation memory cleared for %s", self.anima_name)

    # ── Prompt building ──────────────────────────────────────

    def build_chat_prompt(
        self,
        content: str,
        from_person: str = "human",
        max_history_chars: int | None = None,
    ) -> str:
        """Build the user prompt with conversation history injected.

        Args:
            content: The current user message.
            from_person: Sender identifier.
            max_history_chars: If set, truncate the rendered history text
                to at most this many characters (tail-preserving).
        """
        state = self.load()
        history_block = self._format_history(state, max_chars=max_history_chars)

        if history_block:
            return load_prompt(
                "chat_message_with_history",
                conversation_history=history_block,
                from_person=from_person,
                content=content,
            )
        else:
            return load_prompt(
                "chat_message",
                from_person=from_person,
                content=content,
            )

    def build_structured_messages(
        self,
        content: str,
        fmt: str = "openai",
    ) -> list[dict[str, Any]]:
        """Build structured message history for A2/Fallback modes.

        Preserves tool_use/tool_result structure to prevent the LLM
        from learning to describe tool calls in text instead of actually
        calling them.

        Args:
            content: The current user message content.
            fmt: Message format — ``"openai"`` for LiteLLM/OpenAI-style
                or ``"anthropic"`` for native Anthropic API format.

        Returns:
            List of message dicts ready to pass to the LLM API.
        """
        state = self.load()
        messages: list[dict[str, Any]] = []

        # Compressed summary as context
        if state.compressed_summary:
            messages.append({
                "role": "user",
                "content": (
                    f"[会話の要約（{state.compressed_turn_count}ターン分）]"
                    f"\n\n{state.compressed_summary}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "承知しました。これまでの会話内容を把握しました。",
            })

        # Track total tool records rendered
        rendered_tool_count = 0
        display_turns = state.turns[-_MAX_DISPLAY_TURNS:]

        # Track pending tool_result blocks for Anthropic format merging
        pending_tool_results: list[dict[str, Any]] = []

        for turn in display_turns:
            if turn.role == "human":
                display = turn.content
                if len(display) > _MAX_HUMAN_CHARS_IN_HISTORY:
                    display = display[:_MAX_HUMAN_CHARS_IN_HISTORY] + "..."
                if fmt == "anthropic" and pending_tool_results:
                    # Merge tool_result blocks with this user message
                    merged_content = pending_tool_results + [
                        {"type": "text", "text": display}
                    ]
                    messages.append({"role": "user", "content": merged_content})
                    pending_tool_results = []
                else:
                    messages.append({"role": "user", "content": display})

            elif turn.role == "assistant":
                display = turn.content
                if len(display) > _MAX_RESPONSE_CHARS_IN_HISTORY:
                    display = display[:_MAX_RESPONSE_CHARS_IN_HISTORY] + "..."

                has_tools = (
                    turn.tool_records
                    and rendered_tool_count < _MAX_RENDERED_TOOL_RECORDS
                )

                if has_tools and fmt == "openai":
                    # OpenAI/LiteLLM format
                    tool_calls: list[dict[str, Any]] = []
                    for tr in turn.tool_records:
                        if rendered_tool_count >= _MAX_RENDERED_TOOL_RECORDS:
                            break
                        tool_calls.append({
                            "id": tr.tool_id or f"hist_{rendered_tool_count}",
                            "type": "function",
                            "function": {
                                "name": tr.tool_name,
                                "arguments": json.dumps(
                                    {"_summary": tr.input_summary},
                                    ensure_ascii=False,
                                ),
                            },
                        })
                        rendered_tool_count += 1
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    })
                    # Tool results — match by index (tool_calls and tool_records are built in same order)
                    for i, tc in enumerate(tool_calls):
                        result_text = (
                            turn.tool_records[i].result_summary
                            if i < len(turn.tool_records)
                            else ""
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_text or "(completed)",
                        })

                elif has_tools and fmt == "anthropic":
                    # Anthropic format
                    content_blocks: list[dict[str, Any]] = []
                    if display:
                        content_blocks.append({"type": "text", "text": display})
                    for tr in turn.tool_records:
                        if rendered_tool_count >= _MAX_RENDERED_TOOL_RECORDS:
                            break
                        tid = tr.tool_id or f"hist_{rendered_tool_count}"
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tid,
                            "name": tr.tool_name,
                            "input": {"_summary": tr.input_summary},
                        })
                        pending_tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": tr.result_summary or "(completed)",
                        })
                        rendered_tool_count += 1
                    messages.append({"role": "assistant", "content": content_blocks})

                else:
                    # No tool records or budget exceeded — plain text
                    messages.append({"role": "assistant", "content": display})

        # Current user message (merge with any pending tool_results)
        if pending_tool_results:
            merged_content = pending_tool_results + [
                {"type": "text", "text": content}
            ]
            messages.append({"role": "user", "content": merged_content})
        elif fmt == "anthropic" and messages and messages[-1]["role"] == "user":
            # Merge to avoid consecutive user roles (Anthropic API requirement)
            existing = messages[-1]["content"]
            if isinstance(existing, list):
                existing.append({"type": "text", "text": content})
            else:
                messages[-1]["content"] = [
                    {"type": "text", "text": existing},
                    {"type": "text", "text": content},
                ]
        else:
            messages.append({"role": "user", "content": content})

        return messages

    def _format_history(
        self,
        state: ConversationState,
        max_chars: int | None = None,
    ) -> str:
        """Format conversation history for prompt injection.

        Args:
            state: The conversation state to render.
            max_chars: If set, truncate the rendered text to at most this
                many characters, keeping the tail (most recent turns) and
                prepending an ellipsis marker.
        """
        parts: list[str] = []

        if state.compressed_summary:
            parts.append(
                f"### 会話の要約（{state.compressed_turn_count}ターン分）\n\n"
                f"{state.compressed_summary}"
            )

        if state.turns:
            display_turns = state.turns[-_MAX_DISPLAY_TURNS:]
            turn_lines: list[str] = []
            for t in display_turns:
                # Extract time portion for compact display
                ts = t.timestamp[11:16] if len(t.timestamp) >= 16 else t.timestamp
                role_label = "あなた" if t.role == "assistant" else t.role
                display = t.content
                if t.role == "assistant" and len(display) > _MAX_RESPONSE_CHARS_IN_HISTORY:
                    display = display[:_MAX_RESPONSE_CHARS_IN_HISTORY] + "..."
                elif t.role != "assistant" and len(display) > _MAX_HUMAN_CHARS_IN_HISTORY:
                    display = display[:_MAX_HUMAN_CHARS_IN_HISTORY] + "..."
                if t.role == "assistant" and t.tool_records:
                    tool_names = ", ".join(tr.tool_name for tr in t.tool_records)
                    display += f"\n[実行ツール: {tool_names}]"
                turn_lines.append(f"**[{ts}] {role_label}:**\n{display}")

            if parts:
                parts.append("### 直近の会話\n\n" + "\n\n".join(turn_lines))
            else:
                parts.append("\n\n".join(turn_lines))

        result = "\n\n---\n\n".join(parts) if parts else ""

        if max_chars and len(result) > max_chars:
            result = result[-max_chars:]
            result = "...(前半省略)...\n" + result

        return result

    # ── Compression ──────────────────────────────────────────

    def needs_compression(self) -> bool:
        """Check whether conversation history exceeds the compression threshold."""
        state = self.load()
        if len(state.turns) < 4:
            return False

        # Turn-count trigger: force compression regardless of token estimate
        if len(state.turns) > _MAX_TURNS_BEFORE_COMPRESS:
            return True

        from core.prompt.context import resolve_context_window

        window = resolve_context_window(
            self.model_config.model, self._load_context_window_overrides()
        )

        # Auto-scale threshold for small context models.
        # Formula: min(configured, max(0.10, window / 64000 * 0.30))
        # Results: 128K+ → 0.30, 32K → 0.15, 16K → 0.10
        configured = self.model_config.conversation_history_threshold
        if window < 64_000:
            auto_threshold = max(0.10, window / 64_000 * 0.30)
            effective_threshold = min(configured, auto_threshold)
        else:
            effective_threshold = configured

        threshold_tokens = int(window * effective_threshold)
        return state.total_token_estimate > threshold_tokens

    async def compress_if_needed(self) -> bool:
        """Compress older conversation turns if the threshold is exceeded.

        Returns True if compression was performed.
        """
        if not self.needs_compression():
            return False
        await self._compress()
        return True

    async def _compress(self) -> None:
        """Perform LLM-based compression of older conversation turns."""
        state = self.load()
        if len(state.turns) < 4:
            return

        # Keep a fixed number of recent turns (matches _MAX_DISPLAY_TURNS)
        keep_count = min(_MAX_DISPLAY_TURNS, len(state.turns) - 1)
        to_compress = state.turns[:-keep_count]
        to_keep = state.turns[-keep_count:]

        old_summary = state.compressed_summary
        turn_text = self._format_turns_for_compression(to_compress)

        try:
            summary = await self._call_compression_llm(old_summary, turn_text)
        except Exception:
            logger.exception("Conversation compression failed; keeping raw turns")
            return

        state.turns = to_keep
        state.compressed_summary = summary
        state.compressed_turn_count += len(to_compress)
        self.save()

        logger.info(
            "Conversation compressed for %s: %d turns -> summary (%d chars), "
            "keeping %d recent turns",
            self.anima_name,
            len(to_compress),
            len(summary),
            len(to_keep),
        )

    def _format_turns_for_compression(
        self, turns: list[ConversationTurn]
    ) -> str:
        """Format turns into readable text for the compression prompt."""
        lines: list[str] = []
        for t in turns:
            role = "あなた" if t.role == "assistant" else t.role
            text = f"[{t.timestamp}] {role}: {t.content}"
            if t.tool_records:
                tools = ", ".join(tr.tool_name for tr in t.tool_records)
                text += f"\n  [使用ツール: {tools}]"
            lines.append(text)
        return "\n\n".join(lines)

    async def _call_llm(self, system: str, user_content: str, max_tokens: int = 1000) -> str:
        """Common LLM helper using litellm for provider-agnostic calls."""
        import litellm

        model = self.model_config.fallback_model or self.model_config.model
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def _call_compression_llm(
        self, old_summary: str, new_turns: str
    ) -> str:
        """Call the LLM to produce a compressed conversation summary."""
        system = load_prompt("memory/conversation_compression")

        user_content = ""
        if old_summary:
            user_content += f"## 既存の要約\n\n{old_summary}\n\n---\n\n"
        user_content += f"## 新しい会話ターン\n\n{new_turns}\n\n"
        user_content += "上記を統合した新しい要約を作成してください。"

        return await self._call_llm(system, user_content, max_tokens=2000)

    # ── Session finalization (automatic episode recording) ─────

    async def finalize_session(
        self,
        min_turns: int = 3,
        injected_procedures: list[Path] | None = None,
        session_id: str = "",
    ) -> bool:
        """Finalize the current conversation session (differential).

        Only summarizes turns since last_finalized_turn_index, preventing
        duplicate episode entries. Also extracts state changes and resolution
        information from the conversation.

        Args:
            min_turns: Minimum number of *new* turns to trigger summarization.
            injected_procedures: Procedure paths from ``BuildResult`` for
                auto-outcome tracking.
            session_id: Session identifier for double-count prevention with
                explicit ``report_procedure_outcome`` calls.

        Returns:
            True if session was finalized and written to episodes/, False if skipped.
        """
        state = self.load()

        # Only process turns since last finalization
        new_turns = state.turns[state.last_finalized_turn_index:]
        if len(new_turns) < min_turns:
            logger.debug(
                "Session finalization skipped: only %d new turns (min %d)",
                len(new_turns),
                min_turns,
            )
            return False

        # Gather activity context for richer episode generation
        activity_context = self._gather_activity_context(new_turns)

        # Generate summary with state extraction
        try:
            raw_summary = await self._summarize_session_with_state(
                new_turns, activity_context,
            )
        except Exception:
            logger.exception("Failed to summarize session; skipping episode write")
            return False

        parsed = self._parse_session_summary(raw_summary)

        # 1. Episode recording (differential only)
        from core.memory.manager import MemoryManager

        memory_mgr = MemoryManager(self.anima_dir)
        timestamp = now_jst()
        time_str = timestamp.strftime("%H:%M")
        episode_entry = f"## {time_str} — {parsed.title}\n\n{parsed.episode_body}\n"
        memory_mgr.append_episode(episode_entry)

        # 2. State auto-update (only when parse succeeded)
        if parsed.has_state_changes:
            self._update_state_from_summary(memory_mgr, parsed)

        # 3. Resolution event recording (only when resolved items found)
        if parsed.resolved_items:
            self._record_resolutions(memory_mgr, parsed.resolved_items)

        # 3.5. Auto-track procedure outcomes for injected procedures
        if injected_procedures:
            self._auto_track_procedure_outcomes(
                memory_mgr, new_turns,
                injected_procedures=injected_procedures,
                session_id=session_id,
            )

        # 4. Integrate recorded turns into compressed_summary
        turn_text = self._format_turns_for_compression(new_turns)
        old_summary = state.compressed_summary
        try:
            compressed = await self._call_compression_llm(old_summary, turn_text)
            state.compressed_summary = compressed
        except Exception:
            logger.warning("Compression failed during finalization; keeping raw turns")

        # 5. Update tracking index
        state.last_finalized_turn_index = len(state.turns)
        state.compressed_turn_count += len(new_turns)
        self.save()

        logger.info(
            "Session finalized: %d new turns summarized and written to episodes/%s.md",
            len(new_turns),
            date.today().isoformat(),
        )

        return True

    async def finalize_if_session_ended(self) -> bool:
        """Finalize if session has ended (10-minute idle gap).

        Called from heartbeat to detect session boundaries and trigger
        episode recording for pending conversation turns.

        Returns:
            True if finalization was performed.
        """
        state = self.load()
        if not state.turns:
            return False
        # No unrecorded turns → skip
        new_turns = state.turns[state.last_finalized_turn_index:]
        if not new_turns:
            return False
        last_ts = datetime.fromisoformat(new_turns[-1].timestamp)
        elapsed = (now_jst() - ensure_aware(last_ts)).total_seconds()
        if elapsed < SESSION_GAP_MINUTES * 60:
            return False
        # Load any pending injected procedures stored by the agent
        procedures, session_id = self._load_pending_procedures()
        return await self.finalize_session(
            injected_procedures=procedures or None,
            session_id=session_id,
        )

    def _gather_activity_context(self, turns: list[ConversationTurn]) -> str:
        """Gather non-conversation activities from activity log for episode enrichment.

        Retrieves DM, channel, tool_use, human_notify events that occurred
        during the conversation session timeframe.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)

            # Determine session timeframe from conversation turns
            if not turns:
                return ""
            first_ts = turns[0].timestamp
            last_ts = turns[-1].timestamp

            # Get all non-conversation activities from today
            entries = activity.recent(
                days=1,
                limit=30,
                types=[
                    "dm_sent", "dm_received", "channel_post", "channel_read",
                    "tool_use", "human_notify", "cron_executed",
                ],
            )

            # Filter to session timeframe (between first and last turn)
            session_entries = [
                e for e in entries
                if first_ts <= e.ts <= last_ts
            ]

            if not session_entries:
                return ""

            # Format as context
            lines = ["## セッション中のその他の活動"]
            for e in session_entries:
                text = e.summary or e.content[:100]
                lines.append(f"- [{e.type}] {text}")

            return "\n".join(lines)
        except Exception:
            logger.debug("Failed to gather activity context", exc_info=True)
            return ""

    async def _summarize_session_with_state(
        self, turns: list[ConversationTurn], activity_context: str = "",
    ) -> str:
        """Summarize a conversation session with state change extraction.

        Produces a structured Markdown output with episode summary and
        state change sections for parsing by _parse_session_summary().
        """
        conversation_text = self._format_turns_for_compression(turns)

        system = load_prompt("memory/session_summary")

        user_content = conversation_text
        if activity_context:
            user_content += f"\n\n{activity_context}"

        return await self._call_llm(system, user_content)

    @staticmethod
    def _parse_session_summary(raw: str) -> ParsedSessionSummary:
        """Parse Markdown-formatted LLM output into structured data.

        Falls back to treating the entire raw text as episode body
        when the expected sections are not found.
        """
        # Extract ## エピソード要約 section
        episode_match = re.search(
            r"##\s*エピソード要約\s*\n(.+?)(?=##\s*ステート変更|\Z)",
            raw, re.DOTALL,
        )
        episode_body = episode_match.group(1).strip() if episode_match else raw.strip()

        lines = episode_body.splitlines()
        title = lines[0][:50] if lines else "会話"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else episode_body

        # Extract ## ステート変更 section
        state_match = re.search(
            r"##\s*ステート変更\s*\n(.+)",
            raw, re.DOTALL,
        )

        resolved_items: list[str] = []
        new_tasks: list[str] = []
        current_status = ""

        if state_match:
            state_text = state_match.group(1)

            # ### 解決済み
            resolved_match = re.search(
                r"###\s*解決済み\s*\n(.+?)(?=###|\Z)",
                state_text, re.DOTALL,
            )
            if resolved_match:
                for line in resolved_match.group(1).strip().splitlines():
                    item = line.strip().lstrip("- ").strip()
                    if item and item != "なし":
                        resolved_items.append(item)

            # ### 新規タスク
            tasks_match = re.search(
                r"###\s*新規タスク\s*\n(.+?)(?=###|\Z)",
                state_text, re.DOTALL,
            )
            if tasks_match:
                for line in tasks_match.group(1).strip().splitlines():
                    item = line.strip().lstrip("- ").strip()
                    if item and item != "なし":
                        new_tasks.append(item)

            # ### 現在の状態
            status_match = re.search(
                r"###\s*現在の状態\s*\n(.+?)(?=###|\Z)",
                state_text, re.DOTALL,
            )
            if status_match:
                current_status = status_match.group(1).strip()

        return ParsedSessionSummary(
            title=title,
            episode_body=body,
            resolved_items=resolved_items,
            new_tasks=new_tasks,
            current_status=current_status,
            has_state_changes=bool(resolved_items or new_tasks or current_status),
        )

    def _update_state_from_summary(
        self, memory_mgr: "MemoryManager", parsed: ParsedSessionSummary,
    ) -> None:
        """Auto-update state/current_task.md based on conversation conclusions."""
        current = memory_mgr.read_current_state()
        updated = False

        # Append resolved items with checkmark
        for item in parsed.resolved_items:
            if item not in current:
                marker = f"- ✅ {item}（自動検出: {now_jst().strftime('%m/%d %H:%M')}）"
                current += f"\n{marker}"
                updated = True

        # Append new tasks
        for task in parsed.new_tasks:
            if task not in current:
                current += f"\n- [ ] {task}（自動検出: {now_jst().strftime('%m/%d %H:%M')}）"
                updated = True

        if updated:
            memory_mgr.update_state(current)
            logger.info("State auto-updated from session summary")

    def _auto_track_procedure_outcomes(
        self,
        memory_mgr: MemoryManager,
        new_turns: list[ConversationTurn],
        injected_procedures: list[Path] | None = None,
        session_id: str = "",
    ) -> None:
        """Auto-track outcomes for procedures that were injected during this session.

        Args:
            memory_mgr: MemoryManager for reading/writing procedure metadata.
            new_turns: Conversation turns from the current session.
            injected_procedures: List of procedure paths injected via
                ``BuildResult.injected_procedures``.
            session_id: Session identifier for double-count prevention.
                Procedures already reported via ``report_procedure_outcome``
                with a matching ``_reported_session_id`` are skipped.
        """
        try:
            if not injected_procedures:
                return

            # Only check the LAST assistant turn, not all turns
            assistant_turns = [t for t in new_turns if t.role == "assistant"]
            if assistant_turns:
                last_turn = assistant_turns[-1]
                has_error = bool(_ERROR_PATTERN.search(last_turn.content))
                if has_error and _RESOLVED_PATTERN.search(last_turn.content):
                    has_error = False  # Resolution context overrides error detection
            else:
                has_error = False

            for proc_path in injected_procedures:
                if not proc_path.exists():
                    continue

                meta = memory_mgr.read_procedure_metadata(proc_path)
                if not meta:
                    continue

                # Skip if already reported via explicit tool in this session
                if session_id and meta.get("_reported_session_id") == session_id:
                    logger.debug(
                        "Skipping auto-track for %s: already reported in session %s",
                        proc_path.name, session_id,
                    )
                    continue

                if has_error:
                    meta["failure_count"] = meta.get("failure_count", 0) + 1
                else:
                    meta["success_count"] = meta.get("success_count", 0) + 1

                meta["last_used"] = now_iso()

                s = meta.get("success_count", 0)
                f = meta.get("failure_count", 0)
                meta["confidence"] = s / max(1, s + f)

                body = memory_mgr.read_procedure_content(proc_path)
                memory_mgr.write_procedure_with_meta(proc_path, body, meta)

                logger.debug(
                    "Auto-tracked procedure outcome: %s success=%s confidence=%.2f",
                    proc_path.name, not has_error, meta["confidence"],
                )

        except Exception:
            logger.debug("Failed to auto-track procedure outcomes", exc_info=True)

    def _record_resolutions(
        self, memory_mgr: "MemoryManager", resolved_items: list[str],
    ) -> None:
        """Record resolution events to ActivityLogger and shared registry."""
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(self.anima_dir)

        for item in resolved_items:
            # Layer 1: ActivityLogger issue_resolved event
            try:
                activity.log(
                    "issue_resolved",
                    content=item,
                    summary=f"解決済み: {item[:100]}",
                )
            except Exception:
                logger.debug("Failed to log issue_resolved event", exc_info=True)

            # Layer 3: shared/resolutions.jsonl cross-org record
            try:
                memory_mgr.append_resolution(
                    issue=item,
                    resolver=self.anima_dir.name,
                )
            except Exception:
                logger.debug("Failed to write resolution registry", exc_info=True)
