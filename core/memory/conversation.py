# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

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
from typing import Any

from core.schemas import ModelConfig

logger = logging.getLogger("animaworks.conversation_memory")

# Truncate assistant responses to this length in the history display.
_MAX_RESPONSE_CHARS_IN_HISTORY = 1500

# Rough characters-per-token for estimation (conservative for Japanese).
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "human" or "assistant"
    content: str
    timestamp: str = ""
    token_estimate: int = 0
    attachments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
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
                turns = [
                    ConversationTurn(**t) for t in data.get("turns", [])
                ]
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
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "anima_name": state.anima_name,
            "turns": [asdict(t) for t in state.turns],
            "compressed_summary": state.compressed_summary,
            "compressed_turn_count": state.compressed_turn_count,
            "last_finalized_turn_index": state.last_finalized_turn_index,
        }
        self._state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
    ) -> None:
        """Record a conversation turn.

        Transcript recording has been replaced by the unified activity log
        (core.memory.activity.ActivityLogger).  The conversation.json state
        is still maintained for prompt building and compression.
        """
        state = self.load()
        turn = ConversationTurn(
            role=role, content=content, attachments=attachments or [],
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
        self, content: str, from_person: str = "human"
    ) -> str:
        """Build the user prompt with conversation history injected."""
        from core.paths import load_prompt

        state = self.load()
        history_block = self._format_history(state)

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

    def _format_history(self, state: ConversationState) -> str:
        """Format conversation history for prompt injection."""
        parts: list[str] = []

        if state.compressed_summary:
            parts.append(
                f"### 会話の要約（{state.compressed_turn_count}ターン分）\n\n"
                f"{state.compressed_summary}"
            )

        if state.turns:
            turn_lines: list[str] = []
            for t in state.turns:
                # Extract time portion for compact display
                ts = t.timestamp[11:16] if len(t.timestamp) >= 16 else t.timestamp
                role_label = "あなた" if t.role == "assistant" else t.role
                display = t.content
                if (
                    t.role == "assistant"
                    and len(display) > _MAX_RESPONSE_CHARS_IN_HISTORY
                ):
                    display = (
                        display[:_MAX_RESPONSE_CHARS_IN_HISTORY] + "..."
                    )
                turn_lines.append(f"**[{ts}] {role_label}:**\n{display}")

            if parts:
                parts.append("### 直近の会話\n\n" + "\n\n".join(turn_lines))
            else:
                parts.append("\n\n".join(turn_lines))

        return "\n\n---\n\n".join(parts) if parts else ""

    # ── Compression ──────────────────────────────────────────

    def needs_compression(self) -> bool:
        """Check whether conversation history exceeds the compression threshold."""
        state = self.load()
        if len(state.turns) < 4:
            return False

        from core.prompt.context import _resolve_context_window

        window = _resolve_context_window(self.model_config.model)
        threshold_tokens = int(
            window * self.model_config.conversation_history_threshold
        )
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

        # Keep the most recent 25% of turns (at least 2)
        keep_count = max(2, len(state.turns) // 4)
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
            lines.append(f"[{t.timestamp}] {role}: {t.content}")
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
        system = (
            "あなたは会話の要約者です。以下の会話を簡潔に要約してください。\n"
            "保持すべき情報:\n"
            "- 議論された主なトピック\n"
            "- 下された判断・合意事項\n"
            "- アクションアイテム・未解決の問題\n"
            "- 重要な事実・数値\n"
            "- 会話の感情的なトーン\n"
            "- 相手の名前・関係性\n\n"
            "不要な情報:\n"
            "- 挨拶・フィラー\n"
            "- 重複する内容\n"
            "- タイムスタンプの詳細\n\n"
            "要約は日本語で、箇条書きで、簡潔に書いてください。"
        )

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
    ) -> bool:
        """Finalize the current conversation session (differential).

        Only summarizes turns since last_finalized_turn_index, preventing
        duplicate episode entries. Also extracts state changes and resolution
        information from the conversation.

        Args:
            min_turns: Minimum number of *new* turns to trigger summarization.

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
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M")
        episode_entry = f"## {time_str} — {parsed.title}\n\n{parsed.episode_body}\n"
        memory_mgr.append_episode(episode_entry)

        # 2. State auto-update (only when parse succeeded)
        if parsed.has_state_changes:
            self._update_state_from_summary(memory_mgr, parsed)

        # 3. Resolution event recording (only when resolved items found)
        if parsed.resolved_items:
            self._record_resolutions(memory_mgr, parsed.resolved_items)

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
        elapsed = (datetime.now() - last_ts).total_seconds()
        if elapsed < SESSION_GAP_MINUTES * 60:
            return False
        return await self.finalize_session()

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

        system = (
            "あなたは会話記録の要約者です。以下の会話をエピソード記憶として記録し、"
            "同時にステート変更を抽出してください。\n\n"
            "出力形式:\n"
            "## エピソード要約\n"
            "{会話の要約タイトル（20文字以内）}\n\n"
            "**相手**: {相手の名前}\n"
            "**トピック**: {主なトピック、カンマ区切り}\n"
            "**要点**:\n"
            "- {要点1}\n"
            "- {要点2}\n\n"
            "**決定事項**: {あれば記載}\n\n"
            "## ステート変更\n"
            "### 解決済み\n"
            "- {解決した課題があればリスト。なければ「なし」}\n"
            "### 新規タスク\n"
            "- {新たに発生したタスク。なければ「なし」}\n"
            "### 現在の状態\n"
            "{「idle」または現在取り組み中の内容}\n"
        )

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
                marker = f"- ✅ {item}（自動検出: {datetime.now().strftime('%m/%d %H:%M')}）"
                current += f"\n{marker}"
                updated = True

        # Append new tasks
        for task in parsed.new_tasks:
            if task not in current:
                current += f"\n- [ ] {task}（自動検出: {datetime.now().strftime('%m/%d %H:%M')}）"
                updated = True

        if updated:
            memory_mgr.update_state(current)
            logger.info("State auto-updated from session summary")

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
