# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

"""Conversation memory (会話記憶 / ワーキングメモリ) management.

Maintains a rolling history of chat turns per DigitalPerson.
When the accumulated history exceeds the configured threshold,
older turns are compressed into an LLM-generated summary while
recent turns are kept verbatim.

Storage: ``{person_dir}/state/conversation.json``
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

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

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.token_estimate:
            self.token_estimate = len(self.content) // _CHARS_PER_TOKEN


@dataclass
class ConversationState:
    """Full conversation state including compressed summary."""

    person_name: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    compressed_summary: str = ""
    compressed_turn_count: int = 0

    @property
    def total_token_estimate(self) -> int:
        summary_tokens = len(self.compressed_summary) // _CHARS_PER_TOKEN
        turn_tokens = sum(t.token_estimate for t in self.turns)
        return summary_tokens + turn_tokens

    @property
    def total_turn_count(self) -> int:
        return len(self.turns) + self.compressed_turn_count


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Manages per-person conversation history with automatic compression."""

    def __init__(
        self,
        person_dir: Path,
        model_config: ModelConfig,
    ) -> None:
        self.person_dir = person_dir
        self.person_name = person_dir.name
        self.model_config = model_config
        self._state_dir = person_dir / "state"
        self._state_path = self._state_dir / "conversation.json"
        self._transcript_dir = person_dir / "transcripts"
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
                    person_name=data.get("person_name", self.person_name),
                    turns=turns,
                    compressed_summary=data.get("compressed_summary", ""),
                    compressed_turn_count=data.get("compressed_turn_count", 0),
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse conversation state; starting fresh")
                self._state = ConversationState(person_name=self.person_name)
        else:
            self._state = ConversationState(person_name=self.person_name)

        return self._state

    def save(self) -> None:
        """Persist current conversation state to disk."""
        state = self.load()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "person_name": state.person_name,
            "turns": [asdict(t) for t in state.turns],
            "compressed_summary": state.compressed_summary,
            "compressed_turn_count": state.compressed_turn_count,
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

    def _append_transcript(self, role: str, content: str, timestamp: str) -> None:
        """Append a message to the permanent daily transcript (JSONL)."""
        try:
            self._transcript_dir.mkdir(parents=True, exist_ok=True)
            date_str = timestamp[:10] if len(timestamp) >= 10 else datetime.now().strftime("%Y-%m-%d")
            if not self._valid_date(date_str):
                date_str = datetime.now().strftime("%Y-%m-%d")
            path = self._transcript_dir / f"{date_str}.jsonl"
            entry = json.dumps(
                {"role": role, "content": content, "timestamp": timestamp},
                ensure_ascii=False,
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            logger.exception("Failed to append transcript for %s", self.person_name)

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

    def append_turn(self, role: str, content: str) -> None:
        """Record a conversation turn."""
        state = self.load()
        turn = ConversationTurn(role=role, content=content)
        state.turns.append(turn)
        self._append_transcript(role, content, turn.timestamp)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._state = ConversationState(person_name=self.person_name)
        if self._state_path.exists():
            self._state_path.unlink()
        logger.info("Conversation memory cleared for %s", self.person_name)

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
            self.person_name,
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

    async def _call_compression_llm(
        self, old_summary: str, new_turns: str
    ) -> str:
        """Call the LLM to produce a compressed conversation summary."""
        import anthropic

        model = self.model_config.fallback_model or self.model_config.model
        api_key = self.model_config.api_key
        if not api_key:
            api_key = os.environ.get(self.model_config.api_key_env)

        client_kwargs: dict[str, str] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.model_config.api_base_url:
            client_kwargs["base_url"] = self.model_config.api_base_url

        client = anthropic.AsyncAnthropic(**client_kwargs)

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

        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )

        return "\n".join(b.text for b in response.content if b.type == "text")

    # ── Session finalization (automatic episode recording) ─────

    async def finalize_session(
        self,
        min_turns: int = 2,
    ) -> bool:
        """Finalize the current conversation session.

        Summarizes the conversation using LLM and appends to episodes/{date}.md.
        This implements immediate encoding (即時符号化) from the design.

        Args:
            min_turns: Minimum number of turns to trigger summarization.
                       Short conversations (greetings) are skipped.

        Returns:
            True if session was finalized and written to episodes/, False if skipped.
        """
        state = self.load()

        # Skip if conversation is too short
        if len(state.turns) < min_turns:
            logger.debug(
                "Session finalization skipped: only %d turns (min %d)",
                len(state.turns),
                min_turns,
            )
            return False

        # Generate summary
        try:
            summary = await self._summarize_session(state.turns)
        except Exception:
            logger.exception("Failed to summarize session; skipping episode write")
            return False

        # Write to episodes/{date}.md
        from core.memory.manager import MemoryManager

        memory_mgr = MemoryManager(self.person_dir)
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M")

        # Extract title from summary (first line, up to 50 chars)
        summary_lines = summary.strip().splitlines()
        title = summary_lines[0][:50] if summary_lines else "会話"

        episode_entry = f"## {time_str} — {title}\n\n{summary}\n"
        memory_mgr.append_episode(episode_entry)

        logger.info(
            "Session finalized: %d turns summarized and written to episodes/%s.md",
            len(state.turns),
            date.today().isoformat(),
        )

        return True

    async def _summarize_session(self, turns: list[ConversationTurn]) -> str:
        """Summarize a conversation session for episode recording.

        Uses a cheap model (fallback_model or main model) to generate
        a structured summary of the conversation.
        """
        import anthropic

        model = self.model_config.fallback_model or self.model_config.model
        api_key = self.model_config.api_key
        if not api_key:
            api_key = os.environ.get(self.model_config.api_key_env)

        client_kwargs: dict[str, str] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.model_config.api_base_url:
            client_kwargs["base_url"] = self.model_config.api_base_url

        client = anthropic.AsyncAnthropic(**client_kwargs)

        # Format turns for summarization
        conversation_text = self._format_turns_for_compression(turns)

        system = (
            "あなたは会話記録の要約者です。以下の会話をエピソード記憶として記録するための要約を作成してください。\n\n"
            "出力形式:\n"
            "**相手**: {相手の名前}\n"
            "**トピック**: {主なトピック、カンマ区切り}\n"
            "**要点**:\n"
            "- {要点1}\n"
            "- {要点2}\n"
            "**決定事項**: {決定事項があれば}\n"
            "**未解決**: {未解決事項があれば}\n\n"
            "日本語で、簡潔に、事実を中心に記述してください。"
        )

        response = await client.messages.create(
            model=model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": conversation_text}],
        )

        return "\n".join(b.text for b in response.content if b.type == "text")