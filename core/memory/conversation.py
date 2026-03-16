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

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from core.i18n import t
from core.memory._io import atomic_write_text
from core.memory.conversation_compression import (
    _call_compression_llm as _call_compression_llm_fn,
)
from core.memory.conversation_compression import (
    _compress as _compress_fn,
)
from core.memory.conversation_compression import (
    _format_turns_for_compression as _format_turns_for_compression_fn,
)
from core.memory.conversation_compression import (
    compress_if_needed as _compress_if_needed,
)
from core.memory.conversation_compression import (
    needs_compression as _needs_compression,
)
from core.memory.conversation_finalize import (
    _parse_session_summary as _parse_session_summary_fn,
)
from core.memory.conversation_finalize import (
    finalize_if_session_ended as _finalize_if_session_ended,
)
from core.memory.conversation_finalize import (
    finalize_session as _finalize_session,
)
from core.memory.conversation_models import (
    _CHARS_PER_TOKEN,
    _ERROR_PATTERN,
    _MAX_DISPLAY_TURNS,
    _MAX_HUMAN_CHARS_IN_HISTORY,
    _MAX_RENDERED_TOOL_RECORDS,
    _MAX_RESPONSE_CHARS_IN_HISTORY,
    _MAX_STORED_CONTENT_CHARS,
    _MAX_TOOL_INPUT_SUMMARY,
    _MAX_TOOL_RECORDS_PER_TURN,
    _MAX_TOOL_RESULT_SUMMARY,
    _RESOLVED_PATTERN,
    SESSION_GAP_MINUTES,
    ConversationState,
    ConversationTurn,
    ParsedSessionSummary,
    ToolRecord,
)
from core.memory.conversation_prompt import (
    _format_history as _format_history_fn,
)
from core.memory.conversation_prompt import (
    build_chat_prompt as _build_chat_prompt,
)
from core.memory.conversation_prompt import (
    build_structured_messages as _build_structured_messages,
)
from core.memory.conversation_state_update import (
    _record_resolutions as _record_resolutions_fn,
)
from core.memory.conversation_state_update import (
    _update_state_from_summary as _update_state_from_summary_fn,
)
from core.schemas import ModelConfig
from core.time_utils import today_local

if TYPE_CHECKING:
    from core.memory.manager import MemoryManager

logger = logging.getLogger("animaworks.conversation_memory")


class ConversationMemory:
    """Manages per-anima conversation history with automatic compression."""

    _class_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        anima_dir: Path,
        model_config: ModelConfig,
        thread_id: str = "default",
    ) -> None:
        self.anima_dir = anima_dir
        self.anima_name = anima_dir.name
        self.model_config = model_config
        self.thread_id = thread_id
        self._state_dir = anima_dir / "state"
        if thread_id == "default":
            self._state_path = self._state_dir / "conversation.json"
        else:
            conv_dir = self._state_dir / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            self._state_path = conv_dir / f"{thread_id}.json"
        self._transcript_dir = anima_dir / "transcripts"
        self._state: ConversationState | None = None

        _key = f"{anima_dir}:{thread_id}"
        if _key not in self.__class__._class_locks:
            self.__class__._class_locks[_key] = asyncio.Lock()
        self._finalize_lock = self.__class__._class_locks[_key]

    @staticmethod
    async def _call_llm(system: str, user_content: str, max_tokens: int = 1000) -> str:
        """Delegate to standalone _call_llm for backward compat."""
        from core.memory.conversation_compression import _call_llm

        return await _call_llm(system, user_content, max_tokens=max_tokens)

    def _load_context_window_overrides(self) -> dict[str, int] | None:
        try:
            from core.config.models import load_config

            config = load_config()
            return config.model_context_windows or None
        except Exception:
            return None

    def load(self) -> ConversationState:
        if self._state is not None:
            return self._state

        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                turns = []
                for t in data.get("turns", []):
                    raw_records = t.get("tool_records", [])
                    filtered = {k: v for k, v in t.items() if k != "tool_records"}
                    turn = ConversationTurn(**filtered)
                    turn.tool_records = [ToolRecord(**r) for r in raw_records]
                    turns.append(turn)
                self._state = ConversationState(
                    anima_name=data.get("anima_name", self.anima_name),
                    turns=turns,
                    compressed_summary=data.get("compressed_summary", ""),
                    compressed_turn_count=data.get("compressed_turn_count", 0),
                    last_finalized_turn_index=data.get("last_finalized_turn_index", 0),
                )
            except (json.JSONDecodeError, TypeError, OSError):
                logger.warning("Failed to parse conversation state; starting fresh")
                self._state = ConversationState(anima_name=self.anima_name)
        else:
            self._state = ConversationState(anima_name=self.anima_name)

        return self._state

    def save(self) -> None:
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

    @property
    def _pending_procedures_path(self) -> Path:
        return self._state_dir / "pending_procedures.json"

    def store_injected_procedures(self, procedures: list[Path], session_id: str = "") -> None:
        if not procedures:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "procedures": [str(p) for p in procedures],
            "session_id": session_id,
        }
        try:
            self._pending_procedures_path.write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to write pending procedures to %s", self._pending_procedures_path, exc_info=True)

    def _load_pending_procedures(self) -> tuple[list[Path], str]:
        path = self._pending_procedures_path
        if not path.exists():
            return [], ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            procedures = [Path(p) for p in data.get("procedures", [])]
            session_id = data.get("session_id", "")
            path.unlink(missing_ok=True)
            return procedures, session_id
        except OSError:
            logger.warning("Failed to read pending procedures from %s", path, exc_info=True)
            return [], ""
        except (json.JSONDecodeError, TypeError):
            path.unlink(missing_ok=True)
            return [], ""

    @staticmethod
    def _valid_date(date: str) -> bool:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date))

    def list_transcript_dates(self) -> list[str]:
        if not self._transcript_dir.exists():
            return []
        return sorted(
            [f.stem for f in self._transcript_dir.glob("*.jsonl")],
            reverse=True,
        )

    def load_transcript(self, date: str) -> list[dict]:
        if not self._valid_date(date):
            logger.warning("Invalid transcript date format: %s", date)
            return []
        path = self._transcript_dir / f"{date}.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("Failed to read transcript from %s", path, exc_info=True)
            return []
        messages = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed transcript line in %s", path)
        return messages

    def write_transcript(
        self,
        role: str,
        content: str,
        *,
        from_person: str = "",
        thread_id: str = "default",
        attachments: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> None:
        from core.time_utils import now_iso

        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        today = today_local().isoformat()
        path = self._transcript_dir / f"{today}.jsonl"

        entry: dict[str, Any] = {
            "ts": now_iso(),
            "role": role,
            "content": content,
        }
        if from_person:
            entry["from"] = from_person
        if thread_id and thread_id != "default":
            entry["thread_id"] = thread_id
        if attachments:
            entry["attachments"] = attachments
        if tool_names:
            entry["tool_names"] = tool_names

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            logger.warning("Failed to write transcript entry to %s", path, exc_info=True)

    def append_turn(
        self,
        role: str,
        content: str,
        attachments: list[str] | None = None,
        tool_records: list[ToolRecord] | None = None,
    ) -> None:
        state = self.load()
        if len(content) > _MAX_STORED_CONTENT_CHARS:
            logger.info(
                "Truncating %s turn content from %d to %d chars",
                role,
                len(content),
                _MAX_STORED_CONTENT_CHARS,
            )
            content = content[:_MAX_STORED_CONTENT_CHARS] + t("conversation.truncated_suffix", length=len(content))
        records = tool_records or []
        if len(records) > _MAX_TOOL_RECORDS_PER_TURN:
            records = records[:_MAX_TOOL_RECORDS_PER_TURN]
        turn = ConversationTurn(
            role=role,
            content=content,
            attachments=attachments or [],
            tool_records=records,
        )
        state.turns.append(turn)

    def clear(self) -> None:
        self._state = ConversationState(anima_name=self.anima_name)
        if self._state_path.exists():
            self._state_path.unlink()
        logger.info("Conversation memory cleared for %s", self.anima_name)

    def build_chat_prompt(
        self,
        content: str,
        from_person: str = "human",
        max_history_chars: int | None = None,
    ) -> str:
        state = self.load()
        return _build_chat_prompt(state, content, from_person, max_history_chars, self.model_config)

    def build_structured_messages(self, content: str, fmt: str = "openai") -> list[dict[str, Any]]:
        state = self.load()
        return _build_structured_messages(state, content, fmt, self.model_config)

    def _format_history(self, state: ConversationState, max_chars: int | None = None) -> str:
        return _format_history_fn(state, max_chars)

    def _format_turns_for_compression(self, turns: list[ConversationTurn]) -> str:
        return _format_turns_for_compression_fn(turns)

    @staticmethod
    def _parse_session_summary(raw: str) -> ParsedSessionSummary:
        return _parse_session_summary_fn(raw)

    def _update_state_from_summary(self, memory_mgr: MemoryManager, parsed: ParsedSessionSummary) -> None:
        _update_state_from_summary_fn(self.anima_dir, memory_mgr, parsed)

    def _record_resolutions(self, memory_mgr: MemoryManager, resolved_items: list[str]) -> None:
        _record_resolutions_fn(self.anima_dir, memory_mgr, resolved_items)

    def _auto_track_procedure_outcomes(
        self,
        memory_mgr: MemoryManager,
        turns: list[ConversationTurn],
        injected_procedures: list[Path] | None = None,
        session_id: str = "",
    ) -> None:
        from core.memory.conversation_state_update import _auto_track_procedure_outcomes

        _auto_track_procedure_outcomes(
            self.anima_dir,
            memory_mgr,
            turns,
            injected_procedures=injected_procedures,
            session_id=session_id,
        )

    def _gather_activity_context(self, turns: list[ConversationTurn]) -> str:
        from core.memory.conversation_finalize import _gather_activity_context

        return _gather_activity_context(self.anima_dir, turns)

    async def _summarize_session_with_state(
        self,
        new_turns: list[ConversationTurn],
        activity_context: str,
    ) -> str:
        from core.memory.conversation_finalize import _summarize_session_with_state

        return await _summarize_session_with_state(new_turns, activity_context)

    async def _call_compression_llm(self, old_summary: str, new_turns: str) -> str:
        return await _call_compression_llm_fn(old_summary, new_turns)

    async def _compress(self) -> None:
        await _compress_fn(self.load(), self.model_config, self.save, self.anima_name)

    def needs_compression(self) -> bool:
        state = self.load()
        return _needs_compression(state, self.model_config, self._load_context_window_overrides)

    async def compress_if_needed(self) -> bool:
        return await _compress_if_needed(
            self.load(),
            self.model_config,
            self._load_context_window_overrides,
            self.save,
            self.anima_name,
        )

    async def finalize_session(
        self,
        min_turns: int = 3,
        injected_procedures: list[Path] | None = None,
        session_id: str = "",
    ) -> bool:
        return await _finalize_session(
            self.anima_dir,
            self.load(),
            self.model_config,
            self.save,
            min_turns=min_turns,
            injected_procedures=injected_procedures,
            session_id=session_id,
        )

    async def finalize_if_session_ended(self) -> bool:
        async def _compress_inner() -> None:
            from core.memory.conversation_compression import _compress

            await _compress(self.load(), self.model_config, self.save, self.anima_name)

        async def _finalize_inner(procedures: list[Path] | None, sid: str) -> bool:
            return await _finalize_session(
                self.anima_dir,
                self.load(),
                self.model_config,
                self.save,
                injected_procedures=procedures,
                session_id=sid,
            )

        return await _finalize_if_session_ended(
            self._finalize_lock,
            self.load,
            self.save,
            self.needs_compression,
            _compress_inner,
            _finalize_inner,
            self._load_pending_procedures,
            self.anima_name,
        )
