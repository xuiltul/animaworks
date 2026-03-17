from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""PrimingEngine - slim orchestrator for 6-channel memory priming."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import submodules directly to avoid circular import when package __init__ loads engine
from core.memory.priming import (
    budget as _budget,
)
from core.memory.priming import (
    channel_a as _channel_a,
)
from core.memory.priming import (
    channel_b as _channel_b,
)
from core.memory.priming import (
    channel_c as _channel_c,
)
from core.memory.priming import (
    channel_d as _channel_d,
)
from core.memory.priming import (
    channel_e as _channel_e,
)
from core.memory.priming import (
    channel_f as _channel_f,
)
from core.memory.priming import (
    outbound as _outbound,
)
from core.memory.priming.constants import (
    _BUDGET_GREETING,
    _BUDGET_HEARTBEAT,
    _BUDGET_PENDING_TASKS,
    _BUDGET_QUESTION,
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_EPISODES,
    _BUDGET_RELATED_KNOWLEDGE,
    _BUDGET_REQUEST,
    _BUDGET_SENDER_PROFILE,
    _BUDGET_SKILL_MATCH,
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_PRIMING_TOKENS,
)
from core.memory.priming.utils import RetrieverCache, extract_keywords, truncate_head, truncate_tail

logger = logging.getLogger("animaworks.priming")


@dataclass
class PrimingResult:
    """Result of priming memory retrieval."""

    sender_profile: str = ""
    recent_activity: str = ""
    related_knowledge: str = ""
    related_knowledge_untrusted: str = ""
    matched_skills: list[str] = field(default_factory=list)
    pending_tasks: str = ""
    recent_outbound: str = ""
    episodes: str = ""
    pending_human_notifications: str = ""

    def is_empty(self) -> bool:
        """Return True if no memories were primed."""
        return (
            not self.sender_profile
            and not self.recent_activity
            and not self.related_knowledge
            and not self.related_knowledge_untrusted
            and not self.matched_skills
            and not self.pending_tasks
            and not self.recent_outbound
            and not self.episodes
            and not self.pending_human_notifications
        )

    def total_chars(self) -> int:
        """Estimate total character count."""
        return (
            len(self.sender_profile)
            + len(self.recent_activity)
            + len(self.related_knowledge)
            + len(self.related_knowledge_untrusted)
            + sum(len(s) for s in self.matched_skills)
            + len(self.pending_tasks)
            + len(self.recent_outbound)
            + len(self.episodes)
            + len(self.pending_human_notifications)
        )

    def estimated_tokens(self) -> int:
        """Estimate token count."""
        return self.total_chars() // _CHARS_PER_TOKEN


class PrimingEngine:
    """Automatic memory priming engine.

    Executes 6-channel parallel memory retrieval:
      A. Sender profile (direct file read)
      B. Recent activity (unified activity log, replaces old episodes + channels)
      C. Related knowledge (dense vector search)
      D. Skill matching (description-based 3-tier match with vector search)
      E. Pending tasks (persistent task queue summary)
      F. Episodes (dense vector search over episode memory)
    """

    def __init__(
        self,
        anima_dir: Path,
        shared_dir: Path | None = None,
        context_window: int = 0,
    ) -> None:
        self.anima_dir = anima_dir
        self.shared_dir = shared_dir
        self.context_window = context_window
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.skills_dir = anima_dir / "skills"
        self._retriever_cache = RetrieverCache()
        self._retriever: Any | None = None
        self._retriever_initialized = False
        self._config_loaded = False
        self._budget_greeting = _BUDGET_GREETING
        self._budget_question = _BUDGET_QUESTION
        self._budget_request = _BUDGET_REQUEST
        self._budget_heartbeat = _BUDGET_HEARTBEAT
        self._heartbeat_context_pct = 0.05
        self._get_active_parallel_tasks: Callable[[], dict[str, dict]] | None = None

    def _get_or_create_retriever(self):
        """Get or create a retriever instance from the RetrieverCache."""
        if self._retriever is not None:
            return self._retriever
        return self._retriever_cache.get_or_create(self.anima_dir, self.knowledge_dir)

    def _get_retriever(self):
        """Delegate to _get_or_create_retriever (tests may patch either)."""
        return self._get_or_create_retriever()

    def _load_config_budgets(self) -> None:
        if self._config_loaded:
            return
        self._config_loaded = True
        (
            self._budget_greeting,
            self._budget_question,
            self._budget_request,
            self._budget_heartbeat,
            self._heartbeat_context_pct,
        ) = _budget.load_config_budgets()

    @staticmethod
    def _extract_summary(content: str, metadata: dict) -> str:
        """Delegate to standalone extract_summary for backward compat."""
        from core.memory.priming.channel_c import extract_summary

        title, _ = extract_summary(content, metadata)
        return title

    @staticmethod
    def _to_read_memory_path(metadata: dict, anima_name: str) -> str:
        """Delegate to standalone to_read_memory_path for backward compat."""
        from core.memory.priming.channel_c import to_read_memory_path

        return to_read_memory_path(metadata, anima_name)

    def _read_shared_channels(self, limit_per_channel: int = 5) -> list:
        """Delegate to standalone read_shared_channels for backward compat."""
        from core.memory.priming.channel_b import read_shared_channels

        return read_shared_channels(self.anima_dir, self.shared_dir, limit_per_channel=limit_per_channel)

    async def _fallback_episodes_and_channels(self) -> str:
        """Delegate to standalone fallback for backward compat."""
        from core.memory.priming.channel_b import fallback_episodes_and_channels

        return await fallback_episodes_and_channels(self.anima_dir, self.shared_dir)

    def _adjust_token_budget(self, message: str, channel: str, *, intent: str = "") -> int:
        self._load_config_budgets()
        return _budget.adjust_token_budget(
            message,
            channel,
            self.context_window,
            intent=intent,
            budget_greeting=self._budget_greeting,
            budget_question=self._budget_question,
            budget_request=self._budget_request,
            budget_heartbeat=self._budget_heartbeat,
            heartbeat_context_pct=self._heartbeat_context_pct,
        )

    async def prime_memories(
        self,
        message: str,
        sender_name: str = "human",
        channel: str = "chat",
        intent: str = "",
        enable_dynamic_budget: bool = False,
        overflow_files: list[str] | None = None,
    ) -> PrimingResult:
        """Prime memories based on incoming message."""
        logger.debug(
            "Priming memories: sender=%s, message_len=%d, channel=%s",
            sender_name,
            len(message),
            channel,
        )

        if enable_dynamic_budget:
            token_budget = self._adjust_token_budget(message, channel, intent=intent)
        else:
            token_budget = _DEFAULT_MAX_PRIMING_TOKENS

        logger.debug("Token budget: %d", token_budget)

        effective_message = message
        if not effective_message.strip():
            state_path = self.anima_dir / "state" / "current_state.md"
            try:
                if state_path.is_file():
                    effective_message = state_path.read_text(encoding="utf-8")[:300]
            except OSError:
                pass

        keywords = self._extract_keywords(message or effective_message)

        if overflow_files is None:
            channel_c_coro = self._channel_c_related_knowledge(keywords, message=effective_message)
        elif overflow_files:
            channel_c_coro = self._channel_c_related_knowledge(
                keywords,
                restrict_to=overflow_files,
                message=effective_message,
            )
        else:

            async def _noop() -> tuple[str, str]:
                return ("", "")

            channel_c_coro = _noop()

        results = await asyncio.gather(
            self._channel_a_sender_profile(sender_name),
            self._channel_b_recent_activity(sender_name, keywords, channel=channel),
            self._channel_c0_important_knowledge(),
            channel_c_coro,
            self._channel_d_skill_match(message, keywords, channel=channel),
            self._channel_e_pending_tasks(),
            self._collect_recent_outbound(),
            self._channel_f_episodes(keywords, message=message),
            self._collect_pending_human_notifications(channel=channel),
            return_exceptions=True,
        )

        sender_profile = results[0] if isinstance(results[0], str) else ""
        recent_activity = results[1] if isinstance(results[1], str) else ""

        important_knowledge = results[2] if isinstance(results[2], str) else ""
        if isinstance(results[3], tuple):
            related_knowledge, related_knowledge_untrusted = results[3]
        else:
            related_knowledge = ""
            related_knowledge_untrusted = ""
        if important_knowledge:
            related_knowledge = (
                f"{important_knowledge}\n\n{related_knowledge}" if related_knowledge else important_knowledge
            )

        matched_skills = results[4] if isinstance(results[4], list) else []
        pending_tasks = results[5] if isinstance(results[5], str) else ""
        recent_outbound = results[6] if isinstance(results[6], str) else ""
        episodes = results[7] if isinstance(results[7], str) else ""
        pending_human_notifications = results[8] if isinstance(results[8], str) else ""

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Priming channel %d failed: %s", i, r)

        budget_ratio = token_budget / _DEFAULT_MAX_PRIMING_TOKENS
        budget_profile = int(_BUDGET_SENDER_PROFILE * budget_ratio)
        budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY * budget_ratio))
        budget_knowledge = int(_BUDGET_RELATED_KNOWLEDGE * budget_ratio)
        budget_skills = int(_BUDGET_SKILL_MATCH * budget_ratio)
        budget_tasks = int(_BUDGET_PENDING_TASKS * budget_ratio)
        budget_episodes = int(_BUDGET_RELATED_EPISODES * budget_ratio)

        truncated_knowledge = truncate_head(related_knowledge, budget_knowledge)
        knowledge_used_tokens = len(truncated_knowledge) // _CHARS_PER_TOKEN
        remaining_knowledge_budget = max(0, budget_knowledge - knowledge_used_tokens)
        truncated_untrusted = (
            truncate_head(related_knowledge_untrusted, remaining_knowledge_budget)
            if remaining_knowledge_budget > 0
            else ""
        )

        result = PrimingResult(
            sender_profile=truncate_head(sender_profile, budget_profile),
            recent_activity=truncate_tail(recent_activity, budget_activity),
            related_knowledge=truncated_knowledge,
            related_knowledge_untrusted=truncated_untrusted,
            matched_skills=matched_skills[: max(1, budget_skills // 50)],
            pending_tasks=truncate_head(pending_tasks, budget_tasks),
            recent_outbound=recent_outbound,
            episodes=truncate_tail(episodes, budget_episodes),
            pending_human_notifications=pending_human_notifications,
        )

        logger.info(
            "Priming complete: %d chars (~%d tokens), sender_prof=%d, activity=%d, "
            "knowledge=%d, skills=%d, episodes=%d, outbound=%d",
            result.total_chars(),
            result.estimated_tokens(),
            len(result.sender_profile),
            len(result.recent_activity),
            len(result.related_knowledge),
            len(result.matched_skills),
            len(result.episodes),
            len(result.recent_outbound),
        )

        return result

    # ── Channel wrappers (delegate to modules; tests may patch these) ────

    async def _channel_a_sender_profile(self, sender_name: str) -> str:
        return await _channel_a.channel_a_sender_profile(self.anima_dir, sender_name)

    async def _channel_b_recent_activity(
        self,
        sender_name: str,
        keywords: list[str],
        *,
        channel: str = "",
    ) -> str:
        return await _channel_b.channel_b_recent_activity(
            self.anima_dir,
            self.shared_dir,
            sender_name,
            keywords,
            channel=channel,
        )

    async def _channel_c0_important_knowledge(self) -> str:
        return await _channel_c.channel_c0_important_knowledge(
            self.anima_dir,
            self.knowledge_dir,
            self._get_retriever,
        )

    async def _channel_c_related_knowledge(
        self,
        keywords: list[str],
        restrict_to: list[str] | None = None,
        message: str = "",
    ) -> tuple[str, str]:
        return await _channel_c.channel_c_related_knowledge(
            self.anima_dir,
            self.knowledge_dir,
            self._get_retriever,
            keywords,
            restrict_to=restrict_to,
            message=message,
        )

    async def _channel_d_skill_match(
        self,
        message: str,
        keywords: list[str],
        channel: str = "chat",
    ) -> list[str]:
        return await _channel_d.channel_d_skill_match(
            self.anima_dir,
            self.skills_dir,
            self._get_retriever,
            message,
            keywords,
            channel=channel,
        )

    async def _channel_e_pending_tasks(self) -> str:
        return await _channel_e.channel_e_pending_tasks(
            self.anima_dir,
            self._get_active_parallel_tasks,
        )

    async def _collect_recent_outbound(self, max_entries: int = 3) -> str:
        return await _outbound.collect_recent_outbound(self.anima_dir, max_entries=max_entries)

    async def _channel_f_episodes(
        self,
        keywords: list[str],
        *,
        message: str = "",
    ) -> str:
        return await _channel_f.channel_f_episodes(
            self.anima_dir,
            self.episodes_dir,
            self._get_retriever,
            keywords,
            message=message,
        )

    async def _collect_pending_human_notifications(self, *, channel: str = "") -> str:
        return await _outbound.collect_pending_human_notifications(self.anima_dir, channel=channel)

    def _extract_keywords(self, message: str) -> list[str]:
        """Backward compat: delegate to utils.extract_keywords."""
        return extract_keywords(message, self.knowledge_dir)

    def _classify_message_type(self, message: str, channel: str, *, intent: str = "") -> str:
        """Backward compat: delegate to budget.classify_message_type."""
        return _budget.classify_message_type(message, channel, intent=intent)

    async def _read_old_channels(self) -> str:
        """Backward compat: delegate to channel_b.read_old_channels."""
        return await _channel_b.read_old_channels(self.anima_dir, self.shared_dir)

    @staticmethod
    def _search_and_merge(retriever, queries, anima_name, *, memory_type, top_k, include_shared=False, min_score=None):
        """Backward compat: delegate to utils.search_and_merge."""
        from core.memory.priming.utils import search_and_merge as _search_and_merge

        return _search_and_merge(
            retriever,
            queries,
            anima_name,
            memory_type=memory_type,
            top_k=top_k,
            include_shared=include_shared,
            min_score=min_score,
        )

    # Backward compatibility: static methods for tests that call PrimingEngine._build_dual_queries etc.
    @staticmethod
    def _build_dual_queries(message: str, keywords: list[str]) -> list[str]:
        from core.memory.priming.utils import build_dual_queries

        return build_dual_queries(message, keywords)

    @staticmethod
    def _meets_min_length(token: str) -> bool:
        from core.memory.priming.utils import meets_min_length

        return meets_min_length(token)
