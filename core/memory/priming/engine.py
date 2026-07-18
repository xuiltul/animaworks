from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""PrimingEngine - slim orchestrator for six-channel memory priming."""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.file_access_policy import find_denied_root, load_denied_roots

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
    channel_e as _channel_e,
)
from core.memory.priming import (
    channel_f as _channel_f,
)
from core.memory.priming import (
    channel_g as _channel_g,
)
from core.memory.priming import (
    outbound as _outbound,
)
from core.memory.priming.constants import (
    _BUDGET_GRAPH_CONTEXT,
    _BUDGET_GREETING,
    _BUDGET_HEARTBEAT,
    _BUDGET_PENDING_TASKS,
    _BUDGET_QUESTION,
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_EPISODES,
    _BUDGET_RELATED_KNOWLEDGE,
    _BUDGET_REQUEST,
    _BUDGET_SENDER_PROFILE,
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_PRIMING_TOKENS,
)
from core.memory.priming.gate import (
    apply_priming_plan,
    build_candidates_from_result,
    build_priming_plan,
)
from core.memory.priming.result import PrimingResult
from core.memory.priming.utils import RetrieverCache, extract_keywords, truncate_head, truncate_tail

logger = logging.getLogger("animaworks.priming")

# TTL (seconds) before a failed MemoryBackend init is retried once.  Prevents a
# transient failure from permanently disabling graph/episode priming.
_BACKEND_INIT_RETRY_TTL_SECONDS = 300.0


class PrimingEngine:
    """Automatic memory priming engine.

    Executes 6-channel parallel memory retrieval:
      A. Sender profile (direct file read)
      B. Recent activity (unified activity log, replaces old episodes + channels)
      C. Related knowledge (dense vector search)
      E. Pending tasks (persistent task queue summary)
      F. Episodes (dense vector search over episode memory)
      G. Graph context (community summaries + recent facts via MemoryBackend)
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
        self._retriever_cache = RetrieverCache()
        self._retriever: Any | None = None
        self._retriever_initialized = False
        self._config_loaded = False
        self._budget_greeting = _BUDGET_GREETING
        self._budget_question = _BUDGET_QUESTION
        self._budget_request = _BUDGET_REQUEST
        self._budget_heartbeat = _BUDGET_HEARTBEAT
        self._heartbeat_context_pct = 0.05
        self._channel_timeout_seconds = 60.0
        self._get_active_parallel_tasks: Callable[[], dict[str, dict]] | None = None
        self._memory_backend: Any | None = None
        self._memory_backend_init_failed = False
        # Monotonic timestamp of the last failed backend init; ``None`` when the
        # latch was set without a timestamp (e.g. tests) → treated as still latched.
        self._memory_backend_init_failed_at: float | None = None

    def _get_or_create_retriever(self):
        """Get or create a retriever instance from the RetrieverCache."""
        if self._retriever is not None:
            return self._retriever
        return self._retriever_cache.get_or_create(self.anima_dir, self.knowledge_dir)

    def _get_retriever(self):
        """Delegate to _get_or_create_retriever (tests may patch either)."""
        return self._get_or_create_retriever()

    def _get_memory_backend(self):
        """Return lazy-initialized MemoryBackend from config.

        Resolution: per-anima status.json → global config → 'legacy'.
        """
        if self._memory_backend is not None:
            return self._memory_backend
        if self._memory_backend_init_failed:
            failed_at = self._memory_backend_init_failed_at
            if failed_at is None or (time.monotonic() - failed_at) < _BACKEND_INIT_RETRY_TTL_SECONDS:
                return None
            # TTL elapsed: fall through to attempt one re-initialization.
        first_failure = not self._memory_backend_init_failed
        try:
            from core.memory.backend.registry import get_backend, resolve_backend_type

            backend_type = resolve_backend_type(self.anima_dir)
            self._memory_backend = get_backend(backend_type, self.anima_dir)
            self._memory_backend_init_failed = False
            self._memory_backend_init_failed_at = None
            return self._memory_backend
        except Exception:
            if first_failure:
                logger.warning("Failed to init MemoryBackend for priming", exc_info=True)
            else:
                logger.debug("Failed to init MemoryBackend for priming (retry)", exc_info=True)
            self._memory_backend_init_failed = True
            self._memory_backend_init_failed_at = time.monotonic()
            return None

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
        try:
            from core.config.models import load_config

            self._channel_timeout_seconds = float(getattr(load_config().priming, "channel_timeout_seconds", 60.0))
        except Exception:
            logger.debug("Failed to load priming channel timeout; using default", exc_info=True)

    async def _run_priming_channel(self, name: str, coro):
        self._load_config_budgets()
        try:
            return await asyncio.wait_for(coro, timeout=self._channel_timeout_seconds)
        except TimeoutError:
            logger.warning(
                "Priming channel %s timed out after %.1fs; degrading channel only",
                name,
                self._channel_timeout_seconds,
            )
            return ""
        except asyncio.CancelledError:
            raise

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
        recent_human_messages: list[str] | None = None,
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
                denied_roots = load_denied_roots(self.anima_dir)
                resolved_state_path = state_path.resolve()
                if find_denied_root(resolved_state_path, denied_roots) is None and resolved_state_path.is_file():
                    effective_message = resolved_state_path.read_text(encoding="utf-8")[:300]
            except (OSError, RuntimeError):
                pass

        keywords = self._extract_keywords(message or effective_message)

        channel_c_coro = self._channel_c_related_knowledge(
            keywords,
            message=effective_message,
            recent_human_messages=recent_human_messages,
            trigger=channel,
        )

        results = await asyncio.gather(
            self._run_priming_channel("A", self._channel_a_sender_profile(sender_name)),
            self._run_priming_channel(
                "B",
                self._channel_b_recent_activity(sender_name, keywords, channel=channel),
            ),
            self._run_priming_channel("C0", self._channel_c0_important_knowledge()),
            self._run_priming_channel("C", channel_c_coro),
            self._run_priming_channel("E", self._channel_e_pending_tasks()),
            self._run_priming_channel("outbound", self._collect_recent_outbound()),
            self._run_priming_channel(
                "F",
                self._channel_f_episodes(
                    keywords,
                    message=message,
                    recent_human_messages=recent_human_messages,
                    trigger=channel,
                ),
            ),
            self._run_priming_channel(
                "pending_human_notifications",
                self._collect_pending_human_notifications(channel=channel),
            ),
            self._run_priming_channel(
                "G",
                self._channel_g_graph_context(effective_message, trigger=channel),
            ),
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

        pending_tasks = results[4] if isinstance(results[4], str) else ""
        recent_outbound = results[5] if isinstance(results[5], str) else ""
        episodes = results[6] if isinstance(results[6], str) else ""
        pending_human_notifications = results[7] if isinstance(results[7], str) else ""
        graph_context = results[8] if isinstance(results[8], str) else ""

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Priming channel %d failed: %s", i, r)

        raw_result = PrimingResult(
            sender_profile=sender_profile,
            recent_activity=recent_activity,
            related_knowledge=related_knowledge,
            related_knowledge_untrusted=related_knowledge_untrusted,
            pending_tasks=pending_tasks,
            recent_outbound=recent_outbound,
            episodes=episodes,
            pending_human_notifications=pending_human_notifications,
            graph_context=graph_context,
        )
        gate_plan = build_priming_plan(
            effective_message,
            channel,
            intent,
            build_candidates_from_result(raw_result),
            recent_human_messages=recent_human_messages,
        )
        gated_result = apply_priming_plan(raw_result, gate_plan)

        budget_ratio = token_budget / _DEFAULT_MAX_PRIMING_TOKENS
        budget_profile = int(_BUDGET_SENDER_PROFILE * budget_ratio)
        budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY * budget_ratio))
        budget_knowledge = int(_BUDGET_RELATED_KNOWLEDGE * budget_ratio)
        budget_tasks = int(_BUDGET_PENDING_TASKS * budget_ratio)
        budget_episodes = int(_BUDGET_RELATED_EPISODES * budget_ratio)

        truncated_knowledge = truncate_head(gated_result.related_knowledge, budget_knowledge)
        knowledge_used_tokens = len(truncated_knowledge) // _CHARS_PER_TOKEN
        remaining_knowledge_budget = max(0, budget_knowledge - knowledge_used_tokens)
        truncated_untrusted = (
            truncate_head(gated_result.related_knowledge_untrusted, remaining_knowledge_budget)
            if remaining_knowledge_budget > 0
            else ""
        )

        budget_graph = int(_BUDGET_GRAPH_CONTEXT * budget_ratio)

        result = PrimingResult(
            sender_profile=truncate_head(gated_result.sender_profile, budget_profile),
            recent_activity=truncate_tail(gated_result.recent_activity, budget_activity),
            related_knowledge=truncated_knowledge,
            related_knowledge_untrusted=truncated_untrusted,
            pending_tasks=truncate_head(gated_result.pending_tasks, budget_tasks),
            recent_outbound=gated_result.recent_outbound,
            episodes=truncate_tail(gated_result.episodes, budget_episodes),
            pending_human_notifications=gated_result.pending_human_notifications,
            graph_context=truncate_tail(gated_result.graph_context, budget_graph),
            gate_plan=gate_plan,
        )

        logger.info(
            "Priming complete: %d chars (~%d tokens), sender_prof=%d, activity=%d, "
            "knowledge=%d, episodes=%d, outbound=%d",
            result.total_chars(),
            result.estimated_tokens(),
            len(result.sender_profile),
            len(result.recent_activity),
            len(result.related_knowledge),
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
        message: str = "",
        recent_human_messages: list[str] | None = None,
        trigger: str = "chat",
    ) -> tuple[str, str]:
        return await _channel_c.channel_c_related_knowledge(
            self.anima_dir,
            self.knowledge_dir,
            self._get_retriever,
            keywords,
            message=message,
            recent_human_messages=recent_human_messages,
            trigger=trigger,
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
        recent_human_messages: list[str] | None = None,
        trigger: str = "chat",
    ) -> str:
        return await _channel_f.channel_f_episodes(
            self.anima_dir,
            self.episodes_dir,
            self._get_retriever,
            keywords,
            message=message,
            recent_human_messages=recent_human_messages,
            get_memory_backend=self._get_memory_backend,
            trigger=trigger,
        )

    async def _collect_pending_human_notifications(self, *, channel: str = "") -> str:
        return await _outbound.collect_pending_human_notifications(self.anima_dir, channel=channel)

    async def _channel_g_graph_context(self, query: str, *, trigger: str = "chat") -> str:
        backend = self._get_memory_backend()
        if backend is None:
            return ""
        return await _channel_g.collect_graph_context(
            backend,
            query,
            budget_tokens=_BUDGET_GRAPH_CONTEXT,
            anima_dir=self.anima_dir,
            trigger=trigger,
        )

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
