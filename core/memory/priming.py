from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Priming layer - automatic memory retrieval (自動想起).

Implements brain-science-inspired automatic memory activation before agent
execution, reducing the need for explicit search_memory tool calls.

Based on: docs/design/priming-layer-design.md Phase 1
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from core.i18n import t
from core.time_utils import ensure_aware, now_local, today_local
from core.tools._async_compat import run_sync

if TYPE_CHECKING:
    from core.memory.rag.retriever import MemoryRetriever

logger = logging.getLogger("animaworks.priming")

# ── Token budget configuration ────────────────────────────────

# Default maximum tokens for entire priming injection
_DEFAULT_MAX_PRIMING_TOKENS = 2000

# Message type budgets (Phase 3: dynamic budget adjustment)
_BUDGET_GREETING = 500
_BUDGET_QUESTION = 1500
_BUDGET_REQUEST = 3000
_BUDGET_HEARTBEAT = 200

# Channel-specific token budgets (default distribution)
_BUDGET_SENDER_PROFILE = 500
_BUDGET_RECENT_ACTIVITY = 1300  # Unified: old B(600) + E(700)
_BUDGET_RELATED_KNOWLEDGE = 1200
_BUDGET_IMPORTANT_KNOWLEDGE = 300
_BUDGET_SKILL_MATCH = 200
_BUDGET_PENDING_TASKS = 300
_BUDGET_RELATED_EPISODES = 500

# Rough characters-per-token for Japanese/English mixed text
_CHARS_PER_TOKEN = 4

# Pre-compiled regex pattern for language-agnostic keyword extraction
_RE_UNICODE_WORDS = re.compile(r"[\w]+", re.UNICODE)
# Maximum message length to process for keyword extraction
_MAX_KEYWORD_INPUT_LEN = 5000

# Minimal stopwords: only clear function words that pass the length filter.
# Intentionally small — dual-query strategy reduces dependence on keyword quality.
_MINIMAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "had",
        "with",
        "from",
        "this",
        "that",
        "they",
        "been",
        "have",
        "will",
        "would",
        "could",
        "should",
        "about",
        "which",
        "their",
        "there",
        "these",
        "those",
        "being",
        "through",
        "during",
        "before",
        "after",
        "into",
        "の",
        "に",
        "は",
        "を",
        "が",
        "で",
        "と",
        "も",
        "や",
        "へ",
        "より",
        "から",
        "まで",
        "など",
        "について",
        "している",
        "できる",
        "ために",
        "として",
        "における",
        "これ",
        "それ",
        "あれ",
    }
)


# ── Data structures ────────────────────────────────────────────


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


# ── PrimingEngine ──────────────────────────────────────────────


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
        # Lazily initialized by _get_or_create_retriever(); shared by C & D
        self._retriever: MemoryRetriever | None = None
        self._retriever_initialized: bool = False
        # Budget values loaded lazily from config.json
        self._config_loaded: bool = False
        self._budget_greeting: int = _BUDGET_GREETING
        self._budget_question: int = _BUDGET_QUESTION
        self._budget_request: int = _BUDGET_REQUEST
        self._budget_heartbeat: int = _BUDGET_HEARTBEAT
        self._heartbeat_context_pct: float = 0.05
        # Callback to retrieve active parallel tasks (injected by runner)
        self._get_active_parallel_tasks: Callable[[], dict[str, dict]] | None = None

    # ── Main entry point ────────────────────────────────────────

    async def prime_memories(
        self,
        message: str,
        sender_name: str = "human",
        channel: str = "chat",
        intent: str = "",
        enable_dynamic_budget: bool = False,
        overflow_files: list[str] | None = None,
    ) -> PrimingResult:
        """Prime memories based on incoming message.

        Args:
            message: The incoming message text
            sender_name: Name of the sender (for sender profile lookup)
            channel: Message channel (chat, heartbeat, cron, etc.)
            intent: Sender-declared message intent ("delegation" | "report" | "question")
            enable_dynamic_budget: Enable dynamic budget adjustment (Phase 3)

        Returns:
            PrimingResult containing primed memories from all channels
        """
        logger.debug(
            "Priming memories: sender=%s, message_len=%d, channel=%s",
            sender_name,
            len(message),
            channel,
        )

        # Phase 3: Adjust token budget based on message type
        if enable_dynamic_budget:
            token_budget = self._adjust_token_budget(message, channel, intent=intent)
        else:
            token_budget = _DEFAULT_MAX_PRIMING_TOKENS

        logger.debug("Token budget: %d", token_budget)

        # Extract keywords for search (simple rule-based for Phase 1)
        keywords = self._extract_keywords(message)

        # Channel C: conditional based on distilled knowledge injection
        if overflow_files is None:
            # Legacy path: no full injection, run full Channel C
            channel_c_coro = self._channel_c_related_knowledge(
                keywords,
                message=message,
            )
        elif overflow_files:
            # Overflow path: search only among non-injected files
            channel_c_coro = self._channel_c_related_knowledge(
                keywords,
                restrict_to=overflow_files,
                message=message,
            )
        else:
            # All files injected: skip Channel C entirely

            async def _noop() -> tuple[str, str]:
                return ("", "")

            channel_c_coro = _noop()

        # Execute 7 channels + outbound + human notifications in parallel
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

        # Unpack results (handle exceptions gracefully)
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

        # Log exceptions if any
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Priming channel %d failed: %s", i, r)

        # Apply token budget limits (distribute based on budget)
        budget_ratio = token_budget / _DEFAULT_MAX_PRIMING_TOKENS
        budget_profile = int(_BUDGET_SENDER_PROFILE * budget_ratio)
        budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY * budget_ratio))
        budget_knowledge = int(_BUDGET_RELATED_KNOWLEDGE * budget_ratio)
        budget_skills = int(_BUDGET_SKILL_MATCH * budget_ratio)
        budget_tasks = int(_BUDGET_PENDING_TASKS * budget_ratio)
        budget_episodes = int(_BUDGET_RELATED_EPISODES * budget_ratio)

        # Split knowledge budget: trusted/medium gets priority, untrusted gets remainder
        truncated_knowledge = self._truncate_head(related_knowledge, budget_knowledge)
        knowledge_used_tokens = len(truncated_knowledge) // _CHARS_PER_TOKEN
        remaining_knowledge_budget = max(0, budget_knowledge - knowledge_used_tokens)
        truncated_untrusted = (
            self._truncate_head(
                related_knowledge_untrusted,
                remaining_knowledge_budget,
            )
            if remaining_knowledge_budget > 0
            else ""
        )

        result = PrimingResult(
            sender_profile=self._truncate_head(sender_profile, budget_profile),
            recent_activity=self._truncate_tail(recent_activity, budget_activity),
            related_knowledge=truncated_knowledge,
            related_knowledge_untrusted=truncated_untrusted,
            matched_skills=matched_skills[: max(1, budget_skills // 50)],  # ~50 tokens per skill name
            pending_tasks=self._truncate_head(pending_tasks, budget_tasks),
            recent_outbound=recent_outbound,
            episodes=self._truncate_tail(episodes, budget_episodes),
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

    # ── Shared retriever ────────────────────────────────────────

    def _get_or_create_retriever(self) -> MemoryRetriever | None:
        """Return a MemoryRetriever instance, creating one lazily if needed.

        Shared between Channel C (related knowledge) and Channel D (skill
        matching).  Returns ``None`` when RAG dependencies are unavailable
        or the knowledge directory does not exist.

        If ``_retriever`` was injected externally (e.g. in tests), it is
        returned as-is without re-initialization.
        """
        if self._retriever_initialized or self._retriever is not None:
            return self._retriever

        self._retriever_initialized = True

        if not self.knowledge_dir.is_dir():
            return None

        try:
            from core.memory.rag import MemoryRetriever
            from core.memory.rag.indexer import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            anima_name = self.anima_dir.name
            vector_store = get_vector_store(anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, retriever disabled")
                return None
            indexer = MemoryIndexer(vector_store, anima_name, self.anima_dir)
            self._retriever = MemoryRetriever(
                vector_store,
                indexer,
                self.knowledge_dir,
            )
            logger.debug("Shared MemoryRetriever initialized for %s", anima_name)
        except ImportError:
            logger.debug("RAG dependencies not installed, retriever unavailable")
        except Exception as e:
            logger.warning("Failed to initialize MemoryRetriever: %s", e)

        return self._retriever

    # ── Channel implementations ─────────────────────────────────

    async def _channel_a_sender_profile(self, sender_name: str) -> str:
        """Channel A: Direct sender profile lookup.

        Reads shared/users/{sender_name}/index.md if it exists.
        """
        from core.paths import get_shared_dir

        shared_users_dir = get_shared_dir() / "users"
        profile_path = (shared_users_dir / sender_name / "index.md").resolve()
        if not profile_path.is_relative_to(shared_users_dir.resolve()):
            logger.warning("Channel A: path traversal in sender_name=%s", sender_name)
            return ""

        if not profile_path.exists():
            logger.debug("Channel A: No profile found for sender=%s", sender_name)
            return ""

        try:
            content = await run_sync(profile_path.read_text, encoding="utf-8")
            logger.debug(
                "Channel A: Loaded sender profile for %s (%d chars)",
                sender_name,
                len(content),
            )
            return content
        except Exception as e:
            logger.warning("Channel A: Failed to read profile for %s: %s", sender_name, e)
            return ""

    # Event types that are noise for heartbeat/cron priming — tool invocations
    # and heartbeat lifecycle events crowd out actionable messages.
    _HEARTBEAT_NOISE_TYPES = frozenset(
        {
            "tool_use",
            "tool_result",
            "heartbeat_start",
            "heartbeat_end",
            "heartbeat_reflection",
            "inbox_processing_start",
            "inbox_processing_end",
        }
    )

    # Event types to exclude from chat priming — cron task results should not
    # leak into chat sessions (prevents Anima from discussing cron output
    # during user conversations).
    _CHAT_NOISE_TYPES = frozenset(
        {
            "cron_executed",
        }
    )

    async def _channel_b_recent_activity(
        self,
        sender_name: str,
        keywords: list[str],
        *,
        channel: str = "",
    ) -> str:
        """Channel B: Recent activity from unified activity log.

        Replaces old Channel B (episodes) and Channel E (shared channels).
        Reads from activity_log/{date}.jsonl for a unified timeline,
        plus shared/channels/*.jsonl for cross-Anima visibility.
        Falls back to episodes/ if activity_log is empty (migration period).

        When *channel* is ``"heartbeat"`` or starts with ``"cron:"``,
        tool_use / tool_result / heartbeat lifecycle events are filtered
        out so that the limited priming budget contains only actionable
        communication events (messages, channel posts, errors, etc.).
        """
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(self.anima_dir)
        entries = activity.recent(days=2, limit=100)

        is_background = channel in ("heartbeat",) or channel.startswith("cron:")

        if is_background and entries:
            entries = [e for e in entries if e.type not in self._HEARTBEAT_NOISE_TYPES]
        elif entries:
            # Chat sessions: filter out cron results to prevent Anima from
            # discussing scheduled task output during user conversations.
            entries = [e for e in entries if e.type not in self._CHAT_NOISE_TYPES]

        # Always read shared channels for cross-Anima visibility
        channel_entries = self._read_shared_channels(limit_per_channel=5)
        entries.extend(channel_entries)

        if entries:
            prioritized = self._prioritize_entries(entries, sender_name, keywords)
            prioritized = prioritized[:50]
            return activity.format_for_priming(prioritized, budget_tokens=1300)

        # Fallback: read old episodes if no activity log exists yet
        return await self._fallback_episodes_and_channels()

    def _read_shared_channels(self, limit_per_channel: int = 5) -> list:
        """Read recent entries from shared channels for cross-Anima visibility.

        Reads shared/channels/*.jsonl and converts entries to ActivityEntry
        format for unified priming display.  Prioritises 24h human posts
        and @mentions.

        Args:
            limit_per_channel: Max entries per channel (latest N).

        Returns:
            List of ActivityEntry from shared channels.
        """
        from datetime import datetime

        from core.memory.activity import ActivityEntry

        if not self.shared_dir:
            return []

        channels_dir = self.shared_dir / "channels"
        if not channels_dir.is_dir():
            return []

        anima_name = self.anima_dir.name
        mention_tag = f"@{anima_name}"
        now = now_local()
        cutoff_24h = now - timedelta(hours=24)

        result: list[ActivityEntry] = []

        try:
            from core.messenger import is_channel_member

            for channel_file in sorted(channels_dir.glob("*.jsonl")):
                channel_name = channel_file.stem

                # ── ACL check: skip channels this Anima cannot access ──
                if not is_channel_member(self.shared_dir, channel_name, anima_name):
                    continue

                try:
                    content = channel_file.read_text(encoding="utf-8")
                except OSError:
                    continue

                lines = content.strip().splitlines()
                if not lines:
                    continue

                # Parse all entries
                all_entries: list[dict] = []
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        all_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

                if not all_entries:
                    continue

                # Select: latest N + 24h human posts + mentions
                selected: list[dict] = []
                seen_indices: set[int] = set()

                # Latest N entries
                for i in range(max(0, len(all_entries) - limit_per_channel), len(all_entries)):
                    if i not in seen_indices:
                        selected.append(all_entries[i])
                        seen_indices.add(i)

                # 24h human posts and mentions
                for i, entry in enumerate(all_entries):
                    if i in seen_indices:
                        continue
                    ts_str = entry.get("ts", "")
                    try:
                        ts = ensure_aware(datetime.fromisoformat(ts_str))
                    except (ValueError, TypeError):
                        continue
                    is_human = entry.get("source") == "human"
                    is_mention = mention_tag in entry.get("text", "")
                    if (is_human and ts >= cutoff_24h) or is_mention:
                        selected.append(entry)
                        seen_indices.add(i)

                # Convert to ActivityEntry
                for entry in selected:
                    result.append(
                        ActivityEntry(
                            ts=entry.get("ts", ""),
                            type="channel_post",
                            content=entry.get("text", ""),
                            summary=entry.get("text", "")[:100],
                            from_person=entry.get("from", ""),
                            channel=channel_name,
                        )
                    )

        except Exception:
            logger.warning("Failed to read shared channels", exc_info=True)

        _MAX_CHANNEL_ENTRIES = 15
        if len(result) > _MAX_CHANNEL_ENTRIES:
            result.sort(key=lambda e: e.ts, reverse=True)
            result = result[:_MAX_CHANNEL_ENTRIES]

        return result

    _OWN_ACTION_TYPES = frozenset(
        {
            "message_sent",
            "response_sent",
            "message_received",
            "tool_use",
        }
    )

    def _prioritize_entries(
        self,
        entries: list,  # list[ActivityEntry]
        sender_name: str,
        keywords: list[str],
    ) -> list:
        """Prioritize activity entries for priming.

        Priority order:
        1. Own actions (message_sent, response_sent, message_received, tool_use)
        2. Entries involving the current sender (most relevant)
        3. Entries matching keywords (topically relevant)
        4. Most recent entries (temporal relevance, timestamp-based)
        """
        from datetime import datetime

        from core.memory.activity import ActivityEntry

        keywords_lower = {kw.lower() for kw in keywords} if keywords else set()

        # Compute base timestamp for recency scoring
        base_ts: datetime | None = None
        if entries:
            try:
                base_ts = datetime.fromisoformat(entries[0].ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        scored: list[tuple[float, int, ActivityEntry]] = []
        for i, entry in enumerate(entries):
            score = 0.0

            # Own action bonus
            if entry.type in self._OWN_ACTION_TYPES:
                if entry.type == "message_received":
                    from_type = (entry.meta or {}).get("from_type", "")
                    if from_type != "anima":
                        score += 15.0
                    else:
                        origin_chain = (entry.meta or {}).get("origin_chain") or []
                        if "human" in origin_chain:
                            score += 15.0
                else:
                    score += 15.0

            # Sender relevance
            if entry.from_person == sender_name or entry.to_person == sender_name:
                score += 10.0

            # Keyword relevance
            text = (entry.content + " " + entry.summary).lower()
            matching_kw = sum(1 for kw in keywords_lower if kw in text)
            score += matching_kw * 3.0

            # Recency (timestamp-based: 1 point per 10 minutes)
            if base_ts is not None:
                try:
                    entry_ts = datetime.fromisoformat(entry.ts.replace("Z", "+00:00"))
                    elapsed_seconds = (entry_ts - base_ts).total_seconds()
                    score += elapsed_seconds / 600
                except (ValueError, AttributeError):
                    score += i * 0.1
            else:
                score += i * 0.1

            scored.append((score, i, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return entries in chronological order among the top-scored
        top_entries = [e for _, _, e in scored[:50]]
        top_entries.sort(key=lambda e: e.ts)
        return top_entries

    async def _fallback_episodes_and_channels(self) -> str:
        """Fallback: read old episodes + channels when activity_log is empty."""
        parts: list[str] = []

        # Old Channel B logic
        episodes = await self._read_old_episodes()
        if episodes:
            parts.append(episodes)

        # Old Channel E logic
        channels = await self._read_old_channels()
        if channels:
            parts.append(channels)

        return "\n\n---\n\n".join(parts) if parts else ""

    async def _read_old_episodes(self) -> str:
        """Read old episode files (migration fallback for Channel B)."""
        if not self.episodes_dir.is_dir():
            return ""

        parts: list[str] = []
        today = today_local()

        for offset in range(2):
            target_date = today - timedelta(days=offset)
            path = self.episodes_dir / f"{target_date.isoformat()}.md"

            if not path.exists():
                continue

            try:
                content = await run_sync(path.read_text, encoding="utf-8")
                lines = content.strip().splitlines()
                if len(lines) > 30:
                    lines = lines[-30:]
                parts.append("\n".join(lines))
            except Exception as e:
                logger.warning("Channel B fallback: Failed to read episode %s: %s", path, e)

        if not parts:
            return ""

        return "\n\n---\n\n".join(parts)

    async def _read_old_channels(self) -> str:
        """Read old shared channel files (migration fallback for Channel E)."""
        if not self.shared_dir:
            return ""

        from datetime import datetime

        channels_dir = self.shared_dir / "channels"
        if not channels_dir.is_dir():
            return ""

        anima_name = self.anima_dir.name
        mention_tag = f"@{anima_name}"
        now = now_local()
        cutoff_24h = now - timedelta(hours=24)

        parts: list[str] = []

        for channel_name in ("general", "ops"):
            channel_file = channels_dir / f"{channel_name}.jsonl"
            if not channel_file.exists():
                continue

            try:
                content = await run_sync(channel_file.read_text, encoding="utf-8")
            except OSError:
                continue

            lines = content.strip().splitlines()
            if not lines:
                continue

            entries: list[dict] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not entries:
                continue

            selected: list[dict] = []
            seen_indices: set[int] = set()

            for i in range(max(0, len(entries) - 5), len(entries)):
                if i not in seen_indices:
                    selected.append(entries[i])
                    seen_indices.add(i)

            for i, entry in enumerate(entries):
                if i in seen_indices:
                    continue
                ts_str = entry.get("ts", "")
                try:
                    ts = ensure_aware(datetime.fromisoformat(ts_str))
                except (ValueError, TypeError):
                    continue
                is_human = entry.get("source") == "human"
                is_mention = mention_tag in entry.get("text", "")
                if (is_human and ts >= cutoff_24h) or is_mention:
                    selected.append(entry)
                    seen_indices.add(i)

            selected.sort(key=lambda e: e.get("ts", ""))

            channel_parts: list[str] = []
            for entry in selected:
                src = entry.get("source", "anima")
                marker = " [human]" if src == "human" else ""
                channel_parts.append(
                    f"[{entry.get('ts', '?')}] {entry.get('from', '?')}{marker}: {entry.get('text', '')}"
                )

            if channel_parts:
                parts.append(f"### #{channel_name}")
                parts.extend(channel_parts)
                parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    def _extract_summary(self, content: str, metadata: dict) -> str:
        """Extract summary text from an [IMPORTANT] search result."""
        summary = metadata.get("summary")
        if summary:
            return str(summary).strip()
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        source = metadata.get("source_file", "")
        if source:
            return Path(source).stem.replace("-", " ").replace("_", " ")
        return ""

    def _to_read_memory_path(self, metadata: dict, anima_name: str) -> str:
        """Convert chunk metadata to read_memory_file path."""
        source = metadata.get("source_file", "")
        if not source:
            return ""
        if metadata.get("anima") == "shared":
            return f"common_knowledge/{source}" if not source.startswith("common_knowledge/") else source
        return source

    async def _channel_c0_important_knowledge(self) -> str:
        """Channel C0: Always-prime [IMPORTANT] chunks (summary pointers only)."""
        if not self.knowledge_dir.is_dir():
            return ""
        try:
            retriever = self._get_or_create_retriever()
            if retriever is None:
                return ""
            anima_name = self.anima_dir.name
            results = retriever.get_important_chunks(anima_name, include_shared=True)
            if not results:
                return ""
            budget_chars = _BUDGET_IMPORTANT_KNOWLEDGE * _CHARS_PER_TOKEN
            lines: list[tuple[int, str]] = []
            for r in results:
                content = r.document.content
                meta = r.document.metadata
                summary = self._extract_summary(content, meta)
                rel_path = self._to_read_memory_path(meta, anima_name)
                if not rel_path:
                    continue
                line = f'📌 {summary} → read_memory_file(path="{rel_path}")'
                lines.append((len(line), line))
            lines.sort(key=lambda x: x[0])
            out: list[str] = []
            used = 0
            header = "### [IMPORTANT] Knowledge (summary pointers)"
            header_len = len(header) + 1
            if header_len > budget_chars:
                return ""
            out.append(header)
            used += header_len
            for _, line in lines:
                if used + len(line) + 1 > budget_chars:
                    break
                out.append(line)
                used += len(line) + 1
            if len(out) <= 1:
                return ""
            return "\n".join(out)
        except Exception as e:
            logger.debug("Channel C0: get_important_chunks failed: %s", e)
            return ""

    async def _channel_c_related_knowledge(
        self,
        keywords: list[str],
        restrict_to: list[str] | None = None,
        message: str = "",
    ) -> tuple[str, str]:
        """Channel C: Related knowledge search (vector search).

        Uses dense vector retrieval via MemoryRetriever.
        Searches both personal knowledge and shared common_knowledge,
        merging results by score.

        Returns a ``(medium_text, untrusted_text)`` tuple where results
        are split by their provenance-derived trust level.

        Args:
            keywords: Search keywords extracted from message.
            restrict_to: If provided, only return results whose source file
                stem is in this list (used for overflow-only search when
                distilled knowledge injection handles the rest).
            message: Original message text. Prepended (truncated to 200 chars)
                to the keyword query to preserve phrase-level semantics.
        """
        if not self.knowledge_dir.is_dir():
            logger.debug("Channel C: No knowledge dir")
            return ("", "")

        try:
            retriever = self._get_or_create_retriever()
            if retriever is None:
                logger.debug("Channel C: Retriever unavailable")
                return ("", "")

            queries = self._build_dual_queries(message, keywords)
            if not queries:
                logger.debug("Channel C: No keywords and no message")
                return ("", "")
            anima_name = self.anima_dir.name

            # Load min_score from config
            _min_score: float | None = None
            try:
                from core.config.models import load_config as _load_cfg

                _min_score = _load_cfg().rag.min_retrieval_score
            except Exception:
                logger.debug("Failed to load rag.min_retrieval_score from config, using default")

            results = self._search_and_merge(
                retriever,
                queries,
                anima_name,
                memory_type="knowledge",
                top_k=5,
                include_shared=True,
                min_score=_min_score,
            )

            # Filter to overflow files if restriction is specified
            if restrict_to is not None and results:
                restrict_set = set(restrict_to)
                from pathlib import Path as _Path

                results = [
                    r for r in results if _Path(str(r.metadata.get("source_file", r.doc_id))).stem in restrict_set
                ]

            if results:
                from core.execution._sanitize import ORIGIN_UNKNOWN, resolve_trust

                # Record access (Hebbian LTP: memories that fire together wire together)
                retriever.record_access(results, anima_name)

                # Classify results by trust level
                medium_parts: list[str] = []
                untrusted_parts: list[str] = []

                for i, result in enumerate(results):
                    chunk_origin = result.metadata.get("origin", "")
                    chunk_trust = resolve_trust(chunk_origin or ORIGIN_UNKNOWN)
                    source_label = result.metadata.get("anima", anima_name)
                    label = "shared" if source_label == "shared" else "personal"
                    line = f"--- Result {i + 1} [{label}] (score: {result.score:.3f}) ---\n{result.content}\n"
                    if chunk_trust == "untrusted":
                        untrusted_parts.append(line)
                    else:
                        medium_parts.append(line)

                medium_output = "\n".join(medium_parts)
                untrusted_output = "\n".join(untrusted_parts)

                logger.debug(
                    "Channel C: Vector search returned %d results (medium=%d, untrusted=%d)%s",
                    len(results),
                    len(medium_parts),
                    len(untrusted_parts),
                    f" (restricted to {len(restrict_to)} overflow files)" if restrict_to else "",
                )
                return (medium_output, untrusted_output)
            else:
                logger.debug("Channel C: Vector search found no results")
                return ("", "")

        except Exception as e:
            logger.warning("Channel C: Vector search failed: %s", e)
            return ("", "")

    async def _channel_d_skill_match(
        self,
        message: str,
        keywords: list[str],
        channel: str = "chat",
    ) -> list[str]:
        """Channel D: Skill matching via description-based 3-tier search.

        Uses ``match_skills_by_description()`` from ``core.memory.manager``
        which applies Tier 1 (keyword), Tier 2 (vocabulary), and Tier 3
        (vector search) matching.  The MemoryRetriever instance is shared
        with Channel C via ``_get_or_create_retriever()``.

        Returns list of skill/procedure names (not full content, max 5).
        Searches personal skills/, common_skills/, and procedures/.
        """
        _MAX_SKILL_MATCHES = 5

        if not message and not keywords:
            return []

        from core.memory.manager import MemoryManager, match_skills_by_description
        from core.paths import get_common_skills_dir

        # Collect all skill/procedure metas from the three sources.
        # File I/O is offloaded to a thread to avoid blocking the event loop.
        all_metas: list = []
        _seen_names: set[str] = set()

        def _collect_metas() -> list:
            """Synchronous helper — runs in thread via run_sync."""
            metas: list = []
            names: set[str] = set()

            # Personal skills (highest precedence)
            if self.skills_dir.is_dir():
                personal_skill_files = sorted(self.skills_dir.glob("*/SKILL.md"))
                # Backward compatibility: legacy flat skills/*.md layout.
                personal_skill_files.extend(sorted(self.skills_dir.glob("*.md")))
                for f in personal_skill_files:
                    try:
                        meta = MemoryManager._extract_skill_meta(f, is_common=False)
                        if meta.name not in names:
                            metas.append(meta)
                            names.add(meta.name)
                    except Exception:
                        logger.debug("Failed to extract skill meta from %s", f, exc_info=True)

            # Common skills
            common_dir = get_common_skills_dir()
            if common_dir.is_dir():
                common_skill_files = sorted(common_dir.glob("*/SKILL.md"))
                # Backward compatibility: legacy flat common_skills/*.md layout.
                common_skill_files.extend(sorted(common_dir.glob("*.md")))
                for f in common_skill_files:
                    try:
                        meta = MemoryManager._extract_skill_meta(f, is_common=True)
                        if meta.name not in names:
                            metas.append(meta)
                            names.add(meta.name)
                    except Exception:
                        logger.debug("Failed to extract common skill meta from %s", f, exc_info=True)

            # Procedures
            procedures_dir = self.anima_dir / "procedures"
            if procedures_dir.is_dir():
                for f in sorted(procedures_dir.glob("*.md")):
                    try:
                        meta = MemoryManager._extract_skill_meta(f, is_common=False)
                        if meta.name not in names:
                            metas.append(meta)
                            names.add(meta.name)
                    except Exception:
                        logger.debug("Failed to extract procedure meta from %s", f, exc_info=True)

            return metas

        all_metas = await run_sync(_collect_metas)

        if not all_metas:
            return []

        try:
            retriever = self._get_or_create_retriever()
            anima_name = self.anima_dir.name

            matched = match_skills_by_description(
                message,
                all_metas,
                retriever=retriever,
                anima_name=anima_name,
            )

            result = [m.name for m in matched[:_MAX_SKILL_MATCHES]]

            if result:
                logger.debug(
                    "Channel D: Matched %d skills: %s",
                    len(result),
                    result,
                )

            return result

        except Exception as e:
            logger.warning(
                "Channel D: Full skill matching failed, trying Tier 1/2 only: %s",
                e,
            )
            try:
                matched = match_skills_by_description(
                    message,
                    all_metas,
                    retriever=None,
                    anima_name="",
                )
                result = [m.name for m in matched[:_MAX_SKILL_MATCHES]]
                if result:
                    logger.debug(
                        "Channel D: Tier 1/2 fallback matched %d skills: %s",
                        len(result),
                        result,
                    )
                return result
            except Exception as e2:
                logger.warning("Channel D: Tier 1/2 fallback also failed: %s", e2)
                return []

    async def _channel_f_episodes(
        self,
        keywords: list[str],
        *,
        message: str = "",
    ) -> str:
        """Channel F: Episode memory search (vector search).

        Searches episodes/ via dense vector retrieval to surface
        semantically relevant past experiences.  Complements Channel B
        (recent activity timeline) by looking further back in time and
        ranking by semantic similarity rather than recency alone.
        """
        if not self.episodes_dir.is_dir():
            return ""

        try:
            retriever = self._get_or_create_retriever()
            if retriever is None:
                return ""

            queries = self._build_dual_queries(message, keywords)
            if not queries:
                return ""
            anima_name = self.anima_dir.name

            # Load min_score from config
            _min_score: float | None = None
            try:
                from core.config.models import load_config as _load_cfg

                _min_score = _load_cfg().rag.min_retrieval_score
            except Exception:
                logger.debug("Failed to load rag.min_retrieval_score from config, using default")

            results = self._search_and_merge(
                retriever,
                queries,
                anima_name,
                memory_type="episodes",
                top_k=3,
                min_score=_min_score,
            )

            if not results:
                return ""

            retriever.record_access(results, anima_name)

            parts: list[str] = []
            for i, result in enumerate(results):
                source = result.metadata.get("source_file", result.doc_id)
                parts.append(
                    f"--- Episode {i + 1} (score: {result.score:.3f}, source: {source}) ---\n{result.content}\n"
                )

            logger.debug(
                "Channel F: Episode search returned %d results",
                len(results),
            )
            return "\n".join(parts)

        except Exception as e:
            logger.warning("Channel F: Episode search failed: %s", e)
            return ""

    async def _channel_e_pending_tasks(self) -> str:
        """Channel E: Pending task queue summary + active parallel tasks.

        Retrieves pending tasks from the persistent task queue.
        Human-origin tasks are marked with 🔴 HIGH priority.
        Also includes currently running parallel tasks (Level 2 format:
        title + description summary + status + elapsed time).
        Budget: 300 tokens.

        Uses asyncio.to_thread to avoid blocking the event loop
        since TaskQueueManager performs synchronous file I/O.
        """
        parts: list[str] = []

        # Existing: task queue entries
        try:
            from core.memory.task_queue import TaskQueueManager

            manager = TaskQueueManager(self.anima_dir)
            queue_summary = await asyncio.to_thread(
                manager.format_for_priming,
                _BUDGET_PENDING_TASKS,
            )
            if queue_summary:
                parts.append(queue_summary)
        except Exception:
            logger.debug("Channel E (pending_tasks) failed", exc_info=True)

        # New: active parallel tasks from _active_parallel_tasks
        active = self._get_active_parallel_tasks() if self._get_active_parallel_tasks else {}
        if active:
            lines = [t("priming.active_parallel_tasks_header")]
            for tid, info in active.items():
                elapsed = self._format_elapsed(info.get("started_at", ""))
                status = info.get("status", "running")
                deps = info.get("depends_on", [])
                dep_str = f", depends_on: {','.join(deps)}" if deps else ""
                lines.append(f"- [{tid}] {info.get('title', '?')} ({status} {elapsed}{dep_str})")
                desc = info.get("description", "")
                if desc:
                    lines.append(f"  {desc[:100]}")
            parts.append("\n".join(lines))

        # Completed background tasks from state/task_results/
        results_dir = self.anima_dir / "state" / "task_results"
        if results_dir.is_dir():
            try:
                result_files = sorted(
                    results_dir.glob("*.md"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:5]
                if result_files:
                    lines = [t("priming.completed_bg_tasks_header")]
                    for rf in result_files:
                        try:
                            content = rf.read_text(encoding="utf-8").strip()
                            task_id = rf.stem
                            preview = content[:150].replace("\n", " ")
                            lines.append(f"- [{task_id}] {preview}")
                        except Exception:
                            logger.debug("Channel E: failed to read %s", rf.name, exc_info=True)
                    if len(lines) > 1:
                        parts.append("\n".join(lines))
            except Exception:
                logger.debug("Channel E: task_results read failed", exc_info=True)

        return "\n\n".join(parts)

    @staticmethod
    def _format_elapsed(started_at: str) -> str:
        """Format elapsed time from an ISO timestamp."""
        if not started_at:
            return ""
        try:
            from datetime import datetime as _dt

            start = _dt.fromisoformat(started_at)
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            elapsed_s = (_dt.now(UTC) - start).total_seconds()
            if elapsed_s < 60:
                return f"{int(elapsed_s)}s"
            if elapsed_s < 3600:
                return f"{int(elapsed_s / 60)}m"
            return f"{elapsed_s / 3600:.1f}h"
        except (ValueError, TypeError):
            return ""

    # ── Recent outbound collection ────────────────────────────────

    async def _collect_recent_outbound(self, max_entries: int = 3) -> str:
        """Collect recent outbound actions (channel_post, message_sent).

        Reads activity_log for the last 2 hours and formats a short summary.
        This replaces the former ``_build_recent_outbound_section`` in builder.py,
        ensuring builder.py never reads ActivityLogger directly (hippocampus model).
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)
            entries = activity.recent(
                days=1,
                limit=20,
                types=["channel_post", "message_sent"],
            )
        except Exception:
            return ""

        if not entries:
            return ""

        from datetime import datetime, timedelta

        cutoff = now_local() - timedelta(hours=2)

        recent: list = []
        for e in reversed(entries):
            try:
                ts = ensure_aware(datetime.fromisoformat(e.ts))
                if ts >= cutoff:
                    recent.append(e)
            except (ValueError, TypeError):
                continue
            if len(recent) >= max_entries:
                break

        if not recent:
            return ""

        lines = [t("priming.outbound_header"), ""]
        for e in reversed(recent):
            time_str = e.ts[11:16] if len(e.ts) >= 16 else e.ts
            text_preview = (e.summary or e.content or "")[:200]
            if e.type == "channel_post":
                ch = e.channel or "?"
                lines.append(t("priming.outbound_posted", time_str=time_str, ch=ch, text_preview=text_preview))
            elif e.type in ("dm_sent", "message_sent"):
                to = e.to_person or "?"
                lines.append(t("priming.outbound_sent", time_str=time_str, to=to, text_preview=text_preview))
        lines.append("")
        return "\n".join(lines)

    # ── Pending human notifications ─────────────────────────────

    _HUMAN_NOTIFY_BUDGET_TOKENS = 500

    async def _collect_pending_human_notifications(self, *, channel: str = "") -> str:
        """Collect recent call_human notifications for context injection.

        Returns formatted string of human_notify entries from last 24 hours.
        Only active for chat, heartbeat, and message: sessions.
        """
        if channel not in ("chat", "heartbeat") and not channel.startswith("message:"):
            return ""

        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(self.anima_dir)
        entries = activity.recent(days=1, limit=10, types=["human_notify"])
        if not entries:
            return ""

        lines: list[str] = []
        budget_chars = self._HUMAN_NOTIFY_BUDGET_TOKENS * _CHARS_PER_TOKEN
        total = 0
        for entry in reversed(entries):
            ts = entry.ts[:16]
            body = entry.content or entry.summary or ""
            via = entry.via or ""
            line = f"[{ts}] call_human (via {via}):\n{body}"
            if total + len(line) > budget_chars:
                break
            lines.append(line)
            total += len(line)

        if not lines:
            return ""

        lines.reverse()
        header = "## Pending Human Notifications (last 24h)"
        return header + "\n\n" + "\n\n".join(lines)

    # ── Config loading ──────────────────────────────────────────

    def _load_config_budgets(self) -> None:
        """Load budget values from config.json (lazy, once per instance)."""
        if self._config_loaded:
            return
        self._config_loaded = True
        try:
            from core.config.models import load_config

            config = load_config()
            p = config.priming
            self._budget_greeting = p.budget_greeting
            self._budget_question = p.budget_question
            self._budget_request = p.budget_request
            self._budget_heartbeat = p.budget_heartbeat
            self._heartbeat_context_pct = p.heartbeat_context_pct
        except Exception:
            logger.debug("Failed to load priming config; using defaults")

    # ── Dynamic budget adjustment (Phase 3) ─────────────────────

    def _classify_message_type(self, message: str, channel: str, *, intent: str = "") -> str:
        """Classify message type for budget adjustment.

        Args:
            message: Message text
            channel: Message channel
            intent: Sender-declared intent (preferred when provided)

        Returns:
            Message type: "greeting", "question", "request", "heartbeat"
        """
        # Heartbeat channel has fixed budget
        if channel == "heartbeat":
            return "heartbeat"

        # Sender-declared intent takes priority over keyword heuristics.
        intent_map = {
            "delegation": "request",
            "report": "question",
            "question": "question",
        }
        intent_norm = str(intent or "").strip().lower()
        mapped = intent_map.get(intent_norm)
        if mapped:
            return mapped

        message_lower = message.lower()

        # Simple greeting patterns
        greeting_patterns = [
            "こんにちは",
            "おはよう",
            "こんばんは",
            "よろしく",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good evening",
        ]
        if any(p in message_lower for p in greeting_patterns) and len(message) < 50:
            return "greeting"

        # Question patterns
        question_patterns = [
            "?",
            "？",
            "教えて",
            "どう",
            "なぜ",
            "いつ",
            "どこ",
            "誰",
            "what",
            "why",
            "when",
            "where",
            "who",
            "how",
            "can you",
        ]
        if any(p in message_lower for p in question_patterns):
            return "question"

        # Default to request for longer messages
        if len(message) > 100:
            return "request"

        # Default to question
        return "question"

    def _adjust_token_budget(self, message: str, channel: str, *, intent: str = "") -> int:
        """Adjust token budget based on message type.

        For heartbeat, the budget is ``max(config.budget_heartbeat,
        int(context_window * config.heartbeat_context_pct))`` so that
        models with large context windows get proportionally more priming.

        Args:
            message: Message text
            channel: Message channel
            intent: Sender-declared intent (preferred when provided)

        Returns:
            Adjusted token budget
        """
        self._load_config_budgets()
        msg_type = self._classify_message_type(message, channel, intent=intent)

        if msg_type == "heartbeat":
            base = self._budget_heartbeat
            if self.context_window > 0:
                pct_budget = int(self.context_window * self._heartbeat_context_pct)
                budget = max(base, pct_budget)
            else:
                budget = base
        else:
            budget_map = {
                "greeting": self._budget_greeting,
                "question": self._budget_question,
                "request": self._budget_request,
            }
            budget = budget_map.get(msg_type, _DEFAULT_MAX_PRIMING_TOKENS)

        logger.debug("Message type: %s -> budget: %d", msg_type, budget)
        return budget

    # ── Dual-query helpers ────────────────────────────────────────

    @staticmethod
    def _build_dual_queries(message: str, keywords: list[str]) -> list[str]:
        """Build dual queries: message-context + keyword-only.

        Returns 1–2 queries.  Avoids issuing a duplicate when the keyword
        query is identical to the message query.
        """
        queries: list[str] = []
        msg_query = message[:300].strip() if message else ""
        kw_query = " ".join(keywords[:5]).strip() if keywords else ""

        if msg_query:
            queries.append(msg_query)
        if kw_query and kw_query != msg_query:
            queries.append(kw_query)

        return queries

    def _search_and_merge(
        self,
        retriever: MemoryRetriever,
        queries: list[str],
        anima_name: str,
        *,
        memory_type: str,
        top_k: int,
        include_shared: bool = False,
        min_score: float | None = None,
    ) -> list:
        """Execute multiple queries and merge by max-score deduplication."""
        best: dict[str, object] = {}

        for query in queries:
            results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type=memory_type,
                top_k=top_k,
                include_shared=include_shared,
                min_score=min_score,
            )
            for r in results:
                existing = best.get(r.doc_id)
                if existing is None or r.score > existing.score:  # type: ignore[union-attr]
                    best[r.doc_id] = r

        return sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]  # type: ignore[union-attr]

    # ── Keyword extraction ─────────────────────────────────────

    def _extract_keywords(self, message: str) -> list[str]:
        """Language-agnostic keyword extraction.

        1. Known entity matching (knowledge/ filenames — top priority)
        2. Unicode-aware tokenization
        3. Character-category min-length filter + minimal stopwords
        4. Length-descending sort (longer = more specific)

        Input is truncated to ``_MAX_KEYWORD_INPUT_LEN`` to bound regex cost.
        """
        text = message[:_MAX_KEYWORD_INPUT_LEN] if len(message) > _MAX_KEYWORD_INPUT_LEN else message

        known_entities: set[str] = set()
        if self.knowledge_dir.is_dir():
            known_entities = {f.stem.lower() for f in self.knowledge_dir.glob("*.md")}

        tokens = _RE_UNICODE_WORDS.findall(text)

        filtered = [t for t in tokens if self._meets_min_length(t) and t.lower() not in _MINIMAL_STOPWORDS]

        entity_matches = [t for t in filtered if t.lower() in known_entities]
        entity_set = {t.lower() for t in entity_matches}

        general = [t for t in filtered if t.lower() not in entity_set]
        general.sort(key=len, reverse=True)

        seen: set[str] = set()
        combined: list[str] = []
        for w in entity_matches + general:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                combined.append(w)

        return combined[:10]

    @staticmethod
    def _meets_min_length(token: str) -> bool:
        """Check minimum length based on Unicode character category.

        CJK characters carry meaning even as a single character (e.g. 裏, 金, 型).
        Latin and other scripts need at least 3 characters to be meaningful.
        """
        for c in token:
            cp = ord(c)
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3040 <= cp <= 0x309F
                or 0x30A0 <= cp <= 0x30FF
                or 0xAC00 <= cp <= 0xD7AF
                or 0x0E00 <= cp <= 0x0E7F
            ):
                return len(token) >= 1
        return len(token) >= 3

    def _truncate_head(self, text: str, max_tokens: int) -> str:
        """Truncate text keeping the head (front), cutting from the tail.

        Suitable for sender profiles (basic info at the top) and
        ripgrep results (best matches first).
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text

        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_period = max(
            truncated.rfind("。"),
            truncated.rfind("."),
            truncated.rfind("\n"),
        )
        if last_period > max_chars * 0.8:  # If we're close enough
            return truncated[: last_period + 1]

        return truncated + "..."

    def _truncate_tail(self, text: str, max_tokens: int) -> str:
        """Truncate text keeping the tail (end), cutting from the head.

        Suitable for recent episodes where newest entries are most relevant.
        """
        max_chars = max_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text

        # Keep the tail portion
        truncated = text[-max_chars:]
        # Try to start at a clean boundary
        first_newline = truncated.find("\n")
        if first_newline != -1 and first_newline < max_chars * 0.2:
            return truncated[first_newline + 1 :]

        return "..." + truncated


# ── Public API ──────────────────────────────────────────────────


def format_priming_section(result: PrimingResult, sender_name: str = "human") -> str:
    """Format priming result as a Markdown section for system prompt injection.

    Args:
        result: The priming result to format
        sender_name: Name of the message sender

    Returns:
        Formatted markdown section, or empty string if no memories primed
    """
    from core.execution._sanitize import wrap_priming

    if result.is_empty():
        return ""

    parts: list[str] = []
    parts.append(t("priming.section_title"))
    parts.append("")
    parts.append(t("priming.section_intro"))
    parts.append("")

    if result.sender_profile:
        parts.append(t("priming.about_sender", sender_name=sender_name))
        parts.append("")
        parts.append(wrap_priming("sender_profile", result.sender_profile, trust="medium"))
        parts.append("")

    if result.recent_activity:
        parts.append(t("priming.recent_activity_header"))
        parts.append("")
        parts.append(wrap_priming("recent_activity", result.recent_activity, trust="untrusted"))
        parts.append("")

    if result.related_knowledge or result.related_knowledge_untrusted:
        from core.execution._sanitize import ORIGIN_CONSOLIDATION, ORIGIN_EXTERNAL_PLATFORM

        parts.append(t("priming.related_knowledge_header"))
        parts.append("")
        if result.related_knowledge:
            if result.related_knowledge_untrusted:
                parts.append(
                    wrap_priming(
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                        origin=ORIGIN_CONSOLIDATION,
                    )
                )
            else:
                parts.append(
                    wrap_priming(
                        "related_knowledge",
                        result.related_knowledge,
                        trust="medium",
                    )
                )
            parts.append("")
        if result.related_knowledge_untrusted:
            parts.append(
                wrap_priming(
                    "related_knowledge_external",
                    result.related_knowledge_untrusted,
                    trust="untrusted",
                    origin=ORIGIN_EXTERNAL_PLATFORM,
                )
            )
            parts.append("")

    if result.episodes:
        parts.append(t("priming.episodes_header"))
        parts.append("")
        parts.append(wrap_priming("episodes", result.episodes, trust="medium"))
        parts.append("")

    if result.matched_skills:
        parts.append(t("priming.matched_skills_header"))
        parts.append("")
        skills_line = ", ".join(result.matched_skills)
        parts.append(t("priming.skills_list", skills_line=skills_line))
        parts.append("")
        parts.append(t("priming.skills_detail_hint"))
        parts.append("")

    if result.pending_tasks:
        parts.append(t("priming.pending_tasks_header"))
        parts.append("")
        parts.append(wrap_priming("pending_tasks", result.pending_tasks, trust="medium"))
        parts.append("")

    if result.recent_outbound:
        parts.append(wrap_priming("recent_outbound", result.recent_outbound, trust="trusted"))
        parts.append("")

    return "\n".join(parts)
