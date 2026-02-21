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
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from core.time_utils import ensure_aware, now_jst
from core.tools._async_compat import run_sync

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
_BUDGET_RELATED_KNOWLEDGE = 700
_BUDGET_SKILL_MATCH = 200
_BUDGET_PENDING_TASKS = 300

# Rough characters-per-token for Japanese/English mixed text
_CHARS_PER_TOKEN = 4

# Pre-compiled regex patterns for keyword extraction (avoids ReDoS risk)
_RE_KATAKANA = re.compile(r"[\u30A0-\u30FF]{2,}")
_RE_WORDS = re.compile(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+")
# Maximum message length to process for keyword extraction
_MAX_KEYWORD_INPUT_LEN = 5000


# ── Data structures ────────────────────────────────────────────


@dataclass
class PrimingResult:
    """Result of priming memory retrieval."""

    sender_profile: str = ""
    recent_activity: str = ""
    related_knowledge: str = ""
    matched_skills: list[str] = field(default_factory=list)
    pending_tasks: str = ""

    def is_empty(self) -> bool:
        """Return True if no memories were primed."""
        return (
            not self.sender_profile
            and not self.recent_activity
            and not self.related_knowledge
            and not self.matched_skills
            and not self.pending_tasks
        )

    def total_chars(self) -> int:
        """Estimate total character count."""
        return (
            len(self.sender_profile)
            + len(self.recent_activity)
            + len(self.related_knowledge)
            + sum(len(s) for s in self.matched_skills)
            + len(self.pending_tasks)
        )

    def estimated_tokens(self) -> int:
        """Estimate token count."""
        return self.total_chars() // _CHARS_PER_TOKEN


# ── PrimingEngine ──────────────────────────────────────────────


class PrimingEngine:
    """Automatic memory priming engine.

    Executes 5-channel parallel memory retrieval:
      A. Sender profile (direct file read)
      B. Recent activity (unified activity log, replaces old episodes + channels)
      C. Related knowledge (dense vector search)
      D. Skill matching (description-based 3-tier match with vector search)
      E. Pending tasks (persistent task queue summary)
    """

    def __init__(self, anima_dir: Path, shared_dir: Path | None = None) -> None:
        self.anima_dir = anima_dir
        self.shared_dir = shared_dir
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.skills_dir = anima_dir / "skills"
        # Lazily initialized by _get_or_create_retriever(); shared by C & D
        self._retriever: object | None = None
        self._retriever_initialized: bool = False

    # ── Main entry point ────────────────────────────────────────

    async def prime_memories(
        self,
        message: str,
        sender_name: str = "human",
        channel: str = "chat",
        intent: str = "",
        enable_dynamic_budget: bool = False,
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

        # Execute 5 channels in parallel
        results = await asyncio.gather(
            self._channel_a_sender_profile(sender_name),
            self._channel_b_recent_activity(sender_name, keywords),  # Unified channel
            self._channel_c_related_knowledge(keywords),
            self._channel_d_skill_match(message, keywords, channel=channel),
            self._channel_e_pending_tasks(),
            return_exceptions=True,
        )

        # Unpack results (handle exceptions gracefully)
        sender_profile = results[0] if isinstance(results[0], str) else ""
        recent_activity = results[1] if isinstance(results[1], str) else ""
        related_knowledge = results[2] if isinstance(results[2], str) else ""
        matched_skills = results[3] if isinstance(results[3], list) else []
        pending_tasks = results[4] if isinstance(results[4], str) else ""

        # Log exceptions if any
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Priming channel %d failed: %s", i, r)

        # Apply token budget limits (distribute based on budget)
        budget_profile = int(_BUDGET_SENDER_PROFILE * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS)))
        budget_knowledge = int(_BUDGET_RELATED_KNOWLEDGE * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_skills = int(_BUDGET_SKILL_MATCH * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_tasks = int(_BUDGET_PENDING_TASKS * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))

        result = PrimingResult(
            sender_profile=self._truncate_head(sender_profile, budget_profile),
            recent_activity=self._truncate_tail(recent_activity, budget_activity),
            related_knowledge=self._truncate_head(related_knowledge, budget_knowledge),
            matched_skills=matched_skills[:max(1, budget_skills // 50)],  # ~50 tokens per skill name
            pending_tasks=self._truncate_head(pending_tasks, budget_tasks),
        )

        logger.info(
            "Priming complete: %d chars (~%d tokens), sender_prof=%d, activity=%d, "
            "knowledge=%d, skills=%d",
            result.total_chars(),
            result.estimated_tokens(),
            len(result.sender_profile),
            len(result.recent_activity),
            len(result.related_knowledge),
            len(result.matched_skills),
        )

        return result

    # ── Shared retriever ────────────────────────────────────────

    def _get_or_create_retriever(self) -> object | None:
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
            from core.memory.rag.singleton import get_vector_store
            from core.memory.rag.indexer import MemoryIndexer

            vector_store = get_vector_store()
            anima_name = self.anima_dir.name
            indexer = MemoryIndexer(vector_store, anima_name, self.anima_dir)
            self._retriever = MemoryRetriever(
                vector_store, indexer, self.knowledge_dir,
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
        profile_path = shared_users_dir / sender_name / "index.md"

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

    async def _channel_b_recent_activity(self, sender_name: str, keywords: list[str]) -> str:
        """Channel B: Recent activity from unified activity log.

        Replaces old Channel B (episodes) and Channel E (shared channels).
        Reads from activity_log/{date}.jsonl for a unified timeline,
        plus shared/channels/*.jsonl for cross-Anima visibility.
        Falls back to episodes/ if activity_log is empty (migration period).
        """
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(self.anima_dir)
        entries = activity.recent(days=2)  # No limit — fetch all entries

        # Always read shared channels for cross-Anima visibility
        channel_entries = self._read_shared_channels(limit_per_channel=5)
        entries.extend(channel_entries)

        if entries:
            # Prioritize: sender-related entries first, then by recency
            prioritized = self._prioritize_entries(entries, sender_name, keywords)
            # Apply limit after scoring
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
        from core.memory.activity import ActivityEntry
        from datetime import datetime

        if not self.shared_dir:
            return []

        channels_dir = self.shared_dir / "channels"
        if not channels_dir.is_dir():
            return []

        anima_name = self.anima_dir.name
        mention_tag = f"@{anima_name}"
        now = now_jst()
        cutoff_24h = now - timedelta(hours=24)

        result: list[ActivityEntry] = []

        try:
            for channel_file in sorted(channels_dir.glob("*.jsonl")):
                channel_name = channel_file.stem
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
                    result.append(ActivityEntry(
                        ts=entry.get("ts", ""),
                        type="channel_post",
                        content=entry.get("text", ""),
                        summary=entry.get("text", "")[:100],
                        from_person=entry.get("from", ""),
                        channel=channel_name,
                    ))

        except Exception:
            logger.warning("Failed to read shared channels", exc_info=True)

        return result

    def _prioritize_entries(
        self,
        entries: list,  # list[ActivityEntry]
        sender_name: str,
        keywords: list[str],
    ) -> list:
        """Prioritize activity entries for priming.

        Priority order:
        1. Entries involving the current sender (most relevant)
        2. Entries matching keywords (topically relevant)
        3. Most recent entries (temporal relevance)
        """
        from core.memory.activity import ActivityEntry

        keywords_lower = {kw.lower() for kw in keywords} if keywords else set()

        scored: list[tuple[float, int, ActivityEntry]] = []
        for i, entry in enumerate(entries):
            score = 0.0
            # Sender relevance
            if entry.from_person == sender_name or entry.to_person == sender_name:
                score += 10.0
            # Keyword relevance
            text = (entry.content + " " + entry.summary).lower()
            matching_kw = sum(1 for kw in keywords_lower if kw in text)
            score += matching_kw * 3.0
            # Recency (later index = more recent = higher score)
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
        today = date.today()

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
        now = now_jst()
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
                    f"[{entry.get('ts', '?')}] {entry.get('from', '?')}{marker}: "
                    f"{entry.get('text', '')}"
                )

            if channel_parts:
                parts.append(f"### #{channel_name}")
                parts.extend(channel_parts)
                parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    async def _channel_c_related_knowledge(self, keywords: list[str]) -> str:
        """Channel C: Related knowledge search (vector search).

        Uses dense vector retrieval via MemoryRetriever.
        Searches both personal knowledge and shared common_knowledge,
        merging results by score.
        """
        if not self.knowledge_dir.is_dir() or not keywords:
            logger.debug("Channel C: No knowledge dir or no keywords")
            return ""

        try:
            retriever = self._get_or_create_retriever()
            if retriever is None:
                logger.debug("Channel C: Retriever unavailable")
                return ""

            # Build query from keywords
            query = " ".join(keywords[:5])
            anima_name = self.anima_dir.name

            # Vector search (personal + shared common_knowledge)
            results = retriever.search(
                query=query,
                anima_name=anima_name,
                memory_type="knowledge",
                top_k=3,
                include_shared=True,
            )

            if results:
                # Record access (Hebbian LTP: memories that fire together wire together)
                retriever.record_access(results, anima_name)

                # Format results
                parts = []
                for i, result in enumerate(results):
                    source_label = result.metadata.get("anima", anima_name)
                    label = "shared" if source_label == "shared" else "personal"
                    parts.append(
                        f"--- Result {i + 1} [{label}] "
                        f"(score: {result.score:.3f}) ---"
                    )
                    parts.append(result.content)
                    parts.append("")

                output = "\n".join(parts)
                logger.debug(
                    "Channel C: Vector search returned %d results (%d chars)",
                    len(results),
                    len(output),
                )
                return output
            else:
                logger.debug("Channel C: Vector search found no results")
                return ""

        except Exception as e:
            logger.warning("Channel C: Vector search failed: %s", e)
            return ""

    async def _channel_d_skill_match(
        self, message: str, keywords: list[str],
        channel: str = "chat",
    ) -> list[str]:
        """Channel D: Skill matching via description-based 3-tier search.

        Uses ``match_skills_by_description()`` from ``core.memory.manager``
        which applies Tier 1 (keyword), Tier 2 (vocabulary), and Tier 3
        (vector search) matching.  The MemoryRetriever instance is shared
        with Channel C via ``_get_or_create_retriever()``.

        Returns list of skill/procedure names (not full content, max 5).
        Searches personal skills/, common_skills/, and procedures/.
        Skipped for heartbeat/cron triggers to avoid false positives.
        """
        _MAX_SKILL_MATCHES = 5

        if channel in ("heartbeat", "cron"):
            return []

        if not message and not keywords:
            return []

        from core.memory.manager import MemoryManager, match_skills_by_description
        from core.paths import get_common_skills_dir

        # Collect all skill/procedure metas from the three sources.
        # File I/O is offloaded to a thread to avoid blocking the event loop.
        all_metas: list = []
        seen_names: set[str] = set()

        def _collect_metas() -> list:
            """Synchronous helper — runs in thread via run_sync."""
            metas: list = []
            names: set[str] = set()

            # Personal skills (highest precedence)
            if self.skills_dir.is_dir():
                for f in sorted(self.skills_dir.glob("*.md")):
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
                for f in sorted(common_dir.glob("*.md")):
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
                    "Channel D: Matched %d skills: %s", len(result), result,
                )

            return result

        except Exception as e:
            logger.warning("Channel D: Skill matching failed: %s", e)
            return []

    async def _channel_e_pending_tasks(self) -> str:
        """Channel E: Pending task queue summary.

        Retrieves pending tasks from the persistent task queue.
        Human-origin tasks are marked with 🔴 HIGH priority.
        Budget: 300 tokens.

        Uses asyncio.to_thread to avoid blocking the event loop
        since TaskQueueManager performs synchronous file I/O.
        """
        try:
            from core.memory.task_queue import TaskQueueManager
            manager = TaskQueueManager(self.anima_dir)
            return await asyncio.to_thread(
                manager.format_for_priming, _BUDGET_PENDING_TASKS,
            )
        except Exception:
            logger.debug("Channel E (pending_tasks) failed", exc_info=True)
            return ""

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
            "こんにちは", "おはよう", "こんばんは", "よろしく",
            "hello", "hi", "hey", "good morning", "good evening",
        ]
        if any(p in message_lower for p in greeting_patterns) and len(message) < 50:
            return "greeting"

        # Question patterns
        question_patterns = [
            "?", "？", "教えて", "どう", "なぜ", "いつ", "どこ", "誰",
            "what", "why", "when", "where", "who", "how", "can you",
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

        Args:
            message: Message text
            channel: Message channel
            intent: Sender-declared intent (preferred when provided)

        Returns:
            Adjusted token budget
        """
        msg_type = self._classify_message_type(message, channel, intent=intent)

        budget_map = {
            "greeting": _BUDGET_GREETING,
            "question": _BUDGET_QUESTION,
            "request": _BUDGET_REQUEST,
            "heartbeat": _BUDGET_HEARTBEAT,
        }

        budget = budget_map.get(msg_type, _DEFAULT_MAX_PRIMING_TOKENS)
        logger.debug("Message type: %s -> budget: %d", msg_type, budget)
        return budget

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_keywords(self, message: str) -> list[str]:
        """Extract keywords from message with 3-stage extraction.

        1. Proper noun patterns (katakana sequences, capitalized English words)
        2. Known entity matching (knowledge/ filenames)
        3. General keywords (stopword-filtered)

        Input is truncated to ``_MAX_KEYWORD_INPUT_LEN`` to bound regex cost.
        Patterns are pre-compiled module-level constants.
        """
        # Truncate oversized input to bound regex processing time
        text = message[:_MAX_KEYWORD_INPUT_LEN] if len(message) > _MAX_KEYWORD_INPUT_LEN else message

        # Remove common Japanese particles and English stopwords
        stopwords = {
            "の", "に", "は", "を", "が", "で", "と", "から", "まで",
            "も", "や", "へ", "より", "など", "について",
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "through", "during", "it", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could",
        }

        # 1. Proper nouns: katakana sequences (2+ chars)
        katakana_words = _RE_KATAKANA.findall(text)

        # 2. Known entities: match against knowledge/ filenames
        known_entities: set[str] = set()
        if self.knowledge_dir.is_dir():
            known_entities = {f.stem.lower() for f in self.knowledge_dir.glob("*.md")}

        # Split on whitespace and punctuation, keep alphanumeric + Japanese
        words = _RE_WORDS.findall(text)

        # Filter stopwords and short words
        general_keywords = [
            w for w in words
            if len(w) >= 2 and w.lower() not in stopwords
        ]

        # Entity matches from general keywords
        entity_matches = [w for w in general_keywords if w.lower() in known_entities]

        # Sort general keywords by length (longer = more specific)
        general_keywords.sort(key=len, reverse=True)

        # Combine: entity_matches + katakana_words + general keywords (deduplicated)
        seen: set[str] = set()
        combined: list[str] = []
        for w in entity_matches + katakana_words + general_keywords:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                combined.append(w)

        return combined[:10]

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
            return truncated[:last_period + 1]

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
            return truncated[first_newline + 1:]

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
    if result.is_empty():
        return ""

    parts: list[str] = []
    parts.append("## あなたが思い出していること")
    parts.append("")
    parts.append("以下は、この会話に関連してあなたが自然に想起した記憶です。")
    parts.append("")

    if result.sender_profile:
        parts.append(f"### {sender_name} について")
        parts.append("")
        parts.append(result.sender_profile)
        parts.append("")

    if result.recent_activity:
        parts.append("### 直近のアクティビティ")
        parts.append("")
        parts.append(result.recent_activity)
        parts.append("")

    if result.related_knowledge:
        parts.append("### 関連する知識")
        parts.append("")
        parts.append(result.related_knowledge)
        parts.append("")

    if result.matched_skills:
        parts.append("### 使えそうなスキル")
        parts.append("")
        skills_line = ", ".join(result.matched_skills)
        parts.append(f"あなたが持っているスキル: {skills_line}")
        parts.append("")
        parts.append("※詳細はスキルファイルをReadで確認してください。")
        parts.append("")

    if result.pending_tasks:
        parts.append("### 未完了タスク")
        parts.append("")
        parts.append(result.pending_tasks)
        parts.append("")

    return "\n".join(parts)
