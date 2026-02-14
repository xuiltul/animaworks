from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

"""Priming layer - automatic memory retrieval (自動想起).

Implements brain-science-inspired automatic memory activation before agent
execution, reducing the need for explicit search_memory tool calls.

Based on: docs/design/priming-layer-design.md Phase 1
"""

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

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
_BUDGET_RECENT_EPISODES = 600
_BUDGET_RELATED_KNOWLEDGE = 700
_BUDGET_SKILL_MATCH = 200

# Rough characters-per-token for Japanese/English mixed text
_CHARS_PER_TOKEN = 4


# ── Data structures ────────────────────────────────────────────


@dataclass
class PrimingResult:
    """Result of priming memory retrieval."""

    sender_profile: str = ""
    recent_episodes: str = ""
    related_knowledge: str = ""
    matched_skills: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if no memories were primed."""
        return (
            not self.sender_profile
            and not self.recent_episodes
            and not self.related_knowledge
            and not self.matched_skills
        )

    def total_chars(self) -> int:
        """Estimate total character count."""
        return (
            len(self.sender_profile)
            + len(self.recent_episodes)
            + len(self.related_knowledge)
            + sum(len(s) for s in self.matched_skills)
        )

    def estimated_tokens(self) -> int:
        """Estimate token count."""
        return self.total_chars() // _CHARS_PER_TOKEN


# ── PrimingEngine ──────────────────────────────────────────────


class PrimingEngine:
    """Automatic memory priming engine.

    Executes 4-channel parallel memory retrieval:
      A. Sender profile (direct file read)
      B. Recent episodes (last 2 days)
      C. Related knowledge (BM25 keyword search)
      D. Skill matching (filename pattern match)
    """

    def __init__(self, person_dir: Path) -> None:
        self.person_dir = person_dir
        self.episodes_dir = person_dir / "episodes"
        self.knowledge_dir = person_dir / "knowledge"
        self.skills_dir = person_dir / "skills"

    # ── Main entry point ────────────────────────────────────────

    async def prime_memories(
        self,
        message: str,
        sender_name: str = "human",
        channel: str = "chat",
        enable_dynamic_budget: bool = True,
    ) -> PrimingResult:
        """Prime memories based on incoming message.

        Args:
            message: The incoming message text
            sender_name: Name of the sender (for sender profile lookup)
            channel: Message channel (chat, heartbeat, cron, etc.)
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
            token_budget = self._adjust_token_budget(message, channel)
        else:
            token_budget = _DEFAULT_MAX_PRIMING_TOKENS

        logger.debug("Token budget: %d", token_budget)

        # Extract keywords for search (simple rule-based for Phase 1)
        keywords = self._extract_keywords(message)

        # Execute 4 channels in parallel
        results = await asyncio.gather(
            self._channel_a_sender_profile(sender_name),
            self._channel_b_recent_episodes(),
            self._channel_c_related_knowledge(keywords),
            self._channel_d_skill_match(keywords),
            return_exceptions=True,
        )

        # Unpack results (handle exceptions gracefully)
        sender_profile = results[0] if isinstance(results[0], str) else ""
        recent_episodes = results[1] if isinstance(results[1], str) else ""
        related_knowledge = results[2] if isinstance(results[2], str) else ""
        matched_skills = results[3] if isinstance(results[3], list) else []

        # Log exceptions if any
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Priming channel %d failed: %s", i, r)

        # Apply token budget limits (distribute based on budget)
        budget_profile = int(_BUDGET_SENDER_PROFILE * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_episodes = int(_BUDGET_RECENT_EPISODES * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_knowledge = int(_BUDGET_RELATED_KNOWLEDGE * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))
        budget_skills = int(_BUDGET_SKILL_MATCH * (token_budget / _DEFAULT_MAX_PRIMING_TOKENS))

        result = PrimingResult(
            sender_profile=self._truncate(sender_profile, budget_profile),
            recent_episodes=self._truncate(recent_episodes, budget_episodes),
            related_knowledge=self._truncate(related_knowledge, budget_knowledge),
            matched_skills=matched_skills[:max(1, budget_skills // 50)],  # ~50 tokens per skill name
        )

        logger.info(
            "Priming complete: %d chars (~%d tokens), sender_prof=%d, episodes=%d, "
            "knowledge=%d, skills=%d",
            result.total_chars(),
            result.estimated_tokens(),
            len(result.sender_profile),
            len(result.recent_episodes),
            len(result.related_knowledge),
            len(result.matched_skills),
        )

        return result

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
            content = profile_path.read_text(encoding="utf-8")
            logger.debug(
                "Channel A: Loaded sender profile for %s (%d chars)",
                sender_name,
                len(content),
            )
            return content
        except Exception as e:
            logger.warning("Channel A: Failed to read profile for %s: %s", sender_name, e)
            return ""

    async def _channel_b_recent_episodes(self) -> str:
        """Channel B: Recent episode logs.

        Reads last 2 days of episode files (today + yesterday).
        Returns newest entries first, truncated to budget.
        """
        if not self.episodes_dir.is_dir():
            return ""

        parts: list[str] = []
        today = date.today()

        # Read today and yesterday
        for offset in range(2):
            target_date = today - timedelta(days=offset)
            path = self.episodes_dir / f"{target_date.isoformat()}.md"

            if not path.exists():
                continue

            try:
                content = path.read_text(encoding="utf-8")
                # Take last N lines (most recent)
                lines = content.strip().splitlines()
                # Limit to ~30 lines per day to avoid overwhelming
                if len(lines) > 30:
                    lines = lines[-30:]
                parts.append("\n".join(lines))
            except Exception as e:
                logger.warning("Channel B: Failed to read episode %s: %s", path, e)

        if not parts:
            logger.debug("Channel B: No recent episodes found")
            return ""

        result = "\n\n---\n\n".join(parts)
        logger.debug("Channel B: Loaded %d days of episodes (%d chars)", len(parts), len(result))
        return result

    async def _channel_c_related_knowledge(self, keywords: list[str]) -> str:
        """Channel C: Related knowledge search (Hybrid search for Phase 2).

        Uses hybrid retrieval (vector + BM25 + RRF) if available,
        falls back to BM25-only (ripgrep) if RAG not initialized.
        """
        if not self.knowledge_dir.is_dir() or not keywords:
            logger.debug("Channel C: No knowledge dir or no keywords")
            return ""

        # Try hybrid search first (Phase 2)
        try:
            from core.memory.rag import HybridRetriever
            from core.memory.rag.store import ChromaVectorStore
            from core.memory.rag.indexer import MemoryIndexer

            # Initialize RAG components if not already done
            if not hasattr(self, "_hybrid_retriever"):
                vector_store = ChromaVectorStore()
                person_name = self.person_dir.name
                indexer = MemoryIndexer(vector_store, person_name, self.person_dir)
                self._hybrid_retriever = HybridRetriever(
                    vector_store, indexer, self.knowledge_dir
                )

            # Build query from keywords
            query = " ".join(keywords[:5])
            person_name = self.person_dir.name

            # Perform hybrid search
            results = self._hybrid_retriever.search(
                query=query,
                person_name=person_name,
                memory_type="knowledge",
                top_k=3,
            )

            if results:
                # Format results
                parts = []
                for i, result in enumerate(results):
                    parts.append(f"--- Result {i + 1} (score: {result.score:.3f}) ---")
                    parts.append(result.content)
                    parts.append("")

                output = "\n".join(parts)
                logger.debug(
                    "Channel C: Hybrid search returned %d results (%d chars)",
                    len(results),
                    len(output),
                )
                return output
            else:
                logger.debug("Channel C: Hybrid search found no results")
                return ""

        except ImportError:
            logger.debug("Channel C: RAG not installed, falling back to BM25-only")
        except Exception as e:
            logger.warning("Channel C: Hybrid search failed, falling back to BM25: %s", e)

        # Fallback to BM25-only (Phase 1 implementation)
        return await self._bm25_only_search(keywords)

    async def _bm25_only_search(self, keywords: list[str]) -> str:
        """BM25-only search using ripgrep (Phase 1 fallback)."""
        # Build ripgrep pattern (OR of all keywords)
        # Escape special regex chars
        escaped_keywords = [re.escape(kw) for kw in keywords[:5]]  # Limit to top 5
        pattern = "|".join(escaped_keywords)

        try:
            # Run ripgrep with context lines
            result = subprocess.run(
                [
                    "rg",
                    "--ignore-case",
                    "--context", "2",  # Include 2 lines before/after match
                    "--max-count", "3",  # Max 3 matches per file
                    "--no-heading",
                    "--with-filename",
                    pattern,
                    str(self.knowledge_dir),
                ],
                capture_output=True,
                text=True,
                timeout=2.0,  # 2 second timeout
            )

            if result.returncode == 0 and result.stdout:
                logger.debug(
                    "Channel C: Found BM25 matches (%d chars)", len(result.stdout)
                )
                return result.stdout
            else:
                logger.debug("Channel C: No BM25 matches found")
                return ""

        except subprocess.TimeoutExpired:
            logger.warning("Channel C: ripgrep timeout")
            return ""
        except FileNotFoundError:
            logger.warning("Channel C: ripgrep not found, falling back to Python search")
            return await self._fallback_knowledge_search(keywords)
        except Exception as e:
            logger.warning("Channel C: ripgrep failed: %s", e)
            return ""

    async def _fallback_knowledge_search(self, keywords: list[str]) -> str:
        """Fallback knowledge search using Python (when ripgrep unavailable)."""
        if not self.knowledge_dir.is_dir():
            return ""

        results: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords[:5]]

        try:
            for md_file in self.knowledge_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                content_lower = content.lower()

                # Check if any keyword matches
                if any(kw in content_lower for kw in keywords_lower):
                    # Extract matching lines with context
                    lines = content.splitlines()
                    matching_lines = []
                    for i, line in enumerate(lines):
                        if any(kw in line.lower() for kw in keywords_lower):
                            # Add line with context (±2 lines)
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            matching_lines.extend(lines[start:end])
                            matching_lines.append("---")

                    if matching_lines:
                        results.append(
                            f"{md_file.name}:\n" + "\n".join(matching_lines[:20])
                        )

                if len(results) >= 3:  # Limit to 3 files
                    break

        except Exception as e:
            logger.warning("Fallback knowledge search failed: %s", e)

        return "\n\n".join(results) if results else ""

    async def _channel_d_skill_match(self, keywords: list[str]) -> list[str]:
        """Channel D: Skill filename matching.

        Returns list of skill names (not full content) that match keywords.
        """
        if not self.skills_dir.is_dir() or not keywords:
            return []

        matched: list[str] = []
        keywords_lower = [kw.lower() for kw in keywords]

        try:
            for skill_file in self.skills_dir.glob("*.md"):
                skill_name = skill_file.stem

                # Match against filename
                if any(kw in skill_name.lower() for kw in keywords_lower):
                    matched.append(skill_name)
                    continue

                # Match against first few lines of file
                try:
                    content = skill_file.read_text(encoding="utf-8")
                    first_lines = "\n".join(content.splitlines()[:10]).lower()
                    if any(kw in first_lines for kw in keywords_lower):
                        matched.append(skill_name)
                except Exception:
                    pass

                if len(matched) >= 5:  # Limit to 5 skills
                    break

        except Exception as e:
            logger.warning("Channel D: Skill matching failed: %s", e)

        if matched:
            logger.debug("Channel D: Matched %d skills: %s", len(matched), matched)

        return matched

    # ── Dynamic budget adjustment (Phase 3) ─────────────────────

    def _classify_message_type(self, message: str, channel: str) -> str:
        """Classify message type for budget adjustment.

        Args:
            message: Message text
            channel: Message channel

        Returns:
            Message type: "greeting", "question", "request", "heartbeat"
        """
        # Heartbeat channel has fixed budget
        if channel == "heartbeat":
            return "heartbeat"

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

    def _adjust_token_budget(self, message: str, channel: str) -> int:
        """Adjust token budget based on message type.

        Args:
            message: Message text
            channel: Message channel

        Returns:
            Adjusted token budget
        """
        msg_type = self._classify_message_type(message, channel)

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
        """Extract keywords from message (simple rule-based for Phase 1).

        Future: Use morphological analysis (MeCab/Sudachi) for better quality.
        """
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

        # Split on whitespace and punctuation, keep alphanumeric + Japanese
        words = re.findall(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+", message)

        # Filter stopwords and short words
        keywords = [
            w for w in words
            if len(w) >= 2 and w.lower() not in stopwords
        ]

        # Return top 10 by length (heuristic: longer words = more specific)
        keywords.sort(key=len, reverse=True)
        return keywords[:10]

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to stay within token budget."""
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

    if result.recent_episodes:
        parts.append("### 直近の出来事")
        parts.append("")
        parts.append(result.recent_episodes)
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

    return "\n".join(parts)
