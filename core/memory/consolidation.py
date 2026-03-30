from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Memory consolidation engine — pre/post-processing helpers.

The actual consolidation (episode summarisation, knowledge extraction,
contradiction checks, etc.) is now performed by the Anima itself through
its tool-call loop (see ``Anima.run_consolidation()``).

This module retains:
- Episode and resolved-event collection (pre-processing for the Anima)
- RAG index updates and rebuilds (post-processing after the Anima finishes)
- Monthly forgetting (lifecycle.py post-processing)
- Legacy knowledge migration
- LLM output sanitisation (shared utility used by reconsolidation.py)
"""

import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.i18n import t
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.consolidation")


# ── ConsolidationEngine ────────────────────────────────────────


class ConsolidationEngine:
    """Pre/post-processing helpers for memory consolidation.

    The Anima itself now drives the consolidation loop via tool calls.
    This class provides:
    - **Pre-processing**: episode collection, resolved-event collection
    - **Post-processing**: RAG index updates/rebuilds, monthly forgetting
    - **Utilities**: knowledge file listing, LLM output sanitisation,
      legacy knowledge migration
    """

    def __init__(self, anima_dir: Path, anima_name: str, *, rag_store: Any | None = None) -> None:
        """Initialize consolidation engine.

        Args:
            anima_dir: Path to anima's directory (~/.animaworks/animas/{name})
            anima_name: Name of the anima for logging
            rag_store: Optional shared RAG vector store instance.
                When provided, avoids re-creating the singleton internally.
        """
        self.anima_dir = anima_dir
        self.anima_name = anima_name
        self._rag_store = rag_store
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # ── Legacy Migration ─────────────────────────────────────────

    def _migrate_legacy_knowledge(self) -> int:
        """Migrate legacy knowledge files to YAML frontmatter format.

        Detects knowledge files without frontmatter, creates backups, then
        rewrites them with ``---`` YAML frontmatter containing estimated
        metadata.  Controlled by a ``.migrated`` marker file so it runs
        only once per anima.

        Returns:
            Number of files migrated
        """
        marker = self.knowledge_dir / ".migrated"
        if marker.exists():
            return 0

        from core.memory.manager import MemoryManager

        # Use a lightweight MemoryManager to access frontmatter helpers
        mm = MemoryManager(self.anima_dir)

        backup_dir = self.anima_dir / "archive" / "pre_migration"
        migrated = 0

        for path in sorted(self.knowledge_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")

                # Skip files that already have frontmatter
                if text.startswith("---"):
                    continue

                # Create backup
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_dir / path.name)

                # Try to extract created_at from [AUTO-CONSOLIDATED: YYYY-MM-DD HH:MM]
                created_at = now_iso()
                ts_match = re.search(
                    r"\[AUTO-CONSOLIDATED:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]",
                    text,
                )
                if ts_match:
                    try:
                        parsed = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M")
                        created_at = parsed.isoformat()
                    except ValueError:
                        logger.debug("Failed to parse consolidation timestamp", exc_info=True)

                # Strip code fences that LLM may have wrapped around content
                content = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.MULTILINE)
                content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)
                content = content.strip()

                metadata = {
                    "created_at": created_at,
                    "confidence": 0.5,
                    "auto_consolidated": True,
                    "migrated_from_legacy": True,
                    "success_count": 0,
                    "failure_count": 0,
                    "version": 1,
                    "last_used": "",
                }

                mm.write_knowledge_with_meta(path, content, metadata)
                migrated += 1
                logger.info("Migrated legacy knowledge file: %s", path.name)

            except Exception:
                logger.exception("Failed to migrate knowledge file: %s", path.name)
                continue

        # Write marker
        try:
            marker.write_text(
                now_iso() + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to write migration marker to %s", marker, exc_info=True)
        if migrated > 0:
            logger.info(
                "Legacy knowledge migration complete for anima=%s: migrated=%d",
                self.anima_name,
                migrated,
            )
        return migrated

    # ── Episode Collection ─────────────────────────────────────

    def _collect_recent_episodes(self, hours: int = 24) -> list[dict[str, str]]:
        """Collect episode entries from the past N hours.

        Supports both standard (YYYY-MM-DD.md) and suffixed
        (YYYY-MM-DD_xxx.md) episode filenames.  Files without
        ``## HH:MM — Title`` headers are treated as single entries
        using the file's mtime for timestamp.

        Args:
            hours: Number of hours to look back

        Returns:
            List of episode entries, each with 'date', 'time', 'content'
        """
        cutoff = now_local() - timedelta(hours=hours)
        entries: list[dict[str, str]] = []

        # Check today and yesterday's episode files
        for day_offset in range(2):
            target_date = now_local().date() - timedelta(days=day_offset)
            episode_files = sorted(self.episodes_dir.glob(f"{target_date}*.md"))

            for episode_file in episode_files:
                try:
                    content = episode_file.read_text(encoding="utf-8")
                except OSError:
                    logger.warning("Failed to read episode file %s", episode_file, exc_info=True)
                    continue

                # Parse episode entries (format: ## HH:MM — Title)
                found_entries = list(
                    re.finditer(
                        r"^## (\d{2}:\d{2})\s*—\s*(.+?)(?=^##|\Z)",
                        content,
                        re.MULTILINE | re.DOTALL,
                    )
                )

                if found_entries:
                    for match in found_entries:
                        time_str = match.group(1)
                        entry_content = match.group(2).strip()

                        # Parse timestamp
                        try:
                            entry_dt = ensure_aware(
                                datetime.strptime(
                                    f"{target_date} {time_str}",
                                    "%Y-%m-%d %H:%M",
                                )
                            )

                            # Only include if within time window
                            if entry_dt >= cutoff:
                                entries.append(
                                    {
                                        "date": str(target_date),
                                        "time": time_str,
                                        "content": entry_content,
                                    }
                                )
                        except ValueError:
                            logger.warning(
                                "Failed to parse episode timestamp: %s %s",
                                target_date,
                                time_str,
                            )
                else:
                    # Fallback: treat entire file as a single entry using mtime
                    file_mtime = ensure_aware(
                        datetime.fromtimestamp(
                            episode_file.stat().st_mtime,
                        )
                    )
                    if file_mtime >= cutoff:
                        entries.append(
                            {
                                "date": str(target_date),
                                "time": file_mtime.strftime("%H:%M"),
                                "content": content.strip(),
                            }
                        )

        # Deduplicate by content prefix (first 200 chars)
        seen: set[str] = set()
        unique_entries: list[dict[str, str]] = []
        for entry in entries:
            dedup_key = entry["content"][:200].strip()
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_entries.append(entry)
        entries = unique_entries

        # Sort by datetime (newest first)
        entries.sort(
            key=lambda e: datetime.strptime(f"{e['date']} {e['time']}", "%Y-%m-%d %H:%M"),
            reverse=True,
        )

        return entries

    def _collect_resolved_events(self, hours: int = 24) -> list[dict]:
        """Collect issue_resolved events from activity log."""
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)
            entries = activity.recent(days=1, limit=50, types=["issue_resolved"])
            return [{"ts": e.ts, "content": e.content, "summary": e.summary, "meta": e.meta or {}} for e in entries]
        except Exception:
            logger.debug("Failed to collect resolved events", exc_info=True)
            return []

    # ── Error Pattern Collection ─────────────────────────────

    _ERROR_CHAR_BUDGET = 3_000
    _ERROR_ENTRY_LIMIT = 50

    def _collect_error_entries(self, hours: int = 24) -> str:
        """Collect error and failed tool_result entries for pattern analysis.

        Extracts ``error`` events and ``tool_result`` entries with
        ``result_status == "fail"`` from the activity log, formatted for
        injection into the daily consolidation prompt.

        Args:
            hours: Number of hours to look back (default 24).

        Returns:
            Formatted error summary string.  Returns a placeholder
            message when no errors are found.
        """
        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)
            entries = activity.recent(
                days=max(1, (hours + 23) // 24),
                limit=200,
                types=["error", "tool_result"],
            )
        except Exception:
            logger.debug("Failed to collect error entries", exc_info=True)
            return t("consolidation.no_errors")

        if not entries:
            return t("consolidation.no_errors")

        cutoff = now_local() - timedelta(hours=hours)
        lines: list[str] = []
        total_chars = 0
        count = 0

        for entry in entries:
            try:
                ts = ensure_aware(datetime.fromisoformat(entry.ts))
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                pass

            line = self._format_error_entry(entry)
            if line is None:
                continue

            if total_chars + len(line) + 1 > self._ERROR_CHAR_BUDGET:
                break
            lines.append(line)
            total_chars += len(line) + 1
            count += 1
            if count >= self._ERROR_ENTRY_LIMIT:
                break

        if not lines:
            return t("consolidation.no_errors")

        return "\n".join(lines)

    @staticmethod
    def _format_error_entry(entry: Any) -> str | None:
        """Format a single error or failed tool_result entry.

        Returns:
            Formatted line, or ``None`` if the entry should be skipped.
        """
        ts_short = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
        meta = entry.meta or {}

        if entry.type == "error":
            phase = meta.get("phase", "unknown")
            error_text = meta.get("error", "") or entry.summary or ""
            if len(error_text) > 120:
                error_text = error_text[:120] + "..."
            return f"[{ts_short}] ERR phase={phase}: {error_text}"

        if entry.type == "tool_result":
            if meta.get("result_status") != "fail":
                return None
            tool_name = entry.tool or "unknown"
            err_hint = (entry.content or entry.summary or "")[:100]
            return f"[{ts_short}] FAIL tool={tool_name}: {err_hint}"

        return None

    # ── Reflection Extraction ──────────────────────────────────

    @staticmethod
    def _extract_reflections_from_episodes(episodes_text: str) -> str:
        """Extract [REFLECTION] tagged entries from episode text.

        Scans for ``[REFLECTION] ... [/REFLECTION]`` blocks and returns
        them as a bullet list.  Entries shorter than 50 characters are
        filtered out (too short to be meaningful).

        Args:
            episodes_text: Raw episodes summary text.

        Returns:
            Bullet-list string of reflections, or empty string if none found.
        """
        if not episodes_text:
            return ""

        matches = re.findall(
            r"\[REFLECTION\]\s*\n?(.*?)\n?\s*\[/REFLECTION\]",
            episodes_text,
            re.DOTALL,
        )

        reflections = [m.strip() for m in matches if len(m.strip()) >= 50]

        if not reflections:
            return ""

        return "\n".join(f"- {r}" for r in reflections)

    # ── Activity Log Collection ──────────────────────────────────

    # Communication event types — these carry the most signal for consolidation.
    _COMM_TYPES = frozenset(
        {
            "message_received",
            "message_sent",
            "response_sent",
            "human_notify",
            "heartbeat_reflection",
            "channel_post",
            "error",
        }
    )

    _EXCLUDED_TOOL_NAMES = frozenset(
        {
            "read_memory_file",
            "search_memory",
            "ToolSearch",
        }
    )
    _EXCLUDED_TOOL_PREFIXES = ("mcp__aw__",)

    _OVERHEAD_TOKENS = 25_000  # system prompt + template overhead
    _CONTEXT_RATIO = 0.80
    _CHARS_PER_TOKEN = 3

    # ── Activity log budget & full collection ───────────────────

    @staticmethod
    def compute_activity_budget(model: str) -> int:
        """Compute character budget for activity log based on model context window.

        Uses 80% of context window minus overhead for prompt templates.
        """
        from core.prompt.context import resolve_context_window

        ctx = resolve_context_window(model)
        budget_tokens = int(ctx * ConsolidationEngine._CONTEXT_RATIO)
        available = max(budget_tokens - ConsolidationEngine._OVERHEAD_TOKENS, 10_000)
        return available * ConsolidationEngine._CHARS_PER_TOKEN

    @staticmethod
    def _is_excluded_tool(entry: Any) -> bool:
        """Check if a tool_result/tool_use entry should be excluded."""
        tool = getattr(entry, "tool", None) or ""
        if tool in ConsolidationEngine._EXCLUDED_TOOL_NAMES:
            return True
        return any(tool.startswith(prefix) for prefix in ConsolidationEngine._EXCLUDED_TOOL_PREFIXES)

    @staticmethod
    def _format_entry_full(entry: Any) -> str:
        """Format an activity entry with full content for consolidation.

        Unlike _format_tool_entries (meta-only), this includes the actual
        content of tool results, messages, etc.
        """
        ts_short = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
        meta = entry.meta or {}

        # Build context parts
        parts: list[str] = []
        if entry.from_person:
            parts.append(f"from:{entry.from_person}")
        if entry.to_person:
            parts.append(f"to:{entry.to_person}")
        if entry.channel:
            parts.append(f"#{entry.channel}")
        ctx = f" ({', '.join(parts)})" if parts else ""

        type_labels = {
            "message_received": "MSG_IN",
            "response_sent": "RESPONSE",
            "message_sent": "MSG_OUT",
            "human_notify": "NOTIFY",
            "heartbeat_reflection": "HB_REFLECT",
            "channel_post": "CHANNEL",
            "error": "ERROR",
            "tool_result": "TOOL_RESULT",
            "tool_use": "TOOL_USE",
            "cron_executed": "CRON",
            "memory_write": "MEM_WRITE",
            "heartbeat_start": "HB_START",
            "heartbeat_end": "HB_END",
            "consolidation_start": "CONSOL_START",
            "consolidation_end": "CONSOL_END",
        }
        label = type_labels.get(entry.type, entry.type.upper())

        # For tool entries, include tool name
        tool_name = entry.tool or ""
        if tool_name and entry.type in ("tool_result", "tool_use"):
            label = f"{label}:{tool_name}"

        # Build status suffix for tool_result
        status_suffix = ""
        if entry.type == "tool_result":
            status = meta.get("result_status", "ok")
            if status == "fail":
                status_suffix = " [FAIL]"

        # Content: prefer summary for brief context, fall back to content
        summary = entry.summary or ""
        content = entry.content or ""

        # For tool_result, content is the actual result — include it
        if entry.type == "tool_result" and content:
            text = content
        elif summary and content:
            text = f"{summary}\n{content}" if len(summary) < 200 else summary
        elif summary:
            text = summary
        elif content:
            text = content
        else:
            text = "(no content)"

        header = f"[{ts_short}] {label}{status_suffix}{ctx}"

        # If text is short, put on same line
        if len(text) <= 200 and "\n" not in text:
            return f"{header}: {text}"

        # For longer content, use indented block
        indented = "\n".join(f"  {line}" for line in text.split("\n"))
        return f"{header}:\n{indented}"

    def collect_activity_chunks(
        self,
        hours: int = 24,
        model: str | None = None,
    ) -> list[str]:
        """Collect activity entries and split into budget-sized chunks.

        Returns a list of formatted text chunks, each within the model's
        budget. Chunks are split at natural hour boundaries when possible.

        Args:
            hours: Number of hours to look back.
            model: Model name for budget calculation. Uses consolidation
                model from config if not provided.

        Returns:
            List of formatted activity text chunks. Empty list if no entries.
        """
        if model is None:
            from core.config import load_config

            cfg = load_config()
            model = cfg.consolidation.llm_model

        budget = self.compute_activity_budget(model)

        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)

            # Load all entries (no type filter, high limit)
            entries = activity.recent(
                days=max(1, (hours + 23) // 24),
                limit=10_000,
            )
        except Exception:
            logger.debug("Failed to collect activity entries", exc_info=True)
            return []

        if not entries:
            return []

        # Filter by time cutoff
        cutoff = now_local() - timedelta(hours=hours)
        filtered: list = []
        for e in entries:
            try:
                ts = ensure_aware(datetime.fromisoformat(e.ts))
                if ts >= cutoff:
                    filtered.append(e)
            except (ValueError, TypeError):
                filtered.append(e)

        if not filtered:
            return []

        # Apply exclusion list
        included: list = []
        for e in filtered:
            if e.type in ("tool_result", "tool_use") and self._is_excluded_tool(e):
                continue
            # Skip tool_use for mcp__aw__ (duplicate with tool_result)
            if e.type == "tool_use":
                tool = getattr(e, "tool", None) or ""
                for prefix in self._EXCLUDED_TOOL_PREFIXES:
                    if tool.startswith(prefix):
                        break
                else:
                    included.append(e)
                continue
            included.append(e)

        if not included:
            return []

        # Format all entries — use date+hour key for cross-day correctness
        formatted_entries: list[tuple[str, str]] = []  # (date_hour_key, formatted_text)
        for e in included:
            text = self._format_entry_full(e)
            date_hour = e.ts[:13] if len(e.ts) >= 13 else "0000-00-00T00"
            formatted_entries.append((date_hour, text))

        # Split into budget-sized chunks at hour boundaries
        return self._split_into_chunks(formatted_entries, budget)

    @staticmethod
    def _split_into_chunks(
        entries: list[tuple[str, str]],
        budget: int,
    ) -> list[str]:
        """Split formatted entries into budget-sized chunks.

        Tries to break at hour boundaries for natural segmentation.
        """
        if not entries:
            return []

        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0
        current_hour = entries[0][0]

        # Buffer for entries in the current hour
        hour_buffer: list[str] = []
        hour_buffer_size = 0

        for hour_key, text in entries:
            entry_size = len(text) + 1  # +1 for newline

            if hour_key != current_hour:
                # Hour boundary — check if we should start a new chunk
                if current_size + hour_buffer_size > budget and current_lines:
                    # Flush current chunk, start new one with buffer
                    chunks.append("\n".join(current_lines))
                    current_lines = list(hour_buffer)
                    current_size = hour_buffer_size
                else:
                    # Add buffer to current chunk
                    current_lines.extend(hour_buffer)
                    current_size += hour_buffer_size

                hour_buffer = []
                hour_buffer_size = 0
                current_hour = hour_key

            # If single entry exceeds budget, truncate it
            if entry_size > budget:
                text = text[: budget - 100] + "\n  ... (truncated)"
                entry_size = len(text) + 1

            hour_buffer.append(text)
            hour_buffer_size += entry_size

        # Flush remaining hour buffer
        if current_size + hour_buffer_size > budget and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = list(hour_buffer)
        else:
            current_lines.extend(hour_buffer)

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    @staticmethod
    def merge_timeline_parts(parts: list[str]) -> str:
        """Merge multiple structured timeline parts into a single episode.

        Concatenates timeline parts in order, deduplicating any repeated
        section headers (## HH:MM-HH:MM Title) that might appear at
        chunk boundaries.
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        seen_headers: set[str] = set()
        merged_lines: list[str] = []

        for part in parts:
            for line in part.split("\n"):
                # Detect markdown ## headers for dedup
                stripped = line.strip()
                if stripped.startswith("## "):
                    if stripped in seen_headers:
                        continue
                    seen_headers.add(stripped)
                merged_lines.append(line)

        return "\n".join(merged_lines)

    def defrag_recent_episodes(self, days: int = 3) -> list[Path]:
        """Identify recent episode files that need restructuring.

        Returns paths to episode files from the past N days that exist
        and could benefit from defragmentation.

        .. note::
            This is a discovery helper only. The actual defrag rewriting
            (converting old free-form episodes into structured timelines)
            is not yet wired into the consolidation pipeline.
        """
        today = now_local().date()
        targets: list[Path] = []
        for offset in range(days):
            target_date = today - timedelta(days=offset)
            for ep_file in sorted(self.episodes_dir.glob(f"{target_date}*.md")):
                if ep_file.exists() and ep_file.stat().st_size > 0:
                    targets.append(ep_file)
        return targets

    def _collect_activity_entries(self, hours: int = 24) -> str:
        """Collect recent activity log entries for consolidation input.

        Uses a two-phase budget allocation:
          1. Communication events first (messages, responses, errors, etc.)
          2. Remaining budget for ``tool_result`` only — fail entries get
             100-char content, ok entries are meta-only.
             ``tool_use`` events are excluded (redundant with tool_result).

        Args:
            hours: Number of hours to look back (default 24).

        Returns:
            Formatted activity log summary string, truncated to
            approximately 4000 tokens (12000 chars).  Empty string
            if no matching entries are found.
        """
        _CHAR_BUDGET = 12_000  # ~4000 tokens

        try:
            from core.memory.activity import ActivityLogger

            activity = ActivityLogger(self.anima_dir)
            target_types = [
                "message_received",
                "response_sent",
                "heartbeat_reflection",
                "channel_post",
                "error",
                "tool_result",
            ]
            entries = activity.recent(
                days=max(1, (hours + 23) // 24),
                limit=200,
                types=target_types,
            )

            if not entries:
                return ""

            # Filter by hours cutoff
            cutoff = now_local() - timedelta(hours=hours)
            filtered: list = []
            for e in entries:
                try:
                    ts = ensure_aware(datetime.fromisoformat(e.ts))
                    if ts >= cutoff:
                        filtered.append(e)
                except (ValueError, TypeError):
                    filtered.append(e)

            if not filtered:
                return ""

            # Phase 1: Communication events
            comm_entries = [e for e in filtered if e.type in self._COMM_TYPES]
            tool_result_entries = [e for e in filtered if e.type == "tool_result"]

            lines: list[str] = []
            total_chars = 0

            for entry in comm_entries:
                line = self._format_comm_entry(entry)
                if total_chars + len(line) + 1 > _CHAR_BUDGET:
                    break
                lines.append(line)
                total_chars += len(line) + 1

            # Phase 2: tool_result with remaining budget
            remaining = _CHAR_BUDGET - total_chars
            if remaining > 0 and tool_result_entries:
                tool_lines = self._format_tool_entries(tool_result_entries, remaining)
                lines.extend(tool_lines)

            return "\n".join(lines)

        except Exception:
            logger.debug("Failed to collect activity entries", exc_info=True)
            return ""

    @staticmethod
    def _format_comm_entry(entry: Any) -> str:
        """Format a communication entry as a readable line."""
        ts_short = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
        text = entry.summary or entry.content
        if len(text) > 300:
            text = text[:300] + "..."

        parts: list[str] = []
        if entry.from_person:
            parts.append(f"from:{entry.from_person}")
        if entry.to_person:
            parts.append(f"to:{entry.to_person}")
        if entry.channel:
            parts.append(f"#{entry.channel}")
        ctx = f" ({', '.join(parts)})" if parts else ""

        type_map: dict[str, str] = {
            "message_received": "MSG<",
            "message_sent": "MSG>",
            "response_sent": "RESP>",
            "human_notify": "NOTIFY",
            "heartbeat_reflection": "HB",
            "channel_post": "CH.W",
            "error": "ERR",
        }
        icon = type_map.get(entry.type, "•")

        return f"[{ts_short}] {icon} {entry.type}{ctx}: {text}"

    @staticmethod
    def _format_tool_entries(entries: list, budget_chars: int) -> list[str]:
        """Format tool_result entries with budget-aware rendering.

        Fail entries include up to 100 chars of content for debugging.
        Ok entries are rendered as compact meta-only lines matching the
        Priming format: ``[HH:MM] TRES tool → ok (N件, XKB)``.
        """
        lines: list[str] = []
        total = 0

        for entry in entries:
            ts = entry.ts[11:16] if len(entry.ts) >= 16 else entry.ts
            tool = entry.tool or "unknown"
            meta = entry.meta or {}
            status = meta.get("result_status", "ok")

            if status == "fail":
                err_hint = (entry.content or "")[:100]
                line = f"[{ts}] TRES {tool} → fail: {err_hint}"
            else:
                result_bytes = meta.get("result_bytes", 0)
                result_count = meta.get("result_count")

                if result_bytes >= 1024:
                    size_str = f"{result_bytes / 1024:.1f}KB"
                else:
                    size_str = f"{result_bytes}B"

                detail_parts: list[str] = []
                if result_count is not None:
                    detail_parts.append(f"{result_count}件")
                detail_parts.append(size_str)

                detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
                line = f"[{ts}] TRES {tool} → ok{detail}"

            if total + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            total += len(line) + 1

        return lines

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def _sanitize_llm_output(text: str) -> str:
        """Remove code fences from LLM output.

        LLMs sometimes wrap their entire response in ```markdown fences.
        This method strips those wrapper fences while preserving any
        intentional code blocks within the content.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned text with wrapper code fences removed
        """
        text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def _is_archive_dir(rel: Path) -> bool:
        """Check if a path is inside an archive subdirectory.

        Handles all known archive directory naming conventions:
        ``archive/``, ``_archived/``, ``.archive/``.
        """
        if not rel.parts:
            return False
        first = rel.parts[0].lower().lstrip("_").lstrip(".")
        return first == "archive" or first == "archived"

    def _list_knowledge_files(self) -> list[str]:
        """List all existing knowledge files.

        Returns:
            List of knowledge file paths (relative to knowledge/)
        """
        if not self.knowledge_dir.exists():
            return []

        files = []
        for path in self.knowledge_dir.rglob("*.md"):
            rel_path = path.relative_to(self.knowledge_dir)
            files.append(str(rel_path))

        return sorted(files)

    # ── Merge Candidate Detection ─────────────────────────────────

    def _list_knowledge_files_with_meta(self) -> list[dict[str, Any]]:
        """List all existing knowledge files with frontmatter metadata.

        Returns:
            List of dicts with keys: path, created_at, confidence,
            auto_consolidated, success_count.  Files in archive/ are excluded.
        """
        if not self.knowledge_dir.exists():
            return []

        from core.memory.frontmatter import parse_frontmatter

        results: list[dict[str, Any]] = []
        for path in sorted(self.knowledge_dir.rglob("*.md")):
            rel = path.relative_to(self.knowledge_dir)
            if self._is_archive_dir(rel):
                continue

            meta_fields: dict[str, Any] = {"path": str(rel)}
            try:
                text = path.read_text(encoding="utf-8")
                meta, _ = parse_frontmatter(text)
                meta_fields["created_at"] = meta.get("created_at", "")
                meta_fields["confidence"] = meta.get("confidence", "")
                meta_fields["auto_consolidated"] = meta.get("auto_consolidated", False)
                meta_fields["success_count"] = meta.get("success_count", 0)
            except Exception:
                pass
            results.append(meta_fields)

        return results

    def _find_merge_candidates(
        self,
        similarity_threshold: float = 0.75,
        max_pairs: int = 20,
    ) -> list[tuple[str, str, float]]:
        """Find knowledge file pairs that are candidates for merging.

        Uses RAG vector similarity to detect semantically similar files.
        All knowledge files are eligible (no low-activation requirement).
        Files in archive/ subdirectories are excluded.

        Args:
            similarity_threshold: Minimum vector similarity for a pair (0.0-1.0).
            max_pairs: Maximum number of pairs to return.

        Returns:
            List of (file_a, file_b, similarity) tuples, sorted by
            similarity descending.  Paths are relative to knowledge/.
        """
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.retriever import MemoryRetriever
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store(self.anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable for merge candidate search")
                return []
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)
            retriever = MemoryRetriever(vector_store, indexer, self.knowledge_dir)
        except (ImportError, Exception) as exc:
            logger.debug("RAG not available for merge candidate search: %s", exc)
            return []

        # Read all non-archived knowledge files
        from core.memory.frontmatter import parse_frontmatter

        file_contents: dict[str, str] = {}
        for path in sorted(self.knowledge_dir.rglob("*.md")):
            rel = path.relative_to(self.knowledge_dir)
            if self._is_archive_dir(rel):
                continue
            try:
                text = path.read_text(encoding="utf-8")
                _, body = parse_frontmatter(text)
                if body.strip():
                    file_contents[str(rel)] = body.strip()
            except Exception:
                continue

        if len(file_contents) < 2:
            return []

        # Query each file against RAG to find similar peers
        seen_pairs: set[tuple[str, str]] = set()
        candidates: list[tuple[str, str, float]] = []

        for rel_path, content in file_contents.items():
            try:
                results = retriever.search(
                    query=content[:500],
                    anima_name=self.anima_name,
                    memory_type="knowledge",
                    top_k=5,
                )
            except Exception:
                continue

            for result in results:
                raw_sim = getattr(result, "source_scores", {}).get("vector", result.score)
                if raw_sim < similarity_threshold:
                    continue

                source_file = str(result.metadata.get("source_file", ""))
                if not source_file:
                    continue

                # Normalise to relative path under knowledge/
                if source_file.startswith("knowledge/"):
                    match_rel = source_file[len("knowledge/") :]
                elif source_file.startswith("knowledge\\"):
                    match_rel = source_file[len("knowledge\\") :]
                else:
                    match_rel = source_file

                match_rel_path = Path(match_rel)
                if match_rel == rel_path:
                    continue
                if self._is_archive_dir(match_rel_path):
                    continue
                # Skip if the matched file isn't in our content map
                if match_rel not in file_contents:
                    continue

                pair_key = tuple(sorted([rel_path, match_rel]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidates.append((rel_path, match_rel, raw_sim))

        # Sort by similarity descending, cap at max_pairs
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_pairs]

    # ── Origin detection ─────────────────────────────────────────

    _EXTERNAL_ORIGINS = frozenset({"external_web", "mixed", "consolidation_external"})

    def _has_external_origin_in_files(self, filenames: list[str]) -> bool:
        """Check if any knowledge file in *filenames* contains an external origin.

        Reads the YAML frontmatter ``origin:`` field.  Returns ``True`` if
        at least one file has an origin that indicates external (untrusted)
        data provenance.
        """
        from core.memory.frontmatter import parse_frontmatter

        for filename in filenames:
            filepath = self.knowledge_dir / filename
            if not filepath.exists():
                continue
            try:
                text = filepath.read_text(encoding="utf-8")
                meta, _ = parse_frontmatter(text)
                origin = meta.get("origin", "")
                if origin in self._EXTERNAL_ORIGINS:
                    return True
            except Exception:
                continue
        return False

    # ── RAG Index ────────────────────────────────────────────────

    def _update_rag_index(
        self, filenames: list[str], *, origin: str = "consolidation", source_files: list[str] | None = None
    ) -> None:
        """Update RAG index for the specified knowledge files.

        Args:
            filenames: List of knowledge file names (relative to knowledge/)
            origin: Provenance origin for the indexed chunks.
            source_files: Optional list of input knowledge files used in
                consolidation.  When provided and any contain external
                origins, *origin* is downgraded to ``consolidation_external``.
        """
        if not filenames:
            return

        effective_origin = origin
        if source_files and origin == "consolidation":
            if self._has_external_origin_in_files(source_files):
                effective_origin = "consolidation_external"
                logger.info(
                    "Downgrading consolidation origin to 'consolidation_external' due to external-origin input files",
                )

        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store(self.anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, skipping index update")
                return
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)

            for filename in filenames:
                filepath = self.knowledge_dir / filename
                if filepath.exists():
                    indexer.index_file(filepath, memory_type="knowledge", origin=effective_origin)
                    logger.debug("Updated RAG index for: %s (origin=%s)", filename, effective_origin)

        except ImportError:
            logger.debug("RAG not available, skipping index update")
        except Exception as e:
            logger.warning("Failed to update RAG index: %s", e)

    def _rebuild_rag_index(self) -> None:
        """Rebuild RAG index for all knowledge and episode files."""
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = self._rag_store or get_vector_store(self.anima_name)
            if vector_store is None:
                logger.debug("RAG vector store unavailable, skipping index rebuild")
                return
            indexer = MemoryIndexer(vector_store, self.anima_name, self.anima_dir)

            # Re-index all knowledge files, respecting per-file origin
            from core.memory.frontmatter import parse_frontmatter as _parse_fm

            for knowledge_file in self.knowledge_dir.rglob("*.md"):
                file_origin = "consolidation"
                try:
                    text = knowledge_file.read_text(encoding="utf-8")
                    meta, _ = _parse_fm(text)
                    file_origin = meta.get("origin", file_origin) or file_origin
                except Exception:
                    pass
                indexer.index_file(knowledge_file, memory_type="knowledge", origin=file_origin)
                logger.debug("Re-indexed knowledge: %s (origin=%s)", knowledge_file.name, file_origin)

            # Re-index all episode files (origin unknown on rebuild)
            for episode_file in self.episodes_dir.glob("*.md"):
                indexer.index_file(episode_file, memory_type="episodes")
                logger.debug("Re-indexed episode: %s", episode_file.name)

            logger.info("RAG index rebuild complete for anima=%s", self.anima_name)

        except ImportError:
            logger.debug("RAG not available, skipping index rebuild")
        except Exception:
            logger.exception("Failed to rebuild RAG index")

    # ── Monthly Forgetting ──────────────────────────────────────

    async def monthly_forget(self) -> dict[str, Any]:
        """Perform monthly forgetting: archive and remove forgotten memories.

        This is the final stage of the forgetting pipeline, removing
        memories that have remained at low activation for extended periods.
        Also cleans up old procedure version archives.
        """
        logger.info("Starting monthly forgetting for anima=%s", self.anima_name)
        try:
            from core.memory.forgetting import ForgettingEngine

            forgetter = ForgettingEngine(self.anima_dir, self.anima_name)
            result = forgetter.complete_forgetting()

            # Clean up old procedure version archives
            try:
                archive_result = forgetter.cleanup_procedure_archives()
                result["procedure_archive_cleanup"] = archive_result
                logger.info(
                    "Procedure archive cleanup for anima=%s: deleted=%d, kept=%d",
                    self.anima_name,
                    archive_result.get("deleted_count", 0),
                    archive_result.get("kept_count", 0),
                )
            except Exception:
                logger.exception(
                    "Procedure archive cleanup failed for anima=%s",
                    self.anima_name,
                )

            # Rebuild RAG index after deletions
            self._rebuild_rag_index()

            logger.info(
                "Monthly forgetting complete for anima=%s: forgotten=%d, archived=%d files",
                self.anima_name,
                result.get("forgotten_chunks", 0),
                len(result.get("archived_files", [])),
            )
            return result

        except Exception:
            logger.exception("Monthly forgetting failed for anima=%s", self.anima_name)
            return {"forgotten_chunks": 0, "archived_files": [], "error": True}
