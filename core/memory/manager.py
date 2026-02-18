from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import logging
import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_company_dir, get_shared_dir
from core.schemas import ModelConfig, SkillMeta

logger = logging.getLogger("animaworks.memory")


# ── Skill matching ────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """NFKC normalization + lowercase for keyword matching."""
    return unicodedata.normalize("NFKC", text).lower()


def _extract_bracket_keywords(desc_norm: str) -> list[str]:
    """Extract keywords from 「」-delimited tokens."""
    return re.findall(r"「(.+?)」", desc_norm)


def _extract_comma_keywords(desc_norm: str) -> list[str]:
    """Extract keywords by splitting on commas, periods, and newlines.

    Used as fallback when no 「」 brackets are present.
    Returns short phrases (2–20 chars) that are likely meaningful keywords.
    """
    segments = re.split(r"[、,。.\n]", desc_norm)
    return [s.strip() for s in segments if 2 <= len(s.strip()) <= 20]


def _match_tier1(desc_norm: str, message_norm: str) -> bool:
    """Tier 1: Bracket keyword match (「」) + comma/delimiter keyword match.

    Returns True if any keyword extracted from the description is a
    substring of the message.
    """
    # Primary: 「」 bracket keywords
    keywords = _extract_bracket_keywords(desc_norm)
    if keywords:
        return any(kw in message_norm for kw in keywords)
    # Fallback: comma/delimiter separated keywords
    keywords = _extract_comma_keywords(desc_norm)
    if keywords:
        return any(kw in message_norm for kw in keywords)
    return False


# Common English stop words to exclude from Tier 2 vocabulary matching.
# These words appear in almost any text and would cause false positives.
_TIER2_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "use", "used",
    "when", "into", "also", "can", "are", "was", "has", "have", "had",
    "not", "but", "its", "any", "all", "each", "more", "such", "than",
    "tool", "file", "new", "via", "etc", "using", "other",
})


def _match_tier2(desc_norm: str, message_norm: str) -> bool:
    """Tier 2: Description vocabulary match.

    Extracts words (≥3 chars) from description and checks if ≥2 appear
    in the message. This handles English descriptions and natural-language
    style descriptions without explicit keyword delimiters.

    Stop words are filtered to avoid false positives from common English
    words. Word boundary matching (``\\b``) is used for ASCII words to
    prevent substring collisions (e.g. 'git' matching inside 'digital').
    """
    # Split on whitespace and punctuation, keep meaningful tokens
    raw_words = re.findall(r"[\w]{3,}", desc_norm)
    words = [w for w in raw_words if w not in _TIER2_STOP_WORDS]
    if not words:
        return False
    # Require at least 2 word matches to avoid false positives
    match_count = 0
    for w in words:
        if w.isascii():
            # Word boundary match for ASCII to avoid substring collisions
            if re.search(rf"\b{re.escape(w)}\b", message_norm):
                match_count += 1
        else:
            # Substring match for CJK (no word boundaries in Japanese)
            if w in message_norm:
                match_count += 1
    return match_count >= 2


def match_skills_by_description(
    message: str,
    skills: list[SkillMeta],
    *,
    retriever: object | None = None,
    anima_name: str = "",
) -> list[SkillMeta]:
    """Return skills whose description matches the message (3-tier).

    Tier 1: 「」-delimited and comma/delimiter keyword substring match.
    Tier 2: Description vocabulary match (≥2 words overlap).
    Tier 3: Vector search via RAG retriever (semantic similarity).

    Each tier is applied only to skills not yet matched by prior tiers.
    Results are deduplicated and returned in tier priority order.
    """
    if not message:
        return []
    message_norm = _normalize_text(message)
    matched: list[SkillMeta] = []
    matched_names: set[str] = set()
    remaining: list[SkillMeta] = []

    # ── Tier 1: Bracket / comma keyword match ──────────────
    for skill in skills:
        if not skill.description:
            remaining.append(skill)
            continue
        desc_norm = _normalize_text(skill.description)
        if _match_tier1(desc_norm, message_norm):
            matched.append(skill)
            matched_names.add(skill.name)
        else:
            remaining.append(skill)

    # ── Tier 2: Description vocabulary match ───────────────
    still_remaining: list[SkillMeta] = []
    for skill in remaining:
        if not skill.description:
            still_remaining.append(skill)
            continue
        desc_norm = _normalize_text(skill.description)
        if _match_tier2(desc_norm, message_norm):
            if skill.name not in matched_names:
                matched.append(skill)
                matched_names.add(skill.name)
        else:
            still_remaining.append(skill)

    # ── Tier 3: Vector search (semantic match) ─────────────
    if retriever is not None and anima_name and still_remaining:
        try:
            vector_matched = _match_tier3_vector(
                message, still_remaining, retriever, anima_name,
            )
            for skill in vector_matched:
                if skill.name not in matched_names:
                    matched.append(skill)
                    matched_names.add(skill.name)
        except Exception as e:
            logger.warning("Tier 3 vector search failed: %s", e)

    return matched


def _match_tier3_vector(
    message: str,
    candidates: list[SkillMeta],
    retriever: object,
    anima_name: str,
    top_k: int = 3,
    min_score: float = 0.88,
) -> list[SkillMeta]:
    """Tier 3: Use RAG vector search to find semantically matching skills.

    Searches the personal skills collection and matches results back to
    candidate SkillMeta objects by file path / name.

    Searches both the personal skills collection and the shared
    common_skills collection (``shared_common_skills``) in ChromaDB.
    """
    from core.memory.rag.retriever import MemoryRetriever

    if not isinstance(retriever, MemoryRetriever):
        return []

    # Search in 'skills' memory type (personal + shared common_skills)
    results = retriever.search(
        query=message,
        anima_name=anima_name,
        memory_type="skills",
        top_k=top_k,
        include_shared=True,
    )

    # Build path-to-skill lookup from candidates
    candidate_by_path: dict[str, SkillMeta] = {}
    for skill in candidates:
        candidate_by_path[str(skill.path)] = skill
        # Also index by filename stem for fuzzy matching
        candidate_by_path[skill.path.stem] = skill
        candidate_by_path[skill.name] = skill

    matched: list[SkillMeta] = []
    seen: set[str] = set()
    for r in results:
        if r.score < min_score:
            continue
        # Try to match by file_path or source_file metadata
        file_path = r.metadata.get("file_path", "") or r.metadata.get("source_file", "")
        skill = candidate_by_path.get(str(file_path))
        if skill is None and file_path:
            # Try stem matching from file_path
            stem = Path(file_path).stem
            skill = candidate_by_path.get(stem)
        if skill and skill.name not in seen:
            matched.append(skill)
            seen.add(skill.name)
    return matched


class MemoryManager:
    """File-system based library memory.

    The LLM searches memory autonomously via Grep/Read tools.
    This class handles the Python-side read/write operations.
    """

    def __init__(self, anima_dir: Path, base_dir: Path | None = None) -> None:
        self.anima_dir = anima_dir
        self.company_dir = get_company_dir()
        self.common_skills_dir = get_common_skills_dir()
        self.common_knowledge_dir = get_common_knowledge_dir()
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.procedures_dir = anima_dir / "procedures"
        self.skills_dir = anima_dir / "skills"
        self.state_dir = anima_dir / "state"
        for d in (
            self.episodes_dir,
            self.knowledge_dir,
            self.procedures_dir,
            self.skills_dir,
            self.state_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # RAG indexer is initialized lazily on first access to avoid
        # heavy model loading (sentence-transformers / CUDA) during
        # DigitalAnima construction.  See: _get_indexer()
        self._indexer = None
        self._indexer_initialized = False

    # ── RAG indexer initialization ────────────────────────

    def _init_indexer(self) -> None:
        """Initialize RAG indexer if dependencies are available.

        Called lazily by ``_get_indexer()`` on first access.
        Uses process-level singletons for ChromaVectorStore and embedding
        model to avoid costly repeated initialization.

        Also ensures the ``shared_common_knowledge`` collection is indexed
        from ``~/.animaworks/common_knowledge/``.  The hash-based dedup in
        :meth:`MemoryIndexer.index_file` makes repeated calls a no-op.
        """
        self._indexer_initialized = True
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store()
            anima_name = self.anima_dir.name
            self._indexer = MemoryIndexer(vector_store, anima_name, self.anima_dir)
            logger.debug("RAG indexer initialized for anima=%s", anima_name)

            # Ensure shared collections exist
            self._ensure_shared_knowledge_indexed(vector_store)
            self._ensure_shared_skills_indexed(vector_store)
        except ImportError:
            logger.debug("RAG dependencies not installed, indexing disabled")
        except Exception as e:
            logger.warning("Failed to initialize RAG indexer: %s", e)

    def _ensure_shared_knowledge_indexed(self, vector_store) -> None:
        """Index common_knowledge/ into ``shared_common_knowledge`` collection.

        Uses the existing hash-based dedup so repeated calls (once per
        anima process) are effectively no-ops after the first indexing.
        """
        ck_dir = self.common_knowledge_dir
        if not ck_dir.is_dir() or not any(ck_dir.rglob("*.md")):
            logger.debug("No common_knowledge files found, skipping shared indexing")
            return

        try:
            from core.memory.rag import MemoryIndexer
            from core.paths import get_data_dir

            data_dir = get_data_dir()
            shared_indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=data_dir,
                collection_prefix="shared",
                embedding_model=self._indexer.embedding_model if self._indexer else None,
            )
            indexed = shared_indexer.index_directory(ck_dir, "common_knowledge")
            if indexed > 0:
                logger.info(
                    "Indexed %d chunks into shared_common_knowledge", indexed,
                )
        except Exception as e:
            logger.warning("Failed to index shared common_knowledge: %s", e)

    def _ensure_shared_skills_indexed(self, vector_store) -> None:
        """Index common_skills/ into ``shared_common_skills`` collection.

        Uses the existing hash-based dedup so repeated calls (once per
        anima process) are effectively no-ops after the first indexing.
        """
        cs_dir = self.common_skills_dir
        if not cs_dir.is_dir() or not any(cs_dir.glob("*.md")):
            logger.debug("No common_skills files found, skipping shared skills indexing")
            return

        try:
            from core.memory.rag import MemoryIndexer
            from core.paths import get_data_dir

            data_dir = get_data_dir()
            shared_indexer = MemoryIndexer(
                vector_store,
                anima_name="shared",
                anima_dir=data_dir,
                collection_prefix="shared",
                embedding_model=self._indexer.embedding_model if self._indexer else None,
            )
            indexed = shared_indexer.index_directory(cs_dir, "common_skills")
            if indexed > 0:
                logger.info(
                    "Indexed %d chunks into shared_common_skills", indexed,
                )
        except Exception as e:
            logger.warning("Failed to index shared common_skills: %s", e)

    def _get_indexer(self):
        """Return the RAG indexer, initializing it on first call."""
        if not self._indexer_initialized:
            self._init_indexer()
        return self._indexer

    # ── Read ──────────────────────────────────────────────

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_company_vision(self) -> str:
        return self._read(self.company_dir / "vision.md")

    def read_identity(self) -> str:
        return self._read(self.anima_dir / "identity.md")

    def read_injection(self) -> str:
        return self._read(self.anima_dir / "injection.md")

    def read_specialty_prompt(self) -> str:
        """Read the role-specific specialty prompt."""
        return self._read(self.anima_dir / "specialty_prompt.md")

    def read_permissions(self) -> str:
        return self._read(self.anima_dir / "permissions.md")

    def read_current_state(self) -> str:
        return self._read(self.state_dir / "current_task.md") or "status: idle"

    def read_pending(self) -> str:
        return self._read(self.state_dir / "pending.md")

    def read_heartbeat_config(self) -> str:
        return self._read(self.anima_dir / "heartbeat.md")

    def load_recent_heartbeat_summary(self, limit: int = 5) -> str:
        """Load recent heartbeat summaries for dialogue context injection."""
        import json as _json

        history_dir = self.anima_dir / "shortterm" / "heartbeat_history"
        if not history_dir.exists():
            return ""

        lines: list[str] = []
        for f in sorted(history_dir.glob("*.jsonl"), reverse=True)[:3]:
            file_lines = f.read_text(encoding="utf-8").strip().splitlines()
            lines = file_lines + lines
            if len(lines) >= limit:
                break

        if not lines:
            return ""

        entries: list[str] = []
        for line in lines[-limit:]:
            try:
                e = _json.loads(line)
                ts = e.get("timestamp", "?")
                action = e.get("action", "?")
                summary = e.get("summary", "")[:300]
                if "HEARTBEAT_OK" not in summary:
                    entries.append(f"- {ts}: [{action}] {summary}")
            except (_json.JSONDecodeError, KeyError):
                continue

        return "\n".join(entries)

    def read_cron_config(self) -> str:
        return self._read(self.anima_dir / "cron.md")

    def read_model_config(self) -> ModelConfig:
        """Load model config from unified config.json, with config.md fallback."""
        from core.config import (
            get_config_path,
            load_config,
            resolve_execution_mode,
            resolve_anima_config,
        )

        config_path = get_config_path()
        if config_path.exists():
            config = load_config(config_path)
            anima_name = self.anima_dir.name
            resolved, credential = resolve_anima_config(config, anima_name, anima_dir=self.anima_dir)
            # Derive env var name from credential name (e.g. "anthropic" -> "ANTHROPIC_API_KEY")
            cred_name = resolved.credential
            api_key_env = f"{cred_name.upper()}_API_KEY"
            mode = resolve_execution_mode(
                config, resolved.model, resolved.execution_mode,
            )
            return ModelConfig(
                model=resolved.model,
                fallback_model=resolved.fallback_model,
                max_tokens=resolved.max_tokens,
                max_turns=resolved.max_turns,
                api_key=credential.api_key or None,
                api_key_env=api_key_env,
                api_base_url=credential.base_url,
                context_threshold=resolved.context_threshold,
                max_chains=resolved.max_chains,
                conversation_history_threshold=resolved.conversation_history_threshold,
                execution_mode=resolved.execution_mode,
                supervisor=resolved.supervisor,
                speciality=resolved.speciality,
                resolved_mode=mode,
                thinking=resolved.thinking,
            )

        # Legacy fallback: parse config.md
        return self._read_model_config_from_md()

    def _read_model_config_from_md(self) -> ModelConfig:
        """Legacy parser for config.md (fallback when config.json absent)."""
        raw = self._read(self.anima_dir / "config.md")
        if not raw:
            return ModelConfig()

        # Ignore 備考/設定例 sections to avoid matching example lines
        for marker in ("## 備考", "### 設定例"):
            idx = raw.find(marker)
            if idx != -1:
                raw = raw[:idx]

        def _extract(key: str, default: str) -> str:
            m = re.search(rf"^-\s*{key}\s*:\s*(.+)$", raw, re.MULTILINE)
            return m.group(1).strip() if m else default

        defaults = ModelConfig()
        base_url = _extract("api_base_url", "")
        return ModelConfig(
            model=_extract("model", defaults.model),
            fallback_model=_extract("fallback_model", "") or defaults.fallback_model,
            max_tokens=int(_extract("max_tokens", str(defaults.max_tokens))),
            max_turns=int(_extract("max_turns", str(defaults.max_turns))),
            api_key_env=_extract("api_key_env", defaults.api_key_env),
            api_base_url=base_url or defaults.api_base_url,
        )

    def resolve_api_key(self, config: ModelConfig | None = None) -> str | None:
        """Resolve the actual API key (config.json direct value, then env var fallback)."""
        cfg = config or self.read_model_config()
        if cfg.api_key:
            return cfg.api_key
        return os.environ.get(cfg.api_key_env)

    def read_bootstrap(self) -> str:
        return self._read(self.anima_dir / "bootstrap.md")

    def read_today_episodes(self) -> str:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        return self._read(path)

    def read_file(self, relpath: str) -> str:
        """Read an arbitrary file relative to anima_dir."""
        return self._read(self.anima_dir / relpath)

    def list_knowledge_files(self) -> list[str]:
        return [f.stem for f in sorted(self.knowledge_dir.glob("*.md"))]

    def list_episode_files(self) -> list[str]:
        return [
            f.stem for f in sorted(self.episodes_dir.glob("*.md"), reverse=True)
        ]

    def list_procedure_files(self) -> list[str]:
        return [f.stem for f in sorted(self.procedures_dir.glob("*.md"))]

    def list_skill_files(self) -> list[str]:
        return [f.stem for f in sorted(self.skills_dir.glob("*.md"))]

    @staticmethod
    def _extract_skill_meta(path: Path, *, is_common: bool = False) -> "SkillMeta":
        """Extract SkillMeta from a skill file's YAML frontmatter.

        Supports Claude Code format (name + description frontmatter only).
        Falls back to filename stem and empty description if no frontmatter.
        """
        from core.schemas import SkillMeta

        text = path.read_text(encoding="utf-8")
        name = path.stem
        description = ""

        # Parse YAML frontmatter (--- delimited)
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    fm = yaml.safe_load(parts[1])
                    if isinstance(fm, dict):
                        name = fm.get("name", name)
                        description = fm.get("description", "")
                        if description:
                            description = str(description).strip()
                except Exception:
                    pass

        # Fallback: extract from ## 概要 section (legacy format)
        if not description:
            in_overview = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped == "## 概要":
                    in_overview = True
                    continue
                if in_overview:
                    if stripped.startswith("#"):
                        break
                    if stripped:
                        description = stripped
                        break

        return SkillMeta(
            name=name,
            description=description,
            path=path,
            is_common=is_common,
        )

    def list_skill_metas(self) -> list["SkillMeta"]:
        """Return SkillMeta for each personal skill."""
        return [
            self._extract_skill_meta(f, is_common=False)
            for f in sorted(self.skills_dir.glob("*.md"))
        ]

    def list_common_skill_metas(self) -> list["SkillMeta"]:
        """Return SkillMeta for each common skill."""
        if not self.common_skills_dir.is_dir():
            return []
        return [
            self._extract_skill_meta(f, is_common=True)
            for f in sorted(self.common_skills_dir.glob("*.md"))
        ]

    def list_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (name, description) for each personal skill.

        Compatibility wrapper around list_skill_metas().
        """
        return [(m.name, m.description) for m in self.list_skill_metas()]

    def list_common_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (name, description) for each common skill.

        Compatibility wrapper around list_common_skill_metas().
        """
        return [(m.name, m.description) for m in self.list_common_skill_metas()]

    # ── Cron log ──────────────────────────────────────────

    _CRON_LOG_DIR = "state/cron_logs"
    _CRON_LOG_MAX_LINES = 50

    def append_cron_log(
        self, task_name: str, *, summary: str, duration_ms: int,
    ) -> None:
        """Append a cron execution result to the daily log."""
        log_dir = self.anima_dir / self._CRON_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{date.today().isoformat()}.jsonl"

        import json as _json
        entry = _json.dumps({
            "timestamp": datetime.now().isoformat(),
            "task": task_name,
            "summary": summary[:500],
            "duration_ms": duration_ms,
        }, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

        # Keep file bounded
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > self._CRON_LOG_MAX_LINES:
            path.write_text(
                "\n".join(lines[-self._CRON_LOG_MAX_LINES:]) + "\n",
                encoding="utf-8",
            )

    def append_cron_command_log(
        self,
        task_name: str,
        *,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_ms: int,
    ) -> None:
        """Append a command-type cron execution result to the daily log.

        Logs include exit code, line counts, and previews (first+last 5 lines).
        """
        log_dir = self.anima_dir / self._CRON_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{date.today().isoformat()}.jsonl"

        # Count lines
        stdout_lines_list = stdout.splitlines()
        stderr_lines_list = stderr.splitlines()
        stdout_line_count = len(stdout_lines_list)
        stderr_line_count = len(stderr_lines_list)

        # Generate preview: first 5 + last 5 lines, max 1000 chars total
        def make_preview(lines_list: list[str]) -> str:
            if not lines_list:
                return ""
            if len(lines_list) <= 10:
                preview = "\n".join(lines_list)
            else:
                preview = "\n".join(lines_list[:5] + ["..."] + lines_list[-5:])
            return preview[:1000]

        stdout_preview = make_preview(stdout_lines_list)
        stderr_preview = make_preview(stderr_lines_list)

        import json as _json
        entry = _json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "task": task_name,
                "exit_code": exit_code,
                "stdout_lines": stdout_line_count,
                "stderr_lines": stderr_line_count,
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "duration_ms": duration_ms,
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

        # Keep file bounded
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > self._CRON_LOG_MAX_LINES:
            path.write_text(
                "\n".join(lines[-self._CRON_LOG_MAX_LINES:]) + "\n",
                encoding="utf-8",
            )

    def read_cron_log(self, days: int = 1) -> str:
        """Read cron logs for the last *days* days."""
        log_dir = self.anima_dir / self._CRON_LOG_DIR
        if not log_dir.is_dir():
            return ""

        import json as _json
        parts: list[str] = []
        for i in range(days):
            target = date.today() - timedelta(days=i)
            path = log_dir / f"{target.isoformat()}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    e = _json.loads(line)
                    parts.append(
                        f"- {e['timestamp']}: [{e['task']}] {e['summary'][:200]} "
                        f"({e['duration_ms']}ms)"
                    )
                except (_json.JSONDecodeError, KeyError):
                    continue
        return "\n".join(parts)

    # ── Shared user memory ────────────────────────────────

    @staticmethod
    def _shared_users_dir() -> Path:
        return get_shared_dir() / "users"

    def list_shared_users(self) -> list[str]:
        """List user subdirectories under shared/users/."""
        d = self._shared_users_dir()
        if not d.is_dir():
            return []
        return [p.name for p in sorted(d.iterdir()) if p.is_dir()]

    # ── Write ─────────────────────────────────────────────

    def append_episode(self, entry: str) -> None:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        if not path.exists():
            path.write_text(
                f"# {date.today().isoformat()} 行動ログ\n\n", encoding="utf-8"
            )
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{entry}\n")
        logger.debug("Episode appended, length=%d", len(entry))

        # Index the updated episode file (incremental)
        if self._get_indexer():
            try:
                self._get_indexer().index_file(path, "episodes")
            except Exception as e:
                logger.warning("Failed to index episode file: %s", e)

    def update_state(self, content: str) -> None:
        (self.state_dir / "current_task.md").write_text(content, encoding="utf-8")

    def update_pending(self, content: str) -> None:
        (self.state_dir / "pending.md").write_text(content, encoding="utf-8")

    def append_resolution(self, issue: str, resolver: str) -> None:
        """Append resolution info to shared/resolutions.jsonl."""
        import json as _json

        shared_dir = get_shared_dir()
        path = shared_dir / "resolutions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "issue": issue,
            "resolver": resolver,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

    def read_resolutions(self, days: int = 7) -> list[dict[str, str]]:
        """Read recent resolutions from shared/resolutions.jsonl."""
        import json as _json

        shared_dir = get_shared_dir()
        path = shared_dir / "resolutions.jsonl"
        if not path.exists():
            return []
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        entries: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = _json.loads(line)
                if entry.get("ts", "") >= cutoff:
                    entries.append(entry)
            except _json.JSONDecodeError:
                continue
        return entries

    # ── Knowledge frontmatter helpers ────────────────────────

    def write_knowledge_with_meta(self, path: Path, content: str, metadata: dict) -> None:
        """Write knowledge file with YAML frontmatter metadata.

        Args:
            path: Path to the knowledge file
            content: Markdown content body (without frontmatter)
            metadata: Dictionary of metadata to embed as YAML frontmatter
        """
        import yaml

        frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\n{frontmatter}---\n\n{content}", encoding="utf-8")
        logger.debug("Knowledge written with metadata path='%s' length=%d", path, len(content))

    def read_knowledge_content(self, path: Path) -> str:
        """Read knowledge file body, stripping YAML frontmatter if present.

        Backward-compatible: returns full text if no frontmatter exists.

        Args:
            path: Path to the knowledge file

        Returns:
            Content body without YAML frontmatter
        """
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text

    def read_knowledge_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a knowledge file.

        Applies legacy migration: renames ``superseded_at`` to
        ``valid_until`` when encountered.

        Args:
            path: Path to the knowledge file

        Returns:
            Metadata dictionary, or empty dict if no frontmatter
        """
        import yaml

        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    logger.warning("Failed to parse YAML frontmatter in %s", path)
                    return {}
                # Legacy migration: superseded_at → valid_until
                if "superseded_at" in meta and "valid_until" not in meta:
                    meta["valid_until"] = meta.pop("superseded_at")
                return meta
        return {}

    def write_knowledge(self, topic: str, content: str) -> None:
        safe = re.sub(r"[^\w\-_]", "_", topic)
        path = self.knowledge_dir / f"{safe}.md"
        path.write_text(content, encoding="utf-8")
        logger.debug("Knowledge written topic='%s' length=%d", topic, len(content))

        # Index the new/updated knowledge file
        if self._get_indexer():
            try:
                self._get_indexer().index_file(path, "knowledge")
            except Exception as e:
                logger.warning("Failed to index knowledge file: %s", e)

    # ── Read helpers for Mode B (assisted) ──────────────────

    def read_recent_episodes(self, days: int = 7) -> str:
        """Return concatenated episode logs for the last *days* days."""
        parts: list[str] = []
        today = date.today()
        for offset in range(days):
            d = today - timedelta(days=offset)
            path = self.episodes_dir / f"{d.isoformat()}.md"
            if path.exists():
                parts.append(path.read_text(encoding="utf-8"))
        return "\n\n".join(parts)

    def search_memory_text(
        self, query: str, scope: str = "all"
    ) -> list[tuple[str, str]]:
        """Search memory files by keyword and optional vector similarity.

        Returns ``(filename, matching_line)`` pairs.

        *scope* can be ``"knowledge"``, ``"episodes"``, ``"procedures"``,
        ``"common_knowledge"``, or ``"all"`` (default).

        When RAG dependencies are available the method performs **hybrid
        search**: keyword matches are returned first, followed by
        vector-similarity results that were not already found by keyword.
        """
        dirs: list[Path] = []
        if scope in ("knowledge", "all"):
            dirs.append(self.knowledge_dir)
        if scope in ("episodes", "all"):
            dirs.append(self.episodes_dir)
        if scope in ("procedures", "all"):
            dirs.append(self.procedures_dir)
        if scope in ("common_knowledge", "all"):
            if self.common_knowledge_dir.is_dir():
                dirs.append(self.common_knowledge_dir)

        # Keyword search
        results: list[tuple[str, str]] = []
        q = query.lower()
        for d in dirs:
            for f in d.glob("*.md"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if q in line.lower():
                        results.append((f.name, line.strip()))

        # Hybrid: append vector search results when RAG is available
        if self._indexer is not None and scope in ("knowledge", "common_knowledge", "all"):
            try:
                vector_hits = self._vector_search_memory(query, scope)
                seen_files = {r[0] for r in results}
                for fname, snippet in vector_hits:
                    if fname not in seen_files:
                        results.append((fname, snippet))
                        seen_files.add(fname)
            except Exception as e:
                logger.debug("Vector search augmentation failed: %s", e)

        return results

    def _vector_search_memory(
        self, query: str, scope: str,
    ) -> list[tuple[str, str]]:
        """Perform vector search to augment keyword results.

        Returns ``(filename, first_line_of_content)`` pairs.
        """
        from core.memory.rag.retriever import MemoryRetriever

        anima_name = self.anima_dir.name
        retriever = MemoryRetriever(
            self._indexer.vector_store,
            self._indexer,
            self.knowledge_dir,
        )

        include_shared = scope in ("common_knowledge", "all")
        rag_results = retriever.search(
            query=query,
            anima_name=anima_name,
            memory_type="knowledge",
            top_k=5,
            include_shared=include_shared,
        )

        # Record access (Hebbian LTP: strengthen frequently accessed memories)
        if rag_results:
            retriever.record_access(rag_results, anima_name)

        hits: list[tuple[str, str]] = []
        for r in rag_results:
            source = r.metadata.get("source_file", r.doc_id)
            first_line = r.content.split("\n", 1)[0].strip()
            hits.append((str(source), first_line))
        return hits

    def search_procedures(self, query: str) -> list[tuple[str, str]]:
        """Search procedures/ by keyword."""
        return self.search_memory_text(query, scope="procedures")

    # ── Search (Python-side; LLM uses Grep directly) ─────

    def search_knowledge(self, query: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        q = query.lower()
        for f in self.knowledge_dir.glob("*.md"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if q in line.lower():
                    results.append((f.name, line.strip()))
        logger.debug("search_knowledge query='%s' results=%d", query, len(results))
        return results

    # ── Procedure frontmatter helpers ─────────────────────

    def write_procedure_with_meta(
        self, path: Path, content: str, metadata: dict,
    ) -> None:
        """Write a procedure file with YAML frontmatter metadata.

        Args:
            path: Target file path (absolute or relative to procedures_dir).
            content: Markdown body (without frontmatter).
            metadata: Dict of frontmatter fields (description, tags, etc.).
        """
        import yaml

        target = path if path.is_absolute() else self.procedures_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)

        fm_str = yaml.dump(metadata, default_flow_style=False, allow_unicode=True).rstrip()
        full = f"---\n{fm_str}\n---\n\n{content}"
        target.write_text(full, encoding="utf-8")
        logger.debug("Procedure written with metadata: %s", target.name)

    def read_procedure_content(self, path: Path) -> str:
        """Read procedure file body, stripping YAML frontmatter.

        Args:
            path: Absolute path or path relative to procedures_dir.

        Returns:
            Body text after the frontmatter block, or empty string.
        """
        target = path if path.is_absolute() else self.procedures_dir / path
        if not target.exists():
            return ""
        text = target.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()

    def read_procedure_metadata(self, path: Path) -> dict:
        """Read YAML frontmatter metadata from a procedure file.

        Args:
            path: Absolute path or path relative to procedures_dir.

        Returns:
            Parsed frontmatter dict, or empty dict if absent/unparseable.
        """
        import yaml

        target = path if path.is_absolute() else self.procedures_dir / path
        if not target.exists():
            return {}
        text = target.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        try:
            fm = yaml.safe_load(parts[1])
            return fm if isinstance(fm, dict) else {}
        except Exception:
            return {}

    def list_procedure_metas(self) -> list[SkillMeta]:
        """Return SkillMeta for each procedure file (reuses _extract_skill_meta)."""
        return [
            self._extract_skill_meta(f, is_common=False)
            for f in sorted(self.procedures_dir.glob("*.md"))
        ]

    def migrate_legacy_procedures(self) -> int:
        """Add YAML frontmatter to procedure files that lack it.

        Idempotent: uses ``{procedures_dir}/.migrated`` as a marker.
        Backs up originals to ``archive/pre_migration_procedures/``.

        Returns:
            Number of files migrated.
        """
        import shutil
        from datetime import datetime as _dt

        marker = self.procedures_dir / ".migrated"
        if marker.exists():
            logger.debug("Procedures already migrated (marker exists)")
            return 0

        md_files = sorted(self.procedures_dir.glob("*.md"))
        if not md_files:
            marker.write_text(_dt.now().isoformat(), encoding="utf-8")
            return 0

        backup_dir = self.anima_dir / "archive" / "pre_migration_procedures"
        backup_dir.mkdir(parents=True, exist_ok=True)

        migrated = 0
        for f in md_files:
            text = f.read_text(encoding="utf-8")
            if text.startswith("---"):
                continue  # already has frontmatter

            # Backup original
            shutil.copy2(f, backup_dir / f.name)

            # Derive description from filename (replace _ and - with spaces)
            desc = f.stem.replace("_", " ").replace("-", " ")

            metadata = {
                "description": desc,
                "tags": [],
                "success_count": 0,
                "failure_count": 0,
                "last_used": None,
                "confidence": 0.5,
                "version": 1,
                "created_at": _dt.now().isoformat(),
            }
            self.write_procedure_with_meta(f, text, metadata)
            migrated += 1
            logger.info("Migrated procedure: %s", f.name)

        marker.write_text(_dt.now().isoformat(), encoding="utf-8")
        logger.info("Migrated %d legacy procedures", migrated)
        return migrated