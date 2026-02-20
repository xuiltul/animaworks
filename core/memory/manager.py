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
from datetime import date, timedelta
from pathlib import Path

from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_company_dir, get_shared_dir
from core.schemas import ModelConfig, SkillMeta

# ── Re-exports for backward compatibility ─────────────────
# These were originally defined in this module.  External code
# (builder.py, priming.py, tests) may import them from here.
from core.memory.skill_metadata import (  # noqa: F401
    match_skills_by_description,
    _normalize_text,
    _extract_bracket_keywords,
    _extract_comma_keywords,
    _match_tier1,
    _match_tier2,
    _match_tier3_vector,
    _TIER2_STOP_WORDS,
)

from core.memory.cron_logger import CronLogger
from core.memory.resolution_tracker import ResolutionTracker
from core.memory.config_reader import ConfigReader
from core.memory.skill_metadata import SkillMetadataService
from core.memory.rag_search import RAGMemorySearch
from core.memory.frontmatter import FrontmatterService

logger = logging.getLogger("animaworks.memory")


class MemoryManager:
    """File-system based library memory — Facade.

    Delegates to specialised sub-services while preserving the original
    public interface so that the 60+ call-sites remain unchanged.
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

        # Eagerly initialize delegates (also available lazily via properties
        # for code that bypasses __init__ via __new__).
        self._init_delegates()

    def _init_delegates(self) -> None:
        """Create delegate instances.  Safe to call multiple times.

        Handles the case where ``__init__`` was bypassed via ``__new__``
        by resolving missing directory attributes from ``core.paths``.
        """
        if hasattr(self, "_delegates_ready"):
            return
        self._delegates_ready = True
        # Ensure directory attributes exist (tests may bypass __init__)
        if not hasattr(self, "common_skills_dir"):
            self.common_skills_dir = get_common_skills_dir()
        if not hasattr(self, "common_knowledge_dir"):
            self.common_knowledge_dir = get_common_knowledge_dir()
        if not hasattr(self, "company_dir"):
            self.company_dir = get_company_dir()
        ad = self.anima_dir
        if not hasattr(self, "knowledge_dir"):
            self.knowledge_dir = ad / "knowledge"
        if not hasattr(self, "procedures_dir"):
            self.procedures_dir = ad / "procedures"
        if not hasattr(self, "skills_dir"):
            self.skills_dir = ad / "skills"
        self.__cron = CronLogger(ad)
        self.__resolution = ResolutionTracker()
        self.__config = ConfigReader(ad)
        self.__skill_meta = SkillMetadataService(self.skills_dir, self.common_skills_dir)
        self.__rag = RAGMemorySearch(ad, self.common_knowledge_dir, self.common_skills_dir)
        self.__frontmatter = FrontmatterService(ad, self.knowledge_dir, self.procedures_dir)
        # Apply overrides stored before delegates existed (from tests
        # that set mm._indexer = None before __init__ was called).
        if "_indexer_override" in self.__dict__:
            self.__rag._indexer = self.__dict__.pop("_indexer_override")
        if "_indexer_initialized_override" in self.__dict__:
            self.__rag._indexer_initialized = self.__dict__.pop("_indexer_initialized_override")

    # Lazy delegate accessors — allow tests that bypass __init__
    # (via MemoryManager.__new__) to still work.

    @property
    def _cron(self) -> CronLogger:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__cron

    @property
    def _resolution(self) -> ResolutionTracker:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__resolution

    @property
    def _config_reader(self) -> ConfigReader:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__config

    @property
    def _skill_meta(self) -> SkillMetadataService:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__skill_meta

    @property
    def _rag(self) -> RAGMemorySearch:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__rag

    @property
    def _frontmatter(self) -> FrontmatterService:
        if not hasattr(self, "_delegates_ready"):
            self._init_delegates()
        return self.__frontmatter

    @property
    def task_queue_path(self) -> Path:
        """Path to the persistent task queue JSONL file."""
        return self.state_dir / "task_queue.jsonl"

    # ── Read helpers ──────────────────────────────────────

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

    def read_bootstrap(self) -> str:
        return self._read(self.anima_dir / "bootstrap.md")

    def read_today_episodes(self) -> str:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        return self._read(path)

    def read_file(self, relpath: str) -> str:
        """Read an arbitrary file relative to anima_dir."""
        return self._read(self.anima_dir / relpath)

    # ── List helpers ──────────────────────────────────────

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

    # ── Write helpers ─────────────────────────────────────

    def append_episode(self, entry: str) -> None:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        if not path.exists():
            path.write_text(
                f"# {date.today().isoformat()} 行動ログ\n\n", encoding="utf-8"
            )
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{entry}\n")
            f.flush()
            os.fsync(f.fileno())
        logger.debug("Episode appended, length=%d", len(entry))

        # Index the updated episode file (incremental)
        self._rag.index_file(path, "episodes")

    def update_state(self, content: str) -> None:
        (self.state_dir / "current_task.md").write_text(content, encoding="utf-8")

    def update_pending(self, content: str) -> None:
        (self.state_dir / "pending.md").write_text(content, encoding="utf-8")

    def write_knowledge(self, topic: str, content: str) -> None:
        safe = re.sub(r"[^\w\-_]", "_", topic)
        path = self.knowledge_dir / f"{safe}.md"
        path.write_text(content, encoding="utf-8")
        logger.debug("Knowledge written topic='%s' length=%d", topic, len(content))

        # Index the new/updated knowledge file
        self._rag.index_file(path, "knowledge")

    # ── Read helpers for Mode B (assisted) ────────────────

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

    # ── Backward-compatible RAG proxies ─────────────────
    # Tests and internal code may access these private attributes
    # which now live in RAGMemorySearch.

    @property
    def _indexer(self):
        return self._rag._indexer

    @_indexer.setter
    def _indexer(self, value):
        # When tests bypass __init__ and set _indexer before delegates
        # exist, store the value directly; it will be picked up later.
        if hasattr(self, "_delegates_ready"):
            self._rag._indexer = value
        else:
            self.__dict__["_indexer_override"] = value

    @property
    def _indexer_initialized(self):
        return self._rag._indexer_initialized

    @_indexer_initialized.setter
    def _indexer_initialized(self, value):
        if hasattr(self, "_delegates_ready"):
            self._rag._indexer_initialized = value
        else:
            self.__dict__["_indexer_initialized_override"] = value

    def _init_indexer(self) -> None:
        self._rag._init_indexer()

    def _get_indexer(self):
        return self._rag._get_indexer()

    def _ensure_shared_knowledge_indexed(self, vector_store) -> None:
        self._rag._ensure_shared_knowledge_indexed(vector_store)

    def _ensure_shared_skills_indexed(self, vector_store) -> None:
        self._rag._ensure_shared_skills_indexed(vector_store)

    def _vector_search_memory(
        self, query: str, scope: str,
    ) -> list[tuple[str, str]]:
        return self._rag._vector_search_memory(query, scope, self.knowledge_dir)

    # ── Facade: ConfigReader ──────────────────────────────

    def read_model_config(self) -> ModelConfig:
        """Facade: ConfigReader.read_model_config."""
        return self._config_reader.read_model_config()

    def _read_model_config_from_md(self) -> ModelConfig:
        """Facade: ConfigReader._read_model_config_from_md."""
        return self._config_reader._read_model_config_from_md()

    def resolve_api_key(self, config: ModelConfig | None = None) -> str | None:
        """Facade: ConfigReader.resolve_api_key."""
        return self._config_reader.resolve_api_key(config)

    # ── Facade: SkillMetadataService ──────────────────────

    @staticmethod
    def _extract_skill_meta(path: Path, *, is_common: bool = False) -> SkillMeta:
        """Facade: SkillMetadataService.extract_skill_meta."""
        return SkillMetadataService.extract_skill_meta(path, is_common=is_common)

    def list_skill_metas(self) -> list[SkillMeta]:
        """Facade: SkillMetadataService.list_skill_metas."""
        return self._skill_meta.list_skill_metas()

    def list_common_skill_metas(self) -> list[SkillMeta]:
        """Facade: SkillMetadataService.list_common_skill_metas."""
        return self._skill_meta.list_common_skill_metas()

    def list_skill_summaries(self) -> list[tuple[str, str]]:
        """Facade: SkillMetadataService.list_skill_summaries."""
        return self._skill_meta.list_skill_summaries()

    def list_common_skill_summaries(self) -> list[tuple[str, str]]:
        """Facade: SkillMetadataService.list_common_skill_summaries."""
        return self._skill_meta.list_common_skill_summaries()

    # ── Facade: CronLogger ────────────────────────────────

    def append_cron_log(
        self, task_name: str, *, summary: str, duration_ms: int,
    ) -> None:
        """Facade: CronLogger.append_cron_log."""
        self._cron.append_cron_log(task_name, summary=summary, duration_ms=duration_ms)

    def append_cron_command_log(
        self,
        task_name: str,
        *,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_ms: int,
    ) -> None:
        """Facade: CronLogger.append_cron_command_log."""
        self._cron.append_cron_command_log(
            task_name, exit_code=exit_code, stdout=stdout,
            stderr=stderr, duration_ms=duration_ms,
        )

    def read_cron_log(self, days: int = 1) -> str:
        """Facade: CronLogger.read_cron_log."""
        return self._cron.read_cron_log(days)

    # ── Facade: ResolutionTracker ─────────────────────────

    def append_resolution(self, issue: str, resolver: str) -> None:
        """Facade: ResolutionTracker.append_resolution."""
        self._resolution.append_resolution(issue, resolver)

    def read_resolutions(self, days: int = 7) -> list[dict[str, str]]:
        """Facade: ResolutionTracker.read_resolutions."""
        return self._resolution.read_resolutions(days)

    # ── Facade: RAGMemorySearch ───────────────────────────

    def search_memory_text(
        self, query: str, scope: str = "all",
    ) -> list[tuple[str, str]]:
        """Facade: RAGMemorySearch.search_memory_text."""
        return self._rag.search_memory_text(
            query, scope,
            knowledge_dir=self.knowledge_dir,
            episodes_dir=self.episodes_dir,
            procedures_dir=self.procedures_dir,
            common_knowledge_dir=self.common_knowledge_dir,
        )

    def search_procedures(self, query: str) -> list[tuple[str, str]]:
        """Facade: search via search_memory_text with procedures scope."""
        return self.search_memory_text(query, scope="procedures")

    def search_knowledge(self, query: str) -> list[tuple[str, str]]:
        """Facade: RAGMemorySearch.search_knowledge."""
        return self._rag.search_knowledge(query, self.knowledge_dir)

    # ── Facade: FrontmatterService ────────────────────────

    def write_knowledge_with_meta(self, path: Path, content: str, metadata: dict) -> None:
        """Facade: FrontmatterService.write_knowledge_with_meta."""
        self._frontmatter.write_knowledge_with_meta(path, content, metadata)

    def read_knowledge_content(self, path: Path) -> str:
        """Facade: FrontmatterService.read_knowledge_content."""
        return self._frontmatter.read_knowledge_content(path)

    def read_knowledge_metadata(self, path: Path) -> dict:
        """Facade: FrontmatterService.read_knowledge_metadata."""
        return self._frontmatter.read_knowledge_metadata(path)

    def update_knowledge_metadata(self, path: Path, updates: dict) -> None:
        """Facade: FrontmatterService.update_knowledge_metadata."""
        self._frontmatter.update_knowledge_metadata(path, updates)

    def write_procedure_with_meta(
        self, path: Path, content: str, metadata: dict,
    ) -> None:
        """Facade: FrontmatterService.write_procedure_with_meta."""
        self._frontmatter.write_procedure_with_meta(path, content, metadata)

    def read_procedure_content(self, path: Path) -> str:
        """Facade: FrontmatterService.read_procedure_content."""
        return self._frontmatter.read_procedure_content(path)

    def read_procedure_metadata(self, path: Path) -> dict:
        """Facade: FrontmatterService.read_procedure_metadata."""
        return self._frontmatter.read_procedure_metadata(path)

    def list_procedure_metas(self) -> list[SkillMeta]:
        """Facade: FrontmatterService.list_procedure_metas."""
        return self._frontmatter.list_procedure_metas(self._extract_skill_meta)

    def migrate_legacy_procedures(self) -> int:
        """Facade: FrontmatterService.migrate_legacy_procedures."""
        return self._frontmatter.migrate_legacy_procedures()
