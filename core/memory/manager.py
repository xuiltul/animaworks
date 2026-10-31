from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import logging
import os
from datetime import timedelta
from pathlib import Path

from core.anima.workspace_files import AGENTS_FILE, BOOTSTRAP_FILE, HEARTBEAT_FILE, IDENTITY_FILE, SOUL_FILE
from core.config.file_access_policy import find_denied_root, load_denied_roots
from core.i18n import t
from core.memory.config_reader import ConfigReader
from core.memory.frontmatter import FrontmatterService
from core.memory.io import atomic_write_text
from core.memory.maintenance.cron_logger import CronLogger
from core.memory.maintenance.resolution_tracker import ResolutionTracker
from core.memory.retrieval.rag_search import RAGMemorySearch
from core.memory.skill_metadata import SkillMetadataService
from core.memory.state_lock import StateFileLock
from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_company_dir, get_shared_dir
from core.platform.locks import locked_path
from core.schemas import ModelConfig, SkillMeta
from core.time_utils import now_local, today_local

logger = logging.getLogger("animaworks.memory")


class MemoryManager:
    """File-system based library memory — Facade.

    Delegates memory operations to specialised sub-services.
    """

    def __init__(self, anima_dir: Path, base_dir: Path | None = None, *, read_only: bool = False) -> None:
        self.anima_dir = anima_dir
        self.read_only = read_only
        self.company_dir = get_company_dir()
        self.common_skills_dir = get_common_skills_dir()
        self.common_knowledge_dir = get_common_knowledge_dir()
        self.episodes_dir = anima_dir / "episodes"
        self.knowledge_dir = anima_dir / "knowledge"
        self.procedures_dir = anima_dir / "procedures"
        self.facts_dir = anima_dir / "facts"
        self.skills_dir = anima_dir / "skills"
        self.state_dir = anima_dir / "state"
        if not self.read_only:
            for d in (
                self.episodes_dir,
                self.knowledge_dir,
                self.procedures_dir,
                self.facts_dir,
                self.skills_dir,
                self.state_dir,
            ):
                d.mkdir(parents=True, exist_ok=True)
        self.state_lock = StateFileLock(anima_dir)

        if not self.read_only:
            self._migrate_current_task_to_state()
            self._migrate_pending_to_state()

        # Eagerly initialize delegates (also available lazily via properties
        # for code that bypasses __init__ via __new__).
        self._init_delegates()

    def _migrate_current_task_to_state(self) -> None:
        """Migrate current_task.md to current_state.md (one-time rename).

        - If state/current_task.md exists and state/current_state.md does NOT:
          rename (os.rename).
        - If both exist: log warning, use current_state.md.
        - If anima_dir/current_task.md (legacy root-level) exists: log warning.
        """
        old_task = self.state_dir / "current_task.md"
        new_state = self.state_dir / "current_state.md"
        legacy_root = self.anima_dir / "current_task.md"

        legacy_allowed = self._resolve_host_read_path(legacy_root) is not None
        old_allowed = self._resolve_host_read_path(old_task) is not None
        new_allowed = self._resolve_host_read_path(new_state) is not None

        if legacy_allowed and legacy_root.exists():
            logger.warning("Legacy root-level current_task.md found at %s", legacy_root)

        if old_allowed and new_allowed and old_task.exists() and not new_state.exists():
            try:
                os.rename(old_task, new_state)
                logger.info("Migrated %s -> %s", old_task.name, new_state.name)
            except OSError:
                logger.warning("Failed to migrate %s to %s", old_task, new_state, exc_info=True)
        elif old_allowed and new_allowed and old_task.exists() and new_state.exists():
            try:
                old_task.unlink()
                logger.info("Removed stale current_task.md from %s", self.state_dir)
            except OSError:
                logger.warning(
                    "Both current_task.md and current_state.md exist in %s; using current_state.md",
                    self.state_dir,
                )

    def _migrate_pending_to_state(self) -> None:
        """One-time migration: merge pending.md into current_state.md."""
        pending = self.state_dir / "pending.md"
        if self._resolve_host_read_path(pending) is None or not pending.exists():
            return
        content = self._read(pending).strip()
        if content:
            current = self.read_current_state()
            merged = current.rstrip() + "\n\n## Migrated from pending.md\n\n" + content
            self.update_state(merged)
            logger.info(
                "Migrated pending.md (%d chars) into current_state.md",
                len(content),
            )
        pending.unlink()
        logger.info("Removed deprecated pending.md")

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
        if not hasattr(self, "facts_dir"):
            self.facts_dir = ad / "facts"
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

    # ── Read helpers ──────────────────────────────────────

    def _resolve_host_read_path(self, path: Path) -> Path | None:
        """Resolve *path* and reject configured deny roots before host reads.

        Memory prompt assembly runs in the trusted AnimaWorks host process,
        outside the model's OS sandbox.  Every host-side read therefore needs
        the same canonical deny check to avoid becoming a confused deputy for
        symlinks placed inside otherwise allowed memory directories.
        """
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            logger.warning("Failed to resolve host memory path %s", path, exc_info=True)
            return None

        denied_root = find_denied_root(
            resolved, load_denied_roots(self.anima_dir, read_only=getattr(self, "read_only", False))
        )
        if denied_root is not None:
            logger.warning("Host memory read denied for %s by %s", resolved, denied_root)
            return None
        return resolved

    def _read(self, path: Path) -> str:
        resolved = self._resolve_host_read_path(path)
        if resolved is None or not resolved.exists():
            return ""
        try:
            return resolved.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Failed to read %s", resolved, exc_info=True)
            return ""

    def _company_vision_path(self) -> tuple[Path, str]:
        from core.org.company import get_company

        company = get_company(self.anima_dir.name, animas_dir=self.anima_dir.parent)
        if company is not None:
            companies_dir = (self.company_dir.parent / "companies").resolve()
            company_dir = (companies_dir / company).resolve()
            if company_dir.parent == companies_dir:
                vision_path = company_dir / "vision.md"
                if vision_path.is_file():
                    return vision_path, str(vision_path)
        legacy_path = self.company_dir / "vision.md"
        return legacy_path, str(legacy_path.resolve())

    def read_company_vision(self) -> str:
        return self._read(self._company_vision_path()[0])

    def read_company_vision_summary(self) -> tuple[str, str] | None:
        """Read a concise company vision and its full-text memory pointer, if present."""
        vision_path, reference_path = self._company_vision_path()
        if not vision_path.is_file():
            return None
        summary = self._read(vision_path.with_name("vision_summary.md")).strip()
        if not summary:
            return None
        return summary, reference_path

    def read_identity(self) -> str:
        """Read IDENTITY.md, stripping YAML frontmatter if present."""
        from core.memory.frontmatter import strip_frontmatter

        raw = self._read(self.anima_dir / IDENTITY_FILE)
        if raw and raw.lstrip().startswith("---"):
            return strip_frontmatter(raw)
        return raw

    def read_soul(self) -> str:
        """Read SOUL.md, stripping YAML frontmatter if present."""
        from core.memory.frontmatter import strip_frontmatter

        raw = self._read(self.anima_dir / SOUL_FILE)
        if raw and raw.lstrip().startswith("---"):
            return strip_frontmatter(raw)
        return raw

    def read_injection(self) -> str:
        """Read AGENTS.md, stripping YAML frontmatter if present."""
        from core.memory.frontmatter import strip_frontmatter

        raw = self._read(self.anima_dir / AGENTS_FILE)
        if raw and raw.lstrip().startswith("---"):
            return strip_frontmatter(raw)
        return raw

    def read_specialty_prompt(self) -> str:
        """Read the role-specific specialty prompt."""
        return self._read(self.anima_dir / "specialty_prompt.md")

    def read_permissions(self) -> str:
        """Read permissions as formatted text for prompt injection.

        Loads from permissions.json (structured) and formats as text.
        Falls back to raw permissions.md if JSON not available.
        """
        from core.config.models import _format_permissions_for_prompt, load_permissions

        config = load_permissions(self.anima_dir, read_only=getattr(self, "read_only", False))
        return _format_permissions_for_prompt(config, self.anima_dir.name)

    def read_current_state(self) -> str:
        return self._read(self.state_dir / "current_state.md") or "status: idle"

    def read_heartbeat_config(self) -> str:
        return self._read(self.anima_dir / HEARTBEAT_FILE)

    def load_recent_heartbeat_summary(self, limit: int = 5) -> str:
        """Load recent heartbeat summaries for dialogue context injection."""
        import json as _json

        history_dir = self._resolve_host_read_path(self.anima_dir / "shortterm" / "heartbeat_history")
        if history_dir is None or not history_dir.exists():
            return ""

        lines: list[str] = []
        for f in sorted(history_dir.glob("*.jsonl"), reverse=True)[:3]:
            content = self._read(f)
            if not content:
                continue
            file_lines = content.strip().splitlines()
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
        return self._read(self.anima_dir / BOOTSTRAP_FILE)

    # ── List helpers ──────────────────────────────────────

    def list_knowledge_files(self) -> list[str]:
        return [f.stem for f in self._glob_host_readable(self.knowledge_dir, "*.md")]

    def list_episode_files(self) -> list[str]:
        return [f.stem for f in self._glob_host_readable(self.episodes_dir, "*.md", reverse=True)]

    def list_procedure_files(self) -> list[str]:
        return [f.stem for f in self._glob_host_readable(self.procedures_dir, "*.md")]

    def _glob_host_readable(self, directory: Path, pattern: str, *, reverse: bool = False) -> list[Path]:
        """Return canonical glob matches after pruning denied paths."""
        resolved_dir = self._resolve_host_read_path(directory)
        if resolved_dir is None or not resolved_dir.is_dir():
            return []
        readable: list[Path] = []
        for candidate in resolved_dir.glob(pattern):
            resolved = self._resolve_host_read_path(candidate)
            if resolved is not None:
                readable.append(resolved)
        return sorted(readable, reverse=reverse)

    # ── Shared user memory ────────────────────────────────

    # ── Write helpers ─────────────────────────────────────

    def append_episode(
        self,
        entry: str,
        *,
        origin: str = "",
        _defer_index: bool = False,
    ) -> Path | None:
        path = self.episodes_dir / f"{today_local().isoformat()}.md"
        try:
            lock_path = path.with_name(f"{path.name}.lock")
            with locked_path(lock_path, exclusive=True, thread_lock=True):
                if not path.exists():
                    path.write_text(
                        t("manager.action_log_header", date=today_local().isoformat()),
                        encoding="utf-8",
                    )
                payload = f"\n{entry}\n".encode()
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    offset = 0
                    while offset < len(payload):
                        written = os.write(fd, payload[offset:])
                        if written <= 0:
                            raise OSError("episode append made no progress")
                        offset += written
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except OSError:
            logger.warning("Failed to append episode to %s", path, exc_info=True)
            return None
        logger.debug("Episode appended, length=%d", len(entry))

        # Index outside state-lock critical sections when requested.
        if not _defer_index:
            self._index_episode_file(path, origin=origin)
        return path

    def _index_episode_file(self, path: Path, *, origin: str = "") -> None:
        """Index an episode after its append is safely persisted."""
        self._rag.index_file(path, "episodes", origin=origin)

    def update_state(self, content: str) -> None:
        atomic_write_text(self.state_dir / "current_state.md", content)

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

    def _vector_search_primary(
        self,
        query: str,
        scope: str,
        offset: int = 0,
    ) -> list[dict]:
        return self._rag._vector_search_primary(query, scope, offset, self.knowledge_dir)

    # ── Facade: ConfigReader ──────────────────────────────

    def read_model_config(self) -> ModelConfig:
        """Facade: ConfigReader.read_model_config."""
        return self._config_reader.read_model_config()

    # ── Facade: SkillMetadataService ──────────────────────

    def list_skill_metas(self) -> list[SkillMeta]:
        """Facade: SkillMetadataService.list_skill_metas."""
        return self._skill_meta.list_skill_metas()

    def list_common_skill_metas(self) -> list[SkillMeta]:
        """Facade: SkillMetadataService.list_common_skill_metas."""
        return self._skill_meta.list_common_skill_metas()

    # ── Facade: CronLogger ────────────────────────────────

    def append_cron_event(
        self,
        task_name: str,
        event: str,
        *,
        reason: str = "",
        schedule: str = "",
    ) -> None:
        """Facade: CronLogger.append_cron_event."""
        self._cron.append_cron_event(task_name, event, reason=reason, schedule=schedule)

    def append_cron_log(
        self,
        task_name: str,
        *,
        summary: str,
        duration_ms: int,
        skill_rejections: list[dict[str, str]] | None = None,
    ) -> None:
        """Facade: CronLogger.append_cron_log."""
        self._cron.append_cron_log(
            task_name,
            summary=summary,
            duration_ms=duration_ms,
            skill_rejections=skill_rejections,
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
        """Facade: CronLogger.append_cron_command_log."""
        self._cron.append_cron_command_log(
            task_name,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
        )

    # ── Facade: ResolutionTracker ─────────────────────────

    def append_resolution(self, issue: str, resolver: str) -> None:
        """Facade: ResolutionTracker.append_resolution."""
        self._resolution.append_resolution(issue, resolver)

    def read_resolutions(self, days: int = 7) -> list[dict[str, str]]:
        """Read prompt-visible resolutions through the host deny boundary."""
        import json as _json
        from collections import deque

        content = self._read(get_shared_dir() / "resolutions.jsonl")
        if not content:
            return []

        cutoff = (now_local() - timedelta(days=days)).isoformat()
        lines = deque(content.splitlines(), maxlen=2000)
        entries: list[dict[str, str]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if entry.get("ts", "") < cutoff:
                break
            entries.append(entry)
        entries.reverse()
        return entries

    def filter_resolutions_by_company(self, resolutions: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep resolutions whose resolver belongs to this anima's company."""
        from core.org.company import get_company

        animas_dir = self.anima_dir.parent
        company = get_company(self.anima_dir.name, animas_dir=animas_dir)
        cache: dict[str, str | None] = {}

        def _company_of(resolver: str) -> str | None:
            if resolver not in cache:
                cache[resolver] = get_company(resolver, animas_dir=animas_dir)
            return cache[resolver]

        return [r for r in resolutions if _company_of(r.get("resolver", "")) == company]

    # ── Facade: RAGMemorySearch ───────────────────────────

    def search_memory_text(
        self,
        query: str,
        scope: str = "all",
        *,
        offset: int = 0,
        context_window: int = 128_000,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> list[dict]:
        """Facade: RAGMemorySearch.search_memory_text."""
        return self._rag.search_memory_text(
            query,
            scope,
            offset=offset,
            context_window=context_window,
            knowledge_dir=self.knowledge_dir,
            episodes_dir=self.episodes_dir,
            procedures_dir=self.procedures_dir,
            common_knowledge_dir=self.common_knowledge_dir,
            time_start=time_start,
            time_end=time_end,
        )

    @property
    def last_search_meta(self) -> dict[str, object]:
        """Metadata from the most recent ``search_memory_text`` call."""
        return self._rag.last_search_meta

    # ── Facade: FrontmatterService ────────────────────────

    def write_knowledge_with_meta(self, path: Path, content: str, metadata: dict) -> None:
        """Facade: FrontmatterService.write_knowledge_with_meta."""
        self._frontmatter.write_knowledge_with_meta(path, content, metadata)

    def index_knowledge_file(self, path: Path, *, origin: str = "") -> None:
        """Refresh RAG and sparse indexes after a knowledge write."""
        self._rag.index_file(path, "knowledge", origin=origin)

    def read_peer_profile(self, name: str, *, max_chars: int | None = None) -> str:
        """Read a peer profile through the host deny-root policy."""
        from core.memory.peer_profiles import peer_profile_path

        path = peer_profile_path(self.anima_dir, name)
        if path is None:
            return ""
        content = self._read(path)
        return content[:max_chars] if max_chars is not None else content

    def write_peer_profile(self, name: str, content: str, *, max_chars: int = 1500) -> Path:
        """Atomically replace a safe peer profile without adding it to RAG."""
        from core.memory.peer_profiles import peer_profile_path

        path = peer_profile_path(self.anima_dir, name)
        if path is None:
            raise ValueError(f"Invalid peer profile name: {name!r}")
        resolved = self._resolve_host_read_path(path)
        if resolved is None or not resolved.is_relative_to((self.anima_dir / "peers").resolve()):
            raise ValueError(f"Peer profile path is not writable: {name!r}")
        content = content[: max(0, max_chars)].strip()
        if not content:
            raise ValueError("Peer profile content is empty")
        atomic_write_text(resolved, content.rstrip() + "\n")
        return resolved

    def read_knowledge_content(self, path: Path) -> str:
        """Facade: FrontmatterService.read_knowledge_content."""
        from core.memory.frontmatter import split_frontmatter

        text = self._read(path)
        _, body = split_frontmatter(text)
        return body.strip()

    def read_knowledge_metadata(self, path: Path) -> dict:
        """Facade: FrontmatterService.read_knowledge_metadata."""
        from core.memory.frontmatter import parse_frontmatter

        text = self._read(path)
        meta, _ = parse_frontmatter(text)
        if "superseded_at" in meta and "valid_until" not in meta:
            meta["valid_until"] = meta.pop("superseded_at")
        return meta

    def write_procedure_with_meta(
        self,
        path: Path,
        content: str,
        metadata: dict,
    ) -> None:
        """Facade: FrontmatterService.write_procedure_with_meta."""
        self._frontmatter.write_procedure_with_meta(path, content, metadata)

    def read_procedure_content(self, path: Path) -> str:
        """Facade: FrontmatterService.read_procedure_content."""
        from core.memory.frontmatter import split_frontmatter

        target = path if path.is_absolute() else self.procedures_dir / path
        text = self._read(target)
        _, body = split_frontmatter(text)
        return body.strip()

    def read_procedure_metadata(self, path: Path) -> dict:
        """Facade: FrontmatterService.read_procedure_metadata."""
        from core.memory.frontmatter import parse_frontmatter

        target = path if path.is_absolute() else self.procedures_dir / path
        text = self._read(target)
        meta, _ = parse_frontmatter(text)
        return meta

    def ensure_procedure_frontmatter(self) -> int:
        """Facade: FrontmatterService.ensure_procedure_frontmatter."""
        return self._frontmatter.ensure_procedure_frontmatter()
