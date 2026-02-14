from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from core.paths import get_common_skills_dir, get_company_dir, get_shared_dir
from core.schemas import ModelConfig

logger = logging.getLogger("animaworks.memory")


class MemoryManager:
    """File-system based library memory.

    The LLM searches memory autonomously via Grep/Read tools.
    This class handles the Python-side read/write operations.
    """

    def __init__(self, person_dir: Path, base_dir: Path | None = None) -> None:
        self.person_dir = person_dir
        self.company_dir = get_company_dir()
        self.common_skills_dir = get_common_skills_dir()
        self.episodes_dir = person_dir / "episodes"
        self.knowledge_dir = person_dir / "knowledge"
        self.procedures_dir = person_dir / "procedures"
        self.skills_dir = person_dir / "skills"
        self.state_dir = person_dir / "state"
        for d in (
            self.episodes_dir,
            self.knowledge_dir,
            self.procedures_dir,
            self.skills_dir,
            self.state_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Initialize RAG indexer (Phase 2) if available
        self._indexer = None
        self._init_indexer()

    # ── RAG indexer initialization ────────────────────────

    def _init_indexer(self) -> None:
        """Initialize RAG indexer if dependencies are available."""
        try:
            from core.memory.rag import MemoryIndexer
            from core.memory.rag.store import ChromaVectorStore

            vector_store = ChromaVectorStore()
            person_name = self.person_dir.name
            self._indexer = MemoryIndexer(vector_store, person_name, self.person_dir)
            logger.debug("RAG indexer initialized for person=%s", person_name)
        except ImportError:
            logger.debug("RAG dependencies not installed, indexing disabled")
        except Exception as e:
            logger.warning("Failed to initialize RAG indexer: %s", e)

    # ── Read ──────────────────────────────────────────────

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_company_vision(self) -> str:
        return self._read(self.company_dir / "vision.md")

    def read_identity(self) -> str:
        return self._read(self.person_dir / "identity.md")

    def read_injection(self) -> str:
        return self._read(self.person_dir / "injection.md")

    def read_permissions(self) -> str:
        return self._read(self.person_dir / "permissions.md")

    def read_current_state(self) -> str:
        return self._read(self.state_dir / "current_task.md") or "status: idle"

    def read_pending(self) -> str:
        return self._read(self.state_dir / "pending.md")

    def read_heartbeat_config(self) -> str:
        return self._read(self.person_dir / "heartbeat.md")

    def read_cron_config(self) -> str:
        return self._read(self.person_dir / "cron.md")

    def read_model_config(self) -> ModelConfig:
        """Load model config from unified config.json, with config.md fallback."""
        from core.config import (
            get_config_path,
            load_config,
            resolve_execution_mode,
            resolve_person_config,
        )

        config_path = get_config_path()
        if config_path.exists():
            config = load_config(config_path)
            person_name = self.person_dir.name
            resolved, credential = resolve_person_config(config, person_name)
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
            )

        # Legacy fallback: parse config.md
        return self._read_model_config_from_md()

    def _read_model_config_from_md(self) -> ModelConfig:
        """Legacy parser for config.md (fallback when config.json absent)."""
        raw = self._read(self.person_dir / "config.md")
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
        return self._read(self.person_dir / "bootstrap.md")

    def read_today_episodes(self) -> str:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        return self._read(path)

    def read_file(self, relpath: str) -> str:
        """Read an arbitrary file relative to person_dir."""
        return self._read(self.person_dir / relpath)

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
    def _extract_skill_summary(path: Path) -> str:
        """Extract the first line of the 概要 section from a skill file."""
        text = path.read_text(encoding="utf-8")
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
                    return stripped
        return ""

    def list_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (filename_stem, first_line_of_概要) for each personal skill."""
        return [
            (f.stem, self._extract_skill_summary(f))
            for f in sorted(self.skills_dir.glob("*.md"))
        ]

    def list_common_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (filename_stem, first_line_of_概要) for each common skill."""
        if not self.common_skills_dir.is_dir():
            return []
        return [
            (f.stem, self._extract_skill_summary(f))
            for f in sorted(self.common_skills_dir.glob("*.md"))
        ]

    # ── Cron log ──────────────────────────────────────────

    _CRON_LOG_DIR = "state/cron_logs"
    _CRON_LOG_MAX_LINES = 50

    def append_cron_log(
        self, task_name: str, *, summary: str, duration_ms: int,
    ) -> None:
        """Append a cron execution result to the daily log."""
        log_dir = self.person_dir / self._CRON_LOG_DIR
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

    def read_cron_log(self, days: int = 1) -> str:
        """Read cron logs for the last *days* days."""
        log_dir = self.person_dir / self._CRON_LOG_DIR
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
        if self._indexer:
            try:
                self._indexer.index_file(path, "episodes")
            except Exception as e:
                logger.warning("Failed to index episode file: %s", e)

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
        if self._indexer:
            try:
                self._indexer.index_file(path, "knowledge")
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
        """Search memory files by keyword. Returns (filename, matching_line) pairs.

        *scope* can be ``"knowledge"``, ``"episodes"``, ``"procedures"``, or
        ``"all"`` (default).
        """
        dirs: list[Path] = []
        if scope in ("knowledge", "all"):
            dirs.append(self.knowledge_dir)
        if scope in ("episodes", "all"):
            dirs.append(self.episodes_dir)
        if scope in ("procedures", "all"):
            dirs.append(self.procedures_dir)

        results: list[tuple[str, str]] = []
        q = query.lower()
        for d in dirs:
            for f in d.glob("*.md"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if q in line.lower():
                        results.append((f.name, line.strip()))
        return results

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