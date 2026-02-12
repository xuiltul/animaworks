from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

from core.schemas import ModelConfig


class MemoryManager:
    """File-system based library memory.

    The LLM searches memory autonomously via Grep/Read tools.
    This class handles the Python-side read/write operations.
    """

    def __init__(self, person_dir: Path, base_dir: Path | None = None) -> None:
        self.person_dir = person_dir
        self.base_dir = base_dir or person_dir.parent.parent
        self.company_dir = self.base_dir / "company"
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
        """Parse config.md and return ModelConfig with API key and model settings."""
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
        """Resolve the actual API key from the environment variable name."""
        cfg = config or self.read_model_config()
        return os.environ.get(cfg.api_key_env)

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

    def list_skill_summaries(self) -> list[tuple[str, str]]:
        """Return (filename_stem, first_line_of_概要) for each skill."""
        results: list[tuple[str, str]] = []
        for f in sorted(self.skills_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            summary = ""
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
                        summary = stripped
                        break
            results.append((f.stem, summary))
        return results

    # ── Write ─────────────────────────────────────────────

    def append_episode(self, entry: str) -> None:
        path = self.episodes_dir / f"{date.today().isoformat()}.md"
        if not path.exists():
            path.write_text(
                f"# {date.today().isoformat()} 行動ログ\n\n", encoding="utf-8"
            )
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{entry}\n")

    def update_state(self, content: str) -> None:
        (self.state_dir / "current_task.md").write_text(content, encoding="utf-8")

    def update_pending(self, content: str) -> None:
        (self.state_dir / "pending.md").write_text(content, encoding="utf-8")

    def write_knowledge(self, topic: str, content: str) -> None:
        safe = re.sub(r"[^\w\-_]", "_", topic)
        (self.knowledge_dir / f"{safe}.md").write_text(content, encoding="utf-8")

    # ── Search (Python-side; LLM uses Grep directly) ─────

    def search_knowledge(self, query: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        q = query.lower()
        for f in self.knowledge_dir.glob("*.md"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if q in line.lower():
                    results.append((f.name, line.strip()))
        return results
