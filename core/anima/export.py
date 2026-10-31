# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Export an Anima workspace in OpenClaw or Hermes Agent layout."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.anima.workspace_files import (
    AGENTS_FILE,
    HEARTBEAT_FILE,
    IDENTITY_FILE,
    SOUL_FILE,
)
from core.skills.migration._common import redact_credentials
from core.time_utils import now_iso

_EXPORT_MEMORY_LIMIT = 20_000
_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".csv"}
_NOT_EXPORTED = ("permissions.json", "status.json", "cron.md")
_EPISODE_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class ExportResult:
    """Summary of an Anima workspace export."""

    anima: str
    format: str
    out_dir: Path
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_text(source: Path, destination: Path, result: ExportResult) -> bool:
    try:
        text = _read_text(source)
    except (OSError, UnicodeDecodeError):
        result.warnings.append(f"Could not read {source}; skipped")
        return False
    _write_text(destination, redact_credentials(text))
    result.files.append(str(destination))
    return True


def _copy_skill_tree(skills_dir: Path, destination: Path, result: ExportResult) -> None:
    if not skills_dir.is_dir():
        return
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
            continue
        for source in sorted(skill_dir.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(skill_dir)
            target = destination / skill_dir.name / rel
            if source.suffix.lower() in _TEXT_SUFFIXES:
                _copy_text(source, target, result)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                result.files.append(str(target))


def _find_user_profile(data_dir: Path) -> Path | None:
    users_dir = data_dir / "shared" / "users"
    if not users_dir.is_dir():
        return None
    candidates = [
        user_dir / "index.md"
        for user_dir in users_dir.iterdir()
        if user_dir.is_dir() and (user_dir / "index.md").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _summarize_memory(anima_dir: Path) -> str:
    knowledge_dir = anima_dir / "knowledge"
    blocks: list[str] = ["# MEMORY", ""]
    if knowledge_dir.is_dir():
        for path in sorted(knowledge_dir.rglob("*.md")):
            if not path.is_file():
                continue
            try:
                text = _read_text(path)
            except (OSError, UnicodeDecodeError):
                continue
            lines = [line.strip() for line in text.splitlines()]
            title = next(
                (line.lstrip("#").strip() for line in lines if line.startswith("#") and line.lstrip("#").strip()),
                path.stem,
            )
            summary: list[str] = []
            for line in lines:
                if not line or line.startswith("#") or line.startswith("---"):
                    continue
                summary.append(line)
                if len(summary) >= 2:
                    break
            rel = str(path.relative_to(anima_dir)).replace("\\", "/")
            blocks.extend([f"## {title}", f"<!-- source: {rel} -->"])
            blocks.extend(summary)
            blocks.append("")
    if len(blocks) == 2:
        blocks.append("_No knowledge files._")
    text = redact_credentials("\n".join(blocks).rstrip() + "\n")
    return text[:_EXPORT_MEMORY_LIMIT]


def _copy_episodes(anima_dir: Path, destination: Path, result: ExportResult) -> None:
    episodes_dir = anima_dir / "episodes"
    if not episodes_dir.is_dir():
        return
    for source in sorted(episodes_dir.glob("*.md")):
        date_match = _EPISODE_DATE_RE.search(source.stem)
        if date_match is None:
            continue
        target = destination / f"{date_match.group(0)}.md"
        try:
            text = redact_credentials(_read_text(source))
        except (OSError, UnicodeDecodeError):
            result.warnings.append(f"Could not read episode {source}; skipped")
            continue
        if target.exists():
            previous = target.read_text(encoding="utf-8").rstrip()
            text = f"{previous}\n\n---\n\n{text.lstrip()}"
        _write_text(target, text)
        if str(target) not in result.files:
            result.files.append(str(target))


def _write_manifest(result: ExportResult, out_dir: Path, data_dir: Path) -> None:
    exported = sorted({Path(path).relative_to(out_dir).as_posix() for path in result.files})
    lines = [
        "# AnimaWorks export",
        "",
        f"- Anima: {result.anima}",
        f"- Format: {result.format}",
        f"- Exported at: {now_iso()}",
        f"- Source data directory: {data_dir}",
        "",
        "## Exported",
        "",
    ]
    lines.extend(f"- `{name}`" for name in exported)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    lines.extend(
        [
            "",
            "## Not exported",
            "",
            *(f"- `{name}` (runtime-only)" for name in _NOT_EXPORTED),
            "",
            "Credentials and secrets are redacted in text files.",
            "",
        ]
    )
    _write_text(out_dir / "ANIMAWORKS_EXPORT.md", "\n".join(lines))


def export_anima(data_dir: Path, anima_name: str, export_format: str, out_dir: Path) -> ExportResult:
    """Export ``anima_name`` into ``out_dir``.

    ``openclaw`` keeps the standard workspace files separate and adds
    USER.md, MEMORY.md, dated episodes and skills. ``hermes`` concatenates
    IDENTITY.md and SOUL.md into SOUL.md and uses the memories/ directory.
    """
    if export_format not in {"openclaw", "hermes"}:
        raise ValueError(f"Unsupported export format: {export_format}")
    anima_dir = data_dir / "animas" / anima_name
    if not anima_dir.is_dir():
        raise FileNotFoundError(f"Anima not found: {anima_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(anima=anima_name, format=export_format, out_dir=out_dir)

    if export_format == "openclaw":
        for name in (AGENTS_FILE, SOUL_FILE, IDENTITY_FILE, HEARTBEAT_FILE):
            source = anima_dir / name
            if source.is_file():
                _copy_text(source, out_dir / name, result)
        profile = _find_user_profile(data_dir)
        if profile is not None:
            _copy_text(profile, out_dir / "USER.md", result)
        _write_text(out_dir / "MEMORY.md", _summarize_memory(anima_dir))
        result.files.append(str(out_dir / "MEMORY.md"))
        _copy_episodes(anima_dir, out_dir / "memory", result)
    else:
        soul_parts: list[str] = []
        for name in (IDENTITY_FILE, SOUL_FILE):
            source = anima_dir / name
            if source.is_file():
                try:
                    soul_parts.append(_read_text(source).rstrip())
                except (OSError, UnicodeDecodeError):
                    result.warnings.append(f"Could not read {source}; skipped")
        if soul_parts:
            _write_text(out_dir / "SOUL.md", redact_credentials("\n\n".join(soul_parts) + "\n"))
            result.files.append(str(out_dir / "SOUL.md"))
        agents = anima_dir / AGENTS_FILE
        if agents.is_file():
            _copy_text(agents, out_dir / AGENTS_FILE, result)
        profile = _find_user_profile(data_dir)
        if profile is not None:
            _copy_text(profile, out_dir / "memories" / "USER.md", result)
        _write_text(out_dir / "memories" / "MEMORY.md", _summarize_memory(anima_dir))
        result.files.append(str(out_dir / "memories" / "MEMORY.md"))

    _copy_skill_tree(anima_dir / "skills", out_dir / "skills", result)
    _write_manifest(result, out_dir, data_dir)
    return result


__all__ = ["ExportResult", "export_anima"]
