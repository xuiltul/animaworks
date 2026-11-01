from __future__ import annotations

"""Synchronize bundled shared templates into runtime data directories."""

import hashlib
import logging
import shutil
from pathlib import Path

from core.migrations.registry import StepResult
from core.paths import TEMPLATES_DIR

logger = logging.getLogger(__name__)

SYNC_TREES: tuple[str, ...] = ("common_knowledge", "common_skills", "reference")
RUNTIME_PROMPT_FILES: tuple[str, ...] = (
    "character_design_guide.md",
    "face_types.md",
    "memory/episode_extraction.md",
)
STALE_RUNTIME_FILES: tuple[str, ...] = (
    "prompts/task_delegation_rules.md",
    "prompts/communication_rules_s.md",
    "prompts/hiring_context.md",
    "prompts/meeting_chair.md",
    "prompts/tool_data_interpretation.md",
    "prompts/builder/common_knowledge_hint.md",
    "prompts/builder/reference_hint.md",
    "prompts/builder/human_notification_howto_s.md",
    "prompts/builder/human_notification_howto_other.md",
    "common_knowledge/operations/machine-tool-usage.md",
    "common_knowledge/operations/machine-workflow-engineer.md",
    "common_knowledge/operations/machine-workflow-pdm.md",
    "common_knowledge/operations/machine-workflow-reviewer.md",
    "common_knowledge/operations/machine-workflow-tester.md",
)


def _is_syncable_file(path: Path, root: Path) -> bool:
    """Return whether a bundled file participates in runtime sync and its fingerprint."""
    if path.is_symlink() or not path.is_file():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
        return False
    return path.name != "__init__.py" and path.suffix != ".pyc"


def _locale_fallbacks(locale: str) -> tuple[str, ...]:
    from core.paths import normalize_locale

    return tuple(dict.fromkeys((normalize_locale(locale), "en", "ja")))


def resolve_template_sources(locale: str) -> dict[str, Path]:
    """Resolve shared sync files independently using locale → en → ja fallback."""
    sources: dict[str, Path] = {}
    for fallback in _locale_fallbacks(locale):
        locale_root = TEMPLATES_DIR / fallback
        for tree in SYNC_TREES:
            tree_root = locale_root / tree
            if not tree_root.is_dir() or tree_root.is_symlink():
                continue
            for path in tree_root.rglob("*"):
                if not _is_syncable_file(path, tree_root):
                    continue
                if any(parent.is_symlink() for parent in path.parents if parent != TEMPLATES_DIR):
                    continue
                relative = path.relative_to(locale_root).as_posix()
                sources.setdefault(relative, path)

        prompt_root = locale_root / "prompts"
        if prompt_root.is_dir() and not prompt_root.is_symlink():
            for filename in RUNTIME_PROMPT_FILES:
                path = prompt_root / filename
                if _is_syncable_file(path, prompt_root):
                    sources.setdefault(f"prompts/{filename}", path)

    return sources


def template_fingerprint(locale: str) -> str:
    """Return a stable 12-character ID for the selected locale and file contents."""
    from core.paths import normalize_locale

    locale = normalize_locale(locale)
    digest = hashlib.sha256()
    digest.update(locale.encode("utf-8"))
    digest.update(b"\0")
    for relative, source in sorted(resolve_template_sources(locale).items()):
        content_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def sync_runtime_templates(data_dir: Path, *, locale: str, dry_run: bool) -> StepResult:
    """Overwrite shared runtime templates and remove their retired file copies."""
    try:
        sources = resolve_template_sources(locale)
        copied = 0
        identical = 0
        for relative, source in sorted(sources.items()):
            destination = data_dir / relative
            source_content = source.read_bytes()
            if destination.is_file() and not destination.is_symlink() and destination.read_bytes() == source_content:
                identical += 1
                continue
            copied += 1
            if dry_run:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                destination.unlink()
            shutil.copy2(source, destination)

        stale_files = [
            relative
            for relative in STALE_RUNTIME_FILES
            if (data_dir / relative).is_file() or (data_dir / relative).is_symlink()
        ]
        if not dry_run:
            for relative in stale_files:
                (data_dir / relative).unlink()

        action = "Would copy" if dry_run else "Copied"
        same_action = "Would leave identical" if dry_run else "identical"
        removal_action = "Would remove" if dry_run else "Removed"
        details = [f"{action} {copied} file(s); {identical} {same_action} file(s) skipped"]
        if stale_files:
            details.append(f"{removal_action} stale files: {', '.join(stale_files)}")
        else:
            details.append("No stale files to remove")
        return StepResult(changed=copied + len(stale_files), skipped=identical, details=details)
    except Exception as exc:
        logger.exception("sync_runtime_templates failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))
