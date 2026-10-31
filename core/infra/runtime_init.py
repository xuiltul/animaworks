# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""First-launch initialization: copy templates to runtime data directory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from core.anima.workspace_files import BOOTSTRAP_FILE, IDENTITY_FILE
from core.paths import TEMPLATES_DIR, get_data_dir

logger = logging.getLogger("animaworks.init")

# Directories under templates/ that are infrastructure (always copied).
# anima_templates/ is NOT included — animas are created separately.
_INFRASTRUCTURE_DIRS = {"prompts", "company", "common_skills", "common_knowledge", "reference"}


def ensure_runtime_dir(*, skip_animas: bool = False) -> Path:
    """Ensure the runtime data directory exists, seeding from templates if needed.

    Args:
        skip_animas: If True (used by interactive init), only copy
            infrastructure templates.  If False (default for server startup),
            fall back to legacy behaviour that creates a blank default anima
            when no animas exist yet.

    Returns the runtime data directory path.
    """
    from core.infra.event_export import register_activity_event_exporter

    register_activity_event_exporter()
    data_dir = get_data_dir()

    # Check for proper initialization (config.json is the marker).
    # The data_dir itself may already exist (e.g. created by setup_logging)
    # but not be fully initialized yet.
    config_json = data_dir / "config.json"
    if config_json.exists():
        from core.migrations.tracker import assert_supported_runtime_version

        assert_supported_runtime_version(data_dir)
        # Run supported migrations and synchronize bundled runtime templates.
        try:
            _run_auto_migrations(data_dir)
        except Exception:
            logger.exception("Auto-migration failed — partial state may exist; run 'animaworks migrate' manually")
        # Incremental sync must run on every startup (not as a one-shot
        # migration step) so new common_skills/common_knowledge templates
        # reach existing installations after upgrades.
        _sync_shared_templates(data_dir)
        _ensure_runtime_only_dirs(data_dir)
        logger.debug("Runtime directory already initialized: %s", data_dir)
        return data_dir

    logger.info("Initializing runtime directory at %s", data_dir)

    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(f"Templates directory not found: {TEMPLATES_DIR}. Is the project installed correctly?")

    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy infrastructure templates (prompts, company, common_skills, common_knowledge)
    _copy_infrastructure(data_dir)

    # Create runtime-only directories that have no template
    _ensure_runtime_only_dirs(data_dir)

    # Legacy fallback: if not skipping animas and no animas exist,
    # create a blank default anima so cmd_start() works out of the box.
    # Skip when setup is not yet complete — the setup wizard handles
    # anima creation interactively.
    if not skip_animas:
        from core.config import load_config as _load_config

        _cfg = _load_config(data_dir / "config.json") if (data_dir / "config.json").exists() else None
        if _cfg is not None and _cfg.setup_complete:
            animas_dir = data_dir / "animas"
            if not animas_dir.exists() or not any(animas_dir.iterdir()):
                _legacy_copy_default_anima(data_dir)

    # Generate default config.json
    _create_default_config(data_dir)

    logger.info("Runtime directory initialized: %s", data_dir)
    return data_dir


# Directories synced incrementally on every startup (new entries only).
_INCREMENTAL_SYNC_DIRS = {"common_skills", "common_knowledge", "reference"}


def _sync_shared_templates(data_dir: Path) -> None:
    """Incrementally sync new common_skills and common_knowledge from templates.

    Recursively walks the template directory tree and copies any file that
    does not yet exist in the runtime data directory, preserving user
    modifications to existing files.  Called on every startup so that version
    upgrades automatically deliver new files without ``animaworks init``.
    """
    from core.paths import _get_locale

    locale = _get_locale()
    locale_dir: Path | None = None
    for loc in (locale, "en", "ja"):
        candidate = TEMPLATES_DIR / loc
        if candidate.exists():
            locale_dir = candidate
            break
    if locale_dir is None:
        return

    for dir_name in _INCREMENTAL_SYNC_DIRS:
        src_dir = locale_dir / dir_name
        if not src_dir.is_dir():
            continue
        dst_dir = data_dir / dir_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(src_dir)
            target = dst_dir / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target)
            logger.info("Synced new template %s → %s", rel, dir_name)


def _copy_infrastructure(data_dir: Path) -> None:
    """Copy only infrastructure templates (prompts, company, etc.) to data_dir."""
    from core.paths import _get_locale

    locale = _get_locale()
    locale_dir: Path | None = None
    for loc in (locale, "en", "ja"):
        candidate = TEMPLATES_DIR / loc
        if candidate.exists():
            locale_dir = candidate
            break
    if locale_dir is None:
        logger.warning("No locale template directory found; skipping infrastructure copy")
        return

    for item in locale_dir.iterdir():
        if item.name == "anima_templates":
            continue
        target = data_dir / item.name
        if item.is_dir():
            if item.name in _INFRASTRUCTURE_DIRS and item.name != "prompts":
                shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            # BOOTSTRAP.md is only placed per-anima, not at data root
            if item.name == BOOTSTRAP_FILE:
                continue
            shutil.copy2(item, target)

    # Selected Anima-read and framework prompts are kept as runtime copies.
    from core.migrations.template_sync import RUNTIME_PROMPT_FILES

    locale_fallbacks = tuple(dict.fromkeys((locale, "en", "ja")))
    prompt_sources: dict[str, Path] = {}
    for loc in locale_fallbacks:
        for filename in RUNTIME_PROMPT_FILES:
            source = TEMPLATES_DIR / loc / "prompts" / filename
            if source.is_file() and not source.is_symlink():
                prompt_sources.setdefault(filename, source)
    for filename, source in prompt_sources.items():
        target = data_dir / "prompts" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Fill locale-specific infrastructure gaps file-by-file without overwriting
    # files copied from the selected locale above.
    for loc in locale_fallbacks:
        locale_root = TEMPLATES_DIR / loc
        for directory in _INFRASTRUCTURE_DIRS - {"prompts"}:
            source_root = locale_root / directory
            if not source_root.is_dir() or source_root.is_symlink():
                continue
            for source in source_root.rglob("*"):
                if source.is_symlink() or not source.is_file():
                    continue
                target = data_dir / directory / source.relative_to(source_root)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    # Copy models.json from _shared/config_defaults/ to data_dir root
    models_json_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
    if models_json_src.is_file():
        models_json_dst = data_dir / "models.json"
        if not models_json_dst.exists():
            shutil.copy2(models_json_src, models_json_dst)
            logger.info("Copied models.json template to %s", models_json_dst)

    # Copy permissions.global.json from _shared/config_defaults/ to data_dir root
    gp_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
    if gp_src.is_file():
        gp_dst = data_dir / "permissions.global.json"
        if not gp_dst.exists():
            shutil.copy2(gp_src, gp_dst)
            logger.info("Copied permissions.global.json template to %s", gp_dst)


def _legacy_copy_default_anima(data_dir: Path) -> None:
    """Legacy fallback: create a blank anima when auto-initialising for server."""
    from core.anima.factory import create_blank

    default_name = "default"
    animas_dir = data_dir / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    try:
        create_blank(animas_dir, default_name)
        logger.info("Legacy fallback: created blank default anima '%s'", default_name)
    except Exception:
        logger.warning("Could not create default anima", exc_info=True)


def merge_templates(data_dir: Path) -> list[str]:
    """Copy infrastructure template files that don't exist in the runtime directory.

    Walks locale templates in fallback order and selects each file from the
    first locale that provides it. Selected runtime prompts are copied and
    overwritten; other runtime files are copied only when absent.
    anima_templates/ is skipped (animas are managed separately).

    Returns a list of newly added file paths (relative to data_dir).
    """
    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(f"Templates directory not found: {TEMPLATES_DIR}. Is the project installed correctly?")

    from core.migrations.template_sync import RUNTIME_PROMPT_FILES
    from core.paths import _get_locale

    locale = _get_locale()
    locale_fallbacks = tuple(dict.fromkeys((locale, "en", "ja")))
    sources: dict[Path, Path] = {}
    found_locale_root = False
    for loc in locale_fallbacks:
        locale_dir = TEMPLATES_DIR / loc
        if not locale_dir.is_dir() or locale_dir.is_symlink():
            continue
        found_locale_root = True
        for src in locale_dir.rglob("*"):
            if src.is_symlink() or not src.is_file():
                continue
            rel = src.relative_to(locale_dir)
            parts = rel.parts
            if not parts or parts[0] in {"anima_templates", "roles"}:
                continue
            if rel.name == BOOTSTRAP_FILE and len(parts) == 1:
                continue
            if parts[0] == "prompts" and Path(*parts[1:]).as_posix() not in RUNTIME_PROMPT_FILES:
                continue
            if any(parent.is_symlink() for parent in src.parents if parent != TEMPLATES_DIR):
                continue
            sources.setdefault(rel, src)

    if not found_locale_root:
        logger.warning("No locale template directory found; skipping merge")
        return []

    added: list[str] = []

    for rel, src in sorted(sources.items()):
        dest = data_dir / rel
        is_prompt = rel.parts[0] == "prompts"
        if is_prompt or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            shutil.copy2(src, dest)
            added.append(rel.as_posix())
            logger.debug("Merged template file: %s", rel)
    if added:
        logger.info("Merged %d template file(s) into %s", len(added), data_dir)

    # Copy models.json from _shared/config_defaults/ if missing
    models_json_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
    models_json_dst = data_dir / "models.json"
    if models_json_src.is_file() and not models_json_dst.exists():
        shutil.copy2(models_json_src, models_json_dst)
        added.append("models.json")
        logger.info("Merged models.json template to %s", models_json_dst)

    # Copy permissions.global.json from _shared/config_defaults/ if missing
    gp_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
    gp_dst = data_dir / "permissions.global.json"
    if gp_src.is_file() and not gp_dst.exists():
        shutil.copy2(gp_src, gp_dst)
        added.append("permissions.global.json")
        logger.info("Merged permissions.global.json template to %s", gp_dst)

    # Ensure runtime-only directories exist
    _ensure_runtime_only_dirs(data_dir)

    return added


def reset_runtime_dir(data_dir: Path, *, skip_animas: bool = False) -> Path:
    """Delete the runtime directory entirely and re-initialize from templates.

    This is a destructive operation — all user data (episodes, knowledge,
    state, config) will be lost.
    """
    _validate_safe_path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
        logger.info("Removed runtime directory: %s", data_dir)
    return ensure_runtime_dir(skip_animas=skip_animas)


def _ensure_runtime_only_dirs(data_dir: Path) -> None:
    """Create runtime-only directories that have no template counterpart."""
    (data_dir / "animas").mkdir(parents=True, exist_ok=True)

    # Generate default .ragignore if not present
    ragignore_path = data_dir / ".ragignore"
    if not ragignore_path.exists():
        ragignore_path.write_text(
            "# Files excluded from RAG indexing\n"
            "# Patterns use fnmatch syntax (like gitignore)\n"
            "# Lines starting with # are comments\n"
            "\n"
            "# Index/TOC files (already referenced in system prompt)\n"
            "00_index.md\n"
            "\n"
            "# Archived memory files\n"
            "*/.archive/*\n"
            "*/_archived/*\n"
            "*/knowledge/archive/*\n"
            "*/knowledge/archived/*\n"
            "*/episodes/archive/*\n"
            "*/episodes/archived/*\n"
            "*/procedures/archive/*\n"
            "*/procedures/archived/*\n",
            encoding="utf-8",
        )
    inbox_dir = data_dir / "shared" / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    try:
        inbox_dir.chmod(0o700)
    except OSError:
        pass
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "channels").mkdir(parents=True, exist_ok=True)
    dm_logs_dir = data_dir / "shared" / "dm_logs"
    dm_logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        dm_logs_dir.chmod(0o700)
    except OSError:
        pass
    (data_dir / "tmp" / "attachments").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)
    (data_dir / "reference").mkdir(parents=True, exist_ok=True)

    # Create initial shared channels
    for channel in ("general", "ops"):
        channel_file = data_dir / "shared" / "channels" / f"{channel}.jsonl"
        if not channel_file.exists():
            channel_file.touch()


def _validate_safe_path(data_dir: Path) -> None:
    """Guard against deleting dangerous paths (/, /home, symlinks, etc.)."""
    resolved = data_dir.resolve()
    if data_dir.is_symlink():
        raise ValueError(f"Refusing to delete: {data_dir} is a symlink")
    if resolved == Path.home() or resolved == Path("/") or resolved.parent == Path("/"):
        raise ValueError(f"Refusing to delete {resolved} — path looks too broad. Check ANIMAWORKS_DATA_DIR.")


def _create_default_config(data_dir: Path) -> None:
    """Generate a default config.json for a freshly initialized runtime."""
    from core.config import (
        DEFAULT_MODEL_MODE_PATTERNS,
        AnimaModelConfig,
        AnimaWorksConfig,
        CredentialConfig,
        save_config,
    )

    config = AnimaWorksConfig(
        credentials={"anthropic": CredentialConfig()},
        model_modes=dict(DEFAULT_MODEL_MODE_PATTERNS),
    )

    # Auto-detect animas from the just-copied templates
    animas_dir = data_dir / "animas"
    if animas_dir.exists():
        for d in sorted(animas_dir.iterdir()):
            if d.is_dir() and (d / IDENTITY_FILE).exists():
                config.animas[d.name] = AnimaModelConfig()

    save_config(config, data_dir / "config.json")
    logger.info("Default config.json created at %s", data_dir / "config.json")


# ── Unified migration on startup ────────────────────────────


def _run_auto_migrations(data_dir: Path) -> None:
    """Run all pending migrations automatically on server startup."""
    from core.migrations.registry import MigrationRunner
    from core.migrations.steps import register_all_steps

    runner = MigrationRunner(data_dir)
    register_all_steps(runner)
    report = runner.run_all()
    if report.total_changed:
        logger.info(
            "Auto-migration: %d change(s), %d skip(s), %d error(s)",
            report.total_changed,
            report.total_skipped,
            len(report.errors),
        )
    if report.errors:
        for e in report.errors:
            logger.warning("Migration error: %s", e)
