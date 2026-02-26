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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tooling.prompt_db import ToolPromptStore

from core.paths import TEMPLATES_DIR, get_data_dir

logger = logging.getLogger("animaworks.init")

# Directories under templates/ that are infrastructure (always copied).
# anima_templates/ is NOT included — animas are created separately.
_INFRASTRUCTURE_DIRS = {"prompts", "company", "common_skills", "common_knowledge"}


def _ensure_tool_prompt_db(data_dir: Path) -> None:
    """Create and seed the tool prompt DB if needed."""
    from core.tooling.prompt_db import (
        DEFAULT_DESCRIPTIONS,
        DEFAULT_GUIDES,
        SECTION_CONDITIONS,
        ToolPromptStore,
    )

    tool_db_path = data_dir / "tool_prompts.sqlite3"
    tool_store = ToolPromptStore(tool_db_path)

    # Load section content from runtime prompts directory
    sections: dict[str, tuple[str, str | None]] = {}
    prompts_dir = data_dir / "prompts"

    _SECTION_FILES: dict[str, str] = {
        "behavior_rules": "behavior_rules.md",
        "environment": "environment.md",
        "messaging_s": "messaging_s.md",
        "messaging": "messaging.md",
        "communication_rules_s": "communication_rules_s.md",
        "communication_rules": "communication_rules.md",
        "a_reflection": "a_reflection.md",
        "hiring_context": "hiring_context.md",
    }

    for key, filename in _SECTION_FILES.items():
        filepath = prompts_dir / filename
        if filepath.exists():
            try:
                content = filepath.read_text(encoding="utf-8").strip()
                if content:
                    condition = SECTION_CONDITIONS.get(key)
                    sections[key] = (content, condition)
            except Exception:
                logger.warning("Failed to read section template: %s", filepath)

    # emotion_instruction: built at runtime from valid emotions list
    try:
        from core.prompt.builder import _build_emotion_instruction

        emotion = _build_emotion_instruction()
        if emotion:
            sections["emotion_instruction"] = (emotion, None)
    except Exception:
        logger.warning("Failed to build emotion instruction for seeding")

    tool_store.seed_defaults(
        descriptions=DEFAULT_DESCRIPTIONS,
        guides=DEFAULT_GUIDES,
        sections=sections,
    )

    # Apply incremental migrations for existing DBs
    _migrate_memory_prompts_v1(tool_store, prompts_dir)
    _migrate_praise_loop_prevention_v1(tool_store, prompts_dir)
    _migrate_behavior_rules_must_v1(tool_store, prompts_dir)

    logger.info("Tool prompt DB initialised: %s", tool_db_path)


def _migrate_memory_prompts_v1(
    tool_store: ToolPromptStore,
    prompts_dir: Path,
) -> None:
    """One-shot migration: update memory-related prompts to v1 (active style).

    Idempotent — records migration key ``memory_prompt_v1`` in a ``migrations``
    table and skips if already applied.
    """
    from core.tooling.prompt_db import (
        SECTION_CONDITIONS,
        get_default_description,
        get_default_guide,
    )

    # Ensure migrations table exists
    conn = tool_store._connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS migrations "
            "(key TEXT PRIMARY KEY, applied_at TEXT)"
        )
        row = conn.execute(
            "SELECT 1 FROM migrations WHERE key = ?",
            ("memory_prompt_v1",),
        ).fetchone()
        if row:
            return  # already applied

        # 1-4: Update tool descriptions
        for name in (
            "search_memory",
            "write_memory_file",
            "report_procedure_outcome",
            "report_knowledge_outcome",
        ):
            desc = get_default_description(name)
            if desc:
                tool_store.set_description(name, desc)

        # 5-6: Update tool guides
        for key in ("s_mcp", "non_s"):
            guide = get_default_guide(key)
            if guide:
                tool_store.set_guide(key, guide)

        # 7: Update behavior_rules section from runtime prompts file
        br_path = prompts_dir / "behavior_rules.md"
        if br_path.exists():
            content = br_path.read_text(encoding="utf-8").strip()
            if content:
                condition = SECTION_CONDITIONS.get("behavior_rules")
                tool_store.set_section("behavior_rules", content, condition)

        # Record migration
        from core.time_utils import now_jst

        conn.execute(
            "INSERT INTO migrations (key, applied_at) VALUES (?, ?)",
            ("memory_prompt_v1", now_jst().isoformat()),
        )
        conn.commit()
        logger.info("Applied migration: memory_prompt_v1")
    finally:
        conn.close()


def _migrate_praise_loop_prevention_v1(
    tool_store: ToolPromptStore,
    prompts_dir: Path,
) -> None:
    """One-shot migration: update communication/messaging prompts to prevent praise loops.

    Updates system_sections with new rules:
    - 1往復ルール (1-round-trip rule) in communication_rules
    - Board投稿ルール in messaging templates
    - 送信禁止ルール in unread_messages (via messaging sections)

    Idempotent — records migration key ``praise_loop_prevention_v1`` in a
    ``migrations`` table and skips if already applied.
    """
    from core.tooling.prompt_db import SECTION_CONDITIONS

    conn = tool_store._connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS migrations "
            "(key TEXT PRIMARY KEY, applied_at TEXT)"
        )
        row = conn.execute(
            "SELECT 1 FROM migrations WHERE key = ?",
            ("praise_loop_prevention_v1",),
        ).fetchone()
        if row:
            return  # already applied

        # Update sections from runtime prompts files
        for key in (
            "communication_rules_s",
            "communication_rules",
            "messaging_s",
            "messaging",
        ):
            path = prompts_dir / f"{key}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    condition = SECTION_CONDITIONS.get(key)
                    tool_store.set_section(key, content, condition)

        from core.time_utils import now_jst

        conn.execute(
            "INSERT INTO migrations (key, applied_at) VALUES (?, ?)",
            ("praise_loop_prevention_v1", now_jst().isoformat()),
        )
        conn.commit()
        logger.info("Applied migration: praise_loop_prevention_v1")
    finally:
        conn.close()


def _migrate_behavior_rules_must_v1(
    tool_store: ToolPromptStore,
    prompts_dir: Path,
) -> None:
    """One-shot migration: upgrade behavior_rules to MUST-level memory search constraints.

    Replaces the soft "検索してから行動" guidance with mandatory search_memory
    obligations and explicit hallucination prohibition.

    Idempotent — records migration key ``behavior_rules_must_v1`` in a
    ``migrations`` table and skips if already applied.
    """
    from core.tooling.prompt_db import SECTION_CONDITIONS

    conn = tool_store._connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS migrations "
            "(key TEXT PRIMARY KEY, applied_at TEXT)"
        )
        row = conn.execute(
            "SELECT 1 FROM migrations WHERE key = ?",
            ("behavior_rules_must_v1",),
        ).fetchone()
        if row:
            return  # already applied

        br_path = prompts_dir / "behavior_rules.md"
        if br_path.exists():
            content = br_path.read_text(encoding="utf-8").strip()
            if content:
                condition = SECTION_CONDITIONS.get("behavior_rules")
                tool_store.set_section("behavior_rules", content, condition)

        from core.time_utils import now_jst

        conn.execute(
            "INSERT INTO migrations (key, applied_at) VALUES (?, ?)",
            ("behavior_rules_must_v1", now_jst().isoformat()),
        )
        conn.commit()
        logger.info("Applied migration: behavior_rules_must_v1")
    finally:
        conn.close()


def ensure_runtime_dir(*, skip_animas: bool = False) -> Path:
    """Ensure the runtime data directory exists, seeding from templates if needed.

    Args:
        skip_animas: If True (used by interactive init), only copy
            infrastructure templates.  If False (default for server startup),
            fall back to legacy behaviour that creates a blank default anima
            when no animas exist yet.

    Returns the runtime data directory path.
    """
    data_dir = get_data_dir()

    # Check for proper initialization (config.json is the marker).
    # The data_dir itself may already exist (e.g. created by setup_logging)
    # but not be fully initialized yet.
    config_json = data_dir / "config.json"
    if config_json.exists():
        # Run Person→Anima rename migration before any other access
        try:
            from core.config.migrate import migrate_person_to_anima

            migrate_person_to_anima(data_dir)
        except Exception:
            logger.exception("Person-to-Anima migration failed")
        _maybe_migrate_config(data_dir)
        logger.debug("Runtime directory already initialized: %s", data_dir)
        return data_dir

    logger.info("Initializing runtime directory at %s", data_dir)

    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(
            f"Templates directory not found: {TEMPLATES_DIR}. "
            "Is the project installed correctly?"
        )

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

    _ensure_tool_prompt_db(data_dir)

    logger.info("Runtime directory initialized: %s", data_dir)
    return data_dir


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
            if item.name in _INFRASTRUCTURE_DIRS:
                shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            # bootstrap.md is only placed per-anima, not at data root
            if item.name == "bootstrap.md":
                continue
            shutil.copy2(item, target)

    # Copy models.json from _shared/config_defaults/ to data_dir root
    models_json_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
    if models_json_src.is_file():
        models_json_dst = data_dir / "models.json"
        if not models_json_dst.exists():
            shutil.copy2(models_json_src, models_json_dst)
            logger.info("Copied models.json template to %s", models_json_dst)


def _legacy_copy_default_anima(data_dir: Path) -> None:
    """Legacy fallback: create a blank anima when auto-initialising for server."""
    from core.anima_factory import create_blank

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

    Walks the locale-specific templates tree and copies any file that is missing
    from the runtime data directory.  prompts/ files are always overwritten to
    keep in sync. anima_templates/ is skipped (animas are managed separately).

    Returns a list of newly added file paths (relative to data_dir).
    """
    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(
            f"Templates directory not found: {TEMPLATES_DIR}. "
            "Is the project installed correctly?"
        )

    from core.paths import _get_locale

    locale = _get_locale()
    locale_dir: Path | None = None
    for loc in (locale, "en", "ja"):
        candidate = TEMPLATES_DIR / loc
        if candidate.exists():
            locale_dir = candidate
            break
    if locale_dir is None:
        logger.warning("No locale template directory found; skipping merge")
        return []

    added: list[str] = []

    # Walk locale directory for infrastructure templates
    for src in locale_dir.rglob("*"):
        if src.is_symlink() or src.is_dir():
            continue
        rel = src.relative_to(locale_dir)
        parts = rel.parts
        if parts[0] == "anima_templates":
            continue
        if rel.name == "bootstrap.md" and len(parts) == 1:
            continue
        # Roles: only copy .md files, not defaults.json (those are in _shared)
        if parts[0] == "roles":
            continue
        dest = data_dir / rel
        is_prompt = parts[0] == "prompts"
        if is_prompt or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            added.append(str(rel))
            logger.info("Merged template file: %s", rel)

    # Copy models.json from _shared/config_defaults/ if missing
    models_json_src = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
    models_json_dst = data_dir / "models.json"
    if models_json_src.is_file() and not models_json_dst.exists():
        shutil.copy2(models_json_src, models_json_dst)
        added.append("models.json")
        logger.info("Merged models.json template to %s", models_json_dst)

    # Ensure runtime-only directories exist
    _ensure_runtime_only_dirs(data_dir)

    _ensure_tool_prompt_db(data_dir)

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
    (data_dir / "shared" / "inbox").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "channels").mkdir(parents=True, exist_ok=True)
    (data_dir / "shared" / "dm_logs").mkdir(parents=True, exist_ok=True)
    (data_dir / "tmp" / "attachments").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_knowledge").mkdir(parents=True, exist_ok=True)

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
        raise ValueError(
            f"Refusing to delete {resolved} — path looks too broad. "
            "Check ANIMAWORKS_DATA_DIR."
        )


def _create_default_config(data_dir: Path) -> None:
    """Generate a default config.json for a freshly initialized runtime."""
    from core.config import (
        DEFAULT_MODEL_MODE_PATTERNS,
        AnimaWorksConfig,
        CredentialConfig,
        AnimaModelConfig,
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
            if d.is_dir() and (d / "identity.md").exists():
                config.animas[d.name] = AnimaModelConfig()

    save_config(config, data_dir / "config.json")
    logger.info("Default config.json created at %s", data_dir / "config.json")


def _maybe_migrate_config(data_dir: Path) -> None:
    """Auto-migrate existing config.md setups to config.json if needed."""
    config_path = data_dir / "config.json"
    if config_path.exists():
        return

    animas_dir = data_dir / "animas"
    if not animas_dir.exists():
        return

    has_legacy = any(
        (d / "config.md").exists()
        for d in animas_dir.iterdir()
        if d.is_dir()
    )
    if not has_legacy:
        return

    logger.info("Migrating legacy config.md files to config.json")
    from core.config.migrate import migrate_to_config_json

    migrate_to_config_json(data_dir)