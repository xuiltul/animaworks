# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Migration step implementations for AnimaWorks runtime data."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from core.migrations.registry import MigrationStep, StepResult

logger = logging.getLogger(__name__)

_MEMORY_ARCHIVE_ROOTS = ("knowledge", "episodes", "procedures")
_LEGACY_ARCHIVE_DIRS = ("archived", "_archived", ".archive")
_RAGIGNORE_ARCHIVE_COMMENT = "# Archived memory files (unified)"
_RAGIGNORE_ARCHIVE_PATTERNS = tuple(
    f"*/{memory_root}/{archive_dir}/*"
    for memory_root in _MEMORY_ARCHIVE_ROOTS
    for archive_dir in ("archive", "archived")
)


def _prime_tooling_imports() -> None:
    """Load execution sanitizers before tooling schemas to avoid import cycles."""
    import_module("core.execution._sanitize")


# ── Section mapping for system_sections resync ─────────────────────

_SECTION_FILES: dict[str, str] = {
    "behavior_rules": "behavior_rules.md",
    "environment": "environment.md",
    "messaging_s": "messaging_s.md",
    "messaging": "messaging.md",
    "communication_rules_s": "communication_rules_s.md",
    "communication_rules": "communication_rules.md",
    "a_reflection": "a_reflection.md",
}

# ── Category 1: Structural migrations ────────────────────────────


def step_person_to_anima(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Person → Anima rename. Call migrate_person_to_anima if persons/ exists."""
    details: list[str] = []
    try:
        persons_dir = data_dir / "persons"
        if not persons_dir.exists():
            return StepResult(changed=0, skipped=1, details=["persons/ not found; skip"])
        if dry_run:
            details.append("Would run migrate_person_to_anima (persons/ exists)")
            return StepResult(changed=1, skipped=0, details=details)
        from core.config.migrate import migrate_person_to_anima

        migrate_person_to_anima(data_dir)
        details.append("Person → Anima rename complete")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_person_to_anima failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_config_md_to_json(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Migrate legacy config.md files to config.json."""
    details: list[str] = []
    try:
        animas_dir = data_dir / "animas"
        if not animas_dir.exists():
            return StepResult(changed=0, skipped=1, details=["animas/ not found"])
        has_legacy = any((d / "config.md").exists() for d in animas_dir.iterdir() if d.is_dir())
        if not has_legacy:
            return StepResult(changed=0, skipped=1, details=["No config.md files found"])
        if dry_run:
            details.append("Would migrate config.md → config.json")
            return StepResult(changed=1, skipped=0, details=details)
        from core.config.migrate import migrate_to_config_json

        migrate_to_config_json(data_dir)
        details.append("config.md → config.json migration complete")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_config_md_to_json failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_model_config_to_status(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Migrate model config from config.json animas to status.json."""
    details: list[str] = []
    try:
        config_path = data_dir / "config.json"
        if not config_path.is_file():
            return StepResult(changed=0, skipped=1, details=["config.json not found"])
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        animas_section = raw.get("animas", {})
        if not animas_section:
            return StepResult(changed=0, skipped=1, details=["No animas in config.json"])
        _model_keys = {"model", "fallback_model", "max_tokens", "max_turns", "credential"}
        has_model_fields = any(
            isinstance(cfg, dict) and bool(_model_keys & set(cfg.keys())) for cfg in animas_section.values()
        )
        if not has_model_fields:
            return StepResult(changed=0, skipped=1, details=["No model fields in config.json animas"])
        from core.config.migrate import migrate_model_config_to_status

        results = migrate_model_config_to_status(data_dir, dry_run=dry_run)
        migrated = sum(1 for v in results.values() if v)
        details.extend([f"{k}: {v}" for k, v in results.items() if v])
        return StepResult(changed=migrated, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_model_config_to_status failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_credentials_migration(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Migrate shared/credentials.json to vault.json."""
    details: list[str] = []
    try:
        try:
            from core.config.vault import VaultManager
        except ImportError:
            return StepResult(changed=0, skipped=1, details=["core.config.vault not importable"])
        shared_creds = data_dir / "shared" / "credentials.json"
        if not shared_creds.is_file():
            return StepResult(changed=0, skipped=1, details=["shared/credentials.json not found"])
        if dry_run:
            details.append("Would migrate shared/credentials.json → vault.json")
            return StepResult(changed=1, skipped=0, details=details)
        vault = VaultManager(data_dir)
        count = vault.migrate_shared_credentials()
        details.append(f"Migrated {count} credential entries")
        return StepResult(changed=1 if count else 0, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_credentials_migration failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_vault_reencrypt(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Back up and re-encrypt every vault entry, rolling back on failure."""
    vault_path = data_dir / "vault.json"
    key_path = data_dir / "vault.key"
    if not vault_path.is_file():
        return StepResult(changed=0, skipped=1, details=["vault.json not found"])
    if dry_run:
        return StepResult(
            changed=1,
            skipped=0,
            details=["Would back up vault files and re-encrypt all entries"],
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    vault_backup = vault_path.with_name(f"{vault_path.name}.bak-{timestamp}")
    key_backup = key_path.with_name(f"{key_path.name}.bak-{timestamp}")
    key_existed = key_path.is_file()
    details: list[str] = []

    shutil.copy2(vault_path, vault_backup)
    details.append(f"Backed up {vault_path.name} to {vault_backup.name}")
    if key_existed:
        shutil.copy2(key_path, key_backup)
        details.append(f"Backed up {key_path.name} to {key_backup.name}")

    try:
        from core.config.vault import VaultError, VaultManager

        original = json.loads(vault_path.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise VaultError("vault.json root must be an object")

        vault = VaultManager(data_dir)
        if not key_existed:
            if not vault.generate_key():
                raise VaultError("Vault key generation is unavailable")
        elif vault._load_key() is None:
            raise VaultError("Existing vault key could not be loaded")

        encrypted: dict[str, dict[str, str]] = {}
        entry_count = 0
        for section, entries in original.items():
            if not isinstance(entries, dict):
                raise VaultError(f"Vault section {section!r} must be an object")
            encrypted_section: dict[str, str] = {}
            for key, value in entries.items():
                if not isinstance(value, str):
                    raise VaultError(f"Vault entry {section!r}/{key!r} must be a string")
                encrypted_section[key] = vault.encrypt(value)
                entry_count += 1
            encrypted[section] = encrypted_section

        vault.save_vault(encrypted)
        persisted = json.loads(vault_path.read_text(encoding="utf-8"))
        if persisted != encrypted:
            raise VaultError("Persisted vault content did not match encrypted data")
        for section, entries in persisted.items():
            for key, ciphertext in entries.items():
                if vault.decrypt(ciphertext) != original[section][key]:
                    raise VaultError(f"Round-trip verification failed for {section!r}/{key!r}")

        details.append(f"Re-encrypted and verified {entry_count} vault entries")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        shutil.copy2(vault_backup, vault_path)
        if key_existed:
            shutil.copy2(key_backup, key_path)
        else:
            key_path.unlink(missing_ok=True)
        details.append("Rolled back vault.json and vault.key")
        logger.exception("step_vault_reencrypt failed; rollback complete")
        return StepResult(changed=0, skipped=0, details=details, error=str(exc))


def step_enable_skill_catalog_router(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Enable skill catalog routing in existing config.json files.

    The feature was introduced behind a default-off flag in commit 8f8b37a3.
    Current runtime defaults should route the skill catalog by message, so
    existing runtime configs are migrated to the new default explicitly.
    """
    details: list[str] = []
    try:
        config_path = data_dir / "config.json"
        if not config_path.is_file():
            return StepResult(changed=0, skipped=1, details=["config.json not found"])

        raw = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        prompt = raw.get("prompt")
        if prompt is None:
            prompt = {}
        if not isinstance(prompt, dict):
            return StepResult(changed=0, skipped=1, details=["config.json prompt section is not an object"])

        defaults = {
            "skill_catalog_router_top_k": 5,
            "skill_catalog_router_min_score": 1.15,
            "skill_catalog_router_include_body": True,
        }
        changed_fields = []
        if prompt.get("skill_catalog_router_enabled") is not True:
            changed_fields.append("skill_catalog_router_enabled")
        changed_fields.extend(key for key in defaults if key not in prompt)
        if not changed_fields:
            return StepResult(changed=0, skipped=1, details=["skill catalog router already enabled"])

        if dry_run:
            details.append(f"Would update prompt fields: {', '.join(changed_fields)}")
            return StepResult(changed=1, skipped=0, details=details)

        prompt["skill_catalog_router_enabled"] = True
        for key, value in defaults.items():
            prompt.setdefault(key, value)
        raw["prompt"] = prompt
        config_path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            from core.config import invalidate_cache

            invalidate_cache()
        except Exception:
            logger.debug("Failed to invalidate config cache after skill router migration", exc_info=True)
        details.append(f"Updated prompt fields: {', '.join(changed_fields)}")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_enable_skill_catalog_router failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


# ── Category 2: Per-anima file migrations ───────────────────────


def _iter_anima_dirs(data_dir: Path) -> list[Path]:
    """Return anima directories that have identity.md."""
    animas_dir = data_dir / "animas"
    if not animas_dir.exists():
        return []
    return [d for d in sorted(animas_dir.iterdir()) if d.is_dir() and (d / "identity.md").exists()]


def _archive_destination(
    relative_path: Path,
    legacy_dir_name: str,
    occupied_files: set[Path],
    occupied_dirs: set[Path],
) -> tuple[Path | None, bool]:
    """Return an unused compatible archive-relative destination."""
    if relative_path in occupied_dirs or any(parent in occupied_files for parent in relative_path.parents):
        return None, False
    if relative_path not in occupied_files:
        return relative_path, False

    stem = relative_path.stem
    suffix = relative_path.suffix
    parent = relative_path.parent
    base_name = f"{stem}__from_{legacy_dir_name}"
    candidate = parent / f"{base_name}{suffix}"
    sequence = 2
    while candidate in occupied_files:
        candidate = parent / f"{base_name}_{sequence}{suffix}"
        sequence += 1
    if candidate in occupied_dirs:
        return None, False
    return candidate, True


def step_knowledge_archive_unify(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Move legacy per-memory archive directories into ``archive/``."""
    del verbose
    moved = 0
    collision_renames = 0
    details: list[str] = []
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            if anima_dir.is_symlink():
                logger.warning("Skipping symlinked anima directory during archive migration: %s", anima_dir)
                continue
            for memory_root_name in _MEMORY_ARCHIVE_ROOTS:
                memory_root = anima_dir / memory_root_name
                if memory_root.is_symlink():
                    logger.warning("Skipping symlinked memory root during archive migration: %s", memory_root)
                    continue
                archive_dir = memory_root / "archive"
                if archive_dir.is_symlink():
                    logger.warning("Skipping symlinked canonical archive during migration: %s", archive_dir)
                    continue
                archive_entries = list(archive_dir.rglob("*")) if archive_dir.is_dir() else []
                occupied_files = {
                    path.relative_to(archive_dir) for path in archive_entries if path.is_file() or path.is_symlink()
                }
                occupied_dirs = {
                    path.relative_to(archive_dir) for path in archive_entries if path.is_dir() and not path.is_symlink()
                }

                for legacy_dir_name in _LEGACY_ARCHIVE_DIRS:
                    legacy_dir = memory_root / legacy_dir_name
                    if legacy_dir.is_symlink():
                        logger.warning("Skipping symlinked legacy archive during migration: %s", legacy_dir)
                        continue
                    if not legacy_dir.is_dir():
                        continue
                    legacy_files = sorted(
                        (path for path in legacy_dir.rglob("*") if path.is_file() or path.is_symlink()),
                        key=lambda path: str(path.relative_to(legacy_dir)),
                    )
                    for source in legacy_files:
                        relative_path = source.relative_to(legacy_dir)
                        destination_relative, renamed = _archive_destination(
                            relative_path,
                            legacy_dir_name,
                            occupied_files,
                            occupied_dirs,
                        )
                        if destination_relative is None:
                            logger.warning(
                                "Skipping archive migration due to file/directory collision: %s",
                                source,
                            )
                            details.append(f"Skipped destination type conflict: {source}")
                            continue
                        occupied_files.add(destination_relative)
                        occupied_dirs.update(destination_relative.parents)
                        moved += 1
                        collision_renames += int(renamed)
                        if not dry_run:
                            destination = archive_dir / destination_relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(source), str(destination))

                    if not dry_run:
                        for child_dir in sorted(
                            (path for path in legacy_dir.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts),
                            reverse=True,
                        ):
                            try:
                                child_dir.rmdir()
                            except OSError:
                                pass
                        try:
                            legacy_dir.rmdir()
                        except OSError:
                            pass

        action = "Would move" if dry_run else "Moved"
        details.append(f"{action} {moved} archived memory files")
        details.append(f"Collision-renamed files: {collision_renames}")
        return StepResult(changed=moved, skipped=int(moved == 0), details=details)
    except Exception as exc:
        logger.exception("step_knowledge_archive_unify failed")
        return StepResult(changed=0, skipped=0, details=details, error=str(exc))


def step_ragignore_archive_patterns(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Add unified archive exclusions to an existing ``.ragignore`` file."""
    del verbose
    ragignore_path = data_dir / ".ragignore"
    if ragignore_path.is_symlink():
        logger.warning("Skipping symlinked .ragignore during migration: %s", ragignore_path)
        return StepResult(changed=0, skipped=1, details=[".ragignore is a symlink; skip"])
    if not ragignore_path.is_file():
        return StepResult(changed=0, skipped=1, details=[".ragignore not found; skip"])

    try:
        content = ragignore_path.read_text(encoding="utf-8")
        existing_lines = {line.strip() for line in content.splitlines()}
        missing_patterns = [pattern for pattern in _RAGIGNORE_ARCHIVE_PATTERNS if pattern not in existing_lines]
        if not missing_patterns:
            return StepResult(changed=0, skipped=1, details=["Archive patterns already present"])

        details = [f"Added .ragignore pattern: {pattern}" for pattern in missing_patterns]
        if dry_run:
            return StepResult(
                changed=1, skipped=0, details=[detail.replace("Added", "Would add") for detail in details]
            )

        additions: list[str] = []
        if _RAGIGNORE_ARCHIVE_COMMENT not in existing_lines:
            additions.append(_RAGIGNORE_ARCHIVE_COMMENT)
        additions.extend(missing_patterns)
        base_content = content.rstrip("\n")
        separator = "\n\n" if base_content else ""
        ragignore_path.write_text(
            base_content + separator + "\n".join(additions) + "\n",
            encoding="utf-8",
        )
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_ragignore_archive_patterns failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_legacy_flat_skill_migration(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Convert legacy flat local skill files to trusted SKILL.md bundles."""
    from core.migrations.legacy_flat_skills import migrate_legacy_flat_skills

    return migrate_legacy_flat_skills(data_dir, dry_run=dry_run, verbose=verbose)


def step_current_task_rename(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Rename current_task.md to current_state.md for each anima."""
    details: list[str] = []
    changed = 0
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            state_dir = anima_dir / "state"
            old_task = state_dir / "current_task.md"
            new_state = state_dir / "current_state.md"
            root_task = anima_dir / "current_task.md"
            if root_task.exists():
                if dry_run:
                    details.append(f"{anima_dir.name}: would migrate root current_task.md")
                else:
                    try:
                        if not new_state.exists():
                            state_dir.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(root_task), str(new_state))
                            details.append(f"{anima_dir.name}: root current_task.md → state/")
                        else:
                            root_task.unlink()
                            details.append(f"{anima_dir.name}: removed duplicate root current_task.md")
                        changed += 1
                    except OSError as exc:
                        details.append(f"{anima_dir.name}: failed - {exc}")
                continue
            if old_task.exists() and not new_state.exists():
                if dry_run:
                    details.append(f"{anima_dir.name}: would rename current_task.md → current_state.md")
                else:
                    try:
                        old_task.rename(new_state)
                        details.append(f"{anima_dir.name}: current_task.md → current_state.md")
                        changed += 1
                    except OSError as exc:
                        details.append(f"{anima_dir.name}: failed - {exc}")
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_current_task_rename failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_pending_merge(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Merge pending.md into current_state.md for each anima."""
    details: list[str] = []
    changed = 0
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            pending = anima_dir / "state" / "pending.md"
            if not pending.exists():
                continue
            content = pending.read_text(encoding="utf-8").strip()
            if not content:
                if dry_run:
                    details.append(f"{anima_dir.name}: would remove empty pending.md")
                else:
                    pending.unlink()
                    details.append(f"{anima_dir.name}: removed empty pending.md")
                changed += 1
                continue
            current = anima_dir / "state" / "current_state.md"
            if dry_run:
                details.append(f"{anima_dir.name}: would merge pending.md into current_state.md")
                changed += 1
                continue
            existing = current.read_text(encoding="utf-8") if current.exists() else ""
            merged = existing.rstrip() + "\n\n## Migrated from pending.md\n\n" + content
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_text(merged, encoding="utf-8")
            pending.unlink()
            details.append(f"{anima_dir.name}: merged pending.md into current_state.md")
            changed += 1
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_pending_merge failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_permissions_migration(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Migrate permissions.md to permissions.json for each anima."""
    details: list[str] = []
    changed = 0
    try:
        from core.config.migrate import migrate_permissions_md_to_json

        for anima_dir in _iter_anima_dirs(data_dir):
            md_path = anima_dir / "permissions.md"
            json_path = anima_dir / "permissions.json"
            if not md_path.exists() or json_path.exists():
                continue
            if dry_run:
                details.append(f"{anima_dir.name}: would migrate permissions.md → permissions.json")
                changed += 1
                continue
            try:
                migrate_permissions_md_to_json(anima_dir)
                details.append(f"{anima_dir.name}: permissions.md → permissions.json")
                changed += 1
            except Exception as exc:
                details.append(f"{anima_dir.name}: failed - {exc}")
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_permissions_migration failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_shortterm_layout(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Move shortterm/session_state.json to shortterm/chat/ for each anima."""
    details: list[str] = []
    changed = 0
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            shortterm = anima_dir / "shortterm"
            root_session = shortterm / "session_state.json"
            chat_dir = shortterm / "chat"
            if not root_session.exists():
                continue
            if (chat_dir / "session_state.json").exists():
                if dry_run:
                    details.append(f"{anima_dir.name}: would remove root session_state.json (chat/ has one)")
                else:
                    root_session.unlink()
                changed += 1
                continue
            if dry_run:
                details.append(f"{anima_dir.name}: would move session_state.json → shortterm/chat/")
            else:
                chat_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(root_session), str(chat_dir / "session_state.json"))
                details.append(f"{anima_dir.name}: session_state.json → shortterm/chat/")
            changed += 1
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_shortterm_layout failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_cron_format(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Migrate cron.md from Japanese schedules to cron expressions."""
    details: list[str] = []
    try:
        animas_dir = data_dir / "animas"
        if not animas_dir.exists():
            return StepResult(changed=0, skipped=1, details=["animas/ not found"])
        if dry_run:
            count = sum(1 for d in animas_dir.iterdir() if d.is_dir() and (d / "cron.md").exists())
            if count:
                details.append(f"Would migrate cron.md for {count} anima(s)")
            return StepResult(changed=count, skipped=0, details=details)
        from core.config.migrate import migrate_all_cron

        count = migrate_all_cron(animas_dir)  # takes animas/ dir, not data_dir
        details.append(f"Migrated cron.md for {count} anima(s)")
        return StepResult(changed=count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_cron_format failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_knowledge_frontmatter(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Repair knowledge frontmatter for each anima."""
    details: list[str] = []
    changed = 0
    try:
        try:
            from core.memory.frontmatter import FrontmatterService
        except ImportError:
            return StepResult(changed=0, skipped=1, details=["FrontmatterService not importable"])
        for anima_dir in _iter_anima_dirs(data_dir):
            knowledge_dir = anima_dir / "knowledge"
            if not knowledge_dir.exists():
                continue
            svc = FrontmatterService(anima_dir, knowledge_dir, anima_dir / "procedures")
            if dry_run:
                md_count = len(list(knowledge_dir.glob("*.md")))
                if md_count:
                    details.append(f"{anima_dir.name}: would repair {md_count} knowledge file(s)")
                    changed += 1
                continue
            try:
                n = svc.repair_knowledge_frontmatter()
                if n:
                    details.append(f"{anima_dir.name}: repaired {n} knowledge file(s)")
                    changed += n
            except Exception as exc:
                details.append(f"{anima_dir.name}: failed - {exc}")
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_knowledge_frontmatter failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_procedure_frontmatter(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Ensure procedure frontmatter for each anima."""
    details: list[str] = []
    changed = 0
    try:
        try:
            from core.memory.frontmatter import FrontmatterService
        except ImportError:
            return StepResult(changed=0, skipped=1, details=["FrontmatterService not importable"])
        for anima_dir in _iter_anima_dirs(data_dir):
            procedures_dir = anima_dir / "procedures"
            if not procedures_dir.exists():
                continue
            svc = FrontmatterService(anima_dir, anima_dir / "knowledge", procedures_dir)
            if dry_run:
                md_count = len(list(procedures_dir.glob("*.md")))
                if md_count:
                    details.append(f"{anima_dir.name}: would ensure frontmatter for {md_count} procedure(s)")
                    changed += 1
                continue
            try:
                n = svc.ensure_procedure_frontmatter()
                if n:
                    details.append(f"{anima_dir.name}: added frontmatter to {n} procedure(s)")
                    changed += n
            except Exception as exc:
                details.append(f"{anima_dir.name}: failed - {exc}")
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_procedure_frontmatter failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_current_task_references(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Replace current_task references with current_state in anima config files."""
    details: list[str] = []
    changed = 0
    files_to_scan = ["heartbeat.md", "cron.md", "injection.md"]
    patterns = [
        (re.compile(r"current_task\.md", re.IGNORECASE), "current_state.md"),
        (re.compile(r"\bcurrent_task\b"), "current_state"),
    ]
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            for fname in files_to_scan:
                path = anima_dir / fname
                if not path.exists():
                    continue
                content = path.read_text(encoding="utf-8")
                new_content = content
                for pat, repl in patterns:
                    new_content = pat.sub(repl, new_content)
                if new_content != content:
                    if dry_run:
                        details.append(f"{anima_dir.name}/{fname}: would replace current_task refs")
                    else:
                        path.write_text(new_content, encoding="utf-8")
                        details.append(f"{anima_dir.name}/{fname}: replaced current_task refs")
                    changed += 1
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_current_task_references failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


# ── Category 3: Framework template sync ──────────────────────────


def step_prompt_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Resync prompts/ from templates via merge_templates."""
    details: list[str] = []
    try:
        if dry_run:
            details.append("Would run merge_templates (prompts/ overwritten)")
            return StepResult(changed=1, skipped=0, details=details)
        _prime_tooling_imports()
        from core.init import merge_templates

        added = merge_templates(data_dir)
        details.append(f"Merged {len(added)} template file(s)")
        return StepResult(changed=len(added), skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_prompt_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def _count_copytree_files(src: Path, dst: Path) -> int:
    """Count files that would be copied/overwritten."""
    if not src.exists():
        return 0
    count = 0
    for p in src.rglob("*"):
        if p.is_file():
            count += 1
    return count


def step_common_knowledge_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Overwrite common_knowledge/ from templates."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR, _get_locale

        locale = _get_locale()
        for loc in (locale, "en", "ja"):
            candidate = TEMPLATES_DIR / loc / "common_knowledge"
            if candidate.exists():
                src = candidate
                break
        else:
            return StepResult(changed=0, skipped=1, details=["No common_knowledge template found"])
        dst = data_dir / "common_knowledge"
        count = _count_copytree_files(src, dst)
        if dry_run:
            details.append(f"Would copy {count} file(s) to common_knowledge/")
            return StepResult(changed=count, skipped=0, details=details)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        details.append(f"Copied {count} file(s) to common_knowledge/")
        return StepResult(changed=count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_common_knowledge_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_common_skills_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Overwrite common_skills/ from templates."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR, _get_locale

        locale = _get_locale()
        for loc in (locale, "en", "ja"):
            candidate = TEMPLATES_DIR / loc / "common_skills"
            if candidate.exists():
                src = candidate
                break
        else:
            return StepResult(changed=0, skipped=1, details=["No common_skills template found"])
        dst = data_dir / "common_skills"
        count = _count_copytree_files(src, dst)
        if dry_run:
            details.append(f"Would copy {count} file(s) to common_skills/")
            return StepResult(changed=count, skipped=0, details=details)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        details.append(f"Copied {count} file(s) to common_skills/")
        return StepResult(changed=count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_common_skills_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_skill_description_use_when_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Resync common_skills/ to migrate descriptions to Use-when pattern."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR, _get_locale

        locale = _get_locale()
        for loc in (locale, "en", "ja"):
            candidate = TEMPLATES_DIR / loc / "common_skills"
            if candidate.exists():
                src = candidate
                break
        else:
            return StepResult(changed=0, skipped=1, details=["No common_skills template found"])
        dst = data_dir / "common_skills"
        count = _count_copytree_files(src, dst)
        if dry_run:
            details.append(f"Would copy {count} file(s) to common_skills/ (Use-when migration)")
            return StepResult(changed=count, skipped=0, details=details)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        details.append(f"Copied {count} file(s) to common_skills/ (Use-when migration)")
        return StepResult(changed=count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_skill_description_use_when_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_reference_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Overwrite reference/ from templates."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR, _get_locale

        locale = _get_locale()
        for loc in (locale, "en", "ja"):
            candidate = TEMPLATES_DIR / loc / "reference"
            if candidate.exists():
                src = candidate
                break
        else:
            return StepResult(changed=0, skipped=1, details=["No reference template found"])
        dst = data_dir / "reference"
        count = _count_copytree_files(src, dst)
        if dry_run:
            details.append(f"Would copy {count} file(s) to reference/")
            return StepResult(changed=count, skipped=0, details=details)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        details.append(f"Copied {count} file(s) to reference/")
        return StepResult(changed=count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_reference_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_models_json_create(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Create models.json from template if missing."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR

        dst = data_dir / "models.json"
        if dst.exists():
            return StepResult(changed=0, skipped=1, details=["models.json already exists"])
        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "models.json"
        if not src.is_file():
            return StepResult(changed=0, skipped=1, details=["models.json template not found"])
        if dry_run:
            details.append("Would copy models.json from template")
            return StepResult(changed=1, skipped=0, details=details)
        shutil.copy2(src, dst)
        details.append("Created models.json from template")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_models_json_create failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_global_permissions_create(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Create permissions.global.json from template if missing."""
    details: list[str] = []
    try:
        from core.paths import TEMPLATES_DIR

        dst = data_dir / "permissions.global.json"
        if dst.exists():
            return StepResult(changed=0, skipped=1, details=["permissions.global.json already exists"])
        src = TEMPLATES_DIR / "_shared" / "config_defaults" / "permissions.global.json"
        if not src.is_file():
            return StepResult(changed=0, skipped=1, details=["permissions.global.json template not found"])
        if dry_run:
            details.append("Would copy permissions.global.json from template")
            return StepResult(changed=1, skipped=0, details=details)
        shutil.copy2(src, dst)
        details.append("Created permissions.global.json from template")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_global_permissions_create failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


# ── Category 4: SQLite DB sync ──────────────────────────────────


def step_v056_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """v0.5.6: Resync common_knowledge + prompts for message-quality-protocol."""
    details: list[str] = []
    total = 0
    r1 = step_common_knowledge_resync(data_dir, dry_run, verbose)
    total += r1.changed
    details.extend(r1.details)
    r2 = step_prompt_resync(data_dir, dry_run, verbose)
    total += r2.changed
    details.extend(r2.details)
    return StepResult(changed=total, skipped=0, details=details)


def step_v056_heartbeat_quality_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """v0.5.6: Resync prompts for heartbeat quality improvement (Issue #138)."""
    details: list[str] = []
    total = 0
    r1 = step_prompt_resync(data_dir, dry_run, verbose)
    total += r1.changed
    details.extend(r1.details)
    return StepResult(changed=total, skipped=0, details=details)


def step_task_delegation_to_common_knowledge(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Move task_delegation_rules from prompts/ to common_knowledge/operations/.

    Resyncs common_knowledge (deploys task-delegation-guide.md), prompts
    (deploys updated heartbeat.md/inbox_message.md without the old
    placeholder), and removes the stale prompts/task_delegation_rules.md
    from the runtime directory.
    """
    details: list[str] = []
    total = 0

    r1 = step_common_knowledge_resync(data_dir, dry_run, verbose)
    total += r1.changed
    details.extend(r1.details)

    r2 = step_prompt_resync(data_dir, dry_run, verbose)
    total += r2.changed
    details.extend(r2.details)

    stale = data_dir / "prompts" / "task_delegation_rules.md"
    if stale.is_file():
        if dry_run:
            details.append("Would remove stale prompts/task_delegation_rules.md")
        else:
            stale.unlink()
            details.append("Removed stale prompts/task_delegation_rules.md")
        total += 1

    return StepResult(changed=total, skipped=0, details=details)


def step_v060_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """v0.6.0: Full template resync + models.json update for Mode D/G.

    Deploys Korean locale templates, meeting-room assets, and adds
    cursor/* / gemini/* entries to existing models.json.
    """
    details: list[str] = []
    total = 0

    # Template resync
    for resync_fn in (
        step_common_knowledge_resync,
        step_common_skills_resync,
        step_reference_resync,
        step_prompt_resync,
    ):
        r = resync_fn(data_dir, dry_run, verbose)
        total += r.changed
        details.extend(r.details)

    # models.json: inject cursor/* and gemini/* entries if absent
    models_path = data_dir / "models.json"
    if models_path.exists():
        try:
            raw: dict[str, Any] = json.loads(models_path.read_text(encoding="utf-8"))
            new_entries = {
                "cursor/*": {"mode": "D", "context_window": 1000000},
                "gemini/*": {"mode": "G", "context_window": 1000000},
            }
            added: list[str] = []
            for key, val in new_entries.items():
                if key not in raw:
                    if not dry_run:
                        raw[key] = val
                    added.append(key)

            if added:
                if not dry_run:
                    models_path.write_text(
                        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                details.append(f"Added models.json entries: {', '.join(added)}")
                total += len(added)
            else:
                details.append("models.json already has cursor/*/gemini/* entries")
        except Exception as exc:
            details.append(f"models.json update skipped: {exc}")

    return StepResult(changed=total, skipped=0, details=details)


def step_grok_models_json(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Add Mode X Grok Build entries to an existing runtime models.json."""
    models_path = data_dir / "models.json"
    if not models_path.exists():
        return StepResult(changed=0, skipped=1, details=["models.json not found"])

    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return StepResult(changed=0, skipped=1, details=["models.json is not an object"])

        new_entries = {
            "grok/grok-4.5": {"mode": "X", "context_window": 500000},
            "grok/*": {"mode": "X", "context_window": 500000},
        }
        added = [key for key in new_entries if key not in raw]
        if not added:
            return StepResult(changed=0, skipped=1, details=["models.json already has Grok entries"])

        if not dry_run:
            for key in added:
                raw[key] = new_entries[key]
            models_path.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        action = "Would add" if dry_run else "Added"
        return StepResult(
            changed=len(added),
            skipped=0,
            details=[f"{action} models.json entries: {', '.join(added)}"],
        )
    except (json.JSONDecodeError, OSError) as exc:
        return StepResult(changed=0, skipped=1, details=[f"models.json update skipped: {exc}"])


def step_cross_anima_write_guidance(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Resync common_knowledge + prompts to deploy cross-Anima write boundary guidance.

    Templates now include guidance that subordinates cannot write to another
    Anima's knowledge/ directory, and that common_knowledge/ should be used
    for shared output.  Affected: memory_guide.md, task-delegation-guide.md,
    hierarchy-rules.md.
    """
    details: list[str] = []
    total = 0

    r1 = step_common_knowledge_resync(data_dir, dry_run, verbose)
    total += r1.changed
    details.extend(r1.details)

    r2 = step_prompt_resync(data_dir, dry_run, verbose)
    total += r2.changed
    details.extend(r2.details)

    return StepResult(changed=total, skipped=0, details=details)


# Pre-rename paths under common_knowledge/operations/ (en/ja templates moved
# these files into operations/machine/ in 2026-03).
_STALE_MACHINE_DOC_PATHS: tuple[str, ...] = (
    "operations/machine-tool-usage.md",
    "operations/machine-workflow-engineer.md",
    "operations/machine-workflow-pdm.md",
    "operations/machine-workflow-reviewer.md",
    "operations/machine-workflow-tester.md",
)


def step_common_knowledge_team_design_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Deploy team-design templates, machine/ layout, and updated 00_index.

    Re-runs common_knowledge template sync so new paths (team-design/, etc.)
    appear in ~/.animaworks/common_knowledge/. Removes obsolete flat
    machine-*.md files under operations/ left from older template layouts.
    """
    details: list[str] = []
    total = 0

    r_ck = step_common_knowledge_resync(data_dir, dry_run, verbose)
    total += r_ck.changed
    details.extend(r_ck.details)

    ck_root = data_dir / "common_knowledge"
    removed = 0
    for rel in _STALE_MACHINE_DOC_PATHS:
        path = ck_root / rel
        if not path.is_file():
            continue
        if dry_run:
            details.append(f"Would remove stale {rel}")
        else:
            path.unlink()
            details.append(f"Removed stale {rel}")
        removed += 1
    total += removed

    return StepResult(changed=total, skipped=0, details=details)


def step_v062_skill_removal_and_activity_log(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """v0.6.2: Resync templates for skill tool removal, activity_log scope, completion_gate.

    Covers:
    - common_knowledge/ resync (Channel D removal, skill→read_memory_file, activity_log scope)
    - prompts/ resync (2-phase consolidation, episode_extraction.md, memory_guide)
    - reference/ resync (Channel D removal, priming-channels)
    """
    details: list[str] = []
    total = 0

    for resync_fn in (
        step_common_knowledge_resync,
        step_common_skills_resync,
        step_prompt_resync,
        step_reference_resync,
    ):
        r = resync_fn(data_dir, dry_run, verbose)
        total += r.changed
        details.extend(r.details)

    return StepResult(changed=total, skipped=0, details=details)


def step_v063_behavior_rules_action_rules_skill_sync(
    data_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> StepResult:
    """v0.6.3: Force-sync behavior/action rules, skills docs, references, and DB guides.

    This aggregate step deliberately re-runs the current sync helpers under a
    new migration ID so runtimes that already applied historical resync steps
    still receive the latest behavior_rules, common_knowledge, common_skills,
    and reference files.
    """
    details: list[str] = ["v063 aggregate resync: behavior rules, action rules, skill docs, prompts"]
    total = 0
    skipped = 0
    errors: list[str] = []

    for resync_fn in (
        step_common_knowledge_resync,
        step_common_skills_resync,
        step_reference_resync,
        step_prompt_resync,
    ):
        result = resync_fn(data_dir, dry_run, verbose)
        total += result.changed
        skipped += result.skipped
        details.extend(result.details)
        if result.error:
            errors.append(f"{resync_fn.__name__}: {result.error}")

    error = "; ".join(errors) if errors else None
    return StepResult(changed=total, skipped=skipped, details=details, error=error)


# ── Category 5: Version tracking ────────────────────────────────


def step_update_version(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """No-op step for display; version update is handled by runner."""
    return StepResult(changed=1, skipped=0, details=["migration_state.json"])


# ── Registration ──────────────────────────────────────────────────


def register_all_steps(runner: Any) -> None:
    """Register all migration steps in execution order."""
    steps = [
        MigrationStep("person_to_anima", "Person → Anima rename", "structural", step_person_to_anima),
        MigrationStep("config_md_to_json", "config.md → config.json", "structural", step_config_md_to_json),
        MigrationStep(
            "model_config_to_status", "Model config → status.json", "structural", step_model_config_to_status
        ),
        MigrationStep("credentials_migration", "Credentials → vault", "structural", step_credentials_migration),
        MigrationStep(
            "vault_reencrypt_20260715",
            "Re-encrypt vault entries with a verified key",
            "structural",
            step_vault_reencrypt,
        ),
        MigrationStep(
            "enable_skill_catalog_router",
            "Enable skill catalog router in config",
            "structural",
            step_enable_skill_catalog_router,
        ),
        MigrationStep("current_task_rename", "current_task → current_state", "per_anima", step_current_task_rename),
        MigrationStep("pending_merge", "Merge pending.md into current_state", "per_anima", step_pending_merge),
        MigrationStep(
            "permissions_migration", "permissions.md → permissions.json", "per_anima", step_permissions_migration
        ),
        MigrationStep("shortterm_layout", "shortterm/session_state → chat/", "per_anima", step_shortterm_layout),
        MigrationStep("cron_format", "cron.md Japanese → cron expressions", "per_anima", step_cron_format),
        MigrationStep("knowledge_frontmatter", "Repair knowledge frontmatter", "per_anima", step_knowledge_frontmatter),
        MigrationStep("procedure_frontmatter", "Ensure procedure frontmatter", "per_anima", step_procedure_frontmatter),
        MigrationStep(
            "current_task_references", "Replace current_task refs in config", "per_anima", step_current_task_references
        ),
        MigrationStep("prompt_resync", "Resync prompts/ from templates", "template_sync", step_prompt_resync),
        MigrationStep(
            "memory_hygiene_prompt_resync_20260718",
            "Resync prompts for memory hygiene section",
            "template_sync",
            step_prompt_resync,
        ),
        MigrationStep(
            "common_knowledge_resync", "Resync common_knowledge/", "template_sync", step_common_knowledge_resync
        ),
        MigrationStep("common_skills_resync", "Resync common_skills/", "template_sync", step_common_skills_resync),
        MigrationStep("reference_resync", "Resync reference/", "template_sync", step_reference_resync),
        MigrationStep("models_json_create", "Create models.json if missing", "template_sync", step_models_json_create),
        MigrationStep(
            "global_permissions_create",
            "Create permissions.global.json if missing",
            "template_sync",
            step_global_permissions_create,
        ),
        MigrationStep(
            "v056_resync",
            "v0.5.6: Resync common_knowledge + prompts (message-quality-protocol)",
            "template_sync",
            step_v056_resync,
        ),
        MigrationStep(
            "v056_heartbeat_quality_resync",
            "v0.5.6: Resync prompts (heartbeat quality)",
            "template_sync",
            step_v056_heartbeat_quality_resync,
        ),
        MigrationStep(
            "task_delegation_to_common_knowledge",
            "Move task_delegation_rules to common_knowledge",
            "template_sync",
            step_task_delegation_to_common_knowledge,
        ),
        MigrationStep(
            "v060_resync",
            "v0.6.0: Full template resync + Mode D/G models.json",
            "template_sync",
            step_v060_resync,
        ),
        MigrationStep(
            "grok_models_json",
            "Add Mode X Grok Build models.json entries",
            "template_sync",
            step_grok_models_json,
        ),
        MigrationStep(
            "cross_anima_write_guidance",
            "Deploy cross-Anima write boundary guidance to prompts + common_knowledge",
            "template_sync",
            step_cross_anima_write_guidance,
        ),
        MigrationStep(
            "common_knowledge_team_design_resync",
            "Resync common_knowledge (team-design + operations/machine layout)",
            "template_sync",
            step_common_knowledge_team_design_resync,
        ),
        MigrationStep(
            "skill_description_use_when_resync",
            "Resync common_skills/ with Use-when descriptions",
            "template_sync",
            step_skill_description_use_when_resync,
        ),
        MigrationStep(
            "v062_skill_removal_and_activity_log",
            "v0.6.2: Skill tool removal + activity_log scope + completion_gate",
            "template_sync",
            step_v062_skill_removal_and_activity_log,
        ),
        MigrationStep(
            "v063_behavior_rules_action_rules_skill_sync",
            "v0.6.3: Behavior/action rules + skill docs runtime sync",
            "template_sync",
            step_v063_behavior_rules_action_rules_skill_sync,
        ),
        MigrationStep(
            "legacy_flat_skill_migration",
            "Convert legacy flat skills to trusted SKILL.md bundles",
            "structural",
            step_legacy_flat_skill_migration,
        ),
        MigrationStep(
            "knowledge_archive_unify_20260718",
            "Unify memory archive directory names",
            "per_anima",
            step_knowledge_archive_unify,
        ),
        MigrationStep(
            "ragignore_archive_patterns_20260718",
            "Add unified archive patterns to .ragignore",
            "structural",
            step_ragignore_archive_patterns,
        ),
        MigrationStep("update_version", "Update migration_state.json", "version", step_update_version),
    ]
    for s in steps:
        runner.register(s)
