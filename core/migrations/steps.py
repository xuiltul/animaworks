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
from pathlib import Path
from typing import Any

from core.migrations.registry import MigrationStep, StepResult

logger = logging.getLogger(__name__)

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


# ── Category 2: Per-anima file migrations ───────────────────────


def _iter_anima_dirs(data_dir: Path) -> list[Path]:
    """Return anima directories that have identity.md."""
    animas_dir = data_dir / "animas"
    if not animas_dir.exists():
        return []
    return [d for d in sorted(animas_dir.iterdir()) if d.is_dir() and (d / "identity.md").exists()]


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


def step_tool_prompt_db_init(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Initialize tool prompt DB and run existing migrations."""
    details: list[str] = []
    try:
        if dry_run:
            db_path = data_dir / "tool_prompts.sqlite3"
            details.append("Would ensure tool_prompts.sqlite3 initialized")
            return StepResult(changed=1 if not db_path.exists() else 0, skipped=0, details=details)
        from core.init import _ensure_tool_prompt_db

        _ensure_tool_prompt_db(data_dir)
        details.append("Tool prompt DB initialized")
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_tool_prompt_db_init failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_system_sections_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Resync system_sections from prompts/ into SQLite DB."""
    details: list[str] = []
    try:
        from core.tooling.prompt_db import SECTION_CONDITIONS, ToolPromptStore

        tool_db_path = data_dir / "tool_prompts.sqlite3"
        if not tool_db_path.exists():
            return StepResult(changed=0, skipped=1, details=["tool_prompts.sqlite3 not found"])
        prompts_dir = data_dir / "prompts"
        store = ToolPromptStore(tool_db_path)
        updated: list[str] = []
        for key, filename in _SECTION_FILES.items():
            filepath = prompts_dir / filename
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue
            condition = SECTION_CONDITIONS.get(key)
            if dry_run:
                updated.append(key)
                continue
            store.set_section(key, content, condition)
            updated.append(key)
        if not dry_run:
            try:
                from core.prompt.builder import _build_emotion_instruction

                emotion = _build_emotion_instruction()
                if emotion:
                    store.set_section("emotion_instruction", emotion, None)
                    updated.append("emotion_instruction")
            except Exception as _exc:
                logger.debug("Skipping section resync for emotion_instruction: %s", _exc)
        details.append(f"Resynced sections: {', '.join(updated)}")
        return StepResult(changed=len(updated), skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_system_sections_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_tool_descriptions_resync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Overwrite tool descriptions and guides from defaults."""
    details: list[str] = []
    try:
        from core.paths import _get_locale
        from core.tooling.prompt_db import (
            DEFAULT_DESCRIPTIONS,
            DEFAULT_GUIDES,
            ToolPromptStore,
            get_default_description,
            get_default_guide,
        )

        tool_db_path = data_dir / "tool_prompts.sqlite3"
        if not tool_db_path.exists():
            return StepResult(changed=0, skipped=1, details=["tool_prompts.sqlite3 not found"])
        locale = _get_locale()
        store = ToolPromptStore(tool_db_path)
        changed_count = 0
        if dry_run:
            changed_count = len(DEFAULT_DESCRIPTIONS) + len(DEFAULT_GUIDES)
            details.append(f"Would overwrite {len(DEFAULT_DESCRIPTIONS)} descriptions, {len(DEFAULT_GUIDES)} guides")
            return StepResult(changed=changed_count, skipped=0, details=details)
        for name in DEFAULT_DESCRIPTIONS:
            desc = get_default_description(name, locale)
            if desc:
                store.set_description(name, desc)
                changed_count += 1
        for key in DEFAULT_GUIDES:
            guide = get_default_guide(key, locale)
            store.set_guide(key, guide)
            changed_count += 1
        details.append(f"Overwrote {len(DEFAULT_DESCRIPTIONS)} descriptions, {len(DEFAULT_GUIDES)} guides")
        return StepResult(changed=changed_count, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_tool_descriptions_resync failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


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
    r3 = step_system_sections_resync(data_dir, dry_run, verbose)
    total += r3.changed
    details.extend(r3.details)
    return StepResult(changed=total, skipped=0, details=details)


def step_stale_sections_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove stale entries from system_sections (e.g. hiring_context)."""
    details: list[str] = []
    try:
        from core.tooling.prompt_db import ToolPromptStore

        tool_db_path = data_dir / "tool_prompts.sqlite3"
        if not tool_db_path.exists():
            return StepResult(changed=0, skipped=1, details=["tool_prompts.sqlite3 not found"])
        stale_keys = ["hiring_context"]
        if dry_run:
            details.append(f"Would remove stale keys: {stale_keys}")
            return StepResult(changed=1, skipped=0, details=details)
        store = ToolPromptStore(tool_db_path)
        conn = store._connect()
        removed = 0
        try:
            for key in stale_keys:
                cur = conn.execute(
                    "DELETE FROM system_sections WHERE key = ?",
                    (key,),
                )
                removed += cur.rowcount
            conn.commit()
        finally:
            conn.close()
        details.append(f"Removed {removed} stale section(s)")
        return StepResult(changed=removed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_stale_sections_cleanup failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


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
        step_system_sections_resync,
        step_tool_descriptions_resync,
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
    """v0.6.2: Resync templates + DB for skill tool removal, activity_log scope, completion_gate.

    Covers:
    - common_knowledge/ resync (Channel D removal, skill→read_memory_file, activity_log scope)
    - prompts/ resync (2-phase consolidation, episode_extraction.md, memory_guide)
    - reference/ resync (Channel D removal, priming-channels)
    - DB system_sections resync from prompts/
    - DB tool_descriptions/guides resync (search_memory update, skill removal, completion_gate)
    - DB stale 'skill' tool description cleanup
    """
    details: list[str] = []
    total = 0

    for resync_fn in (
        step_common_knowledge_resync,
        step_common_skills_resync,
        step_prompt_resync,
        step_reference_resync,
        step_system_sections_resync,
        step_tool_descriptions_resync,
    ):
        r = resync_fn(data_dir, dry_run, verbose)
        total += r.changed
        details.extend(r.details)

    # Remove stale 'skill' tool description from DB
    try:
        from core.tooling.prompt_db import ToolPromptStore

        tool_db_path = data_dir / "tool_prompts.sqlite3"
        if tool_db_path.exists():
            stale_tool_names = ["skill"]
            if dry_run:
                details.append(f"Would remove stale tool descriptions: {stale_tool_names}")
                total += 1
            else:
                store = ToolPromptStore(tool_db_path)
                conn = store._connect()
                removed = 0
                try:
                    for name in stale_tool_names:
                        cur = conn.execute(
                            "DELETE FROM tool_descriptions WHERE name = ?",
                            (name,),
                        )
                        removed += cur.rowcount
                    conn.commit()
                finally:
                    conn.close()
                if removed:
                    details.append(f"Removed {removed} stale tool description(s): {stale_tool_names}")
                    total += removed
    except Exception as exc:
        logger.warning("v062: stale skill description cleanup failed: %s", exc)

    return StepResult(changed=total, skipped=0, details=details)


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
        MigrationStep("tool_prompt_db_init", "Init tool prompt DB", "db_sync", step_tool_prompt_db_init),
        MigrationStep("system_sections_resync", "Resync system_sections in DB", "db_sync", step_system_sections_resync),
        MigrationStep(
            "tool_descriptions_resync", "Resync tool descriptions/guides", "db_sync", step_tool_descriptions_resync
        ),
        MigrationStep("stale_sections_cleanup", "Remove stale DB sections", "db_sync", step_stale_sections_cleanup),
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
        MigrationStep("update_version", "Update migration_state.json", "version", step_update_version),
    ]
    for s in steps:
        runner.register(s)
