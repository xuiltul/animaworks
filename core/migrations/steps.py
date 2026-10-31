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
from pathlib import Path
from typing import Any

from core.anima.workspace_files import AGENTS_FILE
from core.migrations.registry import MigrationStep, StepResult
from core.platform.status_store import update_status

logger = logging.getLogger(__name__)


# ── Category 2: Per-anima file migrations ───────────────────────


def _iter_anima_dirs(data_dir: Path) -> list[Path]:
    """Return anima directories that have identity.md."""
    animas_dir = data_dir / "animas"
    if not animas_dir.exists():
        return []
    return [d for d in sorted(animas_dir.iterdir()) if d.is_dir() and (d / "identity.md").exists()]


def _persist_config(config_path: Path, payload: dict[str, Any]) -> None:
    """Persist a sparse migration payload through the locked config writer."""
    from core.config.io import update_config as update

    update(lambda _current: payload, config_path)


_TOOLS_RENAME_RE = re.compile(r"(?<![\w.])core([./])tools(?![\w])")
_TOOLS_RENAME_GLOBS = (
    "common_tools/*.py",
    "animas/*/tools/*.py",
    "common_skills/**/*.md",
    "animas/*/skills/**/*.md",
    "common_knowledge/**/*.md",
    "reference/**/*.md",
)


def step_rename_core_tools_to_integrations(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Rewrite ``core.tools`` references in runtime tools and skills to ``core.integrations``.

    Runtime files are rewritten to the canonical package before they are loaded.
    Originals are copied to ``backups/<timestamp>_tools_rename/`` before rewriting.
    """
    del verbose
    try:
        targets: list[Path] = []
        for pattern in _TOOLS_RENAME_GLOBS:
            for path in sorted(data_dir.glob(pattern)):
                if not path.is_file() or path.is_symlink():
                    continue
                if _TOOLS_RENAME_RE.search(path.read_text(encoding="utf-8", errors="replace")):
                    targets.append(path)
        if not targets:
            return StepResult(changed=0, skipped=1, details=["No core.tools references found"])
        details = [str(path.relative_to(data_dir)) for path in targets]
        if dry_run:
            return StepResult(changed=len(targets), skipped=0, details=[f"Would rewrite: {d}" for d in details])
        backup_root = data_dir / "backups" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tools_rename"
        for path, rel in zip(targets, details, strict=True):
            backup = backup_root / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            text = path.read_text(encoding="utf-8")
            path.write_text(
                _TOOLS_RENAME_RE.sub(lambda m: f"core{m.group(1)}integrations", text),
                encoding="utf-8",
            )
        return StepResult(changed=len(targets), skipped=0, details=[f"Rewrote: {d}" for d in details])
    except Exception as exc:
        logger.exception("step_rename_core_tools_to_integrations failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_trust_state_per_session(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove the legacy anima-wide trust state file."""
    details: list[str] = []
    changed = 0
    try:
        for anima_dir in _iter_anima_dirs(data_dir):
            legacy_state = anima_dir / "run" / "min_trust_seen"
            if not legacy_state.is_file():
                continue
            changed += 1
            if dry_run:
                details.append(f"{anima_dir.name}: would remove run/min_trust_seen")
                continue
            try:
                legacy_state.unlink()
                details.append(f"{anima_dir.name}: removed run/min_trust_seen")
            except OSError as exc:
                changed -= 1
                details.append(f"{anima_dir.name}: failed to remove run/min_trust_seen - {exc}")
        if not details:
            details.append("No shared run/min_trust_seen files found")
        return StepResult(changed=changed, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_trust_state_per_session failed")
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


# dev-engineer injection line as shipped before the scratch-dir guidance (2026-10-20).
_SCRATCH_INJECTION_OLD_LINES: dict[str, str] = {
    "ja": "- 他の作業と競合しないよう、作業は worktree で分離して行う。",
    "en": "- Perform work in a separate worktree to avoid conflicts with other work.",
    "ko": "- 다른 작업과 충돌하지 않도록 작업은 worktree에서 분리하여 수행한다.",
}


def _scratch_injection_new_line(locale: str) -> str | None:
    from core.paths import TEMPLATES_DIR

    template = TEMPLATES_DIR / locale / "anima_templates" / "dev-engineer" / AGENTS_FILE
    if not template.is_file():
        return None
    for line in template.read_text(encoding="utf-8").splitlines():
        if "ANIMAWORKS_SCRATCH_DIR" in line:
            return line
    return None


def step_scratch_dir_injection_20261020(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Point dev-engineer style injections at ``$ANIMAWORKS_SCRATCH_DIR`` for worktrees.

    Only the unmodified template line is replaced, so customized injections are
    left alone. Originals are copied to ``backups/<timestamp>_scratch_injection/``.
    """
    del verbose
    try:
        replacements = {
            old: new
            for locale, old in _SCRATCH_INJECTION_OLD_LINES.items()
            if (new := _scratch_injection_new_line(locale)) is not None
        }
        targets: list[tuple[Path, str]] = []
        for path in sorted((data_dir / "animas").glob("*/injection.md")):
            if not path.is_file() or path.is_symlink():
                continue
            text = path.read_text(encoding="utf-8")
            if "ANIMAWORKS_SCRATCH_DIR" in text:
                continue
            lines = text.splitlines(keepends=True)
            changed = False
            for index, line in enumerate(lines):
                stripped = line.rstrip("\r\n")
                if stripped in replacements:
                    lines[index] = replacements[stripped] + line[len(stripped) :]
                    changed = True
            if changed:
                targets.append((path, "".join(lines)))
        if not targets:
            return StepResult(changed=0, skipped=1, details=["No dev-engineer worktree line to update"])
        details = [str(path.relative_to(data_dir)) for path, _ in targets]
        if dry_run:
            return StepResult(changed=len(targets), skipped=0, details=[f"Would rewrite: {d}" for d in details])
        backup_root = data_dir / "backups" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_scratch_injection"
        for (path, new_text), rel in zip(targets, details, strict=True):
            backup = backup_root / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            path.write_text(new_text, encoding="utf-8")
        return StepResult(changed=len(targets), skipped=0, details=[f"Rewrote: {d}" for d in details])
    except Exception as exc:
        logger.exception("step_scratch_dir_injection_20261020 failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


# ── Category 3: Framework template sync ──────────────────────────


def step_template_sync(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Sync bundled shared templates into runtime data."""
    del verbose
    from core.migrations.template_sync import sync_runtime_templates
    from core.paths import _get_locale

    result = sync_runtime_templates(data_dir, locale=_get_locale(), dry_run=dry_run)
    if not dry_run and result.error is None and result.changed == 0:
        return StepResult(
            changed=1,
            skipped=result.skipped,
            details=[*result.details, "No runtime template changes required; migration marked applied"],
        )
    return result


def step_models_json_mode_b_to_a(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Map retired Mode B models.json entries to Mode A."""
    del verbose
    models_path = data_dir / "models.json"
    if not models_path.is_file():
        return StepResult(changed=0, skipped=1, details=["models.json not found"])

    try:
        models = json.loads(models_path.read_text(encoding="utf-8"))
        if not isinstance(models, dict):
            return StepResult(changed=0, skipped=1, details=["models.json is not an object"])
        patched = [
            pattern
            for pattern, entry in models.items()
            if isinstance(entry, dict) and str(entry.get("mode", "")).upper() == "B"
        ]
        if not patched:
            return StepResult(changed=0, skipped=1, details=["No Mode B entries found in models.json"])
        if dry_run:
            details = [f"Would map Mode B to A in models.json: {', '.join(patched)}"]
        else:
            for pattern in patched:
                models[pattern]["mode"] = "A"
            models_path.write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            details = [f"Mapped Mode B to A in models.json: {', '.join(patched)}"]
        return StepResult(changed=len(patched), skipped=0, details=details)
    except (OSError, ValueError, AttributeError) as exc:
        return StepResult(changed=0, skipped=0, details=[], error=f"models.json: {exc}")


# ── Category 4: Database sync ────────────────────────────────────


# ── Category 5: Version tracking ────────────────────────────────


def step_engine_timeout_config_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Drop supervisor stream-kill settings and rename runner liveness timeout."""
    del verbose
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return StepResult(changed=0, skipped=1, details=["config.json not found; skip"])

    try:
        config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(config, dict):
            return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])
        server = config.get("server")
        if server is None:
            server = {}
            config["server"] = server
        if not isinstance(server, dict):
            return StepResult(changed=0, skipped=1, details=["config.json server section is not an object"])

        details: list[str] = []
        if "busy_hang_threshold" in server:
            if "runner_liveness_timeout" not in server:
                server["runner_liveness_timeout"] = server["busy_hang_threshold"]
                details.append("Moved server.busy_hang_threshold to server.runner_liveness_timeout")
            else:
                details.append("Preserved server.runner_liveness_timeout")
            del server["busy_hang_threshold"]
        if "max_streaming_duration" in server:
            del server["max_streaming_duration"]
            details.append("Removed server.max_streaming_duration")

        if not details:
            return StepResult(changed=0, skipped=1, details=["No retired engine timeout settings found"])
        if dry_run:
            return StepResult(changed=1, skipped=0, details=[f"Would {detail.lower()}" for detail in details])

        _persist_config(config_path, config)
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_engine_timeout_config_cleanup failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_rag_vector_worker_config_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired RAG vector-worker settings without deleting stored data."""
    del verbose
    config_path = data_dir / "config.json"
    retired_keys = (
        "vector_worker_enabled",
        "vector_worker_host",
        "vector_worker_port",
        "vector_worker_startup_timeout_seconds",
        "vector_worker_request_timeout_seconds",
        "vector_worker_restart_backoff_seconds",
        "vector_worker_shutdown_timeout_seconds",
        "vector_worker_fallback_direct",
        "startup_repair_preflight_enabled",
        "startup_repair_window_minutes",
        "repair_stop_anima",
    )
    details: list[str] = []
    changed = 0

    if not config_path.is_file():
        details.append("config.json not found; skip settings cleanup")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            if not isinstance(config, dict):
                details.append("config.json root is not an object; settings cleanup skipped")
            else:
                rag = config.get("rag")
                if rag is not None and not isinstance(rag, dict):
                    details.append("config.json rag section is not an object; settings cleanup skipped")
                else:
                    removed = [key for key in retired_keys if isinstance(rag, dict) and key in rag]
                    details.extend(
                        f"{'Would remove' if dry_run else 'Removed'} rag.{key} from config.json" for key in removed
                    )
                    if removed:
                        changed = 1
                        if not dry_run:
                            for key in removed:
                                del rag[key]
                            _persist_config(config_path, config)
                    else:
                        details.append("No retired RAG vector-worker settings found")
        except Exception as exc:
            logger.exception("step_rag_vector_worker_config_cleanup failed")
            return StepResult(changed=0, skipped=0, details=details, error=str(exc))

    if any((data_dir / "logs").glob("vector-worker.log*")):
        details.append("Left logs/vector-worker.log* untouched; these logs are no longer rotated")
    if (data_dir / "vectordb").is_dir():
        details.append("data_dir/vectordb is unused; it was left in place and may be removed manually")
    return StepResult(changed=changed, skipped=0 if changed else 1, details=details)


def step_retire_chain_timeout_keys(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired model keys (max_turns/max_chains/llm_timeout) from runtime config."""
    del verbose
    retired_keys = {"max_turns", "max_chains", "llm_timeout"}
    changed_files = 0
    details: list[str] = []

    def _remove_keys(record: dict[str, Any]) -> list[str]:
        removed = sorted(retired_keys.intersection(record))
        for key in removed:
            del record[key]
        return removed

    try:
        config_path = data_dir / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            except (ValueError, OSError):
                details.append("config.json is not valid JSON; skip")
                return StepResult(changed=0, skipped=1, details=details)
            if not isinstance(config, dict):
                return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])

            config_changed = False
            defaults = config.get("anima_defaults")
            if isinstance(defaults, dict):
                removed = _remove_keys(defaults)
                if removed:
                    config_changed = True
                    action = "Would remove" if dry_run else "Removed"
                    details.append(f"{action} retired settings from anima_defaults: {', '.join(removed)}")

            animas = config.get("animas")
            if isinstance(animas, dict):
                for name, record in animas.items():
                    if isinstance(record, dict):
                        removed = _remove_keys(record)
                        if removed:
                            config_changed = True
                            action = "Would remove" if dry_run else "Removed"
                            details.append(f"{action} retired settings from animas.{name}: {', '.join(removed)}")

            if config_changed:
                changed_files += 1
                if not dry_run:
                    _persist_config(config_path, config)
        else:
            details.append("config.json not found; skip")

        for anima_dir in _iter_anima_dirs(data_dir):
            status_path = anima_dir / "status.json"
            if not status_path.is_file():
                continue
            removed: list[str] = []
            if dry_run:
                status = json.loads(status_path.read_text(encoding="utf-8") or "{}")
                if not isinstance(status, dict):
                    continue
                removed = _remove_keys(status)
            else:

                def remove_retired_keys(status: dict[str, Any], output: list[str] = removed) -> None:
                    output.extend(_remove_keys(status))

                update_status(anima_dir, remove_retired_keys)
            if not removed:
                continue
            changed_files += 1
            relative_path = status_path.relative_to(data_dir)
            action = "Would remove" if dry_run else "Removed"
            details.append(f"{action} retired settings from {relative_path}: {', '.join(removed)}")

        if not changed_files:
            details.append("No retired model keys found")
        return StepResult(changed=changed_files, skipped=0 if changed_files else 1, details=details)
    except Exception as exc:
        logger.exception("step_retire_chain_timeout_keys failed")
        return StepResult(changed=0, skipped=0, details=details, error=str(exc))


def step_memory_maintenance_config_cleanup_20260927(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove the retired housekeeping hygiene grace-period setting."""
    del verbose
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return StepResult(changed=0, skipped=1, details=["config.json not found; skip"])

    try:
        config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(config, dict):
            return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])
        housekeeping = config.get("housekeeping")
        if housekeeping is None:
            return StepResult(changed=0, skipped=1, details=["housekeeping section not found; skip"])
        if not isinstance(housekeeping, dict):
            return StepResult(changed=0, skipped=1, details=["housekeeping section is not an object"])
        retired_setting = "hygiene_grace_days"
        if retired_setting not in housekeeping:
            return StepResult(changed=0, skipped=1, details=["No retired memory maintenance settings found"])

        del housekeeping[retired_setting]
        detail = "Removed housekeeping.hygiene_grace_days"
        if dry_run:
            return StepResult(changed=1, skipped=0, details=[f"Would {detail.lower()}"])

        _persist_config(config_path, config)
        return StepResult(changed=1, skipped=0, details=[detail])
    except Exception as exc:
        logger.exception("step_memory_maintenance_config_cleanup_20260927 failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_priming_config_cleanup_20260927(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Drop the retired rag.max_graph_hops setting from config.json."""
    del verbose
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return StepResult(changed=0, skipped=1, details=["config.json not found; skip"])

    try:
        config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(config, dict):
            return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])
        rag = config.get("rag")
        if rag is None:
            return StepResult(changed=0, skipped=1, details=["No rag section; skip"])
        if not isinstance(rag, dict):
            return StepResult(changed=0, skipped=1, details=["config.json rag section is not an object"])
        if "max_graph_hops" not in rag:
            return StepResult(changed=0, skipped=1, details=["rag.max_graph_hops not found; skip"])
        if dry_run:
            return StepResult(changed=1, skipped=0, details=["Would drop rag.max_graph_hops"])

        del rag["max_graph_hops"]
        _persist_config(config_path, config)
        return StepResult(changed=1, skipped=0, details=["Dropped rag.max_graph_hops"])
    except Exception as exc:
        logger.exception("step_priming_config_cleanup_20260927 failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_remove_process_model_fields(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired process topology fields from each anima's status.json."""
    del verbose

    details: list[str] = []
    errors: list[str] = []
    changed = 0
    skipped = 0

    for anima_dir in _iter_anima_dirs(data_dir):
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            skipped += 1
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            skipped += 1
            details.append(f"{anima_dir.name}: skipped invalid status.json ({exc})")
            continue
        if not isinstance(status, dict):
            skipped += 1
            details.append(f"{anima_dir.name}: skipped status.json because it is not an object")
            continue

        relative_path = status_path.relative_to(data_dir)
        if dry_run:
            removed_fields = [key for key in ("process_model", "task_process_isolation") if key in status]
            if not removed_fields:
                skipped += 1
                continue
            old_process_model = status.get("process_model")
            details.append(f"Would remove {', '.join(removed_fields)} from {relative_path}")
            if old_process_model in ("legacy", "phase2"):
                details.append(f"{anima_dir.name}: process_model={old_process_model} removed (now phase3)")
            changed += 1
            continue

        removed_fields: list[str] = []
        old_process_model = None

        def remove_process_model_fields(
            current: dict[str, Any],
            output: list[str] = removed_fields,
        ) -> None:
            nonlocal old_process_model
            old_process_model = current.get("process_model")
            for key in ("process_model", "task_process_isolation"):
                if key in current:
                    output.append(key)
                    del current[key]

        try:
            update_status(anima_dir, remove_process_model_fields)
        except Exception as exc:
            skipped += 1
            error = f"{anima_dir.name}: failed to update status.json: {exc}"
            errors.append(error)
            details.append(error)
            logger.exception("step_remove_process_model_fields failed for %s", status_path)
            continue
        if not removed_fields:
            skipped += 1
            continue
        details.append(f"Removed {', '.join(removed_fields)} from {relative_path}")
        if old_process_model in ("legacy", "phase2"):
            details.append(f"{anima_dir.name}: process_model={old_process_model} removed (now phase3)")
        changed += 1

    return StepResult(
        changed=changed,
        skipped=skipped,
        details=details,
        error="; ".join(errors) if errors else None,
    )


def step_update_version(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """No-op step for display; version update is handled by runner."""
    return StepResult(changed=1, skipped=0, details=["migration_state.json"])


# ── Registration ──────────────────────────────────────────────────


def step_phase_b_removal_20260927(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired daily knowledge-mutation settings and state files."""
    del verbose
    details: list[str] = []
    changed = 0
    skipped = 0
    config_path = data_dir / "config.json"

    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            if not isinstance(config, dict):
                skipped += 1
                details.append("config.json root is not an object")
            else:
                consolidation = config.get("consolidation")
                if consolidation is None:
                    skipped += 1
                elif not isinstance(consolidation, dict):
                    skipped += 1
                    details.append("config.json consolidation section is not an object")
                else:
                    retired_settings = (
                        "knowledge_mutation_enabled",
                        "ipc_timeout_per_carryover_item_seconds",
                    )
                    removed_settings = [key for key in retired_settings if key in consolidation]
                    if removed_settings:
                        for key in removed_settings:
                            del consolidation[key]
                        changed += 1
                        if dry_run:
                            details.append(f"Would remove {len(removed_settings)} retired consolidation setting(s)")
                        else:
                            _persist_config(config_path, config)
                            details.append(f"Removed {len(removed_settings)} retired consolidation setting(s)")
                    else:
                        skipped += 1
        except Exception as exc:
            logger.exception("step_phase_b_removal_20260927 failed while updating config.json")
            return StepResult(changed=0, skipped=skipped, details=details, error=str(exc))
    else:
        skipped += 1
        details.append("config.json not found; skip settings cleanup")

    animas_dir = data_dir / "animas"
    state_files = sorted(
        {
            *animas_dir.glob("*/state/consolidation_phase_b_carryover.json"),
            *animas_dir.glob("*/state/consolidation_phase_b_carryover_*.json"),
        }
    )
    state_files = [path for path in state_files if path.is_file()]
    if state_files:
        changed += len(state_files)
        if dry_run:
            details.append(f"Would remove {len(state_files)} carryover state file(s)")
        else:
            for path in state_files:
                path.unlink()
            details.append(f"Removed {len(state_files)} carryover state file(s)")
    else:
        skipped += 1
        details.append("No carryover state files found")

    return StepResult(changed=changed, skipped=skipped, details=details)


def step_retired_mode_to_a(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Map retired Mode B values to canonical Mode A in config and status files."""
    del verbose
    details: list[str] = []
    changed_files = 0

    def _map_record(record: dict[str, Any]) -> bool:
        changed = False
        for key in ("execution_mode", "resolved_mode"):
            value = record.get(key)
            if isinstance(value, str) and value.strip().lower() in {"b", "basic"}:
                record[key] = "A"
                changed = True

        fallback_models = record.get("fallback_models")
        if isinstance(fallback_models, list):
            for index, fallback in enumerate(fallback_models):
                if not isinstance(fallback, str):
                    continue
                leading = len(fallback) - len(fallback.lstrip())
                value = fallback[leading:]
                if value[:2].lower() == "b:":
                    fallback_models[index] = f"{fallback[:leading]}a:{value[2:]}"
                    changed = True
        return changed

    try:
        config_path = data_dir / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            if isinstance(config, dict):
                config_changed = False
                for key in ("anima_defaults", "animas"):
                    section = config.get(key)
                    if isinstance(section, dict):
                        records = section.values() if key == "animas" else (section,)
                        for record in records:
                            if isinstance(record, dict):
                                config_changed = _map_record(record) or config_changed
                model_modes = config.get("model_modes")
                if isinstance(model_modes, dict):
                    for pattern, mode in model_modes.items():
                        if isinstance(mode, str) and mode.strip().lower() in {"b", "basic"}:
                            model_modes[pattern] = "A"
                            config_changed = True
                if config_changed:
                    if dry_run:
                        details.append("Would map retired Mode B values to A in config.json")
                    else:
                        _persist_config(config_path, config)
                        details.append("Mapped retired Mode B values to A in config.json")
                    changed_files += 1

        for anima_dir in _iter_anima_dirs(data_dir):
            status_path = anima_dir / "status.json"
            if not status_path.is_file():
                continue
            relative_path = status_path.relative_to(data_dir)
            if dry_run:
                status = json.loads(status_path.read_text(encoding="utf-8") or "{}")
                if not isinstance(status, dict) or not _map_record(status):
                    continue
                details.append(f"Would map retired Mode B values to A in {relative_path}")
            else:
                status_changed = False

                def map_status_record(status: dict[str, Any]) -> None:
                    nonlocal status_changed
                    status_changed = _map_record(status)

                update_status(anima_dir, map_status_record)
                if not status_changed:
                    continue
                details.append(f"Mapped retired Mode B values to A in {relative_path}")
            changed_files += 1

        if not details:
            details.append("No retired Mode B values found")
        return StepResult(changed=changed_files, skipped=0 if changed_files else 1, details=details)
    except Exception as exc:
        logger.exception("step_retired_mode_to_a failed")
        return StepResult(changed=changed_files, skipped=0, details=details, error=str(exc))


def step_taskboard_metadata_retire(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Retire TaskBoard presentation metadata; tasks are the only board source.

    Backs up the shared DB, transfers ``source_ref`` into canonical task meta,
    reports active-but-archived / orphan WAITING cards, then drops the metadata
    and event tables, and removes the retired housekeeping settings.
    """
    del verbose
    import os
    import sqlite3

    from core.tasks.board.tasks import connect_task_store_for_migration

    db_path = data_dir / "shared" / "taskboard.sqlite3"
    details: list[str] = []
    if not db_path.is_file():
        return StepResult(changed=0, skipped=1, details=["shared/taskboard.sqlite3 not found; skip"])

    try:
        conn = connect_task_store_for_migration(db_path)
        try:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='taskboard_metadata'"
            ).fetchone():
                return StepResult(changed=0, skipped=1, details=["taskboard_metadata table not present; skip"])

            rows = conn.execute(
                "SELECT anima_name, task_id, visibility, column, source_ref FROM taskboard_metadata"
            ).fetchall()
            inconsistent: list[str] = []
            orphans = 0
            source_ref_candidates: list[tuple[str, str, str]] = []
            for anima, tid, visibility, column, source_ref in rows:
                canon = conn.execute(
                    "SELECT entry_json FROM tasks WHERE anima=? AND task_id=?", (anima, tid)
                ).fetchone()
                if canon is None:
                    if column == "waiting":
                        alias_exists = conn.execute(
                            "SELECT 1 FROM task_aliases WHERE viewer=? AND alias=?", (anima, tid)
                        ).fetchone()
                        if not alias_exists:
                            orphans += 1
                    continue
                status = json.loads(canon[0]).get("status")
                if visibility in ("expired", "archived", "tombstoned") and status in ("pending", "in_progress"):
                    inconsistent.append(f"{anima}/{tid}")
                existing_ref = json.loads(canon[0]).get("meta", {}).get("source_ref")
                if source_ref and not source_ref.startswith("task_queue:") and not existing_ref:
                    source_ref_candidates.append((anima, tid, source_ref))

            details.append(f"visible-but-archived cards: {len(inconsistent)}")
            details.extend(f"visible-but-archived: {ref}" for ref in inconsistent)
            details.append(f"orphan WAITING cards: {orphans}")
            details.append(f"source_ref transfers: {len(source_ref_candidates)}")

            changed = bool(inconsistent or orphans or source_ref_candidates)
            if dry_run:
                return StepResult(changed=1 if changed else 0, skipped=0, details=details)

            # Pre-backup (no overwrite)
            backups_dir = data_dir / "shared" / "backups"
            backups_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            backup_path = backups_dir / f"taskboard-pre-metadata-retire-{stamp}.sqlite3"
            fd = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            dest = sqlite3.connect(backup_path)
            try:
                conn.backup(dest)
            finally:
                dest.close()
            details.append(f"backup: {backup_path.name}")

            for anima, tid, source_ref in source_ref_candidates:
                conn.execute(
                    "UPDATE tasks SET entry_json=json_set(entry_json,'$.meta.source_ref',?) "
                    "WHERE anima=? AND task_id=?",
                    (source_ref, anima, tid),
                )

            conn.execute("DROP TABLE taskboard_metadata")
            conn.execute("DROP TABLE taskboard_events")
            conn.commit()

            config_path = data_dir / "config.json"
            if config_path.is_file():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
                    hk = config.get("housekeeping") if isinstance(config, dict) else None
                    removed = False
                    if isinstance(hk, dict):
                        for key in (
                            "taskboard_suppressed_retention_days",
                            "taskboard_orphan_metadata_stale_hours",
                        ):
                            if key in hk:
                                del hk[key]
                                removed = True
                    if removed:
                        _persist_config(config_path, config)
                        details.append("removed retired housekeeping settings")
                except Exception:
                    logger.warning("Failed to clean retired housekeeping settings", exc_info=True)

            return StepResult(changed=1, skipped=0, details=details)
        finally:
            conn.close()
    except Exception as exc:
        logger.exception("step_taskboard_metadata_retire failed")
        return StepResult(changed=0, skipped=0, details=details, error=str(exc))


def step_neo4j_config_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired Neo4j settings and preserve configured fact edge types."""
    del verbose
    details: list[str] = []
    errors: list[str] = []
    changed = 0
    skipped = 0

    config_path = data_dir / "config.json"
    if not config_path.is_file():
        skipped += 1
        details.append("config.json not found; skip")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
            if not isinstance(config, dict):
                skipped += 1
                details.append("config.json root is not an object")
            else:
                memory = config.get("memory")
                if not isinstance(memory, dict):
                    skipped += 1
                    details.append("config.json memory section is not an object; skip")
                else:
                    config_changed = False
                    old_edge_types = memory.get("neo4j_edge_types")
                    if old_edge_types and "fact_edge_types" not in memory:
                        memory["fact_edge_types"] = old_edge_types
                        config_changed = True
                        action = "Would move" if dry_run else "Moved"
                        details.append(f"{action} memory.neo4j_edge_types to memory.fact_edge_types")
                    for key in ("backend", "neo4j", "neo4j_realtime_ingest", "neo4j_edge_types"):
                        if key in memory:
                            del memory[key]
                            config_changed = True
                            action = "Would remove" if dry_run else "Removed"
                            details.append(f"{action} memory.{key}")
                    if config_changed:
                        changed += 1
                        if not dry_run:
                            _persist_config(config_path, config)
                    else:
                        skipped += 1
                        details.append("No retired memory settings found in config.json")
        except Exception as exc:
            errors.append(f"config.json: {exc}")
            details.append(f"config.json: failed to update: {exc}")
            logger.exception("step_neo4j_config_cleanup failed for %s", config_path)

    animas_dir = data_dir / "animas"
    anima_dirs = sorted(path for path in animas_dir.iterdir() if path.is_dir()) if animas_dir.is_dir() else []
    for anima_dir in anima_dirs:
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status_messages: list[str] = []
            status_changed = False

            def cleanup_status(
                status: dict[str, Any],
                messages: list[str] = status_messages,
                anima_name: str = anima_dir.name,
            ) -> None:
                nonlocal status_changed
                legacy_backend = status.get("memory_backend")
                old_edge_types = status.get("neo4j_edge_types")
                if old_edge_types and "fact_edge_types" not in status:
                    status["fact_edge_types"] = old_edge_types
                    status_changed = True
                    action = "would move" if dry_run else "moved"
                    messages.append(f"{anima_name}: {action} neo4j_edge_types to fact_edge_types")
                if "memory_backend" in status:
                    del status["memory_backend"]
                    status_changed = True
                    action = "would remove" if dry_run else "removed"
                    messages.append(f"{anima_name}: {action} memory_backend")
                if "neo4j_edge_types" in status:
                    del status["neo4j_edge_types"]
                    status_changed = True
                    action = "would remove" if dry_run else "removed"
                    messages.append(f"{anima_name}: {action} neo4j_edge_types")
                if legacy_backend == "neo4j":
                    messages.append(f"{anima_name}: Neo4j data is no longer used (legacy RAG continues)")

            if dry_run:
                status = json.loads(status_path.read_text(encoding="utf-8") or "{}")
                if not isinstance(status, dict):
                    skipped += 1
                    details.append(f"{anima_dir.name}: status.json root is not an object")
                    continue
                cleanup_status(status)
            else:
                update_status(anima_dir, cleanup_status)

            details.extend(status_messages)
            if status_changed:
                changed += 1
            else:
                skipped += 1
        except Exception as exc:
            error = f"{anima_dir.name}: failed to update status.json: {exc}"
            errors.append(error)
            details.append(error)
            logger.exception("step_neo4j_config_cleanup failed for %s", status_path)

    return StepResult(changed=changed, skipped=skipped, details=details, error="; ".join(errors) if errors else None)


def step_usage_governor_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired Usage Governor config and archive its runtime state."""
    del verbose
    details: list[str] = []
    config_path = data_dir / "config.json"
    config: dict[str, Any] | None = None
    config_changed = False
    skipped = 0

    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return StepResult(
                changed=0, skipped=0, details=["config.json is not valid JSON; skip cleanup"], error=str(exc)
            )
        if not isinstance(loaded, dict):
            return StepResult(
                changed=0, skipped=0, details=["config.json root is not an object"], error="Invalid config root"
            )
        config = loaded
        server = config.get("server")
        if server is not None and not isinstance(server, dict):
            return StepResult(
                changed=0,
                skipped=0,
                details=["config.json server section is not an object"],
                error="Invalid server section",
            )
        if isinstance(server, dict) and "usage_governor" in server:
            del server["usage_governor"]
            config_changed = True
            details.append(
                "Would remove server.usage_governor from config.json"
                if dry_run
                else "Removed server.usage_governor from config.json"
            )
        else:
            skipped += 1
            details.append("No retired Usage Governor setting found")
    else:
        skipped += 1
        details.append("config.json not found; skip settings cleanup")

    retired_files = ("usage_governor_state.json", "usage_policy.json")
    archive_dir = data_dir / "archive" / "retired"
    sources = [data_dir / name for name in retired_files if (data_dir / name).exists()]
    if archive_dir.exists() and not archive_dir.is_dir():
        return StepResult(
            changed=0,
            skipped=skipped,
            details=details + ["archive/retired exists and is not a directory"],
            error="Archive destination is not a directory",
        )
    for source in sources:
        destination = archive_dir / source.name
        if not source.is_file():
            return StepResult(
                changed=0,
                skipped=skipped,
                details=details + [f"{source.name} is not a regular file"],
                error=f"Cannot archive non-file {source}",
            )
        if destination.exists():
            return StepResult(
                changed=0,
                skipped=skipped,
                details=details + [f"Archive target already exists: {destination}"],
                error=f"Archive target already exists: {destination}",
            )
        details.append(
            f"Would archive {source.name} to archive/retired/"
            if dry_run
            else f"Archived {source.name} to archive/retired/"
        )

    missing_files = [name for name in retired_files if not (data_dir / name).exists()]
    skipped += len(missing_files)
    details.extend(f"{name} not found; skip" for name in missing_files)
    if dry_run:
        return StepResult(changed=int(config_changed) + len(sources), skipped=skipped, details=details)

    changed = 0
    try:
        if sources:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for source in sources:
                shutil.move(str(source), str(archive_dir / source.name))
                changed += 1
        if config_changed and config is not None:
            _persist_config(config_path, config)
            changed += 1
        return StepResult(changed=changed, skipped=skipped, details=details)
    except Exception as exc:
        logger.exception("step_usage_governor_cleanup failed")
        return StepResult(changed=changed, skipped=skipped, details=details, error=str(exc))


_MEMORY_DEAD_STALE_PROMPTS = ("memory/classification.md",)


def step_memory_dead_prompt_cleanup_20260927(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove runtime prompts retired with classify_and_distill."""
    details: list[str] = []
    changed = 0
    skipped = 0
    for name in _MEMORY_DEAD_STALE_PROMPTS:
        stale = data_dir / "prompts" / name
        if not stale.is_file():
            skipped += 1
            continue
        if dry_run:
            details.append(f"Would remove stale prompts/{name}")
        else:
            stale.unlink()
            details.append(f"Removed stale prompts/{name}")
        changed += 1
    return StepResult(changed=changed, skipped=skipped, details=details)


# ── Category 4: Database sync ────────────────────────────────────


def step_retired_config_keys_cleanup(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Remove retired config.json keys that no longer have runtime behavior."""
    del verbose
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return StepResult(changed=0, skipped=1, details=["config.json not found; skip"])

    try:
        config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(config, dict):
            return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])

        retired_keys = {
            "heartbeat": ("orphan_grace_multiplier", "orphan_grace_min_seconds"),
            "background_task": ("max_parallel_llm_tasks",),
            "rag": ("enable_file_watcher",),
            "interaction": ("ttl_days",),
            "system": ("gateway", "worker"),
        }
        removed: list[str] = []
        for section_name, keys in retired_keys.items():
            section = config.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in keys:
                if key in section:
                    del section[key]
                    removed.append(f"{section_name}.{key}")

        if not removed:
            return StepResult(changed=0, skipped=1, details=["No retired config keys found"])

        action = "Would remove" if dry_run else "Removed"
        details = [f"{action} retired config keys: {', '.join(removed)}"]
        if not dry_run:
            _persist_config(config_path, config)
        return StepResult(changed=1, skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_retired_config_keys_cleanup failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def step_memory_config_dead_keys_20260927(data_dir: Path, dry_run: bool, verbose: bool) -> StepResult:
    """Drop retired consolidation and RAG configuration keys."""
    del verbose
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        return StepResult(changed=0, skipped=1, details=["config.json not found; skip"])

    try:
        config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(config, dict):
            return StepResult(changed=0, skipped=1, details=["config.json root is not an object"])

        retired_keys = (
            ("consolidation", "duplicate_threshold"),
            ("rag", "enable_file_watcher"),
        )
        found: list[tuple[str, str]] = []
        for section_name, key in retired_keys:
            section = config.get(section_name)
            if isinstance(section, dict) and key in section:
                found.append((section_name, key))

        if not found:
            return StepResult(changed=0, skipped=1, details=["No retired memory config keys found"])
        if dry_run:
            details = [f"Would remove {section}.{key}" for section, key in found]
            return StepResult(changed=len(found), skipped=0, details=details)

        for section_name, key in found:
            del config[section_name][key]
        details = [f"Removed {section}.{key}" for section, key in found]
        _persist_config(config_path, config)
        return StepResult(changed=len(found), skipped=0, details=details)
    except Exception as exc:
        logger.exception("step_memory_config_dead_keys_20260927 failed")
        return StepResult(changed=0, skipped=0, details=[], error=str(exc))


def register_all_steps(runner: Any) -> None:
    """Register all migration steps in execution order."""
    from core.migrations.template_sync import template_fingerprint
    from core.paths import _get_locale

    try:
        template_sync_id = f"template_sync_{template_fingerprint(_get_locale())}"
    except Exception:
        logger.exception("Could not fingerprint bundled templates; using fallback migration ID")
        template_sync_id = "template_sync_unknown"

    steps = [
        MigrationStep(
            "permissions_migration", "permissions.md → permissions.json", "per_anima", step_permissions_migration
        ),
        MigrationStep(
            "trust_state_per_session",
            "Remove shared run/min_trust_seen (now per tool session)",
            "per_anima",
            step_trust_state_per_session,
        ),
        MigrationStep(
            template_sync_id,
            "Sync shared templates and selected runtime prompts from bundled templates",
            "template_sync",
            step_template_sync,
        ),
        MigrationStep(
            "v0140_harness_diet_resync",
            "Map retired Mode B models.json entries to Mode A",
            "template_sync",
            step_models_json_mode_b_to_a,
        ),
        MigrationStep(
            "rename_core_tools_to_integrations",
            "Rewrite core.tools references in runtime tools/skills to core.integrations",
            "structural",
            step_rename_core_tools_to_integrations,
        ),
        MigrationStep(
            "engine_timeout_config_cleanup",
            "Remove retired engine timeout settings and rename runner liveness timeout",
            "structural",
            step_engine_timeout_config_cleanup,
        ),
        MigrationStep(
            "memory_maintenance_config_cleanup_20260927",
            "Drop housekeeping.hygiene_grace_days and hygiene first_seen state",
            "structural",
            step_memory_maintenance_config_cleanup_20260927,
        ),
        MigrationStep(
            "priming_config_cleanup_20260927",
            "Drop rag.max_graph_hops",
            "structural",
            step_priming_config_cleanup_20260927,
        ),
        MigrationStep(
            "phase_b_removal_20260927",
            "Drop Phase B knowledge mutation settings and carryover state",
            "structural",
            step_phase_b_removal_20260927,
        ),
        MigrationStep(
            "scratch_dir_injection_20261020",
            "Point dev-engineer injection worktree guidance at ANIMAWORKS_SCRATCH_DIR",
            "per_anima",
            step_scratch_dir_injection_20261020,
        ),
        MigrationStep(
            "retired_mode_to_a",
            "Map retired execution mode B to A in config.json and status.json",
            "structural",
            step_retired_mode_to_a,
        ),
        MigrationStep(
            "remove_process_model_fields",
            "Remove retired process_model/task_process_isolation from status.json",
            "per_anima",
            step_remove_process_model_fields,
        ),
        MigrationStep(
            "retire_chain_timeout_keys",
            "Remove retired model keys (max_turns/max_chains/llm_timeout)",
            "structural",
            step_retire_chain_timeout_keys,
        ),
        MigrationStep(
            "rag_vector_worker_config_cleanup",
            "Remove retired RAG vector-worker config without deleting data",
            "structural",
            step_rag_vector_worker_config_cleanup,
        ),
        MigrationStep(
            "taskboard_metadata_retire",
            "Retire TaskBoard presentation metadata (tasks are the only board source)",
            "db_sync",
            step_taskboard_metadata_retire,
        ),
        MigrationStep(
            "neo4j_config_cleanup",
            "Remove retired Neo4j configuration keys",
            "structural",
            step_neo4j_config_cleanup,
        ),
        MigrationStep(
            "usage_governor_cleanup_20260927",
            "Remove Usage Governor state and config",
            "structural",
            step_usage_governor_cleanup,
        ),
        MigrationStep(
            "memory_dead_prompt_cleanup_20260927",
            "Remove runtime prompts retired with classify_and_distill",
            "template_sync",
            step_memory_dead_prompt_cleanup_20260927,
        ),
        MigrationStep(
            "retired_config_keys_cleanup_20260927",
            "Remove retired config.json keys",
            "structural",
            step_retired_config_keys_cleanup,
        ),
        MigrationStep(
            "memory_config_dead_keys_20260927",
            "Drop consolidation.duplicate_threshold and rag.enable_file_watcher",
            "structural",
            step_memory_config_dead_keys_20260927,
        ),
        MigrationStep("update_version", "Update migration_state.json", "version", step_update_version),
    ]
    for s in steps:
        runner.register(s)
