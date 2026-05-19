from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap lifecycle state and repair helpers."""

import json
import shutil
from pathlib import Path
from typing import Any

from core.time_utils import now_local

STATE_PENDING_USER_INPUT = "pending_user_input"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_NEEDS_REPAIR = "needs_repair"

ALLOWED_STATES = {
    STATE_PENDING_USER_INPUT,
    STATE_RUNNING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_NEEDS_REPAIR,
}

UNDEFINED_MARKERS = ("未定義", "undefined")
PRESERVED_STATUS_KEYS = {
    "model",
    "credential",
    "execution_mode",
    "background_model",
    "background_credential",
    "background_execution_mode",
}


def bootstrap_state_path(anima_dir: Path) -> Path:
    return anima_dir / "state" / "bootstrap_state.json"


def _now_iso() -> str:
    return now_local().isoformat()


def _bootstrap_artifacts(anima_dir: Path) -> list[Path]:
    return [
        anima_dir / "bootstrap.md.auto_resolved",
        anima_dir / "bootstrap.md.failed",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_bootstrap_state(anima_dir: Path) -> dict[str, Any]:
    return _read_json(bootstrap_state_path(anima_dir))


def write_bootstrap_state(anima_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    state_dir = anima_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["version"] = 1
    payload["updated_at"] = _now_iso()
    if payload.get("state") not in ALLOWED_STATES:
        payload["state"] = STATE_NEEDS_REPAIR
        payload["reason"] = "invalid_bootstrap_state"

    path = bootstrap_state_path(anima_dir)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return payload


def _base_state(
    state: str,
    *,
    mode: str = "",
    reason: str = "",
    validation_errors: list[str] | None = None,
    last_error: str = "",
    retry_count: int | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "state": state,
        "mode": mode,
        "reason": reason,
        "started_at": started_at,
        "completed_at": completed_at,
        "updated_at": _now_iso(),
        "last_error": last_error,
        "validation_errors": validation_errors or [],
        "retry_count": int(retry_count or 0),
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def file_is_defined(path: Path) -> bool:
    if not path.exists():
        return False
    content = _read_text(path).strip()
    if not content:
        return False
    lower = content.lower()
    return not any(marker in lower for marker in UNDEFINED_MARKERS)


def _identity_is_undefined(anima_dir: Path) -> bool:
    return not file_is_defined(anima_dir / "identity.md")


def _injection_is_undefined(anima_dir: Path) -> bool:
    return not file_is_defined(anima_dir / "injection.md")


def _pending_task_files(anima_dir: Path) -> list[Path]:
    candidates = [
        anima_dir / "state" / "background_tasks" / "pending",
        anima_dir / "state" / "background_tasks" / "pending" / "processing",
        anima_dir / "state" / "pending",
        anima_dir / "state" / "pending" / "processing",
        anima_dir / "state" / "pending" / "deferred",
    ]
    found: list[Path] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            data = _read_json(path)
            task_id = str(data.get("task_id") or data.get("id") or path.stem)
            if task_id.startswith("bootstrap-"):
                found.append(path)
    return found


def _apply_flags(anima_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    state = str(payload.get("state") or STATE_NEEDS_REPAIR)
    has_bootstrap = (anima_dir / "bootstrap.md").exists()
    has_character_sheet = (anima_dir / "character_sheet.md").exists()
    has_blocking_artifact = any(path.exists() for path in _bootstrap_artifacts(anima_dir))

    payload["needs_user_input"] = (
        state == STATE_PENDING_USER_INPUT and has_bootstrap and not has_character_sheet and not has_blocking_artifact
    )
    payload["needs_repair"] = state == STATE_NEEDS_REPAIR
    payload["needs_background_bootstrap"] = (
        has_bootstrap
        and has_character_sheet
        and state not in {STATE_RUNNING, STATE_FAILED, STATE_NEEDS_REPAIR}
        and not has_blocking_artifact
    )
    payload["needs_bootstrap"] = state != STATE_COMPLETED
    return payload


def get_bootstrap_status(anima_dir: Path) -> dict[str, Any]:
    """Return derived bootstrap lifecycle status without writing files."""
    persisted = read_bootstrap_state(anima_dir)
    persisted_state = str(persisted.get("state") or "")
    retry_count = int(persisted.get("retry_count") or 0)

    if not anima_dir.exists():
        return _apply_flags(
            anima_dir,
            _base_state(
                STATE_COMPLETED,
                reason="anima_dir_missing",
                retry_count=retry_count,
            ),
        )

    artifact_names = [path.name for path in _bootstrap_artifacts(anima_dir) if path.exists()]
    if artifact_names:
        return _apply_flags(
            anima_dir,
            _base_state(
                STATE_NEEDS_REPAIR,
                mode=str(persisted.get("mode") or ""),
                reason="bootstrap_artifact_requires_repair",
                validation_errors=[f"unexpected_bootstrap_artifact:{name}" for name in artifact_names],
                retry_count=retry_count,
            ),
        )

    has_bootstrap = (anima_dir / "bootstrap.md").exists()
    has_character_sheet = (anima_dir / "character_sheet.md").exists()

    if persisted_state in {STATE_RUNNING, STATE_FAILED, STATE_NEEDS_REPAIR}:
        payload = _base_state(
            persisted_state,
            mode=str(persisted.get("mode") or ""),
            reason=str(persisted.get("reason") or ""),
            validation_errors=list(persisted.get("validation_errors") or []),
            last_error=str(persisted.get("last_error") or ""),
            retry_count=retry_count,
            started_at=persisted.get("started_at"),
            completed_at=persisted.get("completed_at"),
        )
        return _apply_flags(anima_dir, payload)

    if has_bootstrap:
        if has_character_sheet:
            payload = _base_state(
                STATE_PENDING_USER_INPUT,
                mode="character_sheet",
                reason="character_sheet_background_ready",
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        if _identity_is_undefined(anima_dir):
            payload = _base_state(
                STATE_PENDING_USER_INPUT,
                mode="interactive",
                reason="identity_undefined_without_character_sheet",
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        if _injection_is_undefined(anima_dir):
            payload = _base_state(
                STATE_NEEDS_REPAIR,
                reason="injection_undefined_with_bootstrap",
                validation_errors=["injection_undefined"],
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        payload = _base_state(
            STATE_NEEDS_REPAIR,
            reason="bootstrap_file_left_after_definition",
            validation_errors=["bootstrap_file_left_after_definition"],
            retry_count=retry_count,
        )
        return _apply_flags(anima_dir, payload)

    if persisted_state == STATE_COMPLETED:
        payload = _base_state(
            STATE_COMPLETED,
            mode=str(persisted.get("mode") or ""),
            reason=str(persisted.get("reason") or "completed"),
            retry_count=retry_count,
            completed_at=persisted.get("completed_at"),
        )
        return _apply_flags(anima_dir, payload)

    if persisted_state == STATE_PENDING_USER_INPUT:
        if file_is_defined(anima_dir / "identity.md") and file_is_defined(anima_dir / "injection.md"):
            payload = _base_state(
                STATE_COMPLETED,
                mode=str(persisted.get("mode") or "interactive"),
                reason="interactive_bootstrap_completed",
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        if file_is_defined(anima_dir / "identity.md") and not file_is_defined(anima_dir / "injection.md"):
            payload = _base_state(
                STATE_NEEDS_REPAIR,
                mode=str(persisted.get("mode") or "interactive"),
                reason="interactive_bootstrap_incomplete",
                validation_errors=["injection_undefined"],
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        if not has_bootstrap:
            errors = ["bootstrap_missing"]
            if not file_is_defined(anima_dir / "identity.md"):
                errors.append("identity_undefined")
            if not file_is_defined(anima_dir / "injection.md"):
                errors.append("injection_undefined")
            payload = _base_state(
                STATE_NEEDS_REPAIR,
                mode=str(persisted.get("mode") or "interactive"),
                reason="interactive_bootstrap_prompt_missing",
                validation_errors=errors,
                retry_count=retry_count,
            )
            return _apply_flags(anima_dir, payload)
        payload = _base_state(
            STATE_PENDING_USER_INPUT,
            mode=str(persisted.get("mode") or "interactive"),
            reason=str(persisted.get("reason") or "pending_user_input"),
            retry_count=retry_count,
        )
        return _apply_flags(anima_dir, payload)

    payload = _base_state(
        STATE_COMPLETED,
        reason="no_bootstrap_artifacts",
        retry_count=retry_count,
    )
    return _apply_flags(anima_dir, payload)


def initialize_bootstrap_state(anima_dir: Path) -> dict[str, Any]:
    """Persist the current derived bootstrap state for a newly created Anima."""
    status = get_bootstrap_status(anima_dir)
    payload = {k: v for k, v in status.items() if not k.startswith("needs_")}
    return write_bootstrap_state(anima_dir, payload)


def mark_bootstrap_running(anima_dir: Path, *, mode: str = "background") -> dict[str, Any]:
    persisted = read_bootstrap_state(anima_dir)
    payload = _base_state(
        STATE_RUNNING,
        mode=mode,
        reason="background_bootstrap_running",
        retry_count=int(persisted.get("retry_count") or 0),
        started_at=_now_iso(),
    )
    return write_bootstrap_state(anima_dir, payload)


def mark_bootstrap_failed(anima_dir: Path, error: str, *, reason: str = "bootstrap_failed") -> dict[str, Any]:
    persisted = read_bootstrap_state(anima_dir)
    payload = _base_state(
        STATE_FAILED,
        mode=str(persisted.get("mode") or "background"),
        reason=reason,
        last_error=error,
        retry_count=int(persisted.get("retry_count") or 0) + 1,
        started_at=persisted.get("started_at"),
    )
    return write_bootstrap_state(anima_dir, payload)


def validate_bootstrap(anima_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not file_is_defined(anima_dir / "identity.md"):
        errors.append("identity_undefined")
    if not file_is_defined(anima_dir / "injection.md"):
        errors.append("injection_undefined")
    for artifact in _bootstrap_artifacts(anima_dir):
        if artifact.exists():
            errors.append(f"unexpected_bootstrap_artifact:{artifact.name}")
    if (anima_dir / "character_sheet.md").exists():
        errors.append("character_sheet_unprocessed")
    for path in _pending_task_files(anima_dir):
        errors.append(f"bootstrap_pending_task:{path.relative_to(anima_dir)}")

    state = STATE_COMPLETED if not errors else STATE_NEEDS_REPAIR
    payload = _base_state(
        state,
        reason="validation_passed" if not errors else "validation_failed",
        validation_errors=errors,
        retry_count=int(read_bootstrap_state(anima_dir).get("retry_count") or 0),
    )
    return _apply_flags(anima_dir, payload)


def _archive_bootstrap_file(anima_dir: Path, bootstrap_file: Path) -> Path:
    archive_dir = anima_dir / "state" / "bootstrap_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"bootstrap-{timestamp}.md"
    counter = 1
    while archive_path.exists():
        archive_path = archive_dir / f"bootstrap-{timestamp}-{counter}.md"
        counter += 1
    shutil.move(str(bootstrap_file), str(archive_path))
    return archive_path


def _archive_named_file(anima_dir: Path, source: Path, filename: str) -> Path:
    archive_dir = anima_dir / "state" / "bootstrap_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / filename
    counter = 1
    while archive_path.exists():
        stem = archive_path.stem
        suffix = archive_path.suffix
        archive_path = archive_dir / f"{stem}-{counter}{suffix}"
        counter += 1
    shutil.move(str(source), str(archive_path))
    return archive_path


def finalize_bootstrap_run(anima_dir: Path) -> dict[str, Any]:
    """Validate bootstrap output and persist completed or needs_repair state."""
    status = validate_bootstrap(anima_dir)
    if status["state"] != STATE_COMPLETED:
        payload = {k: v for k, v in status.items() if not k.startswith("needs_")}
        payload["mode"] = read_bootstrap_state(anima_dir).get("mode") or "background"
        return write_bootstrap_state(anima_dir, payload)

    bootstrap_file = anima_dir / "bootstrap.md"
    if bootstrap_file.exists():
        archive_path = _archive_bootstrap_file(anima_dir, bootstrap_file)
        status["reason"] = "completed_bootstrap_archived"
        status["archived_bootstrap"] = str(archive_path)

    payload = {k: v for k, v in status.items() if not k.startswith("needs_")}
    payload["completed_at"] = _now_iso()
    return write_bootstrap_state(anima_dir, payload)


def _remove_contents(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def repair_bootstrap_retry(anima_dir: Path, *, retry_counts_file: Path | None = None) -> dict[str, Any]:
    """Prepare a partially initialized Anima for another bootstrap attempt."""
    restored = False
    bootstrap_file = anima_dir / "bootstrap.md"
    archive_dir = anima_dir / "state" / "bootstrap_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for artifact in _bootstrap_artifacts(anima_dir):
        if not artifact.exists():
            continue
        if not bootstrap_file.exists() and not restored:
            shutil.move(str(artifact), str(bootstrap_file))
            restored = True
        else:
            timestamp = now_local().strftime("%Y%m%d_%H%M%S")
            shutil.move(str(artifact), str(archive_dir / f"{artifact.name}.{timestamp}"))

    _remove_contents(anima_dir / "shortterm")
    (anima_dir / "shortterm").mkdir(parents=True, exist_ok=True)

    for session_file in (anima_dir / "state").glob("current_session_*.json"):
        session_file.unlink(missing_ok=True)

    _remove_bootstrap_pending_tasks(anima_dir)

    if retry_counts_file and retry_counts_file.exists():
        data = _read_json(retry_counts_file)
        data.pop(anima_dir.name, None)
        retry_counts_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = _base_state(
        STATE_PENDING_USER_INPUT,
        mode="character_sheet" if (anima_dir / "character_sheet.md").exists() else "interactive",
        reason="retry_prepared",
        retry_count=0,
    )
    write_bootstrap_state(anima_dir, payload)
    return get_bootstrap_status(anima_dir)


def _remove_bootstrap_pending_tasks(anima_dir: Path) -> None:
    for pending_dir in [
        anima_dir / "state" / "pending",
        anima_dir / "state" / "background_tasks" / "pending",
    ]:
        if not pending_dir.exists():
            continue
        for path in pending_dir.rglob("*.json"):
            data = _read_json(path)
            task_id = str(data.get("task_id") or data.get("id") or path.stem)
            if task_id.startswith("bootstrap-"):
                path.unlink(missing_ok=True)


def repair_bootstrap_complete(anima_dir: Path, *, retry_counts_file: Path | None = None) -> dict[str, Any]:
    """Mark a fully defined Anima as bootstrapped and archive stale bootstrap inputs."""
    if not anima_dir.exists():
        raise FileNotFoundError(f"Anima not found: {anima_dir.name}")

    errors: list[str] = []
    if not file_is_defined(anima_dir / "identity.md"):
        errors.append("identity_undefined")
    if not file_is_defined(anima_dir / "injection.md"):
        errors.append("injection_undefined")
    if errors:
        payload = _base_state(
            STATE_NEEDS_REPAIR,
            reason="complete_repair_requires_defined_identity",
            validation_errors=errors,
            retry_count=int(read_bootstrap_state(anima_dir).get("retry_count") or 0),
        )
        write_bootstrap_state(anima_dir, payload)
        return get_bootstrap_status(anima_dir)

    archived_bootstrap: list[str] = []
    bootstrap_file = anima_dir / "bootstrap.md"
    if bootstrap_file.exists():
        archived_bootstrap.append(str(_archive_bootstrap_file(anima_dir, bootstrap_file)))

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    for artifact in _bootstrap_artifacts(anima_dir):
        if artifact.exists():
            archived_bootstrap.append(str(_archive_named_file(anima_dir, artifact, f"{artifact.name}.{timestamp}")))

    character_sheet = anima_dir / "character_sheet.md"
    archived_character_sheet = ""
    if character_sheet.exists():
        archived_character_sheet = str(
            _archive_named_file(anima_dir, character_sheet, f"character_sheet-{timestamp}.md")
        )

    _remove_bootstrap_pending_tasks(anima_dir)

    if retry_counts_file and retry_counts_file.exists():
        data = _read_json(retry_counts_file)
        data.pop(anima_dir.name, None)
        retry_counts_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = _base_state(
        STATE_COMPLETED,
        mode="repaired",
        reason="completed_after_bootstrap_repair",
        retry_count=0,
        completed_at=_now_iso(),
    )
    if archived_bootstrap:
        payload["archived_bootstrap_artifacts"] = archived_bootstrap
    if archived_character_sheet:
        payload["archived_character_sheet"] = archived_character_sheet
    write_bootstrap_state(anima_dir, payload)
    return get_bootstrap_status(anima_dir)


def preserved_status_settings(status_data: dict[str, Any]) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for key, value in status_data.items():
        if key in PRESERVED_STATUS_KEYS or key.startswith("background_"):
            preserved[key] = value
    return preserved


def repair_bootstrap_fresh(animas_dir: Path, name: str, *, archive_root: Path) -> tuple[Path, Path]:
    """Archive and recreate a blank Anima, preserving model credential settings."""
    from core.anima_factory import create_blank

    anima_dir = animas_dir / name
    if not anima_dir.exists():
        raise FileNotFoundError(f"Anima not found: {name}")

    status_data = _read_json(anima_dir / "status.json")
    preserved = preserved_status_settings(status_data)

    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_root / f"{name}_bootstrap_repair_{timestamp}.zip"
    counter = 1
    while archive_path.exists():
        archive_path = archive_root / f"{name}_bootstrap_repair_{timestamp}_{counter}.zip"
        counter += 1

    shutil.make_archive(str(archive_path.with_suffix("")), "zip", str(anima_dir))
    shutil.rmtree(anima_dir)
    new_dir = create_blank(animas_dir, name)

    status_path = new_dir / "status.json"
    new_status = _read_json(status_path)
    new_status.update(preserved)
    status_path.write_text(json.dumps(new_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    initialize_bootstrap_state(new_dir)
    return new_dir, archive_path
