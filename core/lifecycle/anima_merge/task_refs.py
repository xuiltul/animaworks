from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic task-reference migration for an Anima merge.

The source Anima directory is treated as immutable.  Planning and applying are
separate operations so callers can durably journal the mapping before the first
write and reuse exactly the same mapping when resuming an interrupted merge.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.taskboard.tasks import TaskStore, task_database_path

from .taskboard_refs import rewrite_taskboard, taskboard_ids

_NAME_FIELDS = frozenset(
    {
        "anima",
        "anima_name",
        "assignee",
        "bounce_delegator",
        "delegated_to",
        "disabled_delegatee",
        "name",
        "submitted_by",
        "target",
        "updated_by",
    }
)
_LOCAL_TASK_ID_FIELDS = frozenset({"task_id", "tracking_task_id"})
_PENDING_ROOTS = (Path("state/background_tasks/pending"),)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _queue_ids(anima_dir: Path) -> set[str]:
    # Read-only planning must not trigger a migration or create a database.
    database = task_database_path(anima_dir)
    if database.is_file():
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "tasks" in tables:
                result = {row[0] for row in db.execute("SELECT task_id FROM tasks WHERE anima=?", (anima_dir.name,))}
                result.update(
                    row[0] for row in db.execute("SELECT alias FROM task_aliases WHERE viewer=?", (anima_dir.name,))
                )
                if db.execute("SELECT 1 FROM task_imports WHERE anima=?", (anima_dir.name,)).fetchone():
                    return result
            else:
                result = set()
    else:
        result = set()
    # Unmigrated JSONL is only an import-planning surface, never runtime authority.
    for descriptor in (anima_dir / "state" / "pending").rglob("*.json"):
        payload = _read_json_object(descriptor)
        if payload and payload.get("task_type") == "llm" and isinstance(payload.get("task_id"), str):
            result.add(payload["task_id"])
    path = anima_dir / "state" / "task_queue.jsonl"
    if not path.is_file():
        return result
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and isinstance(value.get("task_id"), str) and value["task_id"]:
                    result.add(value["task_id"])
    except OSError:
        pass
    return result


def _pending_ids(anima_dir: Path) -> set[str]:
    result: set[str] = set()
    for relative_root in _PENDING_ROOTS:
        root = anima_dir / relative_root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            # Lease sidecars end in .lease, so are naturally excluded.
            result.add(path.stem)
            value = _read_json_object(path)
            if value is not None and isinstance(value.get("task_id"), str) and value["task_id"]:
                result.add(value["task_id"])
    return result


def _result_ids(anima_dir: Path) -> set[str]:
    root = anima_dir / "state" / "task_results"
    if not root.is_dir():
        return set()
    return {
        path.name if path.is_dir() else path.stem for path in root.iterdir() if path.is_dir() or path.suffix == ".md"
    }


def _all_owned_ids(anima_dir: Path, db_path: Path, anima_name: str) -> set[str]:
    return _queue_ids(anima_dir) | _pending_ids(anima_dir) | _result_ids(anima_dir) | taskboard_ids(db_path, anima_name)


def _validate_mapping(mapping: dict[str, str], source_ids: set[str]) -> None:
    if set(mapping) != source_ids:
        missing = sorted(source_ids - set(mapping))
        extra = sorted(set(mapping) - source_ids)
        raise ValueError(f"Task ID mapping domain mismatch (missing={missing}, extra={extra})")
    if any(not isinstance(old, str) or not isinstance(new, str) or not old or not new for old, new in mapping.items()):
        raise ValueError("Task ID mapping keys and values must be non-empty strings")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Task ID mapping contains duplicate target IDs")


def _safe_filename_id(task_id: str) -> str:
    if task_id in {"", ".", ".."} or Path(task_id).name != task_id or "/" in task_id or "\\" in task_id:
        raise ValueError(f"Task ID is not safe for use as a filename: {task_id!r}")
    return task_id


@dataclass(frozen=True)
class TaskIdPlan:
    """A serializable, deterministic source-owner task ID mapping."""

    source: str
    target: str
    mapping: dict[str, str]

    def artifacts(self) -> dict[str, Any]:
        return {
            "task_id_mapping": dict(sorted(self.mapping.items())),
            "task_ids_discovered": len(self.mapping),
            "task_ids_remapped": sum(old != new for old, new in self.mapping.items()),
        }


class TaskReferenceRewriter:
    """Plan and apply task migrations for one source/target Anima pair."""

    def __init__(self, data_dir: Path, source: str, target: str) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.source = source
        self.target = target
        self.animas_dir = self.data_dir / "animas"
        self.source_dir = self.animas_dir / source
        self.target_dir = self.animas_dir / target
        self.taskboard_path = self.data_dir / "shared" / "taskboard.sqlite3"

    def plan(self) -> TaskIdPlan:
        """Discover all task surfaces and allocate collision-free target IDs."""
        source_ids = _all_owned_ids(self.source_dir, self.taskboard_path, self.source)
        reserved = _all_owned_ids(self.target_dir, self.taskboard_path, self.target)
        mapping: dict[str, str] = {}
        for old_id in sorted(source_ids):
            candidate = old_id
            if candidate in reserved:
                base = f"{old_id}__from_{self.source}"
                candidate = base
                index = 2
                while candidate in reserved:
                    candidate = f"{base}_{index}"
                    index += 1
            mapping[old_id] = candidate
            reserved.add(candidate)
        return TaskIdPlan(source=self.source, target=self.target, mapping=mapping)

    def apply(self, mapping: dict[str, str] | TaskIdPlan) -> dict[str, Any]:
        """Apply a previously journaled mapping, without modifying source files."""
        resolved = mapping.mapping if isinstance(mapping, TaskIdPlan) else dict(mapping)
        source_ids = _all_owned_ids(self.source_dir, self.taskboard_path, self.source)
        # On resume the TaskBoard source rows have already moved.  They therefore
        # disappear from discovery, while the immutable source files remain.  A
        # journaled mapping may consequently be a strict superset.
        if not source_ids.issubset(resolved):
            _validate_mapping(resolved, source_ids)
        if any(
            not isinstance(old, str) or not isinstance(new, str) or not old or not new for old, new in resolved.items()
        ):
            raise ValueError("Task ID mapping keys and values must be non-empty strings")
        if len(set(resolved.values())) != len(resolved):
            raise ValueError("Task ID mapping contains duplicate target IDs")

        queue_artifacts = self._rewrite_queues(resolved)
        pending_artifacts = self._copy_pending_descriptors(resolved)
        result_artifacts = self._copy_task_results(resolved)
        taskboard_artifacts = rewrite_taskboard(
            self.taskboard_path,
            self.source,
            self.target,
            resolved,
            _rewrite_value,
        )
        return {
            "task_id_mapping": dict(sorted(resolved.items())),
            **queue_artifacts,
            **pending_artifacts,
            **result_artifacts,
            "taskboard": taskboard_artifacts,
        }

    def _rewrite_queues(self, mapping: dict[str, str]) -> dict[str, Any]:
        # Import each legacy source once before the atomic ownership transfer.
        store = TaskStore(task_database_path(self.source_dir))
        for anima_dir in sorted(self.animas_dir.iterdir()):
            if anima_dir.is_dir():
                store.import_legacy(anima_dir)

        def transform(owner: str, value: Any) -> Any:
            rewritten = _rewrite_value(
                value, source=self.source, target=self.target, mapping=mapping, owner_is_source=owner == self.source
            )
            return _rewrite_anima_paths(rewritten, self.source_dir, self.target_dir)

        return store.transfer_anima_tasks(self.source, self.target, mapping, transform)

    def _copy_pending_descriptors(self, mapping: dict[str, str]) -> dict[str, Any]:
        copied: list[dict[str, str]] = []
        for relative_root in _PENDING_ROOTS:
            source_root = self.source_dir / relative_root
            if not source_root.is_dir():
                continue
            for source_path in sorted(source_root.rglob("*.json")):
                payload = _read_json_object(source_path)
                if payload is None:
                    # Malformed historical/failed descriptors are left in source
                    # provenance rather than copied into the target pipeline.
                    continue
                rewritten = _rewrite_value(
                    payload,
                    source=self.source,
                    target=self.target,
                    mapping=mapping,
                    owner_is_source=True,
                )
                rewritten = _rewrite_anima_paths(rewritten, self.source_dir, self.target_dir)
                relative = source_path.relative_to(source_root)
                old_stem = source_path.stem
                new_stem = mapping.get(old_stem, old_stem)
                _safe_filename_id(new_stem)
                destination_relative = relative.with_name(f"{new_stem}{source_path.suffix}")
                destination = self.target_dir / relative_root / destination_relative
                content = json.dumps(rewritten, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                if destination.exists():
                    if destination.read_text(encoding="utf-8") != content:
                        raise ValueError(f"Conflicting target pending descriptor: {destination}")
                    continue
                atomic_write_text(destination, content)
                copied.append(
                    {
                        "source": source_path.relative_to(self.data_dir).as_posix(),
                        "target": destination.relative_to(self.data_dir).as_posix(),
                    }
                )
        return {"pending_descriptors_copied": copied}

    def _copy_task_results(self, mapping: dict[str, str]) -> dict[str, Any]:
        source_root = self.source_dir / "state" / "task_results"
        target_root = self.target_dir / "state" / "task_results"
        copied: list[dict[str, str]] = []
        if not source_root.is_dir():
            return {"task_results_copied": copied}
        for source_path in sorted(source_root.rglob("*.md")):
            relative = source_path.relative_to(source_root)
            old_id = relative.parts[0] if len(relative.parts) > 1 else source_path.stem
            if old_id not in mapping:
                continue
            new_id = _safe_filename_id(mapping[old_id])
            destination = (
                target_root / new_id / Path(*relative.parts[1:])
                if len(relative.parts) > 1
                else target_root / f"{new_id}.md"
            )
            content = source_path.read_text(encoding="utf-8")
            if destination.exists():
                if destination.read_text(encoding="utf-8") != content:
                    raise ValueError(f"Conflicting target task result: {destination}")
                continue
            atomic_write_text(destination, content)
            copied.append(
                {
                    "source": source_path.relative_to(self.data_dir).as_posix(),
                    "target": destination.relative_to(self.data_dir).as_posix(),
                }
            )
        return {"task_results_copied": copied}


def _rewrite_value(
    value: Any,
    *,
    source: str,
    target: str,
    mapping: dict[str, str],
    owner_is_source: bool,
    key: str = "",
    delegated_owner_is_source: bool = False,
) -> Any:
    """Schema-aware recursive rewrite that never treats task IDs as global."""
    if isinstance(value, list):
        if key == "relay_chain":
            return [target if item == source else item for item in value]
        if key == "depends_on" and owner_is_source:
            return [mapping.get(item, item) if isinstance(item, str) else item for item in value]
        return [
            _rewrite_value(
                item,
                source=source,
                target=target,
                mapping=mapping,
                owner_is_source=owner_is_source,
                key=key,
                delegated_owner_is_source=delegated_owner_is_source,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        if isinstance(value, str):
            if key in _NAME_FIELDS and value == source:
                return target
            if key == "reply_to" and value == source:
                return target
            if key in _LOCAL_TASK_ID_FIELDS and owner_is_source:
                return mapping.get(value, value)
            if key == "delegated_task_id" and delegated_owner_is_source:
                return mapping.get(value, value)
            if key in {"source_ref", "replaced_by"}:
                prefix = f"task_queue:{source}:"
                if value.startswith(prefix):
                    old_id = value[len(prefix) :]
                    return f"task_queue:{target}:{mapping.get(old_id, old_id)}"
                if key == "replaced_by" and owner_is_source:
                    return mapping.get(value, value)
            if key == "anima_dir":
                source_suffix = f"/animas/{source}"
                if value.endswith(source_suffix):
                    return f"{value[: -len(source_suffix)]}/animas/{target}"
        return value

    dict_owner_is_source = owner_is_source or value.get("anima_name") == source
    delegated_to = value.get("delegated_to")
    target_name = value.get("target")
    child_owned_by_source = delegated_to == source or target_name == source
    result: dict[str, Any] = {}
    for child_key, child_value in value.items():
        child_owner_is_source = dict_owner_is_source
        # Nested metadata/task descriptors belong to their containing queue row.
        result[child_key] = _rewrite_value(
            child_value,
            source=source,
            target=target,
            mapping=mapping,
            owner_is_source=child_owner_is_source,
            key=child_key,
            delegated_owner_is_source=child_owned_by_source,
        )
    return result


def _rewrite_anima_paths(value: Any, source_dir: Path, target_dir: Path) -> Any:
    """Rewrite only absolute paths rooted in the immutable source directory."""
    if isinstance(value, list):
        return [_rewrite_anima_paths(item, source_dir, target_dir) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_anima_paths(item, source_dir, target_dir) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    source_text = str(source_dir)
    if value == source_text:
        return str(target_dir)
    prefix = f"{source_text}/"
    if value.startswith(prefix):
        return f"{target_dir}/{value[len(prefix) :]}"
    return value


def build_task_id_mapping(data_dir: Path, source: str, target: str) -> dict[str, str]:
    """Convenience API returning the journal-ready mapping only."""
    return TaskReferenceRewriter(data_dir, source, target).plan().mapping


def rewrite_task_references(
    data_dir: Path,
    source: str,
    target: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Convenience API applying a previously persisted task ID mapping."""
    return TaskReferenceRewriter(data_dir, source, target).apply(mapping)
