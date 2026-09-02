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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry

from .taskboard_refs import rewrite_taskboard, taskboard_ids

_ACTIVE_STATUSES = frozenset({"pending", "in_progress", "delegated"})
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
_PENDING_ROOTS = (
    Path("state/pending"),
    Path("state/background_tasks/pending"),
)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _queue_ids(anima_dir: Path) -> set[str]:
    path = anima_dir / "state" / "task_queue.jsonl"
    result: set[str] = set()
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
    return {path.stem for path in root.glob("*.md") if path.is_file()}


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

    def _active_source_tasks(self, mapping: dict[str, str]) -> list[dict[str, Any]]:
        current = _replay_queue(self.source_dir / "state" / "task_queue.jsonl")
        migrated: list[dict[str, Any]] = []
        for old_id, entry in sorted(current.items()):
            if entry.status not in _ACTIVE_STATUSES or old_id not in mapping:
                continue
            raw = entry.model_dump(mode="json")
            rewritten = _rewrite_value(
                raw,
                source=self.source,
                target=self.target,
                mapping=mapping,
                owner_is_source=True,
            )
            rewritten = _rewrite_anima_paths(rewritten, self.source_dir, self.target_dir)
            migrated.append(rewritten)
        return migrated

    def _rewrite_queues(self, mapping: dict[str, str]) -> dict[str, Any]:
        updated: list[str] = []
        recreated: list[str] = []
        active_source = self._active_source_tasks(mapping)

        for anima_dir in sorted(self.animas_dir.iterdir()) if self.animas_dir.is_dir() else []:
            if not anima_dir.is_dir() or anima_dir.name == self.source:
                continue
            path = anima_dir / "state" / "task_queue.jsonl"
            manager = TaskQueueManager(anima_dir)
            with manager._locked_queue():  # noqa: SLF001 - same lock as runtime appends
                changed, recreated_here = self._rewrite_queue_file(
                    path,
                    mapping,
                    active_source if anima_dir.name == self.target else [],
                )
            recreated.extend(recreated_here)
            if changed:
                updated.append(path.relative_to(self.data_dir).as_posix())

        return {
            "task_queue_files_updated": updated,
            "source_tasks_recreated": sorted(recreated),
        }

    def _rewrite_queue_file(
        self,
        path: Path,
        mapping: dict[str, str],
        active_source: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        original = path.read_text(encoding="utf-8") if path.is_file() else ""
        rewritten_lines: list[str] = []
        changed = False
        for line in original.splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                rewritten_lines.append(line)
                continue
            if not isinstance(raw, dict):
                rewritten_lines.append(line)
                continue
            rewritten = _rewrite_value(
                raw,
                source=self.source,
                target=self.target,
                mapping=mapping,
                owner_is_source=False,
            )
            rewritten_lines.append(_json_line(rewritten) if rewritten != raw else line)
            changed = changed or rewritten != raw

        recreated: list[str] = []
        current_ids = set(_replay_queue_lines(rewritten_lines))
        for migrated in active_source:
            new_id = str(migrated["task_id"])
            if new_id in current_ids:
                continue
            rewritten_lines.append(_json_line(migrated))
            current_ids.add(new_id)
            recreated.append(new_id)
            changed = True
        if changed:
            content = "\n".join(rewritten_lines) + ("\n" if rewritten_lines else "")
            atomic_write_text(path, content)
        return changed, recreated

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
        for source_path in sorted(source_root.glob("*.md")):
            old_id = source_path.stem
            if old_id not in mapping:
                continue
            new_id = _safe_filename_id(mapping[old_id])
            destination = target_root / f"{new_id}.md"
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


def _replay_queue(path: Path) -> dict[str, TaskEntry]:
    if not path.is_file():
        return {}
    return _replay_queue_lines(path.read_text(encoding="utf-8").splitlines())


def _replay_queue_lines(lines: list[str]) -> dict[str, TaskEntry]:
    tasks: dict[str, TaskEntry] = {}
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        if raw.get("_event") == "update":
            existing = tasks.get(task_id)
            if existing is None:
                continue
            if "status" in raw:
                existing.status = raw["status"]
            if "summary" in raw:
                existing.summary = raw["summary"]
            if "updated_at" in raw:
                existing.updated_at = raw["updated_at"]
            if isinstance(raw.get("meta"), dict):
                existing.meta = raw["meta"]
            continue
        try:
            tasks[task_id] = TaskEntry(**raw)
        except Exception:
            continue
    return tasks


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
