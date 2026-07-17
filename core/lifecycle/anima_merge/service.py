from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Collision-safe, resumable Anima memory merge through index rebuilding."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from core.anima_factory import validate_anima_name
from core.memory._io import atomic_write_text
from core.memory.backend.registry import resolve_backend_type
from core.memory.facts import FactRecord, append_fact_records, iter_fact_records
from core.platform.locks import acquire_file_lock, release_file_lock
from core.time_utils import now_iso, now_local

from .journal import MergeJournal, MergePhase

_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_ARCHIVE_ONLY = ("activity_log", "token_usage", "prompt_logs")


class AnimaMergeError(RuntimeError):
    """Raised when a merge cannot safely continue."""


@dataclass(frozen=True)
class MergeResult:
    source: str
    target: str
    dry_run: bool
    manifest_json: Path
    manifest_markdown: Path
    journal_path: Path | None = None
    snapshot_path: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _files(root: Path) -> Iterator[Path]:
    if root.is_dir():
        yield from (path for path in sorted(root.rglob("*")) if path.is_file())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnimaMergeError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnimaMergeError(f"Expected a JSON object in {path}")
    return value


class AnimaMergeService:
    """Merge canonical memory and rebuild target indexes in one runtime data directory."""

    def __init__(
        self,
        data_dir: Path,
        source: str,
        target: str,
        *,
        gateway_url: str = "http://localhost:18500",
        force: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.source = source
        self.target = target
        self.gateway_url = gateway_url.rstrip("/")
        self.force = force
        self.animas_dir = self.data_dir / "animas"
        self.source_dir = self.animas_dir / source
        self.target_dir = self.animas_dir / target
        self.state_dir = self.data_dir / "state"
        self.journal_path = self.state_dir / f"merge_journal_{source}_{target}.json"
        self.lock_path = self.state_dir / "anima_merge.lock"

    def run(self, *, execute: bool = False, resume: bool = False) -> MergeResult:
        """Generate a manifest, or execute the complete merge through tombstone."""
        if resume and not execute:
            raise AnimaMergeError("--resume requires --execute")
        self._validate_names()

        with self._runtime_data_dir(), self._global_lock():
            issues = self.preflight()
            manifest = self.build_manifest(issues)
            manifest_json, manifest_markdown = self.write_manifest(manifest)
            if not execute:
                return MergeResult(
                    source=self.source,
                    target=self.target,
                    dry_run=True,
                    manifest_json=manifest_json,
                    manifest_markdown=manifest_markdown,
                )

            journal = self._open_journal(resume=resume)
            self._run_phase(
                journal,
                MergePhase.PREFLIGHT,
                lambda: {
                    "manifest_json": str(manifest_json),
                    "manifest_markdown": str(manifest_markdown),
                    "warnings": issues,
                    "memory_backend": manifest["memory_backend"],
                },
            )
            self._run_phase(journal, MergePhase.QUIESCE, self.quiesce)
            self._run_phase(journal, MergePhase.SNAPSHOT, lambda: self.snapshot(journal))
            self._run_phase(journal, MergePhase.MERGE_MEMORY, self.merge_memory)
            self._run_phase(
                journal,
                MergePhase.REWRITE_REFS,
                lambda: self.rewrite_refs(journal),
            )
            self._run_phase(
                journal,
                MergePhase.REBUILD_INDEXES,
                lambda: self.rebuild_indexes(journal),
            )
            self._run_phase(journal, MergePhase.VERIFY, lambda: self.verify(journal))
            self._run_phase(journal, MergePhase.TOMBSTONE, lambda: self.tombstone(journal))
            if not journal.is_completed(MergePhase.DONE):
                journal.start(MergePhase.DONE)
                journal.complete(
                    MergePhase.DONE,
                    {
                        "phase_2_complete": True,
                        "phase_3_complete": True,
                        "phase_4_complete": True,
                        "manual_smoke_required": bool(
                            journal.phase_artifacts(MergePhase.VERIFY)
                            .get("smoke_check", {})
                            .get("manual_required", False)
                        ),
                    },
                )
            journal.finish()
            snapshot_raw = journal.data.get("artifacts", {}).get("snapshot_path")
            return MergeResult(
                source=self.source,
                target=self.target,
                dry_run=False,
                manifest_json=manifest_json,
                manifest_markdown=manifest_markdown,
                journal_path=self.journal_path,
                snapshot_path=Path(snapshot_raw) if isinstance(snapshot_raw, str) else None,
            )

    def _validate_names(self) -> None:
        for label, name in (("SOURCE", self.source), ("TARGET", self.target)):
            error = validate_anima_name(name)
            if error:
                raise AnimaMergeError(f"Invalid {label} name '{name}': {error}")

    @contextmanager
    def _global_lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                acquire_file_lock(lock_file, exclusive=True, blocking=False)
            except OSError as exc:
                raise AnimaMergeError(f"Another Anima merge is in progress ({self.lock_path})") from exc
            try:
                yield
            finally:
                release_file_lock(lock_file)

    @contextmanager
    def _runtime_data_dir(self) -> Iterator[None]:
        """Keep path-singleton dependencies inside this service's data root."""

        previous = os.environ.get("ANIMAWORKS_DATA_DIR")
        os.environ["ANIMAWORKS_DATA_DIR"] = str(self.data_dir)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("ANIMAWORKS_DATA_DIR", None)
            else:
                os.environ["ANIMAWORKS_DATA_DIR"] = previous

    def _open_journal(self, *, resume: bool) -> MergeJournal:
        try:
            return MergeJournal(self.journal_path, self.source, self.target, resume=resume)
        except ValueError as exc:
            raise AnimaMergeError(str(exc)) from exc

    @staticmethod
    def _run_phase(journal: MergeJournal, phase: MergePhase, action: Any) -> None:
        if journal.is_completed(phase):
            return
        journal.start(phase)
        try:
            artifacts = action() or {}
        except Exception as exc:
            journal.fail(phase, f"{type(exc).__name__}: {exc}")
            raise
        journal.complete(phase, artifacts)

    # ── Preflight and manifest ────────────────────────────────

    def preflight(self) -> list[str]:
        errors: list[str] = []
        if self.source == self.target:
            errors.append("SOURCE and TARGET must be different")
        for name, anima_dir in ((self.source, self.source_dir), (self.target, self.target_dir)):
            if not anima_dir.is_dir():
                errors.append(f"Anima directory does not exist: animas/{name}")
                continue
            for required in ("identity.md", "status.json"):
                if not (anima_dir / required).is_file():
                    errors.append(f"Missing required file: animas/{name}/{required}")
        if errors:
            raise AnimaMergeError("Preflight failed:\n- " + "\n- ".join(errors))

        source_backend = resolve_backend_type(self.source_dir)
        target_backend = resolve_backend_type(self.target_dir)
        if source_backend != target_backend:
            raise AnimaMergeError(
                f"Preflight failed: memory backend mismatch ({self.source}={source_backend}, "
                f"{self.target}={target_backend})"
            )

        dangerous: list[str] = []
        for anima_dir in (self.source_dir, self.target_dir):
            prefix = f"animas/{anima_dir.name}"
            consolidation = anima_dir / "state" / ".consolidation_mode"
            if consolidation.exists():
                dangerous.append(f"{prefix}/state/.consolidation_mode")
            repair_state = anima_dir / "state" / "rag_repair.json"
            if repair_state.is_file() and self._repair_in_progress(repair_state):
                dangerous.append(f"{prefix}/state/rag_repair.json (in progress)")
            processing = anima_dir / "state" / "pending" / "processing"
            if processing.is_dir():
                for path in _files(processing):
                    dangerous.append(f"{prefix}/{_safe_relative(path, anima_dir)}")
            for path in self._streaming_journals(anima_dir):
                dangerous.append(f"{prefix}/{_safe_relative(path, anima_dir)} (unrecovered streaming journal)")
        if dangerous and not self.force:
            raise AnimaMergeError(
                "Preflight found in-progress state; resolve it or rerun with --force:\n- " + "\n- ".join(dangerous)
            )
        return dangerous

    @staticmethod
    def _repair_in_progress(path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if not isinstance(data, dict):
            return True
        status = str(data.get("status", data.get("phase", ""))).strip().lower()
        if not status:
            return True
        # repair_state.py writes: healthy(初期値)/requested/active/cooldown/disabled/
        # locked/repairing/success/failed。cooldownは再試行待ち(DB不健全の可能性)なのでブロック対象。
        return status not in {
            "idle",
            "done",
            "completed",
            "failed",
            "cancelled",
            "canceled",
            "healthy",
            "success",
            "disabled",
        }

    @staticmethod
    def _streaming_journals(anima_dir: Path) -> list[Path]:
        shortterm = anima_dir / "shortterm"
        if not shortterm.is_dir():
            return []
        matches = list(shortterm.glob("streaming_journal*.jsonl"))
        matches.extend(shortterm.rglob("streaming_journal.jsonl"))
        return sorted(set(path for path in matches if path.is_file()))

    def build_manifest(self, warnings: list[str] | None = None) -> dict[str, Any]:
        source_backend = resolve_backend_type(self.source_dir)
        target_backend = resolve_backend_type(self.target_dir)
        return {
            "version": 1,
            "generated_at": now_iso(),
            "source": self.source,
            "target": self.target,
            "mode": "dry-run",
            "memory_backend": {"source": source_backend, "target": target_backend},
            "tree_summary": {
                "source": self._tree_summary(self.source_dir),
                "target": self._tree_summary(self.target_dir),
            },
            "collisions": self._collisions(),
            "task_id_collisions": self._task_id_collisions(),
            "thread_id_collisions": self._thread_id_collisions(),
            "external_references": self._external_references(),
            "rebuild_indexes": self._rebuild_estimate(target_backend),
            "verify": self._verify_estimate(),
            "preflight_warnings": list(warnings or []),
        }

    def _verify_estimate(self) -> dict[str, Any]:
        from .verification import estimate_probe_counts

        counts = estimate_probe_counts(self.source_dir)
        return {
            "probe_categories": counts,
            "estimated_probes": sum(counts.values()),
            "reference_surfaces": [
                "config",
                "anima_status",
                "taskboard",
                "shared_inbox",
                "reply_routing",
            ],
            "smoke_check": "run_if_server_online",
        }

    def _rebuild_estimate(self, backend: str) -> dict[str, Any]:
        """Estimate target rebuild inputs without mutating either Anima."""
        categories: dict[str, int] = {}
        for memory_type, pattern in (
            ("knowledge", "*.md"),
            ("episodes", "*.md"),
            ("procedures", "*.md"),
            ("skills", "SKILL.md"),
        ):
            categories[memory_type] = sum(
                1
                for root in (self.source_dir / memory_type, self.target_dir / memory_type)
                for path in root.rglob(pattern)
                if path.is_file()
            )

        fact_ids = {
            record.fact_id
            for anima_dir in (self.source_dir, self.target_dir)
            for record in iter_fact_records(anima_dir, include_expired=True)
        }
        categories["facts"] = len(fact_ids)
        conversation = self.target_dir / "state" / "conversation.json"
        categories["conversation_summary"] = int(self._has_conversation_summary(conversation))
        return {
            "target": self.target,
            "estimated_inputs": categories,
            "substeps": ["vectordb", "entities", "bm25", "graph_cache", "neo4j"],
            "neo4j_action": "reingest_target_group" if backend == "neo4j" else "skip_not_configured",
        }

    @staticmethod
    def _has_conversation_summary(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(data, dict) and bool(str(data.get("compressed_summary", "")).strip())

    @staticmethod
    def _tree_summary(anima_dir: Path) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for path in _files(anima_dir):
            relative = path.relative_to(anima_dir)
            category = relative.parts[0] if len(relative.parts) > 1 else "root"
            entry = summary.setdefault(category, {"files": 0, "bytes": 0})
            entry["files"] += 1
            try:
                entry["bytes"] += path.stat().st_size
            except OSError:
                pass
        return dict(sorted(summary.items()))

    def _collisions(self) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {
            "episodes": [],
            "knowledge": [],
            "procedures": [],
            "skills": [],
            "attachments": [],
        }
        target_dates: dict[str, list[str]] = {}
        for path in _files(self.target_dir / "episodes"):
            match = _DATE_PREFIX.match(path.name)
            if match:
                target_dates.setdefault(match.group(1), []).append(_safe_relative(path, self.target_dir))
        for path in _files(self.source_dir / "episodes"):
            match = _DATE_PREFIX.match(path.name)
            if match and match.group(1) in target_dates:
                result["episodes"].append(
                    {"source": _safe_relative(path, self.source_dir), "target_date": match.group(1)}
                )

        for category in ("knowledge", "procedures"):
            source_root = self.source_dir / category
            target_root = self.target_dir / category
            for path in _files(source_root):
                relative = path.relative_to(source_root)
                target_path = target_root / relative
                if target_path.is_file():
                    result[category].append(
                        {
                            "path": f"{category}/{relative.as_posix()}",
                            "same_content": str(_sha256(path) == _sha256(target_path)).lower(),
                        }
                    )

        source_skills = self._skill_directories(self.source_dir)
        target_skills = self._skill_directories(self.target_dir)
        for name in sorted(source_skills.keys() & target_skills.keys()):
            result["skills"].append({"source": source_skills[name], "target": target_skills[name]})

        target_attachments: set[str] = set()
        for path in _files(self.target_dir / "attachments"):
            target_attachments.add(_safe_relative(path, self.target_dir / "attachments"))
        for path in _files(self.source_dir / "attachments"):
            relative = _safe_relative(path, self.source_dir / "attachments")
            if relative in target_attachments:
                result["attachments"].append(
                    {"source": _safe_relative(path, self.source_dir), "basename": path.name}
                )
        return result

    @staticmethod
    def _skill_directories(anima_dir: Path) -> dict[str, str]:
        root = anima_dir / "skills"
        result: dict[str, str] = {}
        if not root.is_dir():
            return result
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if path.name == "quarantine":
                for child in sorted(path.iterdir()):
                    if child.is_dir():
                        result[f"quarantine/{child.name}"] = f"skills/quarantine/{child.name}"
            else:
                result[path.name] = f"skills/{path.name}"
        return result

    @staticmethod
    def _task_ids(anima_dir: Path) -> set[str]:
        path = anima_dir / "state" / "task_queue.jsonl"
        result: set[str] = set()
        if not path.is_file():
            return result
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        task_id = item.get("task_id", item.get("id"))
                        if isinstance(task_id, str) and task_id:
                            result.add(task_id)
        except OSError:
            pass
        return result

    def _task_id_collisions(self) -> list[str]:
        from .task_refs import build_task_id_mapping

        mapping = build_task_id_mapping(self.data_dir, self.source, self.target)
        return sorted(old_id for old_id, new_id in mapping.items() if old_id != new_id)

    def _thread_id_collisions(self) -> list[str]:
        def ids(anima_dir: Path) -> set[str]:
            root = anima_dir / "state" / "conversations"
            return {path.stem for path in root.glob("*.json")} if root.is_dir() else set()

        return sorted(ids(self.source_dir) & ids(self.target_dir))

    def _external_references(self) -> dict[str, list[dict[str, str]]]:
        supervisors: list[dict[str, str]] = []
        if self.animas_dir.is_dir():
            for anima_dir in sorted(self.animas_dir.iterdir()):
                status_path = anima_dir / "status.json"
                if not status_path.is_file():
                    continue
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(status, dict) and status.get("supervisor") == self.source:
                    supervisors.append({"anima": anima_dir.name, "path": f"animas/{anima_dir.name}/status.json"})

        mappings: list[dict[str, str]] = []
        config_path = self.data_dir / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                config = {}
            external = config.get("external_messaging", {}) if isinstance(config, dict) else {}
            self._find_value_paths(external, self.source, "external_messaging", mappings)
        return {"supervisors": supervisors, "external_messaging": mappings}

    @classmethod
    def _find_value_paths(
        cls,
        value: Any,
        wanted: str,
        prefix: str,
        result: list[dict[str, str]],
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                cls._find_value_paths(child, wanted, f"{prefix}.{key}", result)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                cls._find_value_paths(child, wanted, f"{prefix}[{index}]", result)
        elif value == wanted:
            result.append({"path": prefix})

    def write_manifest(self, manifest: dict[str, Any]) -> tuple[Path, Path]:
        output_dir = self.state_dir / "merge_manifests"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = now_local().strftime("%Y%m%d_%H%M%S_%f")
        base = output_dir / f"merge_{self.source}_{self.target}_{stamp}"
        json_path = base.with_suffix(".json")
        markdown_path = base.with_suffix(".md")
        json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown_path.write_text(self._manifest_markdown(manifest), encoding="utf-8")
        return json_path, markdown_path

    @staticmethod
    def _manifest_markdown(manifest: dict[str, Any]) -> str:
        lines = [
            f"# Anima merge dry-run: {manifest['source']} → {manifest['target']}",
            "",
            f"Generated: {manifest['generated_at']}",
            f"Memory backend: {manifest['memory_backend']['source']}",
            "",
            "## Tree summary",
            "",
            "| Anima | Category | Files | Bytes |",
            "|---|---|---:|---:|",
        ]
        for side in ("source", "target"):
            for category, values in manifest["tree_summary"][side].items():
                lines.append(f"| {side} | {category} | {values['files']} | {values['bytes']} |")
        lines.extend(["", "## Collisions", ""])
        for category, collisions in manifest["collisions"].items():
            lines.append(f"- {category}: {len(collisions)}")
            for collision in collisions:
                lines.append(f"  - `{json.dumps(collision, ensure_ascii=False, sort_keys=True)}`")
        lines.extend(
            [
                "",
                "## ID collisions",
                "",
                f"- task IDs: {', '.join(manifest['task_id_collisions']) or '(none)'}",
                f"- thread IDs: {', '.join(manifest['thread_id_collisions']) or '(none)'}",
                "",
                "## External references",
                "",
                f"- supervisor references: {len(manifest['external_references']['supervisors'])}",
                f"- external messaging references: {len(manifest['external_references']['external_messaging'])}",
                "",
                "## REBUILD_INDEXES estimate",
                "",
            ]
        )
        estimate = manifest["rebuild_indexes"]
        for category, count in estimate["estimated_inputs"].items():
            lines.append(f"- {category}: {count}")
        lines.extend([f"- Neo4j: {estimate['neo4j_action']}", "", "## VERIFY plan", ""])
        verify = manifest["verify"]
        for category, count in verify["probe_categories"].items():
            lines.append(f"- {category}: {count}")
        lines.extend(["", "## Preflight warnings", ""])
        warnings = manifest.get("preflight_warnings", [])
        lines.extend(f"- {warning}" for warning in warnings)
        if not warnings:
            lines.append("- (none)")
        return "\n".join(lines) + "\n"

    # ── Execute phases ────────────────────────────────────────

    def quiesce(self) -> dict[str, Any]:
        """Stop both Animas before mutation.

        Disabling through the runtime API may update the source ``status.json``.
        That quiesce side effect is the sole permitted exception to the merge's
        otherwise strict source-directory immutability rule.
        """
        server_running = self._server_running()
        disabled: list[str] = []
        if server_running:
            import requests

            for name in (self.source, self.target):
                try:
                    response = requests.post(f"{self.gateway_url}/api/animas/{name}/disable", timeout=10)
                    response.raise_for_status()
                except Exception as exc:
                    raise AnimaMergeError(f"Server is running, but anima '{name}' could not be disabled: {exc}") from exc
                disabled.append(name)

        for name in (self.source, self.target):
            self._wait_for_process_release(name, timeout=30.0 if server_running else 0.0)
        return {"server_running": server_running, "disabled_via_api": disabled}

    def _server_running(self) -> bool:
        return self._pid_path_alive(self.data_dir / "server.pid")

    @staticmethod
    def _pid_path_alive(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
            if pid <= 0:
                return False
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def _wait_for_process_release(self, name: str, *, timeout: float) -> None:
        pid_path = self.data_dir / "run" / "animas" / f"{name}.pid"
        socket_path = self.data_dir / "run" / "sockets" / f"{name}.sock"
        deadline = time.monotonic() + timeout
        while self._pid_path_alive(pid_path):
            if time.monotonic() >= deadline:
                raise AnimaMergeError(f"Anima process '{name}' is still running ({pid_path})")
            time.sleep(0.1)
        pid_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
        if socket_path.exists():
            raise AnimaMergeError(f"Anima socket was not released: {socket_path}")

    def snapshot(self, journal: MergeJournal) -> dict[str, Any]:
        created = str(journal.data.get("created_at", now_iso()))
        timestamp = re.sub(r"[^0-9]", "", created)[:20]
        snapshot_dir = self.data_dir / "backup" / f"merge_{self.source}_{self.target}_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []

        for name, source_path, destination in (
            (self.source, self.source_dir, snapshot_dir / "animas" / self.source),
            (self.target, self.target_dir, snapshot_dir / "animas" / self.target),
        ):
            del name
            shutil.copytree(source_path, destination, dirs_exist_ok=True, copy_function=shutil.copy2)
            copied.append(str(destination))

        config_path = self.data_dir / "config.json"
        if config_path.is_file():
            destination = snapshot_dir / "config.json"
            shutil.copy2(config_path, destination)
            copied.append(str(destination))

        for name in (self.source, self.target):
            inbox = self.data_dir / "shared" / "inbox" / name
            if inbox.is_dir():
                destination = snapshot_dir / "shared" / "inbox" / name
                shutil.copytree(inbox, destination, dirs_exist_ok=True, copy_function=shutil.copy2)
                copied.append(str(destination))
        copied.extend(self._snapshot_reference_state(snapshot_dir))
        return {"snapshot_path": str(snapshot_dir), "snapshot_files": copied}

    def _snapshot_reference_state(self, snapshot_dir: Path) -> list[str]:
        """Snapshot mutable Phase-3 references before REWRITE_REFS."""
        copied: list[str] = []

        def copy_file(path: Path) -> None:
            if not path.is_file():
                return
            destination = snapshot_dir / path.relative_to(self.data_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(str(destination))

        for anima_dir in sorted(self.animas_dir.iterdir()) if self.animas_dir.is_dir() else []:
            if not anima_dir.is_dir() or anima_dir.name in {self.source, self.target}:
                continue
            copy_file(anima_dir / "status.json")
            copy_file(anima_dir / "state" / "task_queue.jsonl")

        shared_dir = self.data_dir / "shared"
        for pattern in ("channels/*.meta.json", "meetings/*.json"):
            for path in sorted(shared_dir.glob(pattern)):
                copy_file(path)
        for path in (
            self.data_dir / "usage_governor_state.json",
            self.animas_dir / ".bootstrap_retries.json",
            self.data_dir / "run" / "notification_map.json",
            self.data_dir / "run" / "discord_thread_map.json",
        ):
            copy_file(path)

        taskboard = shared_dir / "taskboard.sqlite3"
        if taskboard.is_file():
            destination = snapshot_dir / taskboard.relative_to(self.data_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_conn = sqlite3.connect(taskboard)
            destination_conn = sqlite3.connect(destination)
            try:
                source_conn.backup(destination_conn)
            finally:
                destination_conn.close()
                source_conn.close()
            copied.append(str(destination))

        for path in (
            self.data_dir / "run" / "inbox_wake" / self.source,
            self.data_dir / "run" / "animas" / f"{self.source}.pid",
            self.data_dir / "run" / "animas" / f"{self.source}.lock",
            self.data_dir / "run" / "sockets" / f"{self.source}.sock",
        ):
            copy_file(path)
        events = self.data_dir / "run" / "events" / self.source
        if events.is_dir():
            destination = snapshot_dir / events.relative_to(self.data_dir)
            shutil.copytree(events, destination, dirs_exist_ok=True, copy_function=shutil.copy2)
            copied.append(str(destination))
        return copied

    def merge_memory(self) -> dict[str, Any]:
        recovered = self._recover_abandoned_memory()
        episode_mapping, episode_deduped = self._merge_episodes()
        file_mapping: dict[str, str] = dict(episode_mapping)
        deduplicated: list[str] = list(episode_deduped)
        for category in ("knowledge", "procedures"):
            mapping, deduped = self._merge_markdown_tree(category)
            file_mapping.update(mapping)
            deduplicated.extend(deduped)
        fact_result = self._merge_facts(episode_mapping)
        skill_mapping = self._merge_skills()
        attachment_mapping = self._merge_attachments()
        conversation_files = self._merge_conversation_history()
        archive_plan = self._archive_plan()
        skill_state = self._skill_state_provenance(skill_mapping)
        return {
            "file_mapping": file_mapping,
            "episode_mapping": episode_mapping,
            "deduplicated_files": deduplicated,
            "fact_id_mapping": fact_result["id_mapping"],
            "facts_read": fact_result["read"],
            "facts_appended": fact_result["appended"],
            "facts_appended_ids": fact_result["appended_ids"],
            "skill_mapping": skill_mapping,
            "attachment_mapping": attachment_mapping,
            "skill_state_provenance": skill_state,
            "recovered_memory": recovered,
            "conversation_episodes": conversation_files,
            "archive_plan": archive_plan,
        }

    def _copy_collision_safe(self, source_path: Path, desired: Path, namespace: str) -> tuple[Path, bool]:
        source_hash = _sha256(source_path)
        if not desired.exists():
            desired.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, desired)
            return desired, False
        if desired.is_file() and _sha256(desired) == source_hash:
            return desired, True
        candidate = desired.with_name(f"{desired.stem}{namespace}{desired.suffix}")
        if candidate.exists() and candidate.is_file() and _sha256(candidate) == source_hash:
            return candidate, False
        index = 2
        while candidate.exists():
            candidate = desired.with_name(f"{desired.stem}{namespace}_{index}{desired.suffix}")
            if candidate.is_file() and _sha256(candidate) == source_hash:
                return candidate, False
            index += 1
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, candidate)
        return candidate, False

    def _merge_episodes(self) -> tuple[dict[str, str], list[str]]:
        source_root = self.source_dir / "episodes"
        target_root = self.target_dir / "episodes"
        mapping: dict[str, str] = {}
        deduplicated: list[str] = []
        target_dates = {
            match.group(1)
            for path in _files(target_root)
            if (match := _DATE_PREFIX.match(path.name)) is not None
        }
        for source_path in _files(source_root):
            relative = source_path.relative_to(source_root)
            desired = target_root / relative
            match = _DATE_PREFIX.match(source_path.name)
            namespace = f"_{self.source}"
            if match and match.group(1) in target_dates and not desired.exists():
                desired = desired.with_name(f"{desired.stem}{namespace}{desired.suffix}")
            destination, is_duplicate = self._copy_collision_safe(source_path, desired, namespace)
            source_key = f"episodes/{relative.as_posix()}"
            target_key = f"episodes/{destination.relative_to(target_root).as_posix()}"
            mapping[source_key] = target_key
            if is_duplicate:
                deduplicated.append(source_key)
            date_match = _DATE_PREFIX.match(destination.name)
            if date_match:
                target_dates.add(date_match.group(1))
        return mapping, deduplicated

    def _merge_markdown_tree(self, category: str) -> tuple[dict[str, str], list[str]]:
        source_root = self.source_dir / category
        target_root = self.target_dir / category
        mapping: dict[str, str] = {}
        deduplicated: list[str] = []
        namespace = f"__from_{self.source}"
        for source_path in _files(source_root):
            relative = source_path.relative_to(source_root)
            destination, is_duplicate = self._copy_collision_safe(source_path, target_root / relative, namespace)
            source_key = f"{category}/{relative.as_posix()}"
            target_key = f"{category}/{destination.relative_to(target_root).as_posix()}"
            mapping[source_key] = target_key
            if is_duplicate:
                deduplicated.append(source_key)
        return mapping, deduplicated

    def _merge_attachments(self) -> dict[str, str]:
        """Copy source attachments with basename-safe, resumable renames."""
        source_root = self.source_dir / "attachments"
        target_root = self.target_dir / "attachments"
        mapping: dict[str, str] = {}
        namespace = f"__from_{self.source}"
        for source_path in _files(source_root):
            relative = source_path.relative_to(source_root)
            desired = target_root / relative
            destination, _is_duplicate = self._copy_collision_safe(source_path, desired, namespace)
            source_key = f"attachments/{relative.as_posix()}"
            target_key = f"attachments/{destination.relative_to(target_root).as_posix()}"
            mapping[source_key] = target_key
        return mapping

    def rewrite_refs(self, journal: MergeJournal) -> dict[str, Any]:
        """Rewrite external references through durable, idempotent substeps."""
        from .content_refs import (
            plan_inbox,
            rewrite_inbox,
            rewrite_inbox_task_references,
            rewrite_memory_references,
        )
        from .external_refs import ExternalRefsRewriter
        from .task_refs import build_task_id_mapping, rewrite_task_references

        external = ExternalRefsRewriter(self.data_dir, self.source, self.target)
        self._run_rewrite_substep(
            journal,
            "organization",
            lambda: external.rewrite_organization(external.plan_organization()),
        )

        def rewrite_messaging() -> dict[str, Any]:
            return {
                **external.rewrite_messaging(),
                "credential_disable_candidates": external.discover_credential_candidates(),
            }

        self._run_rewrite_substep(journal, "messaging", rewrite_messaging)

        memory_artifacts = journal.phase_artifacts(MergePhase.MERGE_MEMORY)
        file_mapping = memory_artifacts.get("file_mapping", {})
        episode_mapping = memory_artifacts.get("episode_mapping", {})
        attachment_mapping = memory_artifacts.get("attachment_mapping", {})
        facts_appended_ids = memory_artifacts.get("facts_appended_ids", [])
        for label, value in (
            ("file_mapping", file_mapping),
            ("episode_mapping", episode_mapping),
            ("attachment_mapping", attachment_mapping),
        ):
            if not isinstance(value, dict):
                raise AnimaMergeError(f"Invalid {label} in MERGE_MEMORY journal artifacts")
        if not isinstance(facts_appended_ids, list) or not all(
            isinstance(value, str) for value in facts_appended_ids
        ):
            raise AnimaMergeError("Invalid facts_appended_ids in MERGE_MEMORY journal artifacts")

        self._run_rewrite_substep(
            journal,
            "memory_references",
            lambda: rewrite_memory_references(
                self.target_dir,
                source=self.source,
                target=self.target,
                file_mapping=file_mapping,
                episode_mapping=episode_mapping,
                attachment_mapping=attachment_mapping,
                fact_ids_to_rewrite=facts_appended_ids,
            ),
        )
        inbox_plan = self._run_rewrite_substep(
            journal,
            "inbox_mapping",
            lambda: plan_inbox(self.data_dir, self.source, self.target),
        )
        inbox = self._run_rewrite_substep(
            journal,
            "inbox",
            lambda: rewrite_inbox(
                self.data_dir,
                self.source,
                self.target,
                inbox_plan,
            ),
        )
        task_plan = self._run_rewrite_substep(
            journal,
            "task_id_mapping",
            lambda: {
                "task_id_mapping": build_task_id_mapping(
                    self.data_dir,
                    self.source,
                    self.target,
                )
            },
        )
        task_mapping = task_plan.get("task_id_mapping", {})
        if not isinstance(task_mapping, dict):
            raise AnimaMergeError("Invalid task_id_mapping in REWRITE_REFS journal artifacts")
        message_mapping = inbox.get("message_mapping", [])
        if not isinstance(message_mapping, list):
            raise AnimaMergeError("Invalid message_mapping in REWRITE_REFS journal artifacts")
        self._run_rewrite_substep(
            journal,
            "inbox_task_references",
            lambda: rewrite_inbox_task_references(
                self.data_dir,
                self.target,
                message_mapping,
                task_mapping,
            ),
        )
        self._run_rewrite_substep(
            journal,
            "task_references",
            lambda: rewrite_task_references(
                self.data_dir,
                self.source,
                self.target,
                task_mapping,
            ),
        )
        def rewrite_ancillary_state() -> dict[str, Any]:
            artifacts = external.rewrite_ancillary_state(
                wake_target=bool(inbox.get("messages_moved", 0)),
            )
            if self._server_running():
                artifacts["live_runtime"] = self._sync_live_reference_state()
            return artifacts

        self._run_rewrite_substep(journal, "ancillary_state", rewrite_ancillary_state)

        substeps = journal.data.get("phases", {}).get(MergePhase.REWRITE_REFS.value, {}).get("substeps", {})
        return {
            "rewrite_substeps": {
                name: {
                    "status": record.get("status"),
                    "artifacts": record.get("artifacts", {}),
                    **({"reason": record["reason"]} if "reason" in record else {}),
                }
                for name, record in substeps.items()
                if isinstance(record, dict)
            },
            "task_id_mapping": dict(sorted(task_mapping.items())),
            "messages_moved": int(inbox.get("messages_moved", 0)),
        }

    @staticmethod
    def _run_rewrite_substep(journal: MergeJournal, name: str, action: Any) -> dict[str, Any]:
        phase = MergePhase.REWRITE_REFS
        if journal.is_substep_completed(phase, name):
            return journal.substep_artifacts(phase, name)
        journal.start_substep(phase, name)
        try:
            artifacts = action() or {}
        except Exception as exc:
            journal.fail_substep(phase, name, f"{type(exc).__name__}: {exc}")
            if isinstance(exc, AnimaMergeError):
                raise
            raise AnimaMergeError(f"REWRITE_REFS substep '{name}' failed: {exc}") from exc
        journal.complete_substep(phase, name, artifacts)
        return artifacts

    def _sync_live_reference_state(self) -> dict[str, Any]:
        """Ask a running gateway to update caches that can overwrite disk."""
        import requests

        url = f"{self.gateway_url}/api/system/anima-merge/rewrite-runtime-refs"
        try:
            response = requests.post(
                url,
                json={"source": self.source, "target": self.target},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise AnimaMergeError(f"Live reference-state synchronization failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise AnimaMergeError("Live reference-state synchronization returned invalid JSON")
        if payload.get("config_reloaded") is not True:
            raise AnimaMergeError("Live reference-state synchronization did not reload configuration")
        return payload

    def rebuild_indexes(self, journal: MergeJournal) -> dict[str, Any]:
        """Rebuild target-derived indexes with resumable journal substeps."""
        self._run_rebuild_substep(journal, "vectordb", self._rebuild_vectordb)
        self._run_rebuild_substep(journal, "entities", self._rebuild_entities)
        self._run_rebuild_substep(journal, "bm25", self._rebuild_bm25)
        self._run_rebuild_substep(journal, "graph_cache", self._rebuild_graph_cache)

        if resolve_backend_type(self.target_dir) == "neo4j":
            self._run_rebuild_substep(journal, "neo4j", self._rebuild_neo4j)
        elif not journal.is_substep_completed(MergePhase.REBUILD_INDEXES, "neo4j"):
            journal.skip_substep(
                MergePhase.REBUILD_INDEXES,
                "neo4j",
                "Target memory backend is not Neo4j",
            )

        substeps = journal.data.get("phases", {}).get(MergePhase.REBUILD_INDEXES.value, {}).get("substeps", {})
        return {
            "rebuild_target": self.target,
            "rebuild_substeps": {
                name: {
                    "status": record.get("status"),
                    "artifacts": record.get("artifacts", {}),
                    **({"reason": record["reason"]} if "reason" in record else {}),
                }
                for name, record in substeps.items()
                if isinstance(record, dict)
            },
        }

    @staticmethod
    def _run_rebuild_substep(journal: MergeJournal, name: str, action: Any) -> dict[str, Any]:
        phase = MergePhase.REBUILD_INDEXES
        if journal.is_substep_completed(phase, name):
            return journal.substep_artifacts(phase, name)
        journal.start_substep(phase, name)
        try:
            artifacts = action() or {}
        except Exception as exc:
            journal.fail_substep(phase, name, f"{type(exc).__name__}: {exc}")
            raise
        journal.complete_substep(phase, name, artifacts)
        return artifacts

    def _rebuild_vectordb(self) -> dict[str, Any]:
        from core.memory.rag.repair_rebuild import atomic_rebuild_vectordb

        chunks, archive = atomic_rebuild_vectordb(
            self.target,
            include_shared=False,
            anima_dir=self.target_dir,
        )
        return {
            "chunks_indexed": chunks,
            "archived_vectordb": str(archive) if archive is not None else None,
        }

    def _target_vector_components(self) -> tuple[Any, Any]:
        from core.memory.rag import MemoryIndexer
        from core.memory.rag.singleton import get_vector_store

        vector_store = get_vector_store(self.target)
        if vector_store is None:
            raise AnimaMergeError(f"Vector store unavailable for target '{self.target}'")
        return vector_store, MemoryIndexer(vector_store, self.target, self.target_dir)

    def _rebuild_entities(self) -> dict[str, Any]:
        from core.memory.entity_index import rebuild_entity_registry, sync_entity_collection

        registry = rebuild_entity_registry(self.target_dir)
        vector_store, _indexer = self._target_vector_components()
        if not sync_entity_collection(self.target_dir, registry=registry, vector_store=vector_store):
            raise AnimaMergeError(f"Failed to rebuild entity collection for target '{self.target}'")
        entities = registry.get("entities", {})
        return {"entities": len(entities) if isinstance(entities, dict) else 0}

    def _rebuild_bm25(self) -> dict[str, Any]:
        from core.memory.bm25 import rebuild_longterm_bm25_index

        result = rebuild_longterm_bm25_index(self.target_dir)
        return {"documents": result.documents, "path": str(result.path)}

    def _rebuild_graph_cache(self) -> dict[str, Any]:
        from core.memory.rag.graph import rebuild_graph_cache

        vector_store, indexer = self._target_vector_components()
        rebuilt = rebuild_graph_cache(
            self.target,
            self.target_dir,
            vector_store,
            indexer,
        )
        return {"rebuilt": rebuilt}

    def _rebuild_neo4j(self) -> dict[str, Any]:
        return asyncio.run(self._rebuild_neo4j_async())

    async def _rebuild_neo4j_async(self) -> dict[str, Any]:
        """Reset only the target group and ingest all merged canonical inputs."""
        from core.memory.backend.registry import get_backend

        backend = get_backend("neo4j", self.target_dir)
        files = 0
        facts = 0
        chunks = 0
        try:
            await backend.reset()
            for memory_type, pattern in (
                ("knowledge", "*.md"),
                ("episodes", "*.md"),
                ("procedures", "*.md"),
                ("skills", "SKILL.md"),
            ):
                root = self.target_dir / memory_type
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob(pattern)):
                    if path.is_file():
                        chunks += await backend.ingest_file(path)
                        files += 1

            for record in iter_fact_records(self.target_dir, include_expired=True):
                chunks += await backend.ingest_text(
                    record.text,
                    source=f"fact:{record.fact_id}",
                    metadata={
                        "stable_key": f"fact:{record.fact_id}",
                        "fact_id": record.fact_id,
                        "valid_at": record.valid_at,
                        "source_episode": record.source_episode,
                    },
                )
                facts += 1

            conversation_path = self.target_dir / "state" / "conversation.json"
            if self._has_conversation_summary(conversation_path):
                conversation = _read_json(conversation_path)
                summary = str(conversation.get("compressed_summary", "")).strip()
                chunks += await backend.ingest_text(
                    summary,
                    source="conversation_summary",
                    metadata={"stable_key": f"conversation_summary:{self.target}"},
                )
                files += 1
        finally:
            await backend.close()
        return {"files_ingested": files, "facts_ingested": facts, "chunks_created": chunks}

    # ── VERIFY and TOMBSTONE ──────────────────────────────────

    def verify(self, journal: MergeJournal) -> dict[str, Any]:
        """Verify migrated memory, active references, and the live target."""

        probe_result = self._verify_memory_probes(journal)
        from .verification import source_reference_report

        try:
            references = source_reference_report(self.data_dir, self.source)
        except ValueError as exc:
            raise AnimaMergeError(f"Reference-integrity scan failed: {exc}") from exc
        residual = references["residual_references"]
        if residual:
            raise AnimaMergeError(
                "VERIFY found residual source references:\n- " + "\n- ".join(residual)
            )
        smoke = self._smoke_check_target()
        return {
            "memory_probes": probe_result,
            "reference_integrity": references,
            "smoke_check": smoke,
        }

    def _verify_memory_probes(self, journal: MergeJournal) -> dict[str, Any]:
        probes = self._build_memory_probes(journal)
        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for probe in probes:
            query = str(probe.pop("query"))
            if not query:
                results.append({**probe, "status": "failed", "hits": 0})
                failures.append(f"{probe['category']}:{probe['source']} (empty probe content)")
                continue
            if probe["category"] == "entities":
                expected_entity = str(probe.pop("expected_entity"))
                matched = self._verify_entity_probe(query, expected_entity)
                hits = int(matched)
            else:
                hits_found = self._search_memory_probe(query, str(probe["scope"]))
                matched = self._probe_results_match(hits_found, probe)
                hits = len(hits_found)
            status = "passed" if matched else "failed"
            result = {**probe, "status": status, "hits": hits}
            results.append(result)
            if not matched:
                failures.append(f"{probe['category']}:{probe['source']}")
        if failures:
            raise AnimaMergeError(
                "VERIFY memory probes could not find migrated source content:\n- "
                + "\n- ".join(failures)
            )
        by_category: dict[str, dict[str, int]] = {}
        for result in results:
            bucket = by_category.setdefault(result["category"], {"planned": 0, "passed": 0, "skipped": 0})
            bucket["planned"] += 1
            if result["status"] == "passed":
                bucket["passed"] += 1
            elif result["status"].startswith("skipped"):
                bucket["skipped"] += 1
        return {
            "planned": len(results),
            "passed": sum(item["status"] == "passed" for item in results),
            "skipped": sum(item["status"].startswith("skipped") for item in results),
            "categories": by_category,
            "results": results,
        }

    def _build_memory_probes(self, journal: MergeJournal) -> list[dict[str, Any]]:
        from core.memory.entity_index import normalize_entity_key
        from core.memory.facts import fact_entity_names

        from .verification import probe_query

        artifacts = journal.phase_artifacts(MergePhase.MERGE_MEMORY)
        file_mapping = artifacts.get("file_mapping", {})
        deduplicated = set(artifacts.get("deduplicated_files", []))
        probes: list[dict[str, Any]] = []
        if isinstance(file_mapping, dict):
            for category in ("knowledge", "episodes", "procedures"):
                selected = [
                    (str(source), str(target))
                    for source, target in sorted(file_mapping.items())
                    if str(source).startswith(f"{category}/")
                    and source not in deduplicated
                    # indexerは*.mdのみ索引する(facts/skillsを除く)。非mdはprobe不能
                    and str(source).lower().endswith(".md")
                ][:3]
                for source, target in selected:
                    source_path = self.source_dir / source
                    probes.append(
                        {
                            "category": category,
                            "scope": category,
                            "source": source,
                            "target": target,
                            "query": probe_query(self._read_probe_text(source_path)),
                        }
                    )

        skill_mapping = artifacts.get("skill_mapping", {})
        if isinstance(skill_mapping, dict):
            for source, target in list(sorted(skill_mapping.items()))[:3]:
                source_file = self.source_dir / str(source) / "SKILL.md"
                probes.append(
                    {
                        "category": "skills",
                        "scope": "skills",
                        "source": f"{source}/SKILL.md",
                        "target": f"{target}/SKILL.md",
                        "query": probe_query(self._read_probe_text(source_file)),
                    }
                )

        appended_ids = artifacts.get("facts_appended_ids", [])
        appended = {str(value) for value in appended_ids if isinstance(value, str)}
        source_facts = [
            record
            for record in iter_fact_records(self.source_dir, include_expired=True)
            if record.fact_id in appended
        ]
        for record in source_facts[:3]:
            probes.append(
                {
                    "category": "facts",
                    "scope": "facts",
                    "source": f"fact:{record.fact_id}",
                    "target": f"fact:{record.fact_id}",
                    "expected_fact_id": record.fact_id,
                    "query": probe_query(record.text),
                }
            )

        conversation_paths = artifacts.get("conversation_episodes", [])
        if isinstance(conversation_paths, list):
            selected = [
                str(path)
                for path in conversation_paths
                if isinstance(path, str) and "merged_conversation_" in Path(path).name
            ][:3]
            for target in selected:
                probes.append(
                    {
                        "category": "conversation_summary",
                        "scope": "episodes",
                        "source": "state/conversation.json:compressed_summary",
                        "target": target,
                        "query": probe_query(self._read_probe_text(self.target_dir / target)),
                    }
                )

        entities: dict[str, str] = {}
        for record in source_facts:
            for name in fact_entity_names(record):
                key = normalize_entity_key(name)
                if key:
                    entities.setdefault(key, name)
        for key, name in list(sorted(entities.items()))[:3]:
            probe_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            probes.append(
                {
                    "category": "entities",
                    "scope": "entities",
                    "source": f"entity:{probe_id}",
                    "target": f"entity:{probe_id}",
                    "expected_entity": key,
                    "query": name,
                }
            )
        return probes

    @staticmethod
    def _read_probe_text(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(16_384)
        except OSError:
            return ""

    def _search_memory_probe(self, query: str, scope: str) -> list[dict[str, Any]]:
        from core.memory.rag_search import RAGMemorySearch

        search = RAGMemorySearch(
            self.target_dir,
            self.data_dir / "common_knowledge",
            self.data_dir / "common_skills",
        )
        return search.search_memory_text(
            query,
            scope=scope,
            knowledge_dir=self.target_dir / "knowledge",
            episodes_dir=self.target_dir / "episodes",
            procedures_dir=self.target_dir / "procedures",
            common_knowledge_dir=self.data_dir / "common_knowledge",
            result_limit=20,
        )

    def _verify_entity_probe(self, query: str, expected: str) -> bool:
        from core.memory.entity_index import match_query_entities

        return expected in match_query_entities(self.target_dir, query)

    @staticmethod
    def _probe_results_match(results: list[dict[str, Any]], probe: dict[str, Any]) -> bool:
        expected_fact = probe.get("expected_fact_id")
        expected_path = str(probe.get("target", ""))
        for result in results:
            if expected_fact and result.get("fact_id") == expected_fact:
                return True
            source_file = str(result.get("source_file", ""))
            if expected_path and (source_file == expected_path or source_file.endswith(f"/{expected_path}")):
                return True
        return False

    def _smoke_check_target(self) -> dict[str, Any]:
        if not self._server_running():
            return {
                "status": "skipped_offline",
                "manual_required": True,
                "target_enabled": False,
                "process_started": False,
            }
        import requests

        try:
            response = requests.post(f"{self.gateway_url}/api/animas/{self.target}/enable", timeout=10)
            response.raise_for_status()
        except Exception as exc:
            raise AnimaMergeError(f"Target smoke-check enable failed: {exc}") from exc
        pid_path = self.data_dir / "run" / "animas" / f"{self.target}.pid"
        deadline = time.monotonic() + 30.0
        while not self._pid_path_alive(pid_path):
            if time.monotonic() >= deadline:
                raise AnimaMergeError(f"Target smoke-check process did not start: {pid_path}")
            time.sleep(0.1)
        return {
            "status": "passed",
            "manual_required": False,
            "target_enabled": True,
            "process_started": True,
        }

    def tombstone(self, journal: MergeJournal, *, rollback_window_days: int = 7) -> dict[str, Any]:
        """Disable source while retaining its directory and config registration."""

        if rollback_window_days < 1:
            raise AnimaMergeError("rollback_window_days must be at least 1")
        status_path = self.source_dir / "status.json"
        status = _read_json(status_path)
        already_disabled = status.get("enabled") is False
        if not already_disabled:
            status["enabled"] = False
            atomic_write_text(
                status_path,
                json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        tombstoned_at = now_local()
        archive_plan = journal.phase_artifacts(MergePhase.MERGE_MEMORY).get("archive_plan", [])
        return {
            "source_enabled": False,
            "already_disabled": already_disabled,
            "source_directory_retained": self.source_dir.is_dir(),
            "source_config_retained": self._source_config_present(),
            "tombstoned_at": tombstoned_at.isoformat(),
            "rollback_window_days": rollback_window_days,
            "rollback_deadline": (tombstoned_at + timedelta(days=rollback_window_days)).isoformat(),
            "archive_plan": archive_plan if isinstance(archive_plan, list) else [],
        }

    def _source_config_present(self) -> bool:
        config_path = self.data_dir / "config.json"
        if not config_path.is_file():
            return False
        config = _read_json(config_path)
        return isinstance(config.get("animas"), dict) and self.source in config["animas"]

    def _map_source_episode(self, value: str, mapping: dict[str, str]) -> str:
        if value in mapping:
            return mapping[value]
        prefixed = f"episodes/{value}"
        if prefixed in mapping:
            mapped = mapping[prefixed]
            return mapped.removeprefix("episodes/")
        return value

    def _merge_facts(self, episode_mapping: dict[str, str]) -> dict[str, Any]:
        records: list[FactRecord] = []
        id_mapping: dict[str, str] = {}
        for record in iter_fact_records(self.source_dir, include_expired=True):
            mapped_episode = self._map_source_episode(record.source_episode, episode_mapping)
            data = record.to_dict()
            data["source_episode"] = mapped_episode
            mapped = FactRecord.from_dict(data)
            records.append(mapped)
            id_mapping[record.fact_id] = mapped.fact_id
        appended = append_fact_records(self.target_dir, records)
        return {
            "read": len(records),
            "appended": len(appended),
            "appended_ids": [record.fact_id for record in appended],
            "id_mapping": id_mapping,
        }

    def _merge_skills(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        source_skills = self.source_dir / "skills"
        target_skills = self.target_dir / "skills"
        if not source_skills.is_dir():
            return mapping
        for source_path in sorted(source_skills.iterdir()):
            if not source_path.is_dir():
                continue
            if source_path.name == "quarantine":
                for child in sorted(source_path.iterdir()):
                    if child.is_dir():
                        destination = self._copy_skill_directory(
                            child,
                            target_skills / "quarantine" / child.name,
                        )
                        mapping[f"skills/quarantine/{child.name}"] = (
                            f"skills/quarantine/{destination.name}"
                        )
                continue
            destination = self._copy_skill_directory(source_path, target_skills / source_path.name)
            mapping[f"skills/{source_path.name}"] = f"skills/{destination.name}"
        return mapping

    def _copy_skill_directory(self, source_path: Path, desired: Path) -> Path:
        if not desired.exists():
            shutil.copytree(source_path, desired, copy_function=shutil.copy2)
            return desired
        if self._directory_fingerprint(source_path) == self._directory_fingerprint(desired):
            return desired
        candidate = desired.with_name(f"{desired.name}__from_{self.source}")
        if candidate.exists() and self._directory_fingerprint(source_path) == self._directory_fingerprint(candidate):
            return candidate
        index = 2
        while candidate.exists():
            candidate = desired.with_name(f"{desired.name}__from_{self.source}_{index}")
            if self._directory_fingerprint(source_path) == self._directory_fingerprint(candidate):
                return candidate
            index += 1
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, candidate, copy_function=shutil.copy2)
        return candidate

    @staticmethod
    def _directory_fingerprint(root: Path) -> list[tuple[str, str]]:
        if not root.is_dir():
            return []
        return [(_safe_relative(path, root), _sha256(path)) for path in _files(root)]

    def _recover_abandoned_memory(self) -> list[str]:
        recovered: list[str] = []
        for journal_path in self._streaming_journals(self.source_dir):
            body = self._streaming_journal_markdown(journal_path)
            if body:
                recovered.append(self._write_generated_episode("streaming", journal_path, body))
        shortterm = self.source_dir / "shortterm"
        if shortterm.is_dir():
            for state_path in sorted(shortterm.rglob("session_state.json")):
                body = self._shortterm_markdown(state_path)
                if body:
                    recovered.append(self._write_generated_episode("shortterm", state_path, body))
        current_state = self.source_dir / "state" / "current_state.md"
        if current_state.is_file():
            try:
                content = current_state.read_text(encoding="utf-8")
            except OSError:
                content = ""
            if content.strip():
                body = f"# Abandoned current state from {self.source}\n\n{content.strip()}\n"
                recovered.append(self._write_generated_episode("current_state", current_state, body))
        return recovered

    def _streaming_journal_markdown(self, path: Path) -> str:
        metadata = {"trigger": "", "from": "", "session_id": "", "started_at": "", "last_event_at": ""}
        text_parts: list[str] = []
        tools: list[str] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    timestamp = str(entry.get("ts", ""))
                    if timestamp:
                        metadata["last_event_at"] = timestamp
                    event = entry.get("ev")
                    if event == "start":
                        metadata.update(
                            {
                                "trigger": str(entry.get("trigger", "")),
                                "from": str(entry.get("from", "")),
                                "session_id": str(entry.get("session_id", "")),
                                "started_at": timestamp,
                            }
                        )
                    elif event == "text":
                        text_parts.append(str(entry.get("t", "")))
                    elif event in {"tool_start", "tool_end"}:
                        tools.append(f"- {event}: {entry.get('tool', '')}")
        except OSError:
            return ""
        text = "".join(text_parts).strip()
        if not text and not tools:
            return ""
        lines = [f"# Recovered streaming journal from {self.source}"]
        lines.extend(f"- {key}: {value}" for key, value in metadata.items())
        if tools:
            lines.extend(["", "## Tool events", *tools])
        if text:
            lines.extend(["", "## Recovered text", "", text])
        return "\n".join(lines) + "\n"

    def _shortterm_markdown(self, path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if not isinstance(data, dict):
            return ""
        parts: list[str] = []
        for label, key in (
            ("Original request", "original_prompt"),
            ("Work so far", "accumulated_response"),
            ("Notes", "notes"),
        ):
            value = str(data.get(key, "") or "").strip()
            if value:
                parts.extend([f"## {label}", "", value, ""])
        if not parts:
            return ""
        return f"# Recovered short-term session from {self.source}\n\n" + "\n".join(parts).rstrip() + "\n"

    def _write_generated_episode(self, kind: str, source_path: Path, content: str) -> str:
        identity = f"{_safe_relative(source_path, self.source_dir)}\0{content}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:12]
        destination = self.target_dir / "episodes" / f"merged_{kind}_from_{self.source}_{digest}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(content, encoding="utf-8")
        elif destination.read_text(encoding="utf-8") != content:
            raise AnimaMergeError(f"Generated episode collision: {destination}")
        return _safe_relative(destination, self.target_dir)

    def _merge_conversation_history(self) -> list[str]:
        created: list[str] = []
        conversation = self.source_dir / "state" / "conversation.json"
        if conversation.is_file():
            try:
                data = json.loads(conversation.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            summary = data.get("compressed_summary", "") if isinstance(data, dict) else ""
            if isinstance(summary, str) and summary.strip():
                content = f"# Compressed conversation summary from {self.source}\n\n{summary.strip()}\n"
                created.append(self._write_generated_episode("conversation", conversation, content))

        transcripts = self.source_dir / "transcripts"
        for transcript in _files(transcripts):
            entries: list[str] = []
            try:
                with transcript.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "unknown"))
                        timestamp = str(item.get("ts", ""))
                        content = str(item.get("content", "")).strip()
                        if content:
                            entries.extend([f"## {timestamp} {role}".strip(), "", content, ""])
            except OSError:
                continue
            if entries:
                body = f"# Transcript from {self.source}: {transcript.name}\n\n" + "\n".join(entries).rstrip() + "\n"
                created.append(self._write_generated_episode("transcript", transcript, body))
        return created

    def _archive_plan(self) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        destination_root = self.target_dir / "archive" / f"merged_from_{self.source}"
        for category in _ARCHIVE_ONLY:
            source_root = self.source_dir / category
            if not source_root.exists():
                continue
            files = list(_files(source_root))
            plan.append(
                {
                    "source": f"animas/{self.source}/{category}",
                    "planned_destination": str(destination_root / category),
                    "files": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                    "action": "deferred_to_TOMBSTONE",
                }
            )
        return plan

    def _skill_state_provenance(self, mapping: dict[str, str]) -> dict[str, Any]:
        state = self.source_dir / "state"
        candidates = (
            state / "skill_usage.jsonl",
            state / "skill_promotion.jsonl",
            state / "skill_curator.jsonl",
            state / "skill_hub_lock.jsonl",
        )
        files = [
            {"path": _safe_relative(path, self.source_dir), "sha256": _sha256(path)}
            for path in candidates
            if path.is_file()
        ]
        curator = state / "skill_curator"
        files.extend(
            {"path": _safe_relative(path, self.source_dir), "sha256": _sha256(path)} for path in _files(curator)
        )
        return {"source_anima": self.source, "rename_mapping": mapping, "state_files": files, "activated": False}
