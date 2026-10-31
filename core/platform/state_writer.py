from __future__ import annotations

from core.anima.workspace_files import BOOTSTRAP_FILE, HEARTBEAT_FILE

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-writer persistence for Anima runtime-managed state.

Task runners send semantic state operations over their IPC v2 connection. The
Anima main resolves every destination from its own ``anima_dir``; the wire
protocol never accepts arbitrary filesystem paths.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from core.platform.atomic_io import atomic_write_json, atomic_write_text
from core.platform.process_role import get_process_role_env
from core.time_utils import now_iso, now_local, today_local

logger = logging.getLogger("animaworks.state_writer")


def is_task_runner_process() -> bool:
    """Return whether the inherited process-role environment marks a task runner."""
    return get_process_role_env() == "task_runner"


def _is_task_runner_process() -> bool:
    return is_task_runner_process()


_STATE_WRITE_CHUNK_BYTES = 512 * 1024
_STATE_WRITE_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_PROMPT_LOG_RETENTION_DAYS = 3
_INBOX_OVERFLOW_MAX_AGE_DAYS = 7
_INBOX_OVERFLOW_MAX_FILES = 500
_SHORTTERM_ARCHIVE_MAX_FILES = 100
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,36}$")
_SAFE_SESSION_TYPES = frozenset({"chat", "heartbeat", "cron", "task", "background", "inbox", "task_exec"})
_SAFE_SESSION_ENGINES = frozenset({"agent_sdk", "codex", "cursor", "grok", "agy"})
_PROMPT_ROTATION_DATE: str | None = None


class StateWriterError(RuntimeError):
    """A state write could not be safely completed."""


class StateWriteLink(Protocol):
    """Subset of the task runner IPC link required by ``IpcStateWriter``."""

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def send_unanswered_request(self, method: str, params: dict[str, Any]) -> None: ...


class LocalStateWriter:
    """Anima-main implementation of the semantic state-write operations."""

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = Path(anima_dir)
        self._locks_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
            weakref.WeakKeyDictionary()
        )
        self._thread_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _assert_main_process(self) -> None:
        if _is_task_runner_process():
            raise StateWriterError("LocalStateWriter cannot be used by a task_runner process")

    def _lock_for(self, key: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._locks_guard:
            locks = self._locks_by_loop.setdefault(loop, {})
            return locks.setdefault(key, asyncio.Lock())

    def _thread_lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._thread_locks.setdefault(key, threading.Lock())

    async def _run(self, key: str, function: Callable[[], Any]) -> Any:
        self._assert_main_process()
        async with self._lock_for(key):

            def _locked_call() -> Any:
                with self._thread_lock_for(key):
                    return function()

            return await asyncio.to_thread(_locked_call)

    @staticmethod
    def _component(value: str, label: str = "identifier") -> str:
        if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
            raise ValueError(f"invalid {label}: {value!r}")
        return value

    @staticmethod
    def _thread_id(value: str) -> str:
        if value == "default":
            return value
        if not isinstance(value, str) or not _SAFE_THREAD_ID.fullmatch(value):
            raise ValueError(f"invalid thread_id: {value!r}")
        return value

    @staticmethod
    def _session_type(value: str) -> str:
        if not isinstance(value, str) or value not in _SAFE_SESSION_TYPES:
            raise ValueError(f"invalid session_type: {value!r}")
        return value

    def _conversation_path(self, thread_id: str) -> Path:
        thread_id = self._thread_id(thread_id)
        if thread_id == "default":
            return self.anima_dir / "state" / "conversation.json"
        return self.anima_dir / "state" / "conversations" / f"{thread_id}.json"

    def _shortterm_dir(self, session_type: str, thread_id: str) -> Path:
        session_type = self._session_type(session_type)
        thread_id = self._thread_id(thread_id)
        base = self.anima_dir / "shortterm" / session_type
        return base if thread_id == "default" else base / thread_id

    async def save_conversation(self, thread_id: str, state: dict[str, Any]) -> None:
        path = self._conversation_path(thread_id)
        text = json.dumps(state, ensure_ascii=False, indent=2)
        await self._run(str(path), lambda: atomic_write_text(path, text))

    async def clear_conversation(self, thread_id: str = "default") -> None:
        path = self._conversation_path(thread_id)
        await self._run(str(path), lambda: path.unlink(missing_ok=True))

    async def append_transcript(self, entry: dict[str, Any]) -> None:
        path = self.anima_dir / "transcripts" / f"{today_local().isoformat()}.jsonl"
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

        await self._run(str(path), _append)

    async def save_shortterm(
        self,
        session_type: str,
        thread_id: str,
        state: dict[str, Any],
        markdown: str,
    ) -> Path:
        shortterm_dir = self._shortterm_dir(session_type, thread_id)
        archive_dir = shortterm_dir / "archive"
        json_path = shortterm_dir / "session_state.json"
        md_path = shortterm_dir / "session_state.md"

        def _save() -> Path:
            shortterm_dir.mkdir(parents=True, exist_ok=True)
            archive_dir.mkdir(parents=True, exist_ok=True)
            self._archive_shortterm_files(shortterm_dir, archive_dir)
            try:
                atomic_write_json(json_path, state, indent=2, ensure_ascii=False, trailing_newline=False)
            except OSError as exc:
                from core.exceptions import MemoryWriteError

                raise MemoryWriteError(f"Short-term memory save failed: {exc}") from exc
            try:
                md_path.write_text(markdown, encoding="utf-8")
            except OSError:
                logger.warning("Failed to write short-term memory markdown to %s", md_path, exc_info=True)
            return json_path

        return await self._run(str(shortterm_dir), _save)

    @staticmethod
    def _archive_shortterm_files(shortterm_dir: Path, archive_dir: Path) -> None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = now_local().strftime("%Y%m%d_%H%M%S")
        for suffix in (".json", ".md"):
            source = shortterm_dir / f"session_state{suffix}"
            if source.exists():
                source.rename(archive_dir / f"{ts}{suffix}")
        files = sorted(archive_dir.iterdir(), key=lambda path: path.name)
        excess = len(files) - _SHORTTERM_ARCHIVE_MAX_FILES
        if excess > 0:
            for path in files[:excess]:
                path.unlink(missing_ok=True)

    async def archive_shortterm(self, session_type: str, thread_id: str = "default") -> None:
        shortterm_dir = self._shortterm_dir(session_type, thread_id)
        archive_dir = shortterm_dir / "archive"

        def _archive() -> None:
            if shortterm_dir.exists():
                self._archive_shortterm_files(shortterm_dir, archive_dir)

        await self._run(str(shortterm_dir), _archive)

    async def prune_shortterm_archive(
        self,
        session_type: str,
        thread_id: str = "default",
        max_files: int = _SHORTTERM_ARCHIVE_MAX_FILES,
    ) -> None:
        archive_dir = self._shortterm_dir(session_type, thread_id) / "archive"

        def _prune() -> None:
            if not archive_dir.exists():
                return
            files = sorted(archive_dir.iterdir(), key=lambda path: path.name)
            for path in files[: max(len(files) - max_files, 0)]:
                path.unlink(missing_ok=True)

        await self._run(str(archive_dir), _prune)

    async def migrate_legacy_shortterm(self, session_type: str, thread_id: str = "default") -> None:
        self._assert_main_process()
        shortterm_dir = self._shortterm_dir(session_type, thread_id)
        if session_type != "chat" or thread_id != "default":
            return
        legacy_dir = self.anima_dir / "shortterm"
        archive_dir = shortterm_dir / "archive"

        def _migrate() -> None:
            shortterm_dir.mkdir(parents=True, exist_ok=True)
            for name in ("session_state.json", "session_state.md", "stream_checkpoint.json"):
                source = legacy_dir / name
                destination = shortterm_dir / name
                if source.exists() and not destination.exists():
                    source.rename(destination)
                    logger.info("Migrated legacy shortterm file: %s -> %s", source, destination)
            old_archive = legacy_dir / "archive"
            if old_archive.exists() and not archive_dir.exists():
                old_archive.rename(archive_dir)
            elif old_archive.exists() and archive_dir.exists():
                for source in old_archive.iterdir():
                    destination = archive_dir / source.name
                    if not destination.exists():
                        source.rename(destination)
                if not any(old_archive.iterdir()):
                    old_archive.rmdir()

        await self._run(str(shortterm_dir), _migrate)

    async def save_stream_checkpoint(
        self,
        session_type: str,
        thread_id: str,
        checkpoint: dict[str, Any],
    ) -> Path:
        shortterm_dir = self._shortterm_dir(session_type, thread_id)
        path = shortterm_dir / "stream_checkpoint.json"

        def _save() -> Path:
            shortterm_dir.mkdir(parents=True, exist_ok=True)
            try:
                atomic_write_json(path, checkpoint, indent=2, ensure_ascii=False, trailing_newline=False)
            except OSError:
                logger.warning("Failed to write stream checkpoint to %s", path, exc_info=True)
            return path

        return await self._run(str(path), _save)

    async def clear_stream_checkpoint(self, session_type: str, thread_id: str = "default") -> None:
        path = self._shortterm_dir(session_type, thread_id) / "stream_checkpoint.json"
        await self._run(str(path), lambda: path.unlink(missing_ok=True))

    @staticmethod
    def _session_path(engine: str, anima_dir: Path, session_type: str, thread_id: str) -> Path:
        if engine not in _SAFE_SESSION_ENGINES:
            raise ValueError(f"invalid session engine: {engine!r}")
        LocalStateWriter._session_type(session_type)
        thread_id = LocalStateWriter._thread_id(thread_id)
        if engine == "agent_sdk":
            suffix = f"_{thread_id}" if thread_id != "default" else ""
            return anima_dir / "state" / f"current_session_{session_type}{suffix}.json"

        base = anima_dir / "shortterm" / session_type
        if engine == "codex":
            directory = base if thread_id == "default" else base / thread_id
            return directory / "codex_thread_id.txt"
        if engine == "cursor":
            directory = base if thread_id == "default" else base / thread_id
            return directory / "cursor_chat_id.txt"
        if engine == "agy":
            return base / thread_id / "agy_session_id.txt"
        return base / thread_id / "grok_session_id.txt"

    async def save_session_record(
        self,
        engine: str,
        session_type: str,
        thread_id: str,
        record: dict[str, Any],
    ) -> None:
        path = self._session_path(engine, self.anima_dir, session_type, thread_id)
        session_id = record.get("session_id")
        if not isinstance(session_id, str):
            raise ValueError("session record requires session_id")

        def _save() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if engine == "agent_sdk":
                atomic_write_json(path, record, indent=None)
            else:
                contents = session_id
                if engine in {"cursor", "grok", "agy"}:
                    turn_count = record.get("turn_count", 1)
                    if not isinstance(turn_count, int) or isinstance(turn_count, bool):
                        raise ValueError("session turn_count must be an integer")
                    contents = f"{session_id}\n{turn_count}"
                path.write_text(contents, encoding="utf-8")

        await self._run(str(path), _save)

    async def clear_session(self, engine: str, session_type: str, thread_id: str = "default") -> None:
        path = self._session_path(engine, self.anima_dir, session_type, thread_id)
        await self._run(str(path), lambda: path.unlink(missing_ok=True))

    @staticmethod
    def _entry_date(entry: dict[str, Any]) -> str:
        timestamp = entry.get("ts")
        if isinstance(timestamp, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp[:10]):
            return timestamp[:10]
        return today_local().isoformat()

    async def log_token_usage(self, record: dict[str, Any]) -> None:
        path = self.anima_dir / "token_usage" / f"{self._entry_date(record)}.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"

        def _append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)

        await self._run(str(path), _append)

    async def save_token_usage_rollup(self, rollup: dict[str, int]) -> None:
        path = self.anima_dir / "token_usage" / "rollup.json"
        text = json.dumps(dict(sorted(rollup.items())), ensure_ascii=False, indent=2) + "\n"
        await self._run(str(path), lambda: atomic_write_text(path, text))

    async def log_prompt(self, record: dict[str, Any]) -> None:
        await self._append_prompt_record(record)

    async def log_prompt_end(self, record: dict[str, Any]) -> None:
        await self._append_prompt_record(record)

    async def rotate_prompt_logs(self) -> None:
        global _PROMPT_ROTATION_DATE
        log_dir = self.anima_dir / "prompt_logs"

        def _rotate() -> None:
            global _PROMPT_ROTATION_DATE
            today = now_local().strftime("%Y-%m-%d")
            if today == _PROMPT_ROTATION_DATE:
                return
            _PROMPT_ROTATION_DATE = today
            cutoff = (now_local() - timedelta(days=_PROMPT_LOG_RETENTION_DAYS)).strftime("%Y-%m-%d")
            for path in log_dir.glob("*.jsonl"):
                if path.stem < cutoff:
                    path.unlink(missing_ok=True)

        await self._run(str(log_dir), _rotate)

    async def _append_prompt_record(self, record: dict[str, Any]) -> None:
        global _PROMPT_ROTATION_DATE
        date = self._entry_date(record)
        log_dir = self.anima_dir / "prompt_logs"
        path = log_dir / f"{date}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

        def _append() -> None:
            global _PROMPT_ROTATION_DATE
            log_dir.mkdir(parents=True, exist_ok=True)
            today = now_local().strftime("%Y-%m-%d")
            if today != _PROMPT_ROTATION_DATE:
                _PROMPT_ROTATION_DATE = today
                cutoff = (now_local() - timedelta(days=_PROMPT_LOG_RETENTION_DAYS)).strftime("%Y-%m-%d")
                for old_path in log_dir.glob("*.jsonl"):
                    if old_path.stem < cutoff:
                        old_path.unlink(missing_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)

        await self._run(str(path), _append)

    async def set_consolidation_mode(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise ValueError("consolidation mode must be boolean")
        path = self.anima_dir / "state" / ".consolidation_mode"

        def _set() -> None:
            if active:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("1", encoding="utf-8")
            else:
                path.unlink(missing_ok=True)

        await self._run(str(path), _set)

    async def write_heartbeat_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        path = self.anima_dir / "state" / "heartbeat_checkpoint.json"
        await self._run(
            str(path),
            lambda: atomic_write_json(path, checkpoint, indent=None, trailing_newline=False),
        )

    async def clear_heartbeat_checkpoint(self) -> None:
        path = self.anima_dir / "state" / "heartbeat_checkpoint.json"
        await self._run(str(path), lambda: path.unlink(missing_ok=True))

    async def write_recovery_note(self, content: str) -> None:
        path = self.anima_dir / "state" / "recovery_note.md"

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        await self._run(str(path), _write)

    async def consume_recovery_note(self) -> str:
        path = self.anima_dir / "state" / "recovery_note.md"

        def _consume() -> str:
            if not path.exists():
                return ""
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                logger.debug("Failed to read recovery note %s", path, exc_info=True)
                return ""
            path.unlink(missing_ok=True)
            return content

        return await self._run(str(path), _consume)

    async def update_inbox_read_counts(self, counts: dict[str, int]) -> None:
        path = self.anima_dir / "state" / "inbox_read_counts.json"
        await self._run(
            str(path),
            lambda: atomic_write_json(path, counts, indent=None, trailing_newline=False),
        )

    async def save_task_result(self, task_id: str, attempt_token: str | None, content: str) -> Path:
        task_id = self._component(task_id, "task_id")
        if attempt_token:
            attempt_token = self._component(attempt_token, "attempt_token")
            path = self.anima_dir / "state" / "task_results" / task_id / f"{attempt_token}.md"
        else:
            path = self.anima_dir / "state" / "task_results" / f"{task_id}.md"
        await self._run(str(path), lambda: atomic_write_text(path, content))
        return path

    async def write_inbox_overflow_file(self, message: dict[str, Any]) -> Path:
        path_root = self.anima_dir / "state" / "overflow_inbox"
        timestamp = str(message.get("ts") or now_iso())
        stamp = timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
        sender = re.sub(r"[^A-Za-z0-9_-]", "_", str(message.get("from_person") or "unknown"))[:64] or "unknown"
        base = f"{stamp}_{sender}"
        content = "\n".join(
            [
                "---",
                f"from: {message.get('from_person', 'unknown')}",
                f"ts: {timestamp}",
                f"intent: {message.get('intent', '')}",
                f"type: {message.get('type', 'message')}",
                "---",
                "",
                str(message.get("content") or ""),
            ]
        )

        def _write() -> Path:
            path_root.mkdir(parents=True, exist_ok=True)
            path = path_root / f"{base}.md"
            counter = 2
            while path.exists():
                path = path_root / f"{base}_{counter}.md"
                counter += 1
            path.write_text(content, encoding="utf-8")
            return path

        return await self._run(str(path_root), _write)

    async def cleanup_inbox_overflow(self) -> None:
        path_root = self.anima_dir / "state" / "overflow_inbox"

        def _cleanup() -> None:
            if not path_root.exists():
                return
            files = sorted(path_root.glob("*.md"), key=lambda path: path.stat().st_mtime)
            cutoff = time.time() - _INBOX_OVERFLOW_MAX_AGE_DAYS * 86400
            for path in files:
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    pass
            remaining = sorted(path_root.glob("*.md"), key=lambda path: path.stat().st_mtime)
            excess = len(remaining) - _INBOX_OVERFLOW_MAX_FILES
            for path in remaining[: max(excess, 0)]:
                try:
                    path.unlink()
                except OSError:
                    pass

        await self._run(str(path_root), _cleanup)

    async def write_token_budget_notification(self, month: str, metadata: dict[str, Any]) -> bool:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ValueError(f"invalid budget month: {month!r}")
        marker = self.anima_dir / "state" / "token_budget_notifications" / f"{month}.notified"
        notification = self.anima_dir / "state" / "background_notifications" / f"token_budget_exceeded_{month}.md"
        content = (
            "# Monthly token budget reached\n\n"
            f"- month: {month}\n"
            f"- budget: {metadata['budget']}\n"
            f"- consumed: {metadata['consumed']}\n"
            f"- trigger: {metadata['trigger']}\n"
        )

        def _write() -> bool:
            marker.parent.mkdir(parents=True, exist_ok=True)
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    stream.write(now_iso() + "\n")
            except FileExistsError:
                return False
            try:
                notification.parent.mkdir(parents=True, exist_ok=True)
                notification.write_text(content, encoding="utf-8")
            except Exception:
                try:
                    marker.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            return True

        return await self._run(str(notification.parent), _write)

    async def mark_cron_rejected_notice(self, digest: str) -> None:
        path = self.anima_dir / "state" / "cron_rejected_notice.sha256"
        await self._run(str(path), lambda: atomic_write_text(path, digest + "\n"))

    async def archive_heartbeat_md_snapshot(self) -> str | None:
        source = self.anima_dir / HEARTBEAT_FILE
        archive_dir = self.anima_dir / "archive" / "heartbeat"
        archive_path = archive_dir / f"{HEARTBEAT_FILE}.{now_local().strftime('%Y%m%d')}"

        def _archive() -> str | None:
            try:
                if not source.is_file():
                    raise FileNotFoundError(f"{HEARTBEAT_FILE} not found at {source}")
                archive_dir.mkdir(parents=True, exist_ok=True)
                if not archive_path.exists():
                    shutil.copy2(source, archive_path)
                if not archive_path.is_file():
                    raise OSError(f"archive target is not a file: {archive_path}")
                return str(archive_path.relative_to(self.anima_dir))
            except OSError:
                logger.warning("Failed to archive %s before cleanup", HEARTBEAT_FILE, exc_info=True)
                return None

        return await self._run(str(archive_path), _archive)

    async def consume_background_notifications(self, mode: str) -> list[str]:
        if mode not in {"all", "chat"}:
            raise ValueError(f"invalid background notification mode: {mode!r}")
        directory = self.anima_dir / "state" / "background_notifications"
        excluded = ("cron_health_", "cron_guard_", "token_budget_")

        def _consume() -> list[str]:
            if not directory.is_dir():
                return []
            notifications: list[str] = []
            for path in sorted(directory.glob("*.md")):
                if mode == "chat" and path.name.startswith(excluded):
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                    path.unlink()
                    notifications.append(content)
                except (OSError, UnicodeDecodeError):
                    logger.warning("Failed to read notification: %s", path.name)
            return notifications

        return await self._run(str(directory), _consume)

    async def write_background_notification(self, task_id: str, content: str) -> Path:
        task_id = self._component(task_id, "task_id")
        path = self.anima_dir / "state" / "background_notifications" / f"{task_id}.md"
        await self._run(str(path.parent), lambda: atomic_write_text(path, content))
        return path

    async def write_bootstrap_state(self, state: dict[str, Any]) -> None:
        path = self.anima_dir / "state" / "bootstrap_state.json"
        await self._run(str(path), lambda: atomic_write_json(path, state))

    async def archive_bootstrap_file(self, source_name: str) -> Path:
        allowed = {
            BOOTSTRAP_FILE,
            f"{BOOTSTRAP_FILE}.auto_resolved",
            f"{BOOTSTRAP_FILE}.failed",
            "character_sheet.md",
        }
        if source_name not in allowed:
            raise ValueError(f"unsupported bootstrap archive source: {source_name!r}")
        source = self.anima_dir / source_name
        archive_dir = self.anima_dir / "state" / "bootstrap_archive"

        def _archive() -> Path:
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = now_local().strftime("%Y%m%d_%H%M%S")
            if source_name == BOOTSTRAP_FILE:
                archive_path = archive_dir / f"bootstrap-{timestamp}.md"
            else:
                archive_path = archive_dir / f"{source_name}.{timestamp}"
            counter = 1
            while archive_path.exists():
                if source_name == BOOTSTRAP_FILE:
                    archive_path = archive_dir / f"bootstrap-{timestamp}-{counter}.md"
                else:
                    archive_path = archive_dir / f"{source_name}.{timestamp}-{counter}"
                counter += 1
            shutil.move(str(source), str(archive_path))
            return archive_path

        return await self._run(str(source), _archive)

    async def save_background_task(self, task_id: str, task: dict[str, Any]) -> Path:
        task_id = self._component(task_id, "background_task_id")
        path = self.anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        await self._run(str(path), lambda: atomic_write_json(path, task, indent=2, ensure_ascii=False))
        return path

    async def clear_background_task(self, task_id: str) -> None:
        task_id = self._component(task_id, "background_task_id")
        path = self.anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        await self._run(str(path), lambda: path.unlink(missing_ok=True))

    async def execute_operation(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one validated IPC operation to the same local methods."""
        self._assert_main_process()
        handlers: dict[str, Callable[..., Any]] = {
            "save_conversation": self.save_conversation,
            "clear_conversation": self.clear_conversation,
            "append_transcript": self.append_transcript,
            "save_shortterm": self.save_shortterm,
            "archive_shortterm": self.archive_shortterm,
            "prune_shortterm_archive": self.prune_shortterm_archive,
            "migrate_legacy_shortterm": self.migrate_legacy_shortterm,
            "save_stream_checkpoint": self.save_stream_checkpoint,
            "clear_stream_checkpoint": self.clear_stream_checkpoint,
            "save_session_record": self.save_session_record,
            "clear_session": self.clear_session,
            "log_token_usage": self.log_token_usage,
            "save_token_usage_rollup": self.save_token_usage_rollup,
            "log_prompt": self.log_prompt,
            "log_prompt_end": self.log_prompt_end,
            "rotate_prompt_logs": self.rotate_prompt_logs,
            "set_consolidation_mode": self.set_consolidation_mode,
            "write_heartbeat_checkpoint": self.write_heartbeat_checkpoint,
            "clear_heartbeat_checkpoint": self.clear_heartbeat_checkpoint,
            "write_recovery_note": self.write_recovery_note,
            "consume_recovery_note": self.consume_recovery_note,
            "update_inbox_read_counts": self.update_inbox_read_counts,
            "save_task_result": self.save_task_result,
            "write_inbox_overflow_file": self.write_inbox_overflow_file,
            "cleanup_inbox_overflow": self.cleanup_inbox_overflow,
            "write_token_budget_notification": self.write_token_budget_notification,
            "mark_cron_rejected_notice": self.mark_cron_rejected_notice,
            "archive_heartbeat_md_snapshot": self.archive_heartbeat_md_snapshot,
            "consume_background_notifications": self.consume_background_notifications,
            "write_background_notification": self.write_background_notification,
            "write_bootstrap_state": self.write_bootstrap_state,
            "archive_bootstrap_file": self.archive_bootstrap_file,
            "save_background_task": self.save_background_task,
            "clear_background_task": self.clear_background_task,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise ValueError(f"unsupported state_write operation: {operation!r}")
        result = await handler(**payload)
        if isinstance(result, dict):
            return result
        if result is None:
            return {}
        if isinstance(result, Path):
            return {"path": str(result)}
        if isinstance(result, list):
            return {"items": result}
        if isinstance(result, (str, int, float, bool)):
            return {"value": result}
        raise TypeError(f"unsupported state_write result: {type(result).__name__}")


class IpcStateWriter:
    """Task-runner implementation that waits for Anima-main persistence ACKs."""

    def __init__(self, link: StateWriteLink) -> None:
        self._link = link
        self._send_lock = asyncio.Lock()
        self._loop = asyncio.get_running_loop()
        self._pending: set[asyncio.Task[Any]] = set()

    async def _write(self, operation: str, **payload: Any) -> dict[str, Any]:
        async with self._send_lock:
            body = {"operation": operation, "payload": payload}
            encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
            if len(encoded) + 1024 <= 3 * 1024 * 1024:
                return await self._link.request("state_write", body)

            if len(encoded) > _STATE_WRITE_MAX_PAYLOAD_BYTES:
                raise StateWriterError(
                    f"state_write payload is {len(encoded)} bytes; maximum is {_STATE_WRITE_MAX_PAYLOAD_BYTES}"
                )
            transaction_id = hashlib.sha256(os.urandom(32)).hexdigest()
            await self._link.send_unanswered_request(
                "state_write",
                {
                    "phase": "begin",
                    "transaction_id": transaction_id,
                    "operation": operation,
                    "byte_length": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                },
            )
            for index, start in enumerate(range(0, len(encoded), _STATE_WRITE_CHUNK_BYTES)):
                chunk = encoded[start : start + _STATE_WRITE_CHUNK_BYTES]
                await self._link.send_unanswered_request(
                    "state_write",
                    {
                        "phase": "chunk",
                        "transaction_id": transaction_id,
                        "index": index,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    },
                )
            return await self._link.request(
                "state_write",
                {"phase": "commit", "transaction_id": transaction_id},
            )

    async def save_conversation(self, thread_id: str, state: dict[str, Any]) -> None:
        await self._write("save_conversation", thread_id=thread_id, state=state)

    async def clear_conversation(self, thread_id: str = "default") -> None:
        await self._write("clear_conversation", thread_id=thread_id)

    async def append_transcript(self, entry: dict[str, Any]) -> None:
        await self._write("append_transcript", entry=entry)

    async def save_shortterm(
        self,
        session_type: str,
        thread_id: str,
        state: dict[str, Any],
        markdown: str,
    ) -> Path:
        result = await self._write(
            "save_shortterm",
            session_type=session_type,
            thread_id=thread_id,
            state=state,
            markdown=markdown,
        )
        return Path(str(result.get("path", "")))

    async def archive_shortterm(self, session_type: str, thread_id: str = "default") -> None:
        await self._write("archive_shortterm", session_type=session_type, thread_id=thread_id)

    async def migrate_legacy_shortterm(self, session_type: str, thread_id: str = "default") -> None:
        await self._write("migrate_legacy_shortterm", session_type=session_type, thread_id=thread_id)

    async def prune_shortterm_archive(
        self,
        session_type: str,
        thread_id: str = "default",
        max_files: int = _SHORTTERM_ARCHIVE_MAX_FILES,
    ) -> None:
        await self._write(
            "prune_shortterm_archive",
            session_type=session_type,
            thread_id=thread_id,
            max_files=max_files,
        )

    async def save_stream_checkpoint(
        self,
        session_type: str,
        thread_id: str,
        checkpoint: dict[str, Any],
    ) -> Path:
        result = await self._write(
            "save_stream_checkpoint",
            session_type=session_type,
            thread_id=thread_id,
            checkpoint=checkpoint,
        )
        return Path(str(result.get("path", "")))

    async def clear_stream_checkpoint(self, session_type: str, thread_id: str = "default") -> None:
        await self._write("clear_stream_checkpoint", session_type=session_type, thread_id=thread_id)

    async def save_session_record(
        self,
        engine: str,
        session_type: str,
        thread_id: str,
        record: dict[str, Any],
    ) -> None:
        await self._write(
            "save_session_record",
            engine=engine,
            session_type=session_type,
            thread_id=thread_id,
            record=record,
        )

    async def clear_session(self, engine: str, session_type: str, thread_id: str = "default") -> None:
        await self._write("clear_session", engine=engine, session_type=session_type, thread_id=thread_id)

    async def log_token_usage(self, record: dict[str, Any]) -> None:
        await self._write("log_token_usage", record=record)

    async def save_token_usage_rollup(self, rollup: dict[str, int]) -> None:
        await self._write("save_token_usage_rollup", rollup=rollup)

    async def log_prompt(self, record: dict[str, Any]) -> None:
        await self._write("log_prompt", record=record)

    async def log_prompt_end(self, record: dict[str, Any]) -> None:
        await self._write("log_prompt_end", record=record)

    async def rotate_prompt_logs(self) -> None:
        await self._write("rotate_prompt_logs")

    async def set_consolidation_mode(self, active: bool) -> None:
        await self._write("set_consolidation_mode", active=active)

    async def write_heartbeat_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        await self._write("write_heartbeat_checkpoint", checkpoint=checkpoint)

    async def clear_heartbeat_checkpoint(self) -> None:
        await self._write("clear_heartbeat_checkpoint")

    async def write_recovery_note(self, content: str) -> None:
        await self._write("write_recovery_note", content=content)

    async def consume_recovery_note(self) -> str:
        result = await self._write("consume_recovery_note")
        return str(result.get("value") or "")

    async def update_inbox_read_counts(self, counts: dict[str, int]) -> None:
        await self._write("update_inbox_read_counts", counts=counts)

    async def save_task_result(self, task_id: str, attempt_token: str | None, content: str) -> Path:
        result = await self._write(
            "save_task_result",
            task_id=task_id,
            attempt_token=attempt_token,
            content=content,
        )
        return Path(str(result.get("path", "")))

    async def write_inbox_overflow_file(self, message: dict[str, Any]) -> Path:
        result = await self._write("write_inbox_overflow_file", message=message)
        return Path(str(result.get("path", "")))

    async def cleanup_inbox_overflow(self) -> None:
        await self._write("cleanup_inbox_overflow")

    async def write_token_budget_notification(self, month: str, metadata: dict[str, Any]) -> bool:
        result = await self._write("write_token_budget_notification", month=month, metadata=metadata)
        return result.get("value") is True

    async def mark_cron_rejected_notice(self, digest: str) -> None:
        await self._write("mark_cron_rejected_notice", digest=digest)

    async def archive_heartbeat_md_snapshot(self) -> str | None:
        result = await self._write("archive_heartbeat_md_snapshot")
        value = result.get("value")
        return str(value) if value else None

    async def consume_background_notifications(self, mode: str) -> list[str]:
        result = await self._write("consume_background_notifications", mode=mode)
        items = result.get("items")
        return [str(item) for item in items] if isinstance(items, list) else []

    async def write_background_notification(self, task_id: str, content: str) -> Path:
        result = await self._write("write_background_notification", task_id=task_id, content=content)
        return Path(str(result.get("path", "")))

    async def write_bootstrap_state(self, state: dict[str, Any]) -> None:
        await self._write("write_bootstrap_state", state=state)

    async def archive_bootstrap_file(self, source_name: str) -> Path:
        result = await self._write("archive_bootstrap_file", source_name=source_name)
        return Path(str(result.get("path", "")))

    async def save_background_task(self, task_id: str, task: dict[str, Any]) -> Path:
        result = await self._write("save_background_task", task_id=task_id, task=task)
        return Path(str(result.get("path", "")))

    async def clear_background_task(self, task_id: str) -> None:
        await self._write("clear_background_task", task_id=task_id)

    def run_sync(self, awaitable: Any) -> Any:
        """Run a synchronous adapter from a worker thread, never from this loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise StateWriterError("synchronous state write from the task_runner event-loop thread would deadlock")
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop).result()

    def schedule(self, awaitable: Any) -> None:
        """Retain a fire-and-forget state write until the task contract drains it."""
        task = asyncio.create_task(awaitable)
        self._pending.add(task)

    async def drain(self) -> None:
        """Wait for state writes deliberately scheduled by synchronous callbacks."""
        while self._pending:
            pending = tuple(self._pending)
            try:
                await asyncio.gather(*pending)
            finally:
                self._pending.difference_update(pending)


_state_writer_override: LocalStateWriter | IpcStateWriter | None = None
_local_writers: dict[str, LocalStateWriter] = {}


def configure_state_writer(writer: LocalStateWriter | IpcStateWriter | None) -> None:
    """Install the process-scoped writer (the task runner installs its IPC link)."""
    global _state_writer_override
    _state_writer_override = writer


def get_state_writer(anima_dir: Path | None = None) -> LocalStateWriter | IpcStateWriter:
    """Return the process-appropriate state writer for one Anima directory."""
    if _state_writer_override is not None:
        return _state_writer_override
    if _is_task_runner_process():
        raise StateWriterError("task_runner state writer has not been connected to the Anima main")
    if anima_dir is None:
        env_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR", "").strip()
        if not env_dir:
            raise StateWriterError("anima_dir is required to create a LocalStateWriter")
        anima_dir = Path(env_dir)
    key = str(Path(anima_dir).resolve())
    writer = _local_writers.get(key)
    if writer is None:
        writer = LocalStateWriter(Path(anima_dir))
        _local_writers[key] = writer
    return writer


def run_writer_sync(writer: LocalStateWriter | IpcStateWriter, awaitable: Any) -> Any:
    """Compatibility bridge for unavoidable synchronous API boundaries.

    IPC writes from a non-loop thread are submitted to the task runner's loop
    with ``run_coroutine_threadsafe``. Calling the bridge from that loop is an
    error, preventing a self-deadlock. Local compatibility calls may run a
    private loop, including when invoked by legacy synchronous tests.
    """
    if isinstance(writer, IpcStateWriter):
        return writer.run_sync(awaitable)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if _is_task_runner_process():
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise StateWriterError("synchronous state write from the task_runner event-loop thread is forbidden")
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


def _reset_state_writers_for_tests() -> None:
    """Clear process-scoped writer caches between isolated unit tests."""
    global _state_writer_override
    _state_writer_override = None
    _local_writers.clear()


__all__ = [
    "IpcStateWriter",
    "LocalStateWriter",
    "StateWriterError",
    "configure_state_writer",
    "get_state_writer",
    "is_task_runner_process",
    "run_writer_sync",
]
