from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
from datetime import timedelta
from pathlib import Path

from core.time_utils import now_iso, now_local

logger = logging.getLogger("animaworks.memory")

# ── CronLogger ────────────────────────────────────────────


class CronLogger:
    """Cron execution log recorder and reader.

    Manages daily JSONL log files under ``{anima_dir}/state/cron_logs/``.
    """

    _LOG_DIR = "state/cron_logs"
    _MAX_LINES = 50

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir

    def _log_dir(self) -> Path:
        return self._anima_dir / self._LOG_DIR

    def append_cron_log(
        self,
        task_name: str,
        *,
        summary: str,
        duration_ms: int,
    ) -> None:
        """Append a cron execution result to the daily log."""
        log_dir = self._log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{now_local().date().isoformat()}.jsonl"

        entry = json.dumps(
            {
                "timestamp": now_iso(),
                "task": task_name,
                "summary": summary[:500],
                "duration_ms": duration_ms,
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Keep file bounded — use atomic write for truncation
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > self._MAX_LINES:
            from core.memory._io import atomic_write_text

            atomic_write_text(path, "\n".join(lines[-self._MAX_LINES :]) + "\n")

    def append_cron_command_log(
        self,
        task_name: str,
        *,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_ms: int,
    ) -> None:
        """Append a command-type cron execution result to the daily log.

        Logs include exit code, line counts, and previews (first+last 5 lines).
        """
        log_dir = self._log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{now_local().date().isoformat()}.jsonl"

        # Count lines
        stdout_lines_list = stdout.splitlines()
        stderr_lines_list = stderr.splitlines()
        stdout_line_count = len(stdout_lines_list)
        stderr_line_count = len(stderr_lines_list)

        # Generate preview: first 5 + last 5 lines, max 1000 chars total
        def make_preview(lines_list: list[str]) -> str:
            if not lines_list:
                return ""
            if len(lines_list) <= 10:
                preview = "\n".join(lines_list)
            else:
                preview = "\n".join(lines_list[:5] + ["..."] + lines_list[-5:])
            return preview[:1000]

        stdout_preview = make_preview(stdout_lines_list)
        stderr_preview = make_preview(stderr_lines_list)

        entry = json.dumps(
            {
                "timestamp": now_iso(),
                "task": task_name,
                "exit_code": exit_code,
                "stdout_lines": stdout_line_count,
                "stderr_lines": stderr_line_count,
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "duration_ms": duration_ms,
            },
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Keep file bounded — use atomic write for truncation
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > self._MAX_LINES:
            from core.memory._io import atomic_write_text

            atomic_write_text(path, "\n".join(lines[-self._MAX_LINES :]) + "\n")

    def read_cron_log(self, days: int = 1) -> str:
        """Read cron logs for the last *days* days."""
        log_dir = self._log_dir()
        if not log_dir.is_dir():
            return ""

        parts: list[str] = []
        today = now_local().date()
        for i in range(days):
            target = today - timedelta(days=i)
            path = log_dir / f"{target.isoformat()}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    e = json.loads(line)
                    if "summary" in e:
                        line_text = f"- {e['timestamp']}: [{e['task']}] {e['summary'][:200]} ({e['duration_ms']}ms)"
                    else:
                        exit_code = e.get("exit_code", "?")
                        preview = (e.get("stdout_preview", "") or e.get("stderr_preview", ""))[:100]
                        dur = e.get("duration_ms", 0)
                        line_text = f"- {e['timestamp']}: [{e['task']}] exit={exit_code} {preview} ({dur}ms)"
                    parts.append(line_text)
                except (json.JSONDecodeError, KeyError):
                    continue
        return "\n".join(parts)
