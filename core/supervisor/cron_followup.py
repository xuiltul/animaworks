"""Shared command-cron follow-up policy for legacy and isolated runners."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.schemas import CronTask

logger = logging.getLogger(__name__)


def command_followup_output(task: CronTask, result: dict[str, Any]) -> str | None:
    """Suppress only successful, explicitly quiet output; surface failures.

    ``trigger_heartbeat=False`` delegates notification to the command itself.
    A successful command with stderr is not demonstrably quiet, even if its
    stdout matches a normal-output pattern. The original result remains in
    the command audit log regardless of this decision.
    """
    if not task.trigger_heartbeat:
        return None
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    if result.get("exit_code", 1) != 0 or stderr:
        return json.dumps(
            {"exit_code": result.get("exit_code"), "stdout": stdout, "stderr": stderr},
            ensure_ascii=False,
        )
    if not stdout:
        return None
    if task.skip_pattern:
        try:
            if re.search(task.skip_pattern, stdout):
                return None
        except re.error as exc:
            logger.warning(
                "Invalid skip_pattern %r for task %r: %s; continuing without skip",
                task.skip_pattern,
                task.name,
                exc,
            )
    return stdout
