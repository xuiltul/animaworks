from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Schedule-parsing helpers extracted from lifecycle.py.

Provides pure-function parsers for cron.md and heartbeat.md with no
dependency on LifecycleManager or APScheduler internals.
"""

import logging
import re
from typing import Any

import yaml
from apscheduler.triggers.cron import CronTrigger

from core.schemas import CronTask

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

# Regex for a valid 5-field cron expression.
# Each field may contain digits, *, /, -, and commas.
_CRON_EXPR_RE = re.compile(r"^[\d\*\/\-\,]+(\s+[\d\*\/\-\,]+){4}$")


# ── Heartbeat parsing ────────────────────────────────────


def parse_heartbeat_config(content: str) -> tuple[int, int, int]:
    """Parse heartbeat.md content to extract scheduling parameters.

    Returns:
        Tuple of (interval_minutes, active_start_hour, active_end_hour)
    """
    interval = 30
    m = re.search(r"(\d+)\s*分", content)
    if m:
        interval = int(m.group(1))

    active_start, active_end = 9, 22
    m = re.search(r"(\d{1,2}):\d{0,2}\s*-\s*(\d{1,2})", content)
    if m:
        active_start, active_end = int(m.group(1)), int(m.group(2))

    return interval, active_start, active_end


# ── Cron parsing ─────────────────────────────────────────


def parse_cron_md(content: str) -> list[CronTask]:
    """Parse cron.md to extract CronTask definitions.

    Expected format per section:

        ## Task Title
        schedule: 0 9 * * *
        type: llm
        Description text...

    Or for command-type:

        ## Deploy Task
        schedule: 0 2 * * 1-5
        type: command
        command: /path/to/script.sh

    Or with tool:

        ## Tool Task
        schedule: */5 * * * *
        type: command
        tool: tool_name
        args:
          key: value

    Format rules:
        - ``## Title`` — human-readable label (display only)
        - ``schedule: <cron-expression>`` — standard 5-field cron
        - ``type: llm|command`` — execution type
        - ``command: <cmd>`` — for command type only
        - ``tool: <name>`` — for command type with tool
        - ``args:`` — YAML block for tool arguments
        - ``skip_pattern: <regex>`` — stdout matching this skips heartbeat
        - Remaining text lines become the task description (LLM type)
        - HTML comments (``<!-- -->``) are stripped before parsing
    """
    # Strip HTML comment blocks before parsing
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    tasks: list[CronTask] = []
    cur_name = ""
    cur_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if cur_name:
                tasks.append(_parse_section(cur_name, cur_lines))
            cur_name = line[3:].strip()
            cur_lines = []
        elif cur_name:
            cur_lines.append(line)

    if cur_name:
        tasks.append(_parse_section(cur_name, cur_lines))
    return tasks


def _parse_section(name: str, lines: list[str]) -> CronTask:
    """Parse a single cron task section from its body lines.

    Extracts ``schedule:``, ``type:``, ``command:``, ``tool:``, ``args:``
    directives.  All remaining non-directive lines form the task description.
    """
    schedule = ""
    task_type = "llm"
    command = None
    tool = None
    args = None
    skip_pattern = None
    description_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("schedule:"):
            schedule = stripped[len("schedule:"):].strip()
        elif stripped.startswith("type:"):
            task_type = stripped[5:].strip()
        elif stripped.startswith("command:"):
            command = stripped[8:].strip()
        elif stripped.startswith("tool:"):
            tool = stripped[5:].strip()
        elif stripped.startswith("skip_pattern:"):
            val = stripped[len("skip_pattern:"):].strip()
            if val:
                try:
                    re.compile(val)
                    skip_pattern = val
                except re.error as e:
                    logger.warning("Invalid skip_pattern for task %s: %s", name, e)
        elif stripped.startswith("args:"):
            # Parse YAML args block (indented lines following "args:")
            yaml_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Continue if line is indented or empty
                if next_line.startswith("  ") or not next_line.strip():
                    yaml_lines.append(next_line)
                    i += 1
                else:
                    break
            i -= 1  # Back one line for outer loop increment

            # Parse YAML
            try:
                parsed = yaml.safe_load("\n".join(yaml_lines))
                if parsed and "args" in parsed:
                    args = parsed["args"]
            except yaml.YAMLError as e:
                logger.warning("Failed to parse args YAML for task %s: %s", name, e)
        else:
            # Regular description line
            description_lines.append(line)

        i += 1

    return CronTask(
        name=name,
        schedule=schedule,
        type=task_type,
        description="\n".join(description_lines).strip(),
        command=command,
        tool=tool,
        args=args,
        skip_pattern=skip_pattern,
    )


def parse_schedule(schedule: str) -> CronTrigger | None:
    """Parse a standard 5-field cron expression into an APScheduler CronTrigger.

    Args:
        schedule: A cron expression string, e.g. ``"0 9 * * *"`` or
            ``"*/5 * * * *"``.

    Returns:
        CronTrigger if the expression is valid, None otherwise.
    """
    s = schedule.strip()
    if not s:
        logger.warning("Empty schedule expression")
        return None

    if not _CRON_EXPR_RE.match(s):
        logger.warning("Invalid cron expression: '%s'", s)
        return None

    parts = s.split()
    if len(parts) != 5:
        logger.warning("Cron expression must have exactly 5 fields: '%s'", s)
        return None

    try:
        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )
    except Exception as e:
        logger.warning("Failed to create CronTrigger from '%s': %s", s, e)
        return None
