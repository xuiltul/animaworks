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

import yaml
from apscheduler.triggers.cron import CronTrigger

from core.schemas import CronTask

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

# Regex for a valid 5-field cron expression.
# Each field may contain digits, *, /, -, and commas.
_CRON_EXPR_RE = re.compile(r"^[\d\*\/\-\,]+(\s+[\d\*\/\-\,]+){4}$")


# ── Heartbeat parsing ────────────────────────────────────


def parse_heartbeat_config(content: str) -> tuple[int | None, int | None]:
    """Parse heartbeat.md content to extract active hours.

    Interval is managed by config.json (heartbeat.interval_minutes),
    NOT parsed from heartbeat.md content.

    Returns:
        Tuple of (active_start_hour, active_end_hour).
        Both are None if no time range found in content.
    """
    m = re.search(r"(\d{1,2}):\d{0,2}\s*-\s*(\d{1,2})", content)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


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


def _strip_outer_quotes(val: str) -> str:
    """Strip a matching pair of outer quotes (``"..."`` or ``'...'``).

    Only strips when the inner content does *not* contain the same quote
    character, so values like ``"status": "ACTIVE"`` (where quotes are
    intentional regex literals) are left untouched.
    """
    if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
        inner = val[1:-1]
        if val[0] not in inner:
            return inner
    return val


def _strip_inline_comment(val: str) -> str:
    """Strip trailing inline comment (e.g. ``'value  # comment'`` -> ``'value'``).

    Only strips when ``#`` is preceded by whitespace, so values like
    ``echo#nocomment`` are preserved.
    """
    return re.sub(r"\s+#.*$", "", val)


def _parse_section(name: str, lines: list[str]) -> CronTask:
    """Parse a single cron task section from its body lines.

    Extracts ``schedule:``, ``type:``, ``command:``, ``tool:``, ``args:``
    directives.  All remaining non-directive lines form the task description.

    Format tolerance:
        - YAML list prefix ``- `` is stripped (e.g. ``- schedule:`` -> ``schedule:``)
        - Inline comments ``# ...`` after directive values are stripped
    """
    schedule = ""
    task_type = "llm"
    command = None
    tool = None
    args = None
    skip_pattern = None
    trigger_heartbeat = True
    description_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Strip YAML list prefix (e.g. "- schedule:" -> "schedule:")
        if stripped.startswith("- "):
            stripped = stripped[2:]

        if stripped.startswith("schedule:"):
            schedule = _strip_inline_comment(stripped[len("schedule:") :].strip())
            schedule = _strip_outer_quotes(schedule)
        elif stripped.startswith("type:"):
            task_type = _strip_inline_comment(stripped[5:].strip())
        elif stripped.startswith("command:"):
            command = _strip_inline_comment(stripped[8:].strip())
        elif stripped.startswith("tool:"):
            tool = _strip_inline_comment(stripped[5:].strip())
        elif stripped.startswith("skip_pattern:"):
            val = _strip_inline_comment(stripped[len("skip_pattern:") :].strip())
            val = _strip_outer_quotes(val)
            if val:
                try:
                    re.compile(val)
                    skip_pattern = val
                except re.error as e:
                    logger.warning("Invalid skip_pattern for task %s: %s", name, e)
        elif stripped.startswith("trigger_heartbeat:"):
            val = _strip_inline_comment(stripped.split(":", 1)[1].strip()).lower()
            trigger_heartbeat = val not in ("false", "no", "0")
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

    # Warn if type=llm but description contains code blocks
    # (likely should be type=command instead)
    description = "\n".join(description_lines).strip()
    if task_type == "llm" and "```" in description:
        logger.warning(
            "Cron task '%s' is type=llm but contains a code block. "
            "If the command is deterministic, use type=command instead.",
            name,
        )

    return CronTask(
        name=name,
        schedule=schedule,
        type=task_type,
        description=description,
        command=command,
        tool=tool,
        args=args,
        skip_pattern=skip_pattern,
        trigger_heartbeat=trigger_heartbeat,
    )


def _posix_range_to_iso_values(start: int, end: int, step: int = 1) -> list[int]:
    """Expand a POSIX DOW range to a list of ISO APScheduler values.

    POSIX values in [start, end] (inclusive) are sampled by *step* and each
    converted via ``(posix - 1) % 7`` to the ISO numbering used by APScheduler.

    Args:
        start: First POSIX day value (0/7=Sun, 1=Mon, …, 6=Sat).
        end:   Last POSIX day value (inclusive).
        step:  Step size (default 1).

    Returns:
        Ordered list of ISO day values in the same order as the POSIX range.
    """
    return [(x - 1) % 7 for x in range(start, end + 1, step)]


def _iso_values_to_parts(values: list[int]) -> list[str]:
    """Compress a list of consecutive ISO DOW values into range-notation parts.

    Runs of consecutive integers (diff == 1) are collapsed to ``"a-b"``;
    isolated values are emitted as plain strings.

    Args:
        values: Ordered list of ISO day integers.

    Returns:
        List of string parts ready to be joined with commas.
    """
    if not values:
        return []

    parts: list[str] = []
    group_start = values[0]
    prev = values[0]

    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            parts.append(str(group_start) if prev == group_start else f"{group_start}-{prev}")
            group_start = v
            prev = v

    parts.append(str(group_start) if prev == group_start else f"{group_start}-{prev}")
    return parts


def _posix_dow_to_apsched(dow: str) -> str:
    """Convert POSIX cron day_of_week field to APScheduler CronTrigger format.

    POSIX cron uses 0/7=Sunday, 1=Monday, ..., 6=Saturday.
    APScheduler CronTrigger uses ISO 8601: 0=Monday, 1=Tuesday, ..., 6=Sunday.
    Conversion formula: apscheduler_value = (posix_value - 1) % 7

    Ranges that cross Sunday (POSIX 0) are expanded to individual ISO values
    and re-compressed with range notation so that APScheduler never receives an
    invalid ``start > end`` range (e.g. POSIX ``"0-4"`` → ISO ``"6,0-3"``).

    Handles:
    - Wildcard: ``*`` → ``*`` (no change)
    - Single values: ``1`` → ``0``, ``4`` → ``3``
    - Comma-separated: ``1,4`` → ``0,3``
    - Ranges (non-wrapping): ``1-5`` → ``0-4``
    - Ranges (Sunday-wrapping): ``0-4`` → ``6,0-3``
    - Step with wildcard base: ``*/2`` → ``*/2`` (no day conversion)
    - Step with range base: ``1-5/2`` → ``0-4/2`` (wrapping case: expanded)

    Args:
        dow: The day_of_week field string from a POSIX cron expression.

    Returns:
        Equivalent day_of_week string in APScheduler numbering.
    """
    if dow == "*":
        return dow

    result_parts: list[str] = []
    for part in dow.split(","):
        if "/" in part:
            base, step = part.split("/", 1)
            if base == "*":
                result_parts.append(part)  # */n — no day index conversion needed
            elif "-" in base:
                start, end = base.split("-", 1)
                iso_values = _posix_range_to_iso_values(int(start), int(end), int(step))
                result_parts.extend(_iso_values_to_parts(iso_values))
            else:
                result_parts.append(f"{(int(base) - 1) % 7}/{step}")
        elif "-" in part:
            start, end = part.split("-", 1)
            iso_values = _posix_range_to_iso_values(int(start), int(end))
            new_start_int = (int(start) - 1) % 7
            new_end_int = (int(end) - 1) % 7
            if new_start_int <= new_end_int:
                # No wrap-around: keep compact range notation
                result_parts.append(f"{new_start_int}-{new_end_int}")
            else:
                # Wrap-around (e.g. 0-4 → 6,0-3): expand and re-compress
                result_parts.extend(_iso_values_to_parts(iso_values))
        else:
            result_parts.append(str((int(part) - 1) % 7))

    return ",".join(result_parts)


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
        # Convert POSIX day_of_week (0/7=Sun, 1=Mon … 6=Sat) to APScheduler's
        # ISO 8601 format (0=Mon … 6=Sun) so that e.g. "1,4" (Mon+Thu) fires
        # on Monday and Thursday, not Tuesday and Friday.
        apsched_dow = _posix_dow_to_apsched(parts[4])
        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=apsched_dow,
        )
    except Exception as e:
        logger.warning("Failed to create CronTrigger from '%s': %s", s, e)
        return None
