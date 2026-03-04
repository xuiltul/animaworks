from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Auto-updater for Claude Code CLI and claude-agent-sdk.

Periodically checks for new versions and installs updates automatically.
When ``claude-agent-sdk`` is updated, Mode S Animas are restarted so they
pick up the new bundled CLI.

Integrated into the server's APScheduler via :func:`run_update_check`.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.supervisor.manager import ProcessSupervisor

logger = logging.getLogger("animaworks.auto_updater")

SDK_PACKAGE = "claude-agent-sdk"

# ── Version queries ─────────────────────────────────────────


def _run(cmd: list[str], *, timeout: int = 30) -> str:
    """Run a command and return stripped stdout, or '' on failure."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("Command failed %s: %s", cmd, exc)
        return ""


def installed_sdk_version() -> str:
    raw = _run(["pip3", "show", SDK_PACKAGE])
    for line in raw.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return ""


def latest_sdk_version() -> str:
    raw = _run(["pip3", "index", "versions", SDK_PACKAGE])
    if not raw:
        return ""
    first_line = raw.splitlines()[0]
    # Format: "claude-agent-sdk (0.1.44)"
    start = first_line.find("(")
    end = first_line.find(")")
    if start >= 0 and end > start:
        return first_line[start + 1 : end]
    return ""


def installed_cli_version() -> str:
    raw = _run(["claude", "--version"])
    return raw.split()[0] if raw else ""


def latest_cli_version() -> str:
    return _run(["npm", "view", "@anthropic-ai/claude-code", "version"])


# ── Update actions ──────────────────────────────────────────


def upgrade_sdk() -> bool:
    """Upgrade claude-agent-sdk via pip. Returns True on success."""
    r = subprocess.run(
        ["pip3", "install", "--break-system-packages", "--quiet", "--upgrade", SDK_PACKAGE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0


def upgrade_cli() -> bool:
    """Upgrade Claude Code CLI. Returns True on success."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        logger.warning("claude binary not found; skipping CLI update")
        return False
    r = subprocess.run(
        ["claude", "update"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0


# ── Mode S Anima detection ──────────────────────────────────


def _mode_s_anima_names(animas_dir: Path) -> list[str]:
    """Return names of enabled Animas using Mode S (claude-* models)."""
    names: list[str] = []
    if not animas_dir.is_dir():
        return names
    for status_file in sorted(animas_dir.glob("*/status.json")):
        try:
            data = json.loads(status_file.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("enabled") and str(data.get("model", "")).startswith("claude-"):
            names.append(status_file.parent.name)
    return names


# ── Main check routine ──────────────────────────────────────


async def run_update_check(
    supervisor: ProcessSupervisor | None = None,
    animas_dir: Path | None = None,
) -> dict[str, str]:
    """Check for and apply updates. Returns a summary dict.

    Designed to be called from APScheduler or manually.
    Runs blocking subprocess calls in a thread executor.
    """
    loop = asyncio.get_running_loop()
    result: dict[str, str] = {}

    # SDK check
    cur_sdk = await loop.run_in_executor(None, installed_sdk_version)
    new_sdk = await loop.run_in_executor(None, latest_sdk_version)

    sdk_updated = False
    if cur_sdk and new_sdk and cur_sdk != new_sdk:
        logger.info("claude-agent-sdk update available: %s → %s", cur_sdk, new_sdk)
        ok = await loop.run_in_executor(None, upgrade_sdk)
        if ok:
            final = await loop.run_in_executor(None, installed_sdk_version)
            logger.info("claude-agent-sdk upgraded: %s → %s", cur_sdk, final)
            result["sdk"] = f"{cur_sdk} → {final}"
            sdk_updated = True
        else:
            logger.error("claude-agent-sdk upgrade failed")
            result["sdk"] = "upgrade failed"
    else:
        result["sdk"] = f"{cur_sdk} (up to date)"

    # CLI check
    cur_cli = await loop.run_in_executor(None, installed_cli_version)
    new_cli = await loop.run_in_executor(None, latest_cli_version)

    if cur_cli and new_cli and cur_cli != new_cli:
        logger.info("Claude Code CLI update available: %s → %s", cur_cli, new_cli)
        ok = await loop.run_in_executor(None, upgrade_cli)
        if ok:
            final = await loop.run_in_executor(None, installed_cli_version)
            logger.info("Claude Code CLI upgraded: %s → %s", cur_cli, final)
            result["cli"] = f"{cur_cli} → {final}"
        else:
            logger.error("Claude Code CLI upgrade failed")
            result["cli"] = "upgrade failed"
    else:
        result["cli"] = f"{cur_cli} (up to date)"

    # Restart Mode S Animas if SDK was updated (bundled CLI changes)
    if sdk_updated and supervisor and animas_dir:
        names = await loop.run_in_executor(None, _mode_s_anima_names, animas_dir)
        if names:
            logger.info("Restarting %d Mode S Anima(s) after SDK update: %s", len(names), names)
            restarted: list[str] = []
            for name in names:
                try:
                    await supervisor.restart_anima(name)
                    restarted.append(name)
                    await asyncio.sleep(2)
                except Exception:
                    logger.exception("Failed to restart Anima: %s", name)
            result["restarted"] = ", ".join(restarted) if restarted else "none"
        else:
            result["restarted"] = "no Mode S Animas"

    return result
