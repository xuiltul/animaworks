#!/usr/bin/env python3
"""Git ops via subprocess; log to /tmp/git_ops_output.txt (+ mirror); self-delete."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/home/main/dev/animaworks-bak")
OUT = Path("/tmp/git_ops_output.txt")
MIRROR = REPO / "_git_ops_output_mirror.txt"
SELF = Path(__file__).resolve()

COMMIT_MSG = """fix: repair unit test suite — mock mismatches, stale baselines, asset caching

- Heartbeat decomposition: mock _get_current_state_max_chars to avoid MagicMock > int comparison
- Codex tests: mock _run_codex_command to prevent real CLI invocation
- Supervisor route: create identity.md in test helper for identity check
- Asset route: implement ETag + Cache-Control (max-age) for static assets
- Config routes: set auto_apply_presets=True; update i18n label for anthropic_auth
- App lifespan: set usage_governor=None on mock state
- Slack socket: add get_credential mock for shared handler test
- Usage governor: fix fetch mock signatures to accept skip_cache kwarg
- Codex SDK executor: adapt PATH assertion for Linux/POSIX
- i18n hardcode: update violation baselines for new files
- Tooling tests: update dispatch count, prompt_db assertions, schema expectations
"""


def log(line: str = "") -> None:
    text = line + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(text)
    try:
        with MIRROR.open("a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


def run_git(args: list[str]) -> int:
    log(f"$ git {' '.join(args)}")
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        log(proc.stdout.rstrip())
    if proc.stderr:
        log("[stderr]")
        log(proc.stderr.rstrip())
    log(f"[exit {proc.returncode}]")
    log("")
    return proc.returncode


def main() -> int:
    OUT.write_text("", encoding="utf-8")
    try:
        MIRROR.write_text("", encoding="utf-8")
    except OSError:
        pass
    log("=== git ops ===")
    log("")

    for cmd in (
        ["status", "--short"],
        ["diff", "--stat"],
        ["log", "--oneline", "-5"],
        ["add", "-A"],
    ):
        if run_git(cmd) != 0:
            return 1

    msg_path = Path("/tmp/git_ops_commit_msg.txt")
    msg_path.write_text(COMMIT_MSG, encoding="utf-8")
    log(f"$ git commit -F {msg_path}")
    proc = subprocess.run(
        ["git", "commit", "-F", str(msg_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        log(proc.stdout.rstrip())
    if proc.stderr:
        log("[stderr]")
        log(proc.stderr.rstrip())
    log(f"[exit {proc.returncode}]")
    log("")
    try:
        msg_path.unlink(missing_ok=True)
    except OSError:
        pass
    if proc.returncode != 0:
        return proc.returncode

    if run_git(["push", "origin", "main"]) != 0:
        return 1

    run_git(["status"])

    try:
        SELF.unlink()
        log(f"deleted {SELF}")
    except OSError as e:
        log(f"failed to delete self: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
