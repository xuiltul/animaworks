"""CLI commands for viewing anima logs."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_logs(args: argparse.Namespace) -> None:
    """View anima logs (tail -f style)."""
    from core.paths import get_data_dir

    log_dir = get_data_dir() / "logs"

    if args.all:
        # Show all logs (server + all animas)
        _tail_all_logs(log_dir)
    else:
        # Show specific anima log
        if not args.anima:
            print("Error: --anima is required (or use --all)")
            sys.exit(1)

        _tail_anima_log(
            log_dir=log_dir,
            anima_name=args.anima,
            lines=args.lines,
            date=args.date
        )


def _tail_anima_log(
    log_dir: Path,
    anima_name: str,
    lines: int = 50,
    date: str | None = None
) -> None:
    """Tail a specific anima's log file."""
    anima_log_dir = log_dir / "animas" / anima_name

    if not anima_log_dir.exists():
        print(f"Error: No log directory for anima '{anima_name}'")
        print(f"Expected: {anima_log_dir}")
        sys.exit(1)

    # Determine log file
    if date:
        log_file = anima_log_dir / f"{date}.log"
        if not log_file.exists():
            print(f"Error: No log file for date {date}")
            sys.exit(1)
        follow = False
    else:
        # Use current.log symlink or find latest
        current_link = anima_log_dir / "current.log"
        if current_link.exists():
            if current_link.is_symlink():
                log_file = anima_log_dir / current_link.readlink()
            else:
                # Fallback: read text file reference
                target_name = current_link.read_text().strip()
                log_file = anima_log_dir / target_name
        else:
            # Find latest log file
            log_files = sorted(
                anima_log_dir.glob("*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not log_files:
                print(f"Error: No log files found in {anima_log_dir}")
                sys.exit(1)
            log_file = log_files[0]
        follow = True

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Tailing log: {log_file}")
    print("-" * 60)

    # Show last N lines
    _show_last_lines(log_file, lines)

    # Follow mode (like tail -f)
    if follow:
        try:
            _follow_file(log_file)
        except KeyboardInterrupt:
            print("\n[Stopped]")


def _tail_all_logs(log_dir: Path) -> None:
    """Tail all logs (server + all animas)."""
    # Find all anima log directories
    animas_log_dir = log_dir / "animas"

    if not animas_log_dir.exists():
        print("No anima logs found")
        return

    anima_dirs = [d for d in animas_log_dir.iterdir() if d.is_dir()]

    print(f"Monitoring {len(anima_dirs)} anima logs")
    print("-" * 60)

    # Collect all current log files
    log_files = {}

    # Server log
    server_log = log_dir / "server.log"
    if server_log.exists():
        log_files["[SERVER]"] = server_log

    # Anima logs
    for anima_dir in anima_dirs:
        anima_name = anima_dir.name
        current_link = anima_dir / "current.log"

        if current_link.exists():
            if current_link.is_symlink():
                log_file = anima_dir / current_link.readlink()
            else:
                target_name = current_link.read_text().strip()
                log_file = anima_dir / target_name

            if log_file.exists():
                log_files[f"[{anima_name}]"] = log_file

    if not log_files:
        print("No log files found")
        return

    # Show last 10 lines from each
    for prefix, log_file in log_files.items():
        print(f"\n{prefix} {log_file.name}")
        _show_last_lines(log_file, 10, prefix=prefix)

    print("\n" + "=" * 60)
    print("Following all logs... (Ctrl+C to stop)")
    print("=" * 60)

    # Follow all files
    try:
        _follow_multiple_files(log_files)
    except KeyboardInterrupt:
        print("\n[Stopped]")


def _show_last_lines(log_file: Path, n: int, prefix: str = "") -> None:
    """Show last N lines of a file."""
    try:
        lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
        for line in lines[-n:]:
            if prefix:
                print(f"{prefix} {line}")
            else:
                print(line)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")


def _follow_file(log_file: Path) -> None:
    """Follow a single log file (like tail -f)."""
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        # Seek to end
        f.seek(0, 2)

        while True:
            line = f.readline()
            if line:
                print(line.rstrip())
            else:
                time.sleep(0.1)


def _follow_multiple_files(log_files: dict[str, Path]) -> None:
    """Follow multiple log files simultaneously."""
    file_handles = {}

    # Open all files and seek to end
    for prefix, log_file in log_files.items():
        try:
            f = open(log_file, 'r', encoding='utf-8', errors='replace')
            f.seek(0, 2)  # Seek to end
            file_handles[prefix] = f
        except Exception as e:
            print(f"Error opening {log_file}: {e}")

    try:
        while True:
            any_output = False

            for prefix, f in file_handles.items():
                try:
                    line = f.readline()
                    if line:
                        print(f"{prefix} {line.rstrip()}")
                        any_output = True
                except Exception:
                    pass

            if not any_output:
                time.sleep(0.1)
    finally:
        # Close all files
        for f in file_handles.values():
            try:
                f.close()
            except Exception:
                pass
