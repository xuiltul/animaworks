"""CLI commands for anima process management."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_anima_restart(args: argparse.Namespace) -> None:
    """Restart a specific anima process."""
    import requests

    from core.paths import get_data_dir

    # Check if server is running
    pid_file = get_data_dir() / "server.pid"

    if not pid_file.exists():
        print("Error: Server is not running")
        sys.exit(1)

    # Use gateway URL if provided, otherwise default to localhost
    gateway_url = args.gateway_url or "http://localhost:18500"

    try:
        response = requests.post(
            f"{gateway_url}/api/animas/{args.anima}/restart",
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        print(f"Anima '{args.anima}' restarted successfully")
        print(f"PID: {result.get('pid', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to restart anima: {e}")
        sys.exit(1)


def cmd_anima_status(args: argparse.Namespace) -> None:
    """Show status of anima processes."""
    import requests

    from core.paths import get_data_dir

    # Check if server is running
    pid_file = get_data_dir() / "server.pid"

    if not pid_file.exists():
        print("Server is not running")
        return

    # Read PID
    try:
        server_pid = int(pid_file.read_text().strip())
        print(f"Server PID: {server_pid}")
    except Exception:
        print("Server PID file corrupted")

    # Use gateway URL if provided
    gateway_url = args.gateway_url or "http://localhost:18500"

    try:
        # Get status from API
        if args.anima:
            # Specific anima
            response = requests.get(
                f"{gateway_url}/api/animas/{args.anima}",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            _print_anima_status(args.anima, data.get("status", {}))
        else:
            # All animas
            response = requests.get(
                f"{gateway_url}/api/animas",
                timeout=10.0
            )
            response.raise_for_status()
            animas = response.json()

            print(f"\nTotal animas: {len(animas)}")
            print("-" * 60)

            for anima in animas:
                name = anima.get("name", "unknown")
                # Get individual status
                try:
                    status_resp = requests.get(
                        f"{gateway_url}/api/animas/{name}",
                        timeout=5.0
                    )
                    status_resp.raise_for_status()
                    data = status_resp.json()
                    _print_anima_status(name, data.get("status", {}))
                except Exception as e:
                    print(f"\n{name}:")
                    print(f"  Status: ERROR ({e})")
                print()

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to get status: {e}")
        sys.exit(1)


def _print_anima_status(name: str, status: dict) -> None:
    """Print formatted anima status."""
    print(f"\n{name}:")
    print(f"  State: {status.get('state', 'unknown')}")
    print(f"  PID: {status.get('pid', 'N/A')}")
    print(f"  Status: {status.get('status', 'unknown')}")

    if status.get('uptime_sec'):
        uptime = status['uptime_sec']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"  Uptime: {hours}h {minutes}m {seconds}s")

    if status.get('restart_count'):
        print(f"  Restarts: {status['restart_count']}")

    if status.get('current_task'):
        print(f"  Current task: {status['current_task']}")


def cmd_anima_delete(args: argparse.Namespace) -> None:
    """Delete an anima with optional archive."""
    from datetime import datetime

    import requests

    from core.config.models import unregister_anima_from_config
    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    # Validate anima exists
    if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
        print(f"Error: Anima '{name}' not found (missing identity.md)")
        sys.exit(1)

    # Confirmation prompt
    if not args.force:
        answer = input(
            f"Are you sure you want to delete anima '{name}'? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    # Try to disable via server if running
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    if server_running:
        try:
            requests.post(
                f"{gateway_url}/api/animas/{name}/disable",
                timeout=10,
            )
        except Exception as e:
            logger.warning("Failed to disable anima via API: %s", e)

    # Archive before deletion
    if not args.no_archive:
        archive_dir = data_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = archive_dir / f"{name}_{timestamp}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(anima_dir))
        print(f"Archived to: {zip_path}")

    # Delete directory
    shutil.rmtree(anima_dir)

    # Unregister from config
    unregister_anima_from_config(data_dir, name)

    # Check for orphaned supervisor references
    for other_dir in animas_dir.iterdir():
        if not other_dir.is_dir():
            continue
        status_file = other_dir / "status.json"
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                if status_data.get("supervisor") == name:
                    print(
                        f"Warning: Anima '{other_dir.name}' has "
                        f"deleted anima '{name}' as supervisor"
                    )
            except Exception:
                pass

    # Reload server config if running
    if server_running:
        try:
            requests.post(f"{gateway_url}/api/system/reload", timeout=10)
        except Exception:
            pass

    print(f"Anima '{name}' deleted successfully.")


def cmd_anima_disable(args: argparse.Namespace) -> None:
    """Disable an anima."""
    import requests

    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    # Validate anima exists
    if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
        print(f"Error: Anima '{name}' not found (missing identity.md)")
        sys.exit(1)

    # Check if server is running
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    api_success = False
    if server_running:
        try:
            response = requests.post(
                f"{gateway_url}/api/animas/{name}/disable",
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            print(f"Disabled anima '{name}': {result}")
            api_success = True
        except Exception:
            pass

    if not api_success:
        # Direct file update (offline mode)
        status_file = anima_dir / "status.json"
        status_data: dict = {}
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        status_data["enabled"] = False
        status_file.write_text(
            json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Disabled anima '{name}' (offline mode)")


def cmd_anima_enable(args: argparse.Namespace) -> None:
    """Enable an anima."""
    import requests

    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    # Validate anima exists
    if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
        print(f"Error: Anima '{name}' not found (missing identity.md)")
        sys.exit(1)

    # Check if server is running
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    api_success = False
    if server_running:
        try:
            response = requests.post(
                f"{gateway_url}/api/animas/{name}/enable",
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            print(f"Enabled anima '{name}': {result}")
            api_success = True
        except Exception:
            pass

    if not api_success:
        # Direct file update (offline mode)
        status_file = anima_dir / "status.json"
        status_data: dict = {}
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        status_data["enabled"] = True
        status_file.write_text(
            json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Enabled anima '{name}' (offline mode)")


def cmd_anima_list(args: argparse.Namespace) -> None:
    """List all animas."""
    import requests

    from core.paths import get_animas_dir, get_data_dir

    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    # Try API first (unless --local)
    if not args.local:
        pid_file = data_dir / "server.pid"
        if pid_file.exists():
            try:
                response = requests.get(
                    f"{gateway_url}/api/animas",
                    timeout=10,
                )
                response.raise_for_status()
                animas = response.json()

                print(f"{'Name':<20} {'Enabled':<10} {'Supervisor':<20}")
                print("-" * 50)
                for anima in animas:
                    name = anima.get("name", "unknown")
                    enabled = anima.get("enabled", "?")
                    supervisor = anima.get("supervisor", "-")
                    print(f"{name:<20} {str(enabled):<10} {supervisor or '-':<20}")
                print(f"\nTotal: {len(animas)}")
                return
            except Exception:
                pass

    # Local fallback: scan filesystem
    if not animas_dir.exists():
        print("No animas directory found.")
        return

    count = 0
    print(f"{'Name':<20} {'Enabled':<10} {'Role':<15}")
    print("-" * 45)

    for entry in sorted(animas_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "identity.md").exists():
            continue

        name = entry.name
        enabled = "?"
        role = "-"

        status_file = entry / "status.json"
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                enabled = str(status_data.get("enabled", "?"))
                role = status_data.get("role", "-")
            except Exception:
                pass

        print(f"{name:<20} {enabled:<10} {role:<15}")
        count += 1

    print(f"\nTotal: {count}")
