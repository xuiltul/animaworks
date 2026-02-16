"""CLI commands for person process management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_person_restart(args: argparse.Namespace) -> None:
    """Restart a specific person process."""
    import requests

    from core.paths import get_data_dir, get_run_dir

    # Check if server is running
    run_dir = get_run_dir()
    pid_file = run_dir / "server.pid"

    if not pid_file.exists():
        print("Error: Server is not running")
        sys.exit(1)

    # Use gateway URL if provided, otherwise default to localhost
    gateway_url = args.gateway_url or "http://localhost:18500"

    try:
        response = requests.post(
            f"{gateway_url}/api/persons/{args.person}/restart",
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        print(f"Person '{args.person}' restarted successfully")
        print(f"PID: {result.get('pid', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to restart person: {e}")
        sys.exit(1)


def cmd_person_status(args: argparse.Namespace) -> None:
    """Show status of person processes."""
    import requests

    from core.paths import get_run_dir

    # Check if server is running
    run_dir = get_run_dir()
    pid_file = run_dir / "server.pid"

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
        if args.person:
            # Specific person
            response = requests.get(
                f"{gateway_url}/api/persons/{args.person}/status",
                timeout=10.0
            )
            response.raise_for_status()
            status = response.json()
            _print_person_status(args.person, status)
        else:
            # All persons
            response = requests.get(
                f"{gateway_url}/api/persons",
                timeout=10.0
            )
            response.raise_for_status()
            persons = response.json()

            print(f"\nTotal persons: {len(persons)}")
            print("-" * 60)

            for person in persons:
                name = person.get("name", "unknown")
                # Get individual status
                try:
                    status_resp = requests.get(
                        f"{gateway_url}/api/persons/{name}/status",
                        timeout=5.0
                    )
                    status_resp.raise_for_status()
                    status = status_resp.json()
                    _print_person_status(name, status)
                except Exception as e:
                    print(f"\n{name}:")
                    print(f"  Status: ERROR ({e})")
                print()

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to get status: {e}")
        sys.exit(1)


def _print_person_status(name: str, status: dict) -> None:
    """Print formatted person status."""
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
