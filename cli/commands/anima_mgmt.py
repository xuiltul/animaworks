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
from datetime import datetime
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
        response = requests.post(f"{gateway_url}/api/animas/{args.anima}/restart", timeout=30.0)
        response.raise_for_status()
        result = response.json()
        print(f"Anima '{args.anima}' restarted successfully")
        print(f"PID: {result.get('pid', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to restart anima: {e}")
        sys.exit(1)


def cmd_anima_reload(args: argparse.Namespace) -> None:
    """Hot-reload anima config from status.json without process restart."""
    import requests

    from core.paths import get_data_dir

    pid_file = get_data_dir() / "server.pid"
    if not pid_file.exists():
        print("Error: Server is not running")
        sys.exit(1)

    gateway_url = args.gateway_url or "http://localhost:18500"

    if args.all:
        try:
            response = requests.post(
                f"{gateway_url}/api/animas/reload-all",
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            for name, r in result.get("results", {}).items():
                status = r.get("status", "unknown")
                if status == "ok":
                    changes = r.get("changes", [])
                    print(f"  {name}: reloaded (model={r.get('model', '?')}, changes={changes})")
                else:
                    print(f"  {name}: {r.get('error', status)}")
            print("All animas reloaded.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to reload all animas: {e}")
            sys.exit(1)
        return

    if not args.anima:
        print("Error: anima name is required (or use --all)")
        sys.exit(1)

    try:
        response = requests.post(
            f"{gateway_url}/api/animas/{args.anima}/reload",
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        changes = result.get("changes", [])
        model = result.get("model", "?")
        print(f"Anima '{args.anima}' config reloaded (model={model}, changes={changes})")
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to reload anima config: {e}")
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
            response = requests.get(f"{gateway_url}/api/animas/{args.anima}", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            _print_anima_status(args.anima, data.get("status", {}))
        else:
            # All animas
            response = requests.get(f"{gateway_url}/api/animas", timeout=10.0)
            response.raise_for_status()
            animas = response.json()

            print(f"\nTotal animas: {len(animas)}")
            print("-" * 60)

            for anima in animas:
                name = anima.get("name", "unknown")
                # Get individual status
                try:
                    status_resp = requests.get(f"{gateway_url}/api/animas/{name}", timeout=5.0)
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
    from core.paths import get_animas_dir

    print(f"\n{name}:")
    print(f"  State: {status.get('state', 'unknown')}")

    model, mode = _read_model_from_status_json(get_animas_dir() / name)
    if model:
        mode_suffix = f" (Mode {mode})" if mode else ""
        print(f"  Model: {model}{mode_suffix}")

    print(f"  PID: {status.get('pid', 'N/A')}")
    print(f"  Status: {status.get('status', 'unknown')}")

    if status.get("uptime_sec"):
        uptime = status["uptime_sec"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"  Uptime: {hours}h {minutes}m {seconds}s")

    if status.get("restart_count"):
        print(f"  Restarts: {status['restart_count']}")

    if status.get("active_label"):
        print(f"  Working on: {status['active_label']}")


def _read_model_from_status_json(anima_dir: Path) -> tuple[str, str]:
    """Read model and resolve execution mode from status.json.

    Returns:
        (model_name, execution_mode) — either may be empty string.
    """
    status_file = anima_dir / "status.json"
    if not status_file.exists():
        return ("", "")
    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
        model = data.get("model", "")
        if not model:
            return ("", "")
        mode = data.get("execution_mode", "")
        if not mode and model:
            try:
                from core.config.models import load_config, resolve_execution_mode

                mode = resolve_execution_mode(load_config(), model)
            except Exception:
                pass
        return (model, mode)
    except Exception:
        return ("", "")


def cmd_anima_info(args: argparse.Namespace) -> None:
    """Show detailed configuration for a specific anima from status.json."""
    from core.paths import get_animas_dir

    name: str = args.anima
    anima_dir = get_animas_dir() / name

    if not anima_dir.exists():
        print(f"Error: Anima '{name}' not found")
        sys.exit(1)

    status_file = anima_dir / "status.json"
    if not status_file.exists():
        print(f"Error: status.json not found for '{name}'")
        sys.exit(1)

    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading status.json: {exc}")
        sys.exit(1)

    if getattr(args, "json_output", False):
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    model = data.get("model", "-")
    explicit_mode = data.get("execution_mode", "")
    try:
        from core.config.models import load_config, resolve_execution_mode

        mode = resolve_execution_mode(
            load_config(),
            model if model != "-" else "",
            explicit_override=explicit_mode or None,
        )
    except Exception:
        mode = explicit_mode or "?"

    _MODE_LABELS = {"S": "S (SDK)", "C": "C (Codex)", "A": "A (Autonomous)", "B": "B (Basic)"}

    print(f"Anima:            {name}")
    print(f"Enabled:          {data.get('enabled', '?')}")
    print(f"Role:             {data.get('role', '-')}")
    print(f"Model:            {model}")
    print(f"Execution Mode:   {_MODE_LABELS.get(mode, mode)}")
    if data.get("credential"):
        print(f"Credential:       {data['credential']}")
    if data.get("fallback_model"):
        print(f"Fallback Model:   {data['fallback_model']}")
    print(f"Max Turns:        {data.get('max_turns', '-')}")
    print(f"Max Chains:       {data.get('max_chains', '-')}")
    if data.get("context_threshold"):
        print(f"Context Threshold: {data['context_threshold']}")
    if data.get("max_tokens"):
        print(f"Max Tokens:       {data['max_tokens']}")
    if data.get("llm_timeout"):
        print(f"LLM Timeout:      {data['llm_timeout']}s")
    if data.get("thinking"):
        print(f"Thinking:         {data['thinking']}")
    if data.get("thinking_effort"):
        print(f"Thinking Effort:  {data['thinking_effort']}")
    if data.get("supervisor"):
        print(f"Supervisor:       {data['supervisor']}")
    if data.get("mode_s_auth"):
        print(f"Mode S Auth:      {data['mode_s_auth']}")

    voice = data.get("voice")
    if voice and isinstance(voice, dict):
        print("\nVoice:")
        for k, v in voice.items():
            print(f"  {k}: {v}")


def cmd_anima_permissions(args: argparse.Namespace) -> None:
    """Show permissions configuration for an anima in human-readable format."""
    from core.config.models import _format_permissions_for_prompt, load_permissions
    from core.i18n import t
    from core.paths import get_animas_dir

    name: str = args.anima
    anima_dir = get_animas_dir() / name

    if not anima_dir.exists():
        print(t("cli.permissions_not_found", name=name))
        sys.exit(1)

    config = load_permissions(anima_dir)
    formatted = _format_permissions_for_prompt(config, name)
    print(formatted)

    perm_path = anima_dir / "permissions.json"
    if perm_path.exists():
        print(f"\n{t('cli.permissions_file_path', path=str(perm_path))}")
    else:
        md_path = anima_dir / "permissions.md"
        if md_path.exists():
            print(f"\n{t('cli.permissions_file_path', path=str(md_path))} (legacy, will migrate on load)")


def cmd_anima_delete(args: argparse.Namespace) -> None:
    """Delete an anima with optional archive."""
    import requests

    from core.config.models import unregister_anima_from_config
    from core.paths import get_animas_dir, get_data_dir
    from core.time_utils import now_jst

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
        answer = input(f"Are you sure you want to delete anima '{name}'? [y/N] ")
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
        timestamp = now_jst().strftime("%Y%m%d_%H%M%S")
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
                    print(f"Warning: Anima '{other_dir.name}' has deleted anima '{name}' as supervisor")
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


def cmd_anima_set_role(args: argparse.Namespace) -> None:
    """Change an anima's role."""
    import requests

    from core.anima_factory import SHARED_ROLES_DIR, VALID_ROLES, _apply_role_defaults
    from core.config.local_llm import apply_local_llm_role_to_status
    from core.config.models import load_config
    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    new_role = args.role
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    if new_role not in VALID_ROLES:
        print(f"Error: Invalid role '{new_role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
        print(f"Error: Anima '{name}' not found (missing identity.md)")
        sys.exit(1)

    # Read current status.json
    status_file = anima_dir / "status.json"
    status_data: dict = {}
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    old_role = status_data.get("role", "-")
    status_data["role"] = new_role

    if not args.status_only:
        # Merge defaults.json values into status.json
        defaults_path = SHARED_ROLES_DIR / new_role / "defaults.json"
        if defaults_path.is_file():
            try:
                role_defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
                for key in ("model", "context_threshold", "max_turns", "max_chains", "conversation_history_threshold"):
                    if key in role_defaults:
                        status_data[key] = role_defaults[key]
            except Exception:
                logger.warning("Failed to load role defaults for '%s'", new_role)
        apply_local_llm_role_to_status(status_data, load_config(), new_role)

    status_file.write_text(
        json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.status_only:
        # Re-apply role template files (specialty_prompt.md, permissions.md)
        _apply_role_defaults(anima_dir, new_role)
        print(f"Role changed: {old_role} → {new_role}")
        print("  Updated: status.json, specialty_prompt.md, permissions.md")
    else:
        print(f"Role changed: {old_role} → {new_role} (status.json only)")

    # Auto-restart if server is running and not suppressed
    server_running = (data_dir / "server.pid").exists()
    if server_running and not args.no_restart:
        try:
            response = requests.post(
                f"{gateway_url}/api/animas/{name}/restart",
                timeout=30.0,
            )
            response.raise_for_status()
            print(f"  Restarted '{name}' to apply new role.")
        except Exception as e:
            print(f"  Warning: Auto-restart failed ({e}).")
            print(f"  Run 'animaworks anima restart {name}' to apply changes.")
    elif not server_running:
        print("  Changes will take effect on next server start.")


def cmd_anima_set_model(args: argparse.Namespace) -> None:
    """Set an anima's model (updates status.json)."""
    from core.config.model_config import smart_update_model
    from core.paths import get_data_dir

    try:
        data_dir = get_data_dir()
        animas_dir = data_dir / "animas"
        pid_file = data_dir / "server.pid"

        if args.all:
            model = args.model or args.anima
            if not model:
                print("Error: model is required (e.g. animaworks anima set-model claude-sonnet-4-6 --all)")
                sys.exit(1)
            credential = args.credential
            updated = 0
            for entry in sorted(animas_dir.iterdir()):
                if not entry.is_dir():
                    continue
                status_file = entry / "status.json"
                if not status_file.exists():
                    continue
                try:
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    if not status_data.get("enabled", True):
                        continue
                except Exception:
                    continue
                try:
                    result = smart_update_model(entry, model=model, credential=credential)
                    updated += 1
                    cred_info = f" credential={result['credential']}" if result.get("family_changed") else ""
                    mode_info = f" mode={result['execution_mode']}"
                    print(f"  {entry.name}: model={model}{cred_info}{mode_info}")
                except Exception as e:
                    print(f"  {entry.name}: ERROR - {e}", file=sys.stderr)
            if updated == 0:
                print("No enabled animas found.")
                return
            print(f"Updated model for {updated} anima(s) to '{model}'")
        else:
            if not args.anima or not args.model:
                print(
                    "Error: anima name and model are required (e.g. animaworks anima set-model hinata claude-sonnet-4-6)"
                )
                sys.exit(1)
            anima_dir = animas_dir / args.anima
            if not anima_dir.exists():
                print(f"Error: Anima '{args.anima}' not found")
                sys.exit(1)
            result = smart_update_model(
                anima_dir,
                model=args.model,
                credential=args.credential,
            )
            cred_info = f" (credential={result['credential']})" if result.get("family_changed") else ""
            mode_info = f" [mode={result['execution_mode']}]"
            print(f"Model updated to '{args.model}' for '{args.anima}'{cred_info}{mode_info}")

        if pid_file.exists():
            print("  Server is running. Restart animas to apply changes (animaworks anima restart <name>).")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_anima_set_background_model(args: argparse.Namespace) -> None:
    """Set an anima's background model for heartbeat/cron (updates status.json)."""
    from core.config.models import update_status_model
    from core.paths import get_data_dir

    try:
        data_dir = get_data_dir()
        animas_dir = data_dir / "animas"
        pid_file = data_dir / "server.pid"

        if args.clear:
            if args.all:
                updated = 0
                for entry in sorted(animas_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    status_file = entry / "status.json"
                    if not status_file.exists():
                        continue
                    try:
                        status_data = json.loads(status_file.read_text(encoding="utf-8"))
                        if not status_data.get("enabled", True):
                            continue
                    except Exception:
                        continue
                    try:
                        update_status_model(
                            entry,
                            background_model="",
                            background_credential="",
                        )
                        updated += 1
                        print(f"  {entry.name}: background_model cleared")
                    except Exception as e:
                        print(f"  {entry.name}: ERROR - {e}", file=sys.stderr)
                if updated == 0:
                    print("No enabled animas found.")
                    return
                print(f"Cleared background_model for {updated} anima(s)")
            else:
                name = args.anima
                if not name:
                    print("Error: anima name is required (or use --all)")
                    sys.exit(1)
                anima_dir = animas_dir / name
                if not anima_dir.exists():
                    print(f"Error: Anima '{name}' not found")
                    sys.exit(1)
                update_status_model(
                    anima_dir,
                    background_model="",
                    background_credential="",
                )
                print(f"Cleared background_model for '{name}'")
        elif args.all:
            model = args.model or args.anima
            if not model:
                print("Error: model is required (e.g. animaworks anima set-background-model claude-sonnet-4-6 --all)")
                sys.exit(1)
            credential = args.credential
            updated = 0
            for entry in sorted(animas_dir.iterdir()):
                if not entry.is_dir():
                    continue
                status_file = entry / "status.json"
                if not status_file.exists():
                    continue
                try:
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    if not status_data.get("enabled", True):
                        continue
                except Exception:
                    continue
                try:
                    kwargs: dict = {"background_model": model}
                    if credential:
                        kwargs["background_credential"] = credential
                    update_status_model(entry, **kwargs)
                    updated += 1
                    print(f"  {entry.name}: background_model={model}")
                except Exception as e:
                    print(f"  {entry.name}: ERROR - {e}", file=sys.stderr)
            if updated == 0:
                print("No enabled animas found.")
                return
            print(f"Updated background_model for {updated} anima(s) to '{model}'")
        else:
            if not args.anima or not args.model:
                print(
                    "Error: anima name and model are required "
                    "(e.g. animaworks anima set-background-model hinata claude-sonnet-4-6)"
                )
                sys.exit(1)
            anima_dir = animas_dir / args.anima
            if not anima_dir.exists():
                print(f"Error: Anima '{args.anima}' not found")
                sys.exit(1)
            kwargs_update: dict = {"background_model": args.model}
            if args.credential:
                kwargs_update["background_credential"] = args.credential
            update_status_model(anima_dir, **kwargs_update)
            print(f"Background model updated to '{args.model}' for '{args.anima}'")

        if pid_file.exists():
            print("  Server is running. Restart animas to apply changes (animaworks anima restart <name>).")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_anima_set_outbound_limit(args: argparse.Namespace) -> None:
    """Set or clear per-Anima outbound message limits in status.json."""
    from core.i18n import t
    from core.paths import get_data_dir

    data_dir = get_data_dir()
    name = args.name
    anima_dir = data_dir / "animas" / name
    status_path = anima_dir / "status.json"

    if not anima_dir.exists():
        print(f"Error: Anima '{name}' not found")
        sys.exit(1)

    if not status_path.is_file():
        print(f"Error: status.json not found for '{name}'")
        sys.exit(1)

    data = json.loads(status_path.read_text(encoding="utf-8"))
    fields = ("max_outbound_per_hour", "max_outbound_per_day", "max_recipients_per_run")

    if args.clear:
        for f in fields:
            data.pop(f, None)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(status_path)
        print(t("cli.set_outbound_limit_cleared", name=name))
        return

    if args.per_hour is None and args.per_day is None and args.per_run is None:
        print("Error: Specify at least one of --per-hour, --per-day, --per-run, or --clear")
        sys.exit(1)

    details = []
    if args.per_hour is not None:
        data["max_outbound_per_hour"] = args.per_hour
        details.append(f"per_hour={args.per_hour}")
    if args.per_day is not None:
        data["max_outbound_per_day"] = args.per_day
        details.append(f"per_day={args.per_day}")
    if args.per_run is not None:
        data["max_recipients_per_run"] = args.per_run
        details.append(f"per_run={args.per_run}")

    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(status_path)
    print(t("cli.set_outbound_limit_success", name=name, details=", ".join(details)))


def _parse_since(raw: str | None) -> datetime | None:
    """Parse ``--since HH:MM`` into a timezone-aware datetime (today, JST)."""
    if not raw:
        return None
    from datetime import time as _time

    from core.memory.activity import now_local

    now = now_local()
    try:
        parts = raw.strip().split(":")
        t_obj = _time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        print(f"Error: invalid --since format '{raw}' (expected HH:MM)")
        sys.exit(1)
    return datetime.combine(now.date(), t_obj, tzinfo=now.tzinfo)


def _parse_date(raw: str | None) -> tuple[datetime, datetime] | None:
    """Parse ``--date`` into (since, until) as start/end of that day.

    Accepts YYYY-MM-DD, 'today', or 'yesterday'.
    Returns None if *raw* is falsy.
    """
    if not raw:
        return None
    from datetime import date as _date
    from datetime import time as _time
    from datetime import timedelta

    from core.memory.activity import now_local

    now = now_local()
    val = raw.strip().lower()
    if val == "today":
        target = now.date()
    elif val == "yesterday":
        target = now.date() - timedelta(days=1)
    else:
        try:
            target = _date.fromisoformat(val)
        except ValueError:
            print(f"Error: invalid --date format '{raw}' (expected YYYY-MM-DD, 'today', or 'yesterday')")
            sys.exit(1)

    tz = now.tzinfo
    since = datetime.combine(target, _time(0, 0), tzinfo=tz)
    until = datetime.combine(target, _time(23, 59, 59), tzinfo=tz)
    return since, until


def cmd_anima_audit(args: argparse.Namespace) -> None:
    """Audit a subordinate anima's recent activity."""
    from core.paths import get_animas_dir

    name: str | None = args.anima
    audit_all: bool = getattr(args, "audit_all", False)
    days: int = max(1, min(getattr(args, "days", 1), 30))
    since = _parse_since(getattr(args, "since", None))
    until: datetime | None = None
    hours = days * 24

    date_range = _parse_date(getattr(args, "date", None))
    if date_range:
        since, until = date_range
        hours = 24

    if not name and not audit_all:
        print("Error: specify an anima name or use --all")
        sys.exit(1)

    animas_dir = get_animas_dir()

    from core.memory.audit import AuditAggregator

    if audit_all:
        dirs = sorted([d for d in animas_dir.iterdir() if d.is_dir() and (d / "identity.md").exists()])
        if not dirs:
            print("No animas found.")
            sys.exit(1)
        print(AuditAggregator.generate_merged_timeline(dirs, hours=hours, since=since, until=until))
        return

    if name:
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / "identity.md").exists():
            print(f"Error: Anima '{name}' not found")
            sys.exit(1)
        agg = AuditAggregator(anima_dir)
        print(agg.generate_report(hours=hours, since=since, until=until))
        return


def cmd_anima_rename(args: argparse.Namespace) -> None:
    """Rename an anima (directory, config, references)."""
    import requests

    from core.anima_factory import validate_anima_name
    from core.config.models import rename_anima_in_config
    from core.paths import get_animas_dir, get_data_dir

    old_name: str = args.old_name
    new_name: str = args.new_name
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    old_dir = animas_dir / old_name
    new_dir = animas_dir / new_name
    shared_dir = data_dir / "shared"
    gateway_url = getattr(args, "gateway_url", None) or "http://localhost:18500"

    # ── Validation ──
    if old_name == new_name:
        print("Error: Old and new names are the same")
        sys.exit(1)

    if not old_dir.exists() or not (old_dir / "identity.md").exists():
        print(f"Error: Anima '{old_name}' not found (missing identity.md)")
        sys.exit(1)

    if new_dir.exists():
        print(f"Error: Anima '{new_name}' already exists")
        sys.exit(1)

    name_err = validate_anima_name(new_name)
    if name_err:
        print(f"Error: {name_err}")
        sys.exit(1)

    # ── Confirmation ──
    if not args.force:
        answer = input(f"Rename anima '{old_name}' → '{new_name}'? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    print(f"Renaming anima '{old_name}' → '{new_name}'...")

    # ── Server check: disable old anima if running ──
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()
    if server_running:
        try:
            requests.post(
                f"{gateway_url}/api/animas/{old_name}/disable",
                timeout=10,
            )
            print(f"  Stopped anima process '{old_name}'")
        except Exception as e:
            logger.warning("Failed to disable anima via API: %s", e)

    rollback_needed = False
    try:
        # ── Filesystem: rename anima directory ──
        old_dir.rename(new_dir)
        rollback_needed = True
        print(f"  Renamed directory: animas/{old_name} → animas/{new_name}")

        # ── Filesystem: rename inbox ──
        old_inbox = shared_dir / "inbox" / old_name
        if old_inbox.exists():
            new_inbox = shared_dir / "inbox" / new_name
            old_inbox.rename(new_inbox)
            print(f"  Renamed inbox: shared/inbox/{old_name} → shared/inbox/{new_name}")

        # ── Filesystem: rename DM logs ──
        dm_count = _rename_dm_logs(shared_dir, old_name, new_name)
        if dm_count:
            print(f"  Renamed {dm_count} DM log file(s)")

        # ── Filesystem: clean up stale socket/pid ──
        run_dir = data_dir / "run"
        for stale in (
            run_dir / "sockets" / f"{old_name}.sock",
            run_dir / "animas" / f"{old_name}.pid",
        ):
            if stale.exists():
                stale.unlink(missing_ok=True)

        # ── Config: update config.json ──
        try:
            sup_count = rename_anima_in_config(data_dir, old_name, new_name)
            parts = ["key"]
            if sup_count:
                parts.append(f"{sup_count} supervisor reference(s)")
            print(f"  Updated config.json ({' + '.join(parts)})")
        except Exception as e:
            print(f"  Warning: Failed to update config.json: {e}")

        # ── Config: update status.json supervisor refs ──
        status_updated = 0
        for other_dir in animas_dir.iterdir():
            if not other_dir.is_dir():
                continue
            status_file = other_dir / "status.json"
            if not status_file.exists():
                continue
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                if status_data.get("supervisor") == old_name:
                    status_data["supervisor"] = new_name
                    tmp = status_file.with_suffix(".tmp")
                    tmp.write_text(
                        json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    tmp.replace(status_file)
                    status_updated += 1
            except Exception:
                pass
        if status_updated:
            print(f"  Updated status.json for {status_updated} anima(s) with supervisor reference")

        # ── RAG: cleanup old collections ──
        _cleanup_rag_collections(new_dir, old_name)
        print("  Cleared RAG index (will re-index on next startup)")

        # ── Server: reload + restart ──
        if server_running:
            try:
                requests.post(f"{gateway_url}/api/system/reload", timeout=10)
            except Exception:
                pass
            try:
                requests.post(
                    f"{gateway_url}/api/animas/{new_name}/enable",
                    timeout=10,
                )
                requests.post(
                    f"{gateway_url}/api/animas/{new_name}/restart",
                    timeout=30,
                )
                print(f"  Restarted anima '{new_name}'")
            except Exception as e:
                logger.warning("Failed to restart renamed anima: %s", e)

    except OSError as e:
        if rollback_needed and new_dir.exists() and not old_dir.exists():
            try:
                new_dir.rename(old_dir)
                print("  Rolled back directory rename")
            except OSError:
                pass
        print(f"Error: Failed to rename: {e}")
        sys.exit(1)

    print(f"Anima renamed successfully: {old_name} → {new_name}")


def _rename_dm_logs(shared_dir: Path, old_name: str, new_name: str) -> int:
    """Rename DM log files referencing *old_name*. Returns count of renamed files."""
    dm_dir = shared_dir / "dm_logs"
    if not dm_dir.exists():
        return 0
    count = 0
    for f in sorted(dm_dir.glob("*.jsonl")):
        stem = f.stem
        parts = stem.split("-", 1)
        if len(parts) != 2:
            continue
        if old_name not in parts:
            continue
        new_parts = [new_name if p == old_name else p for p in parts]
        new_parts.sort()
        new_path = dm_dir / f"{new_parts[0]}-{new_parts[1]}.jsonl"
        if new_path.exists():
            with new_path.open("a", encoding="utf-8") as dst:
                dst.write(f.read_text(encoding="utf-8"))
            f.unlink()
        else:
            f.rename(new_path)
        count += 1
    return count


def _cleanup_rag_collections(anima_dir: Path, old_name: str) -> None:
    """Delete old RAG collections and reset index_meta for re-indexing."""
    index_meta = anima_dir / "index_meta.json"
    if index_meta.exists():
        index_meta.write_text("{}\n", encoding="utf-8")

    try:
        from core.memory.rag.store import ChromaVectorStore

        vectordb_dir = anima_dir / "vectordb"
        if vectordb_dir.is_dir():
            store = ChromaVectorStore(persist_dir=vectordb_dir)
            for suffix in ("knowledge", "episodes", "procedures", "skills", "common_knowledge", "conversation_summary"):
                collection_name = f"{old_name}_{suffix}"
                try:
                    store.delete_collection(collection_name)
                except Exception:
                    pass
    except Exception:
        pass


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

                print(f"{'Name':<20} {'Enabled':<10} {'Model':<30} {'Supervisor':<20}")
                print("-" * 80)
                for anima in animas:
                    name = anima.get("name", "unknown")
                    enabled = anima.get("enabled", "?")
                    model = anima.get("model", "-") or "-"
                    supervisor = anima.get("supervisor", "-")
                    print(f"{name:<20} {str(enabled):<10} {model:<30} {supervisor or '-':<20}")
                print(f"\nTotal: {len(animas)}")
                return
            except Exception:
                pass

    # Local fallback: scan filesystem
    if not animas_dir.exists():
        print("No animas directory found.")
        return

    count = 0
    print(f"{'Name':<20} {'Enabled':<10} {'Role':<15} {'Model':<30}")
    print("-" * 75)

    for entry in sorted(animas_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "identity.md").exists():
            continue

        name = entry.name
        enabled = "?"
        role = "-"
        model = "-"

        status_file = entry / "status.json"
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                enabled = str(status_data.get("enabled", "?"))
                role = status_data.get("role", "-")
                model = status_data.get("model", "-") or "-"
            except Exception:
                pass

        print(f"{name:<20} {enabled:<10} {role:<15} {model:<30}")
        count += 1

    print(f"\nTotal: {count}")
