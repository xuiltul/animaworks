"""CLI commands for anima process management."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cli._gateway import gateway_request
from core.anima.settings_store import update_status
from core.anima.workspace_files import IDENTITY_FILE

logger = logging.getLogger(__name__)


def cmd_anima_restart(args: argparse.Namespace) -> None:
    """Restart a specific anima process."""
    from core.paths import get_data_dir

    # Check if server is running
    pid_file = get_data_dir() / "server.pid"

    if not pid_file.exists():
        print("Error: Server is not running")
        sys.exit(1)

    try:
        response = gateway_request(
            args,
            "POST",
            f"/api/animas/{args.anima}/restart",
            timeout=30.0,
            raw_response=True,
        )
        response.raise_for_status()
        result = response.json()
        print(f"Anima '{args.anima}' restarted successfully")
        print(f"PID: {result.get('pid', 'N/A')}")
    except Exception as e:
        print(f"Error: Failed to restart anima: {e}")
        sys.exit(1)


def cmd_anima_reload(args: argparse.Namespace) -> None:
    """Hot-reload anima config from status.json without process restart."""
    from core.paths import get_data_dir

    pid_file = get_data_dir() / "server.pid"
    if not pid_file.exists():
        print("Error: Server is not running")
        sys.exit(1)

    if args.all:
        try:
            response = gateway_request(
                args,
                "POST",
                "/api/animas/reload-all",
                timeout=30.0,
                raw_response=True,
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
        except Exception as e:
            print(f"Error: Failed to reload all animas: {e}")
            sys.exit(1)
        return

    if not args.anima:
        print("Error: anima name is required (or use --all)")
        sys.exit(1)

    try:
        response = gateway_request(
            args,
            "POST",
            f"/api/animas/{args.anima}/reload",
            timeout=10.0,
            raw_response=True,
        )
        response.raise_for_status()
        result = response.json()
        changes = result.get("changes", [])
        model = result.get("model", "?")
        print(f"Anima '{args.anima}' config reloaded (model={model}, changes={changes})")
    except Exception as e:
        print(f"Error: Failed to reload anima config: {e}")
        sys.exit(1)


def cmd_anima_status(args: argparse.Namespace) -> None:
    """Show status of anima processes."""
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

    try:
        # Get status from API
        if args.anima:
            # Specific anima
            response = gateway_request(
                args,
                "GET",
                f"/api/animas/{args.anima}",
                timeout=10.0,
                raw_response=True,
            )
            response.raise_for_status()
            data = response.json()
            _print_anima_status(args.anima, data.get("status", {}))
        else:
            # All animas
            response = gateway_request(args, "GET", "/api/animas", timeout=10.0, raw_response=True)
            response.raise_for_status()
            animas = response.json()

            print(f"\nTotal animas: {len(animas)}")
            print("-" * 60)

            for anima in animas:
                name = anima.get("name", "unknown")
                # Get individual status
                try:
                    status_resp = gateway_request(
                        args,
                        "GET",
                        f"/api/animas/{name}",
                        timeout=5.0,
                        raw_response=True,
                    )
                    status_resp.raise_for_status()
                    data = status_resp.json()
                    _print_anima_status(name, data.get("status", {}))
                except Exception as e:
                    print(f"\n{name}:")
                    print(f"  Status: ERROR ({e})")
                print()

    except Exception as e:
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
    from core.platform.status_store import read_status

    data = read_status(anima_dir)
    model = data.get("model", "")
    if not model:
        return ("", "")
    mode = data.get("execution_mode", "")
    if not mode:
        try:
            from core.config.models import load_config, resolve_execution_mode

            mode = resolve_execution_mode(load_config(), model)
        except Exception:
            logger.debug("Best-effort operation failed", exc_info=True)
    return (model, mode)


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

    _MODE_LABELS = {
        "S": "S (SDK)",
        "C": "C (Codex)",
        "D": "D (Cursor Agent CLI)",
        "G": "G (Gemini CLI)",
        "X": "X (Grok Build CLI)",
        "Y": "Y (Antigravity CLI)",
        "A": "A (Autonomous)",
    }

    print(f"Anima:            {name}")
    print(f"Enabled:          {data.get('enabled', '?')}")
    print(f"Role:             {data.get('role', '-')}")
    print(f"Model:            {model}")
    print(f"Execution Mode:   {_MODE_LABELS.get(mode, mode)}")
    if data.get("credential"):
        print(f"Credential:       {data['credential']}")
    if data.get("fallback_model"):
        print(f"Fallback Model:   {data['fallback_model']}")
    if data.get("context_threshold"):
        print(f"Context Threshold: {data['context_threshold']}")
    if data.get("max_tokens"):
        print(f"Max Tokens:       {data['max_tokens']}")
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
    """Delete an anima through the running server or shared local service."""
    from core.anima.admin import delete_anima_files
    from core.paths import get_animas_dir, get_data_dir
    from core.platform.pid import read_server_pid
    from core.platform.process import is_process_alive

    name = args.anima
    if not name or ".." in name or "/" in name or "\\" in name:
        print("Error: Invalid anima name")
        sys.exit(1)

    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name
    if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
        print(f"Error: Anima '{name}' not found (missing IDENTITY.md)")
        sys.exit(1)

    if not getattr(args, "force", False):
        answer = input(f"Are you sure you want to delete anima '{name}'? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    archive = not getattr(args, "no_archive", False)
    pid_file = data_dir / "server.pid"
    server_pid = read_server_pid(data_dir)
    if pid_file.exists() and server_pid is None:
        print("Error: Server PID file is invalid; refusing to delete while server status is unknown")
        sys.exit(1)
    server_running = server_pid is not None and is_process_alive(server_pid)

    def report_success(archive_path: str | Path | None, supervisor_warnings: list[str]) -> None:
        if archive_path:
            print(f"Archived to: {archive_path}")
        for warning in supervisor_warnings:
            print(f"Warning: {warning}")
        print(f"Anima '{name}' deleted successfully.")

    if server_running:
        try:
            response = gateway_request(
                args,
                "DELETE",
                f"/api/animas/{name}?archive={str(archive).lower()}",
                timeout=30.0,
                raw_response=True,
            )
            response.raise_for_status()
            api_result = response.json()
            if not isinstance(api_result, dict) or api_result.get("status") != "deleted":
                detail = api_result.get("detail") if isinstance(api_result, dict) else None
                raise RuntimeError(str(detail or "server did not confirm anima deletion"))
            archive_path = api_result.get("archive_path")
            warnings = api_result.get("supervisor_warnings") or []
            report_success(
                archive_path if isinstance(archive_path, str) else None,
                [str(warning) for warning in warnings] if isinstance(warnings, list) else [],
            )
            return
        except Exception as exc:
            print(f"Error: Failed to delete anima: {exc}")
            sys.exit(1)

    result = delete_anima_files(data_dir, name, archive=archive)
    if result.archive_path is not None:
        print(f"Archived to: {result.archive_path}")
    if result.error:
        print(f"Error: Failed to delete anima: {result.error}")
        sys.exit(1)
    if not result.deleted:
        print("Error: Anima deletion did not complete")
        sys.exit(1)

    warnings = [
        f"Anima '{other_name}' has deleted anima '{name}' as supervisor" for other_name in result.supervisor_references
    ]
    report_success(None, warnings)


def cmd_anima_disable(args: argparse.Namespace) -> None:
    """Disable an anima."""
    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    # Validate anima exists
    if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
        print(f"Error: Anima '{name}' not found (missing IDENTITY.md)")
        sys.exit(1)

    # Check if server is running
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()

    if server_running:
        try:
            response = gateway_request(
                args,
                "POST",
                f"/api/animas/{name}/disable",
                timeout=10,
                raw_response=True,
            )
            response.raise_for_status()
            print(f"Disabled anima '{name}': {response.json()}")
        except Exception as exc:
            print(f"Error: Failed to disable through running server: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        update_status(anima_dir, lambda status: status.update(enabled=False))
    except (OSError, ValueError) as exc:
        print(f"Error updating status.json: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Disabled anima '{name}' (offline mode)")


def cmd_anima_enable(args: argparse.Namespace) -> None:
    """Enable an anima."""
    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    # Validate anima exists
    if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
        print(f"Error: Anima '{name}' not found (missing IDENTITY.md)")
        sys.exit(1)

    # Check if server is running
    pid_file = data_dir / "server.pid"
    server_running = pid_file.exists()

    if server_running:
        try:
            response = gateway_request(
                args,
                "POST",
                f"/api/animas/{name}/enable",
                timeout=10,
                raw_response=True,
            )
            response.raise_for_status()
            print(f"Enabled anima '{name}': {response.json()}")
        except Exception as exc:
            print(f"Error: Failed to enable through running server: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    try:
        update_status(anima_dir, lambda status: status.update(enabled=True))
    except (OSError, ValueError) as exc:
        print(f"Error updating status.json: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Enabled anima '{name}' (offline mode)")


def cmd_anima_set_role(args: argparse.Namespace) -> None:
    """Change an anima's role."""
    from core.anima.factory import SHARED_ROLES_DIR, VALID_ROLES, _apply_role_defaults
    from core.config.local_llm import apply_local_llm_role_to_status
    from core.config.models import load_config
    from core.paths import get_animas_dir, get_data_dir

    name = args.anima
    new_role = args.role
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    anima_dir = animas_dir / name

    if new_role not in VALID_ROLES:
        print(f"Error: Invalid role '{new_role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
        print(f"Error: Anima '{name}' not found (missing IDENTITY.md)")
        sys.exit(1)

    if (data_dir / "server.pid").exists():
        try:
            response = gateway_request(
                args,
                "PUT",
                f"/api/animas/{name}/role",
                json={"role": new_role, "status_only": bool(args.status_only)},
                timeout=30.0,
                raw_response=True,
            )
            response.raise_for_status()
            result = response.json()
            old_role = result.get("old_role", "-")
            if args.status_only:
                print(f"Role changed: {old_role} → {new_role} (status.json only)")
            else:
                print(f"Role changed: {old_role} → {new_role}")
                print("  Updated: status.json, specialty_prompt.md, permissions.json")
            if not args.no_restart:
                restart = gateway_request(
                    args,
                    "POST",
                    f"/api/animas/{name}/restart",
                    timeout=30.0,
                    raw_response=True,
                )
                restart.raise_for_status()
                print(f"  Restarted '{name}' to apply new role.")
            return
        except Exception as exc:
            print(f"Error: Failed to update role through running server: {exc}", file=sys.stderr)
            sys.exit(1)

    role_defaults: dict[str, object] = {}
    if not args.status_only:
        defaults_path = SHARED_ROLES_DIR / new_role / "defaults.json"
        if defaults_path.is_file():
            try:
                role_defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load role defaults for '%s'", new_role)
        config = load_config()

    old_role = "-"

    def apply_role(status_data: dict[str, object]) -> None:
        nonlocal old_role
        old_role = status_data.get("role", "-")
        status_data["role"] = new_role
        if not args.status_only:
            for key in ("model", "context_threshold", "conversation_history_threshold"):
                if key in role_defaults:
                    status_data[key] = role_defaults[key]
            apply_local_llm_role_to_status(status_data, config, new_role)

    try:
        update_status(anima_dir, apply_role)
    except (OSError, ValueError) as exc:
        print(f"Error updating status.json: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.status_only:
        # Re-apply role template files (specialty_prompt.md, permissions.json)
        _apply_role_defaults(anima_dir, new_role)
        print(f"Role changed: {old_role} → {new_role}")
        print("  Updated: status.json, specialty_prompt.md, permissions.json")
    else:
        print(f"Role changed: {old_role} → {new_role} (status.json only)")

    # Auto-restart if server is running and not suppressed
    server_running = (data_dir / "server.pid").exists()
    if server_running and not args.no_restart:
        try:
            response = gateway_request(
                args,
                "POST",
                f"/api/animas/{name}/restart",
                timeout=30.0,
                raw_response=True,
            )
            response.raise_for_status()
            print(f"  Restarted '{name}' to apply new role.")
        except Exception as e:
            print(f"  Warning: Auto-restart failed ({e}).")
            print(f"  Run 'animaworks anima restart {name}' to apply changes.")
    elif not server_running:
        print("  Changes will take effect on next server start.")


def _read_status_json(anima_dir: Path) -> dict[str, object]:
    from core.platform.status_store import read_status

    return read_status(anima_dir)


def _set_model_via_server(args: argparse.Namespace, name: str, model: str, credential: str | None) -> dict[str, Any]:
    response = gateway_request(
        args,
        "PUT",
        f"/api/animas/{name}/model",
        json={"model": model, "credential": credential or ""},
        timeout=30.0,
        raw_response=True,
    )
    response.raise_for_status()
    return response.json()


def cmd_anima_set_model(args: argparse.Namespace) -> None:
    """Set an anima's model (updates status.json)."""
    from core.config.model_config import smart_update_model
    from core.paths import get_data_dir

    try:
        data_dir = get_data_dir()
        animas_dir = data_dir / "animas"
        pid_file = data_dir / "server.pid"
        server_running = pid_file.exists()

        if args.all:
            model = args.model or args.anima
            if not model:
                print("Error: model is required (e.g. animaworks anima set-model claude-sonnet-5-5 --all)")
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
                    result = (
                        _set_model_via_server(args, entry.name, model, credential)
                        if server_running
                        else smart_update_model(entry, model=model, credential=credential)
                    )
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
                    "Error: anima name and model are required (e.g. animaworks anima set-model hinata claude-sonnet-5-5)"
                )
                sys.exit(1)
            anima_dir = animas_dir / args.anima
            if not anima_dir.exists():
                print(f"Error: Anima '{args.anima}' not found")
                sys.exit(1)
            result = (
                _set_model_via_server(args, args.anima, args.model, args.credential)
                if server_running
                else smart_update_model(
                    anima_dir,
                    model=args.model,
                    credential=args.credential,
                )
            )
            cred_info = f" (credential={result['credential']})" if result.get("family_changed") else ""
            mode_info = f" [mode={result['execution_mode']}]"
            print(f"Model updated to '{args.model}' for '{args.anima}'{cred_info}{mode_info}")

        if server_running:
            print("  Running anima processes were asked to reload the model; stopped animas will use it on start.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _set_background_model_via_server(
    args: argparse.Namespace,
    name: str,
    model: str,
    credential: str | None = None,
    *,
    clear: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "clear": clear}
    if credential is not None:
        payload["credential"] = credential
    response = gateway_request(
        args,
        "PUT",
        f"/api/animas/{name}/background-model",
        json=payload,
        timeout=30.0,
        raw_response=True,
    )
    response.raise_for_status()
    return response.json()


def cmd_anima_set_background_model(args: argparse.Namespace) -> None:
    """Set an anima's background model for heartbeat/cron (updates status.json)."""
    from core.config.models import update_status_model
    from core.paths import get_data_dir

    try:
        data_dir = get_data_dir()
        animas_dir = data_dir / "animas"
        pid_file = data_dir / "server.pid"
        server_running = pid_file.exists()

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
                        if server_running:
                            _set_background_model_via_server(args, entry.name, "", clear=True)
                        else:
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
                if server_running:
                    _set_background_model_via_server(args, name, "", clear=True)
                else:
                    update_status_model(
                        anima_dir,
                        background_model="",
                        background_credential="",
                    )
                print(f"Cleared background_model for '{name}'")
        elif args.all:
            model = args.model or args.anima
            if not model:
                print("Error: model is required (e.g. animaworks anima set-background-model claude-sonnet-5-5 --all)")
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
                    if server_running:
                        _set_background_model_via_server(args, entry.name, model, credential or None)
                    else:
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
                    "(e.g. animaworks anima set-background-model hinata claude-sonnet-5-5)"
                )
                sys.exit(1)
            anima_dir = animas_dir / args.anima
            if not anima_dir.exists():
                print(f"Error: Anima '{args.anima}' not found")
                sys.exit(1)
            if server_running:
                _set_background_model_via_server(args, args.anima, args.model, args.credential or None)
            else:
                kwargs_update: dict = {"background_model": args.model}
                if args.credential:
                    kwargs_update["background_credential"] = args.credential
                update_status_model(anima_dir, **kwargs_update)
            print(f"Background model updated to '{args.model}' for '{args.anima}'")

        if server_running:
            print(
                "  Running anima processes were asked to reload the background model; stopped animas use it on next start."
            )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_since(raw: str | None) -> datetime | None:
    """Parse ``--since HH:MM`` into a timezone-aware datetime (today, JST)."""
    if not raw:
        return None
    from datetime import time as _time

    from core.activity.logger import now_local

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

    from core.activity.logger import now_local

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
    from core.infra.event_export import register_activity_event_exporter
    from core.paths import get_animas_dir

    register_activity_event_exporter()

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

    from core.activity.audit import AuditAggregator

    if audit_all:
        dirs = sorted([d for d in animas_dir.iterdir() if d.is_dir() and (d / IDENTITY_FILE).exists()])
        if not dirs:
            print("No animas found.")
            sys.exit(1)
        print(AuditAggregator.generate_merged_timeline(dirs, hours=hours, since=since, until=until))
        return

    if name:
        anima_dir = animas_dir / name
        if not anima_dir.exists() or not (anima_dir / IDENTITY_FILE).exists():
            print(f"Error: Anima '{name}' not found")
            sys.exit(1)
        agg = AuditAggregator(anima_dir)
        print(agg.generate_report(hours=hours, since=since, until=until))
        return


def cmd_anima_rename(args: argparse.Namespace) -> None:
    """Rename an anima (directory, config, references)."""
    from core.anima.factory import validate_anima_name
    from core.config.models import rename_anima_in_config
    from core.paths import get_animas_dir, get_data_dir

    old_name: str = args.old_name
    new_name: str = args.new_name
    data_dir = get_data_dir()
    animas_dir = get_animas_dir()
    old_dir = animas_dir / old_name
    new_dir = animas_dir / new_name
    shared_dir = data_dir / "shared"

    # ── Validation ──
    if old_name == new_name:
        print("Error: Old and new names are the same")
        sys.exit(1)

    if not old_dir.exists() or not (old_dir / IDENTITY_FILE).exists():
        print(f"Error: Anima '{old_name}' not found (missing IDENTITY.md)")
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

    # The running server is the sole writer/mover of root-owned settings.
    pid_file = data_dir / "server.pid"
    if pid_file.exists():
        try:
            response = gateway_request(
                args,
                "POST",
                f"/api/animas/{old_name}/rename",
                json={"new_name": new_name},
                timeout=120.0,
                raw_response=True,
            )
            response.raise_for_status()
            result = response.json()
            print(f"  Renamed directory: animas/{old_name} → animas/{new_name}")
            if result.get("status_files_updated"):
                print(f"  Updated status.json for {result['status_files_updated']} anima(s)")
            if result.get("dm_logs_renamed"):
                print(f"  Renamed {result['dm_logs_renamed']} DM log file(s)")
            print(f"Anima renamed successfully: {old_name} → {new_name}")
            return
        except Exception as exc:
            print(f"Error: Failed to rename through running server: {exc}", file=sys.stderr)
            sys.exit(1)

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
            changed = False

            def rename_supervisor(status_data: dict[str, Any]) -> None:
                nonlocal changed
                if status_data.get("supervisor") == old_name:
                    status_data["supervisor"] = new_name
                    changed = True

            try:
                update_status(other_dir, rename_supervisor)
                status_updated += int(changed)
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)
        if status_updated:
            print(f"  Updated status.json for {status_updated} anima(s) with supervisor reference")

        # ── RAG: cleanup old collections ──
        repair_queued = _cleanup_rag_collections(new_dir, old_name)
        if repair_queued:
            from core.i18n import t

            print(f"  {t('rag.cli_rename_repair_queued')}")
        else:
            print("  Cleared RAG index (will re-index on next startup)")

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
    """Backward-compatible wrapper for the shared root/CLI helper."""
    from core.anima.admin import rename_dm_logs

    return rename_dm_logs(shared_dir, old_name, new_name)


def _cleanup_rag_collections(anima_dir: Path, old_name: str) -> bool:
    """Backward-compatible wrapper for the shared root/CLI helper."""
    from core.anima.admin import cleanup_rag_collections

    return cleanup_rag_collections(anima_dir, old_name, source="cli")


def cmd_anima_list(args: argparse.Namespace) -> None:
    """List all animas."""
    from core.paths import get_animas_dir, get_data_dir

    data_dir = get_data_dir()
    animas_dir = get_animas_dir()

    # Try API first (unless --local)
    if not args.local:
        pid_file = data_dir / "server.pid"
        if pid_file.exists():
            try:
                response = gateway_request(args, "GET", "/api/animas", timeout=10, raw_response=True)
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
                logger.debug("Best-effort operation failed", exc_info=True)

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
        if not (entry / IDENTITY_FILE).exists():
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
                logger.debug("Best-effort operation failed", exc_info=True)

        print(f"{name:<20} {enabled:<10} {role:<15} {model:<30}")
        count += 1

    print(f"\nTotal: {count}")
