# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Native ``animaworks demo`` command.

Runs the 3-agent demo team directly on the host using an already-installed
Claude Code / Codex CLI login (or an ANTHROPIC_API_KEY), so no extra API key
is required.  This replaces the former Docker demo (``demo/entrypoint.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.anima.workspace_files import HEARTBEAT_FILE

# Model names (must match KNOWN_MODELS in core/config/model_mode.py)
CLAUDE_MODEL_MAIN = "claude-sonnet-5-5"
CLAUDE_MODEL_BACKGROUND = "claude-haiku-4-5-20251001"
CODEX_MODEL_MAIN = "codex/gpt-5.4"
CODEX_MODEL_BACKGROUND = "codex/gpt-5.4-mini"

# Roles that get the "main" model; everything else gets the cheap model.
_MAIN_ROLES = ("manager", "engineer", "writer", "researcher")

_DEFAULT_DATA_DIR = "~/.animaworks-demo"


# ── Auth detection (monkeypatchable) ─────────────────────────


def _claude_code_available() -> bool:
    from core.platform.claude_code import is_claude_code_available

    return is_claude_code_available()


def _codex_login_available() -> bool:
    from core.platform.codex import is_codex_login_available

    return is_codex_login_available()


def detect_auth() -> dict | None:
    """Detect the best available authentication source (highest priority first)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return {
            "label": "ANTHROPIC_API_KEY (api)",
            "family": "claude",
            "mode_s_auth": "api",
            "credentials": {"anthropic": {"type": "api_key", "api_key": api_key}},
        }
    if _claude_code_available():
        return {
            "label": "Claude Code login (subscription)",
            "family": "claude",
            "mode_s_auth": "max",
            "credentials": {"anthropic": {"type": "claude_code_login", "api_key": ""}},
        }
    if _codex_login_available():
        return {
            "label": "Codex login",
            "family": "codex",
            "mode_s_auth": None,
            "credentials": {"openai": {"type": "codex_login", "api_key": ""}},
        }
    return None


# ── Path helpers ─────────────────────────────────────────────


def _resolve_repo_root() -> Path:
    from cli import __file__ as cli_init

    root = Path(cli_init).resolve().parent.parent
    if not (root / "demo").is_dir():
        print(
            "Error: demo/ directory not found. Please run this command from a git "
            "clone of the repository (the demo works best with a fresh clone).",
            file=sys.stderr,
        )
        sys.exit(1)
    return root


def available_presets(repo_root: Path) -> list[str]:
    preset_dir = repo_root / "demo" / "presets"
    if not preset_dir.is_dir():
        return []
    return sorted(p.name for p in preset_dir.iterdir() if p.is_dir() and p.name != "commands")


def _resolve_preset_dir(repo_root: Path, preset: str) -> Path:
    preset_dir = repo_root / "demo" / "presets" / preset
    if not preset_dir.is_dir():
        avail = ", ".join(available_presets(repo_root)) or "(none)"
        print(f"Error: Preset directory not found: {preset_dir}", file=sys.stderr)
        print(f"       Available presets: {avail}", file=sys.stderr)
        sys.exit(1)
    return preset_dir


# ── Config helpers ───────────────────────────────────────────


def _deep_merge(base: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_overlay(data_dir: Path, overlay_path: Path) -> None:
    """Deep-merge config_overlay.json into config.json (entrypoint step 3)."""
    from core.config.io import update_config
    from core.config.schemas import AnimaWorksConfig

    cfg_path = data_dir / "config.json"
    ovl = json.loads(overlay_path.read_text(encoding="utf-8"))

    def apply_overlay(config: AnimaWorksConfig) -> AnimaWorksConfig:
        merged = config.model_dump(mode="json")
        _deep_merge(merged, ovl)
        return AnimaWorksConfig.model_validate(merged)

    update_config(apply_overlay, cfg_path)


def _inject_credentials(auth: dict) -> None:
    """Write detected credentials + mode_s_auth into config.json (entrypoint step 11)."""
    from core.config import CredentialConfig, update_config

    def inject_credentials(config):
        config.credentials = {
            name: CredentialConfig(type=data["type"], api_key=data.get("api_key", ""))
            for name, data in auth["credentials"].items()
        }
        if auth.get("mode_s_auth"):
            config.anima_defaults.mode_s_auth = auth["mode_s_auth"]
        if auth.get("family") == "codex":
            # Overlay defaults target claude; align defaults for codex family.
            config.anima_defaults.model = CODEX_MODEL_MAIN
            config.anima_defaults.background_model = CODEX_MODEL_BACKGROUND
        return config

    update_config(inject_credentials)


# ── Initialization (entrypoint steps 1-10) ───────────────────


def _override_models(data_dir: Path, family: str) -> None:
    """Rewrite per-anima status.json models for demo cost control (entrypoint step 4a)."""
    if family == "codex":
        main_model, bg_model = CODEX_MODEL_MAIN, CODEX_MODEL_BACKGROUND
    else:
        main_model, bg_model = CLAUDE_MODEL_MAIN, CLAUDE_MODEL_BACKGROUND
    from core.anima.settings_store import update_status

    for status_path in (data_dir / "animas").glob("*/status.json"):

        def apply_demo_models(status: dict[str, Any]) -> None:
            role = status.get("role", "general")
            status["model"] = main_model if role in _MAIN_ROLES else bg_model
            status["background_model"] = bg_model

        update_status(status_path.parent, apply_demo_models)
    print("  Demo model override applied.")


def _copy_heartbeat_cron(preset_dir: Path, data_dir: Path) -> None:
    """Copy preset heartbeat/cron templates (entrypoint step 4b)."""
    for source, ext in ((preset_dir / "heartbeat", HEARTBEAT_FILE), (preset_dir / "cron", "cron.md")):
        for md_file in source.glob("*.md"):
            name = md_file.stem
            target_dir = data_dir / "animas" / name
            if target_dir.is_dir():
                shutil.copy(md_file, target_dir / ext)
                print(f"  Custom {ext} installed for: {name}")


def _create_auth(data_dir: Path) -> None:
    """Create local_trust auth.json (entrypoint step 4c)."""
    from core.auth.manager import save_auth
    from core.auth.models import AuthConfig, AuthUser

    owner = AuthUser(username="demo", display_name="Demo User", role="owner")
    save_auth(AuthConfig(auth_mode="local_trust", trust_localhost=True, owner=owner))
    print("  Auth config created (local_trust).")


def _copy_assets(preset_dir: Path, data_dir: Path) -> None:
    """Copy pre-built character assets (entrypoint step 5)."""
    assets_dir = preset_dir / "assets"
    if not assets_dir.is_dir():
        return
    for char_dir in assets_dir.iterdir():
        if not char_dir.is_dir():
            continue
        target = data_dir / "animas" / char_dir.name / "assets"
        target.mkdir(parents=True, exist_ok=True)
        for asset in char_dir.iterdir():
            if asset.is_file():
                shutil.copy(asset, target)
    print("  Character assets installed.")


def _copy_examples(examples: Path, data_dir: Path, repo_root: Path) -> None:
    """Adjust example timestamps in a temp copy and install them (entrypoint step 6)."""
    if not examples.is_dir():
        return
    adjust_script = repo_root / "demo" / "adjust_dates.sh"
    try:
        tmp_examples = Path(tempfile.mkdtemp(prefix="animaworks-demo-"))
        for item in examples.iterdir():
            dst = tmp_examples / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy(item, dst)
        if adjust_script.is_file():
            subprocess.run(["bash", str(adjust_script), str(tmp_examples)], check=False)
        for char_dir in tmp_examples.iterdir():
            if not char_dir.is_dir():
                continue
            char_name = char_dir.name
            if char_name in ("channels", "users"):
                continue
            target = data_dir / "animas" / char_name
            if target.is_dir():
                shutil.copytree(char_dir, target, dirs_exist_ok=True)
        for shared_name in ("channels", "users"):
            src = tmp_examples / shared_name
            if src.is_dir():
                target = data_dir / "shared" / shared_name
                target.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, target, dirs_exist_ok=True)
        print("  Example runtime data installed.")
    finally:
        shutil.rmtree(tmp_examples, ignore_errors=True)


def initialize_demo(data_dir: Path, preset_dir: Path, repo_root: Path, auth: dict) -> None:
    """Run first-run initialization (port of entrypoint steps 1-10)."""
    from core.anima.factory import create_from_md
    from core.config import invalidate_cache, register_anima_in_config
    from core.infra.runtime_init import ensure_runtime_dir, merge_templates

    # 1. Initialize infrastructure (no default anima)
    ensure_runtime_dir(skip_animas=True)
    print("Runtime directory initialized.")

    # 2. Company vision
    vision = preset_dir / "vision.md"
    if vision.exists():
        (data_dir / "company").mkdir(parents=True, exist_ok=True)
        shutil.copy(vision, data_dir / "company" / "vision.md")
        print("Company vision installed.")

    # 3. Config overlay BEFORE anima creation so locale is correct
    overlay = preset_dir / "config_overlay.json"
    if overlay.exists():
        _apply_overlay(data_dir, overlay)
        print("Config overlay applied.")
        for subdir in ("prompts", "common_knowledge", "common_skills"):
            path = data_dir / subdir
            if path.is_dir():
                shutil.rmtree(path)
        merge_templates(data_dir)
        print("Templates re-merged with locale from overlay.")
        invalidate_cache()

    # 4. Create animas from character sheets
    animas_dir = data_dir / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    for md_file in sorted(preset_dir.glob("characters/*.md")):
        name = md_file.stem
        role, supervisor = None, None
        role_file = preset_dir / "roles" / f"{name}.txt"
        if role_file.exists():
            lines = role_file.read_text(encoding="utf-8").strip().splitlines()
            role = lines[0].strip() if lines and lines[0].strip() else None
            if len(lines) > 1 and lines[1].strip():
                supervisor = lines[1].strip()
        print(f"Creating anima: {name}")
        anima_dir = create_from_md(animas_dir, md_file, supervisor=supervisor, role=role)
        register_anima_in_config(data_dir, anima_dir.name)

    # 4a. Override models for cost
    _override_models(data_dir, auth["family"])

    # 4b. Custom heartbeat/cron
    _copy_heartbeat_cron(preset_dir, data_dir)

    # 4c. auth.json (local_trust)
    _create_auth(data_dir)

    # 5. Assets
    _copy_assets(preset_dir, data_dir)

    # 6. Example runtime data (locale-based)
    lang = preset_dir.name.split("-")[0]
    _copy_examples(repo_root / "demo" / "examples" / lang, data_dir, repo_root)


# ── Server start (monkeypatchable) ──────────────────────────


def _start_server(args: argparse.Namespace) -> None:
    from cli.commands.server import cmd_start

    cmd_start(args)


# ── Entry point ─────────────────────────────────────────────


def cmd_demo(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    os.environ["ANIMAWORKS_DATA_DIR"] = str(data_dir)

    from core.config import invalidate_cache

    invalidate_cache()

    from core.platform.pid import read_server_pid
    from core.platform.process import is_process_alive

    running_pid = read_server_pid(data_dir)
    if running_pid is not None and is_process_alive(running_pid):
        print(
            f"Error: AnimaWorks server is already running (pid={running_pid}); stop it before starting demo mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = _resolve_repo_root()
    preset_dir = _resolve_preset_dir(repo_root, args.preset)

    auth = detect_auth()
    if auth is None:
        print(
            "No authentication found. Please provide one of:\n"
            "  - Claude Code (log in with `claude`)\n"
            "  - Codex (run `codex login`)\n"
            "  - the ANTHROPIC_API_KEY environment variable",
            file=sys.stderr,
        )
        sys.exit(1)

    # --reset: wipe the demo data dir and re-initialize
    if getattr(args, "reset", False) and (data_dir / "config.json").exists():
        shutil.rmtree(data_dir)

    if (data_dir / "config.json").exists():
        print("Existing configuration found — skipping initialization.")
    else:
        print("=== First run detected — initializing AnimaWorks ===")
        print(f"Preset: {args.preset}")
        initialize_demo(data_dir, preset_dir, repo_root, auth)
        print("=== Initialization complete ===")

    _inject_credentials(auth)

    # Show detected auth in one line
    print(f"Auth: {auth['label']}")

    # Remove stale PID file before starting
    (data_dir / "server.pid").unlink(missing_ok=True)

    print(f"Open http://localhost:{args.port}/")
    start_args = argparse.Namespace(
        host=args.host,
        port=args.port,
        foreground=True,
    )
    _start_server(start_args)


def setup_demo_command(subparsers) -> None:
    """Register the ``demo`` subcommand."""
    # Do not use _resolve_repo_root() here: it prints an error and exits,
    # which would fire on every `animaworks --help` in non-repo installs.
    from cli import __file__ as cli_init

    repo_root = Path(cli_init).resolve().parent.parent
    presets = available_presets(repo_root)
    parser = subparsers.add_parser(
        "demo",
        help="Run the 3-agent demo team (no API key needed if Claude Code or Codex is logged in)",
    )
    parser.add_argument("--preset", default="en-business", choices=presets or None)
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=18501)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reset", action="store_true")
    parser.set_defaults(func=cmd_demo)
