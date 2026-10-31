#!/usr/bin/env python3
"""Create SWE-bench evaluation team in an ISOLATED runtime directory.

IMPORTANT: This creates agents under swe/runtime/ (or $SWE_RUNTIME_DIR),
NOT in the production ~/.animaworks/. The runner sets ANIMAWORKS_HOME to
this isolated directory before starting the server, so SWE agents never
appear in the production environment.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from core.anima.workspace_files import AGENTS_FILE, HEARTBEAT_FILE, IDENTITY_FILE, SOUL_FILE

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "team.json"
DEFAULT_RUNTIME = Path(__file__).parent / "runtime"


def _get_runtime_dir() -> Path:
    """Return the isolated runtime directory for SWE evaluation."""
    import os

    return Path(os.environ.get("SWE_RUNTIME_DIR", str(DEFAULT_RUNTIME)))


def _ensure_init(runtime_dir: Path) -> None:
    """Initialize AnimaWorks runtime in the isolated directory."""
    if (runtime_dir / "config.json").exists():
        return
    logger.info("Initializing isolated runtime at %s ...", runtime_dir)
    env = {**__import__("os").environ, "ANIMAWORKS_DATA_DIR": str(runtime_dir)}
    subprocess.run(
        [sys.executable, "-m", "main", "init", "--skip-anima"],
        check=True,
        env=env,
    )


def _copy_credentials(runtime_dir: Path) -> None:
    """Copy credentials from production config to isolated runtime."""
    prod_config = Path.home() / ".animaworks" / "config.json"
    if not prod_config.exists():
        return
    prod = json.loads(prod_config.read_text())
    prod_creds = prod.get("credentials", {})
    if not prod_creds:
        return

    rt_config_path = runtime_dir / "config.json"
    rt_config = json.loads(rt_config_path.read_text())
    rt_config.setdefault("credentials", {}).update(prod_creds)
    rt_config_path.write_text(json.dumps(rt_config, indent=2, ensure_ascii=False))
    logger.info("Copied %d credentials to isolated runtime", len(prod_creds))


def _copy_models_json(runtime_dir: Path) -> None:
    """Copy models.json from production to isolated runtime."""
    prod_models = Path.home() / ".animaworks" / "models.json"
    if prod_models.exists():
        shutil.copy2(prod_models, runtime_dir / "models.json")
        logger.info("Copied models.json to isolated runtime")


def setup_team(
    config_path: Path = DEFAULT_CONFIG,
    runtime_dir: Path | None = None,
) -> tuple[list[str], Path]:
    """Create agents in the isolated runtime directory.

    Returns (list of agent names, runtime_dir path).
    """
    if runtime_dir is None:
        runtime_dir = _get_runtime_dir()

    _ensure_init(runtime_dir)
    _copy_credentials(runtime_dir)
    _copy_models_json(runtime_dir)

    team_cfg = json.loads(config_path.read_text())
    agents = team_cfg["agents"]
    animas_dir = runtime_dir / "animas"
    animas_dir.mkdir(exist_ok=True)
    created = []

    for name, agent_cfg in agents.items():
        agent_dir = animas_dir / name
        if agent_dir.exists():
            logger.info("Agent %s already exists, updating config", name)
        else:
            agent_dir.mkdir(parents=True)
            logger.info("Created agent directory: %s", name)

        # status.json
        status = {
            "supervisor": agent_cfg.get("supervisor"),
            "role": agent_cfg.get("role", "general"),
            "enabled": True,
            "model": agent_cfg["model"],
            "max_turns": 100,
            "max_chains": 5,
        }
        if agent_cfg.get("credential"):
            status["credential"] = agent_cfg["credential"]
        (agent_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False)
        )

        # IDENTITY.md / SOUL.md
        identity = agent_cfg["identity"]
        (agent_dir / IDENTITY_FILE).write_text(f"# {name}\n\n{identity}\n")
        (agent_dir / SOUL_FILE).write_text(f"# Soul: {name}\n\n{identity}\n")

        # AGENTS.md
        (agent_dir / AGENTS_FILE).write_text(agent_cfg["injection"] + "\n")

        # permissions.md
        (agent_dir / "permissions.md").write_text(
            "## Allowed Commands\n- All commands allowed for SWE-bench evaluation\n"
        )

        # HEARTBEAT.md — disable heartbeat for SWE agents
        (agent_dir / HEARTBEAT_FILE).write_text(
            "# Heartbeat\n\nHeartbeat disabled for SWE-bench evaluation.\n"
        )

        # Required subdirectories
        for subdir in [
            "state", "episodes", "knowledge", "procedures",
            "skills", "shortterm", "activity_log", "transcripts",
        ]:
            (agent_dir / subdir).mkdir(exist_ok=True)
        (agent_dir / "state" / "pending").mkdir(exist_ok=True)

        # Register in isolated config.json
        rt_config_path = runtime_dir / "config.json"
        config = json.loads(rt_config_path.read_text())
        animas_cfg = config.setdefault("animas", {})
        if name not in animas_cfg:
            animas_cfg[name] = {
                "supervisor": agent_cfg.get("supervisor"),
                "speciality": agent_cfg.get("role", "general"),
            }
            rt_config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False)
            )

        created.append(name)

    logger.info("Team setup complete in %s: %s", runtime_dir, created)
    return created, runtime_dir


def teardown_team(
    config_path: Path = DEFAULT_CONFIG,
    runtime_dir: Path | None = None,
) -> None:
    """Remove the isolated runtime directory entirely."""
    if runtime_dir is None:
        runtime_dir = _get_runtime_dir()
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
        logger.info("Removed isolated runtime: %s", runtime_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse

    p = argparse.ArgumentParser(description="SWE-bench team setup")
    p.add_argument("action", choices=["setup", "teardown"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--runtime-dir", type=Path, default=None)
    args = p.parse_args()

    if args.action == "setup":
        setup_team(args.config, args.runtime_dir)
    else:
        teardown_team(args.config, args.runtime_dir)
