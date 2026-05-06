from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure dependency management for AnimaWorks.

Automatically starts required infrastructure services (e.g. Neo4j) when
Anima configurations require them.  Called during server startup before
Anima processes are spawned.
"""

import asyncio
import json
import logging
import shutil
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

_NEO4J_HOST = "127.0.0.1"
_NEO4J_BOLT_PORT = 7687
_NEO4J_COMPOSE_FILE = "docker-compose.neo4j.yml"
_NEO4J_HEALTH_TIMEOUT_SECONDS = 90
_NEO4J_HEALTH_POLL_INTERVAL = 3


# ── Helpers ──────────────────────────────────────────────────────────────


def _animas_need_neo4j(animas_dir: Path, names: list[str]) -> list[str]:
    """Return anima names whose ``memory_backend`` is ``"neo4j"``."""
    result: list[str] = []
    for name in names:
        status_path = animas_dir / name / "status.json"
        if not status_path.is_file():
            continue
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            if data.get("memory_backend") == "neo4j":
                result.append(name)
        except Exception:
            logger.debug("Failed to read status.json for %s", name, exc_info=True)
    return result


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check whether a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _run_docker_compose(compose_file: Path) -> bool:
    """Run ``docker compose up -d`` and return True on success."""
    compose_cmd = _resolve_compose_command()
    if compose_cmd is None:
        logger.error(
            "infra.neo4j_no_docker_compose",
        )
        return False

    cmd = [*compose_cmd, "-f", str(compose_file), "up", "-d"]
    logger.info("Starting Neo4j via: %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(
            "docker compose up failed (rc=%d): %s",
            proc.returncode,
            stderr.decode(errors="replace").strip(),
        )
        return False

    if stdout.strip():
        logger.debug("docker compose stdout: %s", stdout.decode(errors="replace").strip())
    return True


def _resolve_compose_command() -> list[str] | None:
    """Return the docker compose command as a list, or None if unavailable.

    Prefers ``docker compose`` (V2 plugin) over ``docker-compose`` (V1 standalone).
    """
    if shutil.which("docker"):
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


async def _wait_for_neo4j(timeout: float = _NEO4J_HEALTH_TIMEOUT_SECONDS) -> bool:
    """Poll Neo4j Bolt port until it accepts connections or timeout."""
    elapsed = 0.0
    while elapsed < timeout:
        if _is_port_open(_NEO4J_HOST, _NEO4J_BOLT_PORT, timeout=2.0):
            return True
        await asyncio.sleep(_NEO4J_HEALTH_POLL_INTERVAL)
        elapsed += _NEO4J_HEALTH_POLL_INTERVAL
    return False


# ── Public API ───────────────────────────────────────────────────────────


async def ensure_infra_services(
    animas_dir: Path,
    names_to_start: list[str],
    project_dir: Path,
) -> None:
    """Start infrastructure services required by the anima set.

    Currently handles:
    - **Neo4j**: If any anima in *names_to_start* has
      ``memory_backend: "neo4j"`` in its ``status.json``, ensures the
      Neo4j Docker container is running via ``docker-compose.neo4j.yml``.

    This function logs warnings but never raises — infra failures must not
    prevent the rest of the server from starting.
    """
    neo4j_animas = _animas_need_neo4j(animas_dir, names_to_start)
    if not neo4j_animas:
        return

    logger.info(
        "Neo4j backend required by: %s",
        ", ".join(neo4j_animas),
    )

    # Already running — nothing to do
    if _is_port_open(_NEO4J_HOST, _NEO4J_BOLT_PORT):
        logger.info("Neo4j already accepting connections on port %d", _NEO4J_BOLT_PORT)
        return

    compose_file = project_dir / _NEO4J_COMPOSE_FILE
    if not compose_file.is_file():
        logger.warning(
            "Neo4j compose file not found: %s  — skipping auto-start",
            compose_file,
        )
        return

    ok = await _run_docker_compose(compose_file)
    if not ok:
        logger.warning("Failed to start Neo4j container — animas using neo4j backend may malfunction")
        return

    logger.info("Waiting for Neo4j to become ready (up to %ds)…", _NEO4J_HEALTH_TIMEOUT_SECONDS)
    ready = await _wait_for_neo4j()
    if ready:
        logger.info("Neo4j is ready on port %d", _NEO4J_BOLT_PORT)
    else:
        logger.warning(
            "Neo4j did not become ready within %ds — animas using neo4j backend may malfunction",
            _NEO4J_HEALTH_TIMEOUT_SECONDS,
        )
