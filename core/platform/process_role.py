# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-level environment access for process-role metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from core.anima.workspace_files import AGENTS_FILE, IDENTITY_FILE, SOUL_FILE

PROCESS_ROLE_ENV = "ANIMAWORKS_PROCESS_ROLE"

ProcessRole = Literal["root", "anima", "task_runner", "cli", "mcp"]
_VALID_PROCESS_ROLES = frozenset({"root", "anima", "task_runner", "cli", "mcp"})
_SETTINGS_WRITER_ROLES = frozenset({"root", "cli"})


def get_process_role_env() -> str | None:
    """Read the inherited process-role environment value, if present."""
    return os.environ.get(PROCESS_ROLE_ENV)


def set_process_role_env(role: str) -> None:
    """Set the process-role environment value inherited by future children."""
    os.environ[PROCESS_ROLE_ENV] = role


def set_process_role(role: ProcessRole) -> None:
    """Set this process's role and propagate it to subsequently spawned children."""
    if role not in _VALID_PROCESS_ROLES:
        raise ValueError(f"Unknown process role: {role!r}")
    set_process_role_env(role)


def get_process_role() -> ProcessRole:
    """Return the current role, defaulting to ``cli`` when unset or invalid."""
    role = get_process_role_env()
    if role not in _VALID_PROCESS_ROLES:
        return "cli"
    return cast(ProcessRole, role)


def assert_settings_write_allowed(path: Path | None = None) -> None:
    """Allow root writes and offline CLI writes to root-owned settings."""
    role = get_process_role()
    if role not in _SETTINGS_WRITER_ROLES:
        raise PermissionError(f"Process role {role!r} cannot write root-owned settings")
    if role == "cli":
        from core.paths import get_data_dir
        from core.platform.pid import is_server_running

        if path is None:
            data_dir = get_data_dir()
        else:
            target = Path(os.path.abspath(path))
            if target.name == "config.json":
                data_dir = target.parent
            elif target.name in {"status.json", IDENTITY_FILE, SOUL_FILE, AGENTS_FILE, "permissions.json"}:
                anima_dir = target.parent
                data_dir = anima_dir.parent.parent if anima_dir.parent.name == "animas" else get_data_dir()
            else:
                data_dir = get_data_dir()
        if is_server_running(data_dir):
            raise PermissionError("CLI cannot write root-owned settings while the server is running")
