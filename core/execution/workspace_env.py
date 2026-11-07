from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared scratch directory and package-cache environment for child processes."""

import json
from pathlib import Path

from core.platform.scratch import safe_scratch_component, scratch_root, task_scratch_dir

_PACKAGE_CACHE_DIRS = {
    "UV_CACHE_DIR": "uv",
    "PIP_CACHE_DIR": "pip",
    "npm_config_cache": "npm",
    "npm_config_store_dir": "pnpm-store",
    "GRADLE_USER_HOME": "gradle",
    "NUGET_PACKAGES": "nuget-packages",
    "COMPOSER_CACHE_DIR": "composer",
}


__all__ = ["build_workspace_env", "resolve_scratch_dir", "task_scratch_dir", "update_workspace_env"]


def resolve_scratch_dir(anima_dir: Path) -> Path:
    """Create and return the scratch directory for the current execution."""
    from core.tasks.board.tasks import current_attempt_identity

    try:
        identity = current_attempt_identity()
    except (json.JSONDecodeError, ValueError):
        identity = None

    if identity and identity.get("task_id"):
        scratch_dir = task_scratch_dir(anima_dir, identity["task_id"])
    else:
        from core.execution.session.session_context import current_runtime_session

        runtime_session = current_runtime_session()
        session_type = runtime_session.session_type if runtime_session is not None else "misc"
        scratch_dir = scratch_root(anima_dir) / safe_scratch_component(session_type, fallback="misc")

    parent_dir = scratch_dir.parent
    if parent_dir.is_symlink() or scratch_dir.is_symlink():
        raise OSError(f"Scratch directory must not be a symbolic link: {scratch_dir}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir


def build_workspace_env(anima_dir: Path) -> dict[str, str]:
    """Build environment overrides for scratch and shared package caches."""
    from core.execution.session.board_reply import board_reply_channels_env, run_triggers_env

    env = {"ANIMAWORKS_SCRATCH_DIR": str(resolve_scratch_dir(anima_dir))}
    env.update(board_reply_channels_env())
    env.update(run_triggers_env())

    from core.config.file_access_policy import shared_tool_cache_write_root

    cache_root = shared_tool_cache_write_root(anima_dir)
    if cache_root is not None:
        env.update({name: str(cache_root / directory) for name, directory in _PACKAGE_CACHE_DIRS.items()})
    return env


def update_workspace_env(env: dict[str, str], anima_dir: Path) -> None:
    """Apply workspace overrides and drop inherited cache paths when unavailable."""
    workspace_env = build_workspace_env(anima_dir)
    if not any(name in workspace_env for name in _PACKAGE_CACHE_DIRS):
        for name in _PACKAGE_CACHE_DIRS:
            env.pop(name, None)
    env.update(workspace_env)
