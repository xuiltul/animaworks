from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Guarded root-owned settings writers shared by the server and offline CLI."""

import fcntl
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from core.anima.workspace_files import AGENTS_FILE, IDENTITY_FILE, SOUL_FILE
from core.platform.atomic_io import atomic_write_text, update_json
from core.platform.process_role import assert_settings_write_allowed


def settings_server_running() -> bool:
    """Return whether a live root server owns settings persistence right now."""
    from core.paths import get_data_dir
    from core.platform.pid import is_server_running

    return is_server_running(get_data_dir())


def update_status(
    anima_dir: Path,
    fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Atomically update an anima's root-owned ``status.json``."""
    anima_dir = Path(anima_dir)
    assert_settings_write_allowed(anima_dir / "status.json")
    from core.platform.status_store import update_status as update

    return update(anima_dir, fn)


def update_config(fn: Callable[[Any], Any], path: Path | None = None) -> Any:
    """Transactionally update root-owned ``config.json``."""
    from core.config.io import get_config_path
    from core.config.io import update_config as update

    config_path = Path(path) if path is not None else get_config_path()
    assert_settings_write_allowed(config_path)
    return update(fn, config_path)


def write_identity(anima_dir: Path, content: str) -> None:
    """Lock and atomically replace an anima's ``IDENTITY.md``."""
    _write_text_setting(Path(anima_dir) / IDENTITY_FILE, content)


def write_soul(anima_dir: Path, content: str) -> None:
    """Lock and atomically replace an anima's ``SOUL.md``."""
    _write_text_setting(Path(anima_dir) / SOUL_FILE, content)


def write_agents(anima_dir: Path, content: str) -> None:
    """Lock and atomically replace an anima's ``AGENTS.md``."""
    _write_text_setting(Path(anima_dir) / AGENTS_FILE, content)


def write_injection(anima_dir: Path, content: str) -> None:
    """Backward-compatible alias that writes ``AGENTS.md``."""
    write_agents(anima_dir, content)


def write_permissions(anima_dir: Path, permissions: Mapping[str, Any] | Any) -> None:
    """Lock and atomically replace an anima's ``permissions.json`` object."""
    anima_dir = Path(anima_dir)
    assert_settings_write_allowed(anima_dir / "permissions.json")
    if hasattr(permissions, "model_dump"):
        payload = permissions.model_dump(mode="json")
    elif isinstance(permissions, Mapping):
        payload = dict(permissions)
    else:
        raise TypeError("permissions must be a mapping or a model with model_dump()")
    if not isinstance(payload, dict):
        raise TypeError("permissions must serialize to a JSON object")
    # Validate JSON serializability before acquiring the lock so failures do
    # not leave a partially-created target file.
    json.dumps(payload, ensure_ascii=False)
    update_json(anima_dir / "permissions.json", lambda _current: payload, always_write=True)


def _write_text_setting(path: Path, content: str) -> None:
    assert_settings_write_allowed(path)
    if not isinstance(content, str):
        raise TypeError("setting content must be text")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            atomic_write_text(path, content, encoding="utf-8")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = [
    "settings_server_running",
    "update_status",
    "update_config",
    "write_agents",
    "write_identity",
    "write_injection",
    "write_permissions",
    "write_soul",
]
