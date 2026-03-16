from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Workspace registry — alias-based project directory resolution.

Animas live in their home directory (~/.animaworks/animas/{name}/) but
*work* in project directories (workspaces).  This module manages a shared
registry in ``config.json`` and resolves aliases like ``aischreiber`` or
``aischreiber#3af4be6e`` to absolute paths with existence validation.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from core.i18n import t

logger = logging.getLogger("animaworks.workspace")


# ── Hash helpers ──────────────────────────────────────────────


def workspace_hash(path: str) -> str:
    """Generate 8-char hex hash from an absolute path string."""
    return hashlib.sha256(path.encode()).hexdigest()[:8]


def qualified_alias(alias: str, path: str) -> str:
    """Return ``alias#hash8`` form for display and structured references."""
    return f"{alias}#{workspace_hash(path)}"


# ── Resolution ────────────────────────────────────────────────


def resolve_workspace(alias_or_path: str) -> Path:
    """Resolve a workspace reference to an absolute :class:`Path`.

    Resolution order:
      1. ``alias#hash`` exact match against registry
      2. ``alias`` name exact match (for structured fields)
      3. Hash-only search (8-char hex against all registry entries)
      4. Absolute path (direct, existence check only)
      5. Not found → :exc:`ValueError`

    Raises:
        ValueError: If the workspace cannot be resolved or the directory
            does not exist.
    """
    if not alias_or_path or not alias_or_path.strip():
        raise ValueError(t("workspace.not_found", alias="(empty)"))

    from core.config.models import load_config

    cfg = load_config()
    registry: dict[str, str] = cfg.workspaces

    # 1. alias#hash exact match
    if "#" in alias_or_path:
        alias_part, hash_part = alias_or_path.rsplit("#", 1)
        for reg_alias, reg_path in registry.items():
            if reg_alias == alias_part and workspace_hash(reg_path) == hash_part:
                p = Path(reg_path)
                if not p.is_dir():
                    raise ValueError(t("workspace.dir_not_found", path=reg_path))
                return p

    # 2. alias name exact match
    if alias_or_path in registry:
        p = Path(registry[alias_or_path])
        if not p.is_dir():
            raise ValueError(t("workspace.dir_not_found", path=registry[alias_or_path]))
        return p

    # 3. hash-only search
    for _reg_alias, reg_path in registry.items():
        if workspace_hash(reg_path) == alias_or_path:
            p = Path(reg_path)
            if not p.is_dir():
                raise ValueError(t("workspace.dir_not_found", path=reg_path))
            return p

    # 4. Absolute path
    p = Path(alias_or_path).expanduser().resolve()
    if p.is_absolute() and p.is_dir():
        return p

    raise ValueError(t("workspace.not_found", alias=alias_or_path))


def resolve_default_workspace(anima_dir: Path) -> tuple[Path | None, str]:
    """Read default_workspace from status.json and resolve it.

    Returns:
        (resolved_path, alias) if set and resolved successfully.
        (None, alias) if set but resolution fails.
        (None, "") if not set.
    """
    import json

    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return None, ""
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, ""
    alias = (data.get("default_workspace") or "").strip()
    if not alias:
        return None, ""
    try:
        return resolve_workspace(alias), alias
    except ValueError:
        return None, alias


# ── Registration ──────────────────────────────────────────────


def register_workspace(alias: str, path: str) -> str:
    """Register a workspace directory under *alias*.

    Returns the qualified ``alias#hash8`` string on success.

    Raises:
        ValueError: If *path* does not exist as a directory.
    """
    from core.config.models import load_config, save_config

    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(t("workspace.dir_not_found", path=path))

    cfg = load_config()
    cfg.workspaces[alias] = str(p)
    save_config(cfg)

    qa = qualified_alias(alias, str(p))
    logger.info("Registered workspace: %s → %s", qa, p)
    return qa


def list_workspaces() -> dict[str, str]:
    """Return ``{qualified_alias: absolute_path}`` for all registered workspaces."""
    from core.config.models import load_config

    cfg = load_config()
    return {qualified_alias(alias, path): path for alias, path in cfg.workspaces.items()}


def remove_workspace(alias: str) -> bool:
    """Remove a workspace by *alias*.  Returns True if found and removed."""
    from core.config.models import load_config, save_config

    cfg = load_config()
    if alias in cfg.workspaces:
        del cfg.workspaces[alias]
        save_config(cfg)
        logger.info("Removed workspace alias: %s", alias)
        return True
    return False


def workspace_info(alias: str) -> dict[str, Any] | None:
    """Return workspace details or None if not registered."""
    from core.config.models import load_config

    cfg = load_config()
    path = cfg.workspaces.get(alias)
    if path is None:
        return None
    return {
        "alias": alias,
        "path": path,
        "qualified": qualified_alias(alias, path),
        "exists": Path(path).is_dir(),
    }
