# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Configuration I/O: singleton cache, load, and save."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.config.schemas import AnimaWorksConfig
from core.exceptions import ConfigError

logger = logging.getLogger("animaworks.config")

# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: AnimaWorksConfig | None = None
_config_path: Path | None = None
_config_mtime: float = 0.0


def invalidate_cache() -> None:
    """Reset the module-level singleton cache."""
    global _config, _config_path, _config_mtime
    _config = None
    _config_path = None
    _config_mtime = 0.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_config_path(data_dir: Path | None = None) -> Path:
    """Return the path to config.json inside *data_dir*.

    If *data_dir* is not given, it is resolved via ``core.paths.get_data_dir``
    (imported lazily to avoid circular imports).
    """
    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()
    return data_dir / "config.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AnimaWorksConfig:
    """Load configuration from disk, returning cached instance when possible.

    If *path* is ``None``, :func:`get_config_path` determines the location.
    When the file does not exist the default configuration is returned.

    The cache is automatically invalidated when the file's mtime changes,
    so external edits (org_sync, manual changes) are picked up without
    requiring a server restart.
    """
    global _config, _config_path, _config_mtime

    if path is None:
        path = get_config_path()

    # Check whether the on-disk file has been modified since last load.
    if _config is not None and _config_path == path:
        try:
            disk_mtime = path.stat().st_mtime
        except OSError:
            disk_mtime = 0.0
        if disk_mtime == _config_mtime:
            return _config
        logger.debug("Config file changed on disk (mtime %.3f → %.3f); reloading", _config_mtime, disk_mtime)

    if path.is_file():
        logger.debug("Loading config from %s", path)
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw_text)
            config = AnimaWorksConfig.model_validate(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
        except ConfigError:
            raise
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", path, exc)
            raise ConfigError(f"Failed to load config from {path}: {exc}") from exc
    else:
        logger.info("Config file not found at %s; using defaults", path)
        config = AnimaWorksConfig()

    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0
    return config


def save_config(config: AnimaWorksConfig, path: Path | None = None) -> None:
    """Persist *config* to disk as pretty-printed JSON (mode 0o600).

    Updates the module-level singleton cache so subsequent :func:`load_config`
    calls return the freshly saved config.
    """
    global _config, _config_path, _config_mtime

    if path is None:
        path = get_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = config.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    # Atomic write: write to a PID-unique sibling temp file then rename so
    # that concurrent writers (multiple anima workers) never clobber each
    # other's temp file.  Each process writes to .config.json.<PID>.tmp,
    # then renames it to config.json atomically.
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.rename(path)

    logger.debug("Config saved to %s", path)

    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0


__all__ = [
    "get_config_path",
    "invalidate_cache",
    "load_config",
    "save_config",
]
