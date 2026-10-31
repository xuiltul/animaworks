# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Anima registration in config.json: register, unregister, rename."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.anima.workspace_files import IDENTITY_FILE
from core.config.schemas import AnimaModelConfig

logger = logging.getLogger("animaworks.config")

_SENTINEL = object()

_NONE_SUPERVISOR_VALUES = frozenset({"なし", "(なし)", "（なし）", "-", "---", ""})

_PAREN_EN_NAME_RE = re.compile(r"[（(]([A-Za-z_][A-Za-z0-9_]*)[）)]")


def _resolve_supervisor_name(raw: str) -> str | None:
    """Resolve a raw supervisor value to an anima name.

    Handles formats like ``"琴葉（kotoha）"`` → ``"kotoha"``,
    ``"sakura"`` → ``"sakura"``, ``"(なし)"`` → ``None``.
    """
    raw = raw.strip()
    if not raw or raw in _NONE_SUPERVISOR_VALUES:
        return None

    # Extract English name from parentheses (full-width or half-width)
    m = _PAREN_EN_NAME_RE.search(raw)
    if m:
        return m.group(1).lower()

    # Plain ASCII name
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
        return raw.lower()

    # Non-ASCII without English equivalent — cannot resolve to an anima name
    logger.warning(
        "Supervisor value '%s' has no English name in parentheses; ignoring",
        raw,
    )
    return None


def read_anima_supervisor(anima_dir: Path) -> str | None:
    """Read the supervisor field for an anima from status.json or IDENTITY.md.

    Tries two sources in order:
      1. ``status.json`` — looks for ``"supervisor"`` key.
      2. ``IDENTITY.md`` — parses the Japanese table format ``| 上司 | name |``.

    The raw value is resolved to an English anima name (e.g.
    ``"琴葉（kotoha）"`` → ``"kotoha"``).

    Args:
        anima_dir: Path to the anima's runtime directory
            (e.g. ``~/.animaworks/animas/hinata``).

    Returns:
        The supervisor anima name if found, otherwise ``None``.
    """
    # Source 1: status.json
    status_path = anima_dir / "status.json"
    if status_path.is_file():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            raw = data.get("supervisor", "")
            if raw:
                resolved = _resolve_supervisor_name(raw)
                if resolved:
                    return resolved
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", status_path, exc)

    # Source 2: IDENTITY.md table row  | 上司 | name |
    identity_path = anima_dir / IDENTITY_FILE
    if identity_path.is_file():
        try:
            content = identity_path.read_text(encoding="utf-8")
            m = re.search(r"\|\s*上司\s*\|\s*(.+?)\s*\|", content)
            if m:
                resolved = _resolve_supervisor_name(m.group(1))
                if resolved:
                    return resolved
        except OSError as exc:
            logger.warning("Failed to read %s: %s", identity_path, exc)

    return None


def read_anima_company_checked(anima_dir: Path) -> tuple[bool, str | None]:
    """Read company membership while preserving whether disk was readable.

    The boolean is ``True`` only when the anima directory and ``status.json``
    were both present and the latter contained a JSON object.  The second
    value is the normalized company name, or ``None`` for a confirmed
    unassigned membership.  Filesystem denial, missing paths, and invalid JSON
    return ``(False, None)`` so sandbox callers can distinguish them from an
    explicitly unassigned anima.
    """
    try:
        if not anima_dir.is_dir():
            return False, None
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            return False, None
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", anima_dir / "status.json", exc)
        return False, None

    if not isinstance(data, dict):
        return False, None
    raw = data.get("company")
    if not isinstance(raw, str):
        return True, None
    company = raw.strip()
    return True, company or None


def read_anima_company(anima_dir: Path) -> str | None:
    """Read an anima's company membership from ``status.json``.

    This compatibility wrapper intentionally retains the historical behavior
    of treating unreadable membership as unassigned.  Callers that must tell
    those states apart should use :func:`read_anima_company_checked`.
    """
    _, company = read_anima_company_checked(anima_dir)
    return company


def register_anima_in_config(
    data_dir: Path,
    anima_name: str,
) -> None:
    """Register a newly created anima in config.json with supervisor synced.

    Reads status.json / IDENTITY.md in the anima directory to extract the
    ``supervisor`` field and stores it in the :class:`AnimaModelConfig` entry
    inside config.json.  If the anima already exists in the config the call
    is a no-op.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        anima_name: Name of the anima to register.
    """
    from core.config.io import update_config

    config_path = data_dir / "config.json"
    if not config_path.exists():
        return
    anima_dir = data_dir / "animas" / anima_name
    supervisor = read_anima_supervisor(anima_dir) if anima_dir.exists() else None
    registered = False

    def register(config):
        nonlocal registered
        if anima_name not in config.animas:
            config.animas[anima_name] = AnimaModelConfig(supervisor=supervisor)
            registered = True
        return config

    update_config(register, config_path)
    if registered:
        logger.debug(
            "Registered anima '%s' in config (supervisor=%s)",
            anima_name,
            supervisor,
        )
        try:
            from core.anima.roster import refresh_anima_roster

            refresh_anima_roster()
        except Exception:
            logger.debug("Failed to refresh anima roster after register", exc_info=True)

    # Ensure .env has Slack token slots for this Anima
    try:
        from core.config.env_slots import ensure_slack_env_slots

        if ensure_slack_env_slots(anima_name):
            logger.warning(
                "[%s] Slack tokens not configured — edit .env to add SLACK_BOT_TOKEN__%s and SLACK_APP_TOKEN__%s",
                anima_name,
                anima_name,
                anima_name,
            )
    except Exception:
        logger.debug("Failed to ensure Slack env slots for '%s'", anima_name, exc_info=True)


def unregister_anima_from_config(
    data_dir: Path,
    anima_name: str,
) -> bool:
    """Remove an anima from config.json.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        anima_name: Name of the anima to unregister.

    Returns:
        True if the anima was found and removed, False if it was not present.
    """
    from core.config.io import update_config

    config_path = data_dir / "config.json"
    if not config_path.exists():
        return False
    removed = False

    def unregister(config):
        nonlocal removed
        if anima_name in config.animas:
            del config.animas[anima_name]
            removed = True
        return config

    update_config(unregister, config_path)
    if not removed:
        return False
    logger.debug("Unregistered anima '%s' from config", anima_name)
    try:
        from core.anima.roster import refresh_anima_roster

        refresh_anima_roster()
    except Exception:
        logger.debug("Failed to refresh anima roster after unregister", exc_info=True)
    return True


def rename_anima_in_config(
    data_dir: Path,
    old_name: str,
    new_name: str,
) -> int:
    """Rename an anima in config.json: key, supervisor refs, anima_mapping.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        old_name: Current anima name.
        new_name: New anima name.

    Returns:
        Number of supervisor references updated.

    Raises:
        KeyError: If *old_name* is not found in ``config.animas``.
    """
    from core.config.io import update_config

    config_path = data_dir / "config.json"
    if not config_path.exists():
        raise KeyError(f"config.json not found in {data_dir}")
    supervisor_count = 0

    def rename(config):
        nonlocal supervisor_count
        if old_name not in config.animas:
            raise KeyError(f"Anima '{old_name}' not found in config.animas")

        # 1. Move animas entry
        entry = config.animas.pop(old_name)
        config.animas[new_name] = entry

        # 2. Update supervisor references across all animas
        supervisor_count = 0
        for anima_cfg in config.animas.values():
            if anima_cfg.supervisor == old_name:
                anima_cfg.supervisor = new_name
                supervisor_count += 1

        # 3. Update external_messaging: anima_mapping, app_id_mapping, default_anima
        for channel_cfg in (config.external_messaging.slack, config.external_messaging.chatwork):
            for mapping_attr in ("anima_mapping", "app_id_mapping"):
                mapping = getattr(channel_cfg, mapping_attr, {})
                for key, mapped_name in list(mapping.items()):
                    if mapped_name == old_name:
                        mapping[key] = new_name
            if channel_cfg.default_anima == old_name:
                channel_cfg.default_anima = new_name
        return config

    update_config(rename, config_path)
    logger.debug(
        "Renamed anima '%s' → '%s' in config (%d supervisor refs)",
        old_name,
        new_name,
        supervisor_count,
    )
    return supervisor_count


__all__ = [
    "_SENTINEL",
    "_NONE_SUPERVISOR_VALUES",
    "_PAREN_EN_NAME_RE",
    "_resolve_supervisor_name",
    "read_anima_supervisor",
    "register_anima_in_config",
    "unregister_anima_from_config",
    "rename_anima_in_config",
]
