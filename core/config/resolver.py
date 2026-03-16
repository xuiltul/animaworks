# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Configuration resolution: status.json merge with anima_defaults."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config.schemas import AnimaDefaults, AnimaModelConfig, AnimaWorksConfig, CredentialConfig

logger = logging.getLogger("animaworks.config")


def _load_status_json(anima_dir: Path) -> dict[str, Any]:
    """Load ModelConfig-relevant fields from anima's status.json.

    Returns a dict with field names matching AnimaDefaults fields.
    Missing or invalid files return an empty dict.
    """
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("Failed to read status.json from %s", anima_dir)
        return {}

    # Map status.json fields to AnimaModelConfig field names
    result: dict[str, Any] = {}
    field_mapping = {
        "model": "model",
        "background_model": "background_model",
        "background_credential": "background_credential",
        "context_threshold": "context_threshold",
        "max_turns": "max_turns",
        "max_chains": "max_chains",
        "conversation_history_threshold": "conversation_history_threshold",
        "credential": "credential",
        "execution_mode": "execution_mode",
        "supervisor": "supervisor",
        "max_tokens": "max_tokens",
        "fallback_model": "fallback_model",
        "thinking": "thinking",
        "thinking_effort": "thinking_effort",
        "llm_timeout": "llm_timeout",
        "mode_s_auth": "mode_s_auth",
        "max_outbound_per_hour": "max_outbound_per_hour",
        "max_outbound_per_day": "max_outbound_per_day",
        "max_recipients_per_run": "max_recipients_per_run",
        "default_workspace": "default_workspace",
        "extra_mcp_servers": "extra_mcp_servers",
    }
    # Fields where None is a valid explicit value (e.g. supervisor=null
    # means "top-level / no supervisor").  Empty string is still "not set".
    _nullable_fields = frozenset({"supervisor", "speciality"})
    for status_key, config_key in field_mapping.items():
        if status_key in data:
            value = data[status_key]
            if value is None and status_key in _nullable_fields or value not in (None, ""):
                result[config_key] = value
    return result


def resolve_anima_config(
    config: AnimaWorksConfig,
    anima_name: str,
    anima_dir: Path | None = None,
) -> tuple[AnimaDefaults, CredentialConfig]:
    """Merge status.json with *anima_defaults* (3-layer merge).

    Resolution uses a 3-layer priority (strongest first):

      1. ``status.json`` in *anima_dir* (SSoT for all fields including org)
      2. ``config.json`` per-anima (``config.animas``) — fallback for
         ``supervisor`` and ``speciality`` only
      3. ``config.json`` anima_defaults (global defaults)

    For ``supervisor`` and ``speciality``, explicit ``null`` in status.json
    is respected (means "top-level / no supervisor").

    When *anima_dir* is ``None``, layer 1 is skipped (no status.json).

    Returns:
        A ``(resolved_defaults, credential)`` tuple.

    Raises:
        KeyError: If the resolved credential name is not in
            ``config.credentials``.
    """
    anima_entry = config.animas.get(anima_name, AnimaModelConfig())
    defaults = config.anima_defaults

    status_values = _load_status_json(anima_dir) if anima_dir else {}

    # Merge priority (strongest first):
    #   1. status.json (including explicit null for supervisor/speciality)
    #   2. config.animas (fallback for supervisor/speciality only)
    #   3. anima_defaults (global defaults)
    resolved: dict[str, Any] = {}
    for field_name in AnimaDefaults.model_fields:
        if field_name in status_values:
            resolved[field_name] = status_values[field_name]
        elif field_name in ("supervisor", "speciality"):
            # Fallback to config.animas for org structure fields
            anima_value = getattr(anima_entry, field_name)
            if anima_value is not None:
                resolved[field_name] = anima_value
            else:
                resolved[field_name] = getattr(defaults, field_name)
        else:
            resolved[field_name] = getattr(defaults, field_name)

    resolved_defaults = AnimaDefaults.model_validate(resolved)

    credential_name = resolved_defaults.credential
    if credential_name not in config.credentials:
        raise KeyError(f"Credential '{credential_name}' (for anima '{anima_name}') not found in config.credentials")

    credential = config.credentials[credential_name]
    return resolved_defaults, credential


__all__ = [
    "_load_status_json",
    "resolve_anima_config",
]
