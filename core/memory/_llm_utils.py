# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM helper for memory-management modules.

All internal LLM calls (episode summarization, consolidation, distillation,
contradiction detection, etc.) use the *consolidation model* configured in
``config.json``.  This module provides a thin wrapper that resolves the model
and injects credentials so that individual memory modules don't need to
duplicate this logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("animaworks.memory._llm_utils")

# Provider → env var mapping for LiteLLM auto-detection.
_PROVIDER_ENV_MAP: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_credentials_exported = False


def ensure_credentials_in_env() -> None:
    """Export config.json credentials to environment variables.

    LiteLLM reads API keys from environment variables (e.g. GEMINI_API_KEY).
    If the server process was started without these vars, this function
    reads them from ``config.json`` credentials and sets them once.

    Safe to call multiple times — only runs the export logic once.
    """
    global _credentials_exported
    if _credentials_exported:
        return
    _credentials_exported = True

    try:
        from core.config import load_config
        cfg = load_config()
    except Exception:
        return

    for provider, cred in cfg.credentials.items():
        if not cred.api_key:
            continue
        env_key = _PROVIDER_ENV_MAP.get(provider)
        if env_key and not os.environ.get(env_key):
            os.environ[env_key] = cred.api_key
            logger.debug("Exported %s from config credentials", env_key)


def get_consolidation_llm_kwargs() -> dict[str, Any]:
    """Return ``model`` and optional ``api_key`` for LiteLLM calls.

    Reads ``consolidation.llm_model`` from the organisation config and
    resolves the matching credential (if any) from ``credentials``.
    Also ensures credentials are exported to environment variables.

    Returns:
        Dict with at least ``"model"`` key.  May also contain ``"api_key"``.
    """
    ensure_credentials_in_env()

    from core.config import load_config

    cfg = load_config()
    model = cfg.consolidation.llm_model
    kwargs: dict[str, Any] = {"model": model}

    # Resolve API key: config credentials → env vars
    provider = model.split("/")[0] if "/" in model else ""
    cred = cfg.credentials.get(provider)
    if cred and cred.api_key:
        kwargs["api_key"] = cred.api_key
    else:
        env_key = _PROVIDER_ENV_MAP.get(provider)
        if env_key:
            val = os.environ.get(env_key)
            if val:
                kwargs["api_key"] = val

    return kwargs
