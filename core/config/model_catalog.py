# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Static model catalog and per-request model override validation.

Canonical allowed IDs for chat REST and task-tool ``model`` overrides.
Network-dependent providers (nanoGPT / Ollama) are excluded so the check
stays cheap and deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config.models import KNOWN_MODELS, load_config
from core.platform.codex import is_codex_login_available
from core.platform.grok import is_grok_authenticated

logger = logging.getLogger(__name__)


def _known_codex_models() -> list[str]:
    """Return UI-visible Codex model ids from the shared known-model catalog."""
    return [
        str(item["name"])
        for item in KNOWN_MODELS
        if item.get("mode") == "C" and str(item.get("name", "")).startswith("codex/")
    ]


def _build_static_model_catalog(config: Any) -> list[dict[str, str]]:
    """Return available-model entries for static (non-network) providers.

    Shared by ``/system/available-models`` and model override validation so
    the allowed IDs stay canonical and consistent.  The dynamic (nanoGPT /
    Ollama) providers are appended only by the endpoint.
    """
    models: list[dict[str, str]] = []
    seen: set[str] = set()

    for provider, cred in config.credentials.items():  # type: ignore[attr-defined]
        if not cred.api_key and cred.type not in ("claude_code_login", "codex_login"):
            continue
        if provider == "anthropic":
            for m in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"):
                if m not in seen:
                    models.append({"id": m, "label": m, "credential": "anthropic"})
                    seen.add(m)
        elif provider == "openai":
            # Canonical ``openai/`` ids so bare names resolve to a real mode.
            for m in (
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini",
                "openai/gpt-4.1-nano",
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "openai/o3",
                "openai/o4-mini",
            ):
                label = m.removeprefix("openai/")
                if m not in seen:
                    models.append({"id": m, "label": label, "credential": "openai"})
                    seen.add(m)
            # Codex CLI models (codex_login or api_key)
            if cred.type == "codex_login" or cred.api_key:
                for m in _known_codex_models():
                    if m not in seen:
                        models.append({"id": m, "label": m, "credential": "openai"})
                        seen.add(m)
        elif provider in ("google", "gemini"):
            for m in ("gemini/gemini-2.5-flash",):
                label = m.removeprefix("gemini/")
                if m not in seen:
                    models.append({"id": m, "label": label, "credential": "google"})
                    seen.add(m)
    # Codex CLI models (standalone — no openai credential entry needed)
    if is_codex_login_available():
        for m in _known_codex_models():
            if m not in seen:
                models.append({"id": m, "label": m, "credential": "codex"})
                seen.add(m)

    # Grok Build CLI models (standalone — CLI OAuth, no credential entry needed)
    if is_grok_authenticated():
        for m in ("grok/grok-4.5", "grok/grok-composer-2.5-fast"):
            if m not in seen:
                models.append({"id": m, "label": m, "credential": "grok"})
                seen.add(m)

    return models


def available_model_id_set(config: Any = None) -> set[str]:
    """Return the set of canonical available-model IDs for override validation.

    Excludes network-dependent providers (nanoGPT / Ollama) so the check
    stays cheap and deterministic per request.
    """
    if config is None:
        config = load_config()
    return {entry["id"] for entry in _build_static_model_catalog(config)}


def validate_model_override(anima_name: str, requested_model: str | None) -> str | None:
    """Validate a per-request ``model`` override.

    Returns an error string (with the offending value) when the override
    should be rejected, or ``None`` when the request may proceed.  An empty
    model (unspecified) always passes through, matching the historical
    behaviour.

    Allowed set = canonical available-model ids (static catalog) ∪ the
    target anima's current model ∪ its ``fallback_models`` entries
    (both the full ``mode:model`` entry and its bare model part).  A
    ``mode:model`` request is validated against the model part.
    """
    if not requested_model or not requested_model.strip():
        return None
    model = requested_model.strip()
    if len(model) > 128:
        return f"model too long ({len(model)} chars, max 128): {model[:64]}..."

    config = load_config()
    allowed = available_model_id_set(config)

    # Expand with the target anima's current model + fallback_models.
    from core.config.model_config import load_model_config
    from core.paths import get_animas_dir

    anima_dir = get_animas_dir() / anima_name
    try:
        mc = load_model_config(anima_dir)
        if mc.model:
            allowed.add(mc.model)
        for fb in mc.fallback_models or []:
            if not isinstance(fb, str) or not fb:
                continue
            allowed.add(fb)
            head, _, tail = fb.partition(":")
            if len(head) == 1 and tail:
                allowed.add(tail)
    except Exception:
        logger.debug("Failed to load target anima model config: %s", anima_name, exc_info=True)

    check = model
    if ":" in model:
        head, _, tail = model.partition(":")
        if len(head) == 1 and tail:
            check = tail
    if check not in allowed:
        return f"unknown model override '{model}'"
    return None


validate_chat_model = validate_model_override
