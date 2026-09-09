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

from core.config.model_discovery import discovered_model_ids
from core.config.models import KNOWN_MODELS, load_config
from core.platform.codex import is_codex_login_available
from core.platform.grok import is_grok_authenticated

logger = logging.getLogger(__name__)


def _configured_model_entries(config: Any) -> list[dict[str, str]]:
    """Exact configured models remain available when discovery is offline."""
    from core.config.model_mode import _load_models_json, parse_fallback_entry

    entries: list[dict[str, str]] = []
    for model, data in _load_models_json().items():
        if any(char in model for char in "*?["):
            continue
        entries.append(
            {
                "id": model,
                "label": model,
                "credential": str(data.get("credential") or ""),
                "mode": str(data.get("mode") or ""),
            }
        )
    configured = [getattr(config, "anima_defaults", None), *getattr(config, "animas", {}).values()]
    for item in configured:
        for value in (
            getattr(item, "model", None),
            getattr(item, "background_model", None),
            getattr(item, "fallback_model", None),
            *(getattr(item, "fallback_models", None) or []),
        ):
            if not isinstance(value, str) or not value:
                continue
            parsed = parse_fallback_entry(value, config)
            if parsed is not None:
                mode, model = parsed
                entries.append({"id": model, "label": model, "credential": "", "mode": mode.upper()})
    return entries


def _build_static_model_catalog(config: Any) -> list[dict[str, str]]:
    """Use the canonical compatibility catalog plus explicit configuration."""
    from core.config.model_config import _FAMILY_CREDENTIAL_MAP, _model_family

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    codex_login = is_codex_login_available()
    grok_login = is_grok_authenticated()
    for item in KNOWN_MODELS:
        model = str(item["name"])
        mode = item["mode"]
        provider = _FAMILY_CREDENTIAL_MAP.get(_model_family(model), _model_family(model))
        if provider == "gemini" and provider not in config.credentials and "google" in config.credentials:
            provider = "google"
        credential = config.credentials.get(provider)
        available = credential is not None and (
            bool(credential.api_key) or credential.type in {"claude_code_login", "codex_login"}
        )
        if mode == "C":
            available = available or codex_login
        elif mode == "X":
            available = grok_login
        if not available or model in seen:
            continue
        models.append({"id": model, "label": model.removeprefix("openai/"), "credential": provider, "mode": mode})
        seen.add(model)
    # Configuration overrides catalog metadata without discarding discovered
    # IDs. Wildcard routing patterns are not concrete picker entries.
    by_id = {entry["id"]: entry for entry in models}
    for entry in _configured_model_entries(config):
        by_id[entry["id"]] = {**by_id.get(entry["id"], entry), **{key: value for key, value in entry.items() if value}}
    return list(by_id.values())


def available_model_id_set(config: Any = None) -> set[str]:
    """Return the set of canonical available-model IDs for override validation.

    Backed by :func:`core.config.model_discovery.discovered_model_ids` which
    probes the installed CLIs / endpoints (cached for 300 s).  Each model
    contributes its ``mode:model`` id, its bare ``model``, and the tail
    after ``/`` so a ``mode:model`` request can match against the model part.
    """
    return discovered_model_ids(config=config)


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
        for value in (mc.model, mc.background_model):
            if value:
                allowed.add(value)
        for fb in [*(mc.fallback_models or []), mc.fallback_model]:
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

    # Being in the catalog is not enough: the override still has to resolve
    # to a credential.  It used to be dropped silently inside the anima (a
    # WARNING in its own log) while the request looked accepted and the reply
    # came back on the unchanged model.
    from core.config.model_config import can_build_model_override
    from core.config.model_mode import parse_fallback_entry

    parsed = parse_fallback_entry(model, config)
    if parsed is None:
        return f"unparseable model override '{model}'"
    if not can_build_model_override(parsed[0], parsed[1], config):
        return f"no credential configured for model override '{model}'"
    return None


validate_chat_model = validate_model_override
