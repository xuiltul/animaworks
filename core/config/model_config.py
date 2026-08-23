# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Model configuration resolution: load_model_config, penalties, max_tokens."""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.config.model_mode import _match_models_json
from core.config.schemas import AnimaWorksConfig
from core.i18n import t

if TYPE_CHECKING:
    from core.schemas import ModelConfig

logger = logging.getLogger("animaworks.config")

# ---------------------------------------------------------------------------
# load_model_config
# ---------------------------------------------------------------------------

_SENTINEL = object()


def load_model_config(anima_dir: Path) -> ModelConfig:
    """Build a ModelConfig for *anima_dir* from the unified config.json.

    This is a standalone version of ``MemoryManager.read_model_config()``
    for use in server routes that do not have a live DigitalAnima instance.
    """
    from core.config.models import get_config_path, load_config, resolve_anima_config, resolve_execution_mode
    from core.schemas import ModelConfig

    config_path = get_config_path()
    if not config_path.exists():
        return ModelConfig()

    config = load_config(config_path)
    anima_name = anima_dir.name
    resolved, credential = resolve_anima_config(config, anima_name, anima_dir=anima_dir)

    cred_name = resolved.credential
    api_key_env = f"{cred_name.upper()}_API_KEY"
    cred_type = getattr(credential, "type", None)
    if not isinstance(cred_type, str):
        cred_type = None
    mode = resolve_execution_mode(
        config,
        resolved.model,
        resolved.execution_mode,
    )
    return ModelConfig(
        model=resolved.model,
        fallback_model=resolved.fallback_model,
        fallback_models=resolved.fallback_models,
        background_model=resolved.background_model,
        background_credential=resolved.background_credential,
        max_tokens=resolved.max_tokens,
        credential=cred_name,
        credential_type=cred_type,
        api_key=credential.api_key or None,
        api_key_env=api_key_env,
        api_base_url=credential.base_url,
        context_threshold=resolved.context_threshold,
        max_chains=resolved.max_chains,
        conversation_history_threshold=resolved.conversation_history_threshold,
        execution_mode=resolved.execution_mode,
        supervisor=resolved.supervisor,
        speciality=resolved.speciality,
        resolved_mode=mode,
        thinking=resolved.thinking,
        thinking_effort=resolved.thinking_effort,
        background_thinking_effort=resolved.background_thinking_effort,
        voice_thinking_effort=resolved.voice_thinking_effort,
        heartbeat_enabled=resolved.heartbeat_enabled,
        token_budget_monthly=resolved.token_budget_monthly,
        llm_timeout=resolved.llm_timeout,
        extra_keys=credential.keys or {},
        mode_s_auth=resolved.mode_s_auth,
        extra_mcp_servers=resolved.extra_mcp_servers,
    )


# ---------------------------------------------------------------------------
# Runtime fallback resolution
# ---------------------------------------------------------------------------


def _resolved_mode_for_config(model_config: ModelConfig, config: AnimaWorksConfig) -> str:
    """Return the canonical execution mode for a runtime ``ModelConfig``."""
    from core.config.model_mode import resolve_execution_mode

    mode = model_config.resolved_mode or model_config.execution_mode
    if mode:
        return resolve_execution_mode(config, model_config.model, mode)
    return resolve_execution_mode(config, model_config.model)


def _guard_key_for_model_config(model_config: ModelConfig, config: AnimaWorksConfig) -> str:
    """Build the realm-qualified rate-guard key used by an execution mode."""
    from core.execution.error_classifier import (
        guard_key,
        litellm_realm_of,
        provider_family_of,
    )

    mode = _resolved_mode_for_config(model_config, config)
    if mode == "C":
        realm = "codex"
    elif mode == "S":
        realm = model_config.mode_s_auth or "max"
    elif mode == "D":
        realm = "cursor"
    elif mode == "G":
        realm = "gemini"
    elif mode == "X":
        realm = "grok"
    else:
        realm = litellm_realm_of(model_config.model)
    return guard_key(provider_family_of(model_config.model), realm)


def _fallback_credential_name(model: str) -> str | None:
    """Resolve the configured credential name for a fallback model family.

    DeepSeek GPT-format models (``openai/deepseek-*``) are served behind an
    OpenAI-compatible gateway (e.g. a vLLM host), not the ``openai`` cloud
    endpoint.  Route them to the DeepSeek gateway credential instead: a
    dedicated ``deepseek`` credential when present, else the configured
    ``background_credential`` gateway that hosts these models.
    """
    from core.execution.error_classifier import provider_family_of

    model_family = _model_family(model)
    if model_family == "openai" and model.split("/", 1)[-1].startswith("deepseek-"):
        dedicated = _FAMILY_CREDENTIAL_MAP.get("deepseek")
        if dedicated:
            return dedicated
        from core.config.io import load_config

        cfg = load_config()
        bg_cred = getattr(getattr(cfg, "anima_defaults", None), "background_credential", None)
        if isinstance(bg_cred, str) and bg_cred:
            return bg_cred
        return None
    return _FAMILY_CREDENTIAL_MAP.get(model_family) or _FAMILY_CREDENTIAL_MAP.get(provider_family_of(model))


def build_model_override_config(
    base: ModelConfig,
    mode: str,
    model: str,
    config: AnimaWorksConfig,
) -> ModelConfig | None:
    """Build a ModelConfig for a requested model, resolving credential by mode.

    Shared candidate construction for fallback selection and per-task model
    overrides.  CLI-auth engines (C/D/G/X) authenticate via their own CLI
    credential stores, so the base credential fields are preserved and only
    ``model`` / ``execution_mode`` / ``resolved_mode`` are replaced.  Other
    modes (S/A/etc.) resolve a family credential via
    :func:`_fallback_credential_name` and replace the credential fields;
    returns ``None`` when no credential can be resolved (caller should fall
    back to the base config rather than risk an auth error).
    """
    resolved_mode = mode.upper()
    if resolved_mode in {"C", "D", "G", "X"}:
        return base.model_copy(
            update={
                "model": model,
                "execution_mode": resolved_mode,
                "resolved_mode": resolved_mode,
            },
        )
    credential_name = _fallback_credential_name(model)
    credential = config.credentials.get(credential_name) if credential_name else None
    if credential is None:
        logger.warning(
            "build_model_override_config: no credential configured for model %r (family=%r)",
            model,
            credential_name,
        )
        return None
    credential_type = getattr(credential, "type", None)
    if not isinstance(credential_type, str):
        credential_type = None
    mode_s_auth = (
        infer_mode_s_auth(
            mode=resolved_mode,
            credential_name=credential_name,
            config=config,
        )
        if resolved_mode == "S"
        else None
    )
    return base.model_copy(
        update={
            "model": model,
            "execution_mode": resolved_mode,
            "resolved_mode": resolved_mode,
            "credential": credential_name,
            "credential_type": credential_type,
            "api_key": credential.api_key or None,
            "api_key_env": f"{credential_name.upper()}_API_KEY",
            "api_base_url": credential.base_url,
            "extra_keys": dict(credential.keys or {}),
            "mode_s_auth": mode_s_auth,
        },
    )


def resolve_effective_model_config(model_config: ModelConfig) -> ModelConfig:
    """Select the first usable fallback while the primary realm is blocked.

    The returned copy is ephemeral and never writes ``status.json``.  The
    original object is returned unchanged when the primary is available, no
    fallback is configured, or every fallback is unusable (fail-open).
    """
    from core.schemas import ModelConfig as RuntimeModelConfig

    if not isinstance(model_config, RuntimeModelConfig):
        return model_config
    if not model_config.fallback_models:
        return model_config

    from core.config.io import load_config
    from core.config.model_mode import parse_fallback_entry
    from core.execution.rate_guard import get_rate_guard

    config = load_config()
    guard = get_rate_guard()
    primary_key = _guard_key_for_model_config(model_config, config)
    primary_remaining = guard.blocked_remaining(primary_key)
    if primary_remaining <= 0:
        return model_config
    earliest_config = model_config
    earliest_until = guard.blocked_until(primary_key)
    earliest_remaining = primary_remaining

    for entry in model_config.fallback_models:
        parsed = parse_fallback_entry(entry, config)
        if parsed is None:
            continue
        mode, model = parsed
        resolved_mode = mode.upper()

        if resolved_mode == "X" and shutil.which("grok") is None:
            logger.debug("Skipping fallback %s:%s: grok CLI is unavailable", mode, model)
            continue
        if resolved_mode == "C":
            try:
                codex_available = importlib.util.find_spec("openai_codex") is not None
            except (ImportError, ValueError):
                codex_available = False
            if not codex_available:
                logger.debug("Skipping fallback %s:%s: openai_codex is unavailable", mode, model)
                continue

        candidate = build_model_override_config(model_config, mode, model, config)
        if candidate is None:
            logger.warning(
                "Skipping fallback %s:%s: no credential configured for model family",
                mode,
                model,
            )
            continue
        candidate_key = _guard_key_for_model_config(candidate, config)
        candidate_remaining = guard.blocked_remaining(candidate_key)
        if candidate_remaining > 0:
            candidate_until = guard.blocked_until(candidate_key)
            if candidate_until < earliest_until:
                earliest_config = candidate
                earliest_until = candidate_until
                earliest_remaining = candidate_remaining
            continue

        logger.warning(
            "primary %s blocked (%.0fs) \u2192 fallback %s:%s",
            model_config.model,
            primary_remaining,
            mode,
            model,
        )
        return candidate

    if earliest_config is not model_config:
        logger.warning(
            "all usable fallback candidates blocked; selecting earliest release %s (%.0fs)",
            earliest_config.model,
            earliest_remaining,
        )
    return earliest_config


def fallback_event_meta(
    primary: ModelConfig,
    effective: ModelConfig,
) -> dict[str, str | float] | None:
    """Describe a selected fallback for activity-log callers.

    ``None`` means the effective config is the primary config.  Remaining
    seconds are re-read from the primary guard key so all execution paths can
    share the same metadata construction without retaining mutable state.
    """
    primary_mode = (primary.resolved_mode or primary.execution_mode or "").upper()
    effective_mode = (effective.resolved_mode or effective.execution_mode or "").upper()
    if primary.model == effective.model and primary_mode == effective_mode:
        return None

    from core.config.io import load_config
    from core.execution.rate_guard import get_rate_guard

    config = load_config()
    guard = get_rate_guard()
    primary_key = _guard_key_for_model_config(primary, config)
    remaining = guard.blocked_remaining(primary_key)
    reason = guard.block_reason(primary_key)
    if not isinstance(reason, str) or not reason:
        reason = "unknown"
    return {
        "primary": primary.model,
        "fallback": f"{effective_mode.lower()}:{effective.model}",
        "reason": f"rate_guard_blocked:{reason}",
        "remaining": remaining,
    }


# ---------------------------------------------------------------------------
# resolve_penalties
# ---------------------------------------------------------------------------


def resolve_penalties(model_name: str) -> dict[str, float]:
    """Resolve frequency_penalty and presence_penalty from models.json.

    Returns a dict with only the keys that have non-None values.
    An empty dict means no penalties configured (backward-compatible).
    Values are clamped to [-2.0, 2.0] per the OpenAI API specification.
    """
    entry = _match_models_json(model_name)
    result: dict[str, float] = {}
    if entry is not None:
        for key in ("frequency_penalty", "presence_penalty"):
            val = entry.get(key)
            if val is not None:
                try:
                    result[key] = max(-2.0, min(2.0, float(val)))
                except (ValueError, TypeError):
                    pass
    return result


# ---------------------------------------------------------------------------
# max_tokens resolution
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: int = 8192
_THINKING_MIN_MAX_TOKENS: int = 16384


def _match_model_max_tokens(
    model_name: str,
    config: AnimaWorksConfig | None = None,
) -> int | None:
    """Match *model_name* against ``config.model_max_tokens`` pattern table."""
    if config is None:
        try:
            from core.config.models import load_config

            config = load_config()
        except Exception:
            return None
    table = config.model_max_tokens or {}
    if not table:
        return None
    bare = model_name.split("/", 1)[-1] if "/" in model_name else model_name
    for pattern, value in table.items():
        if fnmatch.fnmatch(model_name, pattern) or fnmatch.fnmatch(bare, pattern):
            return value
    return None


def resolve_max_tokens(
    model_name: str,
    explicit: int | None,
    thinking: bool | None,
    config: AnimaWorksConfig | None = None,
) -> int:
    """Resolve effective max_tokens.

    Priority:
      1. ``config.model_max_tokens`` pattern match (overrides all)
      2. Thinking minimum floor (16384 when thinking enabled, raises low values)
      3. Explicit value (status.json ``max_tokens``)
      4. DEFAULT_MAX_TOKENS (8192)
    """
    matched = _match_model_max_tokens(model_name, config)
    if matched is not None:
        return matched
    base = explicit if (explicit is not None and explicit != DEFAULT_MAX_TOKENS) else DEFAULT_MAX_TOKENS
    if thinking:
        return max(_THINKING_MIN_MAX_TOKENS, base)
    return base


# ---------------------------------------------------------------------------
# update_status_model
# ---------------------------------------------------------------------------


def update_status_model(
    anima_dir: Path,
    *,
    model: str | None = None,
    credential: str | None = None,
    background_model: str | None | object = _SENTINEL,
    background_credential: str | None | object = _SENTINEL,
    memory_backend: str | None | object = _SENTINEL,
) -> None:
    """Update model/credential in an anima's status.json (atomic write).

    For background_model/background_credential, pass empty string ``""`` to
    clear (remove the field).  The default sentinel leaves the field unchanged.
    """
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"status.json not found: {status_path}")
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if model is not None:
        data["model"] = model
    if credential is not None:
        data["credential"] = credential
    if background_model is not _SENTINEL:
        if background_model:
            data["background_model"] = background_model
        else:
            data.pop("background_model", None)
    if background_credential is not _SENTINEL:
        if background_credential:
            data["background_credential"] = background_credential
        else:
            data.pop("background_credential", None)
    if memory_backend is not _SENTINEL:
        if memory_backend:
            data["memory_backend"] = memory_backend
        else:
            data.pop("memory_backend", None)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(status_path)


# ── ModelFamily ──────────────────────────────────────────────────────


def _model_family(model: str) -> str:
    """Extract model family prefix.

    ``claude-*`` → ``"claude"``, ``prefix/rest`` → ``prefix``,
    anything else → the model string itself.
    """
    if model.startswith("claude-") or model == "claude":
        return "claude"
    if "/" in model:
        return model.split("/", 1)[0]
    return model


_FAMILY_CREDENTIAL_MAP: dict[str, str] = {
    "claude": "anthropic",
    "openai": "openai",
    "ollama": "ollama",
    "google": "google",
    "vertex_ai": "vertex_ai",
    "gemini": "gemini",
    "codex": "openai",
    "bedrock": "anthropic",
    "cursor": "anthropic",
    "grok": "grok",
}


# ── infer_mode_s_auth ────────────────────────────────────────────────


def infer_mode_s_auth(
    *,
    mode: str,
    credential_name: str,
    config: AnimaWorksConfig,
) -> str | None:
    """Infer ``mode_s_auth`` value from credential type.

    Returns ``None`` when *mode* is not ``"S"`` or no inference is possible.
    """
    if mode != "S":
        return None
    credential = config.credentials.get(credential_name)
    cred_type = getattr(credential, "type", "") if credential is not None else ""
    if cred_type == "claude_code_login":
        return "max"
    if cred_type == "api_key":
        return "api"
    return getattr(config.anima_defaults, "mode_s_auth", None)


# ── smart_update_model ───────────────────────────────────────────────


def smart_update_model(
    anima_dir: Path,
    *,
    model: str,
    credential: str | None = None,
    config: AnimaWorksConfig | None = None,
) -> dict[str, Any]:
    """Update model with automatic resolution of execution_mode, credential, and related fields.

    When the model family changes (e.g. ``ollama/*`` → ``claude-*``), credential,
    thinking, and max_tokens are auto-adjusted.  ``execution_mode`` and
    ``mode_s_auth`` are always resolved.

    Args:
        anima_dir: Path to the anima directory.
        model: New model name.
        credential: Explicit credential override (skips auto-inference).
        config: AppConfig instance (lazy-loaded if None).

    Returns:
        Dict summarizing applied changes.
    """
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"status.json not found: {status_path}")
    data: dict[str, Any] = json.loads(status_path.read_text(encoding="utf-8"))

    if config is None:
        from core.config.io import load_config

        config = load_config()

    old_model = data.get("model", "")
    old_family = _model_family(old_model)
    new_family = _model_family(model)
    family_changed = old_family != new_family

    data["model"] = model

    from core.config.model_mode import resolve_execution_mode

    next_mode = resolve_execution_mode(config, model)
    data["execution_mode"] = next_mode

    # -- credential resolution --
    if credential is not None:
        old_cred = data.get("credential", "")
        data["credential"] = credential
    elif family_changed:
        old_cred = data.get("credential", "")
        mapped = _FAMILY_CREDENTIAL_MAP.get(new_family)
        if mapped and mapped in config.credentials:
            data["credential"] = mapped
            logger.info(t("model_config.credential_auto_switch", old=old_cred, new=mapped))
        else:
            default_cred = getattr(config.anima_defaults, "credential", None)
            if default_cred:
                data["credential"] = default_cred
                logger.info(t("model_config.credential_fallback_defaults", family=new_family, default=default_cred))
            else:
                logger.warning(t("model_config.credential_keep_current", current=old_cred))

    # -- mode_s_auth resolution --
    next_credential = data.get("credential", "")
    if next_mode == "S" and next_credential:
        inferred_auth = infer_mode_s_auth(mode=next_mode, credential_name=next_credential, config=config)
        if inferred_auth:
            data["mode_s_auth"] = inferred_auth
        else:
            data.pop("mode_s_auth", None)
    else:
        data.pop("mode_s_auth", None)

    # -- clear stale overrides on family change --
    cleared: list[str] = []
    if family_changed and credential is None:
        for field in ("thinking", "max_tokens"):
            if field in data:
                data.pop(field)
                cleared.append(field)

    # -- atomic write --
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(status_path)

    return {
        "model": model,
        "credential": data.get("credential", ""),
        "execution_mode": next_mode,
        "mode_s_auth": data.get("mode_s_auth"),
        "family_changed": family_changed,
        "cleared_fields": cleared,
    }


__all__ = [
    "load_model_config",
    "resolve_effective_model_config",
    "fallback_event_meta",
    "resolve_penalties",
    "resolve_max_tokens",
    "DEFAULT_MAX_TOKENS",
    "_THINKING_MIN_MAX_TOKENS",
    "_match_model_max_tokens",
    "update_status_model",
    "_model_family",
    "infer_mode_s_auth",
    "smart_update_model",
]
