from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for helper-model selection."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.anima.workspace_files import IDENTITY_FILE
from core.config.schemas import (
    DEFAULT_CONSOLIDATION_MODEL,
    AnimaWorksConfig,
    HelperModelFallback,
    HelperModelRole,
)

logger = logging.getLogger("animaworks.config")

HELPER_MODEL_ROLES: tuple[str, ...] = (
    "episode_summary",
    "fact_extraction",
    "fact_reconcile",
    "weekly_consolidation",
    "project_consolidation",
    "conversation_compression",
    "distillation",
    "reconsolidation",
    "asset_reconcile",
    "meeting_summary",
)


@dataclass(frozen=True)
class ResolvedHelperModel:
    """Effective model and policy for one helper role."""

    model: str
    credential: str | None
    fallbacks: list[HelperModelFallback]
    allow_agent_sdk_fallback: bool
    max_output_tokens: int | None
    source: str


@dataclass(frozen=True)
class _ModelCandidate:
    model: str
    credential: str | None
    source: str


# Legacy consolidation settings remain supported, but only for the roles whose
# historical behavior used them. Each tuple is (model field, credential field).
_ROLE_LEGACY_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "fact_extraction": (
        ("live_fact_model", "live_fact_credential"),
        ("fact_reconcile_model", "fact_reconcile_credential"),
        ("llm_model", "llm_credential"),
    ),
    "fact_reconcile": (
        ("fact_reconcile_model", "fact_reconcile_credential"),
        ("llm_model", "llm_credential"),
    ),
    "episode_summary": (("llm_model", "llm_credential"),),
    "distillation": (("llm_model", "llm_credential"),),
    "reconsolidation": (("llm_model", "llm_credential"),),
    "conversation_compression": (("llm_model", "llm_credential"),),
    "asset_reconcile": (("llm_model", "llm_credential"),),
    "meeting_summary": (("llm_model", "llm_credential"),),
    "weekly_consolidation": (
        ("weekly_llm_model", "weekly_llm_credential"),
        ("llm_model", "llm_credential"),
    ),
    "project_consolidation": (
        ("weekly_llm_model", "weekly_llm_credential"),
        ("llm_model", "llm_credential"),
    ),
}
_ROLES_WITH_LEGACY_FALLBACK = frozenset(
    {
        "episode_summary",
        "distillation",
        "reconsolidation",
        "conversation_compression",
        "asset_reconcile",
        "meeting_summary",
    }
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_field_explicit(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    fields_set = getattr(value, "model_fields_set", None)
    if fields_set is not None:
        return name in fields_set
    return hasattr(value, name)


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _role_config(value: Any, *, source: str) -> HelperModelRole | None:
    if isinstance(value, HelperModelRole):
        return value
    if isinstance(value, Mapping):
        try:
            return HelperModelRole.model_validate(value)
        except Exception:
            logger.debug("Ignoring malformed %s helper-model override", source, exc_info=True)
    return None


def _status_legacy_candidate(role: str, status: dict[str, Any], config: Any) -> _ModelCandidate | None:
    if role not in {"fact_extraction", "fact_reconcile"}:
        return None
    model = _nonempty_string(status.get("extraction_model"))
    if model is None:
        return None

    credential = _nonempty_string(status.get("extraction_credential"))
    if credential is None:
        # Preserve the old bare-model custom-endpoint behavior only when the
        # configured consolidation model is exactly the same underlying model.
        consolidation = _get(config, "consolidation")
        legacy_model = _nonempty_string(_get(consolidation, "llm_model"))
        legacy_credential = _nonempty_string(_get(consolidation, "llm_credential"))
        if legacy_model and legacy_credential and legacy_model.split("/", 1)[-1] == model.split("/", 1)[-1]:
            credential = legacy_credential

    return _ModelCandidate(model, credential, "status.extraction_model")


def _config_legacy_candidates(role: str, config: Any) -> list[_ModelCandidate]:
    consolidation = _get(config, "consolidation")
    candidates: list[_ModelCandidate] = []
    for model_field, credential_field in _ROLE_LEGACY_FIELDS.get(role, ()):
        if not _is_field_explicit(consolidation, model_field):
            continue
        model = _nonempty_string(_get(consolidation, model_field))
        if model is None:
            continue
        credential = _nonempty_string(_get(consolidation, credential_field))
        candidates.append(
            _ModelCandidate(
                model,
                credential,
                f"config.consolidation.{model_field}",
            )
        )
    return candidates


def _configured_credential_for_model(model: str, config: Any) -> str | None:
    """Infer only from model-owned credential settings, never anima background settings."""
    try:
        from core.config.model_mode import _match_models_json

        entry = _match_models_json(model)
        if entry is not None:
            credential = _nonempty_string(entry.get("credential"))
            if credential:
                return credential
    except Exception:
        logger.debug("Unable to inspect models.json credential mapping", exc_info=True)

    from core.config.model_config import _FAMILY_CREDENTIAL_MAP, _model_family

    family = _model_family(model)
    return _FAMILY_CREDENTIAL_MAP.get(family)


def _role_field_value(role_config: HelperModelRole | None, name: str) -> tuple[bool, Any]:
    if role_config is None:
        return False, None
    explicit = name in role_config.model_fields_set
    return explicit, getattr(role_config, name, None)


def _coerce_fallbacks(value: Any, config: Any) -> list[HelperModelFallback]:
    if not isinstance(value, list):
        return []
    resolved: list[HelperModelFallback] = []
    seen: set[tuple[str, str | None]] = set()
    for item in value:
        if isinstance(item, HelperModelFallback):
            model = _nonempty_string(item.model)
            credential = _nonempty_string(item.credential)
        elif isinstance(item, Mapping):
            model = _nonempty_string(item.get("model"))
            credential = _nonempty_string(item.get("credential"))
        elif isinstance(item, str):
            model = _nonempty_string(item)
            credential = None
        else:
            continue
        if model is None:
            continue
        credential = credential or _configured_credential_for_model(model, config)
        key = (model, credential)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(HelperModelFallback(model=model, credential=credential))
    return resolved


def _field_from_layers(
    layers: list[tuple[HelperModelRole | None, str]],
    field: str,
) -> tuple[bool, Any, str]:
    for role_config, source in layers:
        is_explicit, value = _role_field_value(role_config, field)
        if is_explicit:
            return True, value, source
    return False, None, ""


def resolve_helper_model(
    role: str,
    anima_dir: Path | None = None,
    *,
    config: Any | None = None,
) -> ResolvedHelperModel:
    """Resolve one helper role without consulting main/background model settings.

    Priority is per-anima ``status.json`` role override, the legacy status
    extraction keys, ``config.json`` role override, the role's legacy config
    keys, ``helper_models.default``, then ``DEFAULT_CONSOLIDATION_MODEL``.
    """
    if role not in HELPER_MODEL_ROLES:
        raise ValueError(f"Unknown helper-model role: {role!r}")

    if config is None:
        try:
            from core.config.io import load_config

            config = load_config()
        except Exception:
            logger.debug("Failed to load config while resolving helper role %s", role, exc_info=True)
            config = AnimaWorksConfig()

    from core.platform.status_store import read_status

    status = read_status(Path(anima_dir)) if anima_dir is not None else {}
    status_models = status.get("helper_models")
    status_role = _role_config(
        status_models.get(role) if isinstance(status_models, Mapping) else None,
        source=f"status.helper_models.{role}",
    )

    helper_models = _get(config, "helper_models")
    config_role = _role_config(_get(helper_models, role), source=f"config.helper_models.{role}")
    default_role = _role_config(_get(helper_models, "default"), source="config.helper_models.default")

    status_legacy = _status_legacy_candidate(role, status, config)
    config_legacy = _config_legacy_candidates(role, config)

    model_candidates: list[_ModelCandidate] = []
    status_model = _nonempty_string(getattr(status_role, "model", None)) if status_role is not None else None
    if status_model:
        model_candidates.append(
            _ModelCandidate(
                status_model,
                _nonempty_string(getattr(status_role, "credential", None)),
                f"status.helper_models.{role}",
            )
        )
    if status_legacy is not None:
        model_candidates.append(status_legacy)

    model = _nonempty_string(getattr(config_role, "model", None)) if config_role is not None else None
    if model:
        model_candidates.append(
            _ModelCandidate(
                model,
                _nonempty_string(getattr(config_role, "credential", None)),
                f"config.helper_models.{role}",
            )
        )
    model_candidates.extend(config_legacy)

    model = _nonempty_string(getattr(default_role, "model", None)) if default_role is not None else None
    if model:
        model_candidates.append(
            _ModelCandidate(model, _nonempty_string(getattr(default_role, "credential", None)), "helper_models.default")
        )

    if model_candidates:
        selected = model_candidates[0]
    else:
        selected = _ModelCandidate(DEFAULT_CONSOLIDATION_MODEL, None, "code.DEFAULT_CONSOLIDATION_MODEL")

    credential = _nonempty_string(getattr(status_role, "credential", None)) if status_role is not None else None
    if credential is None and selected.source == "status.extraction_model" and status_legacy is not None:
        credential = status_legacy.credential
    if credential is None and config_role is not None:
        credential = _nonempty_string(getattr(config_role, "credential", None))
    if credential is None and selected.source.startswith("config.consolidation."):
        credential = selected.credential
    if credential is None:
        credential = next(
            (
                candidate.credential
                for candidate in config_legacy
                if candidate.model == selected.model and candidate.credential
            ),
            None,
        )
    if credential is None and default_role is not None:
        credential = _nonempty_string(getattr(default_role, "credential", None))
    credential = credential or _configured_credential_for_model(selected.model, config)

    layers = [
        (status_role, f"status.helper_models.{role}"),
        (config_role, f"config.helper_models.{role}"),
        (default_role, "helper_models.default"),
    ]
    _fallback_is_explicit, fallback_value, _fallback_source = _field_from_layers(layers, "fallbacks")
    if _fallback_is_explicit:
        fallbacks = _coerce_fallbacks(fallback_value, config)
    elif role in _ROLES_WITH_LEGACY_FALLBACK:
        consolidation = _get(config, "consolidation")
        legacy_fallback_model = _nonempty_string(_get(consolidation, "llm_fallback_model"))
        if _is_field_explicit(consolidation, "llm_fallback_model") and legacy_fallback_model:
            fallback_credential = _nonempty_string(_get(consolidation, "llm_fallback_credential"))
            fallbacks = _coerce_fallbacks(
                [{"model": legacy_fallback_model, "credential": fallback_credential}],
                config,
            )
        else:
            _default_fallback_set, default_fallbacks, _ = _field_from_layers([(default_role, "default")], "fallbacks")
            fallbacks = _coerce_fallbacks(default_fallbacks, config) if _default_fallback_set else []
    else:
        _default_fallback_set, default_fallbacks, _ = _field_from_layers([(default_role, "default")], "fallbacks")
        fallbacks = _coerce_fallbacks(default_fallbacks, config) if _default_fallback_set else []

    allow_is_explicit, allow_value, _allow_source = _field_from_layers(layers, "allow_agent_sdk_fallback")
    allow_agent_sdk_fallback = bool(allow_value) if allow_is_explicit else False

    max_tokens_is_explicit, max_tokens_value, _max_tokens_source = _field_from_layers(layers, "max_output_tokens")
    max_output_tokens: int | None = None
    if max_tokens_is_explicit and max_tokens_value is not None:
        try:
            candidate_tokens = int(max_tokens_value)
        except (TypeError, ValueError):
            candidate_tokens = 0
        if candidate_tokens > 0:
            max_output_tokens = candidate_tokens

    # Do not list the selected model/credential a second time as a fallback.
    fallbacks = [item for item in fallbacks if (item.model, item.credential) != (selected.model, credential)]
    return ResolvedHelperModel(
        model=selected.model,
        credential=credential,
        fallbacks=fallbacks,
        allow_agent_sdk_fallback=allow_agent_sdk_fallback,
        max_output_tokens=max_output_tokens,
        source=selected.source,
    )


def _model_family(model: str) -> str:
    from core.config.model_config import _model_family as model_family

    return model_family(model)


def _credential_family(credential_name: str, credential: Any) -> str | None:
    name_family = {
        "anthropic": "anthropic",
        "openai": "openai",
        "ollama": "ollama",
        "google": "google",
        "vertex_ai": "vertex_ai",
        "gemini": "google",
        "azure": "azure",
        "grok": "grok",
        "xai": "grok",
        "agy": "google",
        "bedrock": "bedrock",
        "codex": "openai",
    }
    by_type = {
        "openai": "openai",
        "codex_login": "openai",
        "codex_azure": "openai",
        "claude_code_login": "anthropic",
        "anthropic": "anthropic",
        "ollama": "ollama",
        "google": "google",
        "gemini": "google",
        "vertex_ai": "vertex_ai",
        "azure": "azure",
        "grok": "grok",
        "bedrock": "bedrock",
    }
    credential_type = _get(credential, "type")
    if isinstance(credential_type, str) and credential_type.lower() in by_type:
        return by_type[credential_type.lower()]
    return name_family.get(credential_name.lower())


def validate_helper_model_credentials(
    animas_dir: Path | None = None,
    *,
    config: Any | None = None,
) -> list[str]:
    """Log one-line warnings for unresolved or contradictory helper credentials."""
    if config is None:
        try:
            from core.config.io import load_config

            config = load_config()
        except Exception:
            logger.warning("Helper model credential validation skipped: config could not be loaded")
            return []

    anima_dirs: list[Path | None] = [None]
    if animas_dir is not None:
        root = Path(animas_dir)
        if (root / "status.json").is_file():
            anima_dirs.append(root)
        elif root.is_dir():
            anima_dirs.extend(
                child
                for child in sorted(root.iterdir())
                if child.is_dir() and ((child / "status.json").is_file() or (child / IDENTITY_FILE).is_file())
            )

    credentials = _get(config, "credentials", {})
    if not isinstance(credentials, Mapping):
        credentials = {}
    diagnostics: list[str] = []
    emitted: set[tuple[str, str, str, str]] = set()

    for anima_dir in anima_dirs:
        anima_name = anima_dir.name if anima_dir is not None else "*"
        for role in HELPER_MODEL_ROLES:
            try:
                resolved = resolve_helper_model(role, anima_dir, config=config)
            except Exception as exc:
                diagnostics.append(f"role={role} anima={anima_name} resolution={type(exc).__name__}")
                continue
            credential_name = resolved.credential
            issue = ""
            if credential_name and credential_name not in credentials:
                issue = "credential_missing"
            elif credential_name:
                from core.config.model_config import _FAMILY_CREDENTIAL_MAP

                model_family = _model_family(resolved.model)
                expected_name = _FAMILY_CREDENTIAL_MAP.get(model_family)
                if expected_name and credential_name != expected_name:
                    credential = credentials.get(credential_name)
                    has_custom_endpoint = bool(_get(credential, "base_url"))
                    actual_family = _credential_family(credential_name, credential)
                    expected_family = _credential_family(expected_name, credentials.get(expected_name))
                    if (
                        not has_custom_endpoint
                        and actual_family
                        and expected_family
                        and actual_family != expected_family
                    ):
                        issue = "provider_mismatch"
            if not issue:
                continue
            key = (role, resolved.model, credential_name or "", issue)
            if key in emitted:
                continue
            emitted.add(key)
            detail = f"role={role} anima={anima_name} model={resolved.model} credential={credential_name} issue={issue}"
            diagnostics.append(detail)
            logger.warning("Helper model credential validation: %s", detail)

    return diagnostics


__all__ = [
    "HELPER_MODEL_ROLES",
    "ResolvedHelperModel",
    "resolve_helper_model",
    "validate_helper_model_credentials",
]
