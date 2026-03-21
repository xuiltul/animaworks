from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config.local_llm import (
    apply_local_llm_presets_to_animas,
    normalize_ollama_base_url,
    normalize_ollama_model_name,
)
from core.config.models import (
    CredentialConfig,
    DEFAULT_LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_LLM_PRESETS,
    DEFAULT_LOCAL_LLM_ROLE_PRESETS,
    LocalLLMConfig,
    load_config,
    save_config,
)
from core.i18n import t
from core.paths import get_animas_dir
from core.platform.claude_code import is_claude_code_available
from core.platform.codex import is_codex_cli_available, is_codex_login_available

logger = logging.getLogger("animaworks.routes.config")


class UpdateAnthropicAuthRequest(BaseModel):
    auth_mode: str = "api_key"
    api_key: str = ""


class UpdateOpenAIAuthRequest(BaseModel):
    auth_mode: str = "api_key"
    api_key: str = ""


class UpdateLocalLLMRequest(BaseModel):
    base_url: str = DEFAULT_LOCAL_LLM_BASE_URL
    default_model: str = DEFAULT_LOCAL_LLM_PRESETS["coding"]
    presets: dict[str, str] = {}
    role_presets: dict[str, str] = {}


def _mask_secrets(obj: object) -> object:
    """Recursively mask sensitive values in a config dict."""
    if isinstance(obj, dict):
        return {k: _mask_value(k, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(item) for item in obj]
    return obj


def _mask_value(key: str, value: object) -> object:
    """Mask a value if its key suggests it contains a secret."""
    if isinstance(value, str) and any(kw in key.lower() for kw in ("key", "token", "secret", "password")):
        if len(value) > 8:
            return value[:3] + "..." + value[-4:]
        return "***"
    if isinstance(value, (dict, list)):
        return _mask_secrets(value)
    return value


def _serialize_openai_auth() -> dict[str, object]:
    """Return current OpenAI auth config and runtime availability."""
    config = load_config()
    credential = config.credentials.get("openai", CredentialConfig())
    auth_mode = credential.type or "api_key"
    config_present = "openai" in config.credentials
    config_api_key_configured = bool(credential.api_key)
    env_api_key_configured = bool(os.environ.get("OPENAI_API_KEY"))
    codex_cli_available = is_codex_cli_available()
    codex_login_available = is_codex_login_available()

    configured = False
    if auth_mode == "codex_login":
        configured = codex_login_available
    elif auth_mode == "api_key":
        configured = config_api_key_configured or env_api_key_configured

    return {
        "auth_mode": auth_mode,
        "config_present": config_present,
        "config_api_key_configured": config_api_key_configured,
        "env_api_key_configured": env_api_key_configured,
        "codex_cli_available": codex_cli_available,
        "codex_login_available": codex_login_available,
        "configured": configured,
    }


def _serialize_anthropic_auth() -> dict[str, object]:
    """Return current Anthropic auth config and runtime availability."""
    config = load_config()
    credential = config.credentials.get("anthropic", CredentialConfig())
    auth_mode = credential.type or "api_key"
    config_present = "anthropic" in config.credentials
    config_api_key_configured = bool(credential.api_key)
    env_api_key_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
    claude_code_available = is_claude_code_available()

    configured = False
    if auth_mode == "claude_code_login":
        configured = claude_code_available
    elif auth_mode == "api_key":
        configured = config_api_key_configured or env_api_key_configured

    return {
        "auth_mode": auth_mode,
        "config_present": config_present,
        "config_api_key_configured": config_api_key_configured,
        "env_api_key_configured": env_api_key_configured,
        "claude_code_available": claude_code_available,
        "configured": configured,
    }


def _list_ollama_models(base_url: str) -> list[str]:
    response = httpx.get(
        f"{base_url}/api/tags",
        timeout=httpx.Timeout(10.0, connect=5.0),
    )
    response.raise_for_status()
    data = response.json()
    models = sorted(
        {
            normalize_ollama_model_name(str(item.get("name", "")).strip())
            for item in data.get("models", [])
            if item.get("name")
        }
    )
    return [model for model in models if model]


def _serialize_local_llm() -> dict[str, object]:
    config = load_config()
    local_llm = LocalLLMConfig.model_validate(config.local_llm.model_dump())
    base_url = normalize_ollama_base_url(local_llm.base_url)
    default_model = normalize_ollama_model_name(local_llm.default_model)
    presets = {
        name: normalize_ollama_model_name(model)
        for name, model in local_llm.presets.items()
    }

    available_models: list[str] = []
    reachable = False
    error: str | None = None
    try:
        available_models = _list_ollama_models(base_url)
        reachable = True
    except Exception as exc:  # pragma: no cover
        error = str(exc)

    ollama_credential = config.credentials.get("ollama")
    configured = (
        config.anima_defaults.credential == "ollama"
        and normalize_ollama_model_name(config.anima_defaults.model) == default_model
        and ollama_credential is not None
        and normalize_ollama_base_url(ollama_credential.base_url) == base_url
    )

    return {
        "base_url": base_url,
        "default_model": default_model,
        "presets": presets,
        "role_presets": dict(local_llm.role_presets),
        "recommended_presets": dict(DEFAULT_LOCAL_LLM_PRESETS),
        "recommended_role_presets": dict(DEFAULT_LOCAL_LLM_ROLE_PRESETS),
        "available_models": available_models,
        "reachable": reachable,
        "error": error,
        "configured": configured,
        "current_default_model": config.anima_defaults.model,
        "current_default_credential": config.anima_defaults.credential,
    }


def create_config_router() -> APIRouter:
    router = APIRouter()

    @router.get("/system/config")
    async def get_config(request: Request):
        """Read and return the AnimaWorks config with masked secrets."""
        config_path = Path.home() / ".animaworks" / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid config JSON: {exc}") from exc

        return _mask_secrets(config)

    @router.get("/system/init-status")
    async def init_status(request: Request):
        """Check initialization status of AnimaWorks."""
        base_dir = Path.home() / ".animaworks"
        config_path = base_dir / "config.json"
        animas_dir = base_dir / "animas"
        shared_dir = base_dir / "shared"

        # Count animas
        animas_count = 0
        if animas_dir.exists():
            for d in animas_dir.iterdir():
                if d.is_dir() and (d / "identity.md").exists():
                    animas_count += 1

        # Check API keys / subscription auth
        has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        anthropic_cred = load_config().credentials.get("anthropic", CredentialConfig())
        has_anthropic_subscription = (
            anthropic_cred.type == "claude_code_login" and is_claude_code_available()
        )
        has_anthropic = has_anthropic_key or has_anthropic_subscription
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_codex_login = is_codex_login_available()
        has_openai_auth = has_openai or has_codex_login
        has_google = bool(os.environ.get("GOOGLE_API_KEY"))

        config_exists = config_path.exists()
        initialized = config_exists and animas_count > 0

        return {
            "checks": [
                {"label": t("config.config_file"), "ok": config_exists},
                {
                    "label": t("config.anima_registration"),
                    "ok": animas_count > 0,
                    "detail": t("config.anima_count_detail", count=animas_count),
                },
                {"label": t("config.shared_dir"), "ok": shared_dir.exists()},
                {"label": t("config.anthropic_auth"), "ok": has_anthropic},
                {"label": t("config.openai_auth"), "ok": has_openai_auth},
                {"label": t("config.google_api_key"), "ok": has_google},
                {"label": t("config.init_complete"), "ok": initialized},
            ],
            "config_exists": config_exists,
            "animas_count": animas_count,
            "api_keys": {
                "anthropic": has_anthropic,
                "openai": has_openai_auth,
                "codex_login": has_codex_login,
                "google": has_google,
            },
            "shared_dir_exists": shared_dir.exists(),
            "initialized": initialized,
        }

    @router.get("/settings/anthropic-auth")
    async def get_anthropic_auth(request: Request):
        """Return current Anthropic auth mode and runtime availability."""
        return _serialize_anthropic_auth()

    @router.get("/settings/openai-auth")
    async def get_openai_auth(request: Request):
        """Return current OpenAI auth mode and runtime availability."""
        return _serialize_openai_auth()

    @router.get("/settings/local-llm")
    async def get_local_llm(request: Request):
        """Return local Ollama-backed model settings and runtime availability."""
        return _serialize_local_llm()

    @router.put("/settings/anthropic-auth")
    async def update_anthropic_auth(body: UpdateAnthropicAuthRequest, request: Request):
        """Persist Anthropic auth mode in config.json for the settings UI."""
        auth_mode = body.auth_mode.strip()
        if auth_mode not in ("api_key", "claude_code_login"):
            raise HTTPException(status_code=400, detail="Invalid auth mode. Must be 'api_key' or 'claude_code_login'.")

        config = load_config()
        current = config.credentials.get("anthropic", CredentialConfig())

        if auth_mode == "claude_code_login":
            if not is_claude_code_available():
                raise HTTPException(status_code=400, detail="Claude Code CLI is not installed.")
            config.credentials["anthropic"] = CredentialConfig(
                type="claude_code_login",
                api_key="",
                base_url=current.base_url,
                keys=dict(current.keys),
            )
            config.anima_defaults.mode_s_auth = "max"
            logger.info("Anthropic auth set to subscription (claude_code_login), mode_s_auth=max")
        else:
            api_key = body.api_key.strip()
            if not api_key:
                raise HTTPException(status_code=400, detail="API key is required for api_key mode.")
            config.credentials["anthropic"] = CredentialConfig(
                type="api_key",
                api_key=api_key,
                base_url=current.base_url,
                keys=dict(current.keys),
            )

        save_config(config)
        return _serialize_anthropic_auth()

    @router.put("/settings/openai-auth")
    async def update_openai_auth(body: UpdateOpenAIAuthRequest, request: Request):
        """Persist OpenAI auth mode in config.json for the settings UI."""
        auth_mode = body.auth_mode.strip()
        if auth_mode not in ("api_key", "codex_login"):
            raise HTTPException(status_code=400, detail=t("config.openai_auth_invalid_mode"))

        config = load_config()
        current = config.credentials.get("openai", CredentialConfig())

        if auth_mode == "codex_login":
            if not is_codex_cli_available():
                raise HTTPException(status_code=400, detail=t("config.codex_cli_not_installed"))
            if not is_codex_login_available():
                raise HTTPException(status_code=400, detail=t("config.codex_login_not_available"))
            config.credentials["openai"] = CredentialConfig(
                type="codex_login",
                api_key="",
                base_url=current.base_url,
                keys=dict(current.keys),
            )
        else:
            api_key = body.api_key.strip()
            if not api_key:
                raise HTTPException(status_code=400, detail=t("config.openai_api_key_required"))
            config.credentials["openai"] = CredentialConfig(
                type="api_key",
                api_key=api_key,
                base_url=current.base_url,
                keys=dict(current.keys),
            )

        save_config(config)
        return _serialize_openai_auth()

    @router.put("/settings/local-llm")
    async def update_local_llm(body: UpdateLocalLLMRequest, request: Request):
        """Persist local LLM settings and make Ollama the default execution target."""
        base_url = normalize_ollama_base_url(body.base_url)
        config = load_config()
        current_local_llm = LocalLLMConfig.model_validate(config.local_llm.model_dump())

        default_model = normalize_ollama_model_name(body.default_model)
        presets = dict(current_local_llm.presets)
        for name, model in body.presets.items():
            if name in presets and model.strip():
                presets[name] = normalize_ollama_model_name(model)

        role_presets = dict(current_local_llm.role_presets)
        for role_name, preset_name in body.role_presets.items():
            if role_name in role_presets and preset_name in presets:
                role_presets[role_name] = preset_name

        available_models = set(_list_ollama_models(base_url))
        requested_models = {default_model, *presets.values()}
        missing = sorted(model for model in requested_models if model not in available_models)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Ollama models not found on {base_url}: {', '.join(missing)}",
            )

        config.local_llm = LocalLLMConfig(
            base_url=base_url,
            default_model=default_model,
            presets=presets,
            role_presets=role_presets,
        )
        config.credentials["ollama"] = CredentialConfig(
            type="ollama",
            api_key="",
            base_url=base_url,
        )
        config.anima_defaults.model = default_model
        config.anima_defaults.credential = "ollama"

        save_config(config)
        return _serialize_local_llm()

    @router.post("/settings/local-llm/apply-role-presets")
    async def apply_local_llm_role_presets(request: Request):
        """Apply the configured role-based local LLM presets to existing animas."""
        config = load_config()
        updated = apply_local_llm_presets_to_animas(get_animas_dir(), config)
        return {"updated": updated, "count": len(updated)}

    return router
