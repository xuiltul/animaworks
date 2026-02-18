from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Setup wizard API routes for first-launch configuration."""

import logging
import shutil
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("animaworks.routes.setup")

# ── Available providers ────────────────────────────────────
AVAILABLE_PROVIDERS = [
    {
        "id": "anthropic",
        "name": "Anthropic",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-3.5-20241022"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "models": ["openai/gpt-4o", "openai/gpt-4o-mini"],
        "env_key": "OPENAI_API_KEY",
    },
    {
        "id": "google",
        "name": "Google",
        "models": ["google/gemini-2.0-flash", "google/gemini-2.5-pro"],
        "env_key": "GOOGLE_API_KEY",
    },
    {
        "id": "ollama",
        "name": "Ollama (Local)",
        "models": ["ollama/gemma3:27b", "ollama/llama3.3:70b"],
        "env_key": None,
    },
]

AVAILABLE_LOCALES = [
    "en", "ja", "zh-CN", "zh-TW", "ko", "es", "fr", "de",
    "pt", "it", "ru", "ar", "hi", "tr", "vi", "th", "id",
]


# ── Request/Response models ────────────────────────────────


class ValidateKeyRequest(BaseModel):
    provider: str
    api_key: str = ""
    ollama_url: str = ""


class AnimaSetup(BaseModel):
    name: str


class UserSetup(BaseModel):
    username: str
    display_name: str = ""
    bio: str = ""


class SetupCompleteRequest(BaseModel):
    locale: str = "ja"
    credentials: dict[str, dict[str, str]] = {}
    anima: AnimaSetup | None = None
    user: UserSetup | None = None


# ── Router factory ─────────────────────────────────────────


def create_setup_router() -> APIRouter:
    """Create the setup wizard API router."""
    router = APIRouter(prefix="/api/setup", tags=["setup"])

    # ── GET /api/setup/environment ──────────────────────────

    @router.get("/environment")
    async def get_environment(request: Request) -> dict[str, Any]:
        """Return environment information for the setup wizard."""
        from core.config import load_config

        config = load_config()
        claude_available = shutil.which("claude") is not None

        return {
            "claude_code_available": claude_available,
            "locale": config.locale,
            "providers": AVAILABLE_PROVIDERS,
            "available_locales": AVAILABLE_LOCALES,
        }

    # ── GET /api/setup/detect-locale ───────────────────────

    @router.get("/detect-locale")
    async def detect_locale(request: Request) -> dict[str, Any]:
        """Detect locale from Accept-Language header."""
        accept_lang = request.headers.get("accept-language", "")
        detected = _parse_accept_language(accept_lang)
        return {
            "detected": detected,
            "available": AVAILABLE_LOCALES,
        }

    # ── POST /api/setup/validate-key ───────────────────────

    @router.post("/validate-key")
    async def validate_key(body: ValidateKeyRequest) -> dict[str, Any]:
        """Validate an API key by making a small test request."""
        provider = body.provider
        api_key = body.api_key

        if provider == "anthropic":
            return await _validate_anthropic_key(api_key)
        elif provider == "openai":
            return await _validate_openai_key(api_key)
        elif provider == "google":
            return await _validate_google_key(api_key)
        elif provider == "ollama":
            return {"valid": True, "message": "Ollama does not require an API key"}
        else:
            return {"valid": False, "message": f"Unknown provider: {provider}"}

    # ── POST /api/setup/complete ───────────────────────────

    @router.post("/complete")
    async def complete_setup(
        body: SetupCompleteRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Finalize setup: save config, create anima, mark complete."""
        from core.config import (
            CredentialConfig,
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )
        from core.paths import get_animas_dir

        config = load_config()

        # Update locale
        config.locale = body.locale

        # Update credentials
        for cred_name, cred_data in body.credentials.items():
            config.credentials[cred_name] = CredentialConfig(
                api_key=cred_data.get("api_key", ""),
                base_url=cred_data.get("base_url"),
            )

        # Create anima if specified
        if body.anima:
            from core.anima_factory import create_blank

            animas_dir = get_animas_dir()
            anima_name = body.anima.name

            try:
                create_blank(animas_dir, anima_name)
                # Read supervisor from status.json if present
                from core.config import read_anima_supervisor

                anima_dir = animas_dir / anima_name
                supervisor = (
                    read_anima_supervisor(anima_dir)
                    if anima_dir.exists()
                    else None
                )
                config.animas[anima_name] = AnimaModelConfig(
                    supervisor=supervisor,
                )
                logger.info("Created anima '%s' during setup", anima_name)
            except FileExistsError:
                logger.warning("Anima '%s' already exists, skipping creation", anima_name)
            except Exception:
                logger.error("Failed to create anima during setup", exc_info=True)
                return JSONResponse(
                    {"error": "Failed to create anima"},
                    status_code=500,
                )

        # Mark setup as complete
        config.setup_complete = True
        save_config(config)
        invalidate_cache()

        # Update app state so the middleware switches behaviour immediately
        request.app.state.setup_complete = True

        # Create auth config with owner info
        if body.user:
            from core.auth.manager import save_auth
            from core.auth.models import AuthConfig, AuthUser

            owner = AuthUser(
                username=body.user.username,
                display_name=body.user.display_name,
                bio=body.user.bio,
            )
            auth_config = AuthConfig(owner=owner)
            save_auth(auth_config)
            logger.info("Created auth config for owner '%s'", body.user.username)

            # Create initial user profile in shared/users/
            from core.paths import get_shared_dir

            user_profile_dir = get_shared_dir() / "users" / body.user.username
            user_profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = user_profile_dir / "index.md"
            if not profile_path.exists():
                profile_lines = [f"# {body.user.display_name or body.user.username}\n"]
                if body.user.bio:
                    profile_lines.append(f"\n{body.user.bio}\n")
                profile_path.write_text("".join(profile_lines), encoding="utf-8")
                logger.info("Created user profile for '%s'", body.user.username)

        # Re-scan animas and start processes
        animas_dir = request.app.state.animas_dir
        new_anima_names: list[str] = []
        if animas_dir.exists():
            for anima_dir in sorted(animas_dir.iterdir()):
                if anima_dir.is_dir() and (anima_dir / "identity.md").exists():
                    new_anima_names.append(anima_dir.name)
        request.app.state.anima_names = new_anima_names

        if new_anima_names:
            try:
                await request.app.state.supervisor.start_all(new_anima_names)
                logger.info("Started %d anima(s) after setup", len(new_anima_names))
            except Exception:
                logger.error("Failed to start animas after setup", exc_info=True)

        logger.info("Setup completed successfully")
        return {"status": "ok", "message": "Setup complete. Reload to access the dashboard."}

    return router


# ── Helper functions ───────────────────────────────────────


def _normalize_locale(
    lang: str,
    zh_simplified: set[str],
    zh_traditional: set[str],
) -> str:
    """Normalize a locale tag to match AVAILABLE_LOCALES entries.

    - Chinese variants are mapped to zh-CN or zh-TW
    - Other languages are reduced to their primary subtag (e.g. en-US → en)
    """
    parts = lang.replace("_", "-").split("-")
    primary = parts[0]

    if primary == "zh":
        if len(parts) < 2:
            return "zh-CN"  # bare "zh" defaults to Simplified
        subtag = parts[1].lower()
        if subtag in zh_traditional:
            return "zh-TW"
        return "zh-CN"

    return primary


def _parse_accept_language(header: str) -> str:
    """Parse Accept-Language header and return best matching locale.

    Supports weighted values like ``ja;q=0.9,en-US;q=0.8``.
    Handles Chinese variants: zh-CN/zh-Hans/zh-SG → zh-CN,
    zh-TW/zh-Hant/zh-HK/zh-MO → zh-TW, bare zh → zh-CN.
    Returns the first match from AVAILABLE_LOCALES, or ``"ja"`` as fallback.
    """
    if not header:
        return "ja"

    # Chinese variant mapping
    _ZH_SIMPLIFIED = {"cn", "hans", "sg"}
    _ZH_TRADITIONAL = {"tw", "hant", "hk", "mo"}

    # Parse entries: "ja;q=0.9,en-US;q=0.8,en;q=0.7"
    entries: list[tuple[float, str]] = []
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue
        if ";q=" in part:
            lang, _, q_str = part.partition(";q=")
            try:
                q = float(q_str.strip())
            except ValueError:
                q = 0.0
        else:
            lang = part
            q = 1.0

        lang = lang.strip().lower()
        normalized = _normalize_locale(lang, _ZH_SIMPLIFIED, _ZH_TRADITIONAL)
        entries.append((q, normalized))

    # Sort by quality descending
    entries.sort(key=lambda e: e[0], reverse=True)

    for _q, lang in entries:
        if lang in AVAILABLE_LOCALES:
            return lang

    return "ja"


async def _validate_anthropic_key(api_key: str) -> dict[str, Any]:
    """Validate an Anthropic API key with a minimal request."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-3.5-20241022",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        if resp.status_code in (200, 201):
            return {"valid": True, "message": "API key is valid"}
        if resp.status_code == 401:
            return {"valid": False, "message": "Invalid API key"}
        return {"valid": False, "message": f"Unexpected status: {resp.status_code}"}
    except Exception as exc:
        return {"valid": False, "message": f"Connection error: {exc}"}


async def _validate_openai_key(api_key: str) -> dict[str, Any]:
    """Validate an OpenAI API key by listing models."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            return {"valid": True, "message": "API key is valid"}
        if resp.status_code == 401:
            return {"valid": False, "message": "Invalid API key"}
        return {"valid": False, "message": f"Unexpected status: {resp.status_code}"}
    except Exception as exc:
        return {"valid": False, "message": f"Connection error: {exc}"}


async def _validate_google_key(api_key: str) -> dict[str, Any]:
    """Validate a Google API key by listing models."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key},
            )
        if resp.status_code == 200:
            return {"valid": True, "message": "API key is valid"}
        if resp.status_code in (400, 401, 403):
            return {"valid": False, "message": "Invalid API key"}
        return {"valid": False, "message": f"Unexpected status: {resp.status_code}"}
    except Exception as exc:
        return {"valid": False, "message": f"Connection error: {exc}"}
