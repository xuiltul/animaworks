from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("animaworks.routes.config")


def _mask_secrets(obj: object) -> object:
    """Recursively mask sensitive values in a config dict."""
    if isinstance(obj, dict):
        return {k: _mask_value(k, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(item) for item in obj]
    return obj


def _mask_value(key: str, value: object) -> object:
    """Mask a value if its key suggests it contains a secret."""
    if isinstance(value, str) and any(
        kw in key.lower() for kw in ("key", "token", "secret", "password")
    ):
        if len(value) > 8:
            return value[:3] + "..." + value[-4:]
        return "***"
    if isinstance(value, (dict, list)):
        return _mask_secrets(value)
    return value


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
            raise HTTPException(
                status_code=500, detail=f"Invalid config JSON: {exc}"
            )

        return _mask_secrets(config)

    @router.get("/system/init-status")
    async def init_status(request: Request):
        """Check initialization status of AnimaWorks."""
        base_dir = Path.home() / ".animaworks"
        config_path = base_dir / "config.json"
        persons_dir = base_dir / "persons"
        shared_dir = base_dir / "shared"

        # Count persons
        persons_count = 0
        if persons_dir.exists():
            for d in persons_dir.iterdir():
                if d.is_dir() and (d / "identity.md").exists():
                    persons_count += 1

        # Check API keys from environment
        api_keys = {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "google": bool(os.environ.get("GOOGLE_API_KEY")),
        }

        config_exists = config_path.exists()
        initialized = config_exists and persons_count > 0

        return {
            "config_exists": config_exists,
            "persons_count": persons_count,
            "api_keys": api_keys,
            "shared_dir_exists": shared_dir.exists(),
            "initialized": initialized,
        }

    return router
