from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

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
        animas_dir = base_dir / "animas"
        shared_dir = base_dir / "shared"

        # Count animas
        animas_count = 0
        if animas_dir.exists():
            for d in animas_dir.iterdir():
                if d.is_dir() and (d / "identity.md").exists():
                    animas_count += 1

        # Check API keys from environment
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_google = bool(os.environ.get("GOOGLE_API_KEY"))

        config_exists = config_path.exists()
        initialized = config_exists and animas_count > 0

        return {
            "checks": [
                {"label": "設定ファイル", "ok": config_exists},
                {"label": "Anima登録", "ok": animas_count > 0,
                 "detail": f"{animas_count}名"},
                {"label": "共有ディレクトリ", "ok": shared_dir.exists()},
                {"label": "Anthropic APIキー", "ok": has_anthropic},
                {"label": "OpenAI APIキー", "ok": has_openai},
                {"label": "Google APIキー", "ok": has_google},
                {"label": "初期化完了", "ok": initialized},
            ],
            "config_exists": config_exists,
            "animas_count": animas_count,
            "api_keys": {
                "anthropic": has_anthropic,
                "openai": has_openai,
                "google": has_google,
            },
            "shared_dir_exists": shared_dir.exists(),
            "initialized": initialized,
        }

    return router
