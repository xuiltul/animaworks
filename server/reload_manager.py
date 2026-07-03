from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Central hot-reload orchestrator for configuration and connections."""

import asyncio
import logging
from typing import Any

from fastapi import FastAPI

logger = logging.getLogger("animaworks.reload_manager")


class ConfigReloadManager:
    """Central hot-reload orchestrator.

    Provides granular and full-system reload capabilities with
    ``asyncio.Lock`` serialization to prevent concurrent reloads.
    """

    def __init__(self, app: FastAPI) -> None:
        self._app = app
        self._lock = asyncio.Lock()

    async def reload_all(self) -> dict[str, Any]:
        """Reload all components: config cache + credentials + Slack + Animas."""
        async with self._lock:
            results: dict[str, Any] = {}
            results["config"] = self._reload_config_cache()
            results["credentials"] = self._reload_credentials()
            results["slack"] = await self._reload_slack()
            results["zoom"] = self._reload_zoom()
            results["animas"] = await self._reload_animas()
            return results

    async def reload_slack(self) -> dict[str, Any]:
        """Reload Slack Socket Mode connections only."""
        async with self._lock:
            return await self._reload_slack()

    async def reload_credentials(self) -> dict[str, Any]:
        """Reload credential caches and dependent connections."""
        async with self._lock:
            result = self._reload_credentials()
            slack_result = await self._reload_slack()
            zoom_result = self._reload_zoom()
            return {**result, "slack": slack_result, "zoom": zoom_result}

    async def reload_animas(self) -> dict[str, Any]:
        """Sync Anima processes with disk state."""
        async with self._lock:
            return await self._reload_animas()

    def _reload_config_cache(self) -> dict[str, Any]:
        """Invalidate config.json + models.json caches."""
        from core.config.models import (
            invalidate_cache,
            invalidate_models_json_cache,
            load_config,
        )

        invalidate_cache()
        invalidate_models_json_cache()
        try:
            config = load_config()
            return {"status": "ok", "animas_count": len(config.animas)}
        except Exception as exc:
            logger.exception("Failed to reload config")
            return {"status": "error", "error": str(exc)}

    def _reload_credentials(self) -> dict[str, Any]:
        """Invalidate vault cache (shared/credentials.json has no cache)."""
        from core.config.vault import invalidate_vault_cache

        invalidate_vault_cache()
        return {"status": "ok"}

    async def _reload_slack(self) -> dict[str, Any]:
        """Reload Slack Socket Mode handlers (diff-based)."""
        manager = getattr(self._app.state, "slack_socket_manager", None)
        if manager is None:
            return {"status": "skipped", "reason": "no_manager"}
        try:
            return await manager.reload()
        except Exception as exc:
            logger.exception("Failed to reload Slack")
            return {"status": "error", "error": str(exc)}

    def _reload_zoom(self) -> dict[str, Any]:
        """Refresh Zoom RTMS gateway config (mappings, thresholds, creds)."""
        manager = getattr(self._app.state, "zoom_gateway_manager", None)
        if manager is None:
            return {"status": "skipped", "reason": "no_manager"}
        try:
            manager.reload()
            return {"status": "ok"}
        except Exception as exc:
            logger.exception("Failed to reload Zoom RTMS gateway")
            return {"status": "error", "error": str(exc)}

    async def _reload_animas(self) -> dict[str, Any]:
        """Sync Anima processes with disk state (add/stop only, no restart)."""
        supervisor = getattr(self._app.state, "supervisor", None)
        if supervisor is None:
            return {"status": "skipped", "reason": "no_supervisor"}
        try:
            await supervisor._reconcile()
            return {"status": "ok"}
        except Exception as exc:
            logger.exception("Failed to reload animas")
            return {"status": "error", "error": str(exc)}
