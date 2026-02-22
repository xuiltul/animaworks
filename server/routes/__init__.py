from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from server.routes.assets import create_assets_router
from server.routes.auth import create_auth_router
from server.routes.channels import create_channels_router
from server.routes.chat import create_chat_router
from server.routes.config_routes import create_config_router
from server.routes.internal import create_internal_router
from server.routes.logs_routes import create_logs_router
from server.routes.memory_routes import create_memory_router
from server.routes.animas import create_animas_router
from server.routes.sessions import create_sessions_router
from server.routes.system import create_system_router
from server.routes.tool_prompts import create_tool_prompts_router
from server.routes.users import create_users_router
from server.routes.webhooks import create_webhooks_router
from server.routes.websocket_route import create_websocket_router


def create_router() -> APIRouter:
    router = APIRouter()
    api = APIRouter(prefix="/api")

    api.include_router(create_animas_router())
    api.include_router(create_channels_router())
    api.include_router(create_chat_router())
    api.include_router(create_memory_router())
    api.include_router(create_sessions_router())
    api.include_router(create_system_router())
    api.include_router(create_config_router())
    api.include_router(create_logs_router())
    api.include_router(create_assets_router())
    api.include_router(create_internal_router())
    api.include_router(create_auth_router())
    api.include_router(create_tool_prompts_router())
    api.include_router(create_users_router())
    api.include_router(create_webhooks_router())

    router.include_router(api)
    router.include_router(create_websocket_router())

    return router
