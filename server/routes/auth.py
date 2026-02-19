from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Authentication API routes."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth.manager import (
    create_session,
    find_user,
    load_auth,
    revoke_session,
    save_auth,
    verify_password,
)

logger = logging.getLogger("animaworks.routes.auth")


class LoginRequest(BaseModel):
    username: str
    password: str


def create_auth_router() -> APIRouter:
    router = APIRouter(tags=["auth"])

    @router.post("/auth/login")
    async def login(body: LoginRequest, request: Request):
        auth_config = load_auth()

        if auth_config.auth_mode == "local_trust":
            return JSONResponse(
                {"error": "Authentication is not enabled"},
                status_code=400,
            )

        if len(body.password) > 128:
            logger.warning("Login rejected: password too long for user '%s'", body.username)
            return JSONResponse(
                {"error": "Invalid credentials"},
                status_code=401,
            )

        user = find_user(auth_config, body.username)
        if not user or not user.password_hash:
            logger.warning("Login failed: unknown user '%s'", body.username)
            return JSONResponse(
                {"error": "Invalid credentials"},
                status_code=401,
            )

        if not verify_password(body.password, user.password_hash):
            logger.warning("Login failed: wrong password for user '%s'", body.username)
            return JSONResponse(
                {"error": "Invalid credentials"},
                status_code=401,
            )

        token = create_session(auth_config, user.username)
        save_auth(auth_config)

        response = JSONResponse({
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        })
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="strict",
            path="/",
        )
        logger.info("User '%s' logged in", user.username)
        return response

    @router.post("/auth/logout")
    async def logout(request: Request):
        token = request.cookies.get("session_token")
        if token:
            revoke_session(token)

        response = JSONResponse({"status": "ok"})
        response.delete_cookie(key="session_token", path="/")
        return response

    @router.get("/auth/me")
    async def me(request: Request):
        user = getattr(request.state, "user", None)
        auth_config = load_auth()
        if not user:
            # In local_trust mode, try to get from auth config
            if auth_config.auth_mode == "local_trust" and auth_config.owner:
                user = auth_config.owner
            else:
                return JSONResponse(
                    {"error": "Not authenticated"},
                    status_code=401,
                )

        return {
            "username": user.username,
            "display_name": user.display_name,
            "bio": user.bio,
            "role": user.role,
            "auth_mode": auth_config.auth_mode,
        }

    return router
