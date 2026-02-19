from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""User management API routes."""

import logging
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.auth.manager import (
    create_session,
    find_user,
    get_all_users,
    hash_password,
    load_auth,
    revoke_all_sessions,
    save_auth,
    verify_password,
)
from core.auth.models import AuthUser

logger = logging.getLogger("animaworks.routes.users")

_USERNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,30}$")


class AddUserRequest(BaseModel):
    username: str
    display_name: str = ""
    password: str
    bio: str = ""


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def create_users_router() -> APIRouter:
    router = APIRouter(tags=["users"])

    @router.get("/users")
    async def list_users(request: Request):
        """List all users (without password hashes)."""
        auth_config = load_auth()
        users = get_all_users(auth_config)
        return [
            {
                "username": u.username,
                "display_name": u.display_name,
                "bio": u.bio,
                "role": u.role,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]

    @router.post("/users")
    async def add_user(body: AddUserRequest, request: Request):
        """Add a new user (owner only)."""
        # Check caller is owner
        caller = getattr(request.state, "user", None)
        if not caller or caller.role != "owner":
            return JSONResponse(
                {"error": "Only the owner can add users"},
                status_code=403,
            )

        # Validate username format
        if not _USERNAME_RE.match(body.username):
            return JSONResponse(
                {"error": "Invalid username format. Use alphanumeric and underscore, 2-31 chars, starting with a letter."},
                status_code=400,
            )

        # Validate password length
        if len(body.password) < 4:
            return JSONResponse(
                {"error": "Password must be at least 4 characters"},
                status_code=400,
            )
        if len(body.password) > 128:
            return JSONResponse(
                {"error": "Password must be at most 128 characters"},
                status_code=400,
            )

        auth_config = load_auth()

        # Check username doesn't already exist
        if find_user(auth_config, body.username):
            return JSONResponse(
                {"error": f"User '{body.username}' already exists"},
                status_code=409,
            )

        # Create user
        new_user = AuthUser(
            username=body.username,
            display_name=body.display_name or body.username,
            bio=body.bio,
            password_hash=hash_password(body.password),
            role="user",
        )
        auth_config.users.append(new_user)

        # Switch to multi_user mode if currently in password mode
        if auth_config.auth_mode == "password":
            auth_config.auth_mode = "multi_user"

        save_auth(auth_config)

        # Create user profile directory
        from core.paths import get_shared_dir

        user_dir = get_shared_dir() / "users" / body.username
        user_dir.mkdir(parents=True, exist_ok=True)
        profile_path = user_dir / "index.md"
        if not profile_path.exists():
            lines = [f"# {body.display_name or body.username}\n"]
            if body.bio:
                lines.append(f"\n{body.bio}\n")
            profile_path.write_text("".join(lines), encoding="utf-8")

        logger.info("Added user '%s'", body.username)
        return {
            "status": "ok",
            "username": new_user.username,
            "display_name": new_user.display_name,
            "role": new_user.role,
        }

    @router.delete("/users/{username}")
    async def delete_user(username: str, request: Request):
        """Delete a user (owner only, cannot delete self)."""
        caller = getattr(request.state, "user", None)
        if not caller or caller.role != "owner":
            return JSONResponse(
                {"error": "Only the owner can delete users"},
                status_code=403,
            )

        if caller.username == username:
            return JSONResponse(
                {"error": "Cannot delete yourself"},
                status_code=400,
            )

        auth_config = load_auth()
        user = find_user(auth_config, username)
        if not user:
            return JSONResponse(
                {"error": f"User '{username}' not found"},
                status_code=404,
            )

        # Cannot delete owner
        if auth_config.owner and auth_config.owner.username == username:
            return JSONResponse(
                {"error": "Cannot delete the owner account"},
                status_code=400,
            )

        # Remove user from list
        auth_config.users = [u for u in auth_config.users if u.username != username]

        # Revoke all sessions for deleted user
        auth_config.sessions = {
            t: s for t, s in auth_config.sessions.items() if s.username != username
        }

        save_auth(auth_config)
        logger.info("Deleted user '%s'", username)
        return {"status": "ok", "deleted": username}

    @router.put("/users/me/password")
    async def change_password(body: ChangePasswordRequest, request: Request):
        """Change (or initially set) the current user's password."""
        caller = getattr(request.state, "user", None)
        auth_config = load_auth()

        # In local_trust mode, middleware skips auth — use owner directly
        if not caller:
            if auth_config.auth_mode == "local_trust" and auth_config.owner:
                caller = auth_config.owner
            else:
                return JSONResponse(
                    {"error": "Not authenticated"},
                    status_code=401,
                )

        if caller.password_hash:
            # Existing password — require current password verification
            if not verify_password(body.current_password, caller.password_hash):
                return JSONResponse(
                    {"error": "Current password is incorrect"},
                    status_code=401,
                )

        if len(body.new_password) < 4:
            return JSONResponse(
                {"error": "New password must be at least 4 characters"},
                status_code=400,
            )
        if len(body.new_password) > 128:
            return JSONResponse(
                {"error": "New password must be at most 128 characters"},
                status_code=400,
            )

        # Update password
        user = find_user(auth_config, caller.username)
        if not user:
            return JSONResponse(
                {"error": "User not found"},
                status_code=404,
            )

        initial_setup = not user.password_hash
        user.password_hash = hash_password(body.new_password)

        # Upgrade auth mode on initial password setup
        if initial_setup and auth_config.auth_mode == "local_trust":
            auth_config.auth_mode = "password"

        save_auth(auth_config)

        # Create session so user stays logged in after initial setup
        if initial_setup:
            token = create_session(auth_config, user.username)
            save_auth(auth_config)
            response = JSONResponse({"status": "ok"})
            response.set_cookie(
                key="session_token",
                value=token,
                httponly=True,
                samesite="strict",
                path="/",
            )
            logger.info("User '%s' set initial password", caller.username)
            return response

        logger.info("User '%s' changed their password", caller.username)
        return {"status": "ok"}

    return router
