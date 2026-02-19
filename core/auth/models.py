from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Authentication data models for AnimaWorks."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AuthUser(BaseModel):
    """A human user account."""

    username: str
    display_name: str = ""
    bio: str = ""
    password_hash: str | None = None
    role: Literal["owner", "admin", "user"] = "user"
    created_at: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    """An active authentication session."""

    username: str
    created_at: datetime = Field(default_factory=datetime.now)


class AuthConfig(BaseModel):
    """Root authentication configuration stored in auth.json."""

    auth_mode: Literal["local_trust", "password", "multi_user"] = "local_trust"
    trust_localhost: bool = True
    owner: AuthUser | None = None
    users: list[AuthUser] = []
    token_version: int = 1
    sessions: dict[str, Session] = {}
    secret_key: str = ""
