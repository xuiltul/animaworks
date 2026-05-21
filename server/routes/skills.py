from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Skill catalog and explicit activation API routes."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.skills.activation import (
    get_active_skill_state,
    list_skill_catalog,
    set_active_skill_refs,
    validate_thread_id,
)

logger = logging.getLogger("animaworks.routes.skills")


class ActiveSkillsRequest(BaseModel):
    thread_id: str = "default"
    refs: list[str] = Field(default_factory=list)
    confirm_risk: bool = False


class TrustSkillRequest(BaseModel):
    ref: str
    trusted_by: str = "user"
    trust_reason: str = "human_instruction"


def create_skills_router() -> APIRouter:
    router = APIRouter()

    @router.get("/animas/{name}/skills")
    async def list_skills(name: str, request: Request, thread_id: str = "default"):
        """List visible skills for an anima and mark thread-local active entries."""
        anima_dir = _resolve_anima_dir(request, name)
        thread_id = _validate_thread_or_400(thread_id)
        skills = await asyncio.to_thread(list_skill_catalog, anima_dir, thread_id=thread_id)
        return {
            "anima": name,
            "thread_id": thread_id,
            "skills": skills,
        }

    @router.get("/animas/{name}/skills/active")
    async def get_active_skills(name: str, request: Request, thread_id: str = "default"):
        """Return the active skills currently configured for a chat thread."""
        anima_dir = _resolve_anima_dir(request, name)
        thread_id = _validate_thread_or_400(thread_id)
        result = await asyncio.to_thread(get_active_skill_state, anima_dir, thread_id=thread_id)
        return {
            "anima": name,
            "thread_id": thread_id,
            **result.to_dict(),
        }

    @router.post("/animas/{name}/skills/trust")
    async def trust_skill(name: str, body: TrustSkillRequest, request: Request):
        """Promote a safe skill to trusted operating guidance."""
        from core.skills.trust import promote_skill_to_trusted

        anima_dir = _resolve_anima_dir(request, name)
        try:
            result = await asyncio.to_thread(
                promote_skill_to_trusted,
                anima_dir,
                body.ref,
                trusted_by=body.trusted_by,
                trust_reason=body.trust_reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "anima": name,
            "status": "trusted",
            **result.to_dict(),
        }

    @router.put("/animas/{name}/skills/active")
    async def update_active_skills(name: str, body: ActiveSkillsRequest, request: Request):
        """Replace active skills for a chat thread."""
        anima_dir = _resolve_anima_dir(request, name)
        thread_id = _validate_thread_or_400(body.thread_id)
        result = await asyncio.to_thread(
            set_active_skill_refs,
            anima_dir,
            body.refs,
            thread_id=thread_id,
            confirm_risk=body.confirm_risk,
        )
        return {
            "anima": name,
            "thread_id": thread_id,
            **result.to_dict(),
        }

    return router


def _resolve_anima_dir(request: Request, name: str):
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid anima name")
    animas_dir = request.app.state.animas_dir
    anima_dir = animas_dir / name
    if not anima_dir.exists():
        raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
    return anima_dir


def _validate_thread_or_400(thread_id: str) -> str:
    try:
        return validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


__all__ = ["ActiveSkillsRequest", "TrustSkillRequest", "create_skills_router"]
