from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.tooling.prompt_db import get_prompt_store

logger = logging.getLogger("animaworks.routes.tool_prompts")


class DescriptionUpdate(BaseModel):
    description: str


class GuideUpdate(BaseModel):
    content: str


class SectionUpdate(BaseModel):
    content: str
    condition: str | None = None


_MAX_SECTION_CONTENT_LENGTH = 51200  # 50 KB


class SchemaPreviewRequest(BaseModel):
    mode: str = "anthropic"  # "anthropic", "litellm", "text"


class SystemPromptPreviewRequest(BaseModel):
    anima_name: str


def create_tool_prompts_router() -> APIRouter:
    router = APIRouter()

    # ── Descriptions CRUD ──────────────────────────────────

    @router.get("/tool-prompts/descriptions")
    async def list_descriptions():
        """List all tool descriptions from the prompt DB."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        return {"descriptions": store.list_descriptions()}

    @router.get("/tool-prompts/descriptions/{name}")
    async def get_description(name: str):
        """Get a single tool description by name."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        desc = store.get_description(name)
        if desc is None:
            raise HTTPException(404, f"Tool '{name}' not found")
        return {"name": name, "description": desc}

    @router.put("/tool-prompts/descriptions/{name}")
    async def update_description(name: str, body: DescriptionUpdate):
        """Update a tool description."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        if not body.description.strip():
            raise HTTPException(400, "Description cannot be empty")
        result = store.set_description(name, body.description.strip())
        return result

    # ── Guides CRUD ────────────────────────────────────────

    @router.get("/tool-prompts/guides")
    async def list_guides():
        """List all tool guides from the prompt DB."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        return {"guides": store.list_guides()}

    @router.get("/tool-prompts/guides/{key}")
    async def get_guide(key: str):
        """Get a single tool guide by key."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        content = store.get_guide(key)
        if content is None:
            raise HTTPException(404, f"Guide '{key}' not found")
        return {"key": key, "content": content}

    @router.put("/tool-prompts/guides/{key}")
    async def update_guide(key: str, body: GuideUpdate):
        """Update a tool guide."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        if not body.content.strip():
            raise HTTPException(400, "Content cannot be empty")
        result = store.set_guide(key, body.content.strip())
        return result

    # ── Sections CRUD ─────────────────────────────────────

    @router.get("/tool-prompts/sections")
    async def list_sections():
        """List all system sections from the prompt DB."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        return {"sections": store.list_sections()}

    @router.get("/tool-prompts/sections/{key}")
    async def get_section(key: str):
        """Get a single system section by key."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        result = store.get_section_with_condition(key)
        if result is None:
            raise HTTPException(404, f"Section '{key}' not found")
        content, condition = result
        return {"key": key, "content": content, "condition": condition}

    @router.put("/tool-prompts/sections/{key}")
    async def update_section(key: str, body: SectionUpdate):
        """Update a system section."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")
        if not body.content.strip():
            raise HTTPException(400, "Content cannot be empty")
        if len(body.content) > _MAX_SECTION_CONTENT_LENGTH:
            raise HTTPException(400, f"Content too large (max {_MAX_SECTION_CONTENT_LENGTH} bytes)")
        result = store.set_section(key, body.content.strip(), body.condition)
        return result

    # ── Preview endpoints ──────────────────────────────────

    @router.post("/tool-prompts/preview/schema")
    async def preview_schema(body: SchemaPreviewRequest):
        """Preview tool schemas with DB descriptions applied."""
        store = get_prompt_store()
        if not store:
            raise HTTPException(500, "Tool prompt DB not available")

        from core.tooling.schemas import (
            build_tool_list,
            to_anthropic_format,
            to_litellm_format,
            to_text_format,
        )

        tools = build_tool_list(
            include_file_tools=True,
            include_search_tools=True,
            include_discovery_tools=True,
            include_notification_tools=True,
            include_task_tools=True,
        )

        if body.mode == "anthropic":
            return {"mode": "anthropic", "tools": to_anthropic_format(tools)}
        elif body.mode == "litellm":
            return {"mode": "litellm", "tools": to_litellm_format(tools)}
        elif body.mode == "text":
            return {"mode": "text", "text": to_text_format(tools)}
        else:
            raise HTTPException(400, f"Unknown mode: {body.mode}")

    @router.post("/tool-prompts/preview/system-prompt")
    async def preview_system_prompt(body: SystemPromptPreviewRequest):
        """Build and return the full system prompt for a given anima."""
        from core.paths import get_data_dir

        data_dir = get_data_dir()
        anima_dir = data_dir / "animas" / body.anima_name

        if not anima_dir.is_dir():
            raise HTTPException(404, f"Anima '{body.anima_name}' not found")

        try:
            from core.memory import MemoryManager
            from core.prompt.builder import build_system_prompt

            memory = MemoryManager(anima_dir)

            # Determine execution mode from config
            execution_mode = "a1"  # default
            try:
                from core.config.models import load_config, resolve_execution_mode

                config = load_config()
                anima_config = config.animas.get(body.anima_name)
                if anima_config and anima_config.model:
                    execution_mode = resolve_execution_mode(
                        config, anima_config.model, anima_config.execution_mode,
                    )
            except Exception:
                pass

            # Determine tool registry
            tool_registry: list[str] = []
            personal_tools: dict[str, str] = {}
            try:
                from core.tools import (
                    TOOL_MODULES,
                    discover_common_tools,
                    discover_personal_tools,
                )

                tool_registry = sorted(TOOL_MODULES.keys())
                common = discover_common_tools()
                personal = discover_personal_tools(anima_dir)
                personal_tools = {**common, **personal}
            except Exception:
                pass

            result = build_system_prompt(
                memory,
                tool_registry=tool_registry,
                personal_tools=personal_tools,
                execution_mode=execution_mode,
            )

            prompt_text = str(result)
            token_estimate = len(prompt_text) // 4

            return {
                "anima_name": body.anima_name,
                "execution_mode": execution_mode,
                "system_prompt": prompt_text,
                "token_estimate": token_estimate,
                "char_count": len(prompt_text),
            }
        except Exception as e:
            logger.exception(
                "Failed to build system prompt preview for %s", body.anima_name,
            )
            raise HTTPException(500, f"Failed to build system prompt: {e}")

    return router
