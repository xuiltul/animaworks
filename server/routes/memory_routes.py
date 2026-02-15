from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from fastapi import APIRouter, HTTPException, Request

from core.memory.conversation import ConversationMemory
from core.memory.manager import MemoryManager

logger = logging.getLogger("animaworks.routes.memory")


def create_memory_router() -> APIRouter:
    router = APIRouter()

    # ── Episodes ──────────────────────────────────────────

    @router.get("/persons/{name}/episodes")
    async def list_episodes(name: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        return {"files": memory.list_episode_files()}

    @router.get("/persons/{name}/episodes/{date}")
    async def get_episode(name: str, date: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        path = memory.episodes_dir / f"{date}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Episode not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
        return {"date": date, "content": content}

    # ── Knowledge ─────────────────────────────────────────

    @router.get("/persons/{name}/knowledge")
    async def list_knowledge(name: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        return {"files": memory.list_knowledge_files()}

    @router.get("/persons/{name}/knowledge/{topic}")
    async def get_knowledge(name: str, topic: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        path = memory.knowledge_dir / f"{topic}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Knowledge not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
        return {"topic": topic, "content": content}

    # ── Procedures ────────────────────────────────────────

    @router.get("/persons/{name}/procedures")
    async def list_procedures(name: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        return {"files": memory.list_procedure_files()}

    @router.get("/persons/{name}/procedures/{proc}")
    async def get_procedure(name: str, proc: str, request: Request):
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)
        path = memory.procedures_dir / f"{proc}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Procedure not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
        return {"name": proc, "content": content}

    # ── Conversation ──────────────────────────────────────

    @router.get("/persons/{name}/conversation")
    async def get_conversation(name: str, request: Request):
        """View current conversation state."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        # Read model config from person directory
        from core.config.models import load_model_config
        model_config = load_model_config(person_dir)

        conv = ConversationMemory(person_dir, model_config)
        state = conv.load()
        return {
            "person": name,
            "total_turn_count": state.total_turn_count,
            "raw_turns": len(state.turns),
            "compressed_turn_count": state.compressed_turn_count,
            "has_summary": bool(state.compressed_summary),
            "summary_preview": (
                state.compressed_summary[:300]
                if state.compressed_summary
                else ""
            ),
            "total_token_estimate": state.total_token_estimate,
            "turns": [
                {
                    "role": t.role,
                    "content": (
                        t.content[:200] + "..."
                        if len(t.content) > 200
                        else t.content
                    ),
                    "timestamp": t.timestamp,
                    "token_estimate": t.token_estimate,
                }
                for t in state.turns
            ],
        }

    @router.get("/persons/{name}/conversation/full")
    async def get_conversation_full(
        name: str, limit: int = 50, offset: int = 0,
        request: Request = None,
    ):
        """View full conversation history (not truncated)."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        from core.config.models import load_model_config
        model_config = load_model_config(person_dir)

        conv = ConversationMemory(person_dir, model_config)
        state = conv.load()

        total = len(state.turns)
        end = max(0, total - offset)
        start = max(0, end - limit)
        paginated = state.turns[start:end]

        return {
            "person": name,
            "total_turn_count": state.total_turn_count,
            "raw_turns": total,
            "compressed_turn_count": state.compressed_turn_count,
            "has_summary": bool(state.compressed_summary),
            "compressed_summary": state.compressed_summary or "",
            "total_token_estimate": state.total_token_estimate,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp,
                    "token_estimate": t.token_estimate,
                }
                for t in paginated
            ],
        }

    @router.delete("/persons/{name}/conversation")
    async def clear_conversation(name: str, request: Request):
        """Clear conversation history for a fresh start."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        from core.config.models import load_model_config
        model_config = load_model_config(person_dir)

        conv = ConversationMemory(person_dir, model_config)
        conv.clear()
        return {"status": "cleared", "person": name}

    @router.post("/persons/{name}/conversation/compress")
    async def compress_conversation(name: str, request: Request):
        """Manually trigger conversation compression."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        from core.config.models import load_model_config
        model_config = load_model_config(person_dir)

        conv = ConversationMemory(person_dir, model_config)
        compressed = await conv.compress_if_needed()
        state = conv.load()
        return {
            "compressed": compressed,
            "person": name,
            "total_turn_count": state.total_turn_count,
            "total_token_estimate": state.total_token_estimate,
        }

    # ── Stats ─────────────────────────────────────────────

    @router.get("/persons/{name}/memory/stats")
    async def memory_stats(name: str, request: Request):
        """Return memory storage statistics for a person."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        memory = MemoryManager(person_dir)

        def dir_stats(directory):
            if not directory.exists():
                return {"count": 0, "total_bytes": 0}
            files = list(directory.glob("*.md"))
            return {
                "count": len(files),
                "total_bytes": sum(f.stat().st_size for f in files),
            }

        return {
            "person": name,
            "episodes": dir_stats(memory.episodes_dir),
            "knowledge": dir_stats(memory.knowledge_dir),
            "procedures": dir_stats(memory.procedures_dir),
        }

    return router
