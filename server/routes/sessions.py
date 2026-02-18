from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from core.memory.conversation import ConversationMemory
from core.memory.manager import MemoryManager
from core.memory.shortterm import ShortTermMemory

logger = logging.getLogger("animaworks.routes.sessions")


def create_sessions_router() -> APIRouter:
    router = APIRouter()

    @router.get("/animas/{name}/sessions")
    async def list_sessions(name: str, request: Request):
        """List all available sessions: active conversation, archives, episodes."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        # Active conversation
        conv = ConversationMemory(anima_dir, model_config)
        conv_state = conv.load()
        active_conv = None
        if conv_state.turns or conv_state.compressed_summary:
            active_conv = {
                "exists": True,
                "turn_count": len(conv_state.turns),
                "total_turn_count": conv_state.total_turn_count,
                "last_timestamp": (
                    conv_state.turns[-1].timestamp if conv_state.turns else ""
                ),
                "first_timestamp": (
                    conv_state.turns[0].timestamp if conv_state.turns else ""
                ),
                "has_summary": bool(conv_state.compressed_summary),
            }

        # Archived sessions — metadata only (no full content read)
        stm = ShortTermMemory(anima_dir)
        archived = []
        archive_dir = stm._archive_dir
        if archive_dir.exists():
            for json_file in sorted(archive_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    ts_str = json_file.stem
                    archived.append(
                        {
                            "id": ts_str,
                            "timestamp": data.get("timestamp", ts_str),
                            "trigger": data.get("trigger", ""),
                            "turn_count": data.get("turn_count", 0),
                            "context_usage_ratio": data.get(
                                "context_usage_ratio", 0,
                            ),
                            "original_prompt_preview": data.get(
                                "original_prompt", "",
                            )[:200],
                            "has_markdown": (
                                archive_dir / f"{ts_str}.md"
                            ).exists(),
                        }
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

        # Episodes — partial read (first 200 chars only, no full file load)
        memory = MemoryManager(anima_dir)
        episodes = []
        for stem in memory.list_episode_files():
            ep_path = memory.episodes_dir / f"{stem}.md"
            preview = ""
            if ep_path.exists():
                with ep_path.open(encoding="utf-8") as f:
                    preview = f.read(200)
            episodes.append({"date": stem, "preview": preview})

        # Transcripts — count lines without loading full content
        transcripts = []
        transcript_dir = anima_dir / "transcripts"
        if transcript_dir.exists():
            for tf in sorted(transcript_dir.glob("*.jsonl"), reverse=True):
                line_count = sum(
                    1 for line in tf.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                transcripts.append(
                    {"date": tf.stem, "message_count": line_count}
                )

        return {
            "anima": name,
            "active_conversation": active_conv,
            "archived_sessions": archived,
            "episodes": episodes,
            "transcripts": transcripts,
        }

    @router.get("/animas/{name}/sessions/{session_id}")
    async def get_session_detail(
        name: str, session_id: str, request: Request,
    ):
        """Get archived session detail."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        stm = ShortTermMemory(anima_dir)
        archive_dir = stm._archive_dir
        json_path = archive_dir / f"{session_id}.json"
        md_path = archive_dir / f"{session_id}.md"

        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=500, detail="Session data corrupted")

        markdown = ""
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")

        return {
            "anima": name,
            "session_id": session_id,
            "data": data,
            "markdown": markdown,
        }

    @router.get("/animas/{name}/transcripts/{date}")
    async def get_transcript(name: str, date: str, request: Request):
        """Get full conversation transcript for a specific date."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config
        model_config = load_model_config(anima_dir)

        conv = ConversationMemory(anima_dir, model_config)
        messages = conv.load_transcript(date)
        return {
            "anima": name,
            "date": date,
            "has_summary": False,
            "compressed_summary": "",
            "compressed_turn_count": 0,
            "turns": messages,
        }

    return router
