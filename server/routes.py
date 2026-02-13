from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("animaworks.routes")


class ChatRequest(BaseModel):
    message: str
    from_person: str = "human"


class ChatResponse(BaseModel):
    response: str
    person: str


def create_router() -> APIRouter:
    router = APIRouter()

    # ── REST API ──────────────────────────────────────────

    api = APIRouter(prefix="/api")

    @api.get("/shared/users")
    async def list_shared_users(request: Request):
        """List registered user names from shared/users/."""
        users_dir = request.app.state.shared_dir / "users"
        if not users_dir.is_dir():
            return []
        return [d.name for d in sorted(users_dir.iterdir()) if d.is_dir()]

    @api.get("/persons")
    async def list_persons(request: Request):
        persons = request.app.state.persons
        return [p.status.model_dump() for p in persons.values()]

    @api.get("/persons/{name}")
    async def get_person(name: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        return {
            "status": person.status.model_dump(),
            "identity": person.memory.read_identity(),
            "injection": person.memory.read_injection(),
            "state": person.memory.read_current_state(),
            "pending": person.memory.read_pending(),
            "knowledge_files": person.memory.list_knowledge_files(),
            "episode_files": person.memory.list_episode_files(),
            "procedure_files": person.memory.list_procedure_files(),
        }

    @api.post("/persons/{name}/chat")
    async def chat(name: str, body: ChatRequest, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}

        ws = request.app.state.ws_manager
        await ws.broadcast(
            {"type": "person.status", "data": {"name": name, "status": "thinking"}}
        )

        response = await person.process_message(body.message, from_person=body.from_person)

        await ws.broadcast(
            {"type": "person.status", "data": {"name": name, "status": "idle"}}
        )
        await ws.broadcast(
            {"type": "chat.response", "data": {"name": name, "message": response}}
        )

        return ChatResponse(response=response, person=name)

    @api.post("/persons/{name}/chat/stream")
    async def chat_stream(name: str, body: ChatRequest, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}

        ws = request.app.state.ws_manager

        async def event_generator():
            full_response = ""
            try:
                await ws.broadcast(
                    {"type": "person.status", "data": {"name": name, "status": "thinking"}}
                )

                async for chunk in person.process_message_stream(
                    body.message, from_person=body.from_person
                ):
                    event_type = chunk.get("type", "unknown")

                    if event_type == "text_delta":
                        data = json.dumps({"text": chunk["text"]}, ensure_ascii=False)
                        yield f"event: text_delta\ndata: {data}\n\n"

                    elif event_type == "tool_start":
                        data = json.dumps(
                            {"tool_name": chunk["tool_name"], "tool_id": chunk["tool_id"]},
                            ensure_ascii=False,
                        )
                        yield f"event: tool_start\ndata: {data}\n\n"

                    elif event_type == "tool_end":
                        data = json.dumps(
                            {"tool_id": chunk["tool_id"], "tool_name": chunk.get("tool_name", "")},
                            ensure_ascii=False,
                        )
                        yield f"event: tool_end\ndata: {data}\n\n"

                    elif event_type == "chain_start":
                        data = json.dumps({"chain": chunk["chain"]}, ensure_ascii=False)
                        yield f"event: chain_start\ndata: {data}\n\n"

                    elif event_type == "cycle_done":
                        cycle_result = chunk.get("cycle_result", {})
                        full_response = cycle_result.get("summary", "")
                        data = json.dumps(cycle_result, ensure_ascii=False, default=str)
                        yield f"event: done\ndata: {data}\n\n"

                    elif event_type == "error":
                        data = json.dumps(
                            {"message": chunk.get("message", "Unknown error")},
                            ensure_ascii=False,
                        )
                        yield f"event: error\ndata: {data}\n\n"

                if full_response:
                    await ws.broadcast(
                        {"type": "chat.response", "data": {"name": name, "message": full_response}}
                    )

            except Exception:
                logger.exception("SSE stream error for person=%s", name)
                data = json.dumps({"message": "Internal server error"}, ensure_ascii=False)
                yield f"event: error\ndata: {data}\n\n"

            finally:
                await ws.broadcast(
                    {"type": "person.status", "data": {"name": name, "status": "idle"}}
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @api.get("/persons/{name}/episodes")
    async def list_episodes(name: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        return {"files": person.memory.list_episode_files()}

    @api.get("/persons/{name}/episodes/{date}")
    async def get_episode(name: str, date: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        path = person.memory.episodes_dir / f"{date}.md"
        if not path.exists():
            return {"error": "Episode not found"}
        return {"date": date, "content": path.read_text(encoding="utf-8")}

    @api.get("/persons/{name}/knowledge")
    async def list_knowledge(name: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        return {"files": person.memory.list_knowledge_files()}

    @api.get("/persons/{name}/knowledge/{topic}")
    async def get_knowledge(name: str, topic: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        path = person.memory.knowledge_dir / f"{topic}.md"
        if not path.exists():
            return {"error": "Knowledge not found"}
        return {"topic": topic, "content": path.read_text(encoding="utf-8")}

    @api.get("/persons/{name}/procedures")
    async def list_procedures(name: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        return {"files": person.memory.list_procedure_files()}

    @api.get("/persons/{name}/procedures/{proc}")
    async def get_procedure(name: str, proc: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        path = person.memory.procedures_dir / f"{proc}.md"
        if not path.exists():
            return {"error": "Procedure not found"}
        return {"name": proc, "content": path.read_text(encoding="utf-8")}

    @api.get("/persons/{name}/conversation")
    async def get_conversation(name: str, request: Request):
        """View current conversation state."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory

        conv = ConversationMemory(person.person_dir, person.model_config)
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

    @api.get("/persons/{name}/conversation/full")
    async def get_conversation_full(
        name: str, request: Request, limit: int = 50, offset: int = 0
    ):
        """View full conversation history (not truncated)."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory

        conv = ConversationMemory(person.person_dir, person.model_config)
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

    @api.get("/persons/{name}/sessions")
    async def list_sessions(name: str, request: Request):
        """List all available sessions: active conversation, archives, episodes."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory
        from core.shortterm_memory import ShortTermMemory

        # Active conversation
        conv = ConversationMemory(person.person_dir, person.model_config)
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

        # Archived sessions
        stm = ShortTermMemory(person.person_dir)
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
                                "context_usage_ratio", 0
                            ),
                            "original_prompt_preview": data.get(
                                "original_prompt", ""
                            )[:200],
                            "has_markdown": (
                                archive_dir / f"{ts_str}.md"
                            ).exists(),
                        }
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

        # Episodes
        episodes = []
        ep_dir = person.memory.episodes_dir
        if ep_dir.exists():
            for ep_file in sorted(ep_dir.glob("*.md"), reverse=True):
                content = ep_file.read_text(encoding="utf-8")
                episodes.append(
                    {"date": ep_file.stem, "preview": content[:200]}
                )

        # Transcripts (permanent message logs)
        transcripts = [
            {"date": date, "message_count": len(conv.load_transcript(date))}
            for date in conv.list_transcript_dates()
        ]

        return {
            "person": name,
            "active_conversation": active_conv,
            "archived_sessions": archived,
            "episodes": episodes,
            "transcripts": transcripts,
        }

    @api.get("/persons/{name}/sessions/{session_id}")
    async def get_session_detail(
        name: str, session_id: str, request: Request
    ):
        """Get archived session detail."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.shortterm_memory import ShortTermMemory

        stm = ShortTermMemory(person.person_dir)
        archive_dir = stm._archive_dir
        json_path = archive_dir / f"{session_id}.json"
        md_path = archive_dir / f"{session_id}.md"

        if not json_path.exists():
            return {"error": "Session not found"}

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            return {"error": "Session data corrupted"}

        markdown = ""
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")

        return {
            "person": name,
            "session_id": session_id,
            "data": data,
            "markdown": markdown,
        }

    @api.get("/persons/{name}/transcripts/{date}")
    async def get_transcript(name: str, date: str, request: Request):
        """Get full conversation transcript for a specific date."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory

        conv = ConversationMemory(person.person_dir, person.model_config)
        messages = conv.load_transcript(date)
        return {
            "person": name,
            "date": date,
            "has_summary": False,
            "compressed_summary": "",
            "compressed_turn_count": 0,
            "turns": messages,
        }

    @api.delete("/persons/{name}/conversation")
    async def clear_conversation(name: str, request: Request):
        """Clear conversation history for a fresh start."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory

        conv = ConversationMemory(person.person_dir, person.model_config)
        conv.clear()
        return {"status": "cleared", "person": name}

    @api.post("/persons/{name}/conversation/compress")
    async def compress_conversation(name: str, request: Request):
        """Manually trigger conversation compression."""
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        from core.conversation_memory import ConversationMemory

        conv = ConversationMemory(person.person_dir, person.model_config)
        compressed = await conv.compress_if_needed()
        state = conv.load()
        return {
            "compressed": compressed,
            "person": name,
            "total_turn_count": state.total_turn_count,
            "total_token_estimate": state.total_token_estimate,
        }

    @api.post("/persons/{name}/trigger")
    async def trigger_heartbeat(name: str, request: Request):
        person = request.app.state.persons.get(name)
        if not person:
            return {"error": "Person not found"}
        result = await person.run_heartbeat()
        return result.model_dump()

    @api.get("/system/status")
    async def system_status(request: Request):
        persons = request.app.state.persons
        scheduler = request.app.state.lifecycle.scheduler
        return {
            "persons": len(persons),
            "scheduler_running": scheduler.running,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "next_run": str(j.next_run_time),
                }
                for j in scheduler.get_jobs()
            ],
        }

    @api.post("/system/reload")
    async def reload_persons(request: Request):
        """Full sync: add new persons, refresh existing, remove deleted."""
        from core.person import DigitalPerson

        persons = request.app.state.persons
        lifecycle = request.app.state.lifecycle
        persons_dir = request.app.state.persons_dir
        shared_dir = request.app.state.shared_dir

        added: list[str] = []
        refreshed: list[str] = []
        removed: list[str] = []

        # Discover current persons on disk
        on_disk: set[str] = set()
        if persons_dir.exists():
            for person_dir in sorted(persons_dir.iterdir()):
                if not person_dir.is_dir():
                    continue
                if not (person_dir / "identity.md").exists():
                    continue
                name = person_dir.name
                on_disk.add(name)

                if name not in persons:
                    # New person
                    person = DigitalPerson(person_dir, shared_dir)
                    persons[name] = person
                    lifecycle.register_person(person)
                    added.append(name)
                    logger.info("Hot-loaded person: %s", name)
                else:
                    # Existing person — re-initialize to pick up file changes
                    lifecycle.unregister_person(name)
                    person = DigitalPerson(person_dir, shared_dir)
                    persons[name] = person
                    lifecycle.register_person(person)
                    refreshed.append(name)
                    logger.info("Refreshed person: %s", name)

        # Remove persons whose directories no longer exist
        for name in list(persons.keys()):
            if name not in on_disk:
                lifecycle.unregister_person(name)
                del persons[name]
                removed.append(name)
                logger.info("Unloaded person: %s", name)

        return {
            "added": added,
            "refreshed": refreshed,
            "removed": removed,
            "total": len(persons),
        }

    # ── WebSocket ─────────────────────────────────────────

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        ws_manager = websocket.app.state.ws_manager
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    router.include_router(api)
    return router
