from __future__ import annotations

import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("animaworks.routes")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    person: str


def create_router() -> APIRouter:
    router = APIRouter()

    # ── REST API ──────────────────────────────────────────

    api = APIRouter(prefix="/api")

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

        response = await person.process_message(body.message, from_person="human")

        await ws.broadcast(
            {"type": "person.status", "data": {"name": name, "status": "idle"}}
        )
        await ws.broadcast(
            {"type": "chat.response", "data": {"name": name, "message": response}}
        )

        return ChatResponse(response=response, person=name)

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
