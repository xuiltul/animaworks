from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.lifecycle import LifecycleManager
from core.person import DigitalPerson
from server.routes import create_router
from server.websocket import WebSocketManager

logger = logging.getLogger("animaworks.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lifecycle.start()
    logger.info("Server started")
    yield
    app.state.lifecycle.shutdown()
    logger.info("Server stopped")


def create_app(persons_dir: Path, shared_dir: Path) -> FastAPI:
    app = FastAPI(title="AnimaWorks", version="0.1.0", lifespan=lifespan)

    ws_manager = WebSocketManager()
    lifecycle = LifecycleManager()
    lifecycle.set_broadcast(ws_manager.broadcast)

    persons: dict[str, DigitalPerson] = {}
    if persons_dir.exists():
        for person_dir in sorted(persons_dir.iterdir()):
            if person_dir.is_dir() and (person_dir / "identity.md").exists():
                person = DigitalPerson(person_dir, shared_dir)
                persons[person.name] = person
                lifecycle.register_person(person)
                logger.info("Loaded person: %s", person.name)

    app.state.persons = persons
    app.state.lifecycle = lifecycle
    app.state.ws_manager = ws_manager
    app.state.persons_dir = persons_dir
    app.state.shared_dir = shared_dir

    app.include_router(create_router())

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
