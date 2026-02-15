from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse

from core.config import load_config
from core.supervisor import ProcessSupervisor
from server.routes import create_router
from server.routes.setup import create_setup_router
from server.websocket import WebSocketManager

logger = logging.getLogger("animaworks.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only start person processes when setup is complete
    if app.state.setup_complete:
        # Register person lifecycle callbacks for reconciliation
        def _on_person_added(name: str) -> None:
            if name not in app.state.person_names:
                app.state.person_names.append(name)
                logger.info("Person added via reconciliation: %s", name)

        def _on_person_removed(name: str) -> None:
            if name in app.state.person_names:
                app.state.person_names.remove(name)
                logger.info("Person removed via reconciliation: %s", name)

        app.state.supervisor.on_person_added = _on_person_added
        app.state.supervisor.on_person_removed = _on_person_removed

        await app.state.supervisor.start_all(app.state.person_names)
        logger.info("Server started with process isolation")
    else:
        logger.info("Server started in setup mode (setup not yet complete)")
    yield
    # Shutdown all processes
    if app.state.setup_complete:
        await app.state.supervisor.shutdown_all()
    logger.info("Server stopped")


def create_app(persons_dir: Path, shared_dir: Path) -> FastAPI:
    app = FastAPI(title="AnimaWorks", version="0.1.0", lifespan=lifespan)

    ws_manager = WebSocketManager()
    config = load_config()

    # Create run directory for sockets and PID files
    run_dir = Path.home() / ".animaworks" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ProcessSupervisor
    from core.paths import get_data_dir
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    supervisor = ProcessSupervisor(
        persons_dir=persons_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        ws_manager=ws_manager,
    )

    # Discover person names from disk (respect status.json)
    from core.supervisor.manager import ProcessSupervisor as _PS

    person_names: list[str] = []
    if persons_dir.exists():
        for person_dir in sorted(persons_dir.iterdir()):
            if person_dir.is_dir() and (person_dir / "identity.md").exists():
                if not _PS.read_person_enabled(person_dir):
                    logger.info("Skipping disabled person: %s", person_dir.name)
                    continue
                person_names.append(person_dir.name)
                logger.info("Discovered person: %s", person_dir.name)

    app.state.supervisor = supervisor
    app.state.person_names = person_names
    app.state.ws_manager = ws_manager
    app.state.persons_dir = persons_dir
    app.state.shared_dir = shared_dir
    app.state.setup_complete = config.setup_complete

    # ── Global exception handler ────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return StarletteJSONResponse(
            {"error": "Internal server error"}, status_code=500,
        )

    # ── Setup guard middleware ──────────────────────────────
    @app.middleware("http")
    async def setup_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        setup_complete = request.app.state.setup_complete

        if not setup_complete:
            # During setup: only /api/setup/* and setup static files are accessible
            if path.startswith("/api/setup") or path.startswith("/setup"):
                response = await call_next(request)
                # Prevent browser caching of setup static files so code
                # updates are picked up immediately without a hard refresh.
                if path.startswith("/setup") and not path.startswith("/api/setup"):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                return response
            # Root redirects to setup wizard
            if path == "/":
                return RedirectResponse("/setup/")
            # Block all other API/dashboard routes
            if path.startswith("/api/"):
                return JSONResponse(
                    {"error": "Setup not yet complete"},
                    status_code=503,
                )
            return RedirectResponse("/setup/")
        else:
            # After setup: block setup API
            if path.startswith("/api/setup"):
                return JSONResponse(
                    {"error": "Setup already completed"},
                    status_code=403,
                )
            # Redirect /setup/* to dashboard
            if path.startswith("/setup"):
                return RedirectResponse("/")
            return await call_next(request)

    # ── Route registration ─────────────────────────────────
    # Always mount both routers; the middleware handles access control.
    app.include_router(create_router())
    app.include_router(create_setup_router())

    # ── Static files ───────────────────────────────────────
    setup_static_dir = Path(__file__).parent / "static" / "setup"
    if setup_static_dir.exists():
        app.mount(
            "/setup",
            StaticFiles(directory=str(setup_static_dir), html=True),
            name="setup_static",
        )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
