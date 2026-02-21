from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse

from core.auth.manager import load_auth, validate_session, find_user
from core.config import load_config
from core.supervisor import ProcessSupervisor
from server.localhost import _is_safe_localhost_request
from server.routes import create_router
from server.routes.setup import create_setup_router
from server.websocket import WebSocketManager
from server.stream_registry import StreamRegistry

logger = logging.getLogger("animaworks.server")

# Paths to exclude from request logging (noisy health checks, etc.)
_NOISY_PATHS = frozenset({
    "/api/system/health",
    "/api/system/status",
    "/ws",
})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration.

    Automatically binds a ``request_id`` into structlog contextvars so that
    all log records emitted during request processing carry the ID.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        structlog.contextvars.clear_contextvars()
        request_id = request.headers.get(
            "X-Request-ID", uuid.uuid4().hex[:12],
        )
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)

        response.headers["X-Request-ID"] = request_id

        # Skip noisy endpoints to reduce log volume
        if request.url.path not in _NOISY_PATHS:
            req_logger = logging.getLogger("animaworks.request")
            req_logger.info(
                "request %s %s -> %d (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )

        return response


async def _reconcile_assets_at_startup(animas_dir: Path) -> None:
    """Background task: generate missing anima assets after startup."""
    try:
        from core.asset_reconciler import reconcile_all_assets

        results = await reconcile_all_assets(animas_dir)
        if results:
            logger.info("Startup asset reconciliation: %d anima(s) processed", len(results))
    except Exception:
        logger.exception("Startup asset reconciliation failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only start anima processes when setup is complete
    if app.state.setup_complete:
        # Register anima lifecycle callbacks for reconciliation
        def _on_anima_added(name: str) -> None:
            if name not in app.state.anima_names:
                app.state.anima_names.append(name)
                # Sync full org structure (registers new anima + repairs others)
                from core.org_sync import sync_org_structure

                sync_org_structure(app.state.animas_dir)
                logger.info("Anima added via reconciliation: %s", name)

        def _on_anima_removed(name: str) -> None:
            if name in app.state.anima_names:
                app.state.anima_names.remove(name)
                logger.info("Anima removed via reconciliation: %s", name)

        app.state.supervisor.on_anima_added = _on_anima_added
        app.state.supervisor.on_anima_removed = _on_anima_removed

        await app.state.supervisor.start_all(app.state.anima_names)

        # Sync org structure from identity.md/status.json → config.json
        try:
            from core.org_sync import sync_org_structure

            sync_org_structure(app.state.animas_dir)
        except Exception:
            logger.exception("Org structure sync failed at startup")

        # Reconcile missing anima assets (fallback for failed bootstrap)
        import asyncio

        asyncio.create_task(_reconcile_assets_at_startup(app.state.animas_dir))

        shared_dir = app.state.shared_dir

        msg_log_scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")

        # ── Orphan anima detection ───────────────────────
        from core.org_sync import detect_orphan_animas

        def _detect_orphans_task() -> None:
            try:
                detect_orphan_animas(app.state.animas_dir, shared_dir)
            except Exception:
                logger.exception("Orphan detection failed")

        msg_log_scheduler.add_job(
            _detect_orphans_task,
            IntervalTrigger(minutes=10),
            id="orphan_anima_detection",
            name="System: Orphan Anima Detection",
            replace_existing=True,
        )

        # ── Asset reconciliation (periodic) ───────────────
        from core.asset_reconciler import reconcile_all_assets

        async def _reconcile_assets_periodic() -> None:
            try:
                await reconcile_all_assets(app.state.animas_dir)
            except asyncio.CancelledError:
                logger.debug("Asset reconciliation cancelled (shutdown)")
            except Exception:
                logger.exception("Periodic asset reconciliation failed")

        msg_log_scheduler.add_job(
            _reconcile_assets_periodic,
            IntervalTrigger(minutes=5),
            id="asset_reconciliation",
            name="System: Asset Reconciliation",
            replace_existing=True,
        )

        msg_log_scheduler.start()
        app.state.msg_log_scheduler = msg_log_scheduler

        # ── WebSocket heartbeat ────────────────────────────────
        await app.state.ws_manager.start_heartbeat()

        # ── Stream Registry cleanup ────────────────────────
        await app.state.stream_registry.start_cleanup_loop()

        # ── Slack Socket Mode ─────────────────────────────────
        try:
            from server.slack_socket import SlackSocketModeManager

            socket_manager = SlackSocketModeManager()
            await socket_manager.start()
            app.state.slack_socket_manager = socket_manager
        except Exception:
            logger.exception("Slack Socket Mode startup failed")
            app.state.slack_socket_manager = None

        logger.info("Server started with process isolation")
    else:
        logger.info("Server started in setup mode (setup not yet complete)")
    yield
    # Shutdown all processes
    if app.state.setup_complete:
        await app.state.ws_manager.stop_heartbeat()
        await app.state.stream_registry.stop_cleanup_loop()
        if getattr(app.state, "slack_socket_manager", None):
            await app.state.slack_socket_manager.stop()
        await app.state.supervisor.shutdown_all()
        if hasattr(app.state, "msg_log_scheduler"):
            app.state.msg_log_scheduler.shutdown(wait=False)
    logger.info("Server stopped")


def create_app(animas_dir: Path, shared_dir: Path) -> FastAPI:
    app = FastAPI(title="AnimaWorks", version="0.1.0", lifespan=lifespan)

    ws_manager = WebSocketManager()

    # Run Person→Anima rename migration before any animas_dir access
    try:
        from core.config.migrate import migrate_person_to_anima
        from core.paths import get_data_dir as _get_data_dir

        migrate_person_to_anima(_get_data_dir())
    except Exception:
        logger.exception("Person-to-Anima migration failed")

    config = load_config()

    # Create run directory for sockets and PID files
    run_dir = Path.home() / ".animaworks" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ProcessSupervisor
    from core.paths import get_data_dir
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    supervisor = ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        ws_manager=ws_manager,
    )

    # Auto-migrate old Japanese cron.md format to standard cron expressions
    try:
        from core.config.migrate import migrate_all_cron

        migrated = migrate_all_cron(animas_dir)
        if migrated:
            logger.info("Auto-migrated %d anima(s) cron.md to standard cron format", migrated)
    except Exception:
        logger.exception("Cron format auto-migration failed")

    # Discover anima names from disk (respect status.json)
    from core.supervisor.manager import ProcessSupervisor as _PS

    anima_names: list[str] = []
    if animas_dir.exists():
        for anima_dir in sorted(animas_dir.iterdir()):
            if anima_dir.is_dir() and (anima_dir / "identity.md").exists():
                if not _PS.read_anima_enabled(anima_dir):
                    logger.info("Skipping disabled anima: %s", anima_dir.name)
                    continue
                anima_names.append(anima_dir.name)
                logger.info("Discovered anima: %s", anima_dir.name)

    app.state.supervisor = supervisor
    app.state.anima_names = anima_names
    app.state.ws_manager = ws_manager
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir
    app.state.setup_complete = config.setup_complete
    app.state.stream_registry = StreamRegistry()

    # ── Global exception handler ────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return StarletteJSONResponse(
            {"error": "Internal server error"}, status_code=500,
        )

    # ── Request logging middleware ─────────────────────────
    # Added before setup_guard so request_id is available in all handlers.
    app.add_middleware(RequestLoggingMiddleware)

    # ── Static asset cache control ─────────────────────────
    # Prevent aggressive browser caching of static assets so code
    # updates are picked up without clearing browser cache.
    @app.middleware("http")
    async def static_cache_control(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html")) or path == "/" or path == "/workspace":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

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

    # ── Auth guard middleware ──────────────────────────────
    # Paths that don't require authentication
    _AUTH_WHITELIST_PREFIXES = ("/api/auth/login", "/api/setup", "/health")

    @app.middleware("http")
    async def auth_guard(request: Request, call_next):
        path = request.url.path

        # Skip during setup
        if not request.app.state.setup_complete:
            return await call_next(request)

        # Load auth config
        auth_config = load_auth()

        # Skip if local_trust mode
        if auth_config.auth_mode == "local_trust":
            return await call_next(request)

        # Localhost trust: skip auth for verified local connections
        if auth_config.trust_localhost and _is_safe_localhost_request(request):
            return await call_next(request)

        # Skip whitelisted paths
        if any(path.startswith(prefix) for prefix in _AUTH_WHITELIST_PREFIXES):
            return await call_next(request)

        # Only protect /api/ and /ws paths
        if not path.startswith("/api/") and path != "/ws":
            return await call_next(request)

        # Validate session token from cookie
        token = request.cookies.get("session_token")
        session = validate_session(token) if token else None
        if not session:
            return JSONResponse(
                {"error": "Unauthorized"},
                status_code=401,
            )

        # Set authenticated user on request state
        user = find_user(auth_config, session.username)
        if not user:
            return JSONResponse(
                {"error": "User not found"},
                status_code=401,
            )
        request.state.user = user
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
