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
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core.auth.manager import find_user, load_auth, validate_session
from core.config import load_config
from core.supervisor import ProcessSupervisor
from server.localhost import _is_safe_localhost_request
from server.routes import create_router
from server.routes.setup import create_setup_router
from server.stream_registry import StreamRegistry
from server.websocket import WebSocketManager

logger = logging.getLogger("animaworks.server")

# Paths to exclude from request logging (noisy health checks, etc.)
_NOISY_PATHS = frozenset(
    {
        "/api/system/health",
        "/api/system/status",
        "/ws",
    }
)


class RequestLoggingMiddleware:
    """Pure ASGI middleware for request logging.

    Avoids BaseHTTPMiddleware which buffers StreamingResponse bodies,
    causing stuttery SSE delivery. Binds ``request_id`` into structlog
    contextvars so all log records carry the ID.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = request.headers.get(
            "X-Request-ID",
            uuid.uuid4().hex[:12],
        )
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        status_code = 500

        async def _send_wrapper(message: dict) -> None:  # type: ignore[type-arg]
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            if request.url.path not in _NOISY_PATHS:
                req_logger = logging.getLogger("animaworks.request")
                req_logger.info(
                    "request %s %s -> %d (%.1fms)",
                    request.method,
                    request.url.path,
                    status_code,
                    duration_ms,
                )


async def _reconcile_assets_at_startup(animas_dir: Path) -> None:
    """Background task: generate missing anima assets after startup."""
    try:
        from core.asset_reconciler import reconcile_all_assets
        from core.config.models import load_config

        enable_3d = True
        image_style = "realistic"
        try:
            cfg = load_config()
            enable_3d = cfg.image_gen.enable_3d
            image_style = cfg.image_gen.image_style or "realistic"
        except Exception:
            logger.debug("Failed to load image_gen config for asset reconciliation; using defaults")

        results = await reconcile_all_assets(
            animas_dir,
            enable_3d=enable_3d,
            image_style=image_style,
        )
        if results:
            logger.info("Startup asset reconciliation: %d anima(s) processed", len(results))
    except Exception:
        logger.exception("Startup asset reconciliation failed")


async def _startup_animas_background(app: FastAPI) -> None:
    """Background task: start anima processes and post-startup services.

    Runs after the web server is already accepting requests so that the
    dashboard is reachable immediately.
    """
    try:
        # Register anima lifecycle callbacks for reconciliation
        def _on_anima_added(name: str) -> None:
            if name not in app.state.anima_names:
                app.state.anima_names.append(name)
                from core.org_sync import sync_org_structure

                sync_org_structure(app.state.animas_dir)
                logger.info("Anima added via reconciliation: %s", name)

        def _on_anima_removed(name: str) -> None:
            if name in app.state.anima_names:
                app.state.anima_names.remove(name)
                logger.info("Anima removed via reconciliation: %s", name)

        app.state.supervisor.on_anima_added = _on_anima_added
        app.state.supervisor.on_anima_removed = _on_anima_removed

        # ── Frontmatter migration (before starting animas) ──────
        try:
            from core.memory.frontmatter import FrontmatterService

            _migrated_total = 0
            _repaired_total = 0
            for _aname in app.state.anima_names:
                _adir = app.state.animas_dir / _aname
                _fm_svc = FrontmatterService(
                    _adir,
                    _adir / "knowledge",
                    _adir / "procedures",
                )
                _migrated_total += _fm_svc.ensure_procedure_frontmatter()
                _migrated_total += _fm_svc.ensure_knowledge_frontmatter()
                _repaired_total += _fm_svc.repair_knowledge_frontmatter()
                _repaired_total += _fm_svc.repair_procedure_frontmatter()
            if _migrated_total:
                logger.info(
                    "Frontmatter migration: added metadata to %d files",
                    _migrated_total,
                )
            if _repaired_total:
                logger.info(
                    "Frontmatter repair: fixed %d files",
                    _repaired_total,
                )
        except Exception:
            logger.exception("Frontmatter migration failed (non-fatal)")

        # Start all anima processes (parallel internally)
        await app.state.supervisor.start_all(app.state.anima_names)

        # Sync org structure from identity.md/status.json → config.json
        try:
            from core.org_sync import sync_org_structure

            sync_org_structure(app.state.animas_dir)
        except Exception:
            logger.exception("Org structure sync failed at startup")

        # Reconcile missing anima assets (fallback for failed bootstrap)
        asyncio.create_task(_reconcile_assets_at_startup(app.state.animas_dir))

        # ── Slack Socket Mode ─────────────────────────────────
        try:
            from server.slack_socket import SlackSocketModeManager

            socket_manager = SlackSocketModeManager()
            await socket_manager.start()
            app.state.slack_socket_manager = socket_manager
        except Exception:
            logger.exception("Slack Socket Mode startup failed")
            app.state.slack_socket_manager = None

        # ── ConfigReloadManager ───────────────────────────────
        from server.reload_manager import ConfigReloadManager

        app.state.reload_manager = ConfigReloadManager(app)

        logger.info("All anima processes started")

    except asyncio.CancelledError:
        logger.info("Anima background startup cancelled (shutdown)")
    except Exception:
        logger.exception("Background anima startup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only start anima processes when setup is complete
    if app.state.setup_complete:
        # ── WebSocket heartbeat (start first so dashboard is responsive) ──
        await app.state.ws_manager.start_heartbeat()

        # ── Stream Registry cleanup ────────────────────────
        await app.state.stream_registry.start_cleanup_loop()

        # ── Periodic schedulers (don't depend on running animas) ──
        shared_dir = app.state.shared_dir

        from core.time_utils import get_app_timezone

        msg_log_scheduler = AsyncIOScheduler(timezone=get_app_timezone())

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
                enable_3d = True
                image_style = "realistic"
                try:
                    from core.config.models import load_config

                    _cfg = load_config()
                    enable_3d = _cfg.image_gen.enable_3d
                    image_style = _cfg.image_gen.image_style or "realistic"
                except Exception:
                    pass
                await reconcile_all_assets(
                    app.state.animas_dir,
                    enable_3d=enable_3d,
                    image_style=image_style,
                )
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

        # ── Claude CLI / SDK auto-update ─────────────────
        from core.auto_updater import run_update_check

        async def _auto_update_claude() -> None:
            try:
                result = await run_update_check(
                    supervisor=app.state.supervisor,
                    animas_dir=app.state.animas_dir,
                )
                sdk_info = result.get("sdk", "")
                cli_info = result.get("cli", "")
                if "→" in sdk_info or "→" in cli_info:
                    logger.info("Auto-update completed: sdk=%s cli=%s", sdk_info, cli_info)
            except asyncio.CancelledError:
                logger.debug("Auto-update cancelled (shutdown)")
            except Exception:
                logger.exception("Auto-update check failed")

        msg_log_scheduler.add_job(
            _auto_update_claude,
            IntervalTrigger(hours=4),
            id="claude_auto_update",
            name="System: Claude CLI/SDK Auto-Update",
            replace_existing=True,
        )

        msg_log_scheduler.start()
        app.state.msg_log_scheduler = msg_log_scheduler

        # ── Set embed URL for child processes ──────────────────
        # Child processes inherit this env var and use HTTP for embeddings
        # instead of loading SentenceTransformer on their own GPU.
        import os

        _embed_config = load_config()
        _server_port = getattr(_embed_config.server, "port", 18500)
        os.environ["ANIMAWORKS_EMBED_URL"] = (
            f"http://127.0.0.1:{_server_port}/api/internal/embed"
        )

        # ── Start anima processes in background (parallel) ──
        # Web server is already accepting requests at this point.
        app.state._anima_startup_task = asyncio.create_task(
            _startup_animas_background(app),
        )

        logger.info("Server started (anima processes launching in background)")
    else:
        logger.info("Server started in setup mode (setup not yet complete)")
    yield
    # Shutdown
    if app.state.setup_complete:
        # Cancel background startup if still running
        startup_task = getattr(app.state, "_anima_startup_task", None)
        if startup_task and not startup_task.done():
            startup_task.cancel()
            try:
                await startup_task
            except asyncio.CancelledError:
                pass

        await app.state.ws_manager.stop_heartbeat()
        app.state.stream_registry.cancel_all_producers()
        await app.state.stream_registry.await_all_producers(timeout=5.0)
        await app.state.stream_registry.stop_cleanup_loop()
        if getattr(app.state, "slack_socket_manager", None):
            await app.state.slack_socket_manager.stop()
        await app.state.supervisor.shutdown_all()
        if hasattr(app.state, "msg_log_scheduler"):
            app.state.msg_log_scheduler.shutdown(wait=False)
        # Clean up embed URL env var to avoid leaking into test suites
        import os

        os.environ.pop("ANIMAWORKS_EMBED_URL", None)
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
    from core.paths import get_data_dir

    run_dir = get_data_dir() / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    from core.supervisor.manager import HealthConfig

    health_cfg = HealthConfig()
    try:
        health_cfg.busy_hang_threshold_sec = float(config.server.busy_hang_threshold)
    except Exception:
        pass

    supervisor = ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        ws_manager=ws_manager,
        health_config=health_cfg,
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
            {"error": "Internal server error"},
            status_code=500,
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
            # Set owner as authenticated user so /api/auth/me and other routes work
            if auth_config.owner:
                request.state.user = auth_config.owner
            return await call_next(request)

        # Skip whitelisted paths
        if any(path.startswith(prefix) for prefix in _AUTH_WHITELIST_PREFIXES):
            return await call_next(request)

        # Only protect /api/ and /ws paths
        if not path.startswith("/api/") and path != "/ws" and not path.startswith("/ws/"):
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
