from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import html
import inspect
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core import startup_progress
from core.auth.manager import find_user, load_auth, validate_session
from core.config import load_config
from core.i18n import t
from core.supervisor import ProcessSupervisor
from server.localhost import _is_safe_localhost_request
from server.routes import create_router
from server.routes.setup import create_setup_router
from server.stream_registry import StreamRegistry
from server.websocket import WebSocketManager

logger = logging.getLogger("animaworks.server")

# Public embeddable avatars (e.g. Slack) — no session cookie required
_PUBLIC_ICON_ASSET_PATH = re.compile(r"^/api/animas/[^/]+/assets/icon(?:_realistic)?\.png$")

# Paths to exclude from request logging (noisy health checks, etc.)
_NOISY_PATHS = frozenset(
    {
        "/api/system/health",
        "/api/system/status",
        "/ws",
    }
)


def _get_app_version() -> str:
    try:
        return version("animaworks")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
            if match:
                return match.group(1)
        except Exception:
            logger.debug("Failed to read application version from pyproject.toml", exc_info=True)
    return "0.0.0"


async def _call_optional_async(obj: object | None, method_name: str) -> None:
    if obj is None:
        return
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


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


def _normalize_base_path(base_path: str | None) -> str:
    """Return a canonical URL base path such as "/app" or ""."""
    if not base_path or not isinstance(base_path, str):
        return ""
    normalized = "/" + str(base_path).strip("/")
    return "" if normalized == "/" else normalized


def _append_app_root_path(app_root_path: str, base_path: str) -> str:
    """Append base_path to an application root path without duplicating it."""
    if not app_root_path:
        return base_path
    normalized_root = app_root_path.rstrip("/")
    if normalized_root == base_path or normalized_root.endswith(f"{base_path}"):
        return normalized_root
    return f"{normalized_root}{base_path}"


class BasePathMiddleware:
    """Strip configured deployment base path before routing.

    This lets direct uvicorn requests to /app/... behave like a reverse proxy
    that forwards /... to the ASGI app while preserving the effective
    application root for downstream tooling.
    """

    def __init__(self, app: ASGIApp, base_path: str) -> None:
        self.app = app
        self.base_path = _normalize_base_path(base_path)
        self._raw_prefix = self.base_path.encode("ascii") if self.base_path else b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.base_path or scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if path == self.base_path:
            stripped_path = "/"
        elif path.startswith(f"{self.base_path}/"):
            stripped_path = path[len(self.base_path) :] or "/"
        else:
            await self.app(scope, receive, send)
            return

        updated_scope = dict(scope)
        updated_scope["path"] = stripped_path
        app_root_path = str(scope.get("app_root_path") or scope.get("root_path") or "")
        updated_scope["app_root_path"] = _append_app_root_path(app_root_path, self.base_path)

        raw_path = scope.get("raw_path")
        if isinstance(raw_path, bytes):
            if raw_path == self._raw_prefix:
                updated_scope["raw_path"] = b"/"
            elif raw_path.startswith(self._raw_prefix + b"/"):
                updated_scope["raw_path"] = raw_path[len(self._raw_prefix) :] or b"/"

        await self.app(updated_scope, receive, send)


def _startup_default_preflight_runner(*, force_all_vectordb: bool = False) -> None:
    from cli.commands.server import _run_rag_startup_preflight

    _run_rag_startup_preflight(force_all_vectordb=force_all_vectordb)


def _format_startup_elapsed(seconds: object) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _startup_progress_label(snapshot: dict[str, object]) -> str:
    done = snapshot.get("done_count")
    total = snapshot.get("total_count")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        return f"{done}/{total}"
    if isinstance(done, int):
        return str(done)
    return "-"


def _render_startup_progress_html(snapshot: dict[str, object]) -> str:
    phase = str(snapshot.get("phase") or "starting")
    status = str(snapshot.get("status") or "starting")
    detail = str(snapshot.get("detail") or "")
    error = snapshot.get("error")
    title = t("startup.page_title")
    phase_label = t(f"startup.phase.{phase}")
    status_text = t("startup.failed") if status == "failed" else t("startup.in_progress")
    elapsed = _format_startup_elapsed(snapshot.get("elapsed_seconds"))
    progress = _startup_progress_label(snapshot)
    escaped_detail = html.escape(detail) if detail else html.escape(t("startup.detail_pending"))
    error_html = ""
    if error:
        error_html = f'<p class="error">{html.escape(str(error))}</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="3">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #667085;
      --border: #d8dee8;
      --accent: #2563eb;
      --danger: #b42318;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #101418;
        --panel: #171d23;
        --text: #edf2f7;
        --muted: #a6b0bd;
        --border: #2d3743;
        --accent: #60a5fa;
        --danger: #f97066;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(640px, calc(100vw - 32px));
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 1.35rem;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .status {{
      margin: 0 0 22px;
      color: var(--muted);
      line-height: 1.5;
    }}
    dl {{
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 10px 18px;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .bar {{
      height: 8px;
      margin: 24px 0 0;
      overflow: hidden;
      border-radius: 999px;
      background: color-mix(in srgb, var(--accent) 18%, transparent);
    }}
    .bar span {{
      display: block;
      width: 34%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      animation: pulse 1.8s ease-in-out infinite alternate;
    }}
    .error {{
      margin: 18px 0 0;
      color: var(--danger);
      overflow-wrap: anywhere;
    }}
    @keyframes pulse {{
      from {{ transform: translateX(-20%); opacity: 0.65; }}
      to {{ transform: translateX(220%); opacity: 1; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p class="status">{html.escape(status_text)}</p>
    <dl>
      <dt>{html.escape(t("startup.label_phase"))}</dt>
      <dd>{html.escape(phase_label)}</dd>
      <dt>{html.escape(t("startup.label_detail"))}</dt>
      <dd>{escaped_detail}</dd>
      <dt>{html.escape(t("startup.label_progress"))}</dt>
      <dd>{html.escape(progress)}</dd>
      <dt>{html.escape(t("startup.label_elapsed"))}</dt>
      <dd>{html.escape(elapsed)}</dd>
    </dl>
    {error_html}
    <div class="bar" aria-hidden="true"><span></span></div>
  </main>
</body>
</html>
"""


def _startup_status_payload(snapshot: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": snapshot.get("status"),
        "phase": snapshot.get("phase"),
        "detail": snapshot.get("detail"),
        "progress": {
            "done_count": snapshot.get("done_count"),
            "total_count": snapshot.get("total_count"),
            "elapsed_seconds": snapshot.get("elapsed_seconds"),
        },
    }
    if snapshot.get("error"):
        payload["error"] = snapshot.get("error")
    return payload


def _startup_gate_exempt(path: str, *, setup_complete: bool) -> bool:
    if not setup_complete:
        return True
    if path in {"/startup-status", "/health", "/api/system/health"}:
        return True
    return path.startswith(("/startup-status/", "/health/", "/api/system/health/"))


def _request_accepts_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    path = request.url.path
    return "text/html" in accept or (not path.startswith("/api/") and path != "/ws")


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


async def _startup_animas_background(app: FastAPI, *, suppress_errors: bool = True) -> None:
    """Background task: start anima processes and post-startup services.

    Runs inside startup initialization after RAG preflight, while the
    readiness gate keeps dashboard/API routes on the progress page.
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

        # Exclude governor-suspended animas from startup (only when Governor is enabled).
        _gov_excluded: set[str] = set()
        try:
            from core.config.models import load_config as _lc_gov

            if _lc_gov().server.usage_governor.enabled:
                import json as _json

                from core.paths import get_data_dir as _get_dd

                _gsp = _get_dd() / "usage_governor_state.json"
                if _gsp.is_file():
                    _gsd = _json.loads(_gsp.read_text("utf-8"))
                    _gov_excluded = set(_gsd.get("suspended_animas", []))
                    if _gov_excluded:
                        logger.info(
                            "Startup: skipping %d governor-suspended animas: %s",
                            len(_gov_excluded),
                            ", ".join(sorted(_gov_excluded)),
                        )
        except Exception:
            logger.debug("Failed to read governor state at startup", exc_info=True)

        _names_to_start = [n for n in app.state.anima_names if n not in _gov_excluded]

        # ── Ensure infrastructure services (Neo4j, etc.) ──────────
        try:
            from core.infra import ensure_infra_services
            from core.paths import PROJECT_DIR

            await ensure_infra_services(app.state.animas_dir, _names_to_start, PROJECT_DIR)
        except Exception:
            logger.warning("Infrastructure service check failed", exc_info=True)

        # Start anima processes (parallel internally)
        await app.state.supervisor.start_all(_names_to_start)

        # Sync org structure from identity.md/status.json → config.json
        try:
            from core.org_sync import sync_org_structure

            sync_org_structure(app.state.animas_dir)
        except Exception:
            logger.exception("Org structure sync failed at startup")

        # Reconcile missing anima assets (fallback for failed bootstrap)
        asyncio.create_task(_reconcile_assets_at_startup(app.state.animas_dir))

        # ── Slack: ensure .env slots + warn about missing tokens ──
        try:
            from core.config.env_slots import check_missing_slack_tokens, ensure_all_anima_slots

            ensure_all_anima_slots()
            missing = check_missing_slack_tokens()
            if missing:
                logger.warning(
                    "Slack tokens missing for: %s — edit .env and restart",
                    ", ".join(missing),
                )
        except Exception:
            logger.debug("Slack env slot check failed", exc_info=True)

        # ── Slack Socket Mode ─────────────────────────────────
        _slack_enabled = False
        try:
            _slack_enabled = load_config().external_messaging.slack.enabled
        except Exception:
            pass

        try:
            from server.slack_socket import SlackSocketModeManager

            socket_manager = SlackSocketModeManager()
            await asyncio.wait_for(socket_manager.start(), timeout=30)
            app.state.slack_socket_manager = socket_manager
        except TimeoutError:
            logger.error("Slack Socket Mode startup timed out (30s)")
            app.state.slack_socket_manager = None
        except Exception as exc:
            logger.error(
                "Slack Socket Mode startup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            app.state.slack_socket_manager = None

        if _slack_enabled and app.state.slack_socket_manager is None:
            logger.critical(
                "Slack is enabled but Socket Mode failed to start — "
                "Slack replies will NOT be received. "
                "Install slack-bolt: pip install 'animaworks[communication]'"
            )

        # ── Discord Gateway ────────────────────────────────────
        try:
            from server.discord_gateway import DiscordGatewayManager

            discord_manager = DiscordGatewayManager()
            await asyncio.wait_for(discord_manager.start(), timeout=35)
            app.state.discord_gateway_manager = discord_manager
        except TimeoutError:
            logger.error("Discord Gateway startup timed out (35s)")
            app.state.discord_gateway_manager = None
        except Exception as exc:
            logger.error(
                "Discord Gateway startup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            app.state.discord_gateway_manager = None

        # ── Discord channel → board sync (initial) ───────────
        if app.state.discord_gateway_manager is not None:
            try:
                from server.discord_channel_sync import DiscordChannelSync

                discord_sync = DiscordChannelSync()
                await discord_sync.sync(app.state.discord_gateway_manager)
                app.state.discord_channel_sync = discord_sync
            except Exception:
                logger.warning("Initial Discord channel sync failed", exc_info=True)
                app.state.discord_channel_sync = None
        else:
            app.state.discord_channel_sync = None

        # ── Zoom RTMS Gateway ──────────────────────────────────
        try:
            from server.zoom_gateway import ZoomRTMSManager

            zoom_manager = ZoomRTMSManager()
            await asyncio.wait_for(zoom_manager.start(), timeout=35)
            app.state.zoom_gateway_manager = zoom_manager
        except TimeoutError:
            logger.error("Zoom RTMS Gateway startup timed out (35s)")
            app.state.zoom_gateway_manager = None
        except Exception as exc:
            logger.error(
                "Zoom RTMS Gateway startup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            app.state.zoom_gateway_manager = None

        # ── Slack channel → board sync (initial) ──────────────
        if app.state.slack_socket_manager is not None:
            try:
                from server.slack_channel_sync import SlackChannelSync

                channel_sync = SlackChannelSync()
                await channel_sync.sync(app.state.slack_socket_manager)
                app.state.slack_channel_sync = channel_sync
            except Exception:
                logger.warning("Initial Slack channel sync failed", exc_info=True)
                app.state.slack_channel_sync = None
        else:
            app.state.slack_channel_sync = None

        # ── ConfigReloadManager ───────────────────────────────
        from server.reload_manager import ConfigReloadManager

        app.state.reload_manager = ConfigReloadManager(app)

        logger.info("All anima processes started")

    except asyncio.CancelledError:
        logger.info("Anima background startup cancelled (shutdown)")
        raise
    except Exception:
        logger.exception("Background anima startup failed")
        if not suppress_errors:
            raise


async def _prepare_startup_vector_worker(app: FastAPI) -> None:
    vector_worker = getattr(app.state, "vector_worker", None)
    previous_vector_url_present = "ANIMAWORKS_VECTOR_URL" in os.environ
    previous_vector_url = os.environ.get("ANIMAWORKS_VECTOR_URL")
    app.state._previous_vector_url_present = previous_vector_url_present
    app.state._previous_vector_url = previous_vector_url
    await _call_optional_async(vector_worker, "start")
    vector_worker_url = getattr(vector_worker, "base_url", None)
    if isinstance(vector_worker_url, str) and vector_worker_url:
        os.environ["ANIMAWORKS_VECTOR_URL"] = vector_worker_url
        logger.info("Server RAG vector access routed through vector worker: %s", vector_worker_url)

    _embed_config = load_config()
    _server_port = getattr(_embed_config.server, "port", 18500)
    app.state.child_env_urls = {
        "ANIMAWORKS_EMBED_URL": f"http://127.0.0.1:{_server_port}/api/internal/embed",
        "ANIMAWORKS_VECTOR_URL": f"http://127.0.0.1:{_server_port}/api/internal/vector",
    }
    app.state.supervisor.child_env_urls = app.state.child_env_urls


async def _start_usage_governor_if_enabled(app: FastAPI) -> None:
    from core.config.models import load_config as _load_cfg_gov

    if _load_cfg_gov().server.usage_governor.enabled:
        from core.paths import get_data_dir
        from server.usage_governor import UsageGovernor

        governor = UsageGovernor(app, get_data_dir(), app.state.animas_dir)
        app.state.usage_governor = governor
        await governor.start()
    else:
        logger.info("Usage Governor is disabled (server.usage_governor.enabled=false)")


async def _run_startup_initialization(app: FastAPI) -> None:
    """Run heavyweight startup work after the ASGI app is accepting requests."""
    try:
        startup_progress.set_phase("preflight", detail=t("startup.detail_vector_worker"), reset_counts=True)
        await _prepare_startup_vector_worker(app)

        preflight_runner = getattr(app.state, "startup_preflight_runner", _startup_default_preflight_runner)
        startup_progress.set_phase("preflight", detail=t("startup.detail_preflight"), reset_counts=True)
        await asyncio.to_thread(preflight_runner, force_all_vectordb=False)

        startup_progress.raise_if_cancelled()
        startup_progress.set_phase(
            "spawning_animas",
            detail=t("startup.detail_spawning"),
            done_count=0,
            total_count=len(getattr(app.state, "anima_names", []) or []),
        )
        await _startup_animas_background(app, suppress_errors=False)
        await _start_usage_governor_if_enabled(app)
        startup_progress.set_phase("ready", detail=t("startup.detail_ready"), reset_counts=True)
        logger.info("Server startup initialization complete")
    except asyncio.CancelledError:
        logger.info("Startup initialization cancelled (shutdown)")
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        startup_progress.set_phase("failed", detail=t("startup.detail_failed"), error=message)
        logger.exception("Startup initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only start anima processes when setup is complete
    if app.state.setup_complete:
        startup_progress.begin_startup(t("startup.detail_starting"))
        # ── Global permissions cache ────────────────────────
        from core.config.global_permissions import GlobalPermissionsCache
        from core.paths import get_global_permissions_path

        gp_cache = GlobalPermissionsCache.get()
        gp_path = get_global_permissions_path()
        try:
            gp_cache.load(gp_path)
        except SystemExit:
            raise
        except FileNotFoundError:
            logger.critical(
                "permissions.global.json not found at %s — "
                "server cannot start without global command security. "
                "Run 'animaworks init' to generate defaults.",
                gp_path,
            )
            raise SystemExit(
                f"Fatal: permissions.global.json not found at {gp_path}. Run 'animaworks init' to generate it."
            ) from None
        except Exception:
            logger.critical("Failed to load permissions.global.json — server cannot start")
            raise

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

        msg_log_scheduler.add_job(
            gp_cache.check_integrity,
            IntervalTrigger(minutes=5),
            id="global_permissions_integrity",
            name="System: Global Permissions Integrity Check",
            replace_existing=True,
        )

        # ── Slack channel → board periodic resync ─────────
        async def _slack_channel_resync() -> None:
            try:
                sync = getattr(app.state, "slack_channel_sync", None)
                mgr = getattr(app.state, "slack_socket_manager", None)
                if sync is not None and mgr is not None:
                    await sync.sync(mgr)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Periodic Slack channel sync failed", exc_info=True)

        msg_log_scheduler.add_job(
            _slack_channel_resync,
            IntervalTrigger(minutes=5),
            id="slack_channel_sync",
            name="System: Slack Channel → Board Sync",
            replace_existing=True,
        )

        msg_log_scheduler.start()
        app.state.msg_log_scheduler = msg_log_scheduler

        # ── Startup initialization ─────────────────────────
        # Heavy RAG preflight/reindex and anima spawning run after lifespan
        # yields so uvicorn can bind and serve the progress page immediately.
        app.state._anima_startup_task = asyncio.create_task(
            _run_startup_initialization(app),
        )

        logger.info("Server started (startup initialization running in background)")
    else:
        startup_progress.set_phase("ready", detail=t("startup.detail_setup_mode"), reset_counts=True)
        logger.info("Server started in setup mode (setup not yet complete)")
    yield
    # Shutdown
    if app.state.setup_complete:
        # Cancel background startup if still running
        startup_task = getattr(app.state, "_anima_startup_task", None)
        if startup_task and not startup_task.done():
            startup_progress.request_cancel()
            try:
                await asyncio.wait_for(asyncio.shield(startup_task), timeout=10.0)
            except TimeoutError:
                startup_task.cancel()
                try:
                    await startup_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

        await app.state.ws_manager.stop_heartbeat()
        app.state.stream_registry.cancel_all_producers()
        await app.state.stream_registry.await_all_producers(timeout=5.0)
        await app.state.stream_registry.stop_cleanup_loop()
        if getattr(app.state, "slack_socket_manager", None):
            await app.state.slack_socket_manager.stop()
        if getattr(app.state, "discord_gateway_manager", None):
            await app.state.discord_gateway_manager.stop()
        if getattr(app.state, "zoom_gateway_manager", None):
            await app.state.zoom_gateway_manager.stop()
        governor = getattr(app.state, "usage_governor", None)
        if governor:
            await governor.stop()
        await app.state.supervisor.shutdown_all()
        vector_worker = getattr(app.state, "vector_worker", None)
        await _call_optional_async(vector_worker, "stop")
        if getattr(app.state, "_previous_vector_url_present", False) is True:
            previous = getattr(app.state, "_previous_vector_url", None)
            if isinstance(previous, str):
                os.environ["ANIMAWORKS_VECTOR_URL"] = previous
        else:
            os.environ.pop("ANIMAWORKS_VECTOR_URL", None)
        if hasattr(app.state, "msg_log_scheduler"):
            app.state.msg_log_scheduler.shutdown(wait=False)
    logger.info("Server stopped")


def create_app(
    animas_dir: Path,
    shared_dir: Path,
    *,
    force_startup_repair_all_vectordb: bool = False,
) -> FastAPI:
    app = FastAPI(title="AnimaWorks", version=_get_app_version(), lifespan=lifespan)

    ws_manager = WebSocketManager()

    # Run Person→Anima rename migration before any animas_dir access
    try:
        from core.config.migrate import migrate_person_to_anima
        from core.paths import get_data_dir as _get_data_dir

        migrate_person_to_anima(_get_data_dir())
    except Exception:
        logger.exception("Person-to-Anima migration failed")

    config = load_config()
    _base_path = _normalize_base_path(getattr(config.server, "base_path", ""))

    # Create run directory for sockets and PID files
    from core.paths import get_data_dir

    run_dir = get_data_dir() / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    from core.memory.rag.vector_worker_client import VectorWorkerManager

    vector_worker = VectorWorkerManager.from_config(config, log_dir=log_dir)

    from core.supervisor.manager import HealthConfig

    health_cfg = HealthConfig()
    try:
        health_cfg.busy_hang_threshold_sec = float(config.server.busy_hang_threshold)
        health_cfg.health_check_warmup_seconds = float(config.server.health_check_warmup_seconds)
        health_cfg.runner_warmup_seconds = float(config.server.runner_warmup_seconds)
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
    app.state.vector_worker = vector_worker
    app.state.force_startup_repair_all_vectordb = False
    app.state.startup_preflight_runner = _startup_default_preflight_runner

    # Meeting room manager
    from server.room_manager import RoomManager

    room_manager = RoomManager(shared_dir / "meetings")
    room_manager.load_all_rooms()
    app.state.room_manager = room_manager
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
        if path.endswith((".js", ".css", ".html")) or path == "/" or path.startswith("/workspace"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    def _base_prefixed(path: str) -> str:
        if not _base_path:
            return path
        return f"{_base_path}{path}" if path.startswith("/") else f"{_base_path}/{path}"

    # ── Setup guard middleware ──────────────────────────────
    @app.middleware("http")
    async def setup_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        setup_complete = request.app.state.setup_complete

        if not setup_complete:
            # During setup: only setup API and the static assets needed by
            # the setup wizard are accessible.  The setup HTML imports shared
            # modules through the versioned static route.
            if (
                path.startswith("/api/setup")
                or path.startswith("/setup")
                or path.startswith("/_v/")
                or path.startswith("/shared/")
            ):
                response = await call_next(request)
                # Prevent browser caching of setup static files so code
                # updates are picked up immediately without a hard refresh.
                if not path.startswith("/api/setup"):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                return response
            # Root redirects to setup wizard
            if path == "/":
                return RedirectResponse(_base_prefixed("/setup/"))
            # Block all other API/dashboard routes
            if path.startswith("/api/"):
                return JSONResponse(
                    {"error": "Setup not yet complete"},
                    status_code=503,
                )
            return RedirectResponse(_base_prefixed("/setup/"))
        else:
            # After setup: block setup API
            if path.startswith("/api/setup"):
                return JSONResponse(
                    {"error": "Setup already completed"},
                    status_code=403,
                )
            # Redirect /setup/* to dashboard
            if path.startswith("/setup"):
                return RedirectResponse(_base_prefixed("/"))
            return await call_next(request)

    # ── Auth guard middleware ──────────────────────────────
    # Paths that don't require authentication
    _AUTH_WHITELIST_PREFIXES = (
        "/api/auth/login",
        "/api/system/health",
        "/api/setup",
        "/api/approve",
        "/health",
    )

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

        # Public icon PNGs (GET/HEAD from asset route only)
        if request.method in ("GET", "HEAD") and _PUBLIC_ICON_ASSET_PATH.match(path):
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

    # ── Startup readiness gate ─────────────────────────────
    # Added after auth_guard so it runs before auth, but before
    # BasePathMiddleware is appended so base-path deployments are stripped
    # before this path logic runs.
    @app.middleware("http")
    async def startup_readiness_gate(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if _startup_gate_exempt(path, setup_complete=request.app.state.setup_complete):
            return await call_next(request)

        progress = startup_progress.snapshot()
        if progress.get("phase") == "ready":
            return await call_next(request)

        headers = {"Retry-After": "5", "Cache-Control": "no-store"}
        if _request_accepts_html(request):
            return HTMLResponse(
                _render_startup_progress_html(progress),
                status_code=503,
                headers=headers,
            )
        return JSONResponse(
            _startup_status_payload(progress),
            status_code=503,
            headers=headers,
        )

    @app.get("/startup-status", include_in_schema=False)
    async def _startup_status():
        return JSONResponse(startup_progress.snapshot(), headers={"Cache-Control": "no-store"})

    @app.get("/health", include_in_schema=False)
    async def _health():
        return {"status": "ok"}

    # ── Route registration ─────────────────────────────────
    # Always mount both routers; the middleware handles access control.
    app.include_router(create_router())
    app.include_router(create_setup_router())

    # ── Version stamp (changes on every server start) ────
    _app_version = str(int(time.time()))
    static_dir = Path(__file__).parent / "static"

    # ── Serve index.html as template with version injection ─
    # Replace __AW_VERSION__ so all resource URLs (CSS, JS, import-map)
    # include the server-start timestamp.  This guarantees the browser
    # loads fresh resources after every restart — no manual cache-clear
    # needed, even on plain HTTP (where Clear-Site-Data is ignored).
    _index_raw = (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    async def _serve_index():
        html = _index_raw.replace("__AW_VERSION__", _app_version).replace("__AW_BASE__", _base_path)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    # ── Versioned static file route ───────────────────────
    # Serves /_v/{version}/path → static/path with no-store.
    # All CSS <link> hrefs, JS module imports, and the import-map use
    # this prefix so the browser fetches fresh files after restart.
    _MIME_MAP = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".html": "text/html",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
    }

    @app.get("/_v/{version}/{path:path}", include_in_schema=False)
    async def _serve_versioned_static(version: str, path: str):
        # Prevent path traversal
        safe = Path(path)
        if ".." in safe.parts:
            raise HTTPException(status_code=400, detail="Invalid path")
        file = static_dir / safe
        if not file.exists() or not file.is_file():
            raise HTTPException(status_code=404)
        media = _MIME_MAP.get(file.suffix.lower(), "application/octet-stream")
        return FileResponse(str(file), media_type=media, headers={"Cache-Control": "no-store"})

    # ── Static files (fallback for non-versioned paths) ───
    # Serve setup/workspace index.html with base-path + version injection
    # (explicit routes take priority over StaticFiles mounts).
    setup_static_dir = static_dir / "setup"
    workspace_static_dir = static_dir / "workspace"

    def _inject_html(raw: str) -> str:
        return raw.replace("__AW_VERSION__", _app_version).replace("__AW_BASE__", _base_path)

    _setup_html_raw = ""
    if (setup_static_dir / "index.html").exists():
        _setup_html_raw = (setup_static_dir / "index.html").read_text(encoding="utf-8")

    _workspace_html_raw = ""
    if (workspace_static_dir / "index.html").exists():
        _workspace_html_raw = (workspace_static_dir / "index.html").read_text(encoding="utf-8")

    if _setup_html_raw:

        @app.get("/setup", include_in_schema=False)
        @app.get("/setup/", include_in_schema=False)
        async def _serve_setup_index():
            return HTMLResponse(_inject_html(_setup_html_raw), headers={"Cache-Control": "no-store"})

    if _workspace_html_raw:

        @app.get("/workspace", include_in_schema=False)
        @app.get("/workspace/", include_in_schema=False)
        async def _serve_workspace_index():
            return HTMLResponse(_inject_html(_workspace_html_raw), headers={"Cache-Control": "no-store"})

    if setup_static_dir.exists():
        app.mount(
            "/setup",
            StaticFiles(directory=str(setup_static_dir), html=False),
            name="setup_static",
        )

    if workspace_static_dir.exists():
        app.mount(
            "/workspace",
            StaticFiles(directory=str(workspace_static_dir), html=False),
            name="workspace_static",
        )

    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=False), name="static")

    if _base_path:
        app.add_middleware(BasePathMiddleware, base_path=_base_path)

    return app
