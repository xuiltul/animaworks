from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
import asyncio
import html
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event

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

from core.anima.workspace_files import IDENTITY_FILE
from core.auth.manager import find_user, load_auth, validate_session
from core.config import load_config
from core.i18n import t
from core.infra import startup_progress
from core.platform.tasks import spawn
from server.localhost import _is_safe_localhost_request
from server.routes import create_router
from server.routes.setup import create_setup_router
from server.stream_registry import StreamRegistry
from server.supervisor import HealthConfig, ProcessSupervisor
from server.websocket import WebSocketManager

logger = logging.getLogger("animaworks.server")


def _confirm_global_permissions_change(prompt: str) -> bool:
    """Ask the server operator to accept changed global permission rules."""
    if not sys.stdin.isatty():
        raise SystemExit(
            "permissions.global.json was modified and non-interactive "
            "session cannot confirm. Start server from an interactive terminal."
        )
    return input(prompt).strip().lower() == "yes"


# Public embeddable avatars (e.g. Slack) — no session cookie required
_PUBLIC_ICON_ASSET_PATH = re.compile(r"^/api/animas/[^/]+/assets/icon(?:_realistic)?\.png$")

_AUTH_WHITELIST_PREFIXES = (
    "/api/auth/login",
    "/api/system/health",
    "/api/setup",
    "/api/approve",
    # Webhook receivers authenticate via per-platform HMAC signature
    # verification (Slack/Chatwork/Zoom), not session cookies.
    "/api/webhooks/",
    "/health",
)

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


def _startup_default_preflight_runner() -> None:
    from core.infra.execution_sdk_preflight import run_execution_sdk_preflight

    run_execution_sdk_preflight()


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
  <script>
    // meta refresh to the same URL is a same-document navigation when the URL
    // has a fragment (e.g. /#/chat), so it never reloads; force a real reload.
    setTimeout(function () {{ location.reload(); }}, 3000);
  </script>
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
    """Background task: generate missing anima assets once during startup."""
    try:
        from core.anima.asset_reconciler import reconcile_all_assets
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


def _schedule_startup_asset_reconciliation(app: FastAPI) -> None:
    """Schedule one fallback asset scan per app runtime."""
    if getattr(app.state, "_asset_reconciliation_scheduled", False):
        return
    app.state._asset_reconciliation_scheduled = True
    spawn(
        _reconcile_assets_at_startup(app.state.animas_dir),
        name="startup-asset-reconciliation",
    )


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
                from core.org.org_sync import sync_org_structure

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

        _names_to_start = list(app.state.anima_names)

        # Start anima processes (parallel internally)
        await app.state.supervisor.start_all(_names_to_start)

        # Sync org structure from IDENTITY.md/status.json → config.json
        try:
            from core.org.org_sync import sync_org_structure

            sync_org_structure(app.state.animas_dir)
        except Exception:
            logger.exception("Org structure sync failed at startup")

        # Reconcile missing anima assets once as a fallback for failed bootstrap.
        _schedule_startup_asset_reconciliation(app)

        # Enclave mode never starts outward channels (Slack/Discord/Zoom/GitHub).
        try:
            _enclave_mode = load_config().enclave.enabled
        except Exception:
            logger.warning("Could not read enclave mode; starting no external gateways", exc_info=True)
            _enclave_mode = True
        if _enclave_mode:
            logger.info("Enclave mode: external gateways are not started")
            for _attr in (
                "slack_socket_manager",
                "discord_gateway_manager",
                "discord_channel_sync",
                "zoom_gateway_manager",
                "github_gateway_manager",
                "slack_channel_sync",
            ):
                setattr(app.state, _attr, None)
        else:
            # ── Slack: ensure .env slots + warn about missing tokens ──
            try:
                from core.config.env_slots import ensure_all_anima_slots
                from core.credentials import check_missing_slack_tokens

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
                logger.debug("Best-effort operation failed", exc_info=True)

            try:
                from server.gateways.slack_socket import SlackSocketModeManager

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
                from server.gateways.discord_gateway import DiscordGatewayManager

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
                    from server.gateways.discord_channel_sync import DiscordChannelSync

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
                from server.gateways.zoom_gateway import ZoomRTMSManager

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

            # ── GitHub Webhook Gateway ─────────────────────────────
            try:
                from server.gateways.github_gateway import GitHubWebhookManager

                github_manager = GitHubWebhookManager()
                await github_manager.start()
                app.state.github_gateway_manager = github_manager
            except Exception as exc:
                logger.error(
                    "GitHub Webhook Gateway startup failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                app.state.github_gateway_manager = None

            # ── Slack channel → board sync (initial) ──────────────
            if app.state.slack_socket_manager is not None:
                try:
                    from server.gateways.slack_channel_sync import SlackChannelSync

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


def _prepare_child_env_urls(app: FastAPI) -> None:
    from core.memory.rag.endpoints import RagEndpoints, child_env

    _embed_config = load_config()
    _server_port = getattr(app.state, "listen_port", getattr(_embed_config.server, "port", 18500))
    app.state.child_env_urls = child_env(RagEndpoints.for_server(_server_port))
    app.state.supervisor.child_env_urls = app.state.child_env_urls


async def _run_startup_initialization(app: FastAPI) -> None:
    """Run heavyweight startup work after the ASGI app is accepting requests."""
    app.state.worker_services_ready = False
    try:
        from core.config.helper_models import validate_helper_model_credentials

        try:
            validate_helper_model_credentials(app.state.animas_dir)
        except Exception as exc:
            logger.warning("Helper model credential validation skipped: %s", type(exc).__name__)

        _prepare_child_env_urls(app)
        from core.memory.rag.vector_registry import configure_server_vector_access

        configure_server_vector_access(app.state.supervisor.send_request, asyncio.get_running_loop())

        preflight_runner = getattr(app.state, "startup_preflight_runner", _startup_default_preflight_runner)
        startup_progress.set_phase("preflight", detail=t("startup.detail_preflight"), reset_counts=True)
        await asyncio.to_thread(preflight_runner)

        startup_progress.raise_if_cancelled()
        app.state.worker_services_ready = True
        startup_progress.set_phase(
            "spawning_animas",
            detail=t("startup.detail_spawning"),
            done_count=0,
            total_count=len(getattr(app.state, "anima_names", []) or []),
        )
        await _startup_animas_background(app, suppress_errors=False)
        startup_progress.set_phase("ready", detail=t("startup.detail_ready"), reset_counts=True)
        logger.info("Server startup initialization complete")
    except asyncio.CancelledError:
        app.state.worker_services_ready = False
        logger.info("Startup initialization cancelled (shutdown)")
        raise
    except Exception as exc:
        app.state.worker_services_ready = False
        message = f"{type(exc).__name__}: {exc}"
        startup_progress.set_phase("failed", detail=t("startup.detail_failed"), error=message)
        logger.exception("Startup initialization failed")


async def _run_model_warmup() -> None:
    """Pre-load embed/rerank/STT models in the background so the first real
    request (e.g. a voice utterance right after restart) doesn't pay the
    20-30s cold-load penalty."""

    def _warm_native() -> None:
        from core.config import load_config
        from core.memory.rag.embedding import thread_safe_encode
        from core.memory.retrieval.reranker import get_reranker

        thread_safe_encode(["ウォームアップ"], purpose="query")
        model_name = load_config().rag.cross_encoder_model
        reranker = get_reranker(model_name) if model_name else get_reranker()
        reranker.score_sync("ウォームアップ", ["テスト"])

    try:
        await asyncio.to_thread(_warm_native)
        logger.info("Model warmup complete: embed+rerank")
    except Exception:
        logger.exception("Model warmup failed: embed/rerank")

    try:
        from core.config import load_config
        from server.routes.voice import _get_stt

        stt = _get_stt(load_config().voice)
        # 1s of 16kHz mono PCM16 silence — enough to force the model load.
        await stt.transcribe_buffer_async(b"\x00\x00" * 16000)
        logger.info("Model warmup complete: stt")
    except ImportError as exc:
        # Voice input is an optional extra; say so once instead of a traceback.
        logger.info("Voice STT warmup skipped: %s", exc)
    except Exception:
        logger.exception("Model warmup failed: stt")


async def _warm_model_catalog() -> None:
    """Prime the server-side model list before a UI opens its picker."""
    try:
        from core.config.model_discovery import discover_models

        models = await asyncio.to_thread(discover_models)
        logger.info("Model catalog warmup complete: %d models", len(models))
    except Exception:
        logger.exception("Model catalog warmup failed")


async def _activate_runtime_services(app: FastAPI) -> None:
    """Start the runtime both at boot and when the setup wizard finishes."""
    app.state.worker_services_ready = False
    startup_progress.begin_startup(t("startup.detail_starting"))
    # ── Global permissions cache ────────────────────────
    from core.config.global_permissions import GlobalPermissionsCache
    from core.paths import get_global_permissions_path

    gp_cache = GlobalPermissionsCache.get()
    gp_path = get_global_permissions_path()
    try:
        gp_cache.load(gp_path, confirm_change=_confirm_global_permissions_change)
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
    from core.org.org_sync import detect_orphan_animas

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

    # ── Claude CLI / SDK auto-update ─────────────────
    from server.supervisor.auto_updater import run_update_check

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

    # ── External tasks collection (optional) ──────
    try:
        from core.config.models import load_config as _load_cfg_ext_tasks

        _ext_tasks_cfg = _load_cfg_ext_tasks().external_tasks
    except Exception:
        _ext_tasks_cfg = None
        logger.debug("Failed to load external_tasks config; collection disabled", exc_info=True)

    if _ext_tasks_cfg is not None and _ext_tasks_cfg.enabled:
        _ext_interval = max(1, int(_ext_tasks_cfg.interval_minutes))

        def _run_external_tasks_collection() -> None:
            from datetime import UTC, datetime

            from core.config.models import load_config as _lc
            from core.paths import get_external_tasks_store_path
            from core.tasks.external.collector import collect_all
            from core.tasks.external.store import ExternalTaskStore

            store = ExternalTaskStore(get_external_tasks_store_path())
            previous = store.load()
            cfg = _lc().external_tasks
            snapshot = collect_all(cfg, previous, datetime.now(UTC))
            store.save(snapshot)

        async def _external_tasks_collection() -> None:
            try:
                await asyncio.to_thread(_run_external_tasks_collection)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("External tasks collection failed")

        msg_log_scheduler.add_job(
            _external_tasks_collection,
            IntervalTrigger(minutes=_ext_interval),
            id="external_tasks_collection",
            name="System: External Tasks Collection",
            replace_existing=True,
        )
        # Immediate non-blocking run on startup
        spawn(_external_tasks_collection(), name="external-tasks-collection")

    msg_log_scheduler.start()
    app.state.msg_log_scheduler = msg_log_scheduler

    # ── Startup initialization ─────────────────────────
    # Heavy RAG preflight/reindex and anima spawning run after lifespan
    # yields so uvicorn can bind and serve the progress page immediately.
    app.state._anima_startup_task = asyncio.create_task(
        _run_startup_initialization(app),
    )
    app.state._model_warmup_task = asyncio.create_task(_run_model_warmup())
    app.state._model_catalog_warmup_task = asyncio.create_task(_warm_model_catalog())

    logger.info("Server started (startup initialization running in background)")


def _enclave_usage_summary(anima_dir: Path, days: int) -> dict[str, object]:
    """Token usage plus budget fields for one anima, served by the enclave ``/v1/usage``."""
    from core.usage.token_budget import read_budget_fields
    from core.usage.token_usage import TokenUsageLogger

    summary = TokenUsageLogger(anima_dir).summarize(days)
    summary.update(read_budget_fields(anima_dir))
    return summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.tasks.board.tasks import ensure_task_store_schema

    ensure_task_store_schema(Path(app.state.shared_dir) / "taskboard.sqlite3")
    app.state.activate_runtime = lambda: _activate_runtime_services(app)
    if app.state.setup_complete:
        await app.state.activate_runtime()
    else:
        startup_progress.set_phase("ready", detail=t("startup.detail_setup_mode"), reset_counts=True)
        logger.info("Server started in setup mode (setup not yet complete)")
    # Enclave gateway: start only when enclave mode is enabled (fail-closed on
    # socket-rebinding errors).
    from core.enclave.gateway_server import start_gateway

    await start_gateway(app, usage_reader=_enclave_usage_summary)

    from server.enclave_delivery import host_delivery_loop, should_start_host_delivery

    delivery_config = load_config()
    app.state.enclave_host_delivery_task = None
    app.state.enclave_host_delivery_stop = None
    if should_start_host_delivery(delivery_config):
        delivery_stop = Event()
        delivery_warnings: dict[str, str] = {}
        app.state.enclave_host_delivery_stop = delivery_stop
        app.state.enclave_host_delivery_task = asyncio.create_task(
            host_delivery_loop(
                delivery_config.enclaves,
                app.state.anima_names,
                app.state.shared_dir,
                stop_event=delivery_stop,
                warning_state=delivery_warnings,
                interval_s=60.0,
            ),
            name="enclave-host-delivery",
        )
        logger.info("Enclave host result delivery started")

    feed_prewarm_task = None
    if app.state.setup_complete:
        feed_prewarm_task = asyncio.create_task(
            _prewarm_conversation_feeds(app),
            name="conversation-feed-prewarm",
        )

    yield
    # Shutdown
    if feed_prewarm_task is not None and not feed_prewarm_task.done():
        feed_prewarm_task.cancel()
        try:
            await feed_prewarm_task
        except asyncio.CancelledError:
            pass
    delivery_stop = getattr(app.state, "enclave_host_delivery_stop", None)
    delivery_task = getattr(app.state, "enclave_host_delivery_task", None)
    if delivery_stop is not None:
        delivery_stop.set()
    if delivery_task is not None:
        try:
            await delivery_task
        except asyncio.CancelledError:
            pass
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
        if getattr(app.state, "github_gateway_manager", None):
            await app.state.github_gateway_manager.stop()
        from core.enclave.gateway_server import stop_gateway

        await stop_gateway(app)
        await app.state.supervisor.shutdown_all()
        from core.memory.rag.vector_registry import configure_server_vector_access

        configure_server_vector_access(None)
        from core.paths import get_data_dir as _shutdown_get_data_dir
        from server.internal_auth import remove_operator_token

        try:
            remove_operator_token(_shutdown_get_data_dir() / "run" / "internal_api.auth")
        except Exception:
            logger.exception("Failed to remove operator token file")
        if hasattr(app.state, "msg_log_scheduler"):
            app.state.msg_log_scheduler.shutdown(wait=False)
    logger.info("Server stopped")


async def _prewarm_conversation_feeds(app: FastAPI) -> None:
    """Warm the first default-thread feed page after startup has settled."""
    await asyncio.sleep(60)
    from core.activity.logger import ActivityLogger

    for name in list(getattr(app.state, "anima_names", []) or []):
        anima_dir = Path(app.state.animas_dir) / name
        if not anima_dir.is_dir():
            continue
        try:
            activity = ActivityLogger(anima_dir)
            await asyncio.to_thread(
                activity.get_conversation_feed,
                thread_id="default",
                min_user_items=10,
                _schedule_prefetch=False,
            )
        except Exception:
            logger.debug("Conversation feed prewarm failed for %s", name, exc_info=True)


def create_app(
    animas_dir: Path,
    shared_dir: Path,
) -> FastAPI:
    from core.infra.event_export import register_activity_event_exporter

    register_activity_event_exporter()
    app = FastAPI(title="AnimaWorks", version=_get_app_version(), lifespan=lifespan)

    ws_manager = WebSocketManager()

    config = load_config()

    from core.enclave import enforce_enclave_runtime
    from core.paths import get_data_dir

    enforce_enclave_runtime(config, get_data_dir(), host=None)
    _base_path = _normalize_base_path(getattr(config.server, "base_path", ""))

    # Create run directory for sockets and PID files
    from core.paths import get_data_dir

    run_dir = get_data_dir() / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    health_cfg = HealthConfig()
    try:
        health_cfg.health_check_warmup_seconds = float(config.server.health_check_warmup_seconds)
        health_cfg.runner_warmup_seconds = float(config.server.runner_warmup_seconds)
    except Exception:
        logger.debug("Best-effort operation failed", exc_info=True)

    supervisor = ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        ws_manager=ws_manager,
        health_config=health_cfg,
    )

    # Discover anima names from disk (respect status.json)
    from server.supervisor.manager import ProcessSupervisor as _PS

    anima_names: list[str] = []
    if animas_dir.exists():
        for anima_dir in sorted(animas_dir.iterdir()):
            if anima_dir.is_dir() and (anima_dir / IDENTITY_FILE).exists():
                if not _PS.read_anima_enabled(anima_dir):
                    logger.info("Skipping disabled anima: %s", anima_dir.name)
                    continue
                anima_names.append(anima_dir.name)
                logger.info("Discovered anima: %s", anima_dir.name)

    app.state.supervisor = supervisor

    # Internal API caller authentication (R04-1): a per-server secret from which
    # per-anima / operator tokens are derived.  The secret lives only in
    # process memory and is regenerated on every start.
    from server.internal_auth import InternalAuth, set_current_auth, write_operator_token

    internal_auth = InternalAuth.generate()
    app.state.internal_auth = internal_auth
    supervisor.internal_auth = internal_auth
    set_current_auth(internal_auth)
    write_operator_token(run_dir / "internal_api.auth")

    app.state.anima_names = anima_names
    app.state.ws_manager = ws_manager
    app.state.animas_dir = animas_dir
    app.state.shared_dir = shared_dir
    app.state.setup_complete = config.setup_complete
    app.state.worker_services_ready = False
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
            # modules through the versioned static route, and the wizard steps
            # import the same-origin fetch wrapper from /modules/api.js.
            if (
                path.startswith("/api/setup")
                or path.startswith("/setup")
                or path.startswith("/_v/")
                or path.startswith("/shared/")
                or path == "/modules/api.js"
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

        # Internal API caller verification: a valid internal token lets an
        # in-process / sandboxed caller through regardless of trust_localhost,
        # so anima processes still work under password mode with
        # trust_localhost=false.
        if (
            path.startswith("/api/internal/")
            and getattr(request.app.state, "internal_auth", None) is not None
            and request.app.state.internal_auth.verify(request.headers.get("X-AnimaWorks-Internal-Auth")) is not None
        ):
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
        if (
            progress.get("phase") != "failed"
            and getattr(request.app.state, "worker_services_ready", False)
            and path.startswith("/api/internal/")
            and _is_safe_localhost_request(request)
        ):
            # Vector startup and preflight have finished before workers are
            # spawned. Workers need embed/vector and persistence endpoints
            # during their own initialization; gating those on all workers
            # being ready creates a startup dependency cycle. Catch-up indexing
            # can change the progress phase before public readiness, so service
            # availability must not depend on that display state. This only
            # bypasses readiness: the normal auth/setup guards still run.
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

    @app.get("/api/workspace/pixel/scene", include_in_schema=False)
    async def _pixel_workspace_scene():
        scene_path = app.state.animas_dir.parent / "workspace_pixel" / "scene.json"
        if not scene_path.is_file():
            raise HTTPException(status_code=404)
        try:
            payload = json.loads(await asyncio.to_thread(scene_path.read_text, encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("Invalid pixel workspace scene override: %s", exc)
            raise HTTPException(status_code=500, detail="Invalid workspace scene") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail="Invalid workspace scene")
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.get("/api/workspace/pixel/assets/{path:path}", include_in_schema=False)
    async def _pixel_workspace_asset(path: str):
        assets_root = (app.state.animas_dir.parent / "workspace_pixel" / "assets").resolve()
        candidate = (assets_root / path).resolve()
        if not candidate.is_relative_to(assets_root):
            raise HTTPException(status_code=400, detail="Invalid asset path")
        if not candidate.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(candidate, headers={"Cache-Control": "no-store"})

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

    _pixel_workspace_html_raw = ""
    pixel_workspace_index = workspace_static_dir / "pixel" / "index.html"
    if pixel_workspace_index.exists():
        _pixel_workspace_html_raw = pixel_workspace_index.read_text(encoding="utf-8")

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

    if _pixel_workspace_html_raw:

        @app.get("/workspace/pixel", include_in_schema=False)
        async def _redirect_pixel_workspace(request: Request):
            if "mock" not in request.query_params:
                return RedirectResponse(_base_prefixed("/workspace/?renderer=pixel"))
            return HTMLResponse(_inject_html(_pixel_workspace_html_raw), headers={"Cache-Control": "no-store"})

        @app.get("/workspace/pixel/", include_in_schema=False)
        async def _serve_pixel_workspace_index(request: Request):
            if "mock" not in request.query_params:
                return RedirectResponse(_base_prefixed("/workspace/?renderer=pixel"))
            return HTMLResponse(_inject_html(_pixel_workspace_html_raw), headers={"Cache-Control": "no-store"})

    battle_index = static_dir / "battle" / "index.html"
    if battle_index.exists():
        battle_html = battle_index.read_text(encoding="utf-8")

        @app.get("/battle", include_in_schema=False)
        @app.get("/battle/", include_in_schema=False)
        async def _serve_battle_index(request: Request):
            if "demo" not in request.query_params and "mock" not in request.query_params:
                return RedirectResponse(_base_prefixed("/workspace/?view=battle"))
            return HTMLResponse(_inject_html(battle_html), headers={"Cache-Control": "no-store"})

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
