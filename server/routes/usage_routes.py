from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""API usage dashboard — Claude (Anthropic OAuth) & OpenAI (Codex ChatGPT).

Fetches subscription rate-limit data from each provider and exposes it
via ``GET /api/usage``.  Results are cached for 60 seconds.
"""

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.platform.claude_code import get_claude_executable
from core.platform.codex import get_codex_device_login, is_codex_login_available

logger = logging.getLogger("animaworks.routes.usage")

# ── Cache ────────────────────────────────────────────────────────────────────

_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_CACHE_TTL = 60  # seconds


def _cached(key: str) -> dict[str, Any] | None:
    entry = _CACHE.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        return entry[0]
    return None


def _set_cache(key: str, data: dict[str, Any]) -> None:
    _CACHE[key] = (data, time.time())


# ── Credential discovery ─────────────────────────────────────────────────────

_CLAUDE_CRED_PATHS: list[str] = [
    # Environment override
    "env:CLAUDE_CREDENTIALS_PATH",
    # Global Claude Code credentials
    "~/.claude/.credentials.json",
]


def _discover_claude_cred_paths() -> list[str]:
    """Build credential search paths, appending project-specific .claude dirs."""
    paths = list(_CLAUDE_CRED_PATHS)
    # Scan drives/common locations for project-specific .credentials.json
    home = Path.home()
    for base in [home, home / "OneDrive", home / "OneDriveBiz"]:
        if not base.is_dir():
            continue
        for child in base.iterdir():
            cred = child / ".claude" / ".credentials.json"
            if cred.is_file():
                paths.append(str(cred))
    # Also check env variable pointing to specific directories
    data_dir_env = os.environ.get("ANIMAWORKS_DATA_DIR")
    if data_dir_env:
        p = Path(data_dir_env).parent / ".claude" / ".credentials.json"
        paths.append(str(p))
    return paths

_CODEX_CRED_PATHS: list[str] = [
    "env:CODEX_CREDENTIALS_PATH",
    "~/.codex/auth.json",
]


def _find_credential_file(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        if candidate.startswith("env:"):
            env_key = candidate[4:]
            env_val = os.environ.get(env_key)
            if env_val:
                p = Path(env_val)
                if p.is_file():
                    return p
            continue
        p = Path(os.path.expanduser(candidate))
        if p.is_file():
            return p
    return None


def _clear_usage_cache(*keys: str) -> None:
    for key in keys:
        _CACHE.pop(key, None)


def _launch_claude_login_terminal(executable: str | None) -> bool:
    """Open a new CMD window and run `claude login`."""
    if not executable:
        return False
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        cmd_target = Path(executable)
        if cmd_target.suffix.lower() == ".cmd":
            bare_target = cmd_target.with_suffix("")
            if bare_target.exists():
                cmd_target = bare_target
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        command = f'set "ANTHROPIC_API_KEY=" && {cmd_target} /login'
        subprocess.Popen(
            ["cmd.exe", "/k", command],
            creationflags=creationflags,
            cwd=str(Path.home()),
            env=env,
        )
        return True
    except Exception:
        logger.warning("Failed to launch Claude login terminal", exc_info=True)
        return False


# ── Claude (Anthropic OAuth) ─────────────────────────────────────────────────

_ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"  # Claude Code public client


def _refresh_claude_token(cred_path: Path, refresh_token: str) -> str | None:
    """Use the refresh token to obtain a new access token and persist it."""
    try:
        body = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _ANTHROPIC_CLIENT_ID,
        }).encode("utf-8")
        req = urllib.request.Request(
            _ANTHROPIC_TOKEN_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "claude-code/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        new_access = data.get("access_token")
        new_refresh = data.get("refresh_token", refresh_token)
        expires_in = data.get("expires_in", 3600)
        if not new_access:
            return None

        # Update the credentials file
        try:
            cred_data = json.loads(cred_path.read_text("utf-8"))
        except Exception:
            cred_data = {}
        oauth = cred_data.setdefault("claudeAiOauth", {})
        oauth["accessToken"] = new_access
        oauth["refreshToken"] = new_refresh
        oauth["expiresAt"] = int(time.time() * 1000) + expires_in * 1000
        cred_path.write_text(json.dumps(cred_data, ensure_ascii=False), encoding="utf-8")
        logger.info("Refreshed Claude OAuth token, persisted to %s", cred_path)
        return new_access
    except Exception as e:
        logger.warning("Claude token refresh failed: %s", e)
        return None


def _select_best_claude_credential() -> tuple[Path | None, str | None, str | None, int]:
    """Return (credential_path, access_token, refresh_token, expires_at_ms)."""
    candidates = _discover_claude_cred_paths()
    cred_files: list[Path] = []
    for candidate in candidates:
        if candidate.startswith("env:"):
            env_val = os.environ.get(candidate[4:])
            if env_val:
                p = Path(env_val)
                if p.is_file():
                    cred_files.append(p)
            continue
        p = Path(os.path.expanduser(candidate))
        if p.is_file() and p not in cred_files:
            cred_files.append(p)

    best_path: Path | None = None
    best_token: str | None = None
    best_refresh: str | None = None
    best_expires: int = 0
    for path in cred_files:
        try:
            data = json.loads(path.read_text("utf-8"))
            oauth = data.get("claudeAiOauth", {})
            token = oauth.get("accessToken")
            refresh = oauth.get("refreshToken")
            expires = oauth.get("expiresAt", 0)
            if token and expires > best_expires:
                best_path = path
                best_token = token
                best_refresh = refresh
                best_expires = expires
        except Exception:
            logger.debug("Failed to read Claude credentials from %s", path, exc_info=True)

    return best_path, best_token, best_refresh, best_expires


def _read_claude_token() -> str | None:
    """Find the best OAuth token; auto-refresh if expired."""
    best_path, best_token, best_refresh, best_expires = _select_best_claude_credential()
    if not best_token:
        return None

    now_ms = int(time.time() * 1000)
    if best_expires < now_ms and best_refresh and best_path:
        # Token expired — try refresh
        logger.info("Claude OAuth token expired, attempting refresh...")
        refreshed = _refresh_claude_token(best_path, best_refresh)
        if refreshed:
            return refreshed
        logger.warning("Token refresh failed; returning expired token for error reporting")

    return best_token


def _relogin_claude() -> tuple[dict[str, Any], int]:
    """Try token refresh first; fall back to Claude Code CLI login guidance."""
    _clear_usage_cache("claude")
    executable = get_claude_executable()
    if not executable:
        return ({
            "success": False,
            "message": "Claude Code CLI not found. Install it, then run 'claude login' in CMD.",
            "manual_command": "claude login",
        }, 400)

    best_path, best_token, best_refresh, best_expires = _select_best_claude_credential()
    if not best_path or not best_token:
        launched = _launch_claude_login_terminal(executable)
        return ({
            "success": launched,
            "message": "Opened a CMD window for 'claude login'." if launched else "No Claude credentials found. Run 'claude login' in CMD.",
            "manual_command": "claude login",
            "executable": executable,
            "terminal_launched": launched,
        }, 200 if launched else 400)

    now_ms = int(time.time() * 1000)
    if best_expires > now_ms:
        mins = max(0, round((best_expires - now_ms) / 1000 / 60))
        launched = _launch_claude_login_terminal(executable)
        return ({
            "success": True,
            "message": (
                f"Claude token is already fresh (expires in ~{mins} min). "
                "Opened a CMD window for 'claude login' anyway so you can force re-auth manually."
                if launched else
                f"Claude token is already fresh (expires in ~{mins} min). If usage still fails, it is likely a provider-side rate limit rather than expired auth."
            ),
            "file": str(best_path),
            "executable": executable,
            "terminal_launched": launched,
            "manual_command": "claude login",
        }, 200)

    if not best_refresh:
        launched = _launch_claude_login_terminal(executable)
        return ({
            "success": launched,
            "message": "Opened a CMD window for 'claude login'." if launched else "Claude token is expired and no refresh token is available. Run 'claude login' in CMD.",
            "manual_command": "claude login",
            "file": str(best_path),
            "executable": executable,
            "terminal_launched": launched,
        }, 200 if launched else 400)

    refreshed = _refresh_claude_token(best_path, best_refresh)
    _clear_usage_cache("claude")
    if refreshed:
        return ({
            "success": True,
            "message": "Claude token refresh succeeded.",
            "file": str(best_path),
            "executable": executable,
        }, 200)

    launched = _launch_claude_login_terminal(executable)
    return ({
        "success": launched,
        "message": "Claude token refresh failed, so a CMD window for 'claude login' was opened." if launched else "Claude token refresh failed. Run 'claude login' in CMD.",
        "manual_command": "claude login",
        "executable": executable,
        "terminal_launched": launched,
    }, 200 if launched else 400)


def _relogin_openai() -> tuple[dict[str, Any], int]:
    """Start Codex browser login when needed, or report active login."""
    _clear_usage_cache("openai")
    if is_codex_login_available():
        return ({
            "success": True,
            "already_logged_in": True,
            "message": "Codex login is already available",
        }, 200)

    payload = get_codex_device_login()
    _clear_usage_cache("openai")
    status_code = 200 if payload.get("ok") else 400
    return ({
        "success": bool(payload.get("ok")),
        **payload,
        "manual_command": "codex login",
    }, status_code)


def _fetch_claude_usage(skip_cache: bool = False) -> dict[str, Any]:
    if not skip_cache:
        cached = _cached("claude")
        if cached is not None:
            return cached

    token = _read_claude_token()
    if not token:
        result: dict[str, Any] = {"error": "no_credentials", "message": "Claude credentials not found"}
        _set_cache("claude", result)
        return result

    try:
        req = urllib.request.Request(
            _ANTHROPIC_USAGE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "User-Agent": "claude-code/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        result = {"provider": "claude"}

        if "five_hour" in raw:
            fh = raw["five_hour"]
            result["five_hour"] = {
                "utilization": fh.get("utilization", 0),
                "remaining": 100 - fh.get("utilization", 0),
                "resets_at": fh.get("resets_at"),
                "window_seconds": 18000,  # 5 hours
            }

        if "seven_day" in raw:
            sd = raw["seven_day"]
            result["seven_day"] = {
                "utilization": sd.get("utilization", 0),
                "remaining": 100 - sd.get("utilization", 0),
                "resets_at": sd.get("resets_at"),
                "window_seconds": 604800,  # 7 days
            }

        if "additional_capacity" in raw:
            ac = raw["additional_capacity"]
            if ac.get("limit", 0) > 0:
                result["additional_capacity"] = {
                    "utilization": ac.get("utilization", 0),
                    "remaining": 100 - ac.get("utilization", 0),
                    "used_tokens": ac.get("used", 0),
                    "limit_tokens": ac.get("limit", 0),
                }

        _set_cache("claude", result)
        return result

    except urllib.error.HTTPError as e:
        if e.code == 401:
            result = {"error": "unauthorized", "message": "Token expired — re-login to Claude Code"}
        elif e.code == 429:
            # Don't cache rate-limit errors — retry sooner
            return {"error": "rate_limited", "message": "Rate limited — retry shortly"}
        else:
            result = {"error": "http_error", "message": f"HTTP {e.code}"}
        _set_cache("claude", result)
        return result
    except Exception as e:
        logger.warning("Claude usage fetch failed: %s", e)
        return {"error": "fetch_failed", "message": str(e)[:200]}


# ── OpenAI (ChatGPT subscription via Codex auth) ─────────────────────────────

_CHATGPT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


def _read_codex_credentials() -> tuple[str | None, str | None]:
    """Return (access_token, account_id) from Codex auth file."""
    path = _find_credential_file(_CODEX_CRED_PATHS)
    if not path:
        return None, None
    try:
        data = json.loads(path.read_text("utf-8"))
        tokens = data.get("tokens", {})
        return tokens.get("access_token"), tokens.get("account_id")
    except Exception:
        logger.debug("Failed to read Codex credentials from %s", path, exc_info=True)
        return None, None


def _window_label(seconds: int) -> str:
    """Convert limit_window_seconds to a human label like '5h' or 'Week'."""
    hours = seconds / 3600
    if hours <= 24:
        return f"{hours:.0f}h"
    days = hours / 24
    if days >= 6:
        return "Week"
    return f"{days:.0f}d"


def _fetch_openai_usage(skip_cache: bool = False) -> dict[str, Any]:
    if not skip_cache:
        cached = _cached("openai")
        if cached is not None:
            return cached

    token, account_id = _read_codex_credentials()
    if not token:
        result: dict[str, Any] = {"error": "no_credentials", "message": "Codex credentials not found"}
        _set_cache("openai", result)
        return result

    try:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "CodexBar",
            "Accept": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        req = urllib.request.Request(_CHATGPT_USAGE_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        result: dict[str, Any] = {"provider": "openai"}
        rl = raw.get("rate_limit", {})

        for key, slot in [("primary", "primary_window"), ("secondary", "secondary_window")]:
            win = rl.get(slot)
            if not win:
                continue
            used_pct = win.get("used_percent", 0)
            reset_at = win.get("reset_at")  # unix seconds
            window_sec = win.get("limit_window_seconds", 0)
            label = _window_label(window_sec) if window_sec else key
            result[label] = {
                "utilization": used_pct,
                "remaining": 100 - used_pct,
                "resets_at": reset_at,  # unix timestamp (seconds)
                "window_seconds": window_sec,
            }

        _set_cache("openai", result)
        return result

    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            result = {"error": "unauthorized", "message": "Codex token expired — re-login to Codex"}
        elif e.code == 429:
            return {"error": "rate_limited", "message": "Rate limited — retry shortly"}
        else:
            result = {"error": "http_error", "message": f"HTTP {e.code}"}
        _set_cache("openai", result)
        return result
    except Exception as e:
        logger.warning("OpenAI usage fetch failed: %s", e)
        return {"error": "fetch_failed", "message": str(e)[:200]}


# ── Route ────────────────────────────────────────────────────────────────────


def create_usage_router() -> APIRouter:
    router = APIRouter()

    @router.get("/usage")
    async def get_usage(request: Request, skip_cache: bool = False) -> dict[str, Any]:
        """Return combined Claude + OpenAI usage data + governor status."""
        governor = getattr(request.app.state, "usage_governor", None)
        governor_info: dict[str, Any] = {"active": False}
        if governor:
            st = governor.state
            governor_info = {
                "active": st.is_governing,
                "suspended_animas": st.suspended_animas,
                "reason": st.reason,
                "since": st.since,
                "last_check": st.last_check,
            }
        return {
            "claude": _fetch_claude_usage(skip_cache=skip_cache),
            "openai": _fetch_openai_usage(skip_cache=skip_cache),
            "cached_at": _CACHE.get("claude", (None, 0))[1],
            "governor": governor_info,
        }

    @router.post("/usage/claude/relogin")
    async def relogin_claude() -> JSONResponse:
        payload, status_code = _relogin_claude()
        return JSONResponse(payload, status_code=status_code)

    @router.post("/usage/openai/relogin")
    async def relogin_openai() -> JSONResponse:
        payload, status_code = _relogin_openai()
        return JSONResponse(payload, status_code=status_code)

    @router.get("/usage/policy")
    async def get_policy(request: Request) -> dict[str, Any]:
        """Return the current usage policy."""
        from core.paths import get_data_dir
        from server.usage_governor import load_policy

        return load_policy(get_data_dir())

    @router.put("/usage/policy")
    async def update_policy(request: Request):
        """Update the usage policy."""
        from core.paths import get_data_dir
        from server.usage_governor import save_policy

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        save_policy(get_data_dir(), body)
        return {"ok": True}

    return router
