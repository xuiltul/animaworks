from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""API usage dashboard — Claude (Anthropic OAuth) & OpenAI (Codex ChatGPT).

Fetches subscription rate-limit data from each provider and exposes it
via ``GET /api/usage``.  Results are cached for 60 seconds.
"""

import base64
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
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
_USAGE_SNAPSHOT_NAME = "usage_snapshot.json"


def _cached(key: str) -> dict[str, Any] | None:
    entry = _CACHE.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        return entry[0]
    return None


def _set_cache(key: str, data: dict[str, Any]) -> None:
    _CACHE[key] = (data, time.time())


def _usage_snapshot_path() -> Path:
    from core.paths import get_data_dir

    return get_data_dir() / _USAGE_SNAPSHOT_NAME


def _load_usage_snapshot() -> dict[str, Any] | None:
    path = _usage_snapshot_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        logger.warning("Failed to read usage snapshot from %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _save_usage_snapshot(payload: dict[str, Any]) -> None:
    path = _usage_snapshot_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to save usage snapshot to %s", path, exc_info=True)


def _provider_has_error(payload: dict[str, Any], key: str) -> bool:
    provider = payload.get(key)
    return not isinstance(provider, dict) or bool(provider.get("error"))


def _merge_usage_snapshot(live_payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = _load_usage_snapshot()
    if not snapshot:
        return live_payload

    used: list[str] = []
    merged = dict(live_payload)
    for provider_key in ("claude", "openai", "nanogpt"):
        if not _provider_has_error(merged, provider_key):
            continue
        snapshot_provider = snapshot.get(provider_key)
        if not isinstance(snapshot_provider, dict) or snapshot_provider.get("error"):
            continue
        merged[provider_key] = snapshot_provider
        used.append(provider_key)

    if used:
        merged["snapshot_used"] = used
        merged["snapshot_path"] = str(_usage_snapshot_path())
        merged["snapshot_cached_at"] = snapshot.get("cached_at")
    return merged


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
    for base in [home, home / "OneDrive"]:
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
        body = json.dumps(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _ANTHROPIC_CLIENT_ID,
            }
        ).encode("utf-8")
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
    login_cmd = f"{executable} /login" if executable else "claude login"
    if not executable:
        return (
            {
                "success": False,
                "message": f"Claude Code CLI not found. Install it, then run '{login_cmd}' in CMD.",
                "manual_command": login_cmd,
            },
            400,
        )

    best_path, best_token, best_refresh, best_expires = _select_best_claude_credential()
    if not best_path or not best_token:
        launched = _launch_claude_login_terminal(executable)
        return (
            {
                "success": launched,
                "message": f"Opened a CMD window for '{login_cmd}'."
                if launched
                else f"No Claude credentials found. Run '{login_cmd}' in CMD.",
                "manual_command": login_cmd,
                "executable": executable,
                "terminal_launched": launched,
            },
            200 if launched else 400,
        )

    now_ms = int(time.time() * 1000)
    if best_expires > now_ms:
        mins = max(0, round((best_expires - now_ms) / 1000 / 60))
        launched = _launch_claude_login_terminal(executable)
        return (
            {
                "success": True,
                "message": (
                    f"Claude token is already fresh (expires in ~{mins} min). "
                    f"Opened a CMD window for '{login_cmd}' anyway so you can force re-auth manually."
                    if launched
                    else f"Claude token is already fresh (expires in ~{mins} min). If usage still fails, it is likely a provider-side rate limit rather than expired auth."
                ),
                "file": str(best_path),
                "executable": executable,
                "terminal_launched": launched,
                "manual_command": login_cmd,
            },
            200,
        )

    if not best_refresh:
        launched = _launch_claude_login_terminal(executable)
        return (
            {
                "success": launched,
                "message": f"Opened a CMD window for '{login_cmd}'."
                if launched
                else f"Claude token is expired and no refresh token is available. Run '{login_cmd}' in CMD.",
                "manual_command": login_cmd,
                "file": str(best_path),
                "executable": executable,
                "terminal_launched": launched,
            },
            200 if launched else 400,
        )

    refreshed = _refresh_claude_token(best_path, best_refresh)
    _clear_usage_cache("claude")
    if refreshed:
        return (
            {
                "success": True,
                "message": "Claude token refresh succeeded.",
                "file": str(best_path),
                "executable": executable,
            },
            200,
        )

    launched = _launch_claude_login_terminal(executable)
    return (
        {
            "success": launched,
            "message": f"Claude token refresh failed, so a CMD window for '{login_cmd}' was opened."
            if launched
            else f"Claude token refresh failed. Run '{login_cmd}' in CMD.",
            "manual_command": login_cmd,
            "executable": executable,
            "terminal_launched": launched,
        },
        200 if launched else 400,
    )


def _relogin_openai() -> tuple[dict[str, Any], int]:
    """Start Codex browser login when needed, or report active login."""
    _clear_usage_cache("openai")
    if is_codex_login_available():
        return (
            {
                "success": True,
                "already_logged_in": True,
                "message": "Codex login is already available",
            },
            200,
        )

    payload = get_codex_device_login()
    _clear_usage_cache("openai")
    status_code = 200 if payload.get("ok") else 400
    return (
        {
            "success": bool(payload.get("ok")),
            **payload,
            "manual_command": "codex login",
        },
        status_code,
    )


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
            # Token might be stale — try refresh once before giving up
            if not skip_cache:
                best_path, _bt, best_refresh, _be = _select_best_claude_credential()
                if best_path and best_refresh:
                    refreshed = _refresh_claude_token(best_path, best_refresh)
                    if refreshed:
                        logger.info("Token refreshed after 429, retrying usage fetch")
                        return _fetch_claude_usage(skip_cache=True)
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
_OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


def _decode_jwt_payload(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        data = json.loads(decoded)
    except Exception:
        logger.debug("Failed to decode JWT payload", exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _read_codex_auth_data() -> tuple[Path | None, dict[str, Any] | None]:
    path = _find_credential_file(_CODEX_CRED_PATHS)
    if not path:
        return None, None
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        logger.debug("Failed to read Codex credentials from %s", path, exc_info=True)
        return path, None
    return path, data if isinstance(data, dict) else None


def _extract_codex_client_id(auth_data: dict[str, Any]) -> str | None:
    tokens = auth_data.get("tokens", {}) if isinstance(auth_data, dict) else {}
    for token_name in ("access_token", "id_token"):
        payload = _decode_jwt_payload(tokens.get(token_name))
        if not payload:
            continue
        client_id = payload.get("client_id")
        if isinstance(client_id, str) and client_id:
            return client_id
        aud = payload.get("aud")
        if isinstance(aud, list) and aud and isinstance(aud[0], str):
            return aud[0]
        if isinstance(aud, str) and aud:
            return aud
    return None


def _extract_codex_account_id(tokens: dict[str, Any]) -> str | None:
    for token_name in ("access_token", "id_token"):
        payload = _decode_jwt_payload(tokens.get(token_name))
        if not payload:
            continue
        auth_claim = payload.get("https://api.openai.com/auth")
        if not isinstance(auth_claim, dict):
            continue
        account_id = auth_claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    account_id = tokens.get("account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    return None


def _persist_codex_auth_data(path: Path, auth_data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(auth_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _refresh_codex_token(auth_path: Path, auth_data: dict[str, Any]) -> tuple[str | None, str | None]:
    tokens = auth_data.get("tokens", {}) if isinstance(auth_data, dict) else {}
    refresh_token = tokens.get("refresh_token")
    client_id = _extract_codex_client_id(auth_data)
    if not isinstance(refresh_token, str) or not refresh_token or not client_id:
        return None, None

    body = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _OPENAI_OAUTH_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.warning("Failed to refresh Codex token", exc_info=True)
        return None, None

    access_token = raw.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return None, None

    if not isinstance(tokens, dict):
        tokens = {}
        auth_data["tokens"] = tokens
    tokens["access_token"] = access_token
    if isinstance(raw.get("id_token"), str) and raw["id_token"]:
        tokens["id_token"] = raw["id_token"]
    if isinstance(raw.get("refresh_token"), str) and raw["refresh_token"]:
        tokens["refresh_token"] = raw["refresh_token"]
    account_id = _extract_codex_account_id(tokens)
    if account_id:
        tokens["account_id"] = account_id
    auth_data["last_refresh"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    try:
        _persist_codex_auth_data(auth_path, auth_data)
    except Exception:
        logger.warning("Failed to persist refreshed Codex auth to %s", auth_path, exc_info=True)

    return access_token, account_id


def _read_codex_credentials() -> tuple[str | None, str | None]:
    """Return (access_token, account_id) from Codex auth file."""
    _path, data = _read_codex_auth_data()
    if not data:
        return None, None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        return None, None
    return tokens.get("access_token"), _extract_codex_account_id(tokens)


def _window_label(seconds: int) -> str:
    """Convert limit_window_seconds to a human label like '5h' or 'Week'."""
    hours = seconds / 3600
    if hours <= 24:
        return f"{hours:.0f}h"
    days = hours / 24
    if days >= 6:
        return "Week"
    return f"{days:.0f}d"


def _fetch_openai_usage(skip_cache: bool = False, allow_refresh: bool = True) -> dict[str, Any]:
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
            if allow_refresh:
                auth_path, auth_data = _read_codex_auth_data()
                if auth_path and auth_data:
                    refreshed_token, _account_id = _refresh_codex_token(auth_path, auth_data)
                    if refreshed_token:
                        logger.info("Codex token refreshed after %s, retrying usage fetch", e.code)
                        return _fetch_openai_usage(skip_cache=True, allow_refresh=False)
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


# ── nanoGPT (subscription usage) ────────────────────────────────────────────

_NANOGPT_USAGE_URL = "https://nano-gpt.com/api/subscription/v1/usage"


def _read_nanogpt_api_key() -> str | None:
    """Read nanoGPT API key from AnimaWorks config credentials."""
    try:
        from core.config.models import load_config

        config = load_config()
        cred = config.credentials.get("nanogpt")
        if cred and cred.api_key:
            return cred.api_key
    except Exception:
        logger.debug("Failed to read nanoGPT credentials", exc_info=True)
    return None


def _fetch_nanogpt_usage(skip_cache: bool = False) -> dict[str, Any]:
    if not skip_cache:
        cached = _cached("nanogpt")
        if cached is not None:
            return cached

    api_key = _read_nanogpt_api_key()
    if not api_key:
        result: dict[str, Any] = {"error": "no_credentials", "message": "nanoGPT credentials not found"}
        _set_cache("nanogpt", result)
        return result

    try:
        req = urllib.request.Request(
            _NANOGPT_USAGE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "animaworks/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        result: dict[str, Any] = {"provider": "nanogpt"}

        # Daily images window
        daily_img = raw.get("dailyImages")
        if daily_img and isinstance(daily_img, dict):
            used_pct = daily_img.get("percentUsed", 0) * 100
            reset_at = daily_img.get("resetAt")
            result["Images"] = {
                "utilization": used_pct,
                "remaining": 100 - used_pct,
                "resets_at": reset_at / 1000 if reset_at and reset_at > 1e12 else reset_at,
                "window_seconds": 86400,
                "used_count": daily_img.get("used", 0),
                "limit_count": daily_img.get("used", 0) + daily_img.get("remaining", 0),
            }

        # Weekly input tokens window
        weekly = raw.get("weeklyInputTokens")
        if weekly and isinstance(weekly, dict):
            used_pct = weekly.get("percentUsed", 0) * 100  # API returns 0-1 fraction
            reset_at = weekly.get("resetAt")  # epoch ms
            # Window is 7 days
            result["Week"] = {
                "utilization": used_pct,
                "remaining": 100 - used_pct,
                "resets_at": reset_at / 1000 if reset_at and reset_at > 1e12 else reset_at,
                "window_seconds": 604800,
                "used_tokens": weekly.get("used", 0),
                "limit_tokens": weekly.get("used", 0) + weekly.get("remaining", 0),
            }

        # Subscription state
        result["state"] = raw.get("state", "unknown")

        _set_cache("nanogpt", result)
        return result

    except urllib.error.HTTPError as e:
        if e.code == 401:
            result = {"error": "unauthorized", "message": "nanoGPT API key invalid"}
        elif e.code == 429:
            return {"error": "rate_limited", "message": "Rate limited — retry shortly"}
        else:
            result = {"error": "http_error", "message": f"HTTP {e.code}"}
        _set_cache("nanogpt", result)
        return result
    except Exception as e:
        logger.warning("nanoGPT usage fetch failed: %s", e)
        return {"error": "fetch_failed", "message": str(e)[:200]}


# ── Route ────────────────────────────────────────────────────────────────────


def create_usage_router() -> APIRouter:
    router = APIRouter()

    @router.get("/usage")
    async def get_usage(request: Request, skip_cache: bool = False) -> dict[str, Any]:
        """Return combined Claude + OpenAI + nanoGPT usage data + governor status."""
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
        payload = {
            "claude": _fetch_claude_usage(skip_cache=skip_cache),
            "openai": _fetch_openai_usage(skip_cache=skip_cache),
            "nanogpt": _fetch_nanogpt_usage(skip_cache=skip_cache),
            "cached_at": time.time(),
            "governor": governor_info,
        }
        payload = _merge_usage_snapshot(payload)
        payload["snapshot_path"] = str(_usage_snapshot_path())
        if not skip_cache:
            _save_usage_snapshot(payload)
        return payload

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
