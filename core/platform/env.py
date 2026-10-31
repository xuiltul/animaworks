from __future__ import annotations

"""Accessors for AnimaWorks process environment configuration.

Every accessor reads the current process environment on each call so tests,
subprocess setup, and runtime overrides observe ``monkeypatch.setenv`` changes.
"""

import os
from pathlib import Path

ANIMA_DIR_ENV = "ANIMAWORKS_ANIMA_DIR"
DATA_DIR_ENV = "ANIMAWORKS_DATA_DIR"
SERVER_URL_ENV = "ANIMAWORKS_SERVER_URL"
ANIMAWORKS_ENV_PREFIX = "ANIMAWORKS_"
DEFAULT_SERVER_URL = "http://localhost:18500"


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an environment value without caching it."""
    return os.environ.get(name, default)


def has_env(name: str) -> bool:
    """Return whether an environment variable is currently present."""
    return name in os.environ


def env_items_with_prefix(prefix: str, *, suffix: str = "") -> dict[str, str]:
    """Return a fresh snapshot of variables matching *prefix* and optional *suffix*."""
    return {
        name: value
        for name, value in os.environ.items()
        if name.startswith(prefix) and (not suffix or name.endswith(suffix))
    }


def read_process_env(pid: int, name: str) -> tuple[bool, str | None]:
    """Read one variable from a process environment without decoding secrets."""
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return False, None
    prefix = f"{name}=".encode()
    for entry in raw.split(b"\0"):
        if entry.startswith(prefix):
            return True, os.fsdecode(entry[len(prefix) :])
    return True, None


def anima_dir_env() -> str | None:
    """Return the current raw per-Anima directory override, if set."""
    return get_env(ANIMA_DIR_ENV)


def data_dir_env() -> str | None:
    """Return the current raw runtime data directory override, if set."""
    return get_env(DATA_DIR_ENV)


def server_url(*, default: str | None = None) -> str:
    """Resolve the process-local server base URL from the environment.

    The server launcher sets the configured listening port before starting
    child processes. ``default=""`` preserves callers that intentionally omit
    internal asset URLs unless an explicit server URL is configured.
    """
    configured = get_env(SERVER_URL_ENV, "") or ""
    if configured.strip():
        return configured.strip().rstrip("/")
    if default == "":
        return ""

    return default.rstrip("/") if default else DEFAULT_SERVER_URL


def server_port() -> int:
    """Return the port of :func:`server_url` (``ANIMAWORKS_SERVER_URL`` or the default)."""
    from urllib.parse import urlsplit

    return urlsplit(server_url()).port or urlsplit(DEFAULT_SERVER_URL).port or 18500


def server_url_env() -> dict[str, str]:
    """Return the server URL variable for an explicitly configured child env."""
    return {SERVER_URL_ENV: server_url()}


def set_server_url(value: str) -> None:
    """Set the process-local server URL, normalizing trailing slashes."""
    os.environ[SERVER_URL_ENV] = value.strip().rstrip("/")
