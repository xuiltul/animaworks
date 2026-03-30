from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Auto-manage per-Anima Slack token slots in the .env file.

When a new Anima is registered, this module appends placeholder
lines for ``SLACK_BOT_TOKEN__{name}`` and ``SLACK_APP_TOKEN__{name}``
to the project ``.env`` file so the user only needs to fill in the
values.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("animaworks.config.env_slots")

_SLOT_KEYS = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN")


def _find_env_file() -> Path | None:
    """Locate the .env file (project root or cwd)."""
    # Try project root (one level up from this file's package)
    candidates = [
        Path(__file__).resolve().parents[2] / ".env",  # repo root
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _env_file_or_create() -> Path:
    """Return the .env path, creating an empty one at repo root if needed."""
    existing = _find_env_file()
    if existing:
        return existing
    # Default: repo root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    env_path.touch()
    return env_path


def ensure_slack_env_slots(anima_name: str) -> bool:
    """Append Slack token placeholders for *anima_name* to ``.env``.

    Returns True if new lines were appended, False if they already exist.
    """
    env_path = _env_file_or_create()
    content = env_path.read_text(encoding="utf-8")

    bot_key = f"SLACK_BOT_TOKEN__{anima_name}"
    if bot_key in content:
        return False

    lines = [
        "",
        f"# ── Slack bot for {anima_name} ──",
        f"# {bot_key}=xoxb-...",
        f"# SLACK_APP_TOKEN__{anima_name}=xapp-...",
    ]
    with env_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(
        "[%s] Slack token slots added to .env — edit the file to configure: %s",
        anima_name,
        bot_key,
    )
    return True


def check_missing_slack_tokens() -> list[str]:
    """Return Anima names whose Slack bot tokens are not set.

    Checks registered animas in config against environment variables,
    vault, and shared credentials.  Returns names where no usable
    ``SLACK_BOT_TOKEN__{name}`` value was found.
    """
    try:
        from core.config.models import load_config

        config = load_config()
    except Exception:
        return []

    from core.tools._base import _lookup_shared_credentials, _lookup_vault_credential

    missing: list[str] = []
    for anima_name in sorted(config.animas):
        key = f"SLACK_BOT_TOKEN__{anima_name}"
        token = (
            _lookup_vault_credential(key)
            or _lookup_shared_credentials(key)
            or os.environ.get(key)
        )
        if not token:
            missing.append(anima_name)
    return missing


def ensure_all_anima_slots() -> int:
    """Ensure .env has Slack token slots for all registered Animas.

    Returns the number of newly added slots.
    """
    try:
        from core.config.models import load_config

        config = load_config()
    except Exception:
        return 0

    added = 0
    for anima_name in sorted(config.animas):
        if ensure_slack_env_slots(anima_name):
            added += 1
    return added
