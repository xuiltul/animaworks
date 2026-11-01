from __future__ import annotations

from core.platform.env import data_dir_env

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Messaging, human notification, and recent tool section building."""

import logging
from pathlib import Path

from core.i18n import t
from core.paths import PROJECT_DIR, load_prompt
from core.prompt.org_context import _is_mcp_mode
from core.prompt.sections import _load_fallback_strings

logger = logging.getLogger("animaworks.prompt_builder")


def _resolve_shared_dir_for_prompt(anima_dir: Path) -> Path | None:
    """Resolve the shared runtime directory without leaking to unrelated data."""
    candidates: list[Path] = []

    if anima_dir.parent.name == "animas":
        candidates.append(anima_dir.parent.parent / "shared")

    candidates.append(anima_dir.parent / "shared")

    if data_dir_env():
        from core.paths import get_shared_dir

        candidates.append(get_shared_dir())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue

    return None


def _collect_accessible_board_channels(anima_dir: Path) -> tuple[list[str], list[str]]:
    """Return accessible restricted/open board channels for the current Anima."""
    shared_dir = _resolve_shared_dir_for_prompt(anima_dir)
    if shared_dir is None:
        return [], []

    channels_dir = shared_dir / "channels"
    if not channels_dir.is_dir():
        return [], []

    from core.messaging.messenger import is_channel_member, load_channel_meta

    restricted: list[str] = []
    open_channels: list[str] = []

    for channel_file in sorted(channels_dir.glob("*.jsonl")):
        channel_name = channel_file.stem
        if not is_channel_member(shared_dir, channel_name, anima_dir.name):
            continue
        meta = load_channel_meta(shared_dir, channel_name)
        if meta is not None and meta.members:
            restricted.append(channel_name)
        else:
            open_channels.append(channel_name)

    return restricted, open_channels


def _get_prompt_locale() -> str:
    try:
        from core.config.models import load_config
        from core.paths import normalize_locale

        locale = normalize_locale(load_config().locale)
        if locale in {"ja", "en", "ko", "zh"}:
            return locale
    except Exception:
        logger.debug("Failed to detect locale, defaulting to ja", exc_info=True)
    return "ja"


def _build_board_channel_guidance(anima_dir: Path) -> str:
    """Build dynamic board-channel guidance for work/completion reports."""
    restricted, open_channels = _collect_accessible_board_channels(anima_dir)
    team_channels = [ch for ch in restricted if ch not in {"general", "ops"}]

    ordered_channels = team_channels + [ch for ch in restricted if ch not in team_channels]
    for fallback in ("general", "ops"):
        if fallback in open_channels:
            ordered_channels.append(fallback)
    ordered_channels.extend(ch for ch in open_channels if ch not in {"general", "ops"})

    visible_names = ", ".join(f"#{name}" for name in ordered_channels[:5])
    locale = _get_prompt_locale()
    if len(ordered_channels) > 5:
        visible_names += t(
            "builder.board_channels_more",
            locale=locale,
            count=len(ordered_channels) - 5,
        )
    visible_channels = visible_names or t("builder.board_channels_none", locale=locale)
    preferred_channels = ", ".join(f"#{name}" for name in team_channels)

    key = "builder.board_guidance_with_team" if team_channels else "builder.board_guidance_without_team"
    return t(
        key,
        locale=locale,
        team_channels=preferred_channels,
        visible_channels=visible_channels,
    )


def _build_messaging_section(
    anima_dir: Path,
    other_animas: list[str],
    execution_mode: str = "s",
) -> str:
    """Build the messaging instructions with resolved paths."""
    _fs = _load_fallback_strings()
    self_name = anima_dir.name
    main_py = PROJECT_DIR / "main.py"
    animas_line = ", ".join(other_animas) if other_animas else _fs.get("no_other_animas", "(no other employees yet)")

    prompt_key = "messaging_s" if _is_mcp_mode(execution_mode) else "messaging"
    return load_prompt(
        prompt_key,
        animas_line=animas_line,
        board_channel_guidance=_build_board_channel_guidance(anima_dir),
        main_py=main_py,
        self_name=self_name,
    )


def _build_human_notification_guidance(execution_mode: str = "") -> str:
    """Build concise human-notification guidance for top-level Animas."""
    return load_prompt("builder/human_notification").strip()
