from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Messaging, human notification, and recent tool section building."""

import logging
import os
from pathlib import Path
from typing import Any

from core.paths import PROJECT_DIR, load_prompt, load_prompt_text
from core.prompt.org_context import _is_mcp_mode
from core.prompt.sections import _load_fallback_strings, _load_section_strings

logger = logging.getLogger("animaworks.prompt_builder")


def _resolve_shared_dir_for_prompt(anima_dir: Path) -> Path | None:
    """Resolve the shared runtime directory without leaking to unrelated data."""
    candidates: list[Path] = []

    if anima_dir.parent.name == "animas":
        candidates.append(anima_dir.parent.parent / "shared")

    candidates.append(anima_dir.parent / "shared")

    if os.environ.get("ANIMAWORKS_DATA_DIR"):
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

    from core.messenger import is_channel_member, load_channel_meta

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

        locale = load_config().locale
        if locale in {"ja", "en", "ko"}:
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

    visible_channels = ", ".join(f"#{name}" for name in ordered_channels) if ordered_channels else "(none)"
    preferred_channels = ", ".join(f"#{name}" for name in team_channels) if team_channels else ""

    locale = _get_prompt_locale()
    if locale == "en":
        if preferred_channels:
            return (
                f"- Available Board channels for you: {visible_channels}\n"
                f"- Routine work reports and task completion updates should go to your restricted team channel(s) first: {preferred_channels}\n"
                "- Use `general` only for org-wide sharing, and `ops` only for cross-team operations/infrastructure updates."
            )
        return (
            f"- Available Board channels for you: {visible_channels}\n"
            "- No restricted team channel is currently available to you, so use the most relevant shared channel: `general` for org-wide sharing, `ops` for operations/infrastructure."
        )

    if locale == "ko":
        if preferred_channels:
            return (
                f"- 현재 접근 가능한 Board 채널: {visible_channels}\n"
                f"- 일반적인 작업 보고와 완료 보고는 제한된 팀/부서 채널을 우선 사용: {preferred_channels}\n"
                "- `general`은 전체 공유용, `ops`는 팀을 넘는 운영/인프라 공유용으로만 사용."
            )
        return (
            f"- 현재 접근 가능한 Board 채널: {visible_channels}\n"
            "- 현재 제한된 팀 채널이 없으므로 가장 관련 있는 공유 채널을 사용: `general`은 전체 공유, `ops`는 운영/인프라 공유."
        )

    if preferred_channels:
        return (
            f"- 現在アクセスできるBoardチャネル: {visible_channels}\n"
            f"- 通常の作業報告・完了報告は、所属部門/チームの限定チャネルを優先: {preferred_channels}\n"
            "- `general` は全体共有、`ops` は部門横断の運用・インフラ共有に限定して使う。"
        )
    return (
        f"- 現在アクセスできるBoardチャネル: {visible_channels}\n"
        "- 参加中の限定チャネルが見当たらないため、最も関連性の高い共有チャネルを使う: `general` は全体共有、`ops` は運用・インフラ共有。"
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


def _load_a_reflection() -> str:
    """Load the A mode reflection/retry prompt template."""
    try:
        return load_prompt_text("a_reflection")
    except Exception:
        logger.debug("a_reflection template not found, skipping")
        return ""


def _build_recent_tool_section(anima_dir: Path, model_config: Any) -> str:
    """Build a summary of recent tool results for system prompt injection.

    Reads the last few turns from ConversationMemory and extracts tool
    records with result summaries, constrained to a ~2000 token budget.
    """
    try:
        from core.memory.conversation import ConversationMemory

        conv_memory = ConversationMemory(anima_dir, model_config)
        state = conv_memory.load()
    except Exception:
        return ""
    if not state.turns:
        return ""

    tool_lines: list[str] = []
    budget_remaining = 2000  # approximate token budget (~8000 chars)
    for turn in reversed(state.turns[-3:]):
        for tr in turn.tool_records[:5]:
            if not tr.result_summary:
                continue
            line = f"- {tr.tool_name}: {tr.result_summary[:500]}"
            budget_remaining -= len(line) // 4
            if budget_remaining <= 0:
                break
            tool_lines.append(line)
        if budget_remaining <= 0:
            break

    if not tool_lines:
        return ""
    _ss = _load_section_strings()
    header = _ss.get("recent_tool_results_header", "## Recent Tool Results")
    return f"{header}\n\n" + "\n".join(tool_lines)


def _build_human_notification_guidance(execution_mode: str = "") -> str:
    """Build the human notification instruction for top-level Animas."""
    if _is_mcp_mode(execution_mode):
        how_to = load_prompt("builder/human_notification_howto_s")
    else:
        how_to = load_prompt("builder/human_notification_howto_other")

    return load_prompt("builder/human_notification", how_to=how_to)
