from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Canonical workspace file names for an Anima directory.

The standard names follow the de-facto convention used by OpenClaw and
Hermes Agent.  ``LEGACY_RENAMES`` is the single source of truth for the
startup migration (A2) and the explicit ``write_memory_file`` /
``read_memory_file`` redirects.
"""

SOUL_FILE = "SOUL.md"
IDENTITY_FILE = "IDENTITY.md"
AGENTS_FILE = "AGENTS.md"
HEARTBEAT_FILE = "HEARTBEAT.md"
BOOTSTRAP_FILE = "BOOTSTRAP.md"

LEGACY_IDENTITY = "identity.md"
LEGACY_RENAMES: dict[str, str] = {
    "injection.md": AGENTS_FILE,
    "heartbeat.md": HEARTBEAT_FILE,
    "bootstrap.md": BOOTSTRAP_FILE,
}

IDENTITY_SECTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "基本プロフィール",
        "プロフィール",
        "外見",
        "イメージカラー",
        "AIモデル情報",
        "Profile",
        "Basic Profile",
        "Appearance",
        "Image Color",
        "AI Model",
        "기본 프로필",
        "프로필",
        "외모",
        "이미지 컬러",
        "AI 모델 정보",
    }
)

_IDENTITY_HEADING_PREFIX = "## "
_SOUL_HEADING_PREFIX = "# Soul"


def is_identity_section(heading: str) -> bool:
    """Return whether an H2 heading belongs to IDENTITY.md."""
    folded = heading.casefold()
    return any(keyword.casefold() in folded for keyword in IDENTITY_SECTION_KEYWORDS)


def _title_name(title_line: str) -> str:
    """Extract the display name from an ``# Identity: X`` title line."""
    text = title_line.lstrip()[2:].strip()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    if text.casefold().startswith("identity"):
        remainder = text[len("identity") :].strip()
        return remainder
    return ""


def _soul_title(title_line: str) -> str:
    """Build the ``# Soul[: X]`` title preserving the original newline."""
    newline = "\r\n" if title_line.endswith("\r\n") else "\n"
    name = _title_name(title_line)
    return f"{_SOUL_HEADING_PREFIX}: {name}{newline}" if name else f"{_SOUL_HEADING_PREFIX}{newline}"


def _h2_indices(lines: list[str]) -> list[int]:
    indices: list[int] = []
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence and stripped.startswith(_IDENTITY_HEADING_PREFIX):
            indices.append(index)
    return indices


def split_legacy_identity(text: str) -> tuple[str, str]:
    """Split legacy ``identity.md`` content into ``(IDENTITY.md, SOUL.md)``.

    Profile-style H2 sections (name, appearance, image color, AI model, …)
    are placed in IDENTITY.md; every other section and the preamble are
    placed in SOUL.md.  The original section bodies, order and blank lines
    are preserved within each output.
    """
    if not text:
        return "", ""

    lines = text.splitlines(keepends=True)
    title_index = next((index for index, line in enumerate(lines) if line.lstrip().startswith("# ")), None)
    title_line = lines[title_index] if title_index is not None else ""
    body_start = title_index + 1 if title_index is not None else 0
    h2_indices = _h2_indices(lines)

    if not h2_indices:
        identity = title_line
        if title_index is None:
            return "", text
        soul_body = "".join(lines[body_start:])
        soul_prefix = _soul_title(title_line)
        if soul_body and not soul_body.startswith(("\n", "\r\n")):
            soul_prefix += "\n"
        return identity, soul_prefix + soul_body

    first_heading = h2_indices[0]
    preamble = lines[body_start:first_heading]
    identity_sections: list[str] = []
    soul_sections: list[str] = []
    for position, start in enumerate(h2_indices):
        end = h2_indices[position + 1] if position + 1 < len(h2_indices) else len(lines)
        section = "".join(lines[start:end])
        heading = lines[start].lstrip()[len(_IDENTITY_HEADING_PREFIX) :].strip()
        if is_identity_section(heading):
            identity_sections.append(section)
        else:
            soul_sections.append(section)

    preamble_is_blank = all(not line.strip() for line in preamble)
    newline = "\r\n" if title_line.endswith("\r\n") else "\n"
    identity_parts: list[str] = [title_line] if title_line else []
    if title_line and preamble_is_blank:
        if preamble:
            identity_parts.extend(preamble)
        elif identity_sections:
            identity_parts.append(newline)
    elif title_line and identity_sections:
        identity_parts.append(newline)
    identity_parts.extend(identity_sections)
    identity = "".join(identity_parts)

    soul_parts: list[str] = []
    if title_line:
        soul_parts.append(_soul_title(title_line))
    if preamble and not preamble_is_blank:
        if title_line and preamble[0] and not preamble[0].startswith(("\n", "\r\n")):
            soul_parts.append(newline)
        soul_parts.extend(preamble)
    elif title_line and soul_sections:
        soul_parts.append(newline)
    soul_parts.extend(soul_sections)
    soul = "".join(soul_parts)

    if not title_line and not identity_sections:
        identity = ""
    return identity, soul


__all__ = [
    "AGENTS_FILE",
    "BOOTSTRAP_FILE",
    "HEARTBEAT_FILE",
    "IDENTITY_FILE",
    "IDENTITY_SECTION_KEYWORDS",
    "LEGACY_IDENTITY",
    "LEGACY_RENAMES",
    "SOUL_FILE",
    "is_identity_section",
    "split_legacy_identity",
]
