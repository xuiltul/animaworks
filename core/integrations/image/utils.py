# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Shared utilities for image/3D generation clients."""

from __future__ import annotations

import base64
import re
from typing import Any

import httpx

from core.integrations._retry import retry_with_backoff
from core.integrations.image.constants import _RETRYABLE_CODES


def _retry(
    fn: Any,
    *,
    max_retries: int = 2,
    delay: float = 5.0,
    retryable_codes: set[int] | None = None,
) -> Any:
    """Execute *fn* with simple retry on transient HTTP errors.

    Delegates to the shared :func:`retry_with_backoff` utility while
    preserving the original filtering logic for non-retryable HTTP
    status codes.
    """
    codes = retryable_codes or _RETRYABLE_CODES

    class _NonRetryableHTTPError(Exception):
        """Wrapper for HTTP errors with non-retryable status codes."""

    def _guarded() -> Any:
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in codes:
                raise _NonRetryableHTTPError from exc
            raise

    try:
        return retry_with_backoff(
            _guarded,
            max_retries=max_retries,
            base_delay=delay,
            max_delay=300.0,
            retry_on=(httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout),
        )
    except _NonRetryableHTTPError as exc:
        raise exc.__cause__ from None  # type: ignore[misc]


def _image_to_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a ``data:`` URI."""
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


# ── Anime → Realistic prompt conversion ──────────────────

_ANIME_QUALITY_TAGS = frozenset(
    {
        "masterpiece",
        "best quality",
        "very aesthetic",
        "absurdres",
        "anime coloring",
        "clean lineart",
        "soft shading",
        "highres",
        "extremely detailed",
    }
)

_LOCALE_ETHNICITY: dict[str, str] = {
    "ja": "Japanese",
    "en": "American",
    "ko": "Korean",
    "zh": "Chinese",
}

_DANBOORU_PERSON_TAGS = frozenset({"1girl", "1boy", "2girls", "2boys"})


# Anime-only hair colours mapped to shades a real person could plausibly have.
# Order matters: the first group whose keyword appears in the tag wins.
_NATURAL_HAIR_COLORS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("rose gold", "pink", "rose", "sakura", "peach"), "rose-brown"),
    (("wine", "red", "crimson", "scarlet", "burgundy"), "dark burgundy-brown"),
    (("orange", "copper", "ginger"), "copper-brown"),
    (("violet", "purple", "lavender", "lilac"), "dark plum-brown"),
    (("mint", "green", "teal", "emerald"), "dark brown-black with a subtle teal tint"),
    (("sky blue", "light blue", "pale blue", "silver", "white", "gray", "grey"), "light ash-gray"),
    (("aqua", "cyan", "blue", "navy"), "dark ash-blue black"),
)
# Kept when a hair colour is rewritten ("pink inner hair" -> "rose-brown inner hair").
_HAIR_STRUCTURE_WORDS = frozenset({"inner", "streaked", "gradient", "two-tone", "multicolored", "tips"})
_UNNATURAL_EYE_COLOR_RE = re.compile(
    r"\b(?:blue|cyan|aqua|green|teal|emerald|purple|violet|red|crimson|pink|gold|golden|amber|yellow|silver|gray|grey)\b"
)


def _naturalize_color_tag(tag: str) -> str:
    """Rewrite an anime-only hair/eye colour tag into a natural real-world shade."""
    lower = tag.lower()
    if lower.endswith(" hair"):
        words = lower[: -len(" hair")].split()
        structure = [w for w in words if w in _HAIR_STRUCTURE_WORDS]
        colour = " ".join(w for w in words if w not in _HAIR_STRUCTURE_WORDS)
        for keywords, natural in _NATURAL_HAIR_COLORS:
            if any(re.search(rf"\b{re.escape(k)}\b", colour) for k in keywords):
                return " ".join([natural, *structure, "hair"])
    elif lower.endswith(" eyes") and _UNNATURAL_EYE_COLOR_RE.search(lower):
        return "dark brown eyes"
    return tag


def _convert_anime_to_realistic(anime_prompt: str, locale: str | None = None) -> str:
    """Convert a Danbooru-style anime prompt to a photographic prompt.

    Strips anime quality/style tags, replaces Danbooru shorthands with
    natural English (with locale-based ethnicity), and prepends
    realistic quality descriptors.
    """
    if locale is None:
        try:
            from core.config.models import load_config

            locale = load_config().locale or "ja"
        except Exception:
            locale = "ja"

    ethnicity = _LOCALE_ETHNICITY.get(locale, "")

    person_map: dict[str, str] = {
        "1girl": f"a young {ethnicity} woman" if ethnicity else "a young woman",
        "1boy": f"a young {ethnicity} man" if ethnicity else "a young man",
        "2girls": f"two young {ethnicity} women" if ethnicity else "two young women",
        "2boys": f"two young {ethnicity} men" if ethnicity else "two young men",
    }

    tags = [t.strip() for t in anime_prompt.split(",") if t.strip()]

    converted: list[str] = []
    for tag in tags:
        lower = tag.lower().strip()
        if lower in _ANIME_QUALITY_TAGS:
            continue
        natural = person_map.get(lower)
        if natural:
            converted.append(natural)
        else:
            converted.append(_naturalize_color_tag(tag))

    realistic_prefix = [
        "professional photograph",
        "studio lighting",
        "high resolution",
        "realistic",
        "photorealistic",
    ]
    return ", ".join(realistic_prefix + converted)
