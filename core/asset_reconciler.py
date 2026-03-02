from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Asset reconciliation — detect and generate missing Anima assets.

Primary mechanism: Anima bootstraps generate their own assets.
This module provides a fallback that runs at server startup and
periodically via the reconciliation loop, generating any missing
assets using the ImageGenPipeline with ``skip_existing=True``.
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, cast

from typing import Literal

from core.exceptions import AnimaWorksError  # noqa: F401
from core.i18n import t
from core.paths import load_prompt

logger = logging.getLogger("animaworks.asset_reconciler")

# Per-anima locks to prevent concurrent generation (bootstrap vs fallback).
_anima_locks: dict[str, asyncio.Lock] = {}

# Required base assets.  If *any* of these are missing the anima
# is considered to have incomplete assets.
REQUIRED_ASSETS: dict[str, str] = {
    "avatar_fullbody": "avatar_fullbody.png",
    "avatar_bustup": "avatar_bustup.png",
    "avatar_chibi": "avatar_chibi.png",
    "model_chibi": "avatar_chibi.glb",
    "model_rigged": "avatar_chibi_rigged.glb",
}

REALISTIC_REQUIRED_ASSETS: dict[str, str] = {
    "avatar_fullbody_realistic": "avatar_fullbody_realistic.png",
    "avatar_bustup_realistic": "avatar_bustup_realistic.png",
}

_3D_ASSET_KEYS = frozenset({"model_chibi", "model_rigged"})

# ── Failure cooldown ──────────────────────────────────────────────
# Tracks per-anima asset generation failures to avoid retrying
# non-transient errors (e.g. Meshy 402 Payment Required) every 30s.

_COOLDOWN_SECONDS = 3600.0  # 1 hour cooldown after non-transient failure

# {anima_name: (failure_time, error_message)}
_failure_cooldowns: dict[str, tuple[float, str]] = {}


def _is_in_cooldown(anima_name: str) -> tuple[bool, str]:
    """Check if an anima is in failure cooldown.

    Returns (is_cooled_down, reason_message).
    """
    entry = _failure_cooldowns.get(anima_name)
    if entry is None:
        return False, ""
    fail_time, error_msg = entry
    elapsed = time.monotonic() - fail_time
    if elapsed >= _COOLDOWN_SECONDS:
        del _failure_cooldowns[anima_name]
        return False, ""
    remaining = int(_COOLDOWN_SECONDS - elapsed)
    return True, f"cooldown ({remaining}s remaining): {error_msg}"


def _record_failure(anima_name: str, error: str) -> None:
    """Record a non-transient failure for cooldown tracking."""
    _failure_cooldowns[anima_name] = (time.monotonic(), error)
    logger.warning(
        "Asset generation for %s entered cooldown (%ds): %s",
        anima_name,
        int(_COOLDOWN_SECONDS),
        error,
    )


def _is_non_transient_error(error_str: str) -> bool:
    """Detect non-transient errors that should trigger cooldown."""
    non_transient_patterns = [
        "402",                    # Payment Required (Meshy credits)
        "401",                    # Unauthorized (bad API key)
        "403",                    # Forbidden
        "ToolConfigError",        # Missing API key
        "No image generation API key",
        "MESHY_API_KEY",
        "NOVELAI_TOKEN",
        "FAL_KEY",
    ]
    return any(p in error_str for p in non_transient_patterns)


def _get_lock(anima_name: str) -> asyncio.Lock:
    """Return (or create) the per-anima generation lock."""
    if anima_name not in _anima_locks:
        _anima_locks[anima_name] = asyncio.Lock()
    return _anima_locks[anima_name]


# ── Asset checking ────────────────────────────────────────────────


def _required_assets_for_style(
    image_style: Literal["anime", "realistic"],
    enable_3d: bool = True,
) -> dict[str, str]:
    """Return the required assets map for the given style."""
    if image_style == "realistic":
        return dict(REALISTIC_REQUIRED_ASSETS)
    assets = dict(REQUIRED_ASSETS)
    if not enable_3d:
        for k in _3D_ASSET_KEYS:
            assets.pop(k, None)
    return assets


def check_anima_assets(
    anima_dir: Path,
    *,
    enable_3d: bool = True,
    image_style: Literal["anime", "realistic"] = "realistic",
) -> dict[str, Any]:
    """Check an anima's asset completeness using metadata logic.

    Args:
        anima_dir: Path to the anima's runtime directory.
        enable_3d: Whether 3D assets are required.
        image_style: Which style's assets to check.

    Returns a dict with:
      - ``complete`` (bool): True if all required assets exist.
      - ``missing`` (list[str]): Keys of missing required assets.
      - ``present`` (list[str]): Keys of present required assets.
      - ``has_assets_dir`` (bool): Whether the assets/ directory exists.
    """
    assets_dir = anima_dir / "assets"
    has_dir = assets_dir.exists()

    missing: list[str] = []
    present: list[str] = []

    required = _required_assets_for_style(image_style, enable_3d)
    for key, filename in required.items():
        path = assets_dir / filename
        if has_dir and path.exists() and path.is_file():
            present.append(key)
        else:
            missing.append(key)

    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "present": present,
        "has_assets_dir": has_dir,
    }


def find_animas_with_missing_assets(
    animas_dir: Path,
    *,
    enable_3d: bool = True,
    image_style: Literal["anime", "realistic"] = "realistic",
) -> list[tuple[str, dict[str, Any]]]:
    """Scan all anima directories and return those with incomplete assets.

    Returns list of ``(anima_name, check_result)`` tuples.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    if not animas_dir.exists():
        return results

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        if not (anima_dir / "identity.md").exists():
            continue
        result = check_anima_assets(
            anima_dir, enable_3d=enable_3d, image_style=image_style,
        )
        if not result["complete"]:
            results.append((anima_dir.name, result))

    return results


# ── Asset generation (fallback) ───────────────────────────────────


async def reconcile_anima_assets(
    anima_dir: Path,
    *,
    prompt: str | None = None,
    enable_3d: bool = True,
    image_style: Literal["anime", "realistic"] = "realistic",
) -> dict[str, Any]:
    """Generate missing assets for a single anima (non-blocking).

    Acquires the per-anima lock so bootstrap and fallback cannot run
    concurrently.  Uses ``skip_existing=True`` for differential
    generation.

    Args:
        anima_dir: Path to the anima's runtime directory.
        prompt: Character prompt for image generation.  If ``None``,
            attempts to extract from identity.md.
        enable_3d: Whether to include 3D model generation steps.
        image_style: Which image style to generate assets for.

    Returns:
        Dict with generation results or skip reason.
    """
    anima_name = anima_dir.name
    lock = _get_lock(anima_name)

    # Check failure cooldown before acquiring lock
    in_cooldown, cooldown_reason = _is_in_cooldown(anima_name)
    if in_cooldown:
        logger.debug(
            "Skipping asset generation for %s: %s", anima_name, cooldown_reason,
        )
        return {"anima": anima_name, "skipped": True, "reason": cooldown_reason}

    if lock.locked():
        logger.info(
            "Asset generation already in progress for %s, skipping",
            anima_name,
        )
        return {"anima": anima_name, "skipped": True, "reason": "locked"}

    async with lock:
        check = check_anima_assets(
            anima_dir, enable_3d=enable_3d, image_style=image_style,
        )
        if check["complete"]:
            logger.debug("Assets complete for %s (post-lock check)", anima_name)
            return {"anima": anima_name, "skipped": True, "reason": "complete"}

        logger.info(
            "Generating missing %s assets for %s (missing: %s)",
            image_style,
            anima_name,
            check["missing"],
        )

        resolved_prompt = prompt or await _extract_prompt(anima_dir, style=image_style)
        if not resolved_prompt:
            logger.warning(
                "No prompt available for %s — cannot generate assets",
                anima_name,
            )
            return {
                "anima": anima_name,
                "skipped": True,
                "reason": "no_prompt",
            }

        steps: list[str] | None = None
        if image_style == "anime" and not enable_3d:
            steps = ["fullbody", "bustup", "chibi"]

        try:
            from core.config.models import ImageGenConfig
            from core.tools.image_gen import ImageGenPipeline

            image_config = ImageGenConfig(image_style=image_style, enable_3d=enable_3d)
            pipeline = ImageGenPipeline(anima_dir, config=image_config)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.generate_all(
                    prompt=resolved_prompt,
                    skip_existing=True,
                    steps=steps,
                ),
            )
            generated = _summarise_result(result)
            logger.info(
                "Asset generation complete for %s: %s",
                anima_name,
                generated,
            )

            # Check for non-transient errors and enter cooldown
            for err in result.errors:
                if _is_non_transient_error(err):
                    _record_failure(anima_name, err)
                    break

            return {
                "anima": anima_name,
                "skipped": False,
                "generated": generated,
                "errors": result.errors,
            }
        except Exception as exc:
            error_str = str(exc)
            if _is_non_transient_error(error_str):
                _record_failure(anima_name, error_str)
            else:
                logger.exception("Asset generation failed for %s", anima_name)
            return {
                "anima": anima_name,
                "skipped": False,
                "error": error_str,
            }


async def reconcile_all_assets(
    animas_dir: Path,
    *,
    ws_manager: Any | None = None,
    enable_3d: bool = True,
    image_style: Literal["anime", "realistic"] = "realistic",
) -> list[dict[str, Any]]:
    """Check all animas and generate missing assets sequentially.

    Args:
        animas_dir: Root animas directory.
        ws_manager: Optional WebSocketManager for broadcasting updates.
        enable_3d: Whether to include 3D model generation.
        image_style: Which image style to generate assets for.

    Returns:
        List of per-anima result dicts.
    """
    incomplete = find_animas_with_missing_assets(
        animas_dir, enable_3d=enable_3d, image_style=image_style,
    )
    if not incomplete:
        logger.debug("All animas have complete assets")
        return []

    logger.info(
        "Asset reconciliation: %d anima(s) with missing %s assets: %s",
        len(incomplete),
        image_style,
        [name for name, _ in incomplete],
    )

    results: list[dict[str, Any]] = []
    for anima_name, _check in incomplete:
        anima_dir = animas_dir / anima_name
        result = await reconcile_anima_assets(
            anima_dir, enable_3d=enable_3d, image_style=image_style,
        )
        results.append(result)

        # Broadcast asset update if something was generated
        if ws_manager and not result.get("skipped"):
            try:
                await ws_manager.broadcast(
                    "anima.assets_updated",
                    {"name": anima_name, "source": "reconciliation"},
                )
            except Exception:
                logger.debug(
                    "Failed to broadcast asset update for %s", anima_name,
                )

    return results


# ── Helpers ───────────────────────────────────────────────────────


def _resolve_prompt(anima_dir: Path, style: str) -> str | None:
    """Resolve the cached prompt for the given style.

    For ``"realistic"`` style, checks ``assets/prompt_realistic.txt`` first.
    Falls back to converting the anime prompt via tag replacement.
    For ``"anime"`` (default), reads ``assets/prompt.txt``.

    Returns ``None`` if no prompt is available.
    """
    assets_dir = anima_dir / "assets"
    if style == "realistic":
        realistic_path = assets_dir / "prompt_realistic.txt"
        if realistic_path.exists():
            text = realistic_path.read_text(encoding="utf-8").strip()
            if text:
                return text
        anime_path = assets_dir / "prompt.txt"
        if anime_path.exists():
            anime_text = anime_path.read_text(encoding="utf-8").strip()
            if anime_text:
                from core.tools._image_clients import _convert_anime_to_realistic
                return _convert_anime_to_realistic(anime_text)
        return None

    prompt_path = assets_dir / "prompt.txt"
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


async def _extract_prompt(
    anima_dir: Path,
    style: str = "anime",
) -> str | None:
    """Try to extract a generation prompt from the anima's files.

    Args:
        anima_dir: Anima runtime directory.
        style: ``"anime"`` or ``"realistic"``.

    Fallback chain:
      1. Regex match for explicit ``image_prompt:`` / ``外見:`` fields
         (searches identity.md, then character_sheet.md)
      2. Cached ``assets/prompt.txt`` (or ``prompt_realistic.txt``)
         from a previous LLM synthesis
      3. LLM synthesis — pass the full character document to LLM which
         extracts visual appearance and converts to tags.
         Tries character_sheet.md first, then identity.md.
    """
    # Collect candidate texts: identity.md first, then character_sheet.md
    candidates: list[str] = []
    for filename in ("identity.md", "character_sheet.md"):
        path = anima_dir / filename
        if path.exists():
            candidates.append(path.read_text(encoding="utf-8"))

    if not candidates:
        return None

    # Step 1: Look for a dedicated image_prompt / appearance field
    patterns = [
        r"(?:image[_ ]?prompt|画像プロンプト|外見|appearance)\s*[:\uff1a]\s*(.+)",
        r"(?:キャラクターデザイン|character[_ ]?design)\s*[:\uff1a]\s*(.+)",
    ]
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                if style == "realistic":
                    from core.tools._image_clients import _convert_anime_to_realistic
                    return _convert_anime_to_realistic(raw)
                return raw

    # Step 2: Check cached prompt for the requested style
    cached = _resolve_prompt(anima_dir, style)
    if cached:
        logger.debug(
            "Using cached %s prompt for %s",
            style,
            anima_dir.name,
        )
        return cached

    # Step 3: Pass the richest available document to LLM for synthesis.
    # Prefer character_sheet.md (has appearance info), fall back to identity.md.
    for text in reversed(candidates):
        result = await _synthesize_prompt_via_llm(anima_dir, text, style=style)
        if result:
            return result

    return None


# ── LLM prompt synthesis ──────────────────────────────────────


async def _synthesize_prompt_via_llm(
    anima_dir: Path,
    character_text: str,
    *,
    style: str = "anime",
) -> str | None:
    """Call LLM to extract appearance from a character sheet into image tags.

    For ``"anime"`` style, generates NovelAI Danbooru tags.
    For ``"realistic"`` style, generates natural-language photographic prompts.

    On success the result is cached to ``assets/prompt.txt`` (anime) or
    ``assets/prompt_realistic.txt`` (realistic).
    On failure returns ``None`` (logs the error, does not raise).
    """
    anima_name = anima_dir.name

    try:
        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)
    except Exception:
        logger.warning(
            "Cannot load model config for %s — skipping LLM synthesis",
            anima_name,
        )
        return None

    if style == "realistic":
        system_prompt_name = "fragments/asset_synthesis_system_realistic"
        user_prompt_key = "asset_reconciler.llm_user_prompt_realistic"
    else:
        system_prompt_name = "fragments/asset_synthesis_system"
        user_prompt_key = "asset_reconciler.llm_user_prompt"

    api_key = model_config.api_key or os.environ.get(model_config.api_key_env)
    kwargs: dict[str, Any] = {
        "model": model_config.model,
        "messages": [
            {"role": "system", "content": load_prompt(system_prompt_name)},
            {
                "role": "user",
                "content": t(user_prompt_key, character_text=character_text),
            },
        ],
        "max_tokens": 256,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if model_config.api_base_url:
        kwargs["api_base"] = model_config.api_base_url

    try:
        import litellm

        response = cast(Any, await litellm.acompletion(**kwargs))
        result = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(
            "LLM prompt synthesis failed for %s (%s): %s",
            anima_name,
            style,
            exc,
        )
        return None

    if not result or result == "NO_APPEARANCE_DATA":
        return None

    cache_filename = "prompt_realistic.txt" if style == "realistic" else "prompt.txt"
    try:
        cache_dir = anima_dir / "assets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / cache_filename).write_text(result + "\n", encoding="utf-8")
        logger.info(
            "Synthesized and cached %s prompt for %s: %.200s",
            style,
            anima_name,
            result,
        )
    except OSError:
        logger.debug("Failed to cache %s for %s", cache_filename, anima_name)

    return result


def _summarise_result(result: Any) -> dict[str, list[str]]:
    """Extract a human-readable summary from a PipelineResult."""
    generated: list[str] = []
    skipped: list[str] = list(result.skipped) if result.skipped else []

    if result.fullbody_path and "fullbody" not in skipped:
        generated.append("fullbody")
    if result.bustup_path and "bustup" not in skipped:
        generated.append("bustup")
    if result.chibi_path and "chibi" not in skipped:
        generated.append("chibi")
    if result.model_path and "3d" not in skipped:
        generated.append("3d")
    if result.rigged_model_path and "rigging" not in skipped:
        generated.append("rigging")
    for anim_name in result.animation_paths:
        if f"anim_{anim_name}" not in skipped:
            generated.append(f"anim_{anim_name}")

    return {"generated": generated, "skipped": skipped}
