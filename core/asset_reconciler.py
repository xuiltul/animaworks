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
from pathlib import Path
from typing import Any

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


def _get_lock(anima_name: str) -> asyncio.Lock:
    """Return (or create) the per-anima generation lock."""
    if anima_name not in _anima_locks:
        _anima_locks[anima_name] = asyncio.Lock()
    return _anima_locks[anima_name]


# ── Asset checking ────────────────────────────────────────────────


def check_anima_assets(anima_dir: Path) -> dict[str, Any]:
    """Check an anima's asset completeness using metadata logic.

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

    for key, filename in REQUIRED_ASSETS.items():
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
        result = check_anima_assets(anima_dir)
        if not result["complete"]:
            results.append((anima_dir.name, result))

    return results


# ── Asset generation (fallback) ───────────────────────────────────


async def reconcile_anima_assets(
    anima_dir: Path,
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Generate missing assets for a single anima (non-blocking).

    Acquires the per-anima lock so bootstrap and fallback cannot run
    concurrently.  Uses ``skip_existing=True`` for differential
    generation.

    Args:
        anima_dir: Path to the anima's runtime directory.
        prompt: Character prompt for image generation.  If ``None``,
            attempts to extract from identity.md.

    Returns:
        Dict with generation results or skip reason.
    """
    anima_name = anima_dir.name
    lock = _get_lock(anima_name)

    if lock.locked():
        logger.info(
            "Asset generation already in progress for %s, skipping",
            anima_name,
        )
        return {"anima": anima_name, "skipped": True, "reason": "locked"}

    async with lock:
        # Re-check after acquiring lock — another task may have generated
        check = check_anima_assets(anima_dir)
        if check["complete"]:
            logger.debug("Assets complete for %s (post-lock check)", anima_name)
            return {"anima": anima_name, "skipped": True, "reason": "complete"}

        logger.info(
            "Generating missing assets for %s (missing: %s)",
            anima_name,
            check["missing"],
        )

        resolved_prompt = prompt or await _extract_prompt(anima_dir)
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

        try:
            from core.tools.image_gen import ImageGenPipeline

            pipeline = ImageGenPipeline(anima_dir)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.generate_all(
                    prompt=resolved_prompt,
                    skip_existing=True,
                ),
            )
            generated = _summarise_result(result)
            logger.info(
                "Asset generation complete for %s: %s",
                anima_name,
                generated,
            )
            return {
                "anima": anima_name,
                "skipped": False,
                "generated": generated,
                "errors": result.errors,
            }
        except Exception as exc:
            logger.exception("Asset generation failed for %s", anima_name)
            return {
                "anima": anima_name,
                "skipped": False,
                "error": str(exc),
            }


async def reconcile_all_assets(
    animas_dir: Path,
    *,
    ws_manager: Any | None = None,
) -> list[dict[str, Any]]:
    """Check all animas and generate missing assets sequentially.

    Args:
        animas_dir: Root animas directory.
        ws_manager: Optional WebSocketManager for broadcasting updates.

    Returns:
        List of per-anima result dicts.
    """
    incomplete = find_animas_with_missing_assets(animas_dir)
    if not incomplete:
        logger.debug("All animas have complete assets")
        return []

    logger.info(
        "Asset reconciliation: %d anima(s) with missing assets: %s",
        len(incomplete),
        [name for name, _ in incomplete],
    )

    results: list[dict[str, Any]] = []
    for anima_name, _check in incomplete:
        anima_dir = animas_dir / anima_name
        result = await reconcile_anima_assets(anima_dir)
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


async def _extract_prompt(anima_dir: Path) -> str | None:
    """Try to extract a generation prompt from the anima's files.

    Fallback chain:
      1. Regex match for explicit ``image_prompt:`` / ``外見:`` fields
         (searches identity.md, then character_sheet.md)
      2. Cached ``assets/prompt.txt`` from a previous LLM synthesis
      3. LLM synthesis — pass the full character document to LLM which
         extracts visual appearance and converts to NovelAI tags.
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
                return match.group(1).strip()

    # Step 2: Check cached prompt from previous LLM synthesis
    prompt_cache = anima_dir / "assets" / "prompt.txt"
    if prompt_cache.exists():
        cached = prompt_cache.read_text(encoding="utf-8").strip()
        if cached:
            logger.debug(
                "Using cached prompt for %s from assets/prompt.txt",
                anima_dir.name,
            )
            return cached

    # Step 3: Pass the richest available document to LLM for synthesis.
    # Prefer character_sheet.md (has appearance info), fall back to identity.md.
    for text in reversed(candidates):
        result = await _synthesize_prompt_via_llm(anima_dir, text)
        if result:
            return result

    return None


# ── LLM prompt synthesis ──────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert at reading Japanese character sheets and converting \
visual appearance into high-quality NovelAI V4.5 image generation prompts.

## Image Generation Pipeline Reference

Target: NovelAI V4.5 (nai-diffusion-4-5-full), Danbooru tag system.
The generated prompt will be used as the base_caption in v4_prompt.
NovelAI's qualityToggle is enabled server-side, which auto-prepends \
additional quality boosters — but you MUST still include quality tags \
in your output for maximum effect (they stack, not conflict).
After full-body generation, the image is passed to Flux Kontext for \
bust-up and chibi variants, so the full-body pose/composition matters.

## Task

The input is a full character sheet in Markdown. It contains personality, \
hobbies, skills, backstory, and visual appearance mixed together. \
Extract ONLY the visual appearance and convert to Danbooru-style tags.

## Quality Tags (MANDATORY — always include first)

masterpiece, best quality, very aesthetic, absurdres, \
anime coloring, clean lineart, soft shading

These quality tags are critical for high-quality output. Never omit them.

## Tag Rules

- Output ONLY a comma-separated tag string, nothing else.
- Start with the quality tags above, then 1girl or 1boy.
- Use Danbooru tag conventions (lowercase, underscores optional).
- Use plain English color names, NOT gemstone/poetic metaphors \
  (サファイアブルー → blue eyes, エメラルドグリーン → green eyes, \
   ハニーブラウン → light brown, プラチナブロンド → platinum blonde).
- Decompose compound descriptions into atomic Danbooru tags \
  (ショートボブ、前髪ぱっつん → short hair, bob cut, blunt bangs; \
   ロングヘア、ツインテール → long hair, twintails).
- Translate accessories to Danbooru tags \
  (ピン → hair clip, リボン → hair ribbon, サイド留め → hair clip).
- Include body type cues when available \
  (petite, slender, medium breasts, etc.).
- Include eye shape/expression when described \
  (narrow eyes, round eyes, tareme, tsurime).
- Ignore all non-visual traits (personality, hobbies, skills, backstory).
- Height/weight: omit unless notably tall/short (use tall or petite).
- Always end with: full body, standing, white background, looking at viewer
- All tags lowercase, separated by comma + space.
- If the document contains no visual appearance information at all, \
output exactly: NO_APPEARANCE_DATA

## Examples

Input (excerpt):
- 髪型: 明るいボブカット。元気な印象のサイド留め
- 髪色: ハニーブラウン
- 瞳の色: ウォームブラウン
- 顔タイプ: 明るく親しみやすい可愛い系。くりっとした目、よく笑う
- 身長: 155cm

Output:
masterpiece, best quality, very aesthetic, absurdres, \
anime coloring, clean lineart, soft shading, \
1girl, light brown hair, short hair, bob cut, hair clip, \
brown eyes, round eyes, cute face, friendly expression, smile, petite, \
full body, standing, white background, looking at viewer

Input (excerpt):
- 髪型: ロングストレート、ローポニーテール
- 髪色: 黒
- 瞳の色: 赤
- 顔タイプ: クール系、切れ長の目、端正な顔立ち

Output:
masterpiece, best quality, very aesthetic, absurdres, \
anime coloring, clean lineart, soft shading, \
1girl, black hair, very long hair, straight hair, low ponytail, \
red eyes, narrow eyes, beautiful, elegant, refined features, \
full body, standing, white background, looking at viewer"""


async def _synthesize_prompt_via_llm(
    anima_dir: Path,
    character_text: str,
) -> str | None:
    """Call LLM to extract appearance from a character sheet into NovelAI tags.

    The LLM reads the full character document, picks out visual traits,
    and returns a comma-separated tag string.

    On success the result is cached to ``assets/prompt.txt``.
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

    api_key = model_config.api_key or os.environ.get(model_config.api_key_env)
    kwargs: dict[str, Any] = {
        "model": model_config.model,
        "messages": [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "以下のキャラクターシートから外見情報を読み取り、"
                    "NovelAI 互換の画像生成タグに変換してください:\n\n"
                    + character_text
                ),
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

        response = await litellm.acompletion(**kwargs)
        result = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(
            "LLM prompt synthesis failed for %s: %s",
            anima_name,
            exc,
        )
        return None

    if not result or result == "NO_APPEARANCE_DATA":
        return None

    # Cache result to assets/prompt.txt
    try:
        cache_dir = anima_dir / "assets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "prompt.txt").write_text(result + "\n", encoding="utf-8")
        logger.info(
            "Synthesized and cached prompt for %s: %.200s",
            anima_name,
            result,
        )
    except OSError:
        logger.debug("Failed to cache prompt.txt for %s", anima_name)

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
