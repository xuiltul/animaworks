from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asset reconciliation — detect and generate missing Person assets.

Primary mechanism: Person bootstraps generate their own assets.
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

# Per-person locks to prevent concurrent generation (bootstrap vs fallback).
_person_locks: dict[str, asyncio.Lock] = {}

# Required base assets.  If *any* of these are missing the person
# is considered to have incomplete assets.
REQUIRED_ASSETS: dict[str, str] = {
    "avatar_fullbody": "avatar_fullbody.png",
    "avatar_bustup": "avatar_bustup.png",
    "avatar_chibi": "avatar_chibi.png",
    "model_chibi": "avatar_chibi.glb",
    "model_rigged": "avatar_chibi_rigged.glb",
}


def _get_lock(person_name: str) -> asyncio.Lock:
    """Return (or create) the per-person generation lock."""
    if person_name not in _person_locks:
        _person_locks[person_name] = asyncio.Lock()
    return _person_locks[person_name]


# ── Asset checking ────────────────────────────────────────────────


def check_person_assets(person_dir: Path) -> dict[str, Any]:
    """Check a person's asset completeness using metadata logic.

    Returns a dict with:
      - ``complete`` (bool): True if all required assets exist.
      - ``missing`` (list[str]): Keys of missing required assets.
      - ``present`` (list[str]): Keys of present required assets.
      - ``has_assets_dir`` (bool): Whether the assets/ directory exists.
    """
    assets_dir = person_dir / "assets"
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


def find_persons_with_missing_assets(
    persons_dir: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Scan all person directories and return those with incomplete assets.

    Returns list of ``(person_name, check_result)`` tuples.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    if not persons_dir.exists():
        return results

    for person_dir in sorted(persons_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        if not (person_dir / "identity.md").exists():
            continue
        result = check_person_assets(person_dir)
        if not result["complete"]:
            results.append((person_dir.name, result))

    return results


# ── Asset generation (fallback) ───────────────────────────────────


async def reconcile_person_assets(
    person_dir: Path,
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Generate missing assets for a single person (non-blocking).

    Acquires the per-person lock so bootstrap and fallback cannot run
    concurrently.  Uses ``skip_existing=True`` for differential
    generation.

    Args:
        person_dir: Path to the person's runtime directory.
        prompt: Character prompt for image generation.  If ``None``,
            attempts to extract from identity.md.

    Returns:
        Dict with generation results or skip reason.
    """
    person_name = person_dir.name
    lock = _get_lock(person_name)

    if lock.locked():
        logger.info(
            "Asset generation already in progress for %s, skipping",
            person_name,
        )
        return {"person": person_name, "skipped": True, "reason": "locked"}

    async with lock:
        # Re-check after acquiring lock — another task may have generated
        check = check_person_assets(person_dir)
        if check["complete"]:
            logger.debug("Assets complete for %s (post-lock check)", person_name)
            return {"person": person_name, "skipped": True, "reason": "complete"}

        logger.info(
            "Generating missing assets for %s (missing: %s)",
            person_name,
            check["missing"],
        )

        resolved_prompt = prompt or await _extract_prompt(person_dir)
        if not resolved_prompt:
            logger.warning(
                "No prompt available for %s — cannot generate assets",
                person_name,
            )
            return {
                "person": person_name,
                "skipped": True,
                "reason": "no_prompt",
            }

        try:
            from core.tools.image_gen import ImageGenPipeline

            pipeline = ImageGenPipeline(person_dir)
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
                person_name,
                generated,
            )
            return {
                "person": person_name,
                "skipped": False,
                "generated": generated,
                "errors": result.errors,
            }
        except Exception as exc:
            logger.exception("Asset generation failed for %s", person_name)
            return {
                "person": person_name,
                "skipped": False,
                "error": str(exc),
            }


async def reconcile_all_assets(
    persons_dir: Path,
    *,
    ws_manager: Any | None = None,
) -> list[dict[str, Any]]:
    """Check all persons and generate missing assets sequentially.

    Args:
        persons_dir: Root persons directory.
        ws_manager: Optional WebSocketManager for broadcasting updates.

    Returns:
        List of per-person result dicts.
    """
    incomplete = find_persons_with_missing_assets(persons_dir)
    if not incomplete:
        logger.debug("All persons have complete assets")
        return []

    logger.info(
        "Asset reconciliation: %d person(s) with missing assets: %s",
        len(incomplete),
        [name for name, _ in incomplete],
    )

    results: list[dict[str, Any]] = []
    for person_name, _check in incomplete:
        person_dir = persons_dir / person_name
        result = await reconcile_person_assets(person_dir)
        results.append(result)

        # Broadcast asset update if something was generated
        if ws_manager and not result.get("skipped"):
            try:
                await ws_manager.broadcast(
                    "person.assets_updated",
                    {"name": person_name, "source": "reconciliation"},
                )
            except Exception:
                logger.debug(
                    "Failed to broadcast asset update for %s", person_name,
                )

    return results


# ── Helpers ───────────────────────────────────────────────────────


async def _extract_prompt(person_dir: Path) -> str | None:
    """Try to extract a generation prompt from the person's identity.md.

    Fallback chain:
      1. Regex match for explicit ``image_prompt:`` / ``外見:`` fields
      2. Cached ``assets/prompt.txt`` from a previous LLM synthesis
      3. LLM synthesis from the appearance table in identity.md
    """
    identity_path = person_dir / "identity.md"
    if not identity_path.exists():
        return None

    text = identity_path.read_text(encoding="utf-8")

    # Step 1: Look for a dedicated image_prompt / appearance field
    patterns = [
        r"(?:image[_ ]?prompt|画像プロンプト|外見|appearance)\s*[:\uff1a]\s*(.+)",
        r"(?:キャラクターデザイン|character[_ ]?design)\s*[:\uff1a]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Step 2: Check cached prompt from previous LLM synthesis
    prompt_cache = person_dir / "assets" / "prompt.txt"
    if prompt_cache.exists():
        cached = prompt_cache.read_text(encoding="utf-8").strip()
        if cached:
            logger.debug(
                "Using cached prompt for %s from assets/prompt.txt",
                person_dir.name,
            )
            return cached

    # Step 3: Extract appearance table and synthesize via LLM
    appearance = _extract_appearance_table(text)
    if not appearance:
        return None

    return await _synthesize_prompt_via_llm(person_dir, appearance)


# ── Appearance extraction ─────────────────────────────────────

# Fields in the identity.md profile table that describe visual appearance.
_APPEARANCE_FIELDS: frozenset[str] = frozenset({
    "髪型", "髪色", "瞳の色", "顔タイプ", "身長", "体重",
    "スリーサイズ", "イメージカラー",
})


def _extract_appearance_table(text: str) -> str | None:
    """Extract appearance-related rows from the profile table in identity.md.

    Looks for Markdown table rows (``| key | value |``) where the key
    matches a known appearance field.  Returns the matching rows as a
    single string, or ``None`` if no appearance data is found.
    """
    rows: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", line)
        if not m:
            continue
        key = m.group(1).strip()
        if key in _APPEARANCE_FIELDS:
            rows.append(f"{key}: {m.group(2).strip()}")
    return "\n".join(rows) if rows else None


# ── LLM prompt synthesis ──────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert at converting Japanese character appearance descriptions \
into NovelAI-compatible anime image generation tags.

Rules:
- Output ONLY a comma-separated tag string, nothing else.
- Start with 1girl or 1boy.
- Use plain English color names, NOT gemstone metaphors \
  (サファイアブルー → blue, エメラルドグリーン → green).
- Decompose compound descriptions into atomic tags \
  (ショートボブ、前髪ぱっつん → short hair, bob cut, blunt bangs).
- Translate hair accessories (ピン → hair clip, リボン → hair ribbon).
- Omit non-visual traits (personality, hobbies).
- Height/weight: omit unless notably tall/short (use tall or petite).
- Always end with: full body, standing, white background
- All tags lowercase, separated by comma + space.

Example input:
髪型: ロングヘア、ツインテール
髪色: 黒
瞳の色: 赤
顔タイプ: クール系

Example output:
1girl, black hair, long hair, twintails, red eyes, cool expression, \
full body, standing, white background"""


async def _synthesize_prompt_via_llm(
    person_dir: Path,
    appearance: str,
) -> str | None:
    """Call the Person's LLM to convert appearance text into NovelAI tags.

    On success the result is cached to ``assets/prompt.txt``.
    On failure returns ``None`` (logs the error, does not raise).
    """
    person_name = person_dir.name

    try:
        from core.config.models import load_model_config

        model_config = load_model_config(person_dir)
    except Exception:
        logger.warning(
            "Cannot load model config for %s — skipping LLM synthesis",
            person_name,
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
                    "以下の外見情報を NovelAI 互換タグに変換してください:\n\n"
                    + appearance
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
            person_name,
            exc,
        )
        return None

    if not result:
        return None

    # Cache result to assets/prompt.txt
    try:
        cache_dir = person_dir / "assets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "prompt.txt").write_text(result + "\n", encoding="utf-8")
        logger.info(
            "Synthesized and cached prompt for %s: %.200s",
            person_name,
            result,
        )
    except OSError:
        logger.debug("Failed to cache prompt.txt for %s", person_name)

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
