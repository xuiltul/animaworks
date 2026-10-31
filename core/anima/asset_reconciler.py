from __future__ import annotations

from core.anima.workspace_files import IDENTITY_FILE

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Asset reconciliation — detect and generate missing Anima assets.

Primary mechanism: Anima bootstraps generate their own assets.
This module provides a fallback scan once at server startup and keeps
explicit per-anima reconciliation available for creation and repair paths.
Missing assets are generated with the ImageGenPipeline and
``skip_existing=True``.
"""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Literal

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
    "pixel_sheet": "pixel_sheet.png",
}

REALISTIC_REQUIRED_ASSETS: dict[str, str] = {
    "avatar_fullbody_realistic": "avatar_fullbody_realistic.png",
    "avatar_bustup_realistic": "avatar_bustup_realistic.png",
    "pixel_sheet": "pixel_sheet.png",
}

_PIXEL_ASSET_KEY = "pixel_sheet"

_3D_ASSET_KEYS = frozenset({"model_chibi", "model_rigged"})

# Failure records are scoped to the exact missing asset and generation inputs.
# This process-local registry prevents repeated startup/explicit scans from
# retrying an unchanged failure condition; changed inputs produce a new key.
_AssetFailureKey = tuple[str, str, bool, str, str, str]
_asset_failure_records: dict[_AssetFailureKey, str] = {}


def _prompt_source_fingerprint(anima_dir: Path, prompt: str | None = None) -> str:
    """Fingerprint prompt inputs without retaining their contents in logs/state."""
    digest = hashlib.sha256()
    if prompt is not None:
        digest.update(b"resolved-prompt\0")
        digest.update(prompt.encode("utf-8"))
        return digest.hexdigest()

    for relative_path in (
        IDENTITY_FILE,
        "character_sheet.md",
        "assets/prompt.txt",
        "assets/prompt_realistic.txt",
    ):
        try:
            source = (anima_dir / relative_path).stat()
            signature = f"{relative_path}:{source.st_mtime_ns}:{source.st_size}"
        except OSError:
            signature = f"{relative_path}:missing"
        digest.update(signature.encode("utf-8"))
    return digest.hexdigest()


def _failure_key(
    anima_dir: Path,
    image_style: str,
    enable_3d: bool,
    asset_key: str,
    stage: str,
    fingerprint: str,
) -> _AssetFailureKey:
    return (str(anima_dir.resolve()), image_style, enable_3d, asset_key, stage, fingerprint)


def _previously_failed_assets(
    anima_dir: Path,
    missing: list[str],
    *,
    image_style: str,
    enable_3d: bool,
    stage: str,
    fingerprint: str,
) -> list[str]:
    return [
        asset_key
        for asset_key in missing
        if _failure_key(anima_dir, image_style, enable_3d, asset_key, stage, fingerprint) in _asset_failure_records
    ]


def _record_failure(
    anima_dir: Path,
    missing: list[str],
    *,
    image_style: str,
    enable_3d: bool,
    stage: str,
    fingerprint: str,
    error: str,
    no_prompt: bool = False,
) -> None:
    """Record each failed asset and emit one concise log for a new condition."""
    newly_recorded: list[str] = []
    for asset_key in missing:
        key = _failure_key(anima_dir, image_style, enable_3d, asset_key, stage, fingerprint)
        if key not in _asset_failure_records:
            _asset_failure_records[key] = error
            newly_recorded.append(asset_key)
    if not newly_recorded:
        return

    missing_text = ", ".join(newly_recorded)
    if no_prompt:
        logger.info(
            "No prompt available for %s (missing assets: %s) — cannot generate assets",
            anima_dir.name,
            missing_text,
        )
        return
    one_line_error = error.replace("\r", " ").replace("\n", " ")[:300]
    logger.warning(
        "Asset generation failed for %s (missing assets: %s): %s",
        anima_dir.name,
        missing_text,
        one_line_error,
    )


def _get_lock(anima_name: str) -> asyncio.Lock:
    """Return (or create) the per-anima generation lock."""
    if anima_name not in _anima_locks:
        _anima_locks[anima_name] = asyncio.Lock()
    return _anima_locks[anima_name]


# ── Asset checking ────────────────────────────────────────────────


def _pixel_generation_available() -> bool:
    """Return whether the codex CLI needed for pixel sheet generation exists."""
    try:
        from core.integrations.image.codex import codex_available

        return bool(codex_available())
    except Exception:
        logger.debug("codex availability check failed", exc_info=True)
        return False


def _required_assets_for_style(
    image_style: Literal["anime", "realistic"],
    enable_3d: bool = True,
) -> dict[str, str]:
    """Return the required assets map for the given style."""
    if image_style == "realistic":
        assets = dict(REALISTIC_REQUIRED_ASSETS)
    else:
        assets = dict(REQUIRED_ASSETS)
        if not enable_3d:
            for k in _3D_ASSET_KEYS:
                assets.pop(k, None)
    if not _pixel_generation_available():
        # Without codex there is no way to produce a pixel sheet, so keep it out
        # of the required set instead of reporting it missing on every scan.
        assets.pop(_PIXEL_ASSET_KEY, None)
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
        if not (anima_dir / IDENTITY_FILE).exists():
            continue
        result = check_anima_assets(
            anima_dir,
            enable_3d=enable_3d,
            image_style=image_style,
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
            attempts to extract from IDENTITY.md.
        enable_3d: Whether to include 3D model generation steps.
        image_style: Which image style to generate assets for.

    Returns:
        Dict with generation results or skip reason.
    """
    anima_name = anima_dir.name
    # A blank first-run Anima has no appearance yet. This is waiting for
    # user input, not an asset-generation failure.
    if prompt is None:
        from core.anima.bootstrap_state import get_bootstrap_status

        if get_bootstrap_status(anima_dir).get("needs_user_input"):
            return {"anima": anima_name, "skipped": True, "reason": "awaiting_profile"}
    lock = _get_lock(anima_name)

    if lock.locked():
        logger.info(
            "Asset generation already in progress for %s, skipping",
            anima_name,
        )
        return {"anima": anima_name, "skipped": True, "reason": "locked"}

    # Skip if an active remake-preview session exists (backup dir present).
    # Generating during a remake session causes duplicate work and step-count
    # confusion because the user's slider settings are not used here.
    active_backups = await asyncio.to_thread(lambda: list(anima_dir.glob("assets_backup_*")))
    if active_backups:
        logger.debug(
            "Skipping reconciliation for %s — active remake session detected (%s)",
            anima_name,
            active_backups[0].name,
        )
        return {"anima": anima_name, "skipped": True, "reason": "remake_active"}

    async with lock:
        check = check_anima_assets(
            anima_dir,
            enable_3d=enable_3d,
            image_style=image_style,
        )
        if check["complete"]:
            logger.debug("Assets complete for %s (post-lock check)", anima_name)
            return {"anima": anima_name, "skipped": True, "reason": "complete"}

        missing = list(check["missing"])
        source_fingerprint = _prompt_source_fingerprint(anima_dir)
        if not prompt:
            previous_no_prompt = _previously_failed_assets(
                anima_dir,
                missing,
                image_style=image_style,
                enable_3d=enable_3d,
                stage="no_prompt",
                fingerprint=source_fingerprint,
            )
            if previous_no_prompt:
                return {
                    "anima": anima_name,
                    "skipped": True,
                    "reason": "previous_failure",
                    "failed_assets": previous_no_prompt,
                }

        try:
            resolved_prompt = prompt or await _extract_prompt(anima_dir, style=image_style)
        except Exception as exc:
            _record_failure(
                anima_dir,
                missing,
                image_style=image_style,
                enable_3d=enable_3d,
                stage="no_prompt",
                fingerprint=source_fingerprint,
                error=str(exc),
                no_prompt=True,
            )
            return {
                "anima": anima_name,
                "skipped": True,
                "reason": "no_prompt",
                "error": str(exc),
            }

        if not resolved_prompt:
            _record_failure(
                anima_dir,
                missing,
                image_style=image_style,
                enable_3d=enable_3d,
                stage="no_prompt",
                fingerprint=source_fingerprint,
                error="no prompt available",
                no_prompt=True,
            )
            return {
                "anima": anima_name,
                "skipped": True,
                "reason": "no_prompt",
            }

        prompt_fingerprint = _prompt_source_fingerprint(anima_dir, resolved_prompt)
        previous_generation_failures = _previously_failed_assets(
            anima_dir,
            missing,
            image_style=image_style,
            enable_3d=enable_3d,
            stage="generation",
            fingerprint=prompt_fingerprint,
        )
        if previous_generation_failures:
            return {
                "anima": anima_name,
                "skipped": True,
                "reason": "previous_failure",
                "failed_assets": previous_generation_failures,
            }

        logger.info(
            "Generating missing %s assets for %s (missing: %s)",
            image_style,
            anima_name,
            missing,
        )

        steps: list[str] | None = None
        if image_style == "anime" and not enable_3d:
            steps = ["fullbody", "bustup", "chibi", "pixel"]

        try:
            from core.config.models import ImageGenConfig, load_config
            from core.integrations.image_gen import ImageGenPipeline

            try:
                image_config = load_config().image_gen.model_copy(
                    update={"image_style": image_style, "enable_3d": enable_3d}
                )
            except Exception:
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
            errors = list(result.errors or [])
            remaining = check_anima_assets(
                anima_dir,
                enable_3d=enable_3d,
                image_style=image_style,
            )["missing"]
            if remaining:
                reason = "; ".join(str(error) for error in errors) or "required assets remain missing"
                _record_failure(
                    anima_dir,
                    list(remaining),
                    image_style=image_style,
                    enable_3d=enable_3d,
                    stage="generation",
                    fingerprint=prompt_fingerprint,
                    error=reason,
                )
            else:
                logger.info(
                    "Asset generation complete for %s: %s",
                    anima_name,
                    generated,
                )

            return {
                "anima": anima_name,
                "skipped": False,
                "generated": generated,
                "errors": errors,
            }
        except Exception as exc:
            error_str = str(exc)
            remaining = check_anima_assets(
                anima_dir,
                enable_3d=enable_3d,
                image_style=image_style,
            )["missing"]
            _record_failure(
                anima_dir,
                list(remaining),
                image_style=image_style,
                enable_3d=enable_3d,
                stage="generation",
                fingerprint=prompt_fingerprint,
                error=error_str,
            )
            return {
                "anima": anima_name,
                "skipped": False,
                "error": error_str,
                "failed_assets": list(remaining),
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
        animas_dir,
        enable_3d=enable_3d,
        image_style=image_style,
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
            anima_dir,
            enable_3d=enable_3d,
            image_style=image_style,
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
                    "Failed to broadcast asset update for %s",
                    anima_name,
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
                from core.integrations._image_clients import _convert_anime_to_realistic

                return _convert_anime_to_realistic(anime_text)
        return None

    prompt_path = assets_dir / "prompt.txt"
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    # Fallback: convert realistic prompt to anime-friendly tags
    realistic_path = assets_dir / "prompt_realistic.txt"
    if realistic_path.exists():
        text = realistic_path.read_text(encoding="utf-8").strip()
        if text:
            logger.debug("Falling back to realistic prompt for anime style: %s", anima_dir.name)
            return text
    return None


_APPEARANCE_LABEL_RE = re.compile(
    r"(?:image[_ ]?prompt|画像プロンプト|外見|appearance|キャラクターデザイン|character[_ ]?design)"
    r"[*\s]*[:：][*\s]*(.*)",
    re.IGNORECASE,
)


def _extract_appearance_field(text: str) -> str | None:
    """Find an appearance label and return its value.

    Tolerates markdown bold markers around the label (``- **外見**:``) and
    collects the following indented lines when the value is written as a
    multi-line bullet section instead of on the same line.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = _APPEARANCE_LABEL_RE.search(line)
        if not match:
            continue
        inline = match.group(1).strip(" \t*")
        if inline:
            return inline
        # Value continues on following, more-indented lines
        indent = len(line) - len(line.lstrip())
        block: list[str] = []
        for follow in lines[i + 1 :]:
            if not follow.strip():
                break
            if len(follow) - len(follow.lstrip()) <= indent:
                break
            block.append(follow.strip().lstrip("-•* \t"))
        if block:
            return "、".join(block)
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
         (searches IDENTITY.md, then character_sheet.md)
      2. Cached ``assets/prompt.txt`` (or ``prompt_realistic.txt``)
         from a previous LLM synthesis
      3. LLM synthesis — pass the full character document to LLM which
         extracts visual appearance and converts to tags.
         Tries character_sheet.md first, then IDENTITY.md.
    """
    # Collect candidate texts: IDENTITY.md first, then character_sheet.md
    candidates: list[str] = []
    for filename in (IDENTITY_FILE, "character_sheet.md"):
        path = anima_dir / filename
        if path.exists():
            candidates.append(path.read_text(encoding="utf-8"))

    if not candidates:
        return None

    # Step 1: Look for a dedicated image_prompt / appearance field
    for text in candidates:
        raw = _extract_appearance_field(text)
        if raw:
            if style == "realistic":
                from core.integrations._image_clients import _convert_anime_to_realistic

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
    # Prefer character_sheet.md (has appearance info), fall back to IDENTITY.md.
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

    if style == "realistic":
        system_prompt_name = "fragments/asset_synthesis_system_realistic"
        user_prompt_key = "asset_reconciler.llm_user_prompt_realistic"
    else:
        system_prompt_name = "fragments/asset_synthesis_system"
        user_prompt_key = "asset_reconciler.llm_user_prompt"

    from core.config.helper_models import resolve_helper_model
    from core.llm.helper_completion import one_shot_helper_completion

    system_content = load_prompt(system_prompt_name)
    user_content = t(user_prompt_key, character_text=character_text)
    helper_model = resolve_helper_model("asset_reconcile", anima_dir)

    try:
        result = (
            await one_shot_helper_completion(
                user_content,
                role="asset_reconcile",
                anima_dir=anima_dir,
                resolved=helper_model,
                system_prompt=system_content,
                # thinking models can spend several hundred tokens on reasoning.
                max_tokens=4096,
            )
            or ""
        ).strip()
    except Exception as exc:
        logger.debug(
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


def _resolve_prompt_synthesis_model(anima_dir: Path) -> tuple[str | None, str]:
    """Compatibility facade for the asset-reconcile helper role."""
    from core.config.helper_models import resolve_helper_model

    helper_model = resolve_helper_model("asset_reconcile", anima_dir)
    return helper_model.model, helper_model.credential or ""


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
    if result.pixel_sheet_path and "pixel" not in skipped:
        generated.append("pixel")
    for anim_name in result.animation_paths:
        if f"anim_{anim_name}" not in skipped:
            generated.append(f"anim_{anim_name}")

    return {"generated": generated, "skipped": skipped}
