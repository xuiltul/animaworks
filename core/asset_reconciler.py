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

        resolved_prompt = prompt or _extract_prompt(person_dir)
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


def _extract_prompt(person_dir: Path) -> str | None:
    """Try to extract a generation prompt from the person's identity.md."""
    import re

    identity_path = person_dir / "identity.md"
    if not identity_path.exists():
        return None

    text = identity_path.read_text(encoding="utf-8")

    # Look for a dedicated image_prompt / appearance section
    patterns = [
        r"(?:image[_ ]?prompt|画像プロンプト|外見|appearance)\s*[:\uff1a]\s*(.+)",
        r"(?:キャラクターデザイン|character[_ ]?design)\s*[:\uff1a]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


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
