# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""ImageGenPipeline – orchestrates the full character asset generation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.tools._base import logger
from core.tools._image_clients import (
    _CHIBI_PROMPT,
    _DEFAULT_ANIMATIONS,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
    _REALISTIC_BUSTUP_PROMPT,
    _REALISTIC_EXPRESSION_GUIDANCE,
    _REALISTIC_EXPRESSION_PROMPTS,
    _image_to_data_uri,
)
from core.tools._image_glb import (
    _download_armature_animation,
    optimize_glb,
    strip_mesh_from_glb,
)

if TYPE_CHECKING:
    from core.config.models import ImageGenConfig

from core.schemas import VALID_EMOTIONS as _VALID_EXPRESSION_NAMES

__all__ = [
    "PipelineResult",
    "ImageGenPipeline",
]


# ── PipelineResult ─────────────────────────────────────────


@dataclass
class PipelineResult:
    """Result of the full character asset generation pipeline."""

    fullbody_path: Path | None = None
    bustup_path: Path | None = None
    bustup_paths: dict[str, Path] = field(default_factory=dict)
    chibi_path: Path | None = None
    model_path: Path | None = None
    rigged_model_path: Path | None = None
    animation_paths: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fullbody": str(self.fullbody_path) if self.fullbody_path else None,
            "bustup": str(self.bustup_path) if self.bustup_path else None,
            "bustup_expressions": {k: str(v) for k, v in self.bustup_paths.items()},
            "chibi": str(self.chibi_path) if self.chibi_path else None,
            "model": str(self.model_path) if self.model_path else None,
            "rigged_model": str(self.rigged_model_path) if self.rigged_model_path else None,
            "animations": {k: str(v) for k, v in self.animation_paths.items()},
            "errors": self.errors,
            "skipped": self.skipped,
        }


# ── ImageGenPipeline ───────────────────────────────────────


class ImageGenPipeline:
    """Orchestrates the full character asset generation pipeline.

    Steps:
      1. NovelAI V4.5 → full-body anime image
      2. Flux Kontext  → bust-up from reference  ─┐ independent
      3. Flux Kontext  → chibi from reference     ─┘
      4. Meshy Image-to-3D → GLB from chibi
      5. Meshy Rigging → rigged GLB + walking/running animations
      6. Meshy Animations → idle/sitting/waving/talking GLBs
    """

    ASSET_NAMES = {
        "fullbody": "avatar_fullbody.png",
        "bustup": "avatar_bustup.png",
        "chibi": "avatar_chibi.png",
        "model": "avatar_chibi.glb",
        "rigged_model": "avatar_chibi_rigged.glb",
    }
    REALISTIC_ASSET_NAMES = {
        "fullbody": "avatar_fullbody_realistic.png",
        "bustup": "avatar_bustup_realistic.png",
    }

    def __init__(
        self,
        anima_dir: Path,
        config: "ImageGenConfig | None" = None,
    ) -> None:
        from core.config.models import ImageGenConfig

        self._anima_dir = anima_dir
        self._assets_dir = anima_dir / "assets"
        self._config = config or ImageGenConfig()

    @property
    def _is_realistic(self) -> bool:
        return getattr(self._config, "image_style", "anime") == "realistic"

    def _asset_name(self, key: str) -> str:
        """Return asset filename based on current style."""
        if self._is_realistic and key in self.REALISTIC_ASSET_NAMES:
            return self.REALISTIC_ASSET_NAMES[key]
        return self.ASSET_NAMES[key]

    def _bustup_filename(self, expression: str) -> str:
        """Return bustup expression filename based on current style."""
        suffix = "_realistic" if self._is_realistic else ""
        if expression == "neutral":
            return f"avatar_bustup{suffix}.png"
        return f"avatar_bustup_{expression}{suffix}.png"

    def generate_bustup_expression(
        self,
        reference_image: bytes,
        expression: str,
        skip_existing: bool = True,
    ) -> Path | None:
        """Generate a single expression variant of the bustup image.

        Args:
            reference_image: Full-body reference image bytes.
            expression: Expression name (e.g. "smile", "troubled").
            skip_existing: Skip if output file already exists.

        Returns:
            Path to generated image, or None on failure.
        """
        if expression not in _VALID_EXPRESSION_NAMES:
            logger.warning("Unknown expression: %s", expression)
            return None

        output_filename = self._bustup_filename(expression)
        output_path = self._assets_dir / output_filename

        if skip_existing and output_path.exists():
            logger.info("Skipping existing: %s", output_path)
            return output_path

        if self._is_realistic:
            prompt = _REALISTIC_EXPRESSION_PROMPTS[expression]
        else:
            prompt = _EXPRESSION_PROMPTS[expression]

        if self._config.style_prefix:
            prompt = self._config.style_prefix + prompt
        if self._config.style_suffix:
            prompt = prompt + self._config.style_suffix

        self._assets_dir.mkdir(parents=True, exist_ok=True)

        from core.tools.image_gen import FluxKontextClient

        kontext = FluxKontextClient()
        if self._is_realistic:
            guidance = _REALISTIC_EXPRESSION_GUIDANCE.get(expression, 4.5)
        else:
            guidance = _EXPRESSION_GUIDANCE.get(expression, 5.0)
        result_bytes = kontext.generate_from_reference(
            reference_image=reference_image,
            prompt=prompt,
            aspect_ratio="3:4",
            guidance_scale=guidance,
        )

        output_path.write_bytes(result_bytes)
        logger.info("Generated expression '%s' (guidance=%.1f): %s", expression, guidance, output_path)
        return output_path

    def generate_all(
        self,
        prompt: str,
        negative_prompt: str = "",
        skip_existing: bool = True,
        steps: list[str] | None = None,
        animations: dict[str, int] | None = None,
        expressions: list[str] | None = None,
        vibe_image: bytes | None = None,
        vibe_strength: float | None = None,
        vibe_info_extracted: float | None = None,
        seed: int | None = None,
        progress_callback: "Callable[[str, str, int], None] | None" = None,
    ) -> PipelineResult:
        """Run the 6-step pipeline synchronously.

        Args:
            prompt: Character appearance tags for NovelAI.
            negative_prompt: Negative prompt for NovelAI.
            skip_existing: Skip steps whose output file already exists.
            steps: Subset of steps to run (default: all).
            animations: Dict of {name: action_id} to generate.
                Default: idle, sitting, waving, talking.
            expressions: Subset of expressions to generate.
            vibe_image: Override Vibe Transfer reference image bytes.
                When provided, takes precedence over config.style_reference.
            vibe_strength: Override Vibe Transfer strength (0.0-1.0).
                Falls back to config.vibe_strength.
            vibe_info_extracted: Override Vibe Transfer info extraction (0.0-1.0).
                Falls back to config.vibe_info_extracted.
            seed: Seed for reproducibility (fullbody generation only).
            progress_callback: Optional callback ``(step, status, pct)`` for
                real-time progress reporting.

        Returns:
            PipelineResult with paths and error info.
        """
        from core.tools.image_gen import (
            FalTextToImageClient,
            FluxKontextClient,
            MeshyClient,
            NovelAIClient,
        )

        self._assets_dir.mkdir(parents=True, exist_ok=True)
        if steps:
            enabled = set(steps)
        elif self._is_realistic:
            enabled = {"fullbody", "bustup"}
        else:
            enabled = {"fullbody", "bustup", "chibi", "3d", "rigging", "animations"}
        anim_map = animations if animations is not None else _DEFAULT_ANIMATIONS
        result = PipelineResult()

        def _notify(step: str, status: str, pct: int = 0) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, status, pct)
                except Exception:
                    logger.debug("progress_callback error (ignored)", exc_info=True)

        # ── Step 1: Full-body ──
        fullbody_bytes: bytes | None = None
        fullbody_path = self._assets_dir / self._asset_name("fullbody")

        if "fullbody" in enabled:
            if skip_existing and fullbody_path.exists():
                result.skipped.append("fullbody")
                fullbody_bytes = fullbody_path.read_bytes()
                result.fullbody_path = fullbody_path
            else:
                try:
                    _notify("fullbody", "generating", 0)
                    if self._is_realistic:
                        if not os.environ.get("FAL_KEY"):
                            raise RuntimeError(
                                "FAL_KEY required for realistic image generation."
                            )
                        logger.info("Step 1: Generating realistic full-body with Fal Flux Pro …")
                        client: NovelAIClient | FalTextToImageClient = FalTextToImageClient()
                    elif os.environ.get("NOVELAI_TOKEN"):
                        logger.info("Step 1: Generating full-body with NovelAI …")
                        client = NovelAIClient()
                    elif os.environ.get("FAL_KEY"):
                        logger.info(
                            "Step 1: Generating full-body with Fal Flux Pro (fallback) …",
                        )
                        client = FalTextToImageClient()
                    else:
                        raise RuntimeError(
                            "No image generation API key configured. "
                            "Set NOVELAI_TOKEN or FAL_KEY."
                        )

                    # ── A: Load style reference for Vibe Transfer ──
                    # Direct vibe_image parameter takes precedence over config
                    effective_vibe: bytes | None = vibe_image
                    if effective_vibe is None and self._config.style_reference:
                        style_path = Path(self._config.style_reference).expanduser()
                        if style_path.exists():
                            effective_vibe = style_path.read_bytes()
                            logger.debug("Loaded style reference: %s", style_path)
                        else:
                            logger.warning(
                                "Style reference not found: %s", style_path
                            )

                    effective_vibe_strength = (
                        vibe_strength
                        if vibe_strength is not None
                        else self._config.vibe_strength
                    )
                    effective_vibe_info = (
                        vibe_info_extracted
                        if vibe_info_extracted is not None
                        else self._config.vibe_info_extracted
                    )

                    # ── B: Apply style prefix/suffix to prompt ──
                    styled_prompt = prompt
                    if self._config.style_prefix:
                        styled_prompt = self._config.style_prefix + styled_prompt
                    if self._config.style_suffix:
                        styled_prompt = styled_prompt + self._config.style_suffix

                    styled_negative = negative_prompt
                    if self._config.negative_prompt_extra:
                        if styled_negative:
                            styled_negative += ", " + self._config.negative_prompt_extra
                        else:
                            styled_negative = self._config.negative_prompt_extra

                    fullbody_bytes = client.generate_fullbody(
                        prompt=styled_prompt,
                        negative_prompt=styled_negative,
                        seed=seed,
                        vibe_image=effective_vibe,
                        vibe_strength=effective_vibe_strength,
                        vibe_info_extracted=effective_vibe_info,
                    )
                    fullbody_path.write_bytes(fullbody_bytes)
                    result.fullbody_path = fullbody_path
                    logger.info("Step 1 complete: %s", fullbody_path)
                    _notify("fullbody", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"fullbody: {exc}")
                    logger.error("Step 1 failed: %s", exc)
                    _notify("fullbody", "error", 0)
        elif fullbody_path.exists():
            fullbody_bytes = fullbody_path.read_bytes()
            result.fullbody_path = fullbody_path

        if fullbody_bytes is None:
            # Cannot proceed without a reference image
            if not result.errors:
                result.errors.append(
                    "fullbody: No full-body image available as reference"
                )
            return result

        # ── Step 2 & 3: Bust-up and Chibi (sequential, same client) ──
        chibi_bytes: bytes | None = None

        if "bustup" in enabled:
            _notify("bustup", "generating", 0)
            expr_list = expressions or list(_EXPRESSION_PROMPTS.keys())
            logger.info("Step 2: Generating bustup expressions (%s): %s",
                        "realistic" if self._is_realistic else "anime", expr_list)

            bustup_ref_bytes: bytes | None = None
            neutral_path = self._assets_dir / self._bustup_filename("neutral")

            if "neutral" in expr_list:
                try:
                    path = self.generate_bustup_expression(
                        reference_image=fullbody_bytes,
                        expression="neutral",
                        skip_existing=skip_existing,
                    )
                    if path and path.exists():
                        bustup_ref_bytes = path.read_bytes()
                        result.bustup_paths["neutral"] = path
                        result.bustup_path = path
                except Exception as exc:
                    result.errors.append(f"bustup_neutral: {exc}")
                    logger.error("Bustup neutral failed: %s", exc)
            elif neutral_path.exists():
                # neutral not requested but exists on disk — use as reference
                bustup_ref_bytes = neutral_path.read_bytes()

            # Step 2b: Generate other expressions using bustup as reference.
            # Fall back to fullbody if bustup reference is unavailable.
            ref_for_expressions = bustup_ref_bytes or fullbody_bytes
            if bustup_ref_bytes:
                logger.info("Using neutral bustup as reference for expression variants")
            else:
                logger.warning("Neutral bustup unavailable, falling back to fullbody reference")

            for expr in expr_list:
                if expr == "neutral":
                    continue  # already handled above
                try:
                    path = self.generate_bustup_expression(
                        reference_image=ref_for_expressions,
                        expression=expr,
                        skip_existing=skip_existing,
                    )
                    if path:
                        result.bustup_paths[expr] = path
                except Exception as exc:
                    result.errors.append(f"bustup_{expr}: {exc}")
                    logger.error("Bustup expression '%s' failed: %s", expr, exc)

            if not result.bustup_path and result.bustup_paths:
                result.bustup_path = next(iter(result.bustup_paths.values()))
            logger.info("Step 2 complete: %d expressions generated", len(result.bustup_paths))
            _notify("bustup", "completed", 100)

        if "chibi" in enabled:
            chibi_path = self._assets_dir / self._asset_name("chibi")
            if skip_existing and chibi_path.exists():
                result.skipped.append("chibi")
                chibi_bytes = chibi_path.read_bytes()
                result.chibi_path = chibi_path
            else:
                try:
                    _notify("chibi", "generating", 0)
                    logger.info("Step 3: Generating chibi with Flux Kontext …")
                    kontext = FluxKontextClient()
                    chibi_bytes = kontext.generate_from_reference(
                        reference_image=fullbody_bytes,
                        prompt=_CHIBI_PROMPT,
                        aspect_ratio="1:1",
                    )
                    chibi_path.write_bytes(chibi_bytes)
                    result.chibi_path = chibi_path
                    logger.info("Step 3 complete: %s", chibi_path)
                    _notify("chibi", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"chibi: {exc}")
                    logger.error("Step 3 failed: %s", exc)
                    _notify("chibi", "error", 0)

        # ── Step 4: 3-D model from chibi ──
        meshy_task_id: str | None = None  # tracked for rigging input
        meshy: MeshyClient | None = None

        if "3d" in enabled:
            if chibi_bytes is None:
                chibi_path = self._assets_dir / self._asset_name("chibi")
                if chibi_path.exists():
                    chibi_bytes = chibi_path.read_bytes()

            model_path = self._assets_dir / self._asset_name("model")
            if skip_existing and model_path.exists():
                result.skipped.append("3d")
                result.model_path = model_path
            elif chibi_bytes is None:
                result.errors.append("3d: No chibi image available for 3D conversion")
            else:
                try:
                    _notify("3d", "generating", 0)
                    logger.info("Step 4: Generating 3D model with Meshy …")
                    meshy = MeshyClient()
                    meshy_task_id = meshy.create_task(chibi_bytes)
                    logger.info("Meshy task created: %s", meshy_task_id)
                    task = meshy.poll_task(meshy_task_id)
                    glb_bytes = meshy.download_model(task, fmt="glb")
                    model_path.write_bytes(glb_bytes)
                    result.model_path = model_path
                    logger.info("Step 4 complete: %s", model_path)
                    _notify("3d", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"3d: {exc}")
                    logger.error("Step 4 failed: %s", exc)
                    _notify("3d", "error", 0)

        # ── Step 5: Rigging ──
        rig_task_id: str | None = None

        if "rigging" in enabled:
            rigged_path = self._assets_dir / self._asset_name("rigged_model")
            if skip_existing and rigged_path.exists():
                result.skipped.append("rigging")
                result.rigged_model_path = rigged_path
            elif meshy_task_id is None:
                result.errors.append(
                    "rigging: No Meshy task_id available "
                    "(run 3d step first or provide model)"
                )
            else:
                try:
                    _notify("rigging", "generating", 0)
                    logger.info("Step 5: Rigging 3D model with Meshy …")
                    if meshy is None:
                        meshy = MeshyClient()
                    rig_task_id = meshy.create_rigging_task(meshy_task_id)
                    logger.info("Meshy rigging task created: %s", rig_task_id)
                    rig_task = meshy.poll_rigging_task(rig_task_id)

                    # Download rigged model
                    rigged_bytes = meshy.download_rigged_model(rig_task, fmt="glb")
                    rigged_path.write_bytes(rigged_bytes)
                    result.rigged_model_path = rigged_path
                    optimize_glb(rigged_path)
                    logger.info("Rigged model saved: %s", rigged_path)

                    # Download built-in walking/running animations
                    basic_anims = meshy.download_rigging_animations(rig_task)
                    for anim_name, anim_bytes in basic_anims.items():
                        anim_path = self._assets_dir / f"anim_{anim_name}.glb"
                        anim_path.write_bytes(anim_bytes)
                        if not strip_mesh_from_glb(anim_path):
                            result.errors.append(f"mesh_strip: failed for {anim_path.name}")
                            logger.warning("Mesh strip failed for %s, animation saved with embedded mesh", anim_path.name)
                        result.animation_paths[anim_name] = anim_path
                        logger.info(
                            "Animation '%s' saved: %s (%d bytes)",
                            anim_name, anim_path, len(anim_bytes),
                        )
                    logger.info("Step 5 complete: rigged + %d animations", len(basic_anims))
                    _notify("rigging", "completed", 100)
                except Exception as exc:
                    result.errors.append(f"rigging: {exc}")
                    logger.error("Step 5 failed: %s", exc)
                    _notify("rigging", "error", 0)

        # ── Step 6: Additional animations ──
        if "animations" in enabled and anim_map:
            if rig_task_id is None:
                result.errors.append(
                    "animations: No rigging task_id available "
                    "(run rigging step first)"
                )
            else:
                _notify("animations", "generating", 0)
                if meshy is None:
                    meshy = MeshyClient()
                for anim_name, action_id in anim_map.items():
                    anim_path = self._assets_dir / f"anim_{anim_name}.glb"
                    if skip_existing and anim_path.exists():
                        result.skipped.append(f"anim_{anim_name}")
                        result.animation_paths[anim_name] = anim_path
                        continue
                    try:
                        logger.info(
                            "Step 6: Generating '%s' animation (action_id=%d) …",
                            anim_name, action_id,
                        )
                        anim_task_id = meshy.create_animation_task(
                            rig_task_id, action_id,
                        )
                        logger.info(
                            "Animation task '%s' created: %s",
                            anim_name, anim_task_id,
                        )
                        anim_task = meshy.poll_animation_task(anim_task_id)
                        if not _download_armature_animation(anim_task, anim_path):
                            raise RuntimeError(
                                f"All download methods failed for {anim_path.name}"
                            )
                        result.animation_paths[anim_name] = anim_path
                        logger.info(
                            "Animation '%s' saved: %s (%d bytes)",
                            anim_name, anim_path, anim_path.stat().st_size,
                        )
                    except Exception as exc:
                        result.errors.append(f"anim_{anim_name}: {exc}")
                        logger.error(
                            "Animation '%s' failed: %s", anim_name, exc,
                        )
                _notify("animations", "completed", 100)

        return result
