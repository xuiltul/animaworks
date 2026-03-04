# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Character image & 3-D model generation tool for AnimaWorks.

Facade module — re-exports all public symbols from sub-modules and
provides the :func:`dispatch` entry point used by the tool handler.

Pipeline:
  1. NovelAI V4.5 → anime full-body image (fallback: fal.ai Flux Pro)
  2. Flux Kontext [pro] (fal.ai) → bust-up from reference
  3. Flux Kontext [pro] (fal.ai) → chibi from reference
  4. Meshy Image-to-3D → GLB model from chibi image
  5. Meshy Rigging → rigged GLB + walking/running animations
  6. Meshy Animations → idle/sitting/waving/talking GLBs
"""
from __future__ import annotations

import sys
import httpx  # noqa: F401 — patch compatibility
import os  # noqa: F401 — patch compatibility
import time  # noqa: F401 — patch compatibility
from pathlib import Path
from typing import Any

from core.tools._base import get_credential, logger  # noqa: F401 — patch compatibility

# ── Re-exports: _image_clients ─────────────────────────────
from core.tools._image_clients import (
    EXECUTION_PROFILE,
    FAL_FLUX_PRO_SUBMIT_URL,
    FAL_KONTEXT_SUBMIT_URL,
    MESHY_ANIMATION_TASK_TPL,
    MESHY_ANIMATION_URL,
    MESHY_IMAGE_TO_3D_URL,
    MESHY_RIGGING_TASK_TPL,
    MESHY_RIGGING_URL,
    MESHY_TASK_URL_TPL,
    NOVELAI_API_URL,
    NOVELAI_ENCODE_URL,
    NOVELAI_MODEL,
    FalTextToImageClient,
    FluxKontextClient,
    MeshyClient,
    NovelAIClient,
    _BUSTUP_PROMPT,
    _CHIBI_PROMPT,
    _DEFAULT_ANIMATIONS,
    _DOWNLOAD_TIMEOUT,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
    _HTTP_TIMEOUT,
    _REALISTIC_BUSTUP_PROMPT,
    _REALISTIC_EXPRESSION_GUIDANCE,
    _REALISTIC_EXPRESSION_PROMPTS,
    _RETRYABLE_CODES,
    _image_to_data_uri,
    _retry,
)

# ── Re-exports: _image_glb ────────────────────────────────
from core.tools._image_glb import (
    _convert_fbx_to_glb,
    _download_armature_animation,
    _ensure_fbx2gltf,
    _ensure_gltf_transform_modules,
    _find_fbx2gltf_binary,
    _run_gltf_transform,
    compress_textures,
    optimize_glb,
    simplify_glb,
    strip_mesh_from_glb,
)

# ── Re-exports: _image_pipeline ────────────────────────────
from core.tools._image_pipeline import ImageGenPipeline, PipelineResult

# ── Re-exports: _image_schemas ─────────────────────────────
from core.tools._image_schemas import get_cli_guide, get_tool_schemas

# ── Re-exports: _image_cli ─────────────────────────────────
from core.tools._image_cli import cli_main

# Re-export for `from core.tools.image_gen import _VALID_EXPRESSION_NAMES`
from core.schemas import VALID_EMOTIONS as _VALID_EXPRESSION_NAMES


# ── Mutable cache proxy ───────────────────────────────
# Tests manipulate _FBX2GLTF_PATH / _GLTF_MODULES_DIR via the facade module.
# These live in _image_glb; we proxy reads and writes so that
# ``import core.tools.image_gen as mod; mod._FBX2GLTF_PATH = X`` propagates.

import core.tools._image_glb as _glb_mod  # noqa: E402

_MUTABLE_GLB_ATTRS = {"_FBX2GLTF_PATH", "_GLTF_MODULES_DIR"}

_this = sys.modules[__name__]
_original_module_class = type(_this)


class _FacadeModule(_original_module_class):
    """Module subclass that proxies mutable GLB cache attributes."""

    def __getattr__(self, name: str) -> Any:
        if name in _MUTABLE_GLB_ATTRS:
            return getattr(_glb_mod, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _MUTABLE_GLB_ATTRS:
            setattr(_glb_mod, name, value)
            return
        super().__setattr__(name, value)


_this.__class__ = _FacadeModule


# ── Helpers ───────────────────────────────────────────

_ANIME_MARKER_TAGS = frozenset({
    "anime coloring", "clean lineart", "soft shading",
    "masterpiece", "best quality", "absurdres",
})


def _looks_like_anime_prompt(prompt: str) -> bool:
    """Heuristic: return True if prompt contains Danbooru-style anime tags."""
    tags = {t.strip().lower() for t in prompt.split(",")}
    return len(tags & _ANIME_MARKER_TAGS) >= 3


# ── Dispatch ──────────────────────────────────────────


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if tool_name == "generate_character_assets":
        from core.config.models import load_config
        from core.paths import get_animas_dir

        anima_dir = Path(args.pop("anima_dir", ""))
        supervisor_name: str | None = args.pop("supervisor_name", None)
        config = load_config()
        image_config = config.image_gen

        prompt = args["prompt"]

        # Auto-convert anime prompt when realistic style is configured
        if image_config.image_style == "realistic" and _looks_like_anime_prompt(prompt):
            from core.tools._image_clients import _convert_anime_to_realistic

            converted = _convert_anime_to_realistic(prompt)
            logger.info(
                "Auto-converted anime prompt to realistic for %s: %.120s → %.120s",
                anima_dir.name,
                prompt,
                converted,
            )
            prompt = converted

        # Use supervisor's fullbody image as Vibe Transfer reference
        if supervisor_name:
            ref_name = (
                "avatar_fullbody_realistic.png"
                if image_config.image_style == "realistic"
                else "avatar_fullbody.png"
            )
            supervisor_fullbody = (
                get_animas_dir() / supervisor_name / "assets" / ref_name
            )
            if supervisor_fullbody.exists():
                image_config = image_config.model_copy(
                    update={"style_reference": str(supervisor_fullbody)},
                )
                logger.info(
                    "Using supervisor image as vibe reference: %s",
                    supervisor_fullbody,
                )

        pipeline = ImageGenPipeline(anima_dir, config=image_config)
        result = pipeline.generate_all(
            prompt=prompt,
            negative_prompt=args.get("negative_prompt", ""),
            skip_existing=args.get("skip_existing", True),
            steps=args.get("steps"),
            animations=args.get("animations"),
        )
        return result.to_dict()

    if tool_name == "generate_fullbody":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        client = NovelAIClient()
        img = client.generate_fullbody(
            prompt=args["prompt"],
            negative_prompt=args.get("negative_prompt", ""),
            width=args.get("width", 1024),
            height=args.get("height", 1536),
            seed=args.get("seed"),
        )
        out = assets_dir / "avatar_fullbody.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_bustup":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        ref_path = assets_dir / "avatar_fullbody.png"
        if not ref_path.exists():
            return {"error": "No full-body reference image found"}
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_path.read_bytes(),
            prompt=args.get("prompt", _BUSTUP_PROMPT),
            aspect_ratio="3:4",
        )
        out = assets_dir / "avatar_bustup.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_chibi":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        ref_path = assets_dir / "avatar_fullbody.png"
        if not ref_path.exists():
            return {"error": "No full-body reference image found"}
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_path.read_bytes(),
            prompt=args.get("prompt", _CHIBI_PROMPT),
            aspect_ratio="1:1",
        )
        out = assets_dir / "avatar_chibi.png"
        out.write_bytes(img)
        return {"path": str(out), "size": len(img)}

    if tool_name == "generate_3d_model":
        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        chibi_path = assets_dir / "avatar_chibi.png"
        if not chibi_path.exists():
            return {"error": "No chibi image found for 3D conversion"}
        client = MeshyClient()
        task_id = client.create_task(
            chibi_path.read_bytes(),
            ai_model=args.get("ai_model", "meshy-6"),
            target_polycount=args.get("target_polycount", 30000),
        )
        task = client.poll_task(task_id)
        glb = client.download_model(task, fmt="glb")
        out = assets_dir / "avatar_chibi.glb"
        out.write_bytes(glb)
        return {"path": str(out), "size": len(glb), "task_id": task_id}

    if tool_name == "generate_rigged_model":
        import httpx as _httpx

        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        glb_path = assets_dir / "avatar_chibi.glb"
        if not glb_path.exists():
            return {"error": "No 3D model found for rigging"}
        client = MeshyClient()
        data_uri = _image_to_data_uri(
            glb_path.read_bytes(), mime="model/gltf-binary",
        )
        body = {
            "model_url": data_uri,
            "height_meters": args.get("height_meters", 1.0),
        }
        resp = _httpx.post(
            MESHY_RIGGING_URL,
            json=body,
            headers=client._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rig_task_id = resp.json()["result"]
        rig_task = client.poll_rigging_task(rig_task_id)
        rigged = client.download_rigged_model(rig_task, fmt="glb")
        rigged_path = assets_dir / "avatar_chibi_rigged.glb"
        rigged_path.write_bytes(rigged)
        basic_anims = client.download_rigging_animations(rig_task)
        anim_results: dict[str, str] = {}
        for anim_name, anim_bytes in basic_anims.items():
            anim_path = assets_dir / f"anim_{anim_name}.glb"
            anim_path.write_bytes(anim_bytes)
            anim_results[anim_name] = str(anim_path)
        return {
            "rigged_model": str(rigged_path),
            "animations": anim_results,
            "rig_task_id": rig_task_id,
        }

    if tool_name == "generate_animations":
        import httpx as _httpx

        anima_dir = Path(args.pop("anima_dir", ""))
        assets_dir = anima_dir / "assets"
        glb_path = assets_dir / "avatar_chibi.glb"
        if not glb_path.exists():
            return {"error": "No 3D model found for animation"}
        client = MeshyClient()
        data_uri = _image_to_data_uri(
            glb_path.read_bytes(), mime="model/gltf-binary",
        )
        body = {"model_url": data_uri, "height_meters": 1.0}
        resp = _httpx.post(
            MESHY_RIGGING_URL,
            json=body,
            headers=client._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rig_task_id = resp.json()["result"]
        client.poll_rigging_task(rig_task_id)
        anim_map = args.get("animations") or _DEFAULT_ANIMATIONS
        anim_results_gen: dict[str, str] = {}
        for anim_name, action_id in anim_map.items():
            anim_task_id = client.create_animation_task(rig_task_id, action_id)
            anim_task = client.poll_animation_task(anim_task_id)
            anim_bytes = client.download_animation(anim_task, fmt="glb")
            anim_path = assets_dir / f"anim_{anim_name}.glb"
            anim_path.write_bytes(anim_bytes)
            anim_results_gen[anim_name] = str(anim_path)
        return {"animations": anim_results_gen, "rig_task_id": rig_task_id}

    raise ValueError(f"Unknown tool: {tool_name}")
