# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Character image & 3-D model generation tool for AnimaWorks.

Facade module — re-exports all public symbols from sub-modules and
provides the :func:`dispatch` entry point used by the tool handler.

Pipeline (7 steps):
  1. NovelAI V4.5 → anime full-body image (fallback: fal.ai Flux Pro)
  2. Flux Kontext [pro] (fal.ai) → bust-up from reference
  3. Flux Kontext [pro] (fal.ai) → icon from neutral bustup
  4. Flux Kontext [pro] (fal.ai) → chibi from reference
  5. Meshy Image-to-3D → GLB model from chibi image
  6. Meshy Rigging → rigged GLB + walking/running animations
  7. Meshy Animations → idle/sitting/waving/talking GLBs
"""

from __future__ import annotations

import os  # noqa: F401 — patch compatibility
import sys
import time  # noqa: F401 — patch compatibility
from pathlib import Path
from typing import Any

import httpx  # noqa: F401 — patch compatibility

# ── Mutable cache proxy ───────────────────────────────
# Tests manipulate _FBX2GLTF_PATH / _GLTF_MODULES_DIR via the facade module.
# These live in _image_glb; we proxy reads and writes so that
# ``import core.integrations.image_gen as mod; mod._FBX2GLTF_PATH = X`` propagates.
import core.integrations._image_glb as _glb_mod  # noqa: E402
from core.credentials import get_credential  # noqa: F401 — patch compatibility
from core.integrations._base import ToolConfigError, dispatch_by_table, logger  # noqa: F401 — patch compatibility

# ── Re-exports: _image_cli ─────────────────────────────────
from core.integrations._image_cli import cli_main

# ── Re-exports: _image_clients ─────────────────────────────
from core.integrations._image_clients import (
    _BUSTUP_PROMPT,
    _CHAT_ICON_PROMPT,
    _CHIBI_PROMPT,
    _DEFAULT_ANIMATIONS,
    _EXPRESSION_GUIDANCE,
    _EXPRESSION_PROMPTS,
    _HTTP_TIMEOUT,
    _REALISTIC_CHAT_ICON_PROMPT,
    EXECUTION_PROFILE,
    FAL_FLUX_PRO_SUBMIT_URL,
    MESHY_RIGGING_URL,
    NOVELAI_API_URL,
    NOVELAI_ENCODE_URL,
    NOVELAI_MODEL,
    CodexFirstClient,
    FalTextToImageClient,
    FluxKontextClient,
    LocalDiffusersClient,
    MeshyClient,
    NovelAIClient,
    _image_to_data_uri,
    _retry,
    codex_available,
)

# ── Re-exports: _image_glb ────────────────────────────────
from core.integrations._image_glb import (
    _convert_fbx_to_glb,
    _download_armature_animation,
    _ensure_fbx2gltf,
    _ensure_gltf_transform_modules,
    _run_gltf_transform,
    compress_textures,
    optimize_glb,
    simplify_glb,
    strip_mesh_from_glb,
)

# ── Re-exports: _image_pipeline ────────────────────────────
from core.integrations._image_pipeline import ImageGenPipeline, PipelineResult

# ── Re-exports: _image_schemas ─────────────────────────────
from core.integrations._image_schemas import get_tool_schemas
from core.schemas import VALID_EMOTIONS as _VALID_EXPRESSION_NAMES

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

_ANIME_MARKER_TAGS = frozenset(
    {
        "anime coloring",
        "clean lineart",
        "soft shading",
        "masterpiece",
        "best quality",
        "absurdres",
    }
)


def _looks_like_anime_prompt(prompt: str) -> bool:
    """Heuristic: return True if prompt contains Danbooru-style anime tags."""
    tags = {t.strip().lower() for t in prompt.split(",")}
    return len(tags & _ANIME_MARKER_TAGS) >= 3


def _use_diffusers_backend(image_config: Any) -> bool:
    return getattr(image_config, "backend", "api") == "diffusers"


def _has_image_credential(credential_name: str, env_var: str) -> bool:
    """Check a backend credential through the shared vault/config/env resolver."""
    try:
        return bool(get_credential(credential_name, "image_gen", env_var=env_var))
    except ToolConfigError:
        return False


def _build_fullbody_api_client(image_config: Any) -> Any:
    """Select the API fullbody client (NovelAI / Fal). Used as codex fallback."""
    if getattr(image_config, "image_style", "anime") == "realistic":
        if not _has_image_credential("fal", "FAL_KEY"):
            raise RuntimeError("FAL_KEY required for realistic image generation.")
        return FalTextToImageClient()
    if _has_image_credential("novelai", "NOVELAI_TOKEN"):
        return NovelAIClient()
    if _has_image_credential("fal", "FAL_KEY"):
        return FalTextToImageClient()
    raise RuntimeError("No image generation backend configured. Enable Diffusers or set NOVELAI_TOKEN/FAL_KEY.")


def _build_fullbody_client(image_config: Any) -> Any:
    if _use_diffusers_backend(image_config):
        return LocalDiffusersClient(image_config)
    if getattr(image_config, "prefer_codex", True) and codex_available():
        return CodexFirstClient(
            fallback_factory=lambda: _build_fullbody_api_client(image_config),
            image_config=image_config,
        )
    return _build_fullbody_api_client(image_config)


def _build_reference_client(image_config: Any) -> Any:
    if _use_diffusers_backend(image_config):
        return LocalDiffusersClient(image_config)
    if getattr(image_config, "prefer_codex", True) and codex_available():
        return CodexFirstClient(
            fallback_factory=FluxKontextClient,
            image_config=image_config,
        )
    return FluxKontextClient()


# ── Dispatch ──────────────────────────────────────────


def _dispatch_character_assets(args: dict[str, Any]) -> Any:
    from core.config.models import load_config
    from core.paths import get_animas_dir

    anima_dir = Path(args.pop("anima_dir", ""))
    supervisor_name: str | None = args.pop("supervisor_name", None)
    config = load_config()
    image_config = config.image_gen
    prompt = args["prompt"]

    # Auto-convert anime prompt when realistic style is configured
    if image_config.image_style == "realistic" and _looks_like_anime_prompt(prompt):
        from core.integrations._image_clients import _convert_anime_to_realistic

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
        ref_name = "avatar_fullbody_realistic.png" if image_config.image_style == "realistic" else "avatar_fullbody.png"
        supervisor_fullbody = get_animas_dir() / supervisor_name / "assets" / ref_name
        if supervisor_fullbody.exists():
            image_config = image_config.model_copy(update={"style_reference": str(supervisor_fullbody)})
            logger.info("Using supervisor image as vibe reference: %s", supervisor_fullbody)

    pipeline = ImageGenPipeline(anima_dir, config=image_config)
    result = pipeline.generate_all(
        prompt=prompt,
        negative_prompt=args.get("negative_prompt", ""),
        skip_existing=args.get("skip_existing", True),
        steps=args.get("steps"),
        animations=args.get("animations"),
    )
    return result.to_dict()


def _dispatch_fullbody(args: dict[str, Any]) -> Any:
    from core.config.models import load_config

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_config = load_config().image_gen
    client = _build_fullbody_client(image_config)
    image = client.generate_fullbody(
        prompt=args["prompt"],
        negative_prompt=args.get("negative_prompt", ""),
        width=args.get("width", 1024),
        height=args.get("height", 1536),
        seed=args.get("seed"),
    )
    output = assets_dir / "avatar_fullbody.png"
    output.write_bytes(image)
    return {"path": str(output), "size": len(image)}


def _dispatch_bustup(args: dict[str, Any]) -> Any:
    from core.config.models import load_config

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    reference_path = assets_dir / "avatar_fullbody.png"
    if not reference_path.exists():
        return {"error": "No full-body reference image found"}
    image_config = load_config().image_gen
    client = _build_reference_client(image_config)
    image = client.generate_from_reference(
        reference_image=reference_path.read_bytes(),
        prompt=args.get("prompt", _BUSTUP_PROMPT),
        aspect_ratio="3:4",
    )
    output = assets_dir / "avatar_bustup.png"
    output.write_bytes(image)
    return {"path": str(output), "size": len(image)}


def _dispatch_icon(args: dict[str, Any]) -> Any:
    from core.config.models import load_config

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    config = load_config().image_gen
    is_realistic = config.image_style == "realistic"
    ref_name = "avatar_bustup_realistic.png" if is_realistic else "avatar_bustup.png"
    reference_path = assets_dir / ref_name
    if not reference_path.exists():
        return {"error": f"No bustup reference image found ({ref_name})"}
    prompt = args.get("prompt") or (_REALISTIC_CHAT_ICON_PROMPT if is_realistic else _CHAT_ICON_PROMPT)
    if config.style_prefix:
        prompt = config.style_prefix + prompt
    if config.style_suffix:
        prompt = prompt + config.style_suffix
    client = _build_reference_client(config)
    image = client.generate_from_reference(
        reference_image=reference_path.read_bytes(),
        prompt=prompt,
        aspect_ratio="1:1",
        # guidance_scale=float(args.get("guidance_scale", 4.0)),
        seed=args.get("seed"),
    )
    output_name = "icon_realistic.png" if is_realistic else "icon.png"
    output = assets_dir / output_name
    output.write_bytes(image)
    try:
        from core.integrations._anima_icon_url import persist_anima_icon_path_template

        persist_anima_icon_path_template()
    except Exception:
        logger.debug("persist_anima_icon_path_template failed after icon generation", exc_info=True)
    return {"path": str(output), "size": len(image)}


def _dispatch_chibi(args: dict[str, Any]) -> Any:
    from core.config.models import load_config

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    reference_path = assets_dir / "avatar_fullbody.png"
    if not reference_path.exists():
        return {"error": "No full-body reference image found"}
    image_config = load_config().image_gen
    client = _build_reference_client(image_config)
    image = client.generate_from_reference(
        reference_image=reference_path.read_bytes(),
        prompt=args.get("prompt", _CHIBI_PROMPT),
        aspect_ratio="1:1",
    )
    output = assets_dir / "avatar_chibi.png"
    output.write_bytes(image)
    return {"path": str(output), "size": len(image)}


def _dispatch_3d_model(args: dict[str, Any]) -> Any:
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
    output = assets_dir / "avatar_chibi.glb"
    output.write_bytes(glb)
    return {"path": str(output), "size": len(glb), "task_id": task_id}


def _dispatch_rigged_model(args: dict[str, Any]) -> Any:
    import httpx as _httpx

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    glb_path = assets_dir / "avatar_chibi.glb"
    if not glb_path.exists():
        return {"error": "No 3D model found for rigging"}
    client = MeshyClient()
    data_uri = _image_to_data_uri(glb_path.read_bytes(), mime="model/gltf-binary")
    body = {"model_url": data_uri, "height_meters": args.get("height_meters", 1.0)}
    response = _httpx.post(
        MESHY_RIGGING_URL,
        json=body,
        headers=client._headers(),
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    rig_task_id = response.json()["result"]
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
    return {"rigged_model": str(rigged_path), "animations": anim_results, "rig_task_id": rig_task_id}


def _dispatch_animations(args: dict[str, Any]) -> Any:
    import httpx as _httpx

    anima_dir = Path(args.pop("anima_dir", ""))
    assets_dir = anima_dir / "assets"
    glb_path = assets_dir / "avatar_chibi.glb"
    if not glb_path.exists():
        return {"error": "No 3D model found for animation"}
    client = MeshyClient()
    data_uri = _image_to_data_uri(glb_path.read_bytes(), mime="model/gltf-binary")
    response = _httpx.post(
        MESHY_RIGGING_URL,
        json={"model_url": data_uri, "height_meters": 1.0},
        headers=client._headers(),
        timeout=_HTTP_TIMEOUT,
    )
    response.raise_for_status()
    rig_task_id = response.json()["result"]
    client.poll_rigging_task(rig_task_id)
    animation_map = args.get("animations") or _DEFAULT_ANIMATIONS
    animation_results: dict[str, str] = {}
    for animation_name, action_id in animation_map.items():
        animation_task_id = client.create_animation_task(rig_task_id, action_id)
        animation_task = client.poll_animation_task(animation_task_id)
        animation_bytes = client.download_animation(animation_task, fmt="glb")
        animation_path = assets_dir / f"anim_{animation_name}.glb"
        animation_path.write_bytes(animation_bytes)
        animation_results[animation_name] = str(animation_path)
    return {"animations": animation_results, "rig_task_id": rig_task_id}


_DISPATCH_HANDLERS = {
    "generate_character_assets": _dispatch_character_assets,
    "generate_fullbody": _dispatch_fullbody,
    "generate_bustup": _dispatch_bustup,
    "generate_icon": _dispatch_icon,
    "generate_chibi": _dispatch_chibi,
    "generate_3d_model": _dispatch_3d_model,
    "generate_rigged_model": _dispatch_rigged_model,
    "generate_animations": _dispatch_animations,
}


def dispatch(tool_name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    return dispatch_by_table(_DISPATCH_HANDLERS, tool_name, args)
