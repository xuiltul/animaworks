from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, field_validator

from server.events import emit
from server.routes.media_proxy import proxy_external_image

logger = logging.getLogger("animaworks.routes.assets")

_ASSET_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}


class AssetGenerateRequest(BaseModel):
    prompt: str | None = None
    negative_prompt: str = ""
    steps: list[str] | None = None
    skip_existing: bool = True
    image_style: str | None = None


class ExpressionGenerateRequest(BaseModel):
    expression: str
    image_style: str | None = None

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        from core.schemas import VALID_EMOTIONS

        if v not in VALID_EMOTIONS:
            raise ValueError(f"Invalid expression: {v}. Valid: {', '.join(sorted(VALID_EMOTIONS))}")
        return v


class RemakePreviewRequest(BaseModel):
    style_from: str | None = None
    vibe_strength: float = 0.6
    vibe_info_extracted: float = 0.8
    prompt: str | None = None
    seed: int | None = None
    image_style: str | None = None
    backup_id: str | None = None
    face_reference_url: str | None = None  # URL of a face photo for IP-Adapter

    @field_validator("vibe_strength", "vibe_info_extracted")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0 (got {v})")
        return v

    @field_validator("backup_id")
    @classmethod
    def validate_backup_id(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^assets_backup_\d{8}_\d{6}$", v):
            raise ValueError(f"Invalid backup_id format: {v}. Expected: assets_backup_YYYYMMDD_HHMMSS")
        return v


class RemakeConfirmRequest(BaseModel):
    backup_id: str
    image_style: str | None = None
    preview_file: str | None = None
    fullbody_only: bool = False

    @field_validator("backup_id")
    @classmethod
    def validate_backup_id(cls, v: str) -> str:
        if not re.match(r"^assets_backup_\d{8}_\d{6}$", v):
            raise ValueError(f"Invalid backup_id format: {v}. Expected: assets_backup_YYYYMMDD_HHMMSS")
        return v


def _next_preview_counter(assets_dir: Path, is_realistic: bool) -> int:
    """Return the next sequential preview counter."""
    pattern = "_preview_*_realistic.png" if is_realistic else "_preview_*.png"
    existing = sorted(assets_dir.glob(pattern))
    if not existing:
        return 1
    last = existing[-1].stem
    # Extract digits: _preview_003 or _preview_003_realistic
    parts = last.replace("_realistic", "").split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p) + 1
    return len(existing) + 1


def _effective_image_config(*, image_style: str | None = None, enable_3d: bool | None = None):
    """Load image generation config and apply per-request overrides."""
    from core.config.models import ImageGenConfig, load_config

    try:
        base_config = load_config().image_gen
        updates: dict[str, object] = {}
        if image_style is not None:
            updates["image_style"] = image_style
        if enable_3d is not None:
            updates["enable_3d"] = enable_3d
        return base_config.model_copy(update=updates)
    except Exception:
        kwargs: dict[str, object] = {}
        if image_style is not None:
            kwargs["image_style"] = image_style
        if enable_3d is not None:
            kwargs["enable_3d"] = enable_3d
        return ImageGenConfig(**kwargs)


def create_assets_router() -> APIRouter:
    router = APIRouter()

    @router.get("/animas/{name}/assets")
    async def list_assets(name: str, request: Request):
        """List available assets for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        assets_dir = anima_dir / "assets"
        if not assets_dir.exists():
            return {"assets": []}
        return {
            "assets": [{"name": f.name, "size": f.stat().st_size} for f in sorted(assets_dir.iterdir()) if f.is_file()]
        }

    @router.get("/animas/{name}/assets/metadata")
    async def get_asset_metadata(name: str, request: Request):
        """Return structured metadata about a anima's available assets."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        assets_dir = anima_dir / "assets"
        base_url = f"/api/animas/{name}/assets"

        anime_asset_files = {
            "avatar_fullbody": "avatar_fullbody.png",
            "avatar_bustup": "avatar_bustup.png",
            "avatar_chibi": "avatar_chibi.png",
            "model_chibi": "avatar_chibi.glb",
            "model_rigged": "avatar_chibi_rigged.glb",
        }

        realistic_asset_files = {
            "avatar_fullbody_realistic": "avatar_fullbody_realistic.png",
            "avatar_bustup_realistic": "avatar_bustup_realistic.png",
        }

        # Resolve current display mode from config
        display_mode = "anime"
        try:
            from core.config.models import load_config

            display_mode = load_config().image_gen.image_style or "anime"
        except Exception:
            pass

        result: dict = {
            "name": name,
            "assets": {},
            "assets_realistic": {},
            "animations": {},
            "colors": None,
            "display_mode": display_mode,
        }

        if assets_dir.exists():
            for key, filename_ in anime_asset_files.items():
                path = assets_dir / filename_
                if path.exists():
                    st = path.stat()
                    result["assets"][key] = {
                        "filename": filename_,
                        "url": f"{base_url}/{filename_}?v={int(st.st_mtime)}",
                        "size": st.st_size,
                    }

            for key, filename_ in realistic_asset_files.items():
                path = assets_dir / filename_
                if path.exists():
                    st = path.stat()
                    result["assets_realistic"][key] = {
                        "filename": filename_,
                        "url": f"{base_url}/{filename_}?v={int(st.st_mtime)}",
                        "size": st.st_size,
                    }

            # Scan expression variants
            from core.schemas import VALID_EMOTIONS

            expressions: dict = {}
            for emotion in sorted(VALID_EMOTIONS):
                if emotion == "neutral":
                    continue
                fname = f"avatar_bustup_{emotion}.png"
                path = assets_dir / fname
                if path.exists():
                    st = path.stat()
                    expressions[emotion] = {
                        "filename": fname,
                        "url": f"{base_url}/{fname}?v={int(st.st_mtime)}",
                        "size": st.st_size,
                    }
            result["expressions"] = expressions

            realistic_expressions: dict = {}
            for emotion in sorted(VALID_EMOTIONS):
                if emotion == "neutral":
                    continue
                fname = f"avatar_bustup_{emotion}_realistic.png"
                path = assets_dir / fname
                if path.exists():
                    st = path.stat()
                    realistic_expressions[emotion] = {
                        "filename": fname,
                        "url": f"{base_url}/{fname}?v={int(st.st_mtime)}",
                        "size": st.st_size,
                    }
            result["expressions_realistic"] = realistic_expressions

            for f in sorted(assets_dir.iterdir()):
                if f.is_file() and f.name.startswith("anim_") and f.suffix == ".glb":
                    anim_name = f.stem[len("anim_") :]
                    st = f.stat()
                    result["animations"][anim_name] = {
                        "filename": f.name,
                        "url": f"{base_url}/{f.name}?v={int(st.st_mtime)}",
                        "size": st.st_size,
                    }

        # Extract image_color from identity.md
        identity_path = anima_dir / "identity.md"
        if identity_path.exists():
            try:
                text = identity_path.read_text(encoding="utf-8")
                match = re.search(
                    r"(?:\u30a4\u30e1\u30fc\u30b8\u30ab\u30e9\u30fc|image[_ ]?color|\u30ab\u30e9\u30fc)\s*[:\uff1a]\s*.*?(#[0-9A-Fa-f]{6})",
                    text,
                )
                if match:
                    result["colors"] = {"image_color": match.group(1)}
            except OSError:
                pass

        from starlette.responses import JSONResponse as _StlJSONResponse

        return _StlJSONResponse(
            content=result,
            headers={"Cache-Control": "no-store"},
        )

    @router.api_route("/animas/{name}/assets/{filename}", methods=["GET", "HEAD"])
    async def get_asset(name: str, filename: str, request: Request):
        """Serve a static asset file from a anima's assets directory."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Validate filename (prevent path traversal)
        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = anima_dir / "assets" / safe_name
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")

        suffix = file_path.suffix.lower()
        content_type = _ASSET_CONTENT_TYPES.get(suffix, "application/octet-stream")

        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    @router.api_route("/animas/{name}/attachments/{filename}", methods=["GET", "HEAD"])
    async def get_attachment(name: str, filename: str, request: Request):
        """Serve a user-uploaded attachment from a anima's attachments directory."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = anima_dir / "attachments" / safe_name
        if not file_path.exists() or not file_path.is_file():
            tmp_path = animas_dir.parent / "tmp" / "attachments" / safe_name
            if tmp_path.exists() and tmp_path.is_file():
                file_path = tmp_path
            else:
                raise HTTPException(status_code=404, detail="Attachment not found")

        suffix = file_path.suffix.lower()
        content_type = _ASSET_CONTENT_TYPES.get(suffix, "application/octet-stream")

        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/media/proxy")
    async def media_proxy(url: str, request: Request):
        return await proxy_external_image(url, request)

    @router.post("/animas/{name}/assets/generate")
    async def generate_assets(
        name: str,
        body: AssetGenerateRequest,
        request: Request,
    ):
        """Trigger character asset generation pipeline."""
        import asyncio

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        if not body.prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        from core.tools.image_gen import ImageGenPipeline

        pipeline_kwargs: dict = {}
        if body.image_style:
            pipeline_kwargs["config"] = _effective_image_config(image_style=body.image_style)

        pipeline = ImageGenPipeline(anima_dir, **pipeline_kwargs)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate_all(
                prompt=body.prompt,
                negative_prompt=body.negative_prompt,
                skip_existing=body.skip_existing,
                steps=body.steps,
            ),
        )

        # Broadcast asset update via WebSocket
        generated: list[str] = []
        if result.fullbody_path:
            generated.append("avatar_fullbody.png")
        if result.bustup_path:
            generated.append("avatar_bustup.png")
        if result.chibi_path:
            generated.append("avatar_chibi.png")
        if result.model_path:
            generated.append("avatar_chibi.glb")
        if result.rigged_model_path:
            generated.append("avatar_chibi_rigged.glb")
        for _, anim_path in result.animation_paths.items():
            generated.append(anim_path.name)

        if generated:
            await emit(
                request,
                "anima.assets_updated",
                {
                    "name": name,
                    "assets": generated,
                    "errors": result.errors,
                },
            )

        return result.to_dict()

    @router.post("/animas/{name}/assets/generate-expression")
    async def generate_expression_on_demand(
        name: str,
        body: ExpressionGenerateRequest,
        request: Request,
    ):
        """Generate a specific bustup expression variant on demand."""
        import asyncio

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        is_realistic = body.image_style == "realistic"
        assets_dir = anima_dir / "assets"
        ref_filename = "avatar_fullbody_realistic.png" if is_realistic else "avatar_fullbody.png"
        reference_path = assets_dir / ref_filename
        if not reference_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No reference image ({ref_filename}) found",
            )

        reference_bytes = reference_path.read_bytes()

        from core.tools.image_gen import ImageGenPipeline

        style = body.image_style or "anime"
        pipeline = ImageGenPipeline(
            anima_dir,
            config=_effective_image_config(image_style=style),
        )

        loop = asyncio.get_running_loop()
        result_path = await loop.run_in_executor(
            None,
            lambda: pipeline.generate_bustup_expression(
                reference_image=reference_bytes,
                expression=body.expression,
                skip_existing=False,
            ),
        )

        if result_path:
            await emit(
                request,
                "anima.assets_updated",
                {
                    "name": name,
                    "assets": [result_path.name],
                    "expression": body.expression,
                },
            )

        return {
            "expression": body.expression,
            "path": str(result_path) if result_path else None,
            "url": f"/api/animas/{name}/assets/{result_path.name}" if result_path else None,
        }

    # ── Remake Preview / Confirm / Cancel ──────────────────

    @router.post("/animas/{name}/assets/remake-preview")
    async def remake_preview(
        name: str,
        body: RemakePreviewRequest,
        request: Request,
    ):
        """Generate a fullbody preview, optionally using Vibe Transfer."""
        import asyncio
        import shutil

        from core.time_utils import now_local

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Resolve image_style: request body > config default (not "anime" hardcode)
        style = body.image_style
        if not style:
            try:
                from core.config.models import load_config

                style = load_config().image_gen.image_style or "realistic"
            except Exception:
                style = "realistic"
        is_realistic = style == "realistic"

        # Resolve vibe image when style_from is provided
        vibe_image: bytes | None = None
        if body.style_from:
            style_dir = animas_dir / body.style_from
            if not style_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Style reference anima not found: {body.style_from}",
                )
            style_ref_filename = "avatar_fullbody_realistic.png" if is_realistic else "avatar_fullbody.png"
            style_fullbody = style_dir / "assets" / style_ref_filename
            if not style_fullbody.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Style reference has no {style_ref_filename}: {body.style_from}",
                )
            vibe_image = style_fullbody.read_bytes()

        # Download face reference image if URL is provided
        face_reference_bytes: bytes | None = None
        if body.face_reference_url:
            logger.info("Downloading face reference from: %s", body.face_reference_url)
            try:
                face_resp = await proxy_external_image(body.face_reference_url, request)
                face_reference_bytes = face_resp.body
                logger.info("Face reference downloaded: %d bytes", len(face_reference_bytes))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download face reference image: {exc}",
                ) from exc

        # Resolve prompt (style-aware) — try cached first, then LLM synthesis, then fallback
        prompt = body.prompt
        if not prompt:
            from core.asset_reconciler import _extract_prompt, _resolve_prompt

            # Fast path: check cached prompt files
            prompt = _resolve_prompt(anima_dir, style="realistic" if is_realistic else "anime")
            if not prompt:
                # Slow path: extract from identity.md / character_sheet.md via LLM
                logger.info("No cached prompt for '%s', attempting LLM synthesis…", name)
                prompt = await _extract_prompt(anima_dir, style="realistic" if is_realistic else "anime")
            if not prompt:
                # Fallback: generate a default prompt from the anima name
                logger.info("No appearance data for '%s', using default prompt", name)
                if is_realistic:
                    prompt = (
                        f"realistic full-length full-body photo of a professional young Japanese person "
                        f"named {name}, solo, single subject, one person only, standing naturally, "
                        f"centered composition, facing camera, plain modern office lobby background, "
                        f"minimal background clutter, friendly approachable expression, neat appearance, "
                        f"clean professional attire, polished office style, natural lighting, high detail, "
                        f"no other people, no crowd, no duplicate body, no extra limbs"
                    )
                else:
                    prompt = (
                        f"1girl, {name}, solo, standing, full body, simple background, "
                        f"office lady, professional attire, neat hair, smile, looking at viewer, "
                        f"high quality, detailed"
                    )
                # Cache for future use
                try:
                    cache_dir = anima_dir / "assets"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_name = "prompt_realistic.txt" if is_realistic else "prompt.txt"
                    (cache_dir / cache_name).write_text(prompt + "\n", encoding="utf-8")
                    logger.info("Cached default %s prompt for '%s'", "realistic" if is_realistic else "anime", name)
                except OSError:
                    pass

        # Reuse existing backup or create a new one
        assets_dir = anima_dir / "assets"
        backup_id = body.backup_id
        if backup_id:
            backup_dir = anima_dir / backup_id
            if not backup_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Backup not found: {backup_id}",
                )
        else:
            ts = now_local().strftime("%Y%m%d_%H%M%S")
            backup_id = f"assets_backup_{ts}"
            backup_dir = anima_dir / backup_id
            if assets_dir.exists():
                shutil.copytree(assets_dir, backup_dir)
                logger.info("Backup created: %s", backup_dir)

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(
            anima_dir,
            config=_effective_image_config(image_style=style or "realistic"),
        )

        gen_kwargs: dict = {
            "prompt": prompt,
            "skip_existing": False,
            "steps": ["fullbody"],
            "seed": body.seed,
        }
        if vibe_image is not None:
            gen_kwargs["vibe_image"] = vibe_image
            gen_kwargs["vibe_strength"] = body.vibe_strength
            gen_kwargs["vibe_info_extracted"] = body.vibe_info_extracted
        if face_reference_bytes is not None:
            gen_kwargs["face_reference_image"] = face_reference_bytes

        ws_manager = getattr(request.app.state, "ws_manager", None)

        # Detect low-VRAM mode so the UI can warn before starting
        _low_vram_mode = False
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _free_vram, _ = _torch.cuda.mem_get_info()
                _low_vram_mode = _free_vram < 7 * 1024 ** 3
        except Exception:
            pass

        # Emit start event immediately so UI shows spinner
        await emit(
            request,
            "anima.image_gen_progress",
            {
                "name": name,
                "step": "fullbody",
                "current": 0,
                "total": 0,
                "pct": 0,
                "low_vram": _low_vram_mode,
            },
        )

        # Capture everything needed for the background task before returning HTTP
        _backup_id_original = body.backup_id
        _seed_used = body.seed
        _app_state = request.app.state

        async def _bg_generate() -> None:
            import shutil as _shutil
            import time as _time

            bg_loop = asyncio.get_running_loop()

            def _on_step(current: int, total: int) -> None:
                _ws = getattr(_app_state, "ws_manager", None)
                if _ws is None:
                    return
                pct = int(current * 100 / total) if total > 0 else 0
                try:
                    asyncio.run_coroutine_threadsafe(
                        _ws.broadcast({
                            "type": "anima.image_gen_progress",
                            "data": {
                                "name": name,
                                "step": "fullbody",
                                "current": current,
                                "total": total,
                                "pct": pct,
                            },
                        }),
                        bg_loop,
                    )
                except Exception:
                    pass

            gen_kwargs["fullbody_step_callback"] = _on_step

            async def _ws_emit(etype: str, data: dict) -> None:
                _ws = getattr(_app_state, "ws_manager", None)
                if _ws:
                    try:
                        await _ws.broadcast({"type": etype, "data": data})
                    except Exception:
                        pass

            try:
                result = await bg_loop.run_in_executor(
                    None,
                    lambda: pipeline.generate_all(**gen_kwargs),
                )

                if result.errors:
                    if not _backup_id_original and backup_dir.exists():
                        if assets_dir.exists():
                            _shutil.rmtree(assets_dir)
                        backup_dir.rename(assets_dir)
                        logger.info("Assets restored from backup after failure: %s", name)
                    await _ws_emit("anima.image_gen_progress", {
                        "name": name,
                        "step": "fullbody",
                        "current": -1,
                        "total": -1,
                        "pct": -1,
                        "error": "; ".join(result.errors),
                    })
                    return

                # Save preview as numbered file for history navigation
                output_filename = "avatar_fullbody_realistic.png" if is_realistic else "avatar_fullbody.png"
                source_path = assets_dir / output_filename
                counter = _next_preview_counter(assets_dir, is_realistic)
                preview_filename = (
                    f"_preview_{counter:03d}_realistic.png" if is_realistic else f"_preview_{counter:03d}.png"
                )
                if source_path.exists():
                    _shutil.copy2(source_path, assets_dir / preview_filename)

                preview_url = f"/api/animas/{name}/assets/{preview_filename}?v={int(_time.time())}"
                await _ws_emit("anima.remake_preview_ready", {
                    "name": name,
                    "preview_url": preview_url,
                    "preview_file": preview_filename,
                    "seed_used": _seed_used,
                    "backup_id": backup_id,
                })

            except Exception as exc:
                logger.exception("Background fullbody generation failed for %s", name)
                await _ws_emit("anima.image_gen_progress", {
                    "name": name,
                    "step": "fullbody",
                    "current": -1,
                    "total": -1,
                    "pct": -1,
                    "error": str(exc),
                })

        asyncio.create_task(_bg_generate())

        # Return 202 immediately — result arrives via WebSocket
        return JSONResponse(
            status_code=202,
            content={"status": "generating", "backup_id": backup_id},
        )

    @router.post("/animas/{name}/assets/remake-confirm")
    async def remake_confirm(
        name: str,
        body: RemakeConfirmRequest,
        request: Request,
    ):
        """Accept the preview and cascade-rebuild all remaining assets."""
        import asyncio
        import shutil as _sh

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Verify backup exists
        backup_dir = anima_dir / body.backup_id
        if not backup_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Backup not found: {body.backup_id}",
            )

        # Resolve image_style: request body > config default
        style = body.image_style
        if not style:
            try:
                from core.config.models import load_config

                style = load_config().image_gen.image_style or "realistic"
            except Exception:
                style = "realistic"
        is_realistic = style == "realistic"
        fullbody_filename = "avatar_fullbody_realistic.png" if is_realistic else "avatar_fullbody.png"
        assets_dir = anima_dir / "assets"

        # Copy selected preview file to the canonical fullbody path
        if body.preview_file:
            preview_path = assets_dir / body.preview_file
            if not preview_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Preview file not found: {body.preview_file}",
                )
            _sh.copy2(preview_path, assets_dir / fullbody_filename)

        fullbody_path = assets_dir / fullbody_filename
        if not fullbody_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"No {fullbody_filename} preview found. Run remake-preview first.",
            )

        # ── Fullbody-only mode: copy fullbody to all other slots ──
        if body.fullbody_only:
            app = request.app

            async def _run_fullbody_copy() -> None:
                try:
                    import shutil as _sh2

                    # Determine target files
                    bustup_name = "avatar_bustup_realistic.png" if is_realistic else "avatar_bustup.png"
                    _sh2.copy2(fullbody_path, assets_dir / bustup_name)
                    await _emit_ws("anima.remake_progress", {"name": name, "step": "bustup", "status": "completed", "progress_pct": 100})

                    # Copy fullbody to all expression variants
                    style_suffix = "_realistic" if is_realistic else ""
                    expressions = ["smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"]
                    for i, expr in enumerate(expressions):
                        expr_name = f"avatar_bustup_{expr}{style_suffix}.png"
                        _sh2.copy2(fullbody_path, assets_dir / expr_name)
                        await _emit_ws("anima.remake_progress", {"name": name, "step": f"expression:{expr}", "status": "completed", "progress_pct": 100})

                    # Clean up backup
                    if backup_dir.exists():
                        _sh2.rmtree(backup_dir, ignore_errors=True)
                        logger.info("Removed backup after fullbody-only copy: %s", backup_dir.name)

                    await _emit_ws("anima.remake_complete", {"name": name, "steps_completed": ["fullbody_copy"], "errors": []})
                except Exception:
                    logger.exception("Fullbody-only copy failed for %s", name)
                    await _emit_ws("anima.remake_complete", {"name": name, "steps_completed": [], "errors": ["Internal error during fullbody copy"]})

            async def _emit_ws(event_type: str, data: dict) -> None:
                ws = getattr(app.state, "ws_manager", None)
                if ws:
                    await ws.broadcast({"type": event_type, "data": data})

            remaining_steps = ["bustup"] + [f"expression:{e}" for e in ["smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"]]
            asyncio.create_task(_run_fullbody_copy())
            return {"status": "started", "steps": remaining_steps, "mode": "fullbody_only"}

        # ── Normal cascade rebuild ──
        # Resolve prompt (style-aware)
        from core.asset_reconciler import _resolve_prompt

        prompt = (
            _resolve_prompt(
                anima_dir,
                style="realistic" if is_realistic else "anime",
            )
            or ""
        )
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="No prompt available in assets. Cannot cascade rebuild.",
            )

        if is_realistic:
            remaining_steps = ["bustup"]
        else:
            remaining_steps = ["bustup", "chibi", "3d", "rigging", "animations"]

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(
            anima_dir,
            config=_effective_image_config(image_style=style or "realistic"),
        )

        app = request.app

        async def _run_cascade() -> None:
            try:
                loop = asyncio.get_running_loop()

                def _progress(step: str, status: str, pct: int) -> None:
                    asyncio.run_coroutine_threadsafe(
                        _emit_ws(
                            "anima.remake_progress",
                            {
                                "name": name,
                                "step": step,
                                "status": status,
                                "progress_pct": pct,
                            },
                        ),
                        loop,
                    )

                result = await loop.run_in_executor(
                    None,
                    lambda: pipeline.generate_all(
                        prompt=prompt,
                        skip_existing=False,
                        steps=remaining_steps,
                        progress_callback=_progress,
                    ),
                )

                completed = []
                if result.bustup_paths:
                    completed.append("bustup")
                if result.chibi_path:
                    completed.append("chibi")
                if result.model_path:
                    completed.append("3d")
                if result.rigged_model_path:
                    completed.append("rigging")
                if result.animation_paths:
                    completed.append("animations")

                # Clean up backup after successful rebuild — the new assets
                # are canonical now; keeping the backup would let a stale
                # DELETE /remake-preview restore outdated files.
                if backup_dir.exists():
                    import shutil as _sh2
                    _sh2.rmtree(backup_dir, ignore_errors=True)
                    logger.info("Removed backup after successful rebuild: %s", backup_dir.name)

                await _emit_ws(
                    "anima.remake_complete",
                    {
                        "name": name,
                        "steps_completed": completed,
                        "errors": result.errors,
                    },
                )
            except Exception:
                logger.exception("Cascade rebuild failed for %s", name)
                await _emit_ws(
                    "anima.remake_complete",
                    {
                        "name": name,
                        "steps_completed": [],
                        "errors": ["Internal error during cascade rebuild"],
                    },
                )

        async def _emit_ws(event_type: str, data: dict) -> None:
            ws = getattr(app.state, "ws_manager", None)
            if ws:
                await ws.broadcast({"type": event_type, "data": data})

        asyncio.create_task(_run_cascade())

        return {
            "status": "started",
            "steps": remaining_steps,
        }

    @router.delete("/animas/{name}/assets/remake-preview")
    async def cancel_remake_preview(name: str, request: Request):
        """Cancel a remake preview by restoring from the most recent backup."""
        import shutil

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Find the most recent backup
        backups = sorted(
            (d for d in anima_dir.iterdir() if d.is_dir() and d.name.startswith("assets_backup_")),
            reverse=True,
        )
        if not backups:
            raise HTTPException(status_code=404, detail="No backup found to restore")

        latest_backup = backups[0]
        assets_dir = anima_dir / "assets"

        # Restore from backup
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        latest_backup.rename(assets_dir)
        logger.info("Restored assets from backup: %s", latest_backup.name)

        # Clean up any remaining preview files
        for pf in assets_dir.glob("_preview_*.png"):
            pf.unlink(missing_ok=True)

        return {"status": "restored", "backup_used": latest_backup.name}

    return router
