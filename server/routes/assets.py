from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from server.events import emit

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


class ExpressionGenerateRequest(BaseModel):
    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        from core.schemas import VALID_EMOTIONS
        if v not in VALID_EMOTIONS:
            raise ValueError(f"Invalid expression: {v}. Valid: {', '.join(sorted(VALID_EMOTIONS))}")
        return v


class RemakePreviewRequest(BaseModel):
    style_from: str
    vibe_strength: float = 0.6
    vibe_info_extracted: float = 0.8
    prompt: str | None = None
    seed: int | None = None

    @field_validator("vibe_strength", "vibe_info_extracted")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0 (got {v})")
        return v


class RemakeConfirmRequest(BaseModel):
    backup_id: str

    @field_validator("backup_id")
    @classmethod
    def validate_backup_id(cls, v: str) -> str:
        if not re.match(r"^assets_backup_\d{8}_\d{6}$", v):
            raise ValueError(
                f"Invalid backup_id format: {v}. "
                "Expected: assets_backup_YYYYMMDD_HHMMSS"
            )
        return v


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
            "assets": [
                {"name": f.name, "size": f.stat().st_size}
                for f in sorted(assets_dir.iterdir())
                if f.is_file()
            ]
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

        asset_files = {
            "avatar_fullbody": "avatar_fullbody.png",
            "avatar_bustup": "avatar_bustup.png",
            "avatar_chibi": "avatar_chibi.png",
            "model_chibi": "avatar_chibi.glb",
            "model_rigged": "avatar_chibi_rigged.glb",
        }

        result: dict = {"name": name, "assets": {}, "animations": {}, "colors": None}

        if assets_dir.exists():
            for key, filename_ in asset_files.items():
                path = assets_dir / filename_
                if path.exists():
                    result["assets"][key] = {
                        "filename": filename_,
                        "url": f"{base_url}/{filename_}",
                        "size": path.stat().st_size,
                    }

            for f in sorted(assets_dir.iterdir()):
                if f.is_file() and f.name.startswith("anim_") and f.suffix == ".glb":
                    anim_name = f.stem[len("anim_"):]
                    result["animations"][anim_name] = {
                        "filename": f.name,
                        "url": f"{base_url}/{f.name}",
                        "size": f.stat().st_size,
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

        return result

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

        # Generate ETag from file metadata
        stat = file_path.stat()
        etag = f'"{stat.st_mtime_ns}-{stat.st_size}"'

        # Return 304 Not Modified if ETag matches
        if_none_match = request.headers.get("if-none-match")
        if if_none_match and (
            if_none_match == etag
            or etag in [t.strip() for t in if_none_match.split(",")]
            or if_none_match.strip() == "*"
        ):
            from starlette.responses import Response
            return Response(
                status_code=304,
                headers={
                    "ETag": etag,
                    "Cache-Control": "public, no-cache",
                },
            )

        return FileResponse(
            file_path,
            media_type=content_type,
            headers={
                "Cache-Control": "public, no-cache",
                "ETag": etag,
            },
        )

    @router.post("/animas/{name}/assets/generate")
    async def generate_assets(
        name: str, body: AssetGenerateRequest, request: Request,
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

        pipeline = ImageGenPipeline(anima_dir)

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
        for anim_name, anim_path in result.animation_paths.items():
            generated.append(anim_path.name)

        if generated:
            await emit(request, "anima.assets_updated", {
                "name": name,
                "assets": generated,
                "errors": result.errors,
            })

        return result.to_dict()

    @router.post("/animas/{name}/assets/generate-expression")
    async def generate_expression_on_demand(
        name: str, body: ExpressionGenerateRequest, request: Request,
    ):
        """Generate a specific bustup expression variant on demand."""
        import asyncio

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        assets_dir = anima_dir / "assets"
        reference_path = assets_dir / "avatar_fullbody.png"
        if not reference_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No reference image (avatar_fullbody.png) found",
            )

        reference_bytes = reference_path.read_bytes()

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(anima_dir)

        loop = asyncio.get_running_loop()
        result_path = await loop.run_in_executor(
            None,
            lambda: pipeline.generate_bustup_expression(
                reference_image=reference_bytes,
                expression=body.expression,
                skip_existing=True,
            ),
        )

        if result_path:
            await emit(request, "anima.assets_updated", {
                "name": name,
                "assets": [result_path.name],
                "expression": body.expression,
            })

        return {
            "expression": body.expression,
            "path": str(result_path) if result_path else None,
        }

    # ── Remake Preview / Confirm / Cancel ──────────────────

    @router.post("/animas/{name}/assets/remake-preview")
    async def remake_preview(
        name: str, body: RemakePreviewRequest, request: Request,
    ):
        """Generate a fullbody preview using Vibe Transfer from another anima."""
        import asyncio
        import shutil

        from core.time_utils import now_jst

        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        style_dir = animas_dir / body.style_from
        if not style_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Style reference anima not found: {body.style_from}",
            )

        style_fullbody = style_dir / "assets" / "avatar_fullbody.png"
        if not style_fullbody.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Style reference has no avatar_fullbody.png: {body.style_from}",
            )

        # Resolve prompt
        prompt = body.prompt
        if not prompt:
            prompt_file = anima_dir / "assets" / "prompt.txt"
            if prompt_file.exists():
                prompt = prompt_file.read_text(encoding="utf-8").strip()
            if not prompt:
                raise HTTPException(
                    status_code=400,
                    detail="No prompt available. Provide prompt or create assets/prompt.txt.",
                )

        # Create backup
        assets_dir = anima_dir / "assets"
        ts = now_jst().strftime("%Y%m%d_%H%M%S")
        backup_id = f"assets_backup_{ts}"
        backup_dir = anima_dir / backup_id
        if assets_dir.exists():
            shutil.copytree(assets_dir, backup_dir)
            logger.info("Backup created: %s", backup_dir)

        vibe_image = style_fullbody.read_bytes()

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(anima_dir)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate_all(
                prompt=prompt,
                skip_existing=False,
                steps=["fullbody"],
                vibe_image=vibe_image,
                vibe_strength=body.vibe_strength,
                vibe_info_extracted=body.vibe_info_extracted,
                seed=body.seed,
            ),
        )

        if result.errors:
            # Restore assets from backup on failure
            if backup_dir.exists():
                if assets_dir.exists():
                    shutil.rmtree(assets_dir)
                backup_dir.rename(assets_dir)
                logger.info("Assets restored from backup after failure: %s", name)
            raise HTTPException(
                status_code=500,
                detail=f"Preview generation failed: {'; '.join(result.errors)}",
            )

        preview_url = f"/api/animas/{name}/assets/avatar_fullbody.png"

        await emit(request, "anima.remake_preview_ready", {
            "name": name,
            "preview_url": preview_url,
            "seed_used": body.seed,
            "backup_id": backup_id,
        })

        return {
            "preview_url": preview_url,
            "seed_used": body.seed,
            "backup_id": backup_id,
        }

    @router.post("/animas/{name}/assets/remake-confirm")
    async def remake_confirm(
        name: str, body: RemakeConfirmRequest, request: Request,
    ):
        """Accept the preview and cascade-rebuild all remaining assets."""
        import asyncio

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

        # Verify fullbody was already generated (from preview step)
        fullbody_path = anima_dir / "assets" / "avatar_fullbody.png"
        if not fullbody_path.exists():
            raise HTTPException(
                status_code=400,
                detail="No fullbody preview found. Run remake-preview first.",
            )

        # Resolve prompt
        prompt_file = anima_dir / "assets" / "prompt.txt"
        prompt = ""
        if prompt_file.exists():
            prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="No prompt.txt found in assets. Cannot cascade rebuild.",
            )

        remaining_steps = ["bustup", "chibi", "3d", "rigging", "animations"]

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(anima_dir)

        app = request.app  # Capture app ref, not request (request lifecycle ends)

        async def _run_cascade() -> None:
            try:
                loop = asyncio.get_running_loop()

                def _progress(step: str, status: str, pct: int) -> None:
                    asyncio.run_coroutine_threadsafe(
                        _emit_ws("anima.remake_progress", {
                            "name": name,
                            "step": step,
                            "status": status,
                            "progress_pct": pct,
                        }),
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

                await _emit_ws("anima.remake_complete", {
                    "name": name,
                    "steps_completed": completed,
                    "errors": result.errors,
                })
            except Exception:
                logger.exception("Cascade rebuild failed for %s", name)
                await _emit_ws("anima.remake_complete", {
                    "name": name,
                    "steps_completed": [],
                    "errors": ["Internal error during cascade rebuild"],
                })

        async def _emit_ws(event_type: str, data: dict) -> None:
            ws = getattr(app.state, "ws_manager", None)
            if ws:
                await ws.broadcast({"type": event_type, "data": data})

        # Run cascade in background so the HTTP response returns immediately
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
            (d for d in anima_dir.iterdir()
             if d.is_dir() and d.name.startswith("assets_backup_")),
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

        return {"status": "restored", "backup_used": latest_backup.name}

    return router
