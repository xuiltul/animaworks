from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

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


def create_assets_router() -> APIRouter:
    router = APIRouter()

    @router.get("/persons/{name}/assets")
    async def list_assets(name: str, request: Request):
        """List available assets for a person."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        assets_dir = person_dir / "assets"
        if not assets_dir.exists():
            return {"assets": []}
        return {
            "assets": [
                {"name": f.name, "size": f.stat().st_size}
                for f in sorted(assets_dir.iterdir())
                if f.is_file()
            ]
        }

    @router.get("/persons/{name}/assets/metadata")
    async def get_asset_metadata(name: str, request: Request):
        """Return structured metadata about a person's available assets."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        assets_dir = person_dir / "assets"
        base_url = f"/api/persons/{name}/assets"

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
        identity_path = person_dir / "identity.md"
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

    @router.api_route("/persons/{name}/assets/{filename}", methods=["GET", "HEAD"])
    async def get_asset(name: str, filename: str, request: Request):
        """Serve a static asset file from a person's assets directory."""
        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        # Validate filename (prevent path traversal)
        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = person_dir / "assets" / safe_name
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

    @router.post("/persons/{name}/assets/generate")
    async def generate_assets(
        name: str, body: AssetGenerateRequest, request: Request,
    ):
        """Trigger character asset generation pipeline."""
        import asyncio

        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        if not body.prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(person_dir)

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
            await emit(request, "person.assets_updated", {
                "name": name,
                "assets": generated,
                "errors": result.errors,
            })

        return result.to_dict()

    @router.post("/persons/{name}/assets/generate-expression")
    async def generate_expression_on_demand(
        name: str, body: ExpressionGenerateRequest, request: Request,
    ):
        """Generate a specific bustup expression variant on demand."""
        import asyncio

        persons_dir = request.app.state.persons_dir
        person_dir = persons_dir / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        assets_dir = person_dir / "assets"
        reference_path = assets_dir / "avatar_fullbody.png"
        if not reference_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No reference image (avatar_fullbody.png) found",
            )

        reference_bytes = reference_path.read_bytes()

        from core.tools.image_gen import ImageGenPipeline

        pipeline = ImageGenPipeline(person_dir)

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
            await emit(request, "person.assets_updated", {
                "name": name,
                "assets": [result_path.name],
                "expression": body.expression,
            })

        return {
            "expression": body.expression,
            "path": str(result_path) if result_path else None,
        }

    return router
