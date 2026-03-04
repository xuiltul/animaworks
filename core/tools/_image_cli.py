# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""CLI entry point for ``animaworks-tool image_gen``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from core.tools._base import logger
from core.tools._image_clients import (
    _BUSTUP_PROMPT,
    _CHIBI_PROMPT,
    _DEFAULT_ANIMATIONS,
    _HTTP_TIMEOUT,
    _image_to_data_uri,
    _retry,
    FluxKontextClient,
    MESHY_RIGGING_URL,
    MeshyClient,
    NovelAIClient,
)
from core.tools._image_pipeline import ImageGenPipeline

__all__ = [
    "cli_main",
]


def cli_main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``animaworks-tool image_gen``."""
    parser = argparse.ArgumentParser(
        description="Character image & 3D model generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- pipeline --
    p_pipe = sub.add_parser("pipeline", help="Run full 6-step pipeline")
    p_pipe.add_argument("prompt", help="Character appearance tags")
    p_pipe.add_argument("-n", "--negative", default="", help="Negative prompt")
    p_pipe.add_argument(
        "-d", "--anima-dir", required=True, help="Anima data directory",
    )
    p_pipe.add_argument(
        "--steps", nargs="*",
        choices=["fullbody", "bustup", "chibi", "3d", "rigging", "animations"],
        help="Steps to run (default: all)",
    )
    p_pipe.add_argument(
        "--no-skip", action="store_true", help="Regenerate even if files exist",
    )
    p_pipe.add_argument("-j", "--json", action="store_true", help="JSON output")

    # -- fullbody --
    p_full = sub.add_parser("fullbody", help="Generate full-body image only")
    p_full.add_argument("prompt", help="Character appearance tags")
    p_full.add_argument("-n", "--negative", default="", help="Negative prompt")
    p_full.add_argument("-o", "--output", default="avatar_fullbody.png")
    p_full.add_argument("-W", "--width", type=int, default=1024)
    p_full.add_argument("-H", "--height", type=int, default=1536)
    p_full.add_argument("-s", "--seed", type=int, default=None)
    p_full.add_argument("-j", "--json", action="store_true")

    # -- bustup --
    p_bust = sub.add_parser("bustup", help="Generate bust-up from reference")
    p_bust.add_argument("reference", help="Path to reference image")
    p_bust.add_argument("-p", "--prompt", default=_BUSTUP_PROMPT)
    p_bust.add_argument("-o", "--output", default="avatar_bustup.png")
    p_bust.add_argument("-j", "--json", action="store_true")

    # -- chibi --
    p_chibi = sub.add_parser("chibi", help="Generate chibi from reference")
    p_chibi.add_argument("reference", help="Path to reference image")
    p_chibi.add_argument("-p", "--prompt", default=_CHIBI_PROMPT)
    p_chibi.add_argument("-o", "--output", default="avatar_chibi.png")
    p_chibi.add_argument("-j", "--json", action="store_true")

    # -- 3d --
    p_3d = sub.add_parser("3d", help="Generate 3D model from image")
    p_3d.add_argument("image", help="Path to input image")
    p_3d.add_argument("-o", "--output", default="avatar_chibi.glb")
    p_3d.add_argument("--ai-model", default="meshy-6")
    p_3d.add_argument("--polycount", type=int, default=30000)
    p_3d.add_argument("-j", "--json", action="store_true")

    # -- rigging --
    p_rig = sub.add_parser("rigging", help="Rig a 3D model with skeleton")
    p_rig.add_argument("model", help="Path to GLB model file")
    p_rig.add_argument("-o", "--output-dir", default=".", help="Output directory")
    p_rig.add_argument(
        "--height", type=float, default=1.0,
        help="Character height in meters (default: 1.0 for chibi)",
    )
    p_rig.add_argument("-j", "--json", action="store_true")

    # -- animations --
    p_anim = sub.add_parser("animations", help="Generate animations for rigged model")
    p_anim.add_argument("model", help="Path to GLB model file")
    p_anim.add_argument("-o", "--output-dir", default=".", help="Output directory")
    p_anim.add_argument(
        "--height", type=float, default=1.0,
        help="Character height in meters (default: 1.0 for chibi)",
    )
    p_anim.add_argument(
        "--actions", nargs="*",
        help="Animation names to generate (default: idle sitting waving talking)",
    )
    p_anim.add_argument("-j", "--json", action="store_true")

    args = parser.parse_args(argv)

    # ── Execute ────────────────────────────────────────────

    if args.command == "pipeline":
        from core.config.models import ImageGenConfig, load_config

        try:
            image_config = load_config().image_gen
        except Exception:
            image_config = ImageGenConfig()

        prompt = args.prompt

        # Auto-convert anime prompt when realistic style is configured
        if image_config.image_style == "realistic":
            from core.tools.image_gen import _looks_like_anime_prompt

            if _looks_like_anime_prompt(prompt):
                from core.tools._image_clients import _convert_anime_to_realistic

                prompt = _convert_anime_to_realistic(prompt)
                print(f"[info] Auto-converted anime prompt to realistic", file=sys.stderr)

        pipe = ImageGenPipeline(Path(args.anima_dir), config=image_config)
        result = pipe.generate_all(
            prompt=prompt,
            negative_prompt=args.negative,
            skip_existing=not args.no_skip,
            steps=args.steps,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            for k, v in result.to_dict().items():
                print(f"{k}: {v}")

    elif args.command == "fullbody":
        client = NovelAIClient()
        img = client.generate_fullbody(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "bustup":
        ref_bytes = Path(args.reference).read_bytes()
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_bytes,
            prompt=args.prompt,
            aspect_ratio="3:4",
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "chibi":
        ref_bytes = Path(args.reference).read_bytes()
        client = FluxKontextClient()
        img = client.generate_from_reference(
            reference_image=ref_bytes,
            prompt=args.prompt,
            aspect_ratio="1:1",
        )
        Path(args.output).write_bytes(img)
        out = {"output": args.output, "size": len(img)}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(img)} bytes)")

    elif args.command == "3d":
        img_bytes = Path(args.image).read_bytes()
        client = MeshyClient()
        task_id = client.create_task(
            img_bytes, ai_model=args.ai_model, target_polycount=args.polycount,
        )
        print(f"Meshy task: {task_id}", file=sys.stderr)
        task = client.poll_task(task_id)
        glb = client.download_model(task, fmt="glb")
        Path(args.output).write_bytes(glb)
        out = {"output": args.output, "size": len(glb), "task_id": task_id}
        if args.json:
            print(json.dumps(out))
        else:
            print(f"Saved: {args.output} ({len(glb)} bytes)")

    elif args.command == "rigging":
        model_bytes = Path(args.model).read_bytes()
        data_uri = _image_to_data_uri(model_bytes, mime="model/gltf-binary")
        client = MeshyClient()

        # Submit rigging with model_url (data URI)
        body: dict[str, Any] = {
            "model_url": data_uri,
            "height_meters": args.height,
        }

        def _rig_call() -> str:
            resp = httpx.post(
                MESHY_RIGGING_URL,
                json=body,
                headers=client._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        rig_task_id = _retry(_rig_call, max_retries=1, delay=10.0)
        print(f"Rigging task: {rig_task_id}", file=sys.stderr)
        rig_task = client.poll_rigging_task(rig_task_id)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Any] = {"rig_task_id": rig_task_id}

        # Download rigged model
        rigged = client.download_rigged_model(rig_task, fmt="glb")
        rigged_path = out_dir / "avatar_chibi_rigged.glb"
        rigged_path.write_bytes(rigged)
        outputs["rigged_model"] = str(rigged_path)

        # Download built-in animations
        basic_anims = client.download_rigging_animations(rig_task)
        outputs["animations"] = {}
        for name, anim_bytes in basic_anims.items():
            anim_path = out_dir / f"anim_{name}.glb"
            anim_path.write_bytes(anim_bytes)
            outputs["animations"][name] = str(anim_path)

        if args.json:
            print(json.dumps(outputs, ensure_ascii=False, indent=2))
        else:
            print(f"Rigged model: {outputs['rigged_model']}")
            for name, path in outputs.get("animations", {}).items():
                print(f"Animation '{name}': {path}")

    elif args.command == "animations":
        model_bytes = Path(args.model).read_bytes()
        data_uri = _image_to_data_uri(model_bytes, mime="model/gltf-binary")
        client = MeshyClient()

        # First rig the model
        rig_body: dict[str, Any] = {
            "model_url": data_uri,
            "height_meters": args.height,
        }

        def _rig_call2() -> str:
            resp = httpx.post(
                MESHY_RIGGING_URL,
                json=rig_body,
                headers=client._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["result"]

        rig_task_id = _retry(_rig_call2, max_retries=1, delay=10.0)
        print(f"Rigging task: {rig_task_id}", file=sys.stderr)
        client.poll_rigging_task(rig_task_id)

        # Determine animations to generate
        anim_map = _DEFAULT_ANIMATIONS.copy()
        if args.actions:
            anim_map = {
                name: _DEFAULT_ANIMATIONS.get(name, 0)
                for name in args.actions
            }

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs_anim: dict[str, Any] = {"rig_task_id": rig_task_id, "animations": {}}

        for name, action_id in anim_map.items():
            print(f"Generating '{name}' (action_id={action_id}) …", file=sys.stderr)
            anim_task_id = client.create_animation_task(rig_task_id, action_id)
            anim_task = client.poll_animation_task(anim_task_id)
            anim_bytes = client.download_animation(anim_task, fmt="glb")
            anim_path = out_dir / f"anim_{name}.glb"
            anim_path.write_bytes(anim_bytes)
            outputs_anim["animations"][name] = str(anim_path)

        if args.json:
            print(json.dumps(outputs_anim, ensure_ascii=False, indent=2))
        else:
            for name, path in outputs_anim["animations"].items():
                print(f"Animation '{name}': {path}")
