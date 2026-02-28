# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tool schemas and CLI guide for image generation."""
from __future__ import annotations

__all__ = [
    "get_tool_schemas",
    "get_cli_guide",
]


def get_tool_schemas() -> list[dict]:
    """Return Anthropic tool_use schemas for image generation tools."""
    return [
        {
            "name": "generate_character_assets",
            "description": (
                "Generate a complete set of character avatar assets: "
                "full-body image, bust-up image, chibi image, 3D model, "
                "rigged model with skeleton, and animations "
                "(idle, sitting, waving, talking, walking, running). "
                "Requires NOVELAI_TOKEN, FAL_KEY, and MESHY_API_KEY."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Character appearance description using anime tags. "
                            "Example: '1girl, black hair, long hair, red eyes, "
                            "school uniform, full body, standing, white background'"
                        ),
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Things to avoid in the generated image.",
                        "default": "",
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "fullbody", "bustup", "chibi",
                                "3d", "rigging", "animations",
                            ],
                        },
                        "description": (
                            "Which pipeline steps to run. "
                            "Default: all six steps."
                        ),
                    },
                    "supervisor_name": {
                        "type": "string",
                        "description": (
                            "Supervisor anima name. When specified, the "
                            "supervisor's avatar_fullbody.png is used as "
                            "the Vibe Transfer reference image for style "
                            "consistency."
                        ),
                    },
                    "skip_existing": {
                        "type": "boolean",
                        "description": (
                            "If true, skip steps whose output file already exists."
                        ),
                        "default": True,
                    },
                    "expressions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "neutral", "smile", "laugh",
                                "troubled", "surprised", "thinking",
                                "embarrassed",
                            ],
                        },
                        "description": (
                            "Which bustup expressions to generate. "
                            "Default: neutral, smile, laugh, troubled, surprised."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "generate_fullbody",
            "description": (
                "Generate an anime full-body character image using NovelAI V4.5. "
                "Saves to assets/avatar_fullbody.png. "
                "Requires NOVELAI_TOKEN."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Character appearance tags for NovelAI.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt.",
                        "default": "",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels.",
                        "default": 1024,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels.",
                        "default": 1536,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for reproducibility.",
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "generate_bustup",
            "description": (
                "Generate a bust-up portrait from a reference image "
                "using Flux Kontext [pro]. Saves to assets/avatar_bustup.png. "
                "Requires FAL_KEY and an existing full-body image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Bust-up generation prompt. "
                            "A default prompt is used if omitted."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_chibi",
            "description": (
                "Generate a chibi / super-deformed version from a reference "
                "image using Flux Kontext [pro]. Saves to assets/avatar_chibi.png. "
                "Requires FAL_KEY and an existing full-body image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Chibi generation prompt. "
                            "A default prompt is used if omitted."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_3d_model",
            "description": (
                "Generate a 3D model (GLB) from a chibi image using "
                "Meshy Image-to-3D. Saves to assets/avatar_chibi.glb. "
                "Requires MESHY_API_KEY and an existing chibi image."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ai_model": {
                        "type": "string",
                        "description": "Meshy model version.",
                        "default": "meshy-6",
                        "enum": ["meshy-5", "meshy-6"],
                    },
                    "target_polycount": {
                        "type": "integer",
                        "description": "Target polygon count.",
                        "default": 30000,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_rigged_model",
            "description": (
                "Rig a 3D model with a humanoid skeleton using Meshy. "
                "Also downloads built-in walking/running animations. "
                "Saves rigged model to assets/avatar_chibi_rigged.glb "
                "and animations to assets/anim_walking.glb, anim_running.glb. "
                "Requires MESHY_API_KEY and an existing 3D model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "height_meters": {
                        "type": "number",
                        "description": (
                            "Approximate character height in meters. "
                            "Aids in scaling and rigging accuracy."
                        ),
                        "default": 1.0,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "generate_animations",
            "description": (
                "Generate animation GLBs for a rigged character using "
                "Meshy's animation library. Default: idle, sitting, waving, "
                "talking. Requires MESHY_API_KEY and a completed rigging step."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "animations": {
                        "type": "object",
                        "description": (
                            "Dict of {name: action_id}. "
                            "Default: {idle: 0, sitting: 32, waving: 28, talking: 307}. "
                            "See Meshy animation library for all available action_ids."
                        ),
                        "additionalProperties": {"type": "integer"},
                    },
                },
                "required": [],
            },
        },
    ]


def get_cli_guide() -> str:
    """Return CLI usage guide for image generation tools."""
    return """\
### 画像・3Dモデル生成 (image_gen)

Aモードのツール名: `generate_character_assets` / `generate_fullbody` / `generate_bustup` 等

```bash
# 全6ステップ一括生成（推奨）
animaworks-tool image_gen pipeline "1girl, black hair, ..." --negative "lowres, bad anatomy, ..." --anima-dir <anima_dir> -j

# 個別ステップ
animaworks-tool image_gen fullbody "prompt" --anima-dir <anima_dir> -j
animaworks-tool image_gen bustup --anima-dir <anima_dir> -j
animaworks-tool image_gen chibi --anima-dir <anima_dir> -j
animaworks-tool image_gen 3d --anima-dir <anima_dir> -j
animaworks-tool image_gen rigging <model.glb> -o <output_dir> -j
animaworks-tool image_gen animations <model.glb> -o <output_dir> -j
```"""
