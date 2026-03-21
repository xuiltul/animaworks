---
name: image-gen-tool
description: >-
  Image and 3D model generation tool. Character fullbody, bustup, chibi, and 3D model generation.
  NovelAI/Flux/Meshy support.
  "image generation" "fullbody" "bustup" "chibi" "3D model" "avatar"
tags: [image, 3d, generation, external]
---

# Image Gen Tool

External tool for character image and 3D model generation.

## Invocation via Bash

Use **Bash** with `animaworks-tool image_gen <subcommand> [args]`. For long-running pipelines, use `animaworks-tool submit image_gen pipeline ...`.

## Actions

### character_assets — Full pipeline generation
```json
{"tool_name": "image_gen", "action": "character_assets", "args": {"prompt": "1girl, ...", "anima_dir": "$ANIMAWORKS_ANIMA_DIR"}}
```

### fullbody — Full-body illustration
```json
{"tool_name": "image_gen", "action": "fullbody", "args": {"prompt": "1girl, standing, ...", "width": 832, "height": 1216}}
```

### bustup — Bust-up illustration
```json
{"tool_name": "image_gen", "action": "bustup", "args": {"reference": "source image path", "prompt": "additional prompt (optional)"}}
```

### chibi — Chibi illustration
```json
{"tool_name": "image_gen", "action": "chibi", "args": {"reference": "source image path"}}
```

### 3d_model — 3D model generation
```json
{"tool_name": "image_gen", "action": "3d_model", "args": {"image": "image path"}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
animaworks-tool image_gen fullbody "1girl, standing, ..."
animaworks-tool image_gen bustup reference.png
animaworks-tool image_gen 3d image.png
```

## Notes

- Long-running; use `animaworks-tool submit image_gen pipeline ...` for background execution
- NovelAI or fal.ai API key required for image generation
- Meshy API key required for 3D generation
