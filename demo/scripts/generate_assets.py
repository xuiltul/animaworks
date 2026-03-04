#!/usr/bin/env python3
"""Generate demo character assets using AnimaWorks image generation pipeline.

Usage:
    python demo/scripts/generate_assets.py [--preset PRESET] [--character NAME]
    python demo/scripts/generate_assets.py --preset ja-anime --character kaito

Requires NOVELAI_TOKEN and/or FAL_KEY environment variables.

Pipeline per style:
  anime:     NovelAI fullbody → Flux Kontext bustup/chibi
  realistic: Flux fullbody/bustup + shared chibi (no _realistic suffix)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DEMO_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# ── Preset definitions ────────────────────────────────────────

PRESETS: dict[str, dict[str, object]] = {
    "ja-anime": {
        "characters": ["kaito", "sora", "hina"],
        "style": "anime",
    },
    "ja-business": {
        "characters": ["kaito", "sora", "hina"],
        "style": "realistic",
    },
    "en-anime": {
        "characters": ["alex", "kai", "nova"],
        "style": "anime",
    },
    "en-business": {
        "characters": ["alex", "kai", "nova"],
        "style": "realistic",
    },
}


# ── Prompt extraction ─────────────────────────────────────────


def _extract_appearance(md_path: Path) -> str | None:
    """Extract appearance description from a character sheet markdown."""
    text = md_path.read_text(encoding="utf-8")

    patterns = [
        r"(?:image[_ ]?prompt|画像プロンプト)\s*[:\uff1a]\s*(.+)",
        r"(?:キャラクターデザイン|character[_ ]?design)\s*[:\uff1a]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fall back to ## 外見 / ## Appearance section
    section_pat = re.compile(
        r"^##\s+(?:外見|Appearance)\s*\n(.*?)(?=\n##|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = section_pat.search(text)
    if match:
        return match.group(1).strip()

    return None


def _build_prompt_for_style(appearance: str, style: str) -> str:
    """Convert appearance description to a generation prompt."""
    if style == "anime":
        return (
            f"1girl, {appearance}, full body, standing, white background, "
            "anime illustration, high quality, detailed"
        )
    return (
        f"A professional photo of a person: {appearance}. "
        "Full body, standing, studio lighting, neutral background, "
        "high quality photograph"
    )


# ── Generation ────────────────────────────────────────────────


def generate_character(
    preset_name: str,
    character_name: str,
    style: str,
) -> None:
    """Generate all asset images for a single character."""
    from core.config.models import ImageGenConfig
    from core.tools._image_pipeline import ImageGenPipeline

    preset_dir = DEMO_DIR / "presets" / preset_name
    md_path = preset_dir / "characters" / f"{character_name}.md"
    assets_dir = preset_dir / "assets" / character_name

    if not md_path.exists():
        print(f"  SKIP {character_name}: character sheet not found at {md_path}")
        return

    appearance = _extract_appearance(md_path)
    if not appearance:
        print(f"  SKIP {character_name}: no appearance description found")
        return

    prompt = _build_prompt_for_style(appearance, style)
    print(f"  Prompt: {prompt[:120]}...")

    assets_dir.mkdir(parents=True, exist_ok=True)

    enable_3d = False
    image_config = ImageGenConfig(image_style=style, enable_3d=enable_3d)

    # ImageGenPipeline expects an anima_dir with assets/ subdirectory.
    # We create a temporary wrapper that points assets/ to our target.
    # Since ImageGenPipeline uses anima_dir / "assets", we pass
    # the parent of assets_dir as anima_dir.
    anima_dir_proxy = assets_dir.parent
    (anima_dir_proxy / "assets").mkdir(parents=True, exist_ok=True)
    # Symlink or just use the right structure — assets_dir IS
    # preset_dir/assets/{name}, so anima_dir_proxy/assets = assets_dir
    # only if anima_dir_proxy = preset_dir/assets/{name} parent.
    # Actually: ImageGenPipeline does self._assets_dir = anima_dir / "assets"
    # We want output in preset_dir/assets/{name}/
    # So anima_dir should be preset_dir/assets/{name} and pipeline
    # writes to preset_dir/assets/{name}/assets/ — that's wrong.
    #
    # Instead, we use the pipeline directly with the correct paths.
    # We'll call the lower-level clients ourselves.

    _generate_with_clients(assets_dir, prompt, style)


def _generate_with_clients(
    assets_dir: Path,
    prompt: str,
    style: str,
) -> None:
    """Generate images using the API clients directly."""
    from core.tools._image_clients import _CHIBI_PROMPT

    assets_dir.mkdir(parents=True, exist_ok=True)

    if style == "anime":
        fullbody_name = "avatar_fullbody.png"
        bustup_name = "avatar_bustup.png"
    else:
        fullbody_name = "avatar_fullbody_realistic.png"
        bustup_name = "avatar_bustup_realistic.png"
    chibi_name = "avatar_chibi.png"

    # ── Step 1: Full-body ──
    fullbody_path = assets_dir / fullbody_name
    fullbody_bytes: bytes | None = None

    if fullbody_path.exists() and fullbody_path.stat().st_size > 100:
        print(f"    fullbody: exists, skipping")
        fullbody_bytes = fullbody_path.read_bytes()
    else:
        if style == "realistic" or not os.environ.get("NOVELAI_TOKEN"):
            if not os.environ.get("FAL_KEY"):
                print(f"    fullbody: SKIP (no FAL_KEY)")
                return
            print(f"    fullbody: generating with Fal Flux Pro...")
            from core.tools._image_clients import FalTextToImageClient

            client = FalTextToImageClient()
            fullbody_bytes = client.generate_fullbody(prompt=prompt)
        else:
            print(f"    fullbody: generating with NovelAI...")
            from core.tools._image_clients import NovelAIClient

            client = NovelAIClient()
            fullbody_bytes = client.generate_fullbody(prompt=prompt)

        fullbody_path.write_bytes(fullbody_bytes)
        size_kb = len(fullbody_bytes) / 1024
        print(f"    fullbody: saved ({size_kb:.0f} KB)")

    # ── Step 2: Bust-up ──
    bustup_path = assets_dir / bustup_name
    if bustup_path.exists() and bustup_path.stat().st_size > 100:
        print(f"    bustup: exists, skipping")
    else:
        if not os.environ.get("FAL_KEY"):
            print(f"    bustup: SKIP (no FAL_KEY)")
        else:
            from core.tools._image_clients import (
                FluxKontextClient,
                _BUSTUP_PROMPT,
                _REALISTIC_BUSTUP_PROMPT,
            )

            print(f"    bustup: generating with Flux Kontext...")
            kontext = FluxKontextClient()
            bustup_prompt = _REALISTIC_BUSTUP_PROMPT if style == "realistic" else _BUSTUP_PROMPT
            bustup_bytes = kontext.generate_from_reference(
                reference_image=fullbody_bytes,
                prompt=bustup_prompt,
                aspect_ratio="3:4",
            )
            bustup_path.write_bytes(bustup_bytes)
            size_kb = len(bustup_bytes) / 1024
            print(f"    bustup: saved ({size_kb:.0f} KB)")

    # ── Step 3: Chibi ──
    chibi_path = assets_dir / chibi_name
    if chibi_path.exists() and chibi_path.stat().st_size > 100:
        print(f"    chibi: exists, skipping")
    else:
        if not os.environ.get("FAL_KEY"):
            print(f"    chibi: SKIP (no FAL_KEY)")
        else:
            from core.tools._image_clients import FluxKontextClient

            print(f"    chibi: generating with Flux Kontext...")
            kontext = FluxKontextClient()
            chibi_bytes = kontext.generate_from_reference(
                reference_image=fullbody_bytes,
                prompt=_CHIBI_PROMPT,
                aspect_ratio="1:1",
            )
            chibi_path.write_bytes(chibi_bytes)
            size_kb = len(chibi_bytes) / 1024
            print(f"    chibi: saved ({size_kb:.0f} KB)")


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo character assets using AnimaWorks image pipeline.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Generate assets for a specific preset only.",
    )
    parser.add_argument(
        "--character",
        help="Generate assets for a specific character only.",
    )
    args = parser.parse_args()

    api_keys = []
    if os.environ.get("NOVELAI_TOKEN"):
        api_keys.append("NOVELAI_TOKEN")
    if os.environ.get("FAL_KEY"):
        api_keys.append("FAL_KEY")

    if not api_keys:
        print("ERROR: No API keys found.")
        print("  Set NOVELAI_TOKEN for anime full-body generation")
        print("  Set FAL_KEY for Flux-based generation (bustup, chibi, realistic)")
        sys.exit(1)

    print(f"API keys available: {', '.join(api_keys)}")
    print()

    presets_to_run = {args.preset: PRESETS[args.preset]} if args.preset else PRESETS

    for preset_name, preset_info in presets_to_run.items():
        style = str(preset_info["style"])
        characters = list(preset_info["characters"])  # type: ignore[arg-type]

        if args.character:
            if args.character not in characters:
                continue
            characters = [args.character]

        print(f"=== {preset_name} (style={style}) ===")
        for char_name in characters:
            print(f"  [{char_name}]")
            try:
                generate_character(preset_name, char_name, style)
            except Exception as exc:
                print(f"  ERROR: {exc}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
