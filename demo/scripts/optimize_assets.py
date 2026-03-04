#!/usr/bin/env python3
"""Optimize demo asset images: resize + PNG compression.

Usage:
    python demo/scripts/optimize_assets.py [--preset PRESET] [--dry-run]

Requires Pillow: pip install Pillow
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
PRESETS_DIR = DEMO_DIR / "presets"

# Maximum dimensions per asset type (width, height)
SIZE_LIMITS: dict[str, tuple[int, int]] = {
    "fullbody": (512, 1024),
    "bustup": (512, 512),
    "chibi": (256, 256),
}

# Minimum file size to consider for optimization (skip placeholders)
MIN_SIZE_BYTES = 200


def _classify_asset(filename: str) -> str | None:
    """Classify an asset filename into fullbody/bustup/chibi."""
    name = filename.lower()
    if "chibi" in name:
        return "chibi"
    if "bustup" in name:
        return "bustup"
    if "fullbody" in name:
        return "fullbody"
    return None


def optimize_image(path: Path, max_size: tuple[int, int], dry_run: bool = False) -> int:
    """Resize and compress a single PNG image.

    Returns bytes saved (negative if image grew, which shouldn't happen).
    """
    from PIL import Image

    original_size = path.stat().st_size

    if original_size < MIN_SIZE_BYTES:
        return 0

    img = Image.open(path)
    max_w, max_h = max_size

    if img.width > max_w or img.height > max_h:
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    if dry_run:
        print(f"    [dry-run] would resize to {img.width}x{img.height}: {path.name}")
        return 0

    img.save(path, format="PNG", optimize=True)
    new_size = path.stat().st_size

    return original_size - new_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize demo asset images (resize + PNG compression).",
    )
    parser.add_argument(
        "--preset",
        help="Optimize a specific preset only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files.",
    )
    args = parser.parse_args()

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    total_saved = 0
    total_files = 0

    if args.preset:
        preset_dirs = [PRESETS_DIR / args.preset]
    else:
        preset_dirs = sorted(p for p in PRESETS_DIR.iterdir() if p.is_dir())

    for preset_dir in preset_dirs:
        assets_root = preset_dir / "assets"
        if not assets_root.exists():
            continue

        preset_name = preset_dir.name
        print(f"=== {preset_name} ===")

        for char_dir in sorted(assets_root.iterdir()):
            if not char_dir.is_dir():
                continue

            for png_file in sorted(char_dir.glob("*.png")):
                asset_type = _classify_asset(png_file.name)
                if asset_type is None:
                    continue

                max_size = SIZE_LIMITS[asset_type]
                original_kb = png_file.stat().st_size / 1024

                if png_file.stat().st_size < MIN_SIZE_BYTES:
                    print(f"  {char_dir.name}/{png_file.name}: placeholder, skipping")
                    continue

                saved = optimize_image(png_file, max_size, dry_run=args.dry_run)
                new_kb = png_file.stat().st_size / 1024
                total_saved += saved
                total_files += 1

                if args.dry_run:
                    print(f"  {char_dir.name}/{png_file.name}: {original_kb:.1f} KB (max {max_size[0]}x{max_size[1]})")
                else:
                    saved_kb = saved / 1024
                    print(f"  {char_dir.name}/{png_file.name}: {original_kb:.1f} KB -> {new_kb:.1f} KB (saved {saved_kb:.1f} KB)")

        print()

    total_saved_kb = total_saved / 1024
    print(f"Total: {total_files} files processed, {total_saved_kb:.1f} KB saved")


if __name__ == "__main__":
    main()
