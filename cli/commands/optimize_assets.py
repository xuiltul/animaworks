# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the optimize-assets subcommand."""
    parser = subparsers.add_parser(
        "optimize-assets",
        help="Optimize existing 3D assets (strip meshes, compress, simplify)",
    )
    parser.add_argument(
        "--anima", "-a",
        help="Optimize assets for a specific anima only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        nargs="?",
        const=0.27,
        default=None,
        metavar="RATIO",
        help="Simplify meshes (default ratio: 0.27 ≈ 30K→8K polygons)",
    )
    parser.add_argument(
        "--texture-compress",
        action="store_true",
        help="Convert textures to WebP format",
    )
    parser.add_argument(
        "--texture-resize",
        type=int,
        default=None,
        metavar="RES",
        help="Resize textures to RES×RES (default: 1024 when --texture-compress is set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="apply_all",
        help="Apply all optimizations: strip + simplify + texture + draco",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating backup of original assets",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    from core.paths import get_animas_dir
    from core.tools.image_gen import (
        compress_textures,
        optimize_glb,
        simplify_glb,
        strip_mesh_from_glb,
    )

    animas_dir = get_animas_dir()
    if not animas_dir.exists():
        print(f"Animas directory not found: {animas_dir}")
        return

    # Resolve --all flag
    do_strip = True  # Always strip animation meshes
    do_draco = True  # Always apply Draco compression
    do_simplify = args.apply_all or args.simplify is not None
    do_texture = args.apply_all or args.texture_compress

    simplify_ratio = args.simplify if args.simplify is not None else 0.27
    texture_resolution = args.texture_resize or 1024

    if args.anima:
        anima_dir = animas_dir / args.anima
        if not anima_dir.is_dir():
            print(f"Anima not found: {args.anima}")
            return
        anima_dirs = [anima_dir]
    else:
        anima_dirs = sorted(
            d for d in animas_dir.iterdir()
            if d.is_dir() and (d / "assets").is_dir()
        )

    total_before = 0
    total_after = 0
    errors: list[str] = []

    for anima_dir in anima_dirs:
        assets_dir = anima_dir / "assets"
        if not assets_dir.exists():
            continue

        name = anima_dir.name
        print(f"\n=== {name} ===")

        # ── Backup ──
        if not args.dry_run and not args.skip_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = anima_dir / f"assets_backup_{timestamp}"
            shutil.copytree(assets_dir, backup_dir)
            print(f"  Backup created: {backup_dir.name}")

        # ── 1. Strip meshes from animation GLBs ──
        for anim_file in sorted(assets_dir.glob("anim_*.glb")):
            size_before = anim_file.stat().st_size
            total_before += size_before
            if args.dry_run:
                print(f"  [DRY-RUN] Would strip mesh from {anim_file.name} ({size_before:,} bytes)")
                total_after += size_before
            elif do_strip:
                if strip_mesh_from_glb(anim_file):
                    size_after = anim_file.stat().st_size
                    total_after += size_after
                    print(f"  Stripped {anim_file.name}: {size_before:,} → {size_after:,} bytes")
                else:
                    total_after += size_before
                    errors.append(f"{name}/{anim_file.name}: strip failed")
                    print(f"  FAILED strip {anim_file.name}")

        # ── 2-4. Process model GLBs ──
        for model_file in sorted(assets_dir.glob("avatar_chibi*.glb")):
            size_before = model_file.stat().st_size
            total_before += size_before

            if args.dry_run:
                steps = []
                if do_simplify:
                    steps.append(f"simplify(ratio={simplify_ratio})")
                if do_texture:
                    steps.append(f"texture({texture_resolution}px,webp)")
                if do_draco:
                    steps.append("draco")
                desc = " + ".join(steps) if steps else "draco"
                print(f"  [DRY-RUN] Would optimize {model_file.name}: {desc} ({size_before:,} bytes)")
                total_after += size_before
                continue

            # 2. Simplify mesh
            if do_simplify:
                if simplify_glb(model_file, target_ratio=simplify_ratio):
                    print(f"  Simplified {model_file.name}: {size_before:,} → {model_file.stat().st_size:,} bytes")
                else:
                    errors.append(f"{name}/{model_file.name}: simplify failed")
                    print(f"  FAILED simplify {model_file.name}")

            # 3. Texture compress
            if do_texture:
                size_pre_tex = model_file.stat().st_size
                if compress_textures(model_file, resolution=texture_resolution):
                    print(f"  Textures {model_file.name}: {size_pre_tex:,} → {model_file.stat().st_size:,} bytes")
                else:
                    errors.append(f"{name}/{model_file.name}: texture compress failed")
                    print(f"  FAILED texture compress {model_file.name}")

            # 4. Draco compression (always last — lossy)
            if do_draco:
                size_pre_draco = model_file.stat().st_size
                if optimize_glb(model_file):
                    print(f"  Draco {model_file.name}: {size_pre_draco:,} → {model_file.stat().st_size:,} bytes")
                else:
                    errors.append(f"{name}/{model_file.name}: draco failed")
                    print(f"  FAILED draco {model_file.name}")

            total_after += model_file.stat().st_size

    # ── Summary ──
    print(f"\nTotal: {total_before:,} → {total_after:,} bytes")
    if total_before > 0:
        reduction = (1 - total_after / total_before) * 100
        print(f"Reduction: {reduction:.1f}%")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
