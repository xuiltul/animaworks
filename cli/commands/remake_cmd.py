# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from core.time_utils import now_jst

logger = logging.getLogger(__name__)

ALL_STEPS = ["fullbody", "bustup", "chibi", "3d", "rigging", "animations"]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the remake-assets subcommand."""
    parser = subparsers.add_parser(
        "remake-assets",
        help="Remake character assets with style transfer from another anima",
        description=(
            "Regenerate character assets using Vibe Transfer to match "
            "the art style of a reference anima. Supports selective step "
            "execution and automatic backup."
        ),
    )
    parser.add_argument(
        "anima",
        help="Name of the anima whose assets to remake",
    )
    parser.add_argument(
        "--style-from",
        required=True,
        help="Anima name to use as style reference (their fullbody image)",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated list of steps to run "
            f"(choices: {', '.join(ALL_STEPS)}). Default: all steps"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override character prompt (default: read from prompt.txt)",
    )
    parser.add_argument(
        "--vibe-strength",
        type=float,
        default=0.6,
        help="Vibe Transfer strength 0.0-1.0 (default: 0.6)",
    )
    parser.add_argument(
        "--vibe-info-extracted",
        type=float,
        default=0.8,
        help="Vibe Transfer information extraction 0.0-1.0 (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (fullbody generation only)",
    )
    parser.add_argument(
        "--image-style",
        choices=["anime", "realistic"],
        default=None,
        help="Image style (default: from config.json image_gen.image_style)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip automatic backup of existing assets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Execute the remake-assets command."""
    from core.config.models import load_config
    from core.paths import get_data_dir

    config = load_config()
    data_dir = get_data_dir()
    animas_dir = data_dir / "animas"

    # ── Validate animas exist ──
    target_dir = animas_dir / args.anima
    style_dir = animas_dir / args.style_from

    if not target_dir.exists():
        print(f"Error: Anima '{args.anima}' not found at {target_dir}")
        return

    if not style_dir.exists():
        print(f"Error: Style reference anima '{args.style_from}' not found at {style_dir}")
        return

    # ── Resolve image style ──
    image_style = args.image_style
    if image_style is None:
        image_style = config.image_gen.image_style or "anime"
    is_realistic = image_style == "realistic"

    # ── Validate style reference image ──
    style_ref_filename = (
        "avatar_fullbody_realistic.png" if is_realistic
        else "avatar_fullbody.png"
    )
    style_fullbody = style_dir / "assets" / style_ref_filename
    if not style_fullbody.exists():
        print(
            f"Error: Style reference anima '{args.style_from}' has no "
            f"{style_ref_filename} at {style_fullbody}"
        )
        return

    # ── Resolve prompt (style-aware) ──
    prompt = args.prompt
    if prompt is None:
        from core.asset_reconciler import _resolve_prompt
        prompt = _resolve_prompt(target_dir, style=image_style)
        if not prompt:
            print(
                f"Error: No prompt available for '{args.anima}' and --prompt not specified.\n"
                f"  Expected: assets/prompt.txt (anime) or assets/prompt_realistic.txt (realistic)\n"
                f"  Use --prompt to provide a character prompt."
            )
            return

    if not prompt:
        print("Error: Prompt is empty. Provide a non-empty prompt via --prompt or prompt.txt.")
        return

    # ── Parse steps ──
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
        invalid = [s for s in steps if s not in ALL_STEPS]
        if invalid:
            print(f"Error: Invalid steps: {', '.join(invalid)}")
            print(f"  Valid steps: {', '.join(ALL_STEPS)}")
            return
    else:
        steps = list(ALL_STEPS)

    # ── Validate vibe parameters ──
    if not 0.0 <= args.vibe_strength <= 1.0:
        print(f"Error: --vibe-strength must be between 0.0 and 1.0 (got {args.vibe_strength})")
        return

    if not 0.0 <= args.vibe_info_extracted <= 1.0:
        print(
            f"Error: --vibe-info-extracted must be between 0.0 and 1.0 "
            f"(got {args.vibe_info_extracted})"
        )
        return

    # ── Dry-run summary ──
    print(f"Remake assets for: {args.anima}")
    print(f"Image style:       {image_style}")
    print(f"Style reference:   {args.style_from} ({style_fullbody})")
    print(f"Vibe strength:     {args.vibe_strength}")
    print(f"Vibe info extract: {args.vibe_info_extracted}")
    print(f"Steps:             {', '.join(steps)}")
    print(f"Prompt:            {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if args.seed is not None:
        print(f"Seed:              {args.seed}")
    print()

    if args.dry_run:
        assets_dir = target_dir / "assets"
        print("[DRY-RUN] The following actions would be performed:")
        print()
        if not args.no_backup and assets_dir.exists():
            ts = now_jst().strftime("%Y%m%d_%H%M%S")
            print(f"  1. Backup: {assets_dir} -> {target_dir / f'assets_backup_{ts}'}")
        else:
            print("  1. Backup: skipped (--no-backup or no existing assets)")

        step_names = {
            "fullbody": "Generate fullbody with NovelAI + Vibe Transfer",
            "bustup": "Generate bustup expressions (neutral, smile, laugh, troubled, surprised)",
            "chibi": "Generate chibi with Flux Kontext",
            "3d": "Generate 3D model with Meshy Image-to-3D",
            "rigging": "Rig 3D model with Meshy Rigging",
            "animations": "Generate animations (idle, sitting, waving, talking)",
        }
        for i, step in enumerate(steps, 2):
            print(f"  {i}. {step}: {step_names.get(step, step)}")

        print()
        print("[DRY-RUN] No API calls were made.")
        return

    # ── Backup existing assets ──
    assets_dir = target_dir / "assets"
    if not args.no_backup and assets_dir.exists():
        ts = now_jst().strftime("%Y%m%d_%H%M%S")
        backup_dir = target_dir / f"assets_backup_{ts}"
        shutil.copytree(assets_dir, backup_dir)
        print(f"Backup created: {backup_dir}")
    else:
        assets_dir.mkdir(parents=True, exist_ok=True)

    # ── Load style reference image ──
    vibe_image = style_fullbody.read_bytes()
    print(f"Loaded style reference: {style_fullbody} ({len(vibe_image):,} bytes)")

    # ── Run pipeline ──
    from core.config.models import ImageGenConfig
    from core.tools.image_gen import ImageGenPipeline

    pipeline = ImageGenPipeline(
        target_dir,
        config=ImageGenConfig(image_style=image_style),
    )

    def _progress(step: str, status: str, pct: int) -> None:
        if status == "generating":
            print(f"  [{step}] Generating...")
        elif status == "completed":
            print(f"  [{step}] Done")
        elif status == "error":
            print(f"  [{step}] FAILED")

    print()
    print("Starting pipeline...")
    result = pipeline.generate_all(
        prompt=prompt,
        skip_existing=False,
        steps=steps,
        vibe_image=vibe_image,
        vibe_strength=args.vibe_strength,
        vibe_info_extracted=args.vibe_info_extracted,
        seed=args.seed,
        progress_callback=_progress,
    )

    # ── Summary ──
    print()
    print("=== Remake Complete ===")
    if result.fullbody_path:
        print(f"  fullbody: {result.fullbody_path}")
    if result.bustup_paths:
        print(f"  bustup:   {len(result.bustup_paths)} expressions")
        for expr, path in result.bustup_paths.items():
            print(f"            {expr}: {path}")
    if result.chibi_path:
        print(f"  chibi:    {result.chibi_path}")
    if result.model_path:
        print(f"  3d:       {result.model_path}")
    if result.rigged_model_path:
        print(f"  rigged:   {result.rigged_model_path}")
    if result.animation_paths:
        print(f"  animations: {len(result.animation_paths)}")
        for name, path in result.animation_paths.items():
            print(f"              {name}: {path}")

    if result.skipped:
        print(f"\n  Skipped: {', '.join(result.skipped)}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    - {err}")
