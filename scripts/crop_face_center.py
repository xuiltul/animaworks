#!/usr/bin/env python3
"""Crop image to square with face centered. Uses heuristic for bust-up avatars."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def crop_face_center_square(
    src: Path,
    dest: Path | None = None,
    *,
    face_y_ratio: float = 0.38,
) -> Path:
    """Crop image to square, centering on estimated face position.

    For bust-up avatar images, face is typically at ~38% from top.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    side = min(w, h)

    # Estimate face center (bust-up: face in upper third)
    face_x = w / 2
    face_y = h * face_y_ratio

    # Center crop on face
    left = int(face_x - side / 2)
    top = int(face_y - side / 2)

    # Clamp to image bounds
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))

    cropped = img.crop((left, top, left + side, top + side))
    out = dest or src.parent / f"{src.stem}_cropped{src.suffix}"
    cropped.save(out, quality=95)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <src_image> [dest_image]",
            file=sys.stderr,
        )
        sys.exit(2)
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)
    out = crop_face_center_square(src, dest)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
