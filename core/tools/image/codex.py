# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Codex CLI image generation client (image_gen tool via local codex)."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from core.tools._base import logger

_CODEX_TIMEOUT = 900  # seconds; generous margin for one image

_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "3:4": (1024, 1365),
}


def codex_available() -> bool:
    """Return True if the codex CLI is installed on PATH."""
    return shutil.which("codex") is not None


def _codex_stderr_reason(stderr: bytes | str) -> str:
    """Return a concise, useful reason from codex stderr.

    Codex often reports actionable failures as ``ERROR: ...`` lines.  Keep the
    last distinct such line rather than returning the whole CLI transcript (or
    the image spec that may have appeared nearby); otherwise the fallback error
    can hide the original cause behind an unrelated configuration error.
    """
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
    error_lines: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        marker = line.find("ERROR:")
        if marker < 0:
            continue
        candidate = line[marker:].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            error_lines.append(candidate)
    if error_lines:
        return error_lines[-1]
    return text.strip()[-300:]


def _codex_failure_reason(exc: Exception) -> str:
    """Extract the concise codex reason from a client exception."""
    return _codex_stderr_reason(str(exc)) or type(exc).__name__


def _cover_crop_resize(image_bytes: bytes, target_size: tuple[int, int]) -> bytes:
    """Resize with cover-crop so output matches *target_size* without stretch."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    tw, th = target_size
    if img.size == (tw, th):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    scale = max(tw / img.width, th / img.height)
    nw = max(1, round(img.width * scale))
    nh = max(1, round(img.height * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    img = img.crop((left, top, left + tw, top + th))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_spec(
    *,
    prompt: str,
    image_style: str = "anime",
    negative_prompt: str = "",
    orientation: str = "portrait",
    reference_roles: list[str] | None = None,
) -> str:
    """Build a markdown spec prompt for codex image_gen."""
    if image_style == "realistic":
        style_line = "Photorealistic, live-action look"
    else:
        style_line = "Anime-style illustration"
    style_line = f"{style_line}. Composition: {orientation}."

    lines = [
        "# Image generation spec",
        "",
        "## Tool",
        "Use the built-in `image_gen` tool. Substituting SVG/HTML/CSS or hand-drawn scripts is not acceptable.",
        "",
        "## Subject",
        prompt,
        "",
        "## Style",
        style_line,
        "",
    ]
    if negative_prompt:
        lines.extend(
            [
                "## Avoid",
                negative_prompt,
                "",
            ]
        )
    if reference_roles:
        lines.append("## Reference image roles")
        for role in reference_roles:
            lines.append(f"- {role}")
        lines.append("")
    lines.extend(
        [
            "## Output contract",
            "Save the generated image as a PNG named `out.png` in the current directory, then finish. Do not create any other files.",
            "",
        ]
    )
    return "\n".join(lines)


class CodexImageClient:
    """Image client that drives the local ``codex`` CLI ``image_gen`` tool."""

    def __init__(self, image_config: Any = None) -> None:
        self._image_config = image_config
        self._image_style = getattr(image_config, "image_style", "anime") if image_config is not None else "anime"

    def generate_fullbody(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1536,
        seed: int | None = None,
        vibe_image: bytes | None = None,
        vibe_strength: float | None = None,
        vibe_info_extracted: float | None = None,
        face_reference_image: bytes | None = None,
        **_ignored: Any,
    ) -> bytes:
        """Generate a full-body character image via codex."""
        del seed, vibe_strength, vibe_info_extracted  # not used by codex path
        refs: list[bytes] = []
        roles: list[str] = []
        if vibe_image:
            refs.append(vibe_image)
            roles.append(f"Match the art style of reference image {len(refs)}")
        if face_reference_image:
            refs.append(face_reference_image)
            roles.append(f"Reflect the facial features of the person in reference image {len(refs)}")

        orientation = "portrait" if height >= width else "square"
        if width == height:
            orientation = "square"
        spec = _build_spec(
            prompt=prompt,
            image_style=self._image_style,
            negative_prompt=negative_prompt,
            orientation=orientation,
            reference_roles=roles or None,
        )
        return self._run(spec, refs, (width, height))

    def generate_from_reference(
        self,
        reference_image: bytes,
        prompt: str,
        aspect_ratio: str = "1:1",
        **_ignored: Any,
    ) -> bytes:
        """Generate an image from a reference image via codex."""
        target = _ASPECT_SIZES.get(aspect_ratio, (1024, 1024))
        tw, th = target
        orientation = "square" if tw == th else "portrait"
        roles = ["Preserve the identity of the character in reference image 1 (hairstyle, hair color, eyes, outfit)"]
        spec = _build_spec(
            prompt=prompt,
            image_style=self._image_style,
            orientation=orientation,
            reference_roles=roles,
        )
        return self._run(spec, [reference_image], target)

    def _run(
        self,
        spec: str,
        reference_images: list[bytes],
        target_size: tuple[int, int],
    ) -> bytes:
        """Run codex in a temp dir, then cover-crop to *target_size*."""
        with tempfile.TemporaryDirectory(prefix="aw-codex-img-") as tmp:
            tmp_path = Path(tmp)
            cmd: list[str] = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                "-C",
                str(tmp_path),
            ]
            for i, ref in enumerate(reference_images):
                ref_path = tmp_path / f"ref_{i}.png"
                ref_path.write_bytes(ref)
                # -i resolves relative paths against the invoking cwd (not -C),
                # so pass absolute paths.
                cmd.extend(["-i", str(ref_path)])
            # -i is variadic and would swallow the prompt; "--" ends option parsing.
            cmd.extend(["--", spec])

            try:
                completed = subprocess.run(
                    cmd,
                    timeout=_CODEX_TIMEOUT,
                    capture_output=True,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"codex image generation timed out after {_CODEX_TIMEOUT}s") from exc

            if completed.returncode != 0:
                reason = _codex_stderr_reason(completed.stderr or b"")
                raise RuntimeError(f"codex image generation failed (rc={completed.returncode}): {reason}")

            out_path = tmp_path / "out.png"
            if not out_path.exists() or out_path.stat().st_size == 0:
                reason = _codex_stderr_reason(completed.stderr or b"")
                raise RuntimeError(f"codex did not produce out.png: {reason}")

            return _cover_crop_resize(out_path.read_bytes(), target_size)


class CodexFirstClient:
    """Try codex first; on failure defer to a lazily-built fallback client."""

    def __init__(
        self,
        fallback_factory: Callable[[], Any],
        image_config: Any = None,
    ) -> None:
        self._fallback_factory = fallback_factory
        self._image_config = image_config
        self._fallback: Any | None = None

    def _get_fallback(self) -> Any:
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def generate_fullbody(self, *args: Any, **kwargs: Any) -> bytes:
        try:
            return CodexImageClient(self._image_config).generate_fullbody(*args, **kwargs)
        except Exception as codex_exc:
            codex_reason = _codex_failure_reason(codex_exc)
            logger.warning("codex image generation failed, falling back to API: %s", codex_exc)
            try:
                return self._get_fallback().generate_fullbody(*args, **kwargs)
            except Exception as fallback_exc:
                raise RuntimeError(f"{fallback_exc} (codex: {codex_reason})") from fallback_exc

    def generate_from_reference(self, *args: Any, **kwargs: Any) -> bytes:
        try:
            return CodexImageClient(self._image_config).generate_from_reference(*args, **kwargs)
        except Exception as codex_exc:
            codex_reason = _codex_failure_reason(codex_exc)
            logger.warning("codex image generation failed, falling back to API: %s", codex_exc)
            try:
                return self._get_fallback().generate_from_reference(*args, **kwargs)
            except Exception as fallback_exc:
                raise RuntimeError(f"{fallback_exc} (codex: {codex_reason})") from fallback_exc
