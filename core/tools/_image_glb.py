# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""GLB/FBX asset conversion, optimisation, and compression."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

from core.tools._base import logger
from core.tools._image_clients import _DOWNLOAD_TIMEOUT

__all__ = [
    "_run_gltf_transform",
    "_ensure_gltf_transform_modules",
    "_ensure_fbx2gltf",
    "_find_fbx2gltf_binary",
    "_convert_fbx_to_glb",
    "_download_armature_animation",
    "strip_mesh_from_glb",
    "optimize_glb",
    "simplify_glb",
    "compress_textures",
]


# ── gltf-transform helpers ────────────────────────────────────


def _run_gltf_transform(args: list[str], glb_path: Path) -> bool:
    """Run gltf-transform CLI command. Returns True on success."""
    import shutil
    import subprocess

    npx = shutil.which("npx")
    if npx is None:
        logger.warning("npx not found; skipping gltf-transform for %s", glb_path)
        return False

    cmd = [npx, "--yes", "@gltf-transform/cli", *args]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except FileNotFoundError:
        logger.warning("gltf-transform not available; skipping for %s", glb_path)
        return False
    except subprocess.CalledProcessError as exc:
        logger.warning("gltf-transform failed for %s: %s", glb_path, exc.stderr[:500].decode("utf-8", errors="replace") if exc.stderr else exc)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("gltf-transform timed out for %s", glb_path)
        return False


_GLTF_MODULES_DIR: Path | None = None


def _ensure_gltf_transform_modules() -> Path:
    """Install @gltf-transform/core and /functions to a persistent cache.

    Returns the ``node_modules`` directory path to be used as ``NODE_PATH``.
    Packages are installed once and reused across calls.
    """
    global _GLTF_MODULES_DIR  # noqa: PLW0603
    if _GLTF_MODULES_DIR is not None and _GLTF_MODULES_DIR.exists():
        return _GLTF_MODULES_DIR

    import subprocess

    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "cache" / "gltf_transform_modules"
    node_modules = cache_dir / "node_modules"

    if not (node_modules / "@gltf-transform" / "core").is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["npm", "install", "--save",
             "@gltf-transform/core", "@gltf-transform/functions"],
            cwd=str(cache_dir),
            check=True, capture_output=True, timeout=120,
        )
        logger.info("Installed @gltf-transform modules to %s", cache_dir)

    _GLTF_MODULES_DIR = node_modules
    return node_modules


# ── fbx2gltf helpers ──────────────────────────────────────


_FBX2GLTF_PATH: Path | None = None


def _ensure_fbx2gltf() -> Path | None:
    """Install fbx2gltf to a persistent cache and return the binary path.

    Resolution order:
      1. Module-level cached path (fast path).
      2. System ``PATH`` via ``shutil.which("FBX2glTF")``.
      3. Persistent npm cache at ``~/.animaworks/cache/fbx2gltf/``.
      4. Fresh ``npm install --save fbx2gltf`` into the cache directory.

    Returns:
        Binary path on success, ``None`` if installation fails.
    """
    global _FBX2GLTF_PATH  # noqa: PLW0603
    if _FBX2GLTF_PATH is not None and _FBX2GLTF_PATH.exists():
        return _FBX2GLTF_PATH

    import shutil

    # Check if already on PATH
    system_path = shutil.which("FBX2glTF")
    if system_path:
        _FBX2GLTF_PATH = Path(system_path)
        return _FBX2GLTF_PATH

    import subprocess

    from core.paths import get_data_dir

    cache_dir = get_data_dir() / "cache" / "fbx2gltf"
    node_modules = cache_dir / "node_modules"

    # Check both .bin symlink and platform-specific binary location
    for found in _find_fbx2gltf_binary(node_modules):
        _FBX2GLTF_PATH = found
        return _FBX2GLTF_PATH

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["npm", "install", "--save", "fbx2gltf"],
            cwd=str(cache_dir),
            check=True,
            capture_output=True,
            timeout=120,
        )
        for found in _find_fbx2gltf_binary(node_modules):
            _FBX2GLTF_PATH = found
            logger.info("Installed fbx2gltf to %s", cache_dir)
            return _FBX2GLTF_PATH
    except Exception as exc:
        logger.warning("Failed to install fbx2gltf: %s", exc)
    return None


def _find_fbx2gltf_binary(node_modules: Path):
    """Yield fbx2gltf binary paths if they exist.

    The npm ``fbx2gltf`` package places the binary at a platform-specific
    path (e.g. ``node_modules/fbx2gltf/bin/Linux/FBX2glTF``) rather than
    the standard ``node_modules/.bin/`` symlink.
    """
    import platform

    # Standard .bin symlink
    candidate = node_modules / ".bin" / "fbx2gltf"
    if candidate.exists():
        yield candidate

    # Platform-specific binary (Linux, Darwin, Windows)
    os_name = platform.system()  # "Linux", "Darwin", "Windows"
    candidate = node_modules / "fbx2gltf" / "bin" / os_name / "FBX2glTF"
    if candidate.exists():
        yield candidate


def _convert_fbx_to_glb(fbx_path: Path, glb_path: Path) -> bool:
    """Convert an FBX file to GLB using fbx2gltf.

    Args:
        fbx_path: Source FBX file.
        glb_path: Destination GLB file.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    import subprocess

    from core.tools.image_gen import _ensure_fbx2gltf

    fbx2gltf = _ensure_fbx2gltf()
    if fbx2gltf is None:
        return False
    try:
        subprocess.run(
            [str(fbx2gltf), "--binary", "--input", str(fbx_path),
             "--output", str(glb_path)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return glb_path.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("fbx2gltf conversion failed for %s: %s", fbx_path, exc)
        return False


def _download_armature_animation(
    task: dict[str, Any],
    glb_path: Path,
) -> bool:
    """Download armature-only animation, converting from FBX if available.

    Tries the optimised path first: ``processed_armature_fbx_url`` (small,
    ~10-50 KB) downloaded as FBX then converted to GLB via *fbx2gltf*.

    Falls back to the legacy path on any failure: full-mesh GLB download
    followed by :func:`strip_mesh_from_glb`.

    Args:
        task: Completed animation task dict from Meshy API.
        glb_path: Destination path for the final ``.glb`` file.

    Returns:
        ``True`` when a usable GLB was written, ``False`` on total failure.
    """
    import tempfile

    result = task.get("result", {})
    armature_url = result.get("processed_armature_fbx_url")

    from core.tools.image_gen import _convert_fbx_to_glb as _cvt_fbx
    from core.tools.image_gen import strip_mesh_from_glb as _strip_mesh

    if armature_url:
        fbx_path: Path | None = None
        try:
            resp = httpx.get(armature_url, timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(
                suffix=".fbx", delete=False,
            ) as f:
                f.write(resp.content)
                fbx_path = Path(f.name)
            if _cvt_fbx(fbx_path, glb_path):
                logger.info(
                    "Armature animation converted: %s (%d bytes)",
                    glb_path,
                    glb_path.stat().st_size,
                )
                return True
        except Exception as exc:
            logger.warning(
                "Armature download/conversion failed for %s, falling back: %s",
                glb_path.name,
                exc,
            )
        finally:
            if fbx_path is not None:
                fbx_path.unlink(missing_ok=True)

    # ── Fallback: full GLB + strip ──
    logger.warning(
        "Falling back to full GLB download + strip for %s", glb_path.name,
    )
    glb_url = result.get("animation_glb_url")
    if not glb_url:
        logger.error("No animation URL available for %s", glb_path.name)
        return False
    resp = httpx.get(glb_url, timeout=_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    glb_path.write_bytes(resp.content)
    if not _strip_mesh(glb_path):
        logger.warning("Fallback strip also failed for %s", glb_path.name)
    return True


def strip_mesh_from_glb(glb_path: Path) -> bool:
    """Remove mesh/material/texture data from a GLB, keeping only skeleton + animation.

    Uses a Node.js script via gltf-transform programmatic API because
    the CLI does not have a dedicated 'strip meshes' command.

    Packages are installed to a persistent cache directory under
    ``~/.animaworks/cache/gltf_transform_modules/`` and referenced via
    ``NODE_PATH`` (npm 10.x compatible).

    Returns True on success, False if skipped or failed.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        logger.warning("node not found; skipping mesh strip for %s", glb_path)
        return False

    script = """\
const { NodeIO } = require("@gltf-transform/core");
const { prune } = require("@gltf-transform/functions");

(async () => {
    const io = new NodeIO();
    const doc = await io.read(process.argv[2]);
    for (const node of doc.getRoot().listNodes()) {
        node.setMesh(null);
    }
    for (const mat of doc.getRoot().listMaterials()) {
        mat.dispose();
    }
    for (const tex of doc.getRoot().listTextures()) {
        tex.dispose();
    }
    await doc.transform(prune());
    await io.write(process.argv[2], doc);
})();
"""

    script_path = None
    try:
        from core.tools.image_gen import _ensure_gltf_transform_modules

        node_modules = _ensure_gltf_transform_modules()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cjs", delete=False,
        ) as fd:
            fd.write(script)
            script_path = fd.name

        env = {**os.environ, "NODE_PATH": str(node_modules)}
        subprocess.run(
            [node, script_path, str(glb_path)],
            check=True, capture_output=True, timeout=120, env=env,
        )
        logger.info("Stripped mesh from %s (now %d bytes)", glb_path, glb_path.stat().st_size)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Mesh strip failed for %s: %s", glb_path, exc)
        return False
    finally:
        if script_path is not None:
            Path(script_path).unlink(missing_ok=True)


def optimize_glb(glb_path: Path) -> bool:
    """Apply Draco compression to a GLB file using gltf-transform.

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".opt.glb")
    try:
        if _run_gltf_transform(["optimize", str(glb_path), str(tmp_path)], glb_path):
            if _run_gltf_transform(["draco", str(tmp_path), str(glb_path)], glb_path):
                tmp_path.unlink(missing_ok=True)
                logger.info("Optimized %s (now %d bytes)", glb_path, glb_path.stat().st_size)
                return True
            else:
                # optimize succeeded but draco failed; keep optimized version
                tmp_path.rename(glb_path)
                return True
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def simplify_glb(glb_path: Path, target_ratio: float = 0.27, error_threshold: float = 0.01) -> bool:
    """Simplify mesh geometry using gltf-transform simplify (meshoptimizer).

    ``target_ratio=0.27`` reduces ~30K-polygon Meshy models to ~8K polygons.
    ``error_threshold`` controls maximum allowed deviation (lower = higher quality).

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".simp.glb")
    try:
        if _run_gltf_transform(
            ["simplify", str(glb_path), str(tmp_path),
             "--ratio", str(target_ratio),
             "--error", str(error_threshold)],
            glb_path,
        ):
            tmp_path.rename(glb_path)
            logger.info("Simplified %s (now %d bytes)", glb_path, glb_path.stat().st_size)
            return True
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def compress_textures(glb_path: Path, resolution: int = 1024) -> bool:
    """Resize textures and convert to WebP format using gltf-transform.

    Applies two-step optimization:
    1. Resize all textures to ``resolution x resolution``
    2. Convert textures to WebP

    Returns True on success, False if skipped or failed.
    """
    tmp_path = glb_path.with_suffix(".tex.glb")
    try:
        # Step 1: resize textures
        if not _run_gltf_transform(
            ["resize", str(glb_path), str(tmp_path),
             "--width", str(resolution), "--height", str(resolution)],
            glb_path,
        ):
            return False
        # Step 2: convert to WebP
        if _run_gltf_transform(
            ["webp", str(tmp_path), str(glb_path)],
            glb_path,
        ):
            tmp_path.unlink(missing_ok=True)
            logger.info(
                "Compressed textures in %s to webp@%d (now %d bytes)",
                glb_path, resolution, glb_path.stat().st_size,
            )
            return True
        # resize succeeded but webp failed — keep resized version
        tmp_path.rename(glb_path)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)
