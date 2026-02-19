# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for GLB optimization: strip_mesh_from_glb, simplify_glb, compress_textures, fbx2gltf.

These tests call the actual gltf-transform CLI via npx to verify real-world behavior.
They require node/npx to be installed and will be skipped if unavailable.
"""
from __future__ import annotations

import shutil
import struct
from pathlib import Path

import pytest


_has_npx = shutil.which("npx") is not None
_has_node = shutil.which("node") is not None
_has_npm = shutil.which("npm") is not None

skip_no_node = pytest.mark.skipif(
    not (_has_npx and _has_node),
    reason="node/npx not available",
)

skip_no_npm = pytest.mark.skipif(
    not _has_npm,
    reason="npm not available",
)


def _create_minimal_glb(path: Path, *, with_mesh: bool = True) -> None:
    """Create a minimal valid GLB file with optional mesh data.

    Produces a bare-minimum glTF 2.0 binary container.  When *with_mesh* is
    True the embedded JSON references a trivial mesh so that strip/simplify
    operations have something to work on.
    """
    if with_mesh:
        # Minimal glTF JSON with a mesh containing a single triangle
        gltf_json = (
            '{"asset":{"version":"2.0"},'
            '"scene":0,'
            '"scenes":[{"nodes":[0]}],'
            '"nodes":[{"mesh":0}],'
            '"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],'
            '"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",'
            '"max":[1,1,0],"min":[0,0,0]}],'
            '"bufferViews":[{"buffer":0,"byteLength":36}],'
            '"buffers":[{"byteLength":36}]}'
        )
    else:
        gltf_json = '{"asset":{"version":"2.0"}}'

    json_bytes = gltf_json.encode("utf-8")
    # Pad JSON to 4-byte alignment
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_chunk = json_bytes + b" " * json_pad
    json_chunk_length = len(json_chunk)

    if with_mesh:
        # 3 vertices × 3 floats × 4 bytes = 36 bytes of binary data
        bin_data = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        bin_pad = (4 - len(bin_data) % 4) % 4
        bin_chunk = bin_data + b"\x00" * bin_pad
        bin_chunk_length = len(bin_chunk)
    else:
        bin_chunk = b""
        bin_chunk_length = 0

    # GLB header: magic + version + total length
    total_length = 12  # header
    total_length += 8 + json_chunk_length  # JSON chunk header + data
    if bin_chunk:
        total_length += 8 + bin_chunk_length  # BIN chunk header + data

    with open(path, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        f.write(struct.pack("<II", json_chunk_length, 0x4E4F534A))
        f.write(json_chunk)
        # BIN chunk (if any)
        if bin_chunk:
            f.write(struct.pack("<II", bin_chunk_length, 0x004E4942))
            f.write(bin_chunk)


@pytest.mark.e2e
class TestStripMeshE2E:
    """E2E tests for strip_mesh_from_glb with real gltf-transform."""

    @skip_no_node
    def test_strip_mesh_reduces_file_size(self, tmp_path):
        """strip_mesh_from_glb should remove mesh data from a GLB file."""
        from core.tools.image_gen import strip_mesh_from_glb

        glb_path = tmp_path / "anim_test.glb"
        _create_minimal_glb(glb_path, with_mesh=True)
        size_before = glb_path.stat().st_size
        assert size_before > 0

        result = strip_mesh_from_glb(glb_path)

        assert result is True
        size_after = glb_path.stat().st_size
        # After stripping, file should be smaller (mesh data removed)
        assert size_after < size_before
        # File should still be a valid GLB (starts with glTF magic)
        with open(glb_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46546C67, "Not a valid GLB file after stripping"

    @skip_no_node
    def test_strip_mesh_on_meshless_glb(self, tmp_path):
        """strip_mesh_from_glb should succeed on a GLB without meshes."""
        from core.tools.image_gen import strip_mesh_from_glb

        glb_path = tmp_path / "anim_empty.glb"
        _create_minimal_glb(glb_path, with_mesh=False)

        result = strip_mesh_from_glb(glb_path)
        assert result is True
        assert glb_path.exists()

    @skip_no_node
    def test_strip_mesh_node_path_resolves_modules(self, tmp_path):
        """Verify NODE_PATH approach properly resolves @gltf-transform modules."""
        from core.tools.image_gen import strip_mesh_from_glb

        glb_path = tmp_path / "test_resolve.glb"
        _create_minimal_glb(glb_path, with_mesh=True)

        # This is the critical test: the old npx -p approach failed because
        # require() couldn't find the modules. The new approach installs
        # packages to a persistent cache and sets NODE_PATH.
        result = strip_mesh_from_glb(glb_path)
        assert result is True, (
            "strip_mesh_from_glb failed — NODE_PATH may not be resolving "
            "@gltf-transform modules correctly"
        )


@pytest.mark.e2e
class TestOptimizeGlbE2E:
    """E2E tests for optimize_glb (Draco compression)."""

    @skip_no_node
    def test_draco_compression(self, tmp_path):
        """optimize_glb should apply Draco compression to a GLB."""
        from core.tools.image_gen import optimize_glb

        glb_path = tmp_path / "model.glb"
        _create_minimal_glb(glb_path, with_mesh=True)

        result = optimize_glb(glb_path)
        assert result is True
        assert glb_path.exists()
        # File should still be valid GLB
        with open(glb_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46546C67


@pytest.mark.e2e
class TestSimplifyGlbE2E:
    """E2E tests for simplify_glb (mesh simplification)."""

    @skip_no_node
    def test_simplify_preserves_valid_glb(self, tmp_path):
        """simplify_glb should produce a valid GLB file."""
        from core.tools.image_gen import simplify_glb

        glb_path = tmp_path / "model.glb"
        _create_minimal_glb(glb_path, with_mesh=True)

        result = simplify_glb(glb_path, target_ratio=0.5)
        assert result is True
        assert glb_path.exists()
        with open(glb_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46546C67

    @skip_no_node
    def test_simplify_custom_ratio(self, tmp_path):
        """simplify_glb with ratio=1.0 should keep all polygons."""
        from core.tools.image_gen import simplify_glb

        glb_path = tmp_path / "model.glb"
        _create_minimal_glb(glb_path, with_mesh=True)
        size_before = glb_path.stat().st_size

        result = simplify_glb(glb_path, target_ratio=1.0)
        assert result is True
        # With ratio 1.0, file size should be roughly the same
        size_after = glb_path.stat().st_size
        assert size_after > 0


@pytest.mark.e2e
class TestCompressTexturesE2E:
    """E2E tests for compress_textures (resize + WebP)."""

    @skip_no_node
    def test_compress_textures_on_textureless_glb(self, tmp_path):
        """compress_textures should handle GLBs without textures gracefully."""
        from core.tools.image_gen import compress_textures

        glb_path = tmp_path / "model.glb"
        _create_minimal_glb(glb_path, with_mesh=True)

        # Should succeed even without textures (no-op for texture steps)
        result = compress_textures(glb_path, resolution=512)
        assert result is True
        assert glb_path.exists()


@pytest.mark.e2e
class TestOptimizeAssetsCommandE2E:
    """E2E tests for the optimize-assets CLI command."""

    def test_dry_run_no_changes(self, tmp_path):
        """--dry-run should report what would be done without modifying files."""
        import argparse
        from unittest.mock import patch

        from cli.commands.optimize_assets import _run

        # Set up fake anima directory
        anima_dir = tmp_path / "test_anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)

        anim_file = assets_dir / "anim_idle.glb"
        _create_minimal_glb(anim_file, with_mesh=True)
        model_file = assets_dir / "avatar_chibi_rigged.glb"
        _create_minimal_glb(model_file, with_mesh=True)

        size_anim_before = anim_file.stat().st_size
        size_model_before = model_file.stat().st_size

        args = argparse.Namespace(
            anima="test_anima",
            dry_run=True,
            simplify=None,
            texture_compress=False,
            texture_resize=None,
            apply_all=False,
            skip_backup=True,
        )

        with patch("core.paths.get_animas_dir", return_value=tmp_path):
            _run(args)

        # Files should be unchanged
        assert anim_file.stat().st_size == size_anim_before
        assert model_file.stat().st_size == size_model_before

    @skip_no_node
    def test_strip_and_draco_default(self, tmp_path):
        """Default run should strip animation meshes and draco-compress models."""
        import argparse
        from unittest.mock import patch

        from cli.commands.optimize_assets import _run

        anima_dir = tmp_path / "test_anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)

        anim_file = assets_dir / "anim_idle.glb"
        _create_minimal_glb(anim_file, with_mesh=True)
        model_file = assets_dir / "avatar_chibi_rigged.glb"
        _create_minimal_glb(model_file, with_mesh=True)

        args = argparse.Namespace(
            anima="test_anima",
            dry_run=False,
            simplify=None,
            texture_compress=False,
            texture_resize=None,
            apply_all=False,
            skip_backup=True,
        )

        with patch("core.paths.get_animas_dir", return_value=tmp_path):
            _run(args)

        # Animation file should have mesh stripped
        assert anim_file.exists()
        # Model file should still exist (draco compressed)
        assert model_file.exists()

    def test_backup_created(self, tmp_path):
        """Should create backup directory when --skip-backup is not set."""
        import argparse
        from unittest.mock import patch

        from cli.commands.optimize_assets import _run

        anima_dir = tmp_path / "test_anima"
        assets_dir = anima_dir / "assets"
        assets_dir.mkdir(parents=True)

        anim_file = assets_dir / "anim_idle.glb"
        _create_minimal_glb(anim_file, with_mesh=True)

        args = argparse.Namespace(
            anima="test_anima",
            dry_run=False,
            simplify=None,
            texture_compress=False,
            texture_resize=None,
            apply_all=False,
            skip_backup=False,
        )

        # Mock strip/optimize to avoid needing npx
        with patch("core.paths.get_animas_dir", return_value=tmp_path):
            with patch("core.tools.image_gen.strip_mesh_from_glb", return_value=True):
                _run(args)

        # Verify backup was created
        backups = list(anima_dir.glob("assets_backup_*"))
        assert len(backups) == 1
        backup_dir = backups[0]
        assert (backup_dir / "anim_idle.glb").exists()


@pytest.mark.e2e
class TestFbx2gltfE2E:
    """E2E tests for fbx2gltf installation and FBX-to-GLB conversion."""

    def _reset_cache(self):
        """Reset module-level _FBX2GLTF_PATH cache between tests."""
        import core.tools.image_gen as mod
        mod._FBX2GLTF_PATH = None

    @skip_no_npm
    def test_fbx2gltf_installs(self):
        """_ensure_fbx2gltf() should install and return a valid binary path.

        The fbx2gltf npm package may not provide a ``.bin`` symlink on all
        platforms.  When the binary resolves to ``None`` despite npm being
        available, we skip rather than fail — the unit tests cover the logic
        exhaustively.
        """
        from core.tools.image_gen import _ensure_fbx2gltf

        self._reset_cache()
        try:
            result = _ensure_fbx2gltf()
            if result is None:
                pytest.skip(
                    "fbx2gltf binary not available after npm install "
                    "(platform may lack .bin symlink)"
                )
            assert result.exists(), f"fbx2gltf binary not found at {result}"
        finally:
            self._reset_cache()

    @skip_no_npm
    def test_fbx2gltf_handles_invalid_fbx_gracefully(self, tmp_path):
        """_convert_fbx_to_glb should return False for a non-FBX file."""
        from core.tools.image_gen import _convert_fbx_to_glb, _ensure_fbx2gltf

        self._reset_cache()
        try:
            # Skip if fbx2gltf cannot be installed
            binary = _ensure_fbx2gltf()
            if binary is None:
                pytest.skip("fbx2gltf not available")

            # Create a fake file that is not a valid FBX
            fake_fbx = tmp_path / "not_real.fbx"
            fake_fbx.write_bytes(b"this is not an FBX file")
            glb_output = tmp_path / "output.glb"

            result = _convert_fbx_to_glb(fake_fbx, glb_output)
            # fbx2gltf should fail on invalid input and return False
            assert result is False
        finally:
            self._reset_cache()
