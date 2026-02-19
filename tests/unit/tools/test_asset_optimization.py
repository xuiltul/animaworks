"""Tests for 3D asset optimization: armature download, mesh stripping, GLB compression."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ── _ensure_fbx2gltf ────────────────────────────────────────────


class TestEnsureFbx2gltf:
    """Tests for _ensure_fbx2gltf binary resolution and installation."""

    def _reset_cache(self):
        """Reset module-level _FBX2GLTF_PATH cache between tests."""
        import core.tools.image_gen as mod
        mod._FBX2GLTF_PATH = None

    def test_returns_cached_path(self, tmp_path):
        """Should return immediately when _FBX2GLTF_PATH is set and exists."""
        import core.tools.image_gen as mod
        from core.tools.image_gen import _ensure_fbx2gltf

        cached = tmp_path / "fbx2gltf"
        cached.touch()
        mod._FBX2GLTF_PATH = cached
        try:
            result = _ensure_fbx2gltf()
            assert result == cached
        finally:
            self._reset_cache()

    def test_finds_system_binary(self):
        """Should use system binary when shutil.which('FBX2glTF') returns a path."""
        from core.tools.image_gen import _ensure_fbx2gltf

        self._reset_cache()
        try:
            with patch("shutil.which", return_value="/usr/local/bin/FBX2glTF"):
                result = _ensure_fbx2gltf()
                assert result == Path("/usr/local/bin/FBX2glTF")
        finally:
            self._reset_cache()

    def test_installs_via_npm(self, tmp_path):
        """Should run npm install and return bin path when not found elsewhere."""
        from core.tools.image_gen import _ensure_fbx2gltf

        self._reset_cache()
        # Prepare path but do NOT create the candidate yet — it should only
        # appear after npm install runs.
        bin_dir = tmp_path / "cache" / "fbx2gltf" / "node_modules" / ".bin"
        candidate = bin_dir / "fbx2gltf"

        def fake_npm_install(*args, **kwargs):
            """Simulate npm install creating the binary."""
            bin_dir.mkdir(parents=True, exist_ok=True)
            candidate.touch()
            return MagicMock(returncode=0)

        try:
            with patch("shutil.which", return_value=None):
                with patch("core.paths.get_data_dir", return_value=tmp_path):
                    with patch("subprocess.run", side_effect=fake_npm_install) as mock_run:
                        result = _ensure_fbx2gltf()

                        assert result == candidate
                        mock_run.assert_called_once()
                        cmd = mock_run.call_args.args[0]
                        assert "npm" in cmd
                        assert "install" in cmd
                        assert "fbx2gltf" in cmd
        finally:
            self._reset_cache()

    def test_returns_none_on_install_failure(self, tmp_path):
        """Should return None when npm install fails."""
        from core.tools.image_gen import _ensure_fbx2gltf

        self._reset_cache()
        try:
            with patch("shutil.which", return_value=None):
                with patch("core.paths.get_data_dir", return_value=tmp_path):
                    with patch("subprocess.run", side_effect=RuntimeError("npm not found")):
                        result = _ensure_fbx2gltf()
                        assert result is None
        finally:
            self._reset_cache()


# ── _convert_fbx_to_glb ─────────────────────────────────────────


class TestConvertFbxToGlb:
    """Tests for _convert_fbx_to_glb FBX-to-GLB conversion."""

    def test_returns_false_when_fbx2gltf_not_found(self):
        """Should return False when _ensure_fbx2gltf() returns None."""
        from core.tools.image_gen import _convert_fbx_to_glb

        with patch("core.tools.image_gen._ensure_fbx2gltf", return_value=None):
            result = _convert_fbx_to_glb(Path("/tmp/test.fbx"), Path("/tmp/test.glb"))
            assert result is False

    def test_calls_fbx2gltf_binary(self, tmp_path):
        """Should call fbx2gltf binary with correct arguments."""
        from core.tools.image_gen import _convert_fbx_to_glb

        fbx_path = tmp_path / "test.fbx"
        glb_path = tmp_path / "test.glb"
        fbx_path.write_bytes(b"fake-fbx")
        # Simulate fbx2gltf creating the output file
        glb_path.write_bytes(b"fake-glb")

        with patch("core.tools.image_gen._ensure_fbx2gltf", return_value=Path("/usr/bin/fbx2gltf")):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result = _convert_fbx_to_glb(fbx_path, glb_path)

                assert result is True
                mock_run.assert_called_once()
                cmd = mock_run.call_args.args[0]
                assert cmd[0] == "/usr/bin/fbx2gltf"
                assert "--binary" in cmd
                assert "--input" in cmd
                assert str(fbx_path) in cmd
                assert "--output" in cmd
                assert str(glb_path) in cmd

    def test_returns_false_on_subprocess_error(self):
        """Should return False when subprocess raises CalledProcessError."""
        import subprocess
        from core.tools.image_gen import _convert_fbx_to_glb

        with patch("core.tools.image_gen._ensure_fbx2gltf", return_value=Path("/usr/bin/fbx2gltf")):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "fbx2gltf")):
                result = _convert_fbx_to_glb(Path("/tmp/test.fbx"), Path("/tmp/test.glb"))
                assert result is False

    def test_returns_false_on_timeout(self):
        """Should return False when subprocess times out."""
        import subprocess
        from core.tools.image_gen import _convert_fbx_to_glb

        with patch("core.tools.image_gen._ensure_fbx2gltf", return_value=Path("/usr/bin/fbx2gltf")):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("fbx2gltf", 120)):
                result = _convert_fbx_to_glb(Path("/tmp/test.fbx"), Path("/tmp/test.glb"))
                assert result is False


# ── _download_armature_animation ─────────────────────────────────


class TestDownloadArmatureAnimation:
    """Tests for _download_armature_animation with fallback logic."""

    def test_downloads_armature_fbx_and_converts(self, tmp_path):
        """Happy path: armature URL present, FBX downloaded, conversion succeeds."""
        from core.tools.image_gen import _download_armature_animation

        glb_path = tmp_path / "anim_idle.glb"
        task = {
            "result": {
                "processed_armature_fbx_url": "https://example.com/armature.fbx",
                "animation_glb_url": "https://example.com/full.glb",
            }
        }

        mock_resp = MagicMock()
        mock_resp.content = b"fake-fbx-data"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            with patch("core.tools.image_gen._convert_fbx_to_glb", return_value=True) as mock_convert:
                # Simulate the glb_path existing after conversion for stat()
                glb_path.write_bytes(b"converted-glb")
                result = _download_armature_animation(task, glb_path)

                assert result is True
                # httpx.get called with armature URL
                mock_get.assert_called_once_with(
                    "https://example.com/armature.fbx",
                    timeout=mock_get.call_args.kwargs["timeout"],
                )
                # _convert_fbx_to_glb called
                mock_convert.assert_called_once()
                convert_args = mock_convert.call_args.args
                assert str(convert_args[0]).endswith(".fbx")
                assert convert_args[1] == glb_path

    def test_falls_back_when_no_armature_url(self, tmp_path):
        """Should fall back to full GLB + strip when no armature URL present."""
        from core.tools.image_gen import _download_armature_animation

        glb_path = tmp_path / "anim_idle.glb"
        task = {
            "result": {
                "animation_glb_url": "https://example.com/full.glb",
            }
        }

        mock_resp = MagicMock()
        mock_resp.content = b"full-glb-data"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            with patch("core.tools.image_gen.strip_mesh_from_glb", return_value=True) as mock_strip:
                result = _download_armature_animation(task, glb_path)

                assert result is True
                # Should have downloaded full GLB
                mock_get.assert_called_once()
                url_called = mock_get.call_args.args[0]
                assert url_called == "https://example.com/full.glb"
                # Should have stripped mesh
                mock_strip.assert_called_once_with(glb_path)
                # GLB file should be written
                assert glb_path.read_bytes() == b"full-glb-data"

    def test_falls_back_when_conversion_fails(self, tmp_path):
        """Should fall back to full GLB + strip when fbx2gltf conversion fails."""
        from core.tools.image_gen import _download_armature_animation

        glb_path = tmp_path / "anim_idle.glb"
        task = {
            "result": {
                "processed_armature_fbx_url": "https://example.com/armature.fbx",
                "animation_glb_url": "https://example.com/full.glb",
            }
        }

        call_count = 0

        def mock_get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "armature" in url:
                resp.content = b"fake-fbx"
            else:
                resp.content = b"full-glb-data"
            return resp

        with patch("httpx.get", side_effect=mock_get_side_effect):
            with patch("core.tools.image_gen._convert_fbx_to_glb", return_value=False):
                with patch("core.tools.image_gen.strip_mesh_from_glb", return_value=True) as mock_strip:
                    result = _download_armature_animation(task, glb_path)

                    assert result is True
                    # Two HTTP calls: armature FBX + fallback full GLB
                    assert call_count == 2
                    mock_strip.assert_called_once_with(glb_path)

    def test_falls_back_when_download_fails(self, tmp_path):
        """Should fall back when armature FBX download raises an exception."""
        from core.tools.image_gen import _download_armature_animation

        glb_path = tmp_path / "anim_idle.glb"
        task = {
            "result": {
                "processed_armature_fbx_url": "https://example.com/armature.fbx",
                "animation_glb_url": "https://example.com/full.glb",
            }
        }

        call_count = 0

        def mock_get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "armature" in url:
                raise httpx.HTTPStatusError(
                    "404 Not Found", request=MagicMock(), response=MagicMock(),
                )
            resp = MagicMock()
            resp.content = b"full-glb-data"
            resp.raise_for_status = MagicMock()
            return resp

        import httpx

        with patch("httpx.get", side_effect=mock_get_side_effect):
            with patch("core.tools.image_gen.strip_mesh_from_glb", return_value=True) as mock_strip:
                result = _download_armature_animation(task, glb_path)

                assert result is True
                assert call_count == 2
                mock_strip.assert_called_once_with(glb_path)

    def test_returns_false_when_no_urls(self):
        """Should return False when neither armature nor GLB URL is available."""
        from core.tools.image_gen import _download_armature_animation

        task = {"result": {}}
        result = _download_armature_animation(task, Path("/tmp/anim.glb"))
        assert result is False

    def test_cleans_up_temp_fbx(self, tmp_path):
        """Should delete the temporary FBX file even on conversion failure."""
        from core.tools.image_gen import _download_armature_animation

        glb_path = tmp_path / "anim_idle.glb"
        task = {
            "result": {
                "processed_armature_fbx_url": "https://example.com/armature.fbx",
                "animation_glb_url": "https://example.com/full.glb",
            }
        }

        created_fbx_paths: list[Path] = []

        mock_resp = MagicMock()
        mock_resp.content = b"fake-fbx"
        mock_resp.raise_for_status = MagicMock()

        # Track the temp FBX path created by _download_armature_animation
        original_named_temp = tempfile.NamedTemporaryFile

        def tracking_temp(*args, **kwargs):
            f = original_named_temp(*args, **kwargs)
            created_fbx_paths.append(Path(f.name))
            return f

        fallback_resp = MagicMock()
        fallback_resp.content = b"full-glb"
        fallback_resp.raise_for_status = MagicMock()

        call_count = 0

        def mock_get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "armature" in url:
                return mock_resp
            return fallback_resp

        with patch("httpx.get", side_effect=mock_get_side_effect):
            with patch("core.tools.image_gen._convert_fbx_to_glb", return_value=False):
                with patch("core.tools.image_gen.strip_mesh_from_glb", return_value=True):
                    with patch("tempfile.NamedTemporaryFile", side_effect=tracking_temp):
                        result = _download_armature_animation(task, glb_path)

        assert result is True
        # Verify temp FBX was created and then cleaned up
        assert len(created_fbx_paths) == 1
        assert not created_fbx_paths[0].exists(), "Temp FBX file was not cleaned up"


# ── create_animation_task post_process ───────────────────────────


class TestCreateAnimationTaskPostProcess:
    """Tests for create_animation_task extract_armature post_process parameter."""

    def _make_client(self):
        with patch("core.tools.image_gen.get_credential", return_value="test-key"):
            from core.tools.image_gen import MeshyClient
            return MeshyClient()

    def test_includes_extract_armature(self):
        """Should include post_process with operation_type extract_armature in request body."""
        client = self._make_client()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"result": "task-123"}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = client.create_animation_task("rig-001", 42)

            assert result == "task-123"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
            assert "post_process" in body
            assert body["post_process"] == {"operation_type": "extract_armature"}
            assert body["rig_task_id"] == "rig-001"
            assert body["action_id"] == 42


# ── download_rigging_animations ──────────────────────────────────


class TestDownloadRiggingAnimations:
    """Tests for MeshyClient.download_rigging_animations armature preference."""

    def _make_client(self):
        with patch("core.tools.image_gen.get_credential", return_value="test-key"):
            from core.tools.image_gen import MeshyClient
            return MeshyClient()

    def test_prefers_armature_glb_url(self):
        """Should prefer armature-only URL over full model URL."""
        client = self._make_client()
        task = {
            "result": {
                "basic_animations": {
                    "walking_glb_url": "https://example.com/walking_full.glb",
                    "walking_armature_glb_url": "https://example.com/walking_armature.glb",
                    "running_glb_url": "https://example.com/running_full.glb",
                    "running_armature_glb_url": "https://example.com/running_armature.glb",
                }
            }
        }
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = b"armature-data"
            mock_get.return_value = mock_resp

            result = client.download_rigging_animations(task)

            # Verify armature URLs were used
            urls_called = [c.args[0] for c in mock_get.call_args_list]
            assert "https://example.com/walking_armature.glb" in urls_called
            assert "https://example.com/running_armature.glb" in urls_called
            assert "https://example.com/walking_full.glb" not in urls_called

    def test_falls_back_to_full_glb(self):
        """Should fall back to full GLB URL when armature URL is missing."""
        client = self._make_client()
        task = {
            "result": {
                "basic_animations": {
                    "walking_glb_url": "https://example.com/walking_full.glb",
                    # No armature URLs
                    "running_glb_url": "https://example.com/running_full.glb",
                }
            }
        }
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = b"full-data"
            mock_get.return_value = mock_resp

            result = client.download_rigging_animations(task)

            urls_called = [c.args[0] for c in mock_get.call_args_list]
            assert "https://example.com/walking_full.glb" in urls_called
            assert "https://example.com/running_full.glb" in urls_called

    def test_empty_basic_animations(self):
        """Should handle empty basic_animations gracefully."""
        client = self._make_client()
        task = {"result": {"basic_animations": {}}}
        result = client.download_rigging_animations(task)
        assert result == {}


# ── strip_mesh_from_glb ──────────────────────────────────────────


class TestStripMeshFromGlb:
    """Tests for strip_mesh_from_glb helper."""

    def test_returns_false_when_node_not_found(self):
        """Should return False and log warning when node is not installed."""
        from core.tools.image_gen import strip_mesh_from_glb

        with patch("shutil.which", return_value=None):
            result = strip_mesh_from_glb(Path("/tmp/test.glb"))
            assert result is False

    def test_returns_false_on_subprocess_error(self):
        """Should return False when subprocess fails."""
        import subprocess
        from core.tools.image_gen import strip_mesh_from_glb

        with patch("shutil.which", return_value="/usr/bin/node"):
            with patch("core.tools.image_gen._ensure_gltf_transform_modules", return_value=Path("/fake/node_modules")):
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "node")):
                    result = strip_mesh_from_glb(Path("/tmp/test.glb"))
                    assert result is False

    def test_returns_false_on_timeout(self):
        """Should return False when subprocess times out."""
        import subprocess
        from core.tools.image_gen import strip_mesh_from_glb

        with patch("shutil.which", return_value="/usr/bin/node"):
            with patch("core.tools.image_gen._ensure_gltf_transform_modules", return_value=Path("/fake/node_modules")):
                with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("node", 120)):
                    result = strip_mesh_from_glb(Path("/tmp/test.glb"))
                    assert result is False

    def test_uses_node_path_with_temp_script(self, tmp_path):
        """Should write script to temp file and set NODE_PATH for module resolution."""
        from core.tools.image_gen import strip_mesh_from_glb

        glb_path = tmp_path / "test.glb"
        glb_path.write_bytes(b"fake-glb")
        fake_modules = Path("/fake/node_modules")
        with patch("shutil.which", return_value="/usr/bin/node"):
            with patch("core.tools.image_gen._ensure_gltf_transform_modules", return_value=fake_modules):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    result = strip_mesh_from_glb(glb_path)

                    assert result is True
                    call_kwargs = mock_run.call_args
                    cmd = call_kwargs.args[0]
                    # Verify node is called directly (not npx)
                    assert cmd[0] == "/usr/bin/node"
                    # Verify NODE_PATH is set in env
                    env = call_kwargs.kwargs.get("env", {})
                    assert env.get("NODE_PATH") == str(fake_modules)

    def test_cleans_up_temp_script_on_failure(self):
        """Should clean up temp script file even when subprocess fails."""
        import subprocess
        import tempfile
        from core.tools.image_gen import strip_mesh_from_glb

        with patch("shutil.which", return_value="/usr/bin/node"):
            with patch("core.tools.image_gen._ensure_gltf_transform_modules", return_value=Path("/fake/node_modules")):
                with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "node")):
                    result = strip_mesh_from_glb(Path("/tmp/test.glb"))
                    assert result is False
                    # Verify no leftover .cjs files in temp dir
                    temp_dir = Path(tempfile.gettempdir())
                    cjs_files = list(temp_dir.glob("tmp*.cjs"))
                    assert len(cjs_files) == 0, f"Leftover temp files: {cjs_files}"


# ── optimize_glb ─────────────────────────────────────────────────


class TestOptimizeGlb:
    """Tests for optimize_glb helper."""

    def test_returns_false_when_npx_not_found(self):
        """Should return False when npx is not installed."""
        from core.tools.image_gen import optimize_glb

        with patch("shutil.which", return_value=None):
            result = optimize_glb(Path("/tmp/test.glb"))
            assert result is False

    def test_calls_optimize_then_draco(self):
        """Should call gltf-transform optimize then draco."""
        from core.tools.image_gen import _run_gltf_transform

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result = _run_gltf_transform(["optimize", "in.glb", "out.glb"], Path("in.glb"))
                assert result is True
                cmd = mock_run.call_args.args[0]
                assert "@gltf-transform/cli" in cmd
                assert "optimize" in cmd

    def test_returns_false_on_subprocess_error(self):
        """Should return False when gltf-transform fails."""
        import subprocess
        from core.tools.image_gen import _run_gltf_transform

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "npx", stderr=b"error")):
                result = _run_gltf_transform(["optimize", "in.glb", "out.glb"], Path("in.glb"))
                assert result is False


# ── simplify_glb ─────────────────────────────────────────────────


class TestSimplifyGlb:
    """Tests for simplify_glb helper."""

    def test_returns_false_when_npx_not_found(self):
        """Should return False when npx is not installed."""
        from core.tools.image_gen import simplify_glb

        with patch("shutil.which", return_value=None):
            result = simplify_glb(Path("/tmp/test.glb"))
            assert result is False

    def test_calls_gltf_transform_simplify(self):
        """Should call gltf-transform simplify with correct args."""
        from core.tools.image_gen import simplify_glb

        glb_path = Path("/tmp/test.glb")
        simp_path = glb_path.with_suffix(".simp.glb")

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Mock rename and stat
                with patch.object(Path, "rename") as mock_rename:
                    with patch.object(Path, "stat") as mock_stat:
                        mock_stat.return_value.st_size = 5000
                        with patch.object(Path, "unlink"):
                            result = simplify_glb(glb_path, target_ratio=0.27, error_threshold=0.01)

                            assert result is True
                            cmd = mock_run.call_args.args[0]
                            assert "simplify" in cmd
                            assert "--ratio" in cmd
                            assert "0.27" in cmd
                            assert "--error" in cmd

    def test_cleans_up_temp_file_on_failure(self):
        """Should clean up .simp.glb temp file on failure."""
        import subprocess
        from core.tools.image_gen import simplify_glb

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "npx", stderr=b"err")):
                with patch.object(Path, "unlink") as mock_unlink:
                    result = simplify_glb(Path("/tmp/test.glb"))
                    assert result is False
                    mock_unlink.assert_called()

    def test_custom_ratio(self):
        """Should pass custom ratio and error values."""
        from core.tools.image_gen import simplify_glb

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch.object(Path, "rename"):
                    with patch.object(Path, "stat") as mock_stat:
                        mock_stat.return_value.st_size = 3000
                        with patch.object(Path, "unlink"):
                            simplify_glb(Path("/tmp/test.glb"), target_ratio=0.5, error_threshold=0.02)
                            cmd = mock_run.call_args.args[0]
                            assert "0.5" in cmd
                            assert "0.02" in cmd


# ── compress_textures ────────────────────────────────────────────


class TestCompressTextures:
    """Tests for compress_textures helper."""

    def test_returns_false_when_npx_not_found(self):
        """Should return False when npx is not installed."""
        from core.tools.image_gen import compress_textures

        with patch("shutil.which", return_value=None):
            result = compress_textures(Path("/tmp/test.glb"))
            assert result is False

    def test_calls_resize_then_webp(self):
        """Should call gltf-transform resize then webp."""
        from core.tools.image_gen import compress_textures

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch.object(Path, "unlink"):
                    with patch.object(Path, "stat") as mock_stat:
                        mock_stat.return_value.st_size = 2000
                        with patch.object(Path, "rename"):
                            result = compress_textures(Path("/tmp/test.glb"), resolution=1024)

                            assert result is True
                            calls = mock_run.call_args_list
                            assert len(calls) >= 2
                            # First call should be resize
                            assert "resize" in calls[0].args[0]
                            assert "1024" in calls[0].args[0]
                            # Second call should be webp
                            assert "webp" in calls[1].args[0]

    def test_returns_true_if_resize_succeeds_but_webp_fails(self):
        """Should keep resized version if webp conversion fails."""
        import subprocess
        from core.tools.image_gen import compress_textures

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args[0]
            if "webp" in cmd:
                raise subprocess.CalledProcessError(1, "npx", stderr=b"webp err")
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run", side_effect=side_effect):
                with patch.object(Path, "rename") as mock_rename:
                    with patch.object(Path, "unlink"):
                        result = compress_textures(Path("/tmp/test.glb"))
                        # Should still return True (resize worked)
                        assert result is True

    def test_returns_false_if_resize_fails(self):
        """Should return False if resize step fails."""
        import subprocess
        from core.tools.image_gen import compress_textures

        with patch("shutil.which", return_value="/usr/bin/npx"):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "npx", stderr=b"err")):
                with patch.object(Path, "unlink"):
                    result = compress_textures(Path("/tmp/test.glb"))
                    assert result is False


# ── optimize-assets CLI command ──────────────────────────────────


class TestOptimizeAssetsCommand:
    """Tests for the optimize-assets CLI command argument parsing."""

    def test_register_creates_subcommand(self):
        """Should register optimize-assets subcommand with all options."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        # Verify the subcommand was registered by parsing known args
        args = parser.parse_args(["optimize-assets", "--dry-run"])
        assert args.dry_run is True

    def test_all_flag_parsed(self):
        """Should parse --all flag correctly."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        args = parser.parse_args(["optimize-assets", "--all"])
        assert args.apply_all is True

    def test_simplify_with_default_ratio(self):
        """Should use default ratio 0.27 when --simplify is used without value."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        args = parser.parse_args(["optimize-assets", "--simplify"])
        assert args.simplify == 0.27

    def test_simplify_with_custom_ratio(self):
        """Should accept custom ratio for --simplify."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        args = parser.parse_args(["optimize-assets", "--simplify", "0.5"])
        assert args.simplify == 0.5

    def test_texture_options(self):
        """Should parse texture options correctly."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        args = parser.parse_args(["optimize-assets", "--texture-compress", "--texture-resize", "512"])
        assert args.texture_compress is True
        assert args.texture_resize == 512

    def test_skip_backup_flag(self):
        """Should parse --skip-backup flag."""
        from cli.commands.optimize_assets import register

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)

        args = parser.parse_args(["optimize-assets", "--skip-backup"])
        assert args.skip_backup is True

    def test_nonexistent_anima_exits_early(self, tmp_path, capsys):
        """Should print error and return when specified anima does not exist."""
        from cli.commands.optimize_assets import _run

        args = argparse.Namespace(
            anima="nonexistent_anima",
            dry_run=False,
            simplify=None,
            texture_compress=False,
            texture_resize=None,
            apply_all=False,
            skip_backup=True,
        )

        with patch("core.paths.get_animas_dir", return_value=tmp_path):
            _run(args)

        captured = capsys.readouterr()
        assert "Anima not found: nonexistent_anima" in captured.out
