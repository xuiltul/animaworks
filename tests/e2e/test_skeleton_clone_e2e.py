# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for SkinnedMesh skeleton cloning fix in the workspace.

Validates that character.js uses SkeletonUtils.clone() instead of
scene.clone(true) for proper skeleton cloning of GLB models, and that
all related dependencies (import map, model-cache integration) are
correctly wired.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _create_static_app() -> FastAPI:
    """Create a minimal FastAPI app that serves workspace static files."""
    app = FastAPI()
    static_dir = _PROJECT_ROOT / "server" / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


@pytest.mark.e2e
class TestSkeletonCloneE2E:
    """E2E tests for the SkinnedMesh skeleton cloning fix."""

    def test_character_js_served_with_skeleton_utils(self) -> None:
        """Start the FastAPI server and verify character.js uses SkeletonUtils.clone().

        Fetches /workspace/modules/character.js via HTTP and asserts that
        the _loadGLTFCached function uses SkeletonUtils.clone() and does
        NOT use scene.clone(true).
        """
        app = _create_static_app()
        client = TestClient(app)

        resp = client.get("/workspace/modules/character.js")
        assert resp.status_code == 200, (
            f"Expected 200 for character.js, got {resp.status_code}"
        )

        content = resp.text

        # The served file must use SkeletonUtils.clone() for proper
        # skeleton cloning of skinned meshes.
        assert "SkeletonUtils.clone(" in content, (
            "character.js must use SkeletonUtils.clone() for skeleton cloning"
        )

        # The old broken pattern scene.clone(true) must NOT be present
        # in the _loadGLTFCached function.  We check the whole file to
        # be safe — there should be no remaining scene.clone(true) calls.
        assert "scene.clone(true)" not in content, (
            "character.js must NOT contain scene.clone(true); "
            "use SkeletonUtils.clone() instead"
        )

    def test_skeleton_utils_module_accessible(self) -> None:
        """Verify the SkeletonUtils module URL is reachable.

        Sends an HTTP HEAD request to the esm.sh URL used in the import
        map to confirm the module will resolve at runtime.
        """
        url = (
            "https://esm.sh/three@0.172.0/examples/jsm/utils/SkeletonUtils.js"
        )
        resp = requests.head(url, timeout=10, allow_redirects=True)
        assert resp.status_code == 200, (
            f"SkeletonUtils module URL is not reachable: {resp.status_code}"
        )

    def test_character_js_import_map_includes_three_addons(self) -> None:
        """Verify the workspace import map includes a three/addons/ mapping.

        Reads server/static/workspace/index.html and checks that the
        import map contains a mapping for 'three/addons/' so that
        SkeletonUtils imports resolve correctly in the browser.
        """
        index_html = (
            _PROJECT_ROOT / "server" / "static" / "workspace" / "index.html"
        )
        assert index_html.exists(), f"index.html not found at {index_html}"

        content = index_html.read_text(encoding="utf-8")

        # The import map must include a three/addons/ entry that maps to
        # the esm.sh CDN path for three.js addon modules.
        assert '"three/addons/"' in content, (
            "import map must include a 'three/addons/' mapping"
        )
        assert "esm.sh/three@" in content, (
            "import map must reference the esm.sh CDN for three.js"
        )
        assert "examples/jsm/" in content, (
            "import map three/addons/ mapping must resolve to examples/jsm/"
        )

    def test_model_cache_clone_integration(self) -> None:
        """Verify character.js and model-cache.js are correctly integrated.

        Reads both source files and checks that:
        - model-cache.js exports modelCache
        - character.js imports modelCache from ./model-cache.js
        - Cached models go through SkeletonUtils.clone() before being
          returned (models from cache are never returned raw)
        """
        character_js = (
            _PROJECT_ROOT
            / "server"
            / "static"
            / "workspace"
            / "modules"
            / "character.js"
        )
        model_cache_js = (
            _PROJECT_ROOT
            / "server"
            / "static"
            / "workspace"
            / "modules"
            / "model-cache.js"
        )

        assert character_js.exists(), f"character.js not found at {character_js}"
        assert model_cache_js.exists(), (
            f"model-cache.js not found at {model_cache_js}"
        )

        char_content = character_js.read_text(encoding="utf-8")
        cache_content = model_cache_js.read_text(encoding="utf-8")

        # model-cache.js must export modelCache
        assert "export const modelCache" in cache_content or (
            "export { modelCache" in cache_content
            or "export {modelCache" in cache_content
        ), "model-cache.js must export modelCache"

        # character.js must import modelCache from ./model-cache.js
        assert "modelCache" in char_content, (
            "character.js must reference modelCache"
        )
        assert "model-cache.js" in char_content, (
            "character.js must import from ./model-cache.js"
        )

        # Every code path in _loadGLTFCached must clone via SkeletonUtils.
        # Extract the _loadGLTFCached function body and verify that all
        # return statements use SkeletonUtils.clone().
        func_start = char_content.find("async function _loadGLTFCached")
        assert func_start != -1, (
            "character.js must contain _loadGLTFCached function"
        )

        # Find the function body by counting braces
        brace_count = 0
        func_body_start = char_content.index("{", func_start)
        i = func_body_start
        while i < len(char_content):
            if char_content[i] == "{":
                brace_count += 1
            elif char_content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            i += 1
        func_body = char_content[func_body_start : i + 1]

        # Count return statements — each must use SkeletonUtils.clone()
        return_lines = [
            line.strip()
            for line in func_body.splitlines()
            if "return" in line and "scene" in line.lower()
        ]
        assert len(return_lines) > 0, (
            "_loadGLTFCached must have return statements with scene references"
        )
        for line in return_lines:
            assert "SkeletonUtils.clone(" in line, (
                f"Return in _loadGLTFCached must use SkeletonUtils.clone(), "
                f"found: {line}"
            )
