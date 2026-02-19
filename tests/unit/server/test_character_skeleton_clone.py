# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SkinnedMesh cloning fix in character.js.

Validates that _loadGLTFCached uses SkeletonUtils.clone() instead of
scene.clone(true), which is required for correct skeleton/bone binding
when reusing cached GLTF models with SkinnedMesh.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CHAR_JS = (
    Path(__file__).resolve().parents[3]
    / "server"
    / "static"
    / "workspace"
    / "modules"
    / "character.js"
)


@pytest.fixture
def character_js_source() -> str:
    """Read the character.js source once per test."""
    return _CHAR_JS.read_text(encoding="utf-8")


def _extract_load_gltf_cached_body(source: str) -> str:
    """Extract the function body of _loadGLTFCached from the JS source.

    Matches from ``async function _loadGLTFCached`` to the closing ``}``
    at the same indentation level by counting braces.
    """
    match = re.search(r"async function _loadGLTFCached\b", source)
    assert match is not None, "_loadGLTFCached function not found in character.js"

    # Find the opening brace
    brace_start = source.index("{", match.start())
    depth = 0
    for i in range(brace_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start : i + 1]

    raise AssertionError("Could not find closing brace for _loadGLTFCached")


# ── Tests ──────────────────────────────────────────────────────


class TestCharacterSkeletonCloneFix:
    """Verify the SkinnedMesh cloning fix in character.js."""

    def test_skeleton_utils_imported(self, character_js_source: str) -> None:
        """character.js must import SkeletonUtils from the Three.js addons."""
        assert (
            'import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js"'
            in character_js_source
        )

    def test_load_gltf_cached_uses_skeleton_utils_clone(
        self, character_js_source: str
    ) -> None:
        """_loadGLTFCached must use SkeletonUtils.clone() for scene cloning."""
        body = _extract_load_gltf_cached_body(character_js_source)
        assert "SkeletonUtils.clone(" in body

    def test_no_scene_clone_true_in_load_gltf_cached(
        self, character_js_source: str
    ) -> None:
        """_loadGLTFCached must NOT call .clone(true) on the scene.

        scene.clone(true) does not rebind skeletons, causing all cloned
        SkinnedMesh instances to share the same bone references and
        animate as a single unit.
        """
        body = _extract_load_gltf_cached_body(character_js_source)
        assert ".clone(true)" not in body

    def test_animation_clips_not_cloned(self, character_js_source: str) -> None:
        """_loadGLTFCached must NOT clone AnimationClips.

        AnimationClips are pure data (keyframe tracks) and can be safely
        shared across multiple AnimationMixer instances without cloning.
        Cloning them wastes memory and is unnecessary.
        """
        body = _extract_load_gltf_cached_body(character_js_source)
        assert ".map(c => c.clone())" not in body

    def test_skeleton_utils_import_is_namespace_import(
        self, character_js_source: str
    ) -> None:
        """SkeletonUtils must be imported as a namespace (import * as ...).

        The SkeletonUtils module exports individual functions (clone, etc.)
        rather than a single default class, so ``import * as SkeletonUtils``
        is the correct form. ``import { SkeletonUtils }`` would fail at
        runtime because there is no named export called SkeletonUtils.
        """
        # Must have namespace import
        assert re.search(
            r"import\s+\*\s+as\s+SkeletonUtils\s+from\b", character_js_source
        ), "Expected namespace import: import * as SkeletonUtils from ..."

        # Must NOT have destructured named import
        assert not re.search(
            r"import\s*\{\s*SkeletonUtils\s*\}\s*from\b", character_js_source
        ), "Found incorrect named import: import { SkeletonUtils } from ..."
