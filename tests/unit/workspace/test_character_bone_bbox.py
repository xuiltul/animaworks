# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for character.js bone-based bounding box and animation-first scaling.

Validates that _createGLBCharacter:
1. Uses skeleton bone world positions for bounding box (not Box3.setFromObject)
2. Loads animations BEFORE computing bounding box to normalise bind-pose
   differences (natsume's Hips ~50° tilt gave half the Y-extent)
3. Applies a sanity cap (maxHeight=0.8) to prevent oversized characters

These tests read the source file and verify the correct code patterns
are present (same approach as test_skeleton_clone_e2e.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CHARACTER_JS = (
    _PROJECT_ROOT / "server" / "static" / "workspace" / "modules" / "character.js"
)


@pytest.fixture
def character_js_content() -> str:
    """Read character.js content for assertions."""
    assert _CHARACTER_JS.exists(), f"character.js not found at {_CHARACTER_JS}"
    return _CHARACTER_JS.read_text(encoding="utf-8")


@pytest.fixture
def create_glb_body(character_js_content: str) -> str:
    """Extract the _createGLBCharacter function body."""
    func_start = character_js_content.find("async function _createGLBCharacter")
    assert func_start != -1, "character.js must contain _createGLBCharacter"

    brace_count = 0
    body_start = character_js_content.index("{", func_start)
    i = body_start
    while i < len(character_js_content):
        if character_js_content[i] == "{":
            brace_count += 1
        elif character_js_content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                break
        i += 1
    return character_js_content[body_start : i + 1]


class TestBoneBasedBoundingBox:
    """Verify character.js uses bone world positions for bounding box."""

    def test_uses_skeleton_bones_for_bbox(self, create_glb_body: str) -> None:
        """_createGLBCharacter must iterate skeleton bones for bounding box."""
        assert "child.skeleton" in create_glb_body, (
            "_createGLBCharacter must access child.skeleton for bone-based bbox"
        )
        assert "skeleton.bones" in create_glb_body, (
            "_createGLBCharacter must iterate skeleton.bones"
        )

    def test_uses_get_world_position(self, create_glb_body: str) -> None:
        """Bone positions must be obtained via getWorldPosition."""
        assert "getWorldPosition" in create_glb_body, (
            "_createGLBCharacter must use bone.getWorldPosition() for accurate bounds"
        )

    def test_uses_expand_by_point(self, create_glb_body: str) -> None:
        """Box3 should be expanded point-by-point from bone positions."""
        assert "expandByPoint" in create_glb_body, (
            "_createGLBCharacter must use box.expandByPoint() for bone-based bbox"
        )

    def test_checks_is_skinned_mesh(self, create_glb_body: str) -> None:
        """Must check isSkinnedMesh before accessing skeleton."""
        assert "isSkinnedMesh" in create_glb_body, (
            "_createGLBCharacter must check child.isSkinnedMesh"
        )

    def test_has_fallback_for_non_skinned(self, create_glb_body: str) -> None:
        """Non-skinned models must still use Box3.setFromObject as fallback."""
        assert "setFromObject" in create_glb_body, (
            "_createGLBCharacter must have Box3.setFromObject fallback "
            "for non-skinned models"
        )
        assert "hasBones" in create_glb_body, (
            "_createGLBCharacter must track hasBones flag for fallback logic"
        )

    def test_no_bare_set_from_object_for_initial_bbox(
        self, create_glb_body: str,
    ) -> None:
        """The initial bounding box must NOT use bare Box3.setFromObject(model).

        The old pattern was:
            const box = new THREE.Box3().setFromObject(model);

        This must be replaced with the bone-based approach. setFromObject
        should only appear inside an if(!hasBones) fallback.
        """
        # The old one-liner pattern should not exist
        assert "new THREE.Box3().setFromObject(model)" not in create_glb_body, (
            "_createGLBCharacter must not use Box3().setFromObject(model) directly; "
            "use bone-based approach with setFromObject only as fallback"
        )

    def test_calls_update_matrix_world(self, create_glb_body: str) -> None:
        """Must call model.updateMatrixWorld(true) before reading bone positions."""
        assert "updateMatrixWorld" in create_glb_body, (
            "_createGLBCharacter must call updateMatrixWorld(true) before "
            "reading bone world positions"
        )

    def test_scaled_box_also_uses_bone_approach(
        self, create_glb_body: str,
    ) -> None:
        """The centering/scaledBox computation must also use bone-based bounds."""
        # There should be two Box3 instances: one for initial height, one for centering
        box_count = create_glb_body.count("new THREE.Box3()")
        assert box_count >= 2, (
            f"Expected at least 2 Box3 instances (initial + centering), found {box_count}"
        )

        # The scaledBox section should also reference skeleton bones
        scaled_idx = create_glb_body.find("scaledBox")
        assert scaled_idx != -1, "Must have scaledBox for centering"

        scaled_section = create_glb_body[scaled_idx:]
        assert "skeleton.bones" in scaled_section or "hasBones" in scaled_section, (
            "Centering (scaledBox) must also use bone-based bounds or hasBones check"
        )


class TestAnimationFirstScaling:
    """Verify animations are loaded and applied BEFORE bounding box computation.

    natsume's bind-pose Hips bone is rotated ~50° on X-axis, giving a
    skeleton Y-extent of ~0.835 vs ~1.6 for other characters.  This caused
    a 2x scale factor.  The fix loads the idle animation and applies
    frame 0 before computing the bounding box, so every skeleton is in a
    comparable upright pose.
    """

    def test_animation_mixer_created_before_bbox(
        self, create_glb_body: str,
    ) -> None:
        """AnimationMixer must be created before Box3 bounding box."""
        mixer_idx = create_glb_body.find("new THREE.AnimationMixer")
        box_idx = create_glb_body.find("new THREE.Box3()")
        assert mixer_idx != -1, "Must create AnimationMixer"
        assert box_idx != -1, "Must create Box3"
        assert mixer_idx < box_idx, (
            "AnimationMixer must be created BEFORE Box3 to ensure "
            "animation is applied before bounding box computation"
        )

    def test_load_animation_clips_before_bbox(
        self, create_glb_body: str,
    ) -> None:
        """_loadAnimationClips must be called before Box3 computation."""
        load_idx = create_glb_body.find("_loadAnimationClips")
        box_idx = create_glb_body.find("new THREE.Box3()")
        assert load_idx != -1, "Must call _loadAnimationClips"
        assert box_idx != -1, "Must create Box3"
        assert load_idx < box_idx, (
            "_loadAnimationClips must be called BEFORE Box3 computation"
        )

    def test_mixer_set_time_before_bbox(
        self, create_glb_body: str,
    ) -> None:
        """mixer.setTime(0) must be called before Box3 to apply idle frame 0."""
        set_time_idx = create_glb_body.find("mixer.setTime(0)")
        box_idx = create_glb_body.find("new THREE.Box3()")
        assert set_time_idx != -1, (
            "Must call mixer.setTime(0) to apply idle animation frame 0"
        )
        assert box_idx != -1, "Must create Box3"
        assert set_time_idx < box_idx, (
            "mixer.setTime(0) must be called BEFORE Box3 bounding box "
            "computation to put bones in upright pose"
        )

    def test_idle_action_play_before_set_time(
        self, create_glb_body: str,
    ) -> None:
        """idleAction.play() must precede mixer.setTime(0)."""
        play_idx = create_glb_body.find("idleAction.play()")
        set_time_idx = create_glb_body.find("mixer.setTime(0)")
        assert play_idx != -1, "Must call idleAction.play()"
        assert set_time_idx != -1, "Must call mixer.setTime(0)"
        assert play_idx < set_time_idx, (
            "idleAction.play() must precede mixer.setTime(0)"
        )

    def test_update_matrix_world_after_animation_before_bbox(
        self, create_glb_body: str,
    ) -> None:
        """updateMatrixWorld must be called after animation, before bbox."""
        set_time_idx = create_glb_body.find("mixer.setTime(0)")
        # First updateMatrixWorld after setTime
        umw_idx = create_glb_body.find("updateMatrixWorld", set_time_idx)
        box_idx = create_glb_body.find("new THREE.Box3()")
        assert set_time_idx != -1, "Must call mixer.setTime(0)"
        assert umw_idx != -1, "Must call updateMatrixWorld after setTime"
        assert box_idx != -1, "Must create Box3"
        assert set_time_idx < umw_idx < box_idx, (
            "updateMatrixWorld must be called after mixer.setTime(0) "
            "and before Box3 computation"
        )

    def test_uses_state_anim_files(self, create_glb_body: str) -> None:
        """Must reference _STATE_ANIM_FILES for animation clip mapping."""
        assert "_STATE_ANIM_FILES" in create_glb_body, (
            "Must use _STATE_ANIM_FILES to map state names to animation files"
        )

    def test_embedded_animation_fallback(self, create_glb_body: str) -> None:
        """Must fall back to embedded gltf.animations when no external idle."""
        assert "gltf.animations" in create_glb_body, (
            "Must check gltf.animations as fallback for idle animation"
        )


class TestScaleSanityCap:
    """Verify scale computation includes a sanity cap (maxHeight=0.8)."""

    def test_max_height_constant_exists(self, create_glb_body: str) -> None:
        """maxHeight constant must be defined in the function."""
        assert "maxHeight" in create_glb_body, (
            "Must define maxHeight sanity cap constant"
        )

    def test_max_height_value_is_0_8(self, create_glb_body: str) -> None:
        """maxHeight must be 0.8 (based on other characters' ~0.7 target)."""
        assert "maxHeight = 0.8" in create_glb_body, (
            "maxHeight must be set to 0.8 to cap oversized characters"
        )

    def test_target_height_is_0_7(self, create_glb_body: str) -> None:
        """targetHeight must remain at 0.7."""
        assert "targetHeight = 0.7" in create_glb_body, (
            "targetHeight must be 0.7 for standard character height"
        )

    def test_scale_cap_logic(self, create_glb_body: str) -> None:
        """Scale computation must use maxHeight to cap the result."""
        assert "maxHeight / height" in create_glb_body, (
            "Scale computation must include maxHeight / height cap logic"
        )

    def test_raw_scale_computed(self, create_glb_body: str) -> None:
        """Must compute rawScale before applying the cap."""
        assert "rawScale" in create_glb_body, (
            "Must compute rawScale (targetHeight / height) before capping"
        )
