"""Unit tests for character.js bone-based bounding box fix.

Validates that _createGLBCharacter uses skeleton bone world positions
to compute bounding box instead of Box3.setFromObject, which misses
sibling armature scale on VRoid/Blender GLB models.

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
