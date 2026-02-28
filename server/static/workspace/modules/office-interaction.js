// ── Office Interaction ──────────────────────
// Camera setup, raycasting, click handling, and desk highlighting.

import * as THREE from "three";
import { COLOR } from "./office-structure.js";

// ── Camera ──────────────────────

/**
 * Create an orthographic camera with isometric-like perspective.
 * @param {number} aspect - container width / height
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {THREE.OrthographicCamera}
 */
export function createCamera(aspect, floorWidth, floorDepth) {
  const frustum = Math.max(8, Math.max(floorWidth, floorDepth) / 2 + 2);
  const camera = new THREE.OrthographicCamera(
    -frustum * aspect,
     frustum * aspect,
     frustum,
    -frustum,
    0.1,
    100,
  );
  camera.position.set(10, 12, 10);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * Update camera frustum on container resize.
 * @param {THREE.OrthographicCamera} camera
 * @param {THREE.WebGLRenderer} renderer
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {number} width
 * @param {number} height
 */
export function updateCameraFrustum(camera, renderer, floorWidth, floorDepth, width, height) {
  const aspect = width / height;
  const frustum = Math.max(8, Math.max(floorWidth, floorDepth) / 2 + 2);
  camera.left = -frustum * aspect;
  camera.right = frustum * aspect;
  camera.top = frustum;
  camera.bottom = -frustum;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// ── Raycasting / Click ──────────────────────

const _raycaster = new THREE.Raycaster();
const _mouse = new THREE.Vector2();

/**
 * Handle pointer click for desk/character selection.
 * @param {MouseEvent} event
 * @param {THREE.WebGLRenderer} renderer
 * @param {THREE.OrthographicCamera} camera
 * @param {THREE.Scene} scene
 * @param {Map<string, string>} meshToName
 * @param {((name: string) => void) | null} characterClickHandler
 */
export function onPointerClick(event, renderer, camera, scene, meshToName, characterClickHandler) {
  const rect = renderer.domElement.getBoundingClientRect();
  _mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  _mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  _raycaster.setFromCamera(_mouse, camera);

  const intersects = _raycaster.intersectObjects(scene.children, true);

  for (const hit of intersects) {
    const name = meshToName.get(hit.object.uuid);
    if (name) {
      if (characterClickHandler) {
        characterClickHandler(name);
      }
      return;
    }
  }
}

// ── Highlight System ──────────────────────

/**
 * Create the shared highlight indicator mesh (a flat ring under the desk).
 * @param {Set} disposables
 * @param {{phong: Function}} helpers
 * @returns {THREE.Mesh}
 */
export function createHighlightMesh(disposables, helpers) {
  const geo = new THREE.RingGeometry(0.6, 0.8, 32);
  disposables.add(geo);

  const mat = helpers.phong({
    color: COLOR.highlight,
    emissive: COLOR.highlight,
    emissiveIntensity: 0.5,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.02;
  mesh.visible = false;
  return mesh;
}

/**
 * Highlight a specific desk by positioning the ring mesh.
 * @param {string} name
 * @param {THREE.Mesh} highlightMesh
 * @param {Record<string, {x: number, y: number, z: number}>} deskLayout
 * @returns {string | null} The highlighted name, or null if not found.
 */
export function highlightDesk(name, highlightMesh, deskLayout) {
  const pos = deskLayout[name];
  if (!pos) {
    console.warn(`[office3d] Unknown desk name: "${name}"`);
    return null;
  }

  highlightMesh.position.set(pos.x, 0.02, pos.z);
  highlightMesh.visible = true;
  return name;
}

/**
 * Remove any active desk highlight.
 * @param {THREE.Mesh} highlightMesh
 */
export function clearHighlight(highlightMesh) {
  if (highlightMesh) {
    highlightMesh.visible = false;
  }
}

/**
 * Register additional meshes for raycasting.
 * @param {string} animaName
 * @param {THREE.Object3D} object
 * @param {Map<string, string>} meshToName
 */
export function registerClickTarget(animaName, object, meshToName) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      meshToName.set(child.uuid, animaName);
    }
  });
}

/**
 * Unregister meshes from raycasting.
 * @param {THREE.Object3D} object
 * @param {Map<string, string>} meshToName
 */
export function unregisterClickTarget(object, meshToName) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      meshToName.delete(child.uuid);
    }
  });
}
