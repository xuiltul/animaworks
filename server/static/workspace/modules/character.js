// ── Character Module (Orchestrator) ──────────────────────
// Manages the character lifecycle: creation, state transitions, animation
// updates and disposal.  Delegates to sub-modules for loading, appearance,
// animation and effects.  Owns _scene and _characters as the single source
// of truth for the character registry.

import * as THREE from "three";
import { probeAsset, fetchAssetMetadata } from "./api.js";
import { createGLBCharacter, createProceduralCharacter, applyMetadataColor } from "./character-loader.js";
import { animMap, resetPose, easeInOutCubic, cleanupState } from "./character-animation.js";
import { STATES, profileCache, ensureGeometries, geo, applyFaceTexture } from "./character-appearance.js";
import { hideStatusSprite, clearParticles } from "./character-effects.js";

// ── Constants ──────────────────────

/** Duration in seconds for lerp-based state transitions. */
const TRANSITION_DURATION = 0.4;

// ── Module State ──────────────────────

/** @type {THREE.Scene | null} */
let _scene = null;

/**
 * @typedef {Object} CharacterRecord
 * @property {string}          name
 * @property {THREE.Group}     group
 * @property {string}          state          - Current animation state
 * @property {string}          prevState      - Previous state (for transitions)
 * @property {number}          transitionT    - 0..1 transition progress
 * @property {Object}          parts          - Named mesh references
 * @property {Object}          profile        - Color profile
 * @property {Record<string, THREE.CanvasTexture>} faceTextures - Pre-rendered face textures per state
 * @property {THREE.Sprite|null} statusSprite - Floating sprite ("!", "Z", etc.)
 * @property {THREE.Object3D[]} particles     - Success particles
 * @property {number}          particleAge    - Time since particles spawned
 */

/** @type {Map<string, CharacterRecord>} */
const _characters = new Map();

// ── Public API ──────────────────────

/**
 * Initialize the character system. Must be called before creating characters.
 * @param {THREE.Scene} scene - The Three.js scene to add characters to.
 */
export function initCharacters(scene) {
  _scene = scene;
  ensureGeometries();
}

/**
 * Create a character and add it to the scene.
 * Tries to load a rigged GLB model first, then un-rigged, then procedural.
 * Also fetches asset metadata for dynamic color profiles.
 *
 * @param {string} name     - Anima name (key in CHARACTER_PROFILES or dynamic).
 * @param {THREE.Vector3} position - World position to place the character.
 * @returns {Promise<THREE.Group>}   The character's root Group.
 */
export async function createCharacter(name, position) {
  if (!_scene) {
    throw new Error("character.js: initCharacters() must be called before createCharacter().");
  }

  if (_characters.has(name)) {
    removeCharacter(name);
  }

  if (!profileCache.has(name)) {
    try {
      const meta = await fetchAssetMetadata(name);
      if (meta?.colors?.image_color) {
        applyMetadataColor(name, meta.colors.image_color);
      }
    } catch { /* proceed without metadata colours */ }
  }

  // ── Try rigged GLB first (has skeleton + animations) ──────────
  const riggedUrl = await probeAsset(name, "avatar_chibi_rigged.glb");
  if (riggedUrl) {
    try {
      const { group, record } = await createGLBCharacter(name, position, riggedUrl, true);
      _scene.add(group);
      _characters.set(name, record);
      return group;
    } catch (err) {
      console.warn(`character.js: Rigged GLB failed for "${name}", trying un-rigged.`, err);
    }
  }

  // ── Try un-rigged GLB ──────────
  const glbUrl = await probeAsset(name, "avatar_chibi.glb");
  if (glbUrl) {
    try {
      const { group, record } = await createGLBCharacter(name, position, glbUrl, false);
      _scene.add(group);
      _characters.set(name, record);
      return group;
    } catch (err) {
      console.warn(`character.js: GLB load failed for "${name}", falling back to procedural.`, err);
    }
  }

  // ── Fallback: procedural SD character ──────────
  const { group, record } = createProceduralCharacter(name, position);
  _scene.add(group);
  _characters.set(name, record);
  return group;
}

/**
 * Remove a character from the scene and dispose its resources.
 * @param {string} name
 */
export function removeCharacter(name) {
  const rec = _characters.get(name);
  if (!rec) return;

  if (rec._mixer) {
    rec._mixer.stopAllAction();
    rec._mixer = null;
  }
  rec._animClips = null;
  rec._currentAction = null;

  hideStatusSprite(rec);
  clearParticles(rec);

  rec.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Sprite) {
      const mat = child.material;
      if (mat) {
        mat.dispose();
      }
    }
  });

  for (const tex of Object.values(rec.faceTextures)) {
    tex.dispose();
  }

  if (_scene) {
    _scene.remove(rec.group);
  }

  _characters.delete(name);
}

/**
 * Change a character's animation state with a smooth transition.
 * For GLB models with loaded animation clips, crossfades between actions.
 * @param {string} name  - Anima name.
 * @param {string} state - One of the STATES values.
 */
export function updateCharacterState(name, state) {
  const rec = _characters.get(name);
  if (!rec) return;

  if (!STATES.includes(/** @type {any} */ (state))) {
    console.warn(`character.js: unknown state "${state}", falling back to "idle".`);
    state = "idle";
  }

  if (rec.state === state) return;

  if (rec._isGLB && rec._animClips) {
    const nextAction = rec._animClips[state] || rec._animClips["idle"];
    const currentAction = rec._currentAction;

    if (nextAction && nextAction !== currentAction) {
      nextAction.reset();
      nextAction.play();
      if (currentAction) {
        currentAction.crossFadeTo(nextAction, TRANSITION_DURATION, true);
      }
      rec._currentAction = nextAction;
    }
  }

  if (!rec._isGLB) {
    cleanupState(rec, rec.state);
  }

  rec.prevState = rec.state;
  rec.state = state;
  rec.transitionT = 0;

  if (!rec._isGLB) {
    applyFaceTexture(rec);
  }
}

/**
 * Update all characters' animations. Call this every frame in the render loop.
 * @param {number} deltaTime   - Seconds since last frame.
 * @param {number} elapsedTime - Total elapsed seconds.
 */
export function updateAllCharacters(deltaTime, elapsedTime) {
  for (const rec of _characters.values()) {
    if (rec._isGLB) {
      if (rec._mixer) {
        rec._mixer.update(deltaTime);
      }
      if (!rec._animClips || Object.keys(rec._animClips).length === 0) {
        const baseY = rec.group.userData._baseY;
        rec.group.position.y = baseY + Math.sin(elapsedTime * 1.5) * 0.02;
        rec.group.rotation.y = Math.sin(elapsedTime * 0.3) * 0.1;
      }
      continue;
    }

    if (rec.transitionT < 1.0) {
      rec.transitionT = Math.min(1.0, rec.transitionT + deltaTime / TRANSITION_DURATION);
    }

    resetPose(rec);

    const animFn = animMap[rec.state];
    if (animFn) {
      animFn(rec, deltaTime, elapsedTime);
    }

    if (rec.transitionT < 1.0) {
      const t = easeInOutCubic(rec.transitionT);
      const baseY = rec.group.userData._baseY;
      const currentY = rec.group.position.y;
      rec.group.position.y = baseY + (currentY - baseY) * t;
    }
  }
}

/**
 * Return an array of all character meshes suitable for raycasting.
 * Each mesh has `userData.animaName` set.
 * @returns {THREE.Object3D[]}
 */
export function getCharacterMeshes() {
  /** @type {THREE.Object3D[]} */
  const meshes = [];
  for (const rec of _characters.values()) {
    rec.group.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshes.push(child);
      }
    });
  }
  return meshes;
}

/**
 * Given an array of raycaster intersections, return the anima name
 * of the first character hit, or null if no character was intersected.
 * @param {THREE.Intersection[]} intersects
 * @returns {string | null}
 */
export function getCharacterAtIntersection(intersects) {
  for (const hit of intersects) {
    const name = hit.object.userData.animaName;
    if (name && _characters.has(name)) {
      return name;
    }
  }
  return null;
}

/**
 * Dispose all characters and shared resources.
 * Call this when tearing down the 3D scene.
 */
export function disposeCharacters() {
  const names = [..._characters.keys()];
  for (const name of names) {
    removeCharacter(name);
  }

  for (const key of Object.keys(geo)) {
    const g = geo[/** @type {keyof typeof geo} */ (key)];
    if (g) {
      g.dispose();
      geo[/** @type {keyof typeof geo} */ (key)] = null;
    }
  }

  _scene = null;
}

/**
 * Get the character's Group directly (for movement control).
 * @param {string} name
 * @returns {THREE.Group | null}
 */
export function getCharacterGroup(name) {
  const rec = _characters.get(name);
  return rec ? rec.group : null;
}

/**
 * Get the character's home position.
 * @param {string} name
 * @returns {THREE.Vector3 | null}
 */
export function getCharacterHome(name) {
  const rec = _characters.get(name);
  if (!rec) return null;
  return new THREE.Vector3(
    rec.group.userData._baseX,
    rec.group.userData._baseY,
    rec.group.userData._baseZ,
  );
}

export { setAppearance } from "./character-appearance.js";
