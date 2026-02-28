// ── Character Loader Module ──────────────────────
// GLTF/GLB model loading, animation clip management, and procedural fallback.
// Returns { group, record } — scene registration is done by the orchestrator.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
import { probeAsset } from "./api.js";
import { modelCache } from "./model-cache.js";
import {
  generateProfile,
  profileCache,
  buildCharacterMesh,
  buildFaceTextures,
  applyFaceTexture,
} from "./character-appearance.js";

// ── GLTF Loader Setup ──────────────────────

/** Shared DRACOLoader for Draco-compressed GLBs. */
const _dracoLoader = new DRACOLoader();
_dracoLoader.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");
_dracoLoader.setDecoderConfig({ type: "wasm" });
_dracoLoader.preload();

/** Shared GLTFLoader instance with Draco support. */
const _gltfLoader = new GLTFLoader();
_gltfLoader.setDRACOLoader(_dracoLoader);

/** In-memory cache for parsed GLTF scenes (avoids re-parsing within session). */
const _parsedCache = new Map();

/**
 * Load and cache a GLTF/GLB model.
 * Uses IndexedDB for persistent cache, plus in-memory cache for session reuse.
 * @param {string} url
 * @returns {Promise<{scene: THREE.Group, animations: THREE.AnimationClip[]}>}
 */
async function _loadGLTFCached(url) {
  if (_parsedCache.has(url)) {
    const cached = _parsedCache.get(url);
    return { scene: SkeletonUtils.clone(cached.scene), animations: cached.animations };
  }
  const gltf = await modelCache.loadGLTF(url, _gltfLoader);
  _parsedCache.set(url, gltf);
  return { scene: SkeletonUtils.clone(gltf.scene), animations: gltf.animations };
}

// ── Constants ──────────────────────

/** Mapping from animation state to asset filename for GLB animation clips. */
export const STATE_ANIM_FILES = {
  idle:      "anim_idle.glb",
  working:   "anim_sitting.glb",
  thinking:  "anim_idle.glb",
  talking:   "anim_talking.glb",
  reporting: "anim_talking.glb",
  success:   "anim_waving.glb",
  sleeping:  "anim_sitting.glb",
  error:     "anim_idle.glb",
  walking:   "anim_walking.glb",
};

// ── GLB Character Creation ──────────────────────

/**
 * Load a GLB model and build a character record.
 * For rigged models, also loads separate animation GLB files and sets up
 * state-driven animation crossfading.
 *
 * Returns { group, record } — the caller is responsible for adding to scene
 * and registering in the characters map.
 *
 * @param {string} name
 * @param {THREE.Vector3} position
 * @param {string} url
 * @param {boolean} isRigged - Whether this is a rigged model with skeleton
 * @returns {Promise<{group: THREE.Group, record: Object}>}
 */
export async function createGLBCharacter(name, position, url, isRigged) {
  const gltf = await _loadGLTFCached(url);

  const group = new THREE.Group();
  group.name = `character_${name}`;

  const model = gltf.scene;

  // ── Load animations BEFORE computing bounds ──────────────────────
  // Bind-pose Hips bone rotations vary across characters (e.g. natsume's
  // skeleton is tilted ~50° at bind pose, giving half the Y-extent of
  // other characters).  By applying the idle animation frame 0 first,
  // every skeleton is in a comparable upright pose for consistent scaling.
  const mixer = new THREE.AnimationMixer(model);
  const animClips = {};  // state name -> THREE.AnimationAction
  if (isRigged) {
    const rawClips = await _loadAnimationClips(name);
    for (const [state, filename] of Object.entries(STATE_ANIM_FILES)) {
      const clip = rawClips[filename];
      if (clip) {
        animClips[state] = mixer.clipAction(clip);
        animClips[state].setLoop(THREE.LoopRepeat);
      }
    }
  }
  // Use embedded animations as fallback idle
  if (gltf.animations && gltf.animations.length > 0 && !animClips["idle"]) {
    animClips["idle"] = mixer.clipAction(gltf.animations[0]);
    animClips["idle"].setLoop(THREE.LoopRepeat);
  }

  // Apply idle animation frame 0 to put bones in the displayed pose
  const idleAction = animClips["idle"] || null;
  if (idleAction) {
    idleAction.play();
    mixer.setTime(0);
  }
  model.updateMatrixWorld(true);

  // ── Normalise scale: fit model into ~0.7 units tall ──────────────
  // For skinned (rigged) models, compute bounds from skeleton bone world
  // positions.  Bone.getWorldPosition() incorporates the armature
  // transform correctly, unlike Box3.setFromObject which misses the
  // armature scale on sibling SkinnedMesh nodes.
  const box = new THREE.Box3();
  let hasBones = false;
  const _tmpVec = new THREE.Vector3();
  model.traverse((child) => {
    if (child.isSkinnedMesh && child.skeleton && child.skeleton.bones.length > 0) {
      for (const bone of child.skeleton.bones) {
        box.expandByPoint(bone.getWorldPosition(_tmpVec));
      }
      hasBones = true;
    }
  });
  if (!hasBones) {
    box.setFromObject(model);
  }

  const height = box.max.y - box.min.y;
  const targetHeight = 0.7;
  const maxHeight = 0.8;  // sanity cap to prevent oversized characters
  const rawScale = height > 0 ? targetHeight / height : 1;
  const scale = (height > 0 && height * rawScale > maxHeight)
    ? maxHeight / height
    : rawScale;
  model.scale.setScalar(scale);

  // Center the model horizontally & place feet at y=0.
  // After applying scale we must recompute the bounds using the same
  // method (bone-based for skinned models) so that centering is correct.
  model.updateMatrixWorld(true);
  const scaledBox = new THREE.Box3();
  if (hasBones) {
    model.traverse((child) => {
      if (child.isSkinnedMesh && child.skeleton && child.skeleton.bones.length > 0) {
        for (const bone of child.skeleton.bones) {
          scaledBox.expandByPoint(bone.getWorldPosition(_tmpVec));
        }
      }
    });
  } else {
    scaledBox.setFromObject(model);
  }
  model.position.x -= (scaledBox.min.x + scaledBox.max.x) / 2;
  model.position.y -= scaledBox.min.y;

  group.add(model);

  // Tag for raycasting
  group.traverse((child) => { child.userData.animaName = name; });

  group.position.copy(position);
  group.userData._baseX = position.x;
  group.userData._baseY = position.y;
  group.userData._baseZ = position.z;

  const profile = profileCache.get(name) || generateProfile(name);

  const record = {
    name,
    group,
    state: "idle",
    prevState: "idle",
    transitionT: 1.0,
    parts: {},
    profile,
    faceTextures: {},
    statusSprite: null,
    particles: [],
    particleAge: 0,
    _isGLB: true,
    _mixer: mixer,
    _animClips: animClips,
    _currentAction: idleAction,
  };

  return { group, record };
}

/**
 * Load animation clips from separate GLB files for an anima.
 * Returns a map of filename -> THREE.AnimationClip.
 * @param {string} name
 * @returns {Promise<Record<string, THREE.AnimationClip>>}
 */
async function _loadAnimationClips(name) {
  /** @type {Record<string, THREE.AnimationClip>} */
  const clips = {};
  const filenames = [...new Set(Object.values(STATE_ANIM_FILES))];

  const promises = filenames.map(async (filename) => {
    const url = await probeAsset(name, filename);
    if (!url) return;

    try {
      const gltf = await _loadGLTFCached(url);
      if (gltf.animations && gltf.animations.length > 0) {
        clips[filename] = gltf.animations[0];
      }
    } catch (err) {
      console.warn(`character-loader.js: Failed to load animation "${filename}" for "${name}":`, err);
    }
  });

  await Promise.all(promises);
  return clips;
}

// ── Procedural Character Creation ──────────────────────

/**
 * Create a procedural SD character (original implementation).
 * Returns { group, record } — the caller is responsible for adding to scene
 * and registering in the characters map.
 *
 * @param {string} name
 * @param {THREE.Vector3} position
 * @returns {{group: THREE.Group, record: Object}}
 */
export function createProceduralCharacter(name, position) {
  const profile = profileCache.get(name) || generateProfile(name);
  const { group, parts, faceTextures } = buildCharacterMesh(name, profile);

  group.position.copy(position);

  group.userData._baseX = position.x;
  group.userData._baseY = position.y;
  group.userData._baseZ = position.z;

  const record = {
    name,
    group,
    state: "idle",
    prevState: "idle",
    transitionT: 1.0,
    parts,
    profile,
    faceTextures,
    statusSprite: null,
    particles: [],
    particleAge: 0,
  };

  applyFaceTexture(record);

  return { group, record };
}

// ── Metadata Color ──────────────────────

/**
 * Apply an image_color hex string from asset metadata as the key colour
 * for a dynamically generated profile.
 * @param {string} name
 * @param {string} hexStr - e.g. "#FFB7C5"
 */
export function applyMetadataColor(name, hexStr) {
  const hex = parseInt(hexStr.replace("#", ""), 16);
  if (isNaN(hex)) return;
  const profile = generateProfile(name);
  profile.hairColor = hex;
  profileCache.set(name, profile);
}
