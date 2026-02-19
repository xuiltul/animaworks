// ── Character Module ──────────────────────
// Procedural SD (super-deformed) character generation and animation.
// Creates 2-3 head-proportion characters from primitive geometry
// and manages state-driven animation transitions.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
import { probeAsset, fetchAssetMetadata } from "./api.js";
import { modelCache } from "./model-cache.js";

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

/** @type {Map<string, {hairColor: number, eyeColor: number, bodyColor: number}>} */
const _profileCache = new Map();

/**
 * Register appearance data for an anima from the server API.
 * Call before createCharacter() with data from /api/animas.
 * @param {string} name
 * @param {{hairColor?: string, eyeColor?: string, bodyColor?: string}} appearance
 */
export function setAppearance(name, appearance) {
  if (!appearance || !name) return;
  const base = _generateProfile(name);
  if (appearance.hairColor) base.hairColor = parseInt(appearance.hairColor.replace("#", ""), 16);
  if (appearance.eyeColor) base.eyeColor = parseInt(appearance.eyeColor.replace("#", ""), 16);
  if (appearance.bodyColor) base.bodyColor = parseInt(appearance.bodyColor.replace("#", ""), 16);
  _profileCache.set(name, base);
}

/** All valid animation states. */
const STATES = /** @type {const} */ ([
  "idle", "working", "thinking", "error",
  "sleeping", "talking", "reporting", "success", "walking",
]);

/** Duration in seconds for lerp-based state transitions. */
const TRANSITION_DURATION = 0.4;

/** Maximum success-burst particles per character. */
const MAX_PARTICLES = 10;

/** Canvas size for face textures. */
const FACE_TEX_SIZE = 64;

/** Mapping from animation state to asset filename for GLB animation clips. */
const _STATE_ANIM_FILES = {
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

/** Shared geometries — created once, reused for every character. */
const _geo = {
  /** @type {THREE.SphereGeometry | null} */   head: null,
  /** @type {THREE.SphereGeometry | null} */   hair: null,
  /** @type {THREE.PlaneGeometry | null} */    face: null,
  /** @type {THREE.CylinderGeometry | null} */ body: null,
  /** @type {THREE.CylinderGeometry | null} */ arm: null,
  /** @type {THREE.CylinderGeometry | null} */ leg: null,
  /** @type {THREE.SphereGeometry | null} */   particle: null,
};

// ── Geometry Factory ──────────────────────

/** Create shared geometries once. */
function _ensureGeometries() {
  if (_geo.head) return;
  _geo.head     = new THREE.SphereGeometry(0.2, 16, 12);
  _geo.hair     = new THREE.SphereGeometry(0.22, 16, 12);
  _geo.face     = new THREE.PlaneGeometry(0.26, 0.26);
  _geo.body     = new THREE.CylinderGeometry(0.12, 0.10, 0.25, 12);
  _geo.arm      = new THREE.CylinderGeometry(0.04, 0.04, 0.15, 8);
  _geo.leg      = new THREE.CylinderGeometry(0.05, 0.05, 0.12, 8);
  _geo.particle = new THREE.SphereGeometry(0.02, 6, 4);
}

// ── Face Texture Generation ──────────────────────

/**
 * Hex color number to CSS string.
 * @param {number} c
 * @returns {string}
 */
function _hexToCSS(c) {
  return "#" + c.toString(16).padStart(6, "0");
}

/**
 * Draw a face expression onto a 64x64 canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} expression - One of STATES
 * @param {number} eyeColor   - Hex color
 */
function _drawFace(ctx, expression, eyeColor) {
  const s = FACE_TEX_SIZE;
  ctx.clearRect(0, 0, s, s);

  const eyeCSS = _hexToCSS(eyeColor);
  const cx = s / 2;

  switch (expression) {
    case "idle": {
      // Normal eyes — two circles
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      // Slight smile
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 38, 5, 0.1 * Math.PI, 0.9 * Math.PI);
      ctx.stroke();
      break;
    }
    case "working": {
      // Focused — slightly narrowed eyes (horizontal ellipses)
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.ellipse(cx - 8, 26, 5, 2.5, 0, 0, Math.PI * 2);
      ctx.ellipse(cx + 8, 26, 5, 2.5, 0, 0, Math.PI * 2);
      ctx.fill();
      // Straight mouth
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 4, 38);
      ctx.lineTo(cx + 4, 38);
      ctx.stroke();
      break;
    }
    case "thinking": {
      // One eye larger (curious)
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx + 8, 25, 5, 0, Math.PI * 2);
      ctx.fill();
      // Small "o" mouth
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 38, 3, 0, Math.PI * 2);
      ctx.stroke();
      break;
    }
    case "error": {
      // Wide eyes
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 25, 5, 0, Math.PI * 2);
      ctx.arc(cx + 8, 25, 5, 0, Math.PI * 2);
      ctx.fill();
      // White highlights
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(cx - 6, 23, 1.5, 0, Math.PI * 2);
      ctx.arc(cx + 10, 23, 1.5, 0, Math.PI * 2);
      ctx.fill();
      // Open mouth
      ctx.fillStyle = "#553333";
      ctx.beginPath();
      ctx.ellipse(cx, 39, 4, 3, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "sleeping": {
      // Closed eyes — two horizontal lines
      ctx.strokeStyle = eyeCSS;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - 12, 26);
      ctx.lineTo(cx - 4, 26);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx + 4, 26);
      ctx.lineTo(cx + 12, 26);
      ctx.stroke();
      // No mouth
      break;
    }
    case "talking": {
      // Normal eyes
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      // Open mouth (ellipse)
      ctx.fillStyle = "#553333";
      ctx.beginPath();
      ctx.ellipse(cx, 39, 4, 3, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "reporting": {
      // Normal eyes, slight smile (same as idle but friendlier)
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      // Wider smile
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 37, 6, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
      break;
    }
    case "success": {
      // Happy eyes — arcs (upside-down U)
      ctx.strokeStyle = eyeCSS;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx - 8, 28, 4, Math.PI, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx + 8, 28, 4, Math.PI, 2 * Math.PI);
      ctx.stroke();
      // Wide smile
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, 37, 7, 0.1 * Math.PI, 0.9 * Math.PI);
      ctx.stroke();
      break;
    }
    case "walking": {
      // Same as idle (light smile)
      _drawFace(ctx, "idle", eyeColor);
      break;
    }
    default: {
      // Fallback — same as idle
      _drawFace(ctx, "idle", eyeColor);
      break;
    }
  }
}

/**
 * Pre-render face textures for every state.
 * @param {number} eyeColor
 * @returns {Record<string, THREE.CanvasTexture>}
 */
function _buildFaceTextures(eyeColor) {
  /** @type {Record<string, THREE.CanvasTexture>} */
  const textures = {};
  for (const state of STATES) {
    const canvas = document.createElement("canvas");
    canvas.width = FACE_TEX_SIZE;
    canvas.height = FACE_TEX_SIZE;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      _drawFace(ctx, state, eyeColor);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    textures[state] = tex;
  }
  return textures;
}

// ── Sprite Factory ──────────────────────

/**
 * Create a text sprite (for "!" or "Z" indicators).
 * @param {string} text
 * @param {string} color - CSS color
 * @returns {THREE.Sprite}
 */
function _createTextSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.font = "bold 48px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.fillText(text, 32, 32);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.15, 0.15, 1);
  return sprite;
}

// ── Random Pastel for Dynamic Animas ──────────────────────

/**
 * Generate a deterministic random pastel profile from a name string.
 * Uses a simple hash so the same name always produces the same colors.
 * @param {string} name
 * @returns {{hairColor: number, eyeColor: number, bodyColor: number, role: string}}
 */
function _generateProfile(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  }
  const abs = Math.abs(hash);

  // Hair: medium saturation
  const hH = abs % 360;
  const hairColor = _hslToHex(hH, 40, 30);

  // Eyes: vivid
  const hE = (abs * 137) % 360;
  const eyeColor = _hslToHex(hE, 60, 50);

  // Body: light pastel skin tone
  const hB = (abs * 53) % 60 + 15; // warm hue range
  const bodyColor = _hslToHex(hB, 40, 92);

  return { hairColor, eyeColor, bodyColor, role: "worker" };
}

/**
 * Convert HSL values to a hex number.
 * @param {number} h - 0-360
 * @param {number} s - 0-100
 * @param {number} l - 0-100
 * @returns {number}
 */
function _hslToHex(h, s, l) {
  s /= 100;
  l /= 100;
  const a = s * Math.min(l, 1 - l);
  /** @param {number} n */
  const f = (n) => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color);
  };
  return (f(0) << 16) | (f(8) << 8) | f(4);
}

// ── Character Construction ──────────────────────

/**
 * Build the procedural SD character mesh group as a faceless silhouette.
 * No eyes, no expressions — just a minimal body shape placeholder.
 * Total height ~0.7 units. Origin at the character's feet.
 * @param {string} name
 * @param {{hairColor: number, eyeColor: number, bodyColor: number, role: string}} profile
 * @returns {{group: THREE.Group, parts: Object, faceTextures: Record<string, THREE.CanvasTexture>}}
 */
function _buildCharacterMesh(name, profile) {
  _ensureGeometries();

  const group = new THREE.Group();
  group.name = `character_${name}`;

  // Single semi-transparent silhouette colour derived from hair colour
  const silhouetteColor = profile.hairColor;
  const silhouetteMat = new THREE.MeshLambertMaterial({
    color: silhouetteColor,
    transparent: true,
    opacity: 0.55,
  });

  // ── Measurements ──────────
  const legH   = 0.12;
  const bodyH  = 0.25;
  const headR  = 0.2;
  const headY  = legH + bodyH + headR;   // ~0.57
  const bodyY  = legH + bodyH / 2;       // ~0.245

  // ── Legs ──────────
  const legL = new THREE.Mesh(_geo.leg, silhouetteMat);
  legL.position.set(-0.06, legH / 2, 0);
  legL.name = "legL";
  group.add(legL);

  const legR = new THREE.Mesh(_geo.leg, silhouetteMat);
  legR.position.set(0.06, legH / 2, 0);
  legR.name = "legR";
  group.add(legR);

  // ── Body ──────────
  const body = new THREE.Mesh(_geo.body, silhouetteMat);
  body.position.set(0, bodyY, 0);
  body.name = "body";
  group.add(body);

  // ── Arms ──────────
  const armL = new THREE.Mesh(_geo.arm, silhouetteMat);
  armL.position.set(-0.17, bodyY + 0.02, 0);
  armL.rotation.z = 0.15;
  armL.name = "armL";
  group.add(armL);

  const armR = new THREE.Mesh(_geo.arm, silhouetteMat);
  armR.position.set(0.17, bodyY + 0.02, 0);
  armR.rotation.z = -0.15;
  armR.name = "armR";
  group.add(armR);

  // ── Hair (behind head) ──────────
  const hair = new THREE.Mesh(_geo.hair, silhouetteMat);
  hair.position.set(0, headY, -0.04);
  hair.name = "hair";
  group.add(hair);

  // ── Head (no face / no eyes) ──────────
  const head = new THREE.Mesh(_geo.head, silhouetteMat);
  head.position.set(0, headY, 0);
  head.name = "head";
  group.add(head);

  // ── userData for raycasting ──────────
  group.traverse((child) => {
    child.userData.animaName = name;
  });

  // face is null — no expression textures for silhouette models
  const parts = { head, hair, face: null, body, armL, armR, legL, legR };
  return { group, parts, faceTextures: {} };
}

// ── Animation Functions ──────────────────────
// Each takes (record, dt, elapsed) and mutates the group's transforms.
// They expect parts to be at their "rest" positions and apply offsets.

/**
 * Reset all animated transforms to rest pose.
 * @param {CharacterRecord} rec
 */
function _resetPose(rec) {
  const { parts, group } = rec;
  const legH  = 0.12;
  const bodyH = 0.25;
  const headR = 0.2;
  const headY = legH + bodyH + headR;
  const bodyY = legH + bodyH / 2;

  group.position.y = rec.group.userData._baseY ?? group.position.y;

  parts.head.position.set(0, headY, 0);
  parts.head.rotation.set(0, 0, 0);
  parts.hair.position.set(0, headY, -0.04);
  parts.hair.rotation.set(0, 0, 0);
  if (parts.face) {
    parts.face.position.set(0, headY, headR + 0.001);
    parts.face.rotation.set(0, 0, 0);
  }
  parts.body.position.set(0, bodyY, 0);
  parts.body.rotation.set(0, 0, 0);
  parts.armL.position.set(-0.17, bodyY + 0.02, 0);
  parts.armL.rotation.set(0, 0, 0.15);
  parts.armR.position.set(0.17, bodyY + 0.02, 0);
  parts.armR.rotation.set(0, 0, -0.15);
  parts.legL.rotation.set(0, 0, 0);
  parts.legR.rotation.set(0, 0, 0);
}

/**
 * Idle: gentle bobbing + slow rotation.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animIdle(rec, _dt, elapsed) {
  const bob = Math.sin(elapsed * 1.5) * 0.02;
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + bob;
  rec.group.rotation.y = Math.sin(elapsed * 0.3) * 0.1;
}

/**
 * Working: slight forward lean, arms move (typing gesture).
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animWorking(rec, _dt, elapsed) {
  const { parts } = rec;

  // Slight forward lean
  parts.body.rotation.x = 0.12;
  parts.head.rotation.x = 0.08;
  parts.hair.rotation.x = 0.08;
  if (parts.face) parts.face.rotation.x = 0.08;

  // Typing gesture — arms move up/down alternately
  const cycle = elapsed * 6;
  parts.armL.rotation.x = Math.sin(cycle) * 0.25;
  parts.armR.rotation.x = Math.sin(cycle + Math.PI) * 0.25;

  // Subtle bob
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 3) * 0.005;
}

/**
 * Thinking: head tilt, one arm to chin.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animThinking(rec, _dt, elapsed) {
  const { parts } = rec;

  // Head tilt to the side
  parts.head.rotation.z = 0.2;
  parts.hair.rotation.z = 0.2;
  if (parts.face) parts.face.rotation.z = 0.2;

  // Right arm to chin
  parts.armR.rotation.x = -0.6;
  parts.armR.rotation.z = -0.4;

  // Gentle sway
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 1.2) * 0.01;
}

/**
 * Error: red flash + exclamation sprite above head.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animError(rec, _dt, elapsed) {
  const { parts } = rec;

  // Shake
  rec.group.position.x = (rec.group.userData._baseX ?? 0) + Math.sin(elapsed * 15) * 0.01;

  // Red flash on body — oscillate emissive
  const flash = (Math.sin(elapsed * 4) + 1) * 0.5;
  const bodyMat = /** @type {THREE.MeshLambertMaterial} */ (parts.body.material);
  bodyMat.emissive.setRGB(flash * 0.6, 0, 0);

  // Show exclamation sprite
  _showStatusSprite(rec, "!", "#ff3333", elapsed);
}

/**
 * Sleeping: body tilts forward, "Z" above head, gentle breathing.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animSleeping(rec, _dt, elapsed) {
  const { parts } = rec;

  // Forward tilt
  parts.body.rotation.x = 0.3;
  parts.head.rotation.x = 0.25;
  parts.hair.rotation.x = 0.25;
  if (parts.face) parts.face.rotation.x = 0.25;

  // Breathing: gentle scale oscillation on body
  const breath = 1 + Math.sin(elapsed * 1.8) * 0.03;
  parts.body.scale.set(1, breath, 1);

  // Z sprite floating upward
  _showStatusSprite(rec, "Z", "#6688cc", elapsed);
  if (rec.statusSprite) {
    rec.statusSprite.position.y = 0.8 + Math.sin(elapsed * 0.8) * 0.05;
    rec.statusSprite.material.opacity = 0.6 + Math.sin(elapsed * 1.5) * 0.3;
  }
}

/**
 * Talking: slight bounce + arm gestures.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animTalking(rec, _dt, elapsed) {
  const { parts } = rec;

  // Bounce
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(elapsed * 4)) * 0.02;

  // Arm gestures
  parts.armL.rotation.z = 0.15 + Math.sin(elapsed * 3) * 0.3;
  parts.armR.rotation.z = -0.15 + Math.sin(elapsed * 3 + 1) * 0.3;

  // Slight head nod
  parts.head.rotation.x = Math.sin(elapsed * 2.5) * 0.08;
  parts.hair.rotation.x = parts.head.rotation.x;
  if (parts.face) parts.face.rotation.x = parts.head.rotation.x;
}

/**
 * Reporting: turn to face supervisor desk (rotate toward origin or a target).
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animReporting(rec, _dt, elapsed) {
  // Rotate toward world origin (approximate supervisor position)
  const targetAngle = Math.atan2(
    -rec.group.position.x,
    -rec.group.position.z,
  );
  rec.group.rotation.y = targetAngle;

  // Subtle idle bob while facing supervisor
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 1.5) * 0.01;

  // Slight arm raise (presenting gesture)
  const { parts } = rec;
  parts.armR.rotation.z = -0.15 + Math.sin(elapsed * 2) * 0.15;
}

/**
 * Success: small jump + particle burst.
 * @param {CharacterRecord} rec
 * @param {number} dt
 * @param {number} elapsed
 */
function _animSuccess(rec, dt, elapsed) {
  const { parts } = rec;

  // Jump arc
  const jumpT = (elapsed * 2) % (Math.PI * 2);
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(jumpT)) * 0.08;

  // Happy arm raise
  parts.armL.rotation.z = 0.8;
  parts.armR.rotation.z = -0.8;

  // Spawn / animate particles
  _updateSuccessParticles(rec, dt);
}

/**
 * Walking: leg swing, arm swing, body bob and sway.
 * @param {CharacterRecord} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animWalking(rec, _dt, elapsed) {
  const { parts } = rec;

  // Leg swing (alternating)
  const legCycle = elapsed * 8;
  parts.legL.rotation.x = Math.sin(legCycle) * 0.4;
  parts.legR.rotation.x = Math.sin(legCycle + Math.PI) * 0.4;

  // Arm counter-swing
  parts.armL.rotation.x = Math.sin(legCycle + Math.PI) * 0.3;
  parts.armR.rotation.x = Math.sin(legCycle) * 0.3;

  // Subtle body bob
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(legCycle * 2)) * 0.015;

  // Subtle body sway
  parts.body.rotation.z = Math.sin(legCycle) * 0.03;
}

// ── Animation Dispatch ──────────────────────

/** @type {Record<string, (rec: CharacterRecord, dt: number, elapsed: number) => void>} */
const _animMap = {
  idle:      _animIdle,
  working:   _animWorking,
  thinking:  _animThinking,
  error:     _animError,
  sleeping:  _animSleeping,
  talking:   _animTalking,
  reporting: _animReporting,
  success:   _animSuccess,
  walking:   _animWalking,
};

// ── Status Sprite Helpers ──────────────────────

/**
 * Ensure a floating status sprite is visible and positioned above the head.
 * @param {CharacterRecord} rec
 * @param {string} text
 * @param {string} color
 * @param {number} elapsed
 */
function _showStatusSprite(rec, text, color, elapsed) {
  if (!rec.statusSprite) {
    rec.statusSprite = _createTextSprite(text, color);
    rec.group.add(rec.statusSprite);
  }
  rec.statusSprite.visible = true;
  rec.statusSprite.position.set(0, 0.85 + Math.sin(elapsed * 2) * 0.03, 0);
}

/**
 * Hide and remove the floating status sprite.
 * @param {CharacterRecord} rec
 */
function _hideStatusSprite(rec) {
  if (rec.statusSprite) {
    rec.statusSprite.visible = false;
    rec.group.remove(rec.statusSprite);
    rec.statusSprite.material.map?.dispose();
    rec.statusSprite.material.dispose();
    rec.statusSprite = null;
  }
}

// ── Success Particles ──────────────────────

/**
 * Spawn success particles if not yet spawned.
 * @param {CharacterRecord} rec
 */
function _spawnSuccessParticles(rec) {
  if (rec.particles.length > 0) return;
  rec.particleAge = 0;

  for (let i = 0; i < MAX_PARTICLES; i++) {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
      transparent: true,
      opacity: 1.0,
    });
    const mesh = new THREE.Mesh(_geo.particle, mat);
    mesh.position.set(0, 0.6, 0);
    // Store velocity in userData
    mesh.userData.vel = new THREE.Vector3(
      (Math.random() - 0.5) * 0.8,
      Math.random() * 1.2 + 0.5,
      (Math.random() - 0.5) * 0.8,
    );
    rec.group.add(mesh);
    rec.particles.push(mesh);
  }
}

/**
 * Update success particles (gravity + fade).
 * @param {CharacterRecord} rec
 * @param {number} dt
 */
function _updateSuccessParticles(rec, dt) {
  _spawnSuccessParticles(rec);
  rec.particleAge += dt;

  for (const p of rec.particles) {
    const vel = /** @type {THREE.Vector3} */ (p.userData.vel);
    vel.y -= 2.0 * dt; // gravity
    p.position.addScaledVector(vel, dt);

    const mat = /** @type {THREE.MeshBasicMaterial} */ (p.material);
    mat.opacity = Math.max(0, 1 - rec.particleAge * 0.8);
  }

  // Remove after 1.5 seconds
  if (rec.particleAge > 1.5) {
    _clearParticles(rec);
  }
}

/**
 * Remove all particles from a character.
 * @param {CharacterRecord} rec
 */
function _clearParticles(rec) {
  for (const p of rec.particles) {
    rec.group.remove(p);
    /** @type {THREE.MeshBasicMaterial} */ (p.material).dispose();
  }
  rec.particles = [];
  rec.particleAge = 0;
}

// ── Face Texture Switching ──────────────────────

/**
 * Switch the face texture to match the current state.
 * @param {CharacterRecord} rec
 */
function _applyFaceTexture(rec) {
  if (!rec.parts.face) return;  // silhouette model — no face
  const tex = rec.faceTextures[rec.state] || rec.faceTextures["idle"];
  if (!tex) return;
  const faceMat = /** @type {THREE.MeshBasicMaterial} */ (rec.parts.face.material);
  if (faceMat.map !== tex) {
    faceMat.map = tex;
    faceMat.needsUpdate = true;
  }
}

// ── State Transition Cleanup ──────────────────────

/**
 * Clean up visuals that are state-specific (sprites, particles, emissive).
 * Called when leaving a state.
 * @param {CharacterRecord} rec
 * @param {string} oldState
 */
function _cleanupState(rec, oldState) {
  if (oldState === "error" || oldState === "sleeping") {
    _hideStatusSprite(rec);
  }
  if (oldState === "error") {
    // Reset emissive
    const bodyMat = /** @type {THREE.MeshLambertMaterial} */ (rec.parts.body.material);
    bodyMat.emissive.setRGB(0, 0, 0);
    // Reset shake position
    rec.group.position.x = rec.group.userData._baseX ?? 0;
  }
  if (oldState === "success") {
    _clearParticles(rec);
  }
  // Reset body scale (sleeping breathing)
  rec.parts.body.scale.set(1, 1, 1);
}

// ── Public API ──────────────────────

/**
 * Initialize the character system. Must be called before creating characters.
 * @param {THREE.Scene} scene - The Three.js scene to add characters to.
 */
export function initCharacters(scene) {
  _scene = scene;
  _ensureGeometries();
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

  // Remove existing character with the same name
  if (_characters.has(name)) {
    removeCharacter(name);
  }

  // Fetch metadata to update dynamic color profile for non-hardcoded animas
  if (!_profileCache.has(name)) {
    try {
      const meta = await fetchAssetMetadata(name);
      if (meta?.colors?.image_color) {
        _applyMetadataColor(name, meta.colors.image_color);
      }
    } catch { /* proceed without metadata colours */ }
  }

  // ── Try rigged GLB first (has skeleton + animations) ──────────
  const riggedUrl = await probeAsset(name, "avatar_chibi_rigged.glb");
  if (riggedUrl) {
    try {
      return await _createGLBCharacter(name, position, riggedUrl, true);
    } catch (err) {
      console.warn(`character.js: Rigged GLB failed for "${name}", trying un-rigged.`, err);
    }
  }

  // ── Try un-rigged GLB ──────────
  const glbUrl = await probeAsset(name, "avatar_chibi.glb");
  if (glbUrl) {
    try {
      return await _createGLBCharacter(name, position, glbUrl, false);
    } catch (err) {
      console.warn(`character.js: GLB load failed for "${name}", falling back to procedural.`, err);
    }
  }

  // ── Fallback: procedural SD character ──────────
  return _createProceduralCharacter(name, position);
}

/**
 * Apply an image_color hex string from asset metadata as the key colour
 * for a dynamically generated profile.
 * @param {string} name
 * @param {string} hexStr - e.g. "#FFB7C5"
 */
function _applyMetadataColor(name, hexStr) {
  const hex = parseInt(hexStr.replace("#", ""), 16);
  if (isNaN(hex)) return;
  const profile = _generateProfile(name);
  profile.hairColor = hex;
  _profileCache.set(name, profile);
}

/**
 * Load a GLB model and register it as a character.
 * For rigged models, also loads separate animation GLB files and sets up
 * state-driven animation crossfading.
 *
 * @param {string} name
 * @param {THREE.Vector3} position
 * @param {string} url
 * @param {boolean} isRigged - Whether this is a rigged model with skeleton
 * @returns {Promise<THREE.Group>}
 */
async function _createGLBCharacter(name, position, url, isRigged) {
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
    for (const [state, filename] of Object.entries(_STATE_ANIM_FILES)) {
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

  _scene.add(group);

  const profile = _profileCache.get(name) || _generateProfile(name);

  /** @type {CharacterRecord} */
  const record = {
    name,
    group,
    state: "idle",
    prevState: "idle",
    transitionT: 1.0,
    parts: {},          // no procedural parts
    profile,
    faceTextures: {},   // no face textures
    statusSprite: null,
    particles: [],
    particleAge: 0,
    _isGLB: true,
    _mixer: mixer,
    _animClips: animClips,
    _currentAction: idleAction,
  };

  _characters.set(name, record);
  return group;
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
  const filenames = [...new Set(Object.values(_STATE_ANIM_FILES))];

  const promises = filenames.map(async (filename) => {
    const url = await probeAsset(name, filename);
    if (!url) return;

    try {
      const gltf = await _loadGLTFCached(url);
      if (gltf.animations && gltf.animations.length > 0) {
        clips[filename] = gltf.animations[0];
      }
    } catch (err) {
      console.warn(`character.js: Failed to load animation "${filename}" for "${name}":`, err);
    }
  });

  await Promise.all(promises);
  return clips;
}

/**
 * Create a procedural SD character (original implementation).
 * @param {string} name
 * @param {THREE.Vector3} position
 * @returns {THREE.Group}
 */
function _createProceduralCharacter(name, position) {
  const profile = _profileCache.get(name) || _generateProfile(name);
  const { group, parts, faceTextures } = _buildCharacterMesh(name, profile);

  group.position.copy(position);

  // Store base position for animation offsets
  group.userData._baseX = position.x;
  group.userData._baseY = position.y;
  group.userData._baseZ = position.z;

  _scene.add(group);

  /** @type {CharacterRecord} */
  const record = {
    name,
    group,
    state: "idle",
    prevState: "idle",
    transitionT: 1.0, // fully transitioned
    parts,
    profile,
    faceTextures,
    statusSprite: null,
    particles: [],
    particleAge: 0,
  };

  _characters.set(name, record);
  _applyFaceTexture(record);

  return group;
}

/**
 * Remove a character from the scene and dispose its resources.
 * @param {string} name
 */
export function removeCharacter(name) {
  const rec = _characters.get(name);
  if (!rec) return;

  // Stop animation mixer
  if (rec._mixer) {
    rec._mixer.stopAllAction();
    rec._mixer = null;
  }
  rec._animClips = null;
  rec._currentAction = null;

  // Clean up state-specific visuals
  _hideStatusSprite(rec);
  _clearParticles(rec);

  // Dispose materials (but NOT shared geometries)
  rec.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Sprite) {
      const mat = child.material;
      if (mat) {
        if (/** @type {THREE.MeshBasicMaterial} */ (mat).map) {
          // Only dispose textures we own (face textures, sprite textures)
          // Shared geometry textures are not disposed here
        }
        mat.dispose();
      }
    }
  });

  // Dispose all face textures
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

  // Validate state
  if (!STATES.includes(/** @type {any} */ (state))) {
    console.warn(`character.js: unknown state "${state}", falling back to "idle".`);
    state = "idle";
  }

  if (rec.state === state) return;

  // GLB model: crossfade animation clips
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

  // Procedural model: clean up old state visuals
  if (!rec._isGLB) {
    _cleanupState(rec, rec.state);
  }

  rec.prevState = rec.state;
  rec.state = state;
  rec.transitionT = 0;

  if (!rec._isGLB) {
    _applyFaceTexture(rec);
  }
}

/**
 * Update all characters' animations. Call this every frame in the render loop.
 * @param {number} deltaTime   - Seconds since last frame.
 * @param {number} elapsedTime - Total elapsed seconds.
 */
export function updateAllCharacters(deltaTime, elapsedTime) {
  for (const rec of _characters.values()) {
    // GLB model: update AnimationMixer
    if (rec._isGLB) {
      if (rec._mixer) {
        rec._mixer.update(deltaTime);
      }
      // Only apply idle bob when no animation clips are loaded (un-rigged GLB)
      if (!rec._animClips || Object.keys(rec._animClips).length === 0) {
        const baseY = rec.group.userData._baseY;
        rec.group.position.y = baseY + Math.sin(elapsedTime * 1.5) * 0.02;
        rec.group.rotation.y = Math.sin(elapsedTime * 0.3) * 0.1;
      }
      continue;
    }

    // Procedural model: full animation system
    // Advance transition
    if (rec.transitionT < 1.0) {
      rec.transitionT = Math.min(1.0, rec.transitionT + deltaTime / TRANSITION_DURATION);
    }

    // Reset to rest pose before applying animation
    _resetPose(rec);

    // Apply current state animation
    const animFn = _animMap[rec.state];
    if (animFn) {
      animFn(rec, deltaTime, elapsedTime);
    }

    // During transition, lerp between rest and animated pose
    if (rec.transitionT < 1.0) {
      const t = _easeInOutCubic(rec.transitionT);
      const baseY = rec.group.userData._baseY;
      const currentY = rec.group.position.y;
      rec.group.position.y = baseY + (currentY - baseY) * t;
    }
  }
}

/**
 * Easing function for smooth transitions.
 * @param {number} t - 0..1
 * @returns {number}
 */
function _easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
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
  // Remove all characters
  const names = [..._characters.keys()];
  for (const name of names) {
    removeCharacter(name);
  }

  // Dispose shared geometries
  for (const key of Object.keys(_geo)) {
    const g = _geo[/** @type {keyof typeof _geo} */ (key)];
    if (g) {
      g.dispose();
      _geo[/** @type {keyof typeof _geo} */ (key)] = null;
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
