// ── Character Appearance Module ──────────────────────
// Face textures, color profiles, shared geometries and mesh construction.

import * as THREE from "three";

// ── Constants ──────────────────────

/** All valid animation states. */
export const STATES = /** @type {const} */ ([
  "idle", "working", "thinking", "error",
  "sleeping", "talking", "reporting", "success", "walking",
]);

/** Canvas size for face textures. */
export const FACE_TEX_SIZE = 64;

// ── Profile Cache ──────────────────────

/** @type {Map<string, {hairColor: number, eyeColor: number, bodyColor: number}>} */
export const profileCache = new Map();

/**
 * Register appearance data for an anima from the server API.
 * Call before createCharacter() with data from /api/animas.
 * @param {string} name
 * @param {{hairColor?: string, eyeColor?: string, bodyColor?: string}} appearance
 */
export function setAppearance(name, appearance) {
  if (!appearance || !name) return;
  const base = generateProfile(name);
  if (appearance.hairColor) base.hairColor = parseInt(appearance.hairColor.replace("#", ""), 16);
  if (appearance.eyeColor) base.eyeColor = parseInt(appearance.eyeColor.replace("#", ""), 16);
  if (appearance.bodyColor) base.bodyColor = parseInt(appearance.bodyColor.replace("#", ""), 16);
  profileCache.set(name, base);
}

// ── Random Pastel for Dynamic Animas ──────────────────────

/**
 * Generate a deterministic random pastel profile from a name string.
 * Uses a simple hash so the same name always produces the same colors.
 * @param {string} name
 * @returns {{hairColor: number, eyeColor: number, bodyColor: number, role: string}}
 */
export function generateProfile(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  }
  const abs = Math.abs(hash);

  const hH = abs % 360;
  const hairColor = _hslToHex(hH, 40, 30);

  const hE = (abs * 137) % 360;
  const eyeColor = _hslToHex(hE, 60, 50);

  const hB = (abs * 53) % 60 + 15;
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

// ── Shared Geometries ──────────────────────

/** Shared geometries — created once, reused for every character. */
export const geo = {
  /** @type {THREE.SphereGeometry | null} */   head: null,
  /** @type {THREE.SphereGeometry | null} */   hair: null,
  /** @type {THREE.PlaneGeometry | null} */    face: null,
  /** @type {THREE.CylinderGeometry | null} */ body: null,
  /** @type {THREE.CylinderGeometry | null} */ arm: null,
  /** @type {THREE.CylinderGeometry | null} */ leg: null,
  /** @type {THREE.SphereGeometry | null} */   particle: null,
};

/** Create shared geometries once. */
export function ensureGeometries() {
  if (geo.head) return;
  geo.head     = new THREE.SphereGeometry(0.2, 16, 12);
  geo.hair     = new THREE.SphereGeometry(0.22, 16, 12);
  geo.face     = new THREE.PlaneGeometry(0.26, 0.26);
  geo.body     = new THREE.CylinderGeometry(0.12, 0.10, 0.25, 12);
  geo.arm      = new THREE.CylinderGeometry(0.04, 0.04, 0.15, 8);
  geo.leg      = new THREE.CylinderGeometry(0.05, 0.05, 0.12, 8);
  geo.particle = new THREE.SphereGeometry(0.02, 6, 4);
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
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 38, 5, 0.1 * Math.PI, 0.9 * Math.PI);
      ctx.stroke();
      break;
    }
    case "working": {
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.ellipse(cx - 8, 26, 5, 2.5, 0, 0, Math.PI * 2);
      ctx.ellipse(cx + 8, 26, 5, 2.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 4, 38);
      ctx.lineTo(cx + 4, 38);
      ctx.stroke();
      break;
    }
    case "thinking": {
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx + 8, 25, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 38, 3, 0, Math.PI * 2);
      ctx.stroke();
      break;
    }
    case "error": {
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 25, 5, 0, Math.PI * 2);
      ctx.arc(cx + 8, 25, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(cx - 6, 23, 1.5, 0, Math.PI * 2);
      ctx.arc(cx + 10, 23, 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#553333";
      ctx.beginPath();
      ctx.ellipse(cx, 39, 4, 3, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "sleeping": {
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
      break;
    }
    case "talking": {
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#553333";
      ctx.beginPath();
      ctx.ellipse(cx, 39, 4, 3, 0, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "reporting": {
      ctx.fillStyle = eyeCSS;
      ctx.beginPath();
      ctx.arc(cx - 8, 26, 4, 0, Math.PI * 2);
      ctx.arc(cx + 8, 26, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, 37, 6, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
      break;
    }
    case "success": {
      ctx.strokeStyle = eyeCSS;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx - 8, 28, 4, Math.PI, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx + 8, 28, 4, Math.PI, 2 * Math.PI);
      ctx.stroke();
      ctx.strokeStyle = "#664444";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, 37, 7, 0.1 * Math.PI, 0.9 * Math.PI);
      ctx.stroke();
      break;
    }
    case "walking": {
      _drawFace(ctx, "idle", eyeColor);
      break;
    }
    default: {
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
export function buildFaceTextures(eyeColor) {
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

// ── Character Construction ──────────────────────

/**
 * Build the procedural SD character mesh group as a faceless silhouette.
 * No eyes, no expressions — just a minimal body shape placeholder.
 * Total height ~0.7 units. Origin at the character's feet.
 * @param {string} name
 * @param {{hairColor: number, eyeColor: number, bodyColor: number, role: string}} profile
 * @returns {{group: THREE.Group, parts: Object, faceTextures: Record<string, THREE.CanvasTexture>}}
 */
export function buildCharacterMesh(name, profile) {
  ensureGeometries();

  const group = new THREE.Group();
  group.name = `character_${name}`;

  const silhouetteColor = profile.hairColor;
  const silhouetteMat = new THREE.MeshLambertMaterial({
    color: silhouetteColor,
    transparent: true,
    opacity: 0.55,
  });

  const legH   = 0.12;
  const bodyH  = 0.25;
  const headR  = 0.2;
  const headY  = legH + bodyH + headR;
  const bodyY  = legH + bodyH / 2;

  const legL = new THREE.Mesh(geo.leg, silhouetteMat);
  legL.position.set(-0.06, legH / 2, 0);
  legL.name = "legL";
  group.add(legL);

  const legR = new THREE.Mesh(geo.leg, silhouetteMat);
  legR.position.set(0.06, legH / 2, 0);
  legR.name = "legR";
  group.add(legR);

  const body = new THREE.Mesh(geo.body, silhouetteMat);
  body.position.set(0, bodyY, 0);
  body.name = "body";
  group.add(body);

  const armL = new THREE.Mesh(geo.arm, silhouetteMat);
  armL.position.set(-0.17, bodyY + 0.02, 0);
  armL.rotation.z = 0.15;
  armL.name = "armL";
  group.add(armL);

  const armR = new THREE.Mesh(geo.arm, silhouetteMat);
  armR.position.set(0.17, bodyY + 0.02, 0);
  armR.rotation.z = -0.15;
  armR.name = "armR";
  group.add(armR);

  const hair = new THREE.Mesh(geo.hair, silhouetteMat);
  hair.position.set(0, headY, -0.04);
  hair.name = "hair";
  group.add(hair);

  const head = new THREE.Mesh(geo.head, silhouetteMat);
  head.position.set(0, headY, 0);
  head.name = "head";
  group.add(head);

  group.traverse((child) => {
    child.userData.animaName = name;
  });

  const parts = { head, hair, face: null, body, armL, armR, legL, legR };
  return { group, parts, faceTextures: {} };
}

// ── Face Texture Switching ──────────────────────

/**
 * Switch the face texture to match the current state.
 * @param {Object} rec - CharacterRecord
 */
export function applyFaceTexture(rec) {
  if (!rec.parts.face) return;
  const tex = rec.faceTextures[rec.state] || rec.faceTextures["idle"];
  if (!tex) return;
  const faceMat = /** @type {THREE.MeshBasicMaterial} */ (rec.parts.face.material);
  if (faceMat.map !== tex) {
    faceMat.map = tex;
    faceMat.needsUpdate = true;
  }
}
