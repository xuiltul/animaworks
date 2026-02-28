// ── Character Effects Module ──────────────────────
// Status sprites (floating "!" / "Z") and success particle burst effects.

import * as THREE from "three";
import { geo } from "./character-appearance.js";

const MAX_PARTICLES = 10;

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

// ── Status Sprite Helpers ──────────────────────

/**
 * Ensure a floating status sprite is visible and positioned above the head.
 * @param {Object} rec - CharacterRecord
 * @param {string} text
 * @param {string} color
 * @param {number} elapsed
 */
export function showStatusSprite(rec, text, color, elapsed) {
  if (!rec.statusSprite) {
    rec.statusSprite = _createTextSprite(text, color);
    rec.group.add(rec.statusSprite);
  }
  rec.statusSprite.visible = true;
  rec.statusSprite.position.set(0, 0.85 + Math.sin(elapsed * 2) * 0.03, 0);
}

/**
 * Hide and remove the floating status sprite.
 * @param {Object} rec - CharacterRecord
 */
export function hideStatusSprite(rec) {
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
 * @param {Object} rec - CharacterRecord
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
    const mesh = new THREE.Mesh(geo.particle, mat);
    mesh.position.set(0, 0.6, 0);
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
 * @param {Object} rec - CharacterRecord
 * @param {number} dt
 */
export function updateSuccessParticles(rec, dt) {
  _spawnSuccessParticles(rec);
  rec.particleAge += dt;

  for (const p of rec.particles) {
    const vel = /** @type {THREE.Vector3} */ (p.userData.vel);
    vel.y -= 2.0 * dt; // gravity
    p.position.addScaledVector(vel, dt);

    const mat = /** @type {THREE.MeshBasicMaterial} */ (p.material);
    mat.opacity = Math.max(0, 1 - rec.particleAge * 0.8);
  }

  if (rec.particleAge > 1.5) {
    clearParticles(rec);
  }
}

/**
 * Remove all particles from a character.
 * @param {Object} rec - CharacterRecord
 */
export function clearParticles(rec) {
  for (const p of rec.particles) {
    rec.group.remove(p);
    /** @type {THREE.MeshBasicMaterial} */ (p.material).dispose();
  }
  rec.particles = [];
  rec.particleAge = 0;
}
