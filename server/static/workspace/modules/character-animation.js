// ── Character Animation Module ──────────────────────
// State-driven animation functions for procedural SD characters.
// Each function takes (record, dt, elapsed) and mutates the group's transforms.

import {
  showStatusSprite,
  hideStatusSprite,
  updateSuccessParticles,
  clearParticles,
} from "./character-effects.js";

// ── Animation Functions ──────────────────────

/**
 * Reset all animated transforms to rest pose.
 * @param {Object} rec - CharacterRecord
 */
export function resetPose(rec) {
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
 * @param {Object} rec
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
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animWorking(rec, _dt, elapsed) {
  const { parts } = rec;

  parts.body.rotation.x = 0.12;
  parts.head.rotation.x = 0.08;
  parts.hair.rotation.x = 0.08;
  if (parts.face) parts.face.rotation.x = 0.08;

  const cycle = elapsed * 6;
  parts.armL.rotation.x = Math.sin(cycle) * 0.25;
  parts.armR.rotation.x = Math.sin(cycle + Math.PI) * 0.25;

  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 3) * 0.005;
}

/**
 * Thinking: head tilt, one arm to chin.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animThinking(rec, _dt, elapsed) {
  const { parts } = rec;

  parts.head.rotation.z = 0.2;
  parts.hair.rotation.z = 0.2;
  if (parts.face) parts.face.rotation.z = 0.2;

  parts.armR.rotation.x = -0.6;
  parts.armR.rotation.z = -0.4;

  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 1.2) * 0.01;
}

/**
 * Error: red flash + exclamation sprite above head.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animError(rec, _dt, elapsed) {
  const { parts } = rec;

  rec.group.position.x = (rec.group.userData._baseX ?? 0) + Math.sin(elapsed * 15) * 0.01;

  const flash = (Math.sin(elapsed * 4) + 1) * 0.5;
  const bodyMat = /** @type {THREE.MeshLambertMaterial} */ (parts.body.material);
  bodyMat.emissive.setRGB(flash * 0.6, 0, 0);

  showStatusSprite(rec, "!", "#ff3333", elapsed);
}

/**
 * Sleeping: body tilts forward, "Z" above head, gentle breathing.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animSleeping(rec, _dt, elapsed) {
  const { parts } = rec;

  parts.body.rotation.x = 0.3;
  parts.head.rotation.x = 0.25;
  parts.hair.rotation.x = 0.25;
  if (parts.face) parts.face.rotation.x = 0.25;

  const breath = 1 + Math.sin(elapsed * 1.8) * 0.03;
  parts.body.scale.set(1, breath, 1);

  showStatusSprite(rec, "Z", "#6688cc", elapsed);
  if (rec.statusSprite) {
    rec.statusSprite.position.y = 0.8 + Math.sin(elapsed * 0.8) * 0.05;
    rec.statusSprite.material.opacity = 0.6 + Math.sin(elapsed * 1.5) * 0.3;
  }
}

/**
 * Talking: slight bounce + arm gestures.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animTalking(rec, _dt, elapsed) {
  const { parts } = rec;

  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(elapsed * 4)) * 0.02;

  parts.armL.rotation.z = 0.15 + Math.sin(elapsed * 3) * 0.3;
  parts.armR.rotation.z = -0.15 + Math.sin(elapsed * 3 + 1) * 0.3;

  parts.head.rotation.x = Math.sin(elapsed * 2.5) * 0.08;
  parts.hair.rotation.x = parts.head.rotation.x;
  if (parts.face) parts.face.rotation.x = parts.head.rotation.x;
}

/**
 * Reporting: turn to face supervisor desk.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animReporting(rec, _dt, elapsed) {
  const targetAngle = Math.atan2(
    -rec.group.position.x,
    -rec.group.position.z,
  );
  rec.group.rotation.y = targetAngle;

  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.sin(elapsed * 1.5) * 0.01;

  const { parts } = rec;
  parts.armR.rotation.z = -0.15 + Math.sin(elapsed * 2) * 0.15;
}

/**
 * Success: small jump + particle burst.
 * @param {Object} rec
 * @param {number} dt
 * @param {number} elapsed
 */
function _animSuccess(rec, dt, elapsed) {
  const { parts } = rec;

  const jumpT = (elapsed * 2) % (Math.PI * 2);
  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(jumpT)) * 0.08;

  parts.armL.rotation.z = 0.8;
  parts.armR.rotation.z = -0.8;

  updateSuccessParticles(rec, dt);
}

/**
 * Walking: leg swing, arm swing, body bob and sway.
 * @param {Object} rec
 * @param {number} _dt
 * @param {number} elapsed
 */
function _animWalking(rec, _dt, elapsed) {
  const { parts } = rec;

  const legCycle = elapsed * 8;
  parts.legL.rotation.x = Math.sin(legCycle) * 0.4;
  parts.legR.rotation.x = Math.sin(legCycle + Math.PI) * 0.4;

  parts.armL.rotation.x = Math.sin(legCycle + Math.PI) * 0.3;
  parts.armR.rotation.x = Math.sin(legCycle) * 0.3;

  const baseY = rec.group.userData._baseY;
  rec.group.position.y = baseY + Math.abs(Math.sin(legCycle * 2)) * 0.015;

  parts.body.rotation.z = Math.sin(legCycle) * 0.03;
}

// ── Animation Dispatch ──────────────────────

/** @type {Record<string, (rec: Object, dt: number, elapsed: number) => void>} */
export const animMap = {
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

// ── Easing ──────────────────────

/**
 * Easing function for smooth transitions.
 * @param {number} t - 0..1
 * @returns {number}
 */
export function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// ── State Transition Cleanup ──────────────────────

/**
 * Clean up visuals that are state-specific (sprites, particles, emissive).
 * Called when leaving a state.
 * @param {Object} rec - CharacterRecord
 * @param {string} oldState
 */
export function cleanupState(rec, oldState) {
  if (oldState === "error" || oldState === "sleeping") {
    hideStatusSprite(rec);
  }
  if (oldState === "error") {
    const bodyMat = /** @type {THREE.MeshLambertMaterial} */ (rec.parts.body.material);
    bodyMat.emissive.setRGB(0, 0, 0);
    rec.group.position.x = rec.group.userData._baseX ?? 0;
  }
  if (oldState === "success") {
    clearParticles(rec);
  }
  rec.parts.body.scale.set(1, 1, 1);
}
