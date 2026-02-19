// ── Movement Module ──────────────────────
// Controls character movement along navigation paths.
// Manages per-character movement state, handles path following,
// rotation interpolation, and collision avoidance between characters.

import * as THREE from "three";
import { findPath } from "./navigation.js";
import { updateCharacterState } from "./character.js";

// ── Constants ──────────────────────

/** Rotation interpolation speed (radians per second). */
const ROTATION_LERP_SPEED = 8.0;

/** Distance threshold to consider a waypoint reached. */
const WAYPOINT_THRESHOLD = 0.05;

/** Minimum distance between characters before repulsion kicks in. */
const COLLISION_DISTANCE = 0.4;

/** Speed of collision repulsion (units per second). */
const COLLISION_PUSH_SPEED = 0.5;

// ── Module State ──────────────────────

/**
 * @typedef {Object} MoverState
 * @property {THREE.Vector3[]} path       - Remaining waypoints
 * @property {number}          pathIndex  - Current waypoint index
 * @property {number}          speed      - Movement speed (units/sec)
 * @property {Function|null}   onArrive   - Arrival callback
 * @property {THREE.Vector3}   homePos    - Home position (desk)
 * @property {THREE.Group}     group      - Character group reference
 * @property {string}          prevState  - Animation state before walking
 */

/** @type {Map<string, MoverState>} */
const _movers = new Map();

/** @type {import("./navigation.js").NavGrid | null} */
let _navGrid = null;

/** @type {number} */
let _floorWidth = 16;

/** @type {number} */
let _floorDepth = 12;

// ── Initialization ──────────────────────

/**
 * Initialize the movement system with navigation data.
 *
 * @param {import("./navigation.js").NavGrid} navGrid
 * @param {number} floorWidth
 * @param {number} floorDepth
 */
export function initMovement(navGrid, floorWidth, floorDepth) {
  _navGrid = navGrid;
  _floorWidth = floorWidth;
  _floorDepth = floorDepth;
}

/**
 * Register a character for movement control.
 *
 * @param {string} name
 * @param {THREE.Group} group
 * @param {THREE.Vector3} homePos
 */
export function registerCharacter(name, group, homePos) {
  _movers.set(name, {
    path: [],
    pathIndex: 0,
    speed: 1.5,
    onArrive: null,
    homePos: homePos.clone(),
    group,
    prevState: "idle",
  });
}

// ── Movement Commands ──────────────────────

/**
 * Start moving a character to a target world position.
 * Finds a path using A* and begins movement.
 *
 * @param {string} name
 * @param {THREE.Vector3} targetWorldPos
 * @param {Function|null} [onArrive=null] - Called when character arrives
 * @param {number} [speed=1.5] - Movement speed in units/sec
 * @returns {boolean} True if a path was found
 */
export function moveTo(name, targetWorldPos, onArrive = null, speed = 1.5) {
  const mover = _movers.get(name);
  if (!mover || !_navGrid) return false;

  const startPos = mover.group.position;
  const path = findPath(
    _navGrid,
    { x: startPos.x, z: startPos.z },
    { x: targetWorldPos.x, z: targetWorldPos.z },
    _floorWidth,
    _floorDepth,
  );

  if (path.length === 0) return false;

  // Store previous animation state so we can restore it on arrival
  mover.prevState = "idle";
  mover.path = path;
  mover.pathIndex = 0;
  mover.speed = speed;
  mover.onArrive = onArrive;

  // Switch to walking animation
  updateCharacterState(name, "walking");

  return true;
}

/**
 * Move a character back to its home (desk) position.
 *
 * @param {string} name
 * @param {Function|null} [onArrive=null]
 * @returns {boolean}
 */
export function moveToHome(name, onArrive = null) {
  const mover = _movers.get(name);
  if (!mover) return false;
  return moveTo(name, mover.homePos, onArrive);
}

/**
 * Stop a character's movement immediately.
 *
 * @param {string} name
 */
export function stopMovement(name) {
  const mover = _movers.get(name);
  if (!mover) return;

  mover.path = [];
  mover.pathIndex = 0;
  mover.onArrive = null;

  // Restore previous animation state
  updateCharacterState(name, mover.prevState);
}

/**
 * Check whether a character is currently moving.
 *
 * @param {string} name
 * @returns {boolean}
 */
export function isMoving(name) {
  const mover = _movers.get(name);
  return mover ? mover.path.length > 0 : false;
}

/**
 * Get a character's current world position.
 *
 * @param {string} name
 * @returns {THREE.Vector3 | null}
 */
export function getPosition(name) {
  const mover = _movers.get(name);
  return mover ? mover.group.position.clone() : null;
}

// ── Per-Frame Update ──────────────────────

/** Reusable vectors to avoid per-frame allocations. */
const _tmpDir = new THREE.Vector3();
const _tmpPush = new THREE.Vector3();

/**
 * Update all active character movements. Call once per frame.
 *
 * @param {number} deltaTime - Seconds since last frame
 */
export function updateMovements(deltaTime) {
  // Move each character along its path
  for (const [name, mover] of _movers) {
    if (mover.path.length === 0) continue;

    const target = mover.path[mover.pathIndex];
    const pos = mover.group.position;

    // Direction to next waypoint (XZ plane)
    _tmpDir.set(target.x - pos.x, 0, target.z - pos.z);
    const dist = _tmpDir.length();

    if (dist < WAYPOINT_THRESHOLD) {
      // Reached waypoint — advance
      mover.pathIndex++;

      if (mover.pathIndex >= mover.path.length) {
        // Arrived at final destination
        pos.x = target.x;
        pos.z = target.z;
        const cb = mover.onArrive;
        mover.path = [];
        mover.pathIndex = 0;
        mover.onArrive = null;

        // Restore previous animation state
        updateCharacterState(name, mover.prevState);

        if (cb) cb();
      }
      continue;
    }

    // Move toward waypoint
    _tmpDir.normalize();
    const step = Math.min(mover.speed * deltaTime, dist);
    pos.x += _tmpDir.x * step;
    pos.z += _tmpDir.z * step;

    // Smoothly rotate to face movement direction
    const targetAngle = Math.atan2(_tmpDir.x, _tmpDir.z);
    let currentAngle = mover.group.rotation.y;

    // Shortest angle difference
    let angleDiff = targetAngle - currentAngle;
    while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
    while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

    mover.group.rotation.y = currentAngle + angleDiff * Math.min(1, ROTATION_LERP_SPEED * deltaTime);
  }

  // Collision avoidance between all moving characters
  const names = [..._movers.keys()];
  for (let i = 0; i < names.length; i++) {
    const a = _movers.get(names[i]);
    if (!a || a.path.length === 0) continue;

    for (let j = i + 1; j < names.length; j++) {
      const b = _movers.get(names[j]);
      if (!b || b.path.length === 0) continue;

      _tmpPush.set(
        a.group.position.x - b.group.position.x,
        0,
        a.group.position.z - b.group.position.z,
      );
      const dist = _tmpPush.length();

      if (dist > 0 && dist < COLLISION_DISTANCE) {
        _tmpPush.normalize().multiplyScalar(COLLISION_PUSH_SPEED * deltaTime);
        a.group.position.x += _tmpPush.x;
        a.group.position.z += _tmpPush.z;
        b.group.position.x -= _tmpPush.x;
        b.group.position.z -= _tmpPush.z;
      }
    }
  }
}
