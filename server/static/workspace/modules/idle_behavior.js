/**
 * idle_behavior.js — Autonomous idle actions for office characters.
 *
 * Characters periodically perform idle behaviors (get a drink, visit a
 * colleague, browse the bookshelf, etc.) when no real work event is active.
 */

import { moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { updateCharacterState, getCharacterGroup, getCharacterHome } from "./character.js";

// ── Constants ──────────────────────────────────────

const IDLE_MIN_DELAY = 30;   // seconds before next behavior
const IDLE_MAX_DELAY = 90;   // seconds
const _rand = (lo, hi) => lo + Math.random() * (hi - lo);

// ── Behavior definitions ───────────────────────────

const BEHAVIORS = [
  {
    id: "desk_fidget",
    weight: 3,
    needsMovement: false,
    duration: [3, 6],
  },
  {
    id: "get_drink",
    weight: 2,
    needsMovement: true,
    target: "kitchen",
    stayDuration: [3, 5],
  },
  {
    id: "visit_colleague",
    weight: 2,
    needsMovement: true,
    target: "random_desk",
    stayDuration: [4, 8],
  },
  {
    id: "browse_bookshelf",
    weight: 1,
    needsMovement: true,
    target: "bookshelf",
    stayDuration: [4, 7],
  },
  {
    id: "look_out_window",
    weight: 1,
    needsMovement: true,
    target: "window_center",
    stayDuration: [5, 10],
  },
  {
    id: "stretch_walk",
    weight: 2,
    needsMovement: true,
    target: "center_open",
    stayDuration: [2, 4],
  },
  {
    id: "check_plant",
    weight: 1,
    needsMovement: true,
    target: "nearest_plant",
    stayDuration: [3, 5],
  },
  {
    id: "stand_think",
    weight: 2,
    needsMovement: false,
    duration: [4, 8],
  },
];

// Pre-compute cumulative weights for weighted random selection
const _totalWeight = BEHAVIORS.reduce((s, b) => s + b.weight, 0);

// ── Module state ───────────────────────────────────

/** @type {Map<string, CharacterIdleRecord>} */
const _records = new Map();

/**
 * @typedef {Object} CharacterIdleRecord
 * @property {string}  name
 * @property {boolean} enabled
 * @property {"waiting"|"executing"|"returning"} phase
 * @property {number}  timer        — countdown until next behavior (seconds)
 * @property {string|null} currentBehaviorId
 * @property {number}  stayTimer    — time remaining at POI
 * @property {string|null} previousState — anim state before behavior started
 */

let _globalEnabled = true;

/** @type {Map<string, {x:number, y:number, z:number}>} */
let _pois = new Map();

/** @type {string[]} all character names (for visit_colleague) */
let _allNames = [];

// ── POI helpers ────────────────────────────────────

/**
 * Build POI map from floor dimensions.
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {Map<string, {x:number, y:number, z:number}>}
 */
export function computePOIs(floorWidth, floorDepth) {
  const halfW = floorWidth / 2;
  const halfD = floorDepth / 2;

  const pois = new Map();

  pois.set("kitchen",       { x: halfW - 1.5,     y: 0, z: halfD - 1.5 });
  pois.set("bookshelf",     { x: -halfW + 0.8,    y: 0, z: 0 });
  pois.set("window_left",   { x: -halfW + 1.5,    y: 0, z: -halfD + 1.0 });
  pois.set("window_right",  { x: halfW - 1.5,     y: 0, z: -halfD + 1.0 });
  pois.set("window_center", { x: 0,               y: 0, z: -halfD + 1.0 });
  pois.set("center_open",   { x: 0,               y: 0, z: halfD / 2 });

  // Plant POIs — offset +0.5 from actual plant position
  const plantCorners = [
    [-halfW + 1, -halfD + 1],
    [ halfW - 1, -halfD + 1],
    [-halfW + 1,  halfD - 1],
    [ halfW - 1,  halfD - 1],
  ];
  plantCorners.forEach(([px, pz], i) => {
    // Offset toward center so character stands beside the plant
    const ox = px > 0 ? -0.5 : 0.5;
    const oz = pz > 0 ? -0.5 : 0.5;
    pois.set(`plant_${i + 1}`, { x: px + ox, y: 0, z: pz + oz });
  });

  return pois;
}

// ── Weighted random ────────────────────────────────

function _pickBehavior() {
  let r = Math.random() * _totalWeight;
  for (const b of BEHAVIORS) {
    r -= b.weight;
    if (r <= 0) return b;
  }
  return BEHAVIORS[BEHAVIORS.length - 1];
}

// ── Resolve target position ────────────────────────

function _resolveTarget(behavior, characterName) {
  const { target } = behavior;

  if (target === "random_desk") {
    // Pick a random colleague desk (not self)
    const others = _allNames.filter((n) => n !== characterName);
    if (others.length === 0) return null;
    const pick = others[Math.floor(Math.random() * others.length)];
    const home = getCharacterHome(pick);
    if (!home) return null;
    // Stand slightly in front of the colleague's desk (z + 0.8)
    return { x: home.x, y: 0, z: home.z + 0.8 };
  }

  if (target === "nearest_plant") {
    // Find nearest plant POI to character's home
    const home = getCharacterHome(characterName);
    if (!home) return _pois.get("plant_1") || null;
    let best = null;
    let bestDist = Infinity;
    for (const [key, pos] of _pois) {
      if (!key.startsWith("plant_")) continue;
      const dx = pos.x - home.x;
      const dz = pos.z - home.z;
      const d = dx * dx + dz * dz;
      if (d < bestDist) {
        bestDist = d;
        best = pos;
      }
    }
    return best;
  }

  // window_center can randomly become window_left or window_right
  if (target === "window_center") {
    const choices = ["window_left", "window_center", "window_right"];
    const pick = choices[Math.floor(Math.random() * choices.length)];
    return _pois.get(pick) || null;
  }

  return _pois.get(target) || null;
}

// ── Reset timer ────────────────────────────────────

function _resetTimer(rec) {
  rec.timer = _rand(IDLE_MIN_DELAY, IDLE_MAX_DELAY);
  rec.phase = "waiting";
  rec.currentBehaviorId = null;
  rec.stayTimer = 0;
}

// ── Execute behavior ───────────────────────────────

function _executeBehavior(rec) {
  const behavior = _pickBehavior();
  rec.currentBehaviorId = behavior.id;

  if (!behavior.needsMovement) {
    // Desk-local behavior (desk_fidget or stand_think)
    rec.phase = "executing";
    const dur = _rand(behavior.duration[0], behavior.duration[1]);
    rec.stayTimer = dur;

    // Set visual state
    const animState = behavior.id === "desk_fidget" ? "thinking" : "thinking";
    rec.previousState = "idle";
    updateCharacterState(rec.name, animState);
    return;
  }

  // Movement-based behavior
  const targetPos = _resolveTarget(behavior, rec.name);
  if (!targetPos) {
    // Could not resolve target — skip and reset
    _resetTimer(rec);
    return;
  }

  rec.phase = "executing";
  rec.previousState = "idle";

  const moved = moveTo(rec.name, targetPos, () => {
    // Arrived at POI — stay for a duration
    const stay = _rand(behavior.stayDuration[0], behavior.stayDuration[1]);
    rec.stayTimer = stay;

    // Set appropriate anim at POI
    const poiAnim = behavior.id === "visit_colleague" ? "talking" : "idle";
    updateCharacterState(rec.name, poiAnim);
  });

  if (!moved) {
    // Path not found — skip
    _resetTimer(rec);
  }
}

// ── Public API ─────────────────────────────────────

/**
 * Initialize idle behavior system.
 * @param {Map<string, any>|object} characters — character records (name → data)
 * @param {object} movementSystem — { moveTo, moveToHome, ... } (unused, we import directly)
 * @param {Map<string, {x:number, y:number, z:number}>} pois — POI map
 */
export function initIdleBehaviors(characters, movementSystem, pois) {
  _records.clear();
  _pois = pois instanceof Map ? pois : new Map(Object.entries(pois));

  const names = characters instanceof Map
    ? Array.from(characters.keys())
    : Object.keys(characters);

  _allNames = names;

  for (const name of names) {
    _records.set(name, {
      name,
      enabled: true,
      phase: "waiting",
      timer: _rand(IDLE_MIN_DELAY, IDLE_MAX_DELAY),
      currentBehaviorId: null,
      stayTimer: 0,
      previousState: null,
    });
  }
}

/**
 * Per-frame update.  Call from the main render loop.
 * @param {number} deltaTime — seconds since last frame
 */
export function updateIdleBehaviors(deltaTime) {
  if (!_globalEnabled) return;

  for (const rec of _records.values()) {
    if (!rec.enabled) continue;

    switch (rec.phase) {
      case "waiting": {
        rec.timer -= deltaTime;
        if (rec.timer <= 0) {
          // Don't start a behavior if already moving (e.g. from an external event)
          if (isMoving(rec.name)) {
            _resetTimer(rec);
            break;
          }
          _executeBehavior(rec);
        }
        break;
      }

      case "executing": {
        // If the character is still moving toward POI, wait
        if (isMoving(rec.name)) break;

        // Countdown stay duration at POI
        rec.stayTimer -= deltaTime;
        if (rec.stayTimer <= 0) {
          // Done at POI — return home
          rec.phase = "returning";
          updateCharacterState(rec.name, "idle");

          const behavior = BEHAVIORS.find((b) => b.id === rec.currentBehaviorId);
          if (behavior && behavior.needsMovement) {
            moveToHome(rec.name, () => {
              updateCharacterState(rec.name, "idle");
              _resetTimer(rec);
            });
          } else {
            // Desk-local behavior done
            updateCharacterState(rec.name, "idle");
            _resetTimer(rec);
          }
        }
        break;
      }

      case "returning": {
        // Wait until movement completes (callback handles reset)
        if (!isMoving(rec.name)) {
          // Movement already completed or was never started
          _resetTimer(rec);
        }
        break;
      }
    }
  }
}

/**
 * Cancel any in-progress behavior and return the character home.
 * Used when an external event (status change, message) occurs.
 * @param {string} characterName
 */
export function cancelBehavior(characterName) {
  const rec = _records.get(characterName);
  if (!rec) return;

  if (rec.phase === "waiting") return; // nothing to cancel

  // Stop any current movement
  if (isMoving(characterName)) {
    stopMovement(characterName);
  }

  // Return home
  updateCharacterState(characterName, "idle");
  moveToHome(characterName, () => {
    updateCharacterState(characterName, "idle");
  });

  _resetTimer(rec);
}

/**
 * Enable or disable idle behaviors for a single character.
 * @param {string} characterName
 * @param {boolean} enabled
 */
export function setBehaviorEnabled(characterName, enabled) {
  const rec = _records.get(characterName);
  if (!rec) return;
  rec.enabled = enabled;
  if (!enabled) {
    cancelBehavior(characterName);
  }
}

/**
 * Enable or disable the entire idle behavior system.
 * @param {boolean} enabled
 */
export function setGlobalEnabled(enabled) {
  _globalEnabled = enabled;
  if (!enabled) {
    for (const name of _records.keys()) {
      cancelBehavior(name);
    }
  }
}
