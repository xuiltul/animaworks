/**
 * interactions.js — Interaction visualization for the 3D office.
 *
 * Provides message particles (bezier-curve projectiles), speech bubbles
 * (canvas-textured sprites), and full conversation sequences where a
 * character walks to a colleague's desk and they exchange dialogue.
 */

import * as THREE from "three";
import { moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { updateCharacterState, getCharacterGroup, getCharacterHome } from "./character.js";

// ── Module state ───────────────────────────────────

/** @type {THREE.Scene|null} */
let _scene = null;

/** @type {Map<string, any>} character records */
let _characters = null;

/** @type {object} movement API */
let _movement = null;

/** @type {MessageParticle[]} */
const _activeParticles = [];

/** @type {SpeechBubble[]} */
const _activeBubbles = [];

/** @type {ConversationSequence|null} */
let _activeConversation = null;

/** @type {ConversationSequence[]} */
const _queue = [];

// ── Easing ─────────────────────────────────────────

function _easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// ── MessageParticle ────────────────────────────────

const PARTICLE_DURATION = 1.5;     // seconds
const TRAIL_LENGTH = 5;
const PARTICLE_RADIUS = 0.06;

class MessageParticle {
  /**
   * @param {THREE.Scene} scene
   * @param {THREE.Vector3} from  — sender head position (y+0.9)
   * @param {THREE.Vector3} to    — receiver head position (y+0.9)
   * @param {number} color        — hex color
   * @param {Function} onComplete
   */
  constructor(scene, from, to, color, onComplete) {
    this._scene = scene;
    this._from = from.clone();
    this._to = to.clone();
    this._onComplete = onComplete;
    this._elapsed = 0;

    // Bezier control point — midpoint raised +2.0
    this._control = new THREE.Vector3(
      (from.x + to.x) / 2,
      Math.max(from.y, to.y) + 2.0,
      (from.z + to.z) / 2,
    );

    // Main particle sprite
    const mat = new THREE.SpriteMaterial({
      color,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthTest: false,
    });
    this._sprite = new THREE.Sprite(mat);
    this._sprite.scale.set(PARTICLE_RADIUS * 2, PARTICLE_RADIUS * 2, 1);
    this._sprite.position.copy(from);
    scene.add(this._sprite);

    // Trail sprites
    this._trail = [];
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      const trailMat = new THREE.SpriteMaterial({
        color,
        transparent: true,
        opacity: 0.3 - i * 0.05,
        blending: THREE.AdditiveBlending,
        depthTest: false,
      });
      const ts = new THREE.Sprite(trailMat);
      ts.scale.set(PARTICLE_RADIUS * 1.5, PARTICLE_RADIUS * 1.5, 1);
      ts.position.copy(from);
      ts.visible = false;
      scene.add(ts);
      this._trail.push(ts);
    }

    this._prevPositions = [];
    this._disposed = false;
  }

  /**
   * Sample position on the quadratic bezier at parameter t.
   * @param {number} t — 0..1
   * @returns {THREE.Vector3}
   */
  _bezierAt(t) {
    const a = this._from;
    const b = this._control;
    const c = this._to;
    const omt = 1 - t;
    return new THREE.Vector3(
      omt * omt * a.x + 2 * omt * t * b.x + t * t * c.x,
      omt * omt * a.y + 2 * omt * t * b.y + t * t * c.y,
      omt * omt * a.z + 2 * omt * t * b.z + t * t * c.z,
    );
  }

  /**
   * @param {number} dt — seconds
   * @returns {boolean} true when complete
   */
  update(dt) {
    if (this._disposed) return true;

    this._elapsed += dt;
    const rawT = Math.min(this._elapsed / PARTICLE_DURATION, 1.0);
    const t = _easeInOut(rawT);

    const pos = this._bezierAt(t);
    this._sprite.position.copy(pos);

    // Push to history and update trail
    this._prevPositions.unshift(pos.clone());
    if (this._prevPositions.length > TRAIL_LENGTH) {
      this._prevPositions.length = TRAIL_LENGTH;
    }
    for (let i = 0; i < this._trail.length; i++) {
      if (i < this._prevPositions.length) {
        this._trail[i].position.copy(this._prevPositions[i]);
        this._trail[i].visible = true;
      }
    }

    if (rawT >= 1.0) {
      this.dispose();
      if (this._onComplete) this._onComplete();
      return true;
    }
    return false;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._scene.remove(this._sprite);
    this._sprite.material.dispose();
    for (const ts of this._trail) {
      this._scene.remove(ts);
      ts.material.dispose();
    }
    this._trail.length = 0;
  }
}

// ── SpeechBubble ───────────────────────────────────

const BUBBLE_FADE_DURATION = 0.3;   // seconds
const BUBBLE_CANVAS_W = 256;
const BUBBLE_CANVAS_H = 64;
const BUBBLE_MAX_CHARS = 40;

class SpeechBubble {
  /**
   * @param {THREE.Scene} scene
   * @param {THREE.Group} characterGroup
   * @param {string} text
   * @param {number} duration — total display time in seconds (default 3)
   */
  constructor(scene, characterGroup, text, duration = 3.0) {
    this._scene = scene;
    this._group = characterGroup;
    this._duration = duration;
    this._elapsed = 0;
    this._disposed = false;

    // Truncate
    const display = text.length > BUBBLE_MAX_CHARS
      ? text.slice(0, BUBBLE_MAX_CHARS - 3) + "..."
      : text;

    // Render to canvas
    const canvas = document.createElement("canvas");
    canvas.width = BUBBLE_CANVAS_W;
    canvas.height = BUBBLE_CANVAS_H;
    const ctx = canvas.getContext("2d");

    // Background — rounded rect
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    _roundRect(ctx, 0, 0, BUBBLE_CANVAS_W, BUBBLE_CANVAS_H, 8);
    ctx.fill();

    // Text
    ctx.fillStyle = "#333";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(display, BUBBLE_CANVAS_W / 2, BUBBLE_CANVAS_H / 2);

    // Texture + sprite
    this._texture = new THREE.CanvasTexture(canvas);
    this._texture.needsUpdate = true;

    const mat = new THREE.SpriteMaterial({
      map: this._texture,
      transparent: true,
      opacity: 0,
      depthTest: false,
    });
    this._sprite = new THREE.Sprite(mat);
    this._sprite.scale.set(2.0, 0.5, 1);

    // Position above character head
    this._sprite.position.set(
      characterGroup.position.x,
      characterGroup.position.y + 1.0,
      characterGroup.position.z,
    );
    scene.add(this._sprite);
  }

  /**
   * @param {number} dt
   * @returns {boolean} true when complete
   */
  update(dt) {
    if (this._disposed) return true;

    this._elapsed += dt;

    // Follow character
    this._sprite.position.set(
      this._group.position.x,
      this._group.position.y + 1.0,
      this._group.position.z,
    );

    // Fade logic
    const fadeIn = Math.min(this._elapsed / BUBBLE_FADE_DURATION, 1.0);
    const remaining = this._duration - this._elapsed;
    const fadeOut = remaining < BUBBLE_FADE_DURATION
      ? Math.max(remaining / BUBBLE_FADE_DURATION, 0)
      : 1.0;

    this._sprite.material.opacity = Math.min(fadeIn, fadeOut);

    if (this._elapsed >= this._duration) {
      this.dispose();
      return true;
    }
    return false;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._scene.remove(this._sprite);
    this._sprite.material.dispose();
    this._texture.dispose();
  }
}

/**
 * Draw a rounded rectangle path on a canvas context.
 */
function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ── ConversationSequence ───────────────────────────

/**
 * Full conversation: character walks to colleague, they face each other,
 * exchange speech bubbles, then the visitor returns home.
 */
class ConversationSequence {
  /**
   * @param {object} opts
   * @param {string} opts.fromName
   * @param {string} opts.toName
   * @param {string[]} opts.messages
   */
  constructor(opts) {
    this.fromName = opts.fromName;
    this.toName = opts.toName;
    this.messages = opts.messages || [];
    this._phase = "walking";  // walking | talking | returning | done
    this._msgIndex = 0;
    this._currentBubble = null;
    this._disposed = false;
    this._previousStates = {};

    // Start: walk fromName to toName's desk
    const toHome = getCharacterHome(this.toName);
    if (!toHome) {
      this._phase = "done";
      return;
    }

    const target = { x: toHome.x, y: 0, z: toHome.z + 0.8 };
    const moved = moveTo(this.fromName, target, () => {
      this._onArrived();
    });
    if (!moved) {
      this._phase = "done";
    }
  }

  _onArrived() {
    this._phase = "talking";

    // Face each other
    const fromGroup = getCharacterGroup(this.fromName);
    const toGroup = getCharacterGroup(this.toName);
    if (fromGroup && toGroup) {
      const dx = toGroup.position.x - fromGroup.position.x;
      const dz = toGroup.position.z - fromGroup.position.z;
      fromGroup.rotation.y = Math.atan2(dx, dz);
      toGroup.rotation.y = Math.atan2(-dx, -dz);
    }

    // Set both to talking state
    this._previousStates[this.fromName] = "idle";
    this._previousStates[this.toName] = "idle";
    updateCharacterState(this.fromName, "talking");
    updateCharacterState(this.toName, "talking");

    // Start first message
    this._showNextMessage();
  }

  _showNextMessage() {
    if (this._msgIndex >= this.messages.length) {
      // All messages done — return home
      this._phase = "returning";
      updateCharacterState(this.fromName, "idle");
      updateCharacterState(this.toName, "idle");

      moveToHome(this.fromName, () => {
        this._phase = "done";
      });
      return;
    }

    const text = this.messages[this._msgIndex];
    // Alternate between from and to
    const speaker = this._msgIndex % 2 === 0 ? this.fromName : this.toName;
    const group = getCharacterGroup(speaker);
    if (group && _scene) {
      this._currentBubble = new SpeechBubble(_scene, group, text, 3.0);
    }
    this._msgIndex++;
  }

  /**
   * @param {number} dt
   * @returns {boolean} true when done
   */
  update(dt) {
    if (this._disposed || this._phase === "done") return true;

    if (this._phase === "talking" && this._currentBubble) {
      const done = this._currentBubble.update(dt);
      if (done) {
        this._currentBubble = null;
        this._showNextMessage();
      }
    }

    // walking and returning are handled by movement callbacks
    return this._phase === "done";
  }

  cancel() {
    if (this._disposed) return;

    if (this._currentBubble) {
      this._currentBubble.dispose();
      this._currentBubble = null;
    }

    if (isMoving(this.fromName)) {
      stopMovement(this.fromName);
    }

    updateCharacterState(this.fromName, "idle");
    updateCharacterState(this.toName, "idle");

    moveToHome(this.fromName);
    this._phase = "done";
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    if (this._currentBubble) {
      this._currentBubble.dispose();
      this._currentBubble = null;
    }
  }
}

// ── Public API ─────────────────────────────────────

/**
 * Initialize the interaction system.
 * @param {THREE.Scene} scene
 * @param {Map<string, any>|object} characters
 * @param {object} movementSystem — { moveTo, isMoving, stopMovement }
 */
export function initInteractions(scene, characters, movementSystem) {
  _scene = scene;
  _characters = characters;
  _movement = movementSystem;
}

/**
 * Show a short message effect: particle arc + sender speech bubble.
 * @param {string} fromName
 * @param {string} toName
 * @param {string} text
 */
export function showMessageEffect(fromName, toName, text) {
  if (!_scene) return;

  const fromGroup = getCharacterGroup(fromName);
  const toGroup = getCharacterGroup(toName);
  if (!fromGroup || !toGroup) return;

  const fromPos = new THREE.Vector3(
    fromGroup.position.x,
    fromGroup.position.y + 0.9,
    fromGroup.position.z,
  );
  const toPos = new THREE.Vector3(
    toGroup.position.x,
    toGroup.position.y + 0.9,
    toGroup.position.z,
  );

  // Particle
  const color = 0x4488ff;
  const particle = new MessageParticle(_scene, fromPos, toPos, color, null);
  _activeParticles.push(particle);

  // Speech bubble on sender
  if (text) {
    const bubble = new SpeechBubble(_scene, fromGroup, text, 3.0);
    _activeBubbles.push(bubble);
  }
}

/**
 * Enqueue a full conversation sequence.
 * @param {string} fromName
 * @param {string} toName
 * @param {string[]} messages — alternating from/to lines
 */
export function showConversation(fromName, toName, messages) {
  const seq = new ConversationSequence({ fromName, toName, messages });
  if (_activeConversation) {
    _queue.push(seq);
  } else {
    _activeConversation = seq;
  }
}

/**
 * Per-frame update. Call from the main render loop.
 * @param {number} deltaTime
 */
export function updateInteractions(deltaTime) {
  // Update particles
  for (let i = _activeParticles.length - 1; i >= 0; i--) {
    if (_activeParticles[i].update(deltaTime)) {
      _activeParticles.splice(i, 1);
    }
  }

  // Update bubbles
  for (let i = _activeBubbles.length - 1; i >= 0; i--) {
    if (_activeBubbles[i].update(deltaTime)) {
      _activeBubbles.splice(i, 1);
    }
  }

  // Update active conversation
  if (_activeConversation) {
    if (_activeConversation.update(deltaTime)) {
      _activeConversation.dispose();
      _activeConversation = null;

      // Dequeue next
      if (_queue.length > 0) {
        _activeConversation = _queue.shift();
      }
    }
  }
}

/**
 * Clean up all active effects and queued conversations.
 */
export function dispose() {
  for (const p of _activeParticles) p.dispose();
  _activeParticles.length = 0;

  for (const b of _activeBubbles) b.dispose();
  _activeBubbles.length = 0;

  if (_activeConversation) {
    _activeConversation.cancel();
    _activeConversation.dispose();
    _activeConversation = null;
  }

  for (const s of _queue) {
    s.cancel();
    s.dispose();
  }
  _queue.length = 0;
}
