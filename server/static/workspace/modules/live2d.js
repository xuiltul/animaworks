// ── Bust-up Display Module ──────────────────────
// Displays AI-generated bust-up images with expression switching.
// Falls back to a coloured silhouette when no image asset exists.

import { probeAsset } from "./api.js";

// ── Character Profiles ──────────────────────

/**
 * @typedef {Object} CharacterProfile
 * @property {string} hair    - Hair color hex or hsl
 * @property {string} eyes    - Iris color hex or hsl
 * @property {string} clothing - Clothing color hex or hsl
 * @property {string} skin    - Skin tone hex or hsl
 */

/** @type {Map<string, CharacterProfile>} */
const _profileCache = new Map();

/**
 * Register appearance data for an anima from the server API.
 * Call before setCharacter() with data from /api/animas.
 * @param {string} name
 * @param {{hairColor?: string, eyeColor?: string, bodyColor?: string, clothingColor?: string}} appearance
 */
export function setLive2dAppearance(name, appearance) {
  if (!appearance || !name) return;
  const base = generateProfile(name);
  if (appearance.hairColor) base.hair = appearance.hairColor;
  if (appearance.eyeColor) base.eyes = appearance.eyeColor;
  if (appearance.bodyColor) base.skin = appearance.bodyColor;
  if (appearance.clothingColor) base.clothing = appearance.clothingColor;
  _profileCache.set(name.toLowerCase(), base);
}

/** Valid expression names. */
const EXPRESSIONS = ["neutral", "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"];

// ── Private State ──────────────────────

/** @type {HTMLCanvasElement|null} */
let _canvas = null;

/** @type {CanvasRenderingContext2D|null} */
let _ctx = null;

/** @type {ResizeObserver|null} */
let _resizeObserver = null;

/** @type {string|null} */
let _characterName = null;

/** @type {CharacterProfile|null} */
let _profile = null;

/** @type {string} */
let _expression = "neutral";

/** @type {boolean} — true when showing AI-generated bust-up image instead of Canvas */
let _imageMode = false;

/** @type {boolean} — true when no image asset exists; shows coloured silhouette */
let _silhouetteMode = false;

/** @type {HTMLImageElement|null} — bust-up image element (shown when _imageMode=true) */
let _bustupImg = null;

/** @type {Object<string, string>} — expression name → image URL map for multi-expression support */
let _bustupImages = {};

/** @type {function|null} */
let _clickCallback = null;

/** @type {number} Canvas logical width. */
let _width = 0;

/** @type {number} Canvas logical height. */
let _height = 0;

/** @type {number} Device pixel ratio. */
let _dpr = 1;

// ── Deterministic Hash ──────────────────────

/**
 * Simple deterministic hash from a string (djb2).
 * Returns an unsigned 32-bit integer.
 * @param {string} str
 * @returns {number}
 */
function hashString(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h + str.charCodeAt(i)) >>> 0;
  }
  return h;
}

/**
 * Generate a deterministic profile for unknown character names.
 * @param {string} name
 * @returns {CharacterProfile}
 */
function generateProfile(name) {
  const h = hashString(name);
  const hue1 = h % 360;
  const hue2 = (h * 7 + 123) % 360;
  const hue3 = (h * 13 + 47) % 360;
  const lightness = 20 + (h % 30);

  return {
    hair: `hsl(${hue1}, 40%, ${lightness}%)`,
    eyes: `hsl(${hue2}, 60%, 45%)`,
    clothing: `hsl(${hue3}, 50%, 35%)`,
    skin: `hsl(${(hue1 + 30) % 360}, 30%, 92%)`,
  };
}

// ── Canvas Sizing ──────────────────────

/**
 * Synchronise the canvas buffer size with its CSS container, respecting devicePixelRatio.
 */
function syncCanvasSize() {
  if (!_canvas) return;

  const parent = _canvas.parentElement;
  if (!parent) return;

  const rect = parent.getBoundingClientRect();
  _dpr = window.devicePixelRatio || 1;
  _width = rect.width;
  _height = rect.height;

  _canvas.width = Math.round(_width * _dpr);
  _canvas.height = Math.round(_height * _dpr);
  _canvas.style.width = `${_width}px`;
  _canvas.style.height = `${_height}px`;

  if (_ctx) {
    _ctx.setTransform(_dpr, 0, 0, _dpr, 0, 0);
  }
}

// ── Drawing Primitives ──────────────────────

/**
 * Draw a filled ellipse.
 * @param {number} cx - centre X
 * @param {number} cy - centre Y
 * @param {number} radX - horizontal radius
 * @param {number} radY - vertical radius
 * @param {string} color
 */
function fillEllipse(cx, cy, radX, radY, color) {
  _ctx.beginPath();
  _ctx.ellipse(cx, cy, Math.abs(radX), Math.abs(radY), 0, 0, Math.PI * 2);
  _ctx.fillStyle = color;
  _ctx.fill();
}

// ── Silhouette Drawing ──────────────────────

/**
 * Draw a generic grey silhouette when no character is loaded.
 */
function drawSilhouette() {
  if (!_ctx || _width <= 0 || _height <= 0) return;

  _ctx.clearRect(0, 0, _width, _height);

  // Soft grey background
  const grad = _ctx.createLinearGradient(0, 0, 0, _height);
  grad.addColorStop(0, "#e8e8ec");
  grad.addColorStop(1, "#d0d0d8");
  _ctx.fillStyle = grad;
  _ctx.fillRect(0, 0, _width, _height);

  const cx = _width * 0.5;
  const headCy = _height * 0.32;
  const headR = Math.min(_width, _height) * 0.14;

  // Head silhouette
  fillEllipse(cx, headCy, headR, headR * 1.1, "#b0b0b8");

  // Body silhouette
  const bodyTop = headCy + headR * 1.1;
  _ctx.beginPath();
  _ctx.moveTo(cx - headR * 0.5, bodyTop);
  _ctx.lineTo(cx - headR * 2.2, _height);
  _ctx.lineTo(cx + headR * 2.2, _height);
  _ctx.lineTo(cx + headR * 0.5, bodyTop);
  _ctx.closePath();
  _ctx.fillStyle = "#b0b0b8";
  _ctx.fill();

  // Question mark
  _ctx.font = `${headR * 0.8}px sans-serif`;
  _ctx.fillStyle = "#9090a0";
  _ctx.textAlign = "center";
  _ctx.textBaseline = "middle";
  _ctx.fillText("?", cx, headCy);
}

/**
 * Draw a coloured silhouette for a loaded character that has no bust-up image.
 * Uses the character's hair/clothing colours instead of generic grey.
 */
function drawCharacterSilhouette() {
  if (!_ctx || !_profile || _width <= 0 || _height <= 0) return;

  _ctx.clearRect(0, 0, _width, _height);

  // Soft gradient background
  const grad = _ctx.createLinearGradient(0, 0, 0, _height);
  grad.addColorStop(0, "#e8e8ec");
  grad.addColorStop(1, "#d0d0d8");
  _ctx.fillStyle = grad;
  _ctx.fillRect(0, 0, _width, _height);

  const cx = _width * 0.5;
  const headCy = _height * 0.30;
  const headR = Math.min(_width, _height) * 0.15;

  // Use character's hair colour for the silhouette, semi-transparent
  const baseColor = _profile.hair || "#888898";

  _ctx.save();
  _ctx.globalAlpha = 0.45;

  // Head
  fillEllipse(cx, headCy, headR, headR * 1.1, baseColor);

  // Hair halo (slightly larger, behind)
  fillEllipse(cx, headCy - headR * 0.1, headR * 1.15, headR * 1.25, baseColor);

  // Shoulders + body trapezoid
  const bodyTop = headCy + headR * 1.0;
  _ctx.beginPath();
  _ctx.moveTo(cx - headR * 0.6, bodyTop);
  // Shoulders
  _ctx.quadraticCurveTo(cx - headR * 1.8, bodyTop + headR * 0.3, cx - headR * 2.0, bodyTop + headR * 1.5);
  _ctx.lineTo(cx - headR * 1.6, _height);
  _ctx.lineTo(cx + headR * 1.6, _height);
  _ctx.lineTo(cx + headR * 2.0, bodyTop + headR * 1.5);
  _ctx.quadraticCurveTo(cx + headR * 1.8, bodyTop + headR * 0.3, cx + headR * 0.6, bodyTop);
  _ctx.closePath();
  _ctx.fillStyle = baseColor;
  _ctx.fill();

  _ctx.restore();
}

/**
 * Draw the appropriate silhouette based on current state.
 * Called after init, resize, and mode switches.
 */
function _redrawSilhouette() {
  if (_imageMode) return;
  if (_profile && _silhouetteMode) {
    drawCharacterSilhouette();
  } else if (!_profile) {
    drawSilhouette();
  }
}

// ── Click Handler ──────────────────────

/**
 * Internal click event handler.
 * @param {MouseEvent} e
 */
function handleClick(e) {
  if (_clickCallback) {
    _clickCallback(e);
  }
}

// ── Public API ──────────────────────

/**
 * Initialise the bust-up renderer on a canvas element.
 * Sets up the rendering context and resize observer.
 * @param {HTMLCanvasElement} canvas - The canvas element to draw on
 */
export function initBustup(canvas) {
  if (!canvas || !(canvas instanceof HTMLCanvasElement)) {
    throw new Error("initBustup requires a valid HTMLCanvasElement");
  }

  // Clean up any previous session
  disposeBustup();

  _canvas = canvas;
  _ctx = canvas.getContext("2d");

  // Initial sizing
  syncCanvasSize();

  // Observe container resize
  const parent = canvas.parentElement;
  if (parent && typeof ResizeObserver !== "undefined") {
    _resizeObserver = new ResizeObserver(() => {
      syncCanvasSize();
      _redrawSilhouette();
    });
    _resizeObserver.observe(parent);
  }

  // Click handler
  _canvas.addEventListener("click", handleClick);
  _canvas.style.cursor = "pointer";

  // Draw initial silhouette (no character loaded yet)
  drawSilhouette();
}

/**
 * Stop rendering and clean up all resources.
 * Safe to call multiple times.
 */
export function disposeBustup() {
  if (_resizeObserver) {
    _resizeObserver.disconnect();
    _resizeObserver = null;
  }

  if (_canvas) {
    _canvas.removeEventListener("click", handleClick);
    _canvas.style.cursor = "";
    _canvas = null;
  }

  _ctx = null;
  _characterName = null;
  _profile = null;
  _expression = "neutral";
  _clickCallback = null;
  _imageMode = false;
  _silhouetteMode = false;

  if (_bustupImg) {
    _bustupImg.removeEventListener("click", handleClick);
    _bustupImg.remove();
    _bustupImg = null;
  }
}

/**
 * Switch the displayed character by name.
 * If an AI-generated bust-up image exists, display it in Image Mode.
 * Falls back to a coloured silhouette when no image is available.
 * @param {string} name - Character name (e.g. "alice", "bob")
 */
export async function setCharacter(name) {
  if (!name || typeof name !== "string") {
    _characterName = null;
    _profile = null;
    _silhouetteMode = false;
    _bustupImages = {};
    _switchToSilhouetteMode();
    return;
  }

  const key = name.toLowerCase();
  _characterName = key;
  _profile = _profileCache.get(key) || generateProfile(key);

  // Reset expression
  _expression = "neutral";

  // Preload all expression images
  _bustupImages = {};
  for (const expr of EXPRESSIONS) {
    try {
      const filename = expr === "neutral"
        ? "avatar_bustup.png"
        : `avatar_bustup_${expr}.png`;
      const url = await probeAsset(key, filename);
      if (url) {
        _bustupImages[expr] = url;
      }
    } catch {
      // expression image not available — skip
    }
  }

  // Default image not found — fall back to silhouette
  if (!_bustupImages["neutral"]) {
    _silhouetteMode = true;
    _switchToSilhouetteMode();
    return;
  }

  _silhouetteMode = false;
  _switchToImageMode(_bustupImages["neutral"]);
}

/**
 * Switch to image mode: hide Canvas, show `<img>` with bust-up asset.
 * @param {string} url
 */
function _switchToImageMode(url) {
  _imageMode = true;

  if (_canvas) _canvas.style.display = "none";

  const parent = _canvas?.parentElement;
  if (!parent) return;

  if (!_bustupImg) {
    _bustupImg = document.createElement("img");
    _bustupImg.className = "bustup-image";
    _bustupImg.style.cssText =
      "width:100%;height:100%;object-fit:contain;cursor:pointer;display:block;";
    _bustupImg.addEventListener("click", handleClick);
    parent.appendChild(_bustupImg);
  }

  _bustupImg.src = url;
  _bustupImg.style.display = "block";
}

/**
 * Switch to silhouette mode: hide bust-up image, show Canvas with silhouette.
 */
function _switchToSilhouetteMode() {
  _imageMode = false;

  if (_bustupImg) _bustupImg.style.display = "none";
  if (_canvas) _canvas.style.display = "";

  _redrawSilhouette();
}

/**
 * Change the character's facial expression.
 * In Image Mode, switches the displayed bust-up image to the matching expression variant.
 * @param {string} expression - One of: 'neutral', 'smile', 'laugh', 'troubled', 'surprised', 'thinking', 'embarrassed'
 */
export function setExpression(expression) {
  const expr = (expression || "neutral").toLowerCase();
  if (!EXPRESSIONS.includes(expr)) {
    console.warn(`Unknown expression "${expression}", falling back to "neutral"`);
    return setExpression("neutral");
  }

  if (expr === _expression) return;
  _expression = expr;

  // Switch image in image mode
  if (_imageMode && _bustupImg) {
    const url = _bustupImages[expr] || _bustupImages["neutral"];
    if (url) {
      _bustupImg.src = url;
    }
  }
}

/**
 * Enable or disable lip-sync animation.
 * No-op: Image mode does not animate lip-sync.
 * Retained for API compatibility with app.js callers.
 * @param {boolean} _isTalking
 */
export function setTalking(_isTalking) {
  // No-op in image/silhouette mode.
}

/**
 * Register a click handler for character interaction.
 * Only one handler is active at a time; calling again replaces the previous.
 * @param {function} callback - Receives the native MouseEvent
 */
export function onClick(callback) {
  _clickCallback = typeof callback === "function" ? callback : null;
}

/**
 * Return the canvas element being used for rendering.
 * @returns {HTMLCanvasElement|null}
 */
export function getCanvas() {
  return _canvas;
}
