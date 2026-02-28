// ── 3D Office Scene ──────────────────────
// Three.js-based isometric office for the AnimaWorks Workspace.
// Orchestrator: delegates construction to sub-modules, owns scene state
// and the render loop.  Characters are placed by character.js via getDesks().

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { buildFloor, buildWalls, buildLighting } from "./office-structure.js";
import { createHelpers, createDesk, buildDecorations, buildBookshelf, buildKitchenCorner } from "./office-furniture.js";
import { buildOrgTree, computeTreeLayout, computeFloorDimensions, buildDesks, buildConnectors } from "./office-layout.js";
import {
  createCamera,
  updateCameraFrustum as _updateCameraFrustum,
  onPointerClick as _onPointerClick,
  createHighlightMesh,
  highlightDesk as _highlightDesk,
  clearHighlight as _clearHighlight,
  registerClickTarget as _registerClickTarget,
  unregisterClickTarget as _unregisterClickTarget,
} from "./office-interaction.js";

// ── Module State ──────────────────────

/** @type {THREE.Scene | null} */
let _scene = null;

/** @type {THREE.OrthographicCamera | null} */
let _camera = null;

/** @type {THREE.WebGLRenderer | null} */
let _renderer = null;

/** @type {OrbitControls | null} */
let _controls = null;

/** @type {HTMLElement | null} */
let _container = null;

/** @type {number | null} */
let _animationFrameId = null;

/** @type {ResizeObserver | null} */
let _resizeObserver = null;

/**
 * Dynamic desk layout computed from anima data.
 * @type {Record<string, {x: number, y: number, z: number}>}
 */
let _deskLayout = {};

let _floorWidth = 16;
let _floorDepth = 12;

/** @type {Map<string, THREE.Group>} */
const _deskGroups = new Map();

/** Maps mesh UUID to anima name (for raycaster hit lookup). */
const _meshToName = new Map();

/** @type {THREE.Mesh | null} */
let _highlightMesh = null;

/** @type {string | null} */
let _highlightedName = null;

/** @type {((name: string) => void) | null} */
let _characterClickHandler = null;

/** @type {Set<THREE.BufferGeometry | THREE.Material | THREE.Texture>} */
const _disposables = new Set();

/** @type {((dt: number, elapsed: number) => void) | null} */
let _characterUpdateFn = null;

const _clock = new THREE.Clock();

// ── Render Loop ──────────────────────

function renderLoop() {
  _animationFrameId = requestAnimationFrame(renderLoop);

  const dt = _clock.getDelta();
  const elapsed = _clock.getElapsedTime();

  if (_characterUpdateFn) {
    _characterUpdateFn(dt, elapsed);
  }
  if (_controls) {
    _controls.update();
  }
  if (_renderer && _scene && _camera) {
    _renderer.render(_scene, _camera);
  }
}

// ── Internal Helpers ──────────────────────

function _onWindowResize() {
  if (!_container) return;
  if (!_camera || !_renderer) return;
  _updateCameraFrustum(_camera, _renderer, _floorWidth, _floorDepth, _container.clientWidth, _container.clientHeight);
}

function _handlePointerClick(event) {
  if (!_renderer || !_camera || !_scene) return;
  _onPointerClick(event, _renderer, _camera, _scene, _meshToName, _characterClickHandler);
}

// ── Public API ──────────────────────

/**
 * Initialize the 3D office scene inside the given container element.
 * @param {HTMLElement} container
 * @param {Array<{name: string, role?: string, supervisor?: string}>} [animas]
 */
export function initOffice(container, animas = []) {
  if (_renderer) {
    disposeOffice();
  }

  _container = container;
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  // Compute dynamic desk layout
  if (animas.length > 0) {
    const roots = buildOrgTree(animas);
    _deskLayout = computeTreeLayout(roots);
  } else {
    _deskLayout = {};
  }

  const dims = computeFloorDimensions(_deskLayout);
  _floorWidth = dims.width;
  _floorDepth = dims.depth;

  // Scene
  _scene = new THREE.Scene();
  _scene.background = new THREE.Color(0xf0ece3);

  // Camera
  _camera = createCamera(width / height, _floorWidth, _floorDepth);

  // Renderer
  _renderer = new THREE.WebGLRenderer({ antialias: true });
  _renderer.setSize(width, height);
  _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  _renderer.shadowMap.enabled = true;
  _renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(_renderer.domElement);

  // Controls
  _controls = new OrbitControls(_camera, _renderer.domElement);
  _controls.enableDamping = true;
  _controls.dampingFactor = 0.08;
  _controls.enablePan = true;
  _controls.enableRotate = true;
  _controls.enableZoom = true;
  _controls.minZoom = 0.5;
  _controls.maxZoom = 10;
  _controls.maxPolarAngle = Math.PI / 2.5;
  _controls.minPolarAngle = Math.PI / 6;
  _controls.target.set(0, 0, 0);
  _controls.update();

  // Create disposal-tracked helpers
  const helpers = createHelpers(_disposables);

  // Build scene
  buildFloor(_scene, _floorWidth, _floorDepth, helpers);
  buildWalls(_scene, _floorWidth, _floorDepth, helpers);
  buildDesks(_scene, _deskLayout, _deskGroups, (name, pos) =>
    createDesk(name, pos, _disposables, _meshToName, helpers),
  );
  buildConnectors(_scene, animas, _deskLayout, _disposables);
  buildDecorations(_scene, _floorWidth, _floorDepth, helpers);
  buildBookshelf(_scene, _floorWidth, helpers);
  buildKitchenCorner(_scene, _floorWidth, _floorDepth, helpers);
  buildLighting(_scene, _floorWidth, _floorDepth);

  // Highlight mesh
  _highlightMesh = createHighlightMesh(_disposables, helpers);
  _scene.add(_highlightMesh);

  // Events
  _renderer.domElement.addEventListener("click", _handlePointerClick);

  // Resize handling
  if (typeof ResizeObserver !== "undefined") {
    _resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0 && _camera && _renderer) {
          _updateCameraFrustum(_camera, _renderer, _floorWidth, _floorDepth, w, h);
        }
      }
    });
    _resizeObserver.observe(container);
  } else {
    window.addEventListener("resize", _onWindowResize);
  }

  renderLoop();
}

/**
 * Dispose of all Three.js resources and stop the render loop.
 */
export function disposeOffice() {
  if (_animationFrameId !== null) {
    cancelAnimationFrame(_animationFrameId);
    _animationFrameId = null;
  }

  if (_renderer) {
    _renderer.domElement.removeEventListener("click", _handlePointerClick);
  }

  if (_resizeObserver) {
    _resizeObserver.disconnect();
    _resizeObserver = null;
  } else {
    window.removeEventListener("resize", _onWindowResize);
  }

  if (_controls) {
    _controls.dispose();
    _controls = null;
  }

  for (const resource of _disposables) {
    if (typeof resource.dispose === "function") {
      resource.dispose();
    }
  }
  _disposables.clear();

  if (_renderer) {
    _renderer.dispose();
    if (_renderer.domElement.parentElement) {
      _renderer.domElement.parentElement.removeChild(_renderer.domElement);
    }
    _renderer = null;
  }

  _scene = null;
  _camera = null;
  _container = null;
  _highlightMesh = null;
  _highlightedName = null;
  _deskLayout = {};
  _deskGroups.clear();
  _meshToName.clear();
}

/** @returns {Record<string, {x: number, y: number, z: number}>} */
export function getDesks() {
  /** @type {Record<string, {x: number, y: number, z: number}>} */
  const result = {};
  for (const [name, pos] of Object.entries(_deskLayout)) {
    result[name] = { x: pos.x, y: pos.y, z: pos.z };
  }
  return result;
}

/** @param {string} name */
export function highlightDesk(name) {
  if (!_highlightMesh) return;
  const result = _highlightDesk(name, _highlightMesh, _deskLayout);
  if (result) _highlightedName = result;
}

export function clearHighlight() {
  if (_highlightMesh) {
    _clearHighlight(_highlightMesh);
  }
  _highlightedName = null;
}

/** @param {(name: string) => void} fn */
export function setCharacterClickHandler(fn) {
  _characterClickHandler = fn;
}

/** @returns {THREE.Scene | null} */
export function getScene() {
  return _scene;
}

/**
 * @param {string} animaName
 * @param {THREE.Object3D} object
 */
export function registerClickTarget(animaName, object) {
  _registerClickTarget(animaName, object, _meshToName);
}

/** @param {THREE.Object3D} object */
export function unregisterClickTarget(object) {
  _unregisterClickTarget(object, _meshToName);
}

/** @returns {{ cx: number, cz: number, hw: number, hd: number }[]} */
export function getObstacles() {
  const obstacles = [];
  const halfW = _floorWidth / 2;
  const halfD = _floorDepth / 2;

  for (const pos of Object.values(_deskLayout)) {
    obstacles.push({ cx: pos.x, cz: pos.z, hw: 0.8, hd: 0.5 });
  }

  for (const [px, pz] of [[-halfW+1, -halfD+1],[halfW-1,-halfD+1],[-halfW+1,halfD-1],[halfW-1,halfD-1]]) {
    obstacles.push({ cx: px, cz: pz, hw: 0.3, hd: 0.3 });
  }
  if (_floorWidth >= 14 && _floorDepth >= 10) {
    obstacles.push({ cx: 0, cz: 0, hw: 0.4, hd: 0.4 });
  }

  obstacles.push({ cx: -halfW + 0.5, cz: 0, hw: 0.5, hd: 0.25 });
  obstacles.push({ cx: halfW - 1.5, cz: halfD - 1.5, hw: 0.5, hd: 0.35 });

  return obstacles;
}

/** @returns {{ width: number, depth: number }} */
export function getFloorDimensions() {
  return { width: _floorWidth, depth: _floorDepth };
}

/** @param {(deltaTime: number, elapsedTime: number) => void} fn */
export function setCharacterUpdateHook(fn) {
  _characterUpdateFn = fn;
}
