// ── 3D Office Scene ──────────────────────
// Three.js-based isometric office for the AnimaWorks Workspace.
// Renders a modern open-plan office where AI Animas work at desks.
// Desks are dynamically placed based on organizational tree structure.
// This module owns the scene, camera, renderer, and static furniture.
// Characters are placed by character.js via getDesks().

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Constants ──────────────────────

const COLOR = {
  floor:       0xd4a373,
  wallSide:    0xe8e8e8,
  wallBack:    0xe0e0e0,
  windowGlass: 0x88bbdd,
  desk:        0x8b7355,
  deskLeg:     0x6b5335,
  monitor:     0x333333,
  monitorGlow: 0x4488cc,
  plantPot:    0x8b6914,
  plantLeaf:   0x4a8c3f,
  shelf:       0x9b7b55,
  book1:       0xc44e52,
  book2:       0x4e7cc4,
  book3:       0x52c44e,
  highlight:   0xffcc00,
  connector:   0x556677,
};

/** Spacing between desks in a row (X axis). */
const DESK_SPACING_X = 3.5;
/** Spacing between hierarchy levels (Z axis). */
const DESK_SPACING_Z = 4.0;
/** Minimum floor padding around desks. */
const FLOOR_PADDING = 4;

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

/** @type {THREE.Raycaster} */
let _raycaster = new THREE.Raycaster();

/** @type {THREE.Vector2} */
let _mouse = new THREE.Vector2();

/**
 * Dynamic desk layout computed from anima data.
 * @type {Record<string, {x: number, y: number, z: number}>}
 */
let _deskLayout = {};

/** Current floor dimensions (updated dynamically). */
let _floorWidth = 16;
let _floorDepth = 12;

/**
 * Maps anima name to the desk group mesh (for raycasting / highlighting).
 * @type {Map<string, THREE.Group>}
 */
const _deskGroups = new Map();

/**
 * Maps mesh UUID to anima name (for raycaster hit lookup).
 * @type {Map<string, string>}
 */
const _meshToName = new Map();

/** @type {THREE.Mesh | null} */
let _highlightMesh = null;

/** @type {string | null} */
let _highlightedName = null;

/** @type {((name: string) => void) | null} */
let _characterClickHandler = null;

/**
 * Tracks all disposable resources for cleanup.
 * @type {Set<THREE.BufferGeometry | THREE.Material | THREE.Texture>}
 */
const _disposables = new Set();

// ── Tree Layout Algorithm ──────────────────────

/**
 * @typedef {Object} TreeNode
 * @property {string} name
 * @property {string|null} role
 * @property {string|null} supervisor
 * @property {TreeNode[]} children
 */

/**
 * Build an organizational tree from anima data.
 * Animas with role "commander" or no supervisor are roots.
 * Others are placed under their supervisor.
 *
 * @param {Array<{name: string, role?: string, supervisor?: string}>} animas
 * @returns {TreeNode[]} Root nodes of the tree.
 */
function buildOrgTree(animas) {
  /** @type {Map<string, TreeNode>} */
  const nodeMap = new Map();

  // Create nodes for all animas
  for (const p of animas) {
    nodeMap.set(p.name, {
      name: p.name,
      role: p.role || null,
      supervisor: p.supervisor || null,
      children: [],
    });
  }

  /** @type {TreeNode[]} */
  const roots = [];

  // Build parent-child relationships
  for (const node of nodeMap.values()) {
    if (node.role === "commander" || !node.supervisor || !nodeMap.has(node.supervisor)) {
      roots.push(node);
    } else {
      const parent = nodeMap.get(node.supervisor);
      if (parent) {
        parent.children.push(node);
      }
    }
  }

  // If no roots found, treat all animas as roots (flat layout)
  if (roots.length === 0) {
    return [...nodeMap.values()];
  }

  return roots;
}

/**
 * Compute desk positions from the organizational tree.
 * Commander(s) at the top (negative Z), workers in rows below.
 * Each level is centered horizontally.
 *
 * @param {TreeNode[]} roots
 * @returns {Record<string, {x: number, y: number, z: number}>}
 */
function computeTreeLayout(roots) {
  /** @type {Record<string, {x: number, y: number, z: number}>} */
  const layout = {};

  // Flatten tree into levels using BFS
  /** @type {TreeNode[][]} */
  const levels = [];
  let currentLevel = [...roots];

  while (currentLevel.length > 0) {
    levels.push(currentLevel);
    /** @type {TreeNode[]} */
    const nextLevel = [];
    for (const node of currentLevel) {
      nextLevel.push(...node.children);
    }
    currentLevel = nextLevel;
  }

  // Compute the maximum width needed (for centering)
  const maxNodesInLevel = Math.max(...levels.map((l) => l.length));

  // Position each level
  const totalDepth = levels.length;
  const startZ = -((totalDepth - 1) * DESK_SPACING_Z) / 2;

  for (let levelIdx = 0; levelIdx < levels.length; levelIdx++) {
    const level = levels[levelIdx];
    const z = startZ + levelIdx * DESK_SPACING_Z;
    const levelWidth = (level.length - 1) * DESK_SPACING_X;
    const startX = -levelWidth / 2;

    for (let i = 0; i < level.length; i++) {
      const node = level[i];
      layout[node.name] = {
        x: startX + i * DESK_SPACING_X,
        y: 0,
        z,
      };
    }
  }

  return layout;
}

/**
 * Compute dynamic floor dimensions from desk layout.
 * @param {Record<string, {x: number, y: number, z: number}>} layout
 * @returns {{width: number, depth: number}}
 */
function computeFloorDimensions(layout) {
  const positions = Object.values(layout);
  if (positions.length === 0) {
    return { width: 16, depth: 12 };
  }

  let minX = Infinity, maxX = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (const pos of positions) {
    if (pos.x < minX) minX = pos.x;
    if (pos.x > maxX) maxX = pos.x;
    if (pos.z < minZ) minZ = pos.z;
    if (pos.z > maxZ) maxZ = pos.z;
  }

  const width = Math.max(12, (maxX - minX) + FLOOR_PADDING * 2);
  const depth = Math.max(10, (maxZ - minZ) + FLOOR_PADDING * 2);

  return { width, depth };
}

// ── Geometry / Material Helpers ──────────────────────

/**
 * Create a BoxGeometry and track it for disposal.
 * @param {number} w
 * @param {number} h
 * @param {number} d
 * @returns {THREE.BoxGeometry}
 */
function box(w, h, d) {
  const g = new THREE.BoxGeometry(w, h, d);
  _disposables.add(g);
  return g;
}

/**
 * Create a PlaneGeometry and track it for disposal.
 * @param {number} w
 * @param {number} h
 * @returns {THREE.PlaneGeometry}
 */
function plane(w, h) {
  const g = new THREE.PlaneGeometry(w, h);
  _disposables.add(g);
  return g;
}

/**
 * Create a CylinderGeometry and track it for disposal.
 * @param {number} rt - radius top
 * @param {number} rb - radius bottom
 * @param {number} h  - height
 * @param {number} [seg=16] - radial segments
 * @returns {THREE.CylinderGeometry}
 */
function cylinder(rt, rb, h, seg = 16) {
  const g = new THREE.CylinderGeometry(rt, rb, h, seg);
  _disposables.add(g);
  return g;
}

/**
 * Create a ConeGeometry and track it for disposal.
 * @param {number} r - radius
 * @param {number} h - height
 * @param {number} [seg=16] - radial segments
 * @returns {THREE.ConeGeometry}
 */
function cone(r, h, seg = 16) {
  const g = new THREE.ConeGeometry(r, h, seg);
  _disposables.add(g);
  return g;
}

/**
 * Create a MeshLambertMaterial and track it for disposal.
 * @param {THREE.MeshLambertMaterialParameters} params
 * @returns {THREE.MeshLambertMaterial}
 */
function lambert(params) {
  const m = new THREE.MeshLambertMaterial(params);
  _disposables.add(m);
  return m;
}

/**
 * Create a MeshPhongMaterial and track it for disposal.
 * @param {THREE.MeshPhongMaterialParameters} params
 * @returns {THREE.MeshPhongMaterial}
 */
function phong(params) {
  const m = new THREE.MeshPhongMaterial(params);
  _disposables.add(m);
  return m;
}

// ── Scene Construction ──────────────────────

/**
 * Build the floor plane.
 * @param {THREE.Scene} scene
 */
function buildFloor(scene) {
  const geo = plane(_floorWidth, _floorDepth);
  const mat = lambert({ color: COLOR.floor });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  scene.add(mesh);
}

/**
 * Build side and back walls with windows.
 * @param {THREE.Scene} scene
 */
function buildWalls(scene) {
  const wallMat = lambert({ color: COLOR.wallSide });
  const wallHeight = 3;
  const halfFloorW = _floorWidth / 2;
  const halfFloorD = _floorDepth / 2;

  // Left wall
  const leftGeo = box(0.1, wallHeight, _floorDepth);
  const leftWall = new THREE.Mesh(leftGeo, wallMat);
  leftWall.position.set(-halfFloorW, wallHeight / 2, 0);
  leftWall.castShadow = true;
  leftWall.receiveShadow = true;
  scene.add(leftWall);

  // Right wall
  const rightWall = new THREE.Mesh(leftGeo, wallMat);
  rightWall.position.set(halfFloorW, wallHeight / 2, 0);
  rightWall.castShadow = true;
  rightWall.receiveShadow = true;
  scene.add(rightWall);

  // Back wall — solid sections on left and right, window in center
  const backMat = lambert({ color: COLOR.wallBack });
  const pillarWidth = 2;

  // Left pillar
  const pillarGeo = box(pillarWidth, wallHeight, 0.1);
  const leftPillar = new THREE.Mesh(pillarGeo, backMat);
  leftPillar.position.set(-halfFloorW + pillarWidth / 2, wallHeight / 2, -halfFloorD);
  leftPillar.castShadow = true;
  scene.add(leftPillar);

  // Right pillar
  const rightPillar = new THREE.Mesh(pillarGeo, backMat);
  rightPillar.position.set(halfFloorW - pillarWidth / 2, wallHeight / 2, -halfFloorD);
  rightPillar.castShadow = true;
  scene.add(rightPillar);

  // Window frame top
  const windowWidth = _floorWidth - pillarWidth * 2;
  const frameThickness = 0.3;
  const frameGeo = box(windowWidth, frameThickness, 0.1);
  const topFrame = new THREE.Mesh(frameGeo, backMat);
  topFrame.position.set(0, wallHeight - frameThickness / 2, -halfFloorD);
  scene.add(topFrame);

  // Window frame bottom (sill)
  const sillHeight = 0.8;
  const sillGeo = box(windowWidth, sillHeight, 0.15);
  const sill = new THREE.Mesh(sillGeo, backMat);
  sill.position.set(0, sillHeight / 2, -halfFloorD);
  scene.add(sill);

  // Window glass (semi-transparent)
  const glassHeight = wallHeight - sillHeight - frameThickness;
  const glassGeo = plane(windowWidth, glassHeight);
  const glassMat = phong({
    color: COLOR.windowGlass,
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
  });
  const glass = new THREE.Mesh(glassGeo, glassMat);
  glass.position.set(0, sillHeight + glassHeight / 2, -halfFloorD + 0.01);
  scene.add(glass);

  // Window dividers (vertical mullions)
  const mullionCount = Math.max(3, Math.floor(windowWidth / 2.5));
  const mullionGeo = box(0.05, glassHeight, 0.05);
  const mullionMat = lambert({ color: 0xcccccc });
  for (let i = 1; i < mullionCount; i++) {
    const xPos = -windowWidth / 2 + (windowWidth / mullionCount) * i;
    const mullion = new THREE.Mesh(mullionGeo, mullionMat);
    mullion.position.set(xPos, sillHeight + glassHeight / 2, -halfFloorD);
    scene.add(mullion);
  }
}

/**
 * Create a single desk group with surface, legs, and monitor.
 * @param {string} animaName
 * @param {{x: number, y: number, z: number}} pos
 * @returns {THREE.Group}
 */
function createDesk(animaName, pos) {
  const group = new THREE.Group();
  group.position.set(pos.x, 0, pos.z);

  const deskMat = lambert({ color: COLOR.desk });
  const legMat = lambert({ color: COLOR.deskLeg });

  // Desk surface
  const surfaceGeo = box(1.2, 0.05, 0.6);
  const surface = new THREE.Mesh(surfaceGeo, deskMat);
  surface.position.y = 0.4;
  surface.castShadow = true;
  surface.receiveShadow = true;
  group.add(surface);

  // Desk legs (4) — from floor (y=0) to underside of surface
  const legH = 0.4 - 0.05 / 2;  // 0.725
  const legGeo = box(0.03, legH, 0.03);
  const legOffsets = [
    { x: -0.55, z: -0.25 },
    { x:  0.55, z: -0.25 },
    { x: -0.55, z:  0.25 },
    { x:  0.55, z:  0.25 },
  ];
  for (const off of legOffsets) {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.set(off.x, legH / 2, off.z);
    leg.castShadow = true;
    group.add(leg);
  }

  // Monitor
  const monitorMat = phong({
    color: COLOR.monitor,
    emissive: COLOR.monitorGlow,
    emissiveIntensity: 0.15,
  });
  const screenGeo = box(0.4, 0.3, 0.02);
  const screen = new THREE.Mesh(screenGeo, monitorMat);
  screen.position.set(0, 0.4 + 0.05 / 2 + 0.15 + 0.02, 0.15);
  screen.castShadow = true;
  group.add(screen);

  // Monitor stand
  const standGeo = box(0.03, 0.15, 0.03);
  const stand = new THREE.Mesh(standGeo, legMat);
  stand.position.set(0, 0.4 + 0.05 / 2 + 0.075, 0.15);
  group.add(stand);

  // Name label
  const label = createNameLabel(animaName);
  if (label) {
    label.position.set(0, 0.02, 0.5);
    label.rotation.x = -Math.PI / 2;
    group.add(label);
  }

  // Register all child meshes for raycasting
  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      _meshToName.set(child.uuid, animaName);
    }
  });

  return group;
}

/**
 * Render an anima name to a CanvasTexture and return a label mesh.
 * @param {string} name
 * @returns {THREE.Mesh | null}
 */
function createNameLabel(name) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  canvas.width = 256;
  canvas.height = 64;

  ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
  roundRect(ctx, 4, 4, 248, 56, 10);
  ctx.fill();

  ctx.fillStyle = "#333333";
  ctx.font = "bold 28px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(name, 128, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  _disposables.add(texture);

  const mat = lambert({
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
  });
  const geo = plane(0.8, 0.2);
  const mesh = new THREE.Mesh(geo, mat);
  return mesh;
}

/**
 * Draw a rounded rectangle path on a 2D context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} r - corner radius
 */
function roundRect(ctx, x, y, w, h, r) {
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

/**
 * Build all desks dynamically from the computed layout and add them to the scene.
 * @param {THREE.Scene} scene
 */
function buildDesks(scene) {
  for (const [name, pos] of Object.entries(_deskLayout)) {
    const group = createDesk(name, pos);
    scene.add(group);
    _deskGroups.set(name, group);
  }
}

/**
 * Draw organizational hierarchy connector lines between supervisor and subordinate desks.
 * Walks the same org tree used for layout to guarantee consistency.
 * @param {THREE.Scene} scene
 * @param {Array<{name: string, role?: string, supervisor?: string}>} animas
 */
function buildConnectors(scene, animas) {
  const LINE_W = 0.1;
  const Y = 0.03;
  const lineMat = new THREE.MeshBasicMaterial({
    color: COLOR.connector,
    transparent: true,
    opacity: 0.7,
  });
  _disposables.add(lineMat);

  // Draw a flat box from (x1,z1) to (x2,z2) on the floor
  const addSegment = (x1, z1, x2, z2) => {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.sqrt(dx * dx + dz * dz);
    if (len < 0.001) return;
    const geo = new THREE.BoxGeometry(len, 0.01, LINE_W);
    _disposables.add(geo);
    const mesh = new THREE.Mesh(geo, lineMat);
    mesh.position.set((x1 + x2) / 2, Y, (z1 + z2) / 2);
    mesh.rotation.y = -Math.atan2(dz, dx);
    scene.add(mesh);
  };

  // Rebuild the same tree used for layout, then walk it recursively
  const roots = buildOrgTree(animas);

  /** @param {TreeNode} node */
  const walkNode = (node) => {
    const parentPos = _deskLayout[node.name];
    if (!parentPos) return;

    // Collect children that have desk positions
    const kids = node.children.filter((c) => _deskLayout[c.name]);
    if (kids.length > 0) {
      const childPositions = kids.map((c) => _deskLayout[c.name]);

      // Bus Z = midpoint between parent and children row
      const childZ = childPositions[0].z;
      const busZ = (parentPos.z + childZ) / 2;

      // 1. Vertical: parent desk → bus
      addSegment(parentPos.x, parentPos.z + 1.0, parentPos.x, busZ);

      // 2. Horizontal bus spanning parent and all children
      const xs = [parentPos.x, ...childPositions.map((p) => p.x)];
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      if (minX < maxX) {
        addSegment(minX, busZ, maxX, busZ);
      }

      // 3. Vertical: bus → each child desk
      for (const cp of childPositions) {
        addSegment(cp.x, busZ, cp.x, cp.z - 1.0);
      }
    }

    // Recurse into children
    for (const child of node.children) {
      walkNode(child);
    }
  };

  for (const root of roots) {
    walkNode(root);
  }
}

/**
 * Build decorative elements: plants around the office.
 * Positions are computed relative to current floor size.
 * @param {THREE.Scene} scene
 */
function buildDecorations(scene) {
  const potMat = lambert({ color: COLOR.plantPot });
  const leafMat = lambert({ color: COLOR.plantLeaf });

  const halfW = _floorWidth / 2 - 1;
  const halfD = _floorDepth / 2 - 1;

  // Corner plants
  buildPlant(scene, -halfW, -halfD, potMat, leafMat, 0.2, 0.5);
  buildPlant(scene,  halfW, -halfD, potMat, leafMat, 0.2, 0.5);
  buildPlant(scene, -halfW,  halfD, potMat, leafMat, 0.15, 0.4);
  buildPlant(scene,  halfW,  halfD, potMat, leafMat, 0.15, 0.4);

  // A center-ish plant if the office is large enough
  if (_floorWidth >= 14 && _floorDepth >= 10) {
//    buildPlant(scene, 0, 0, potMat, leafMat, 0.25, 0.6);
  }
}

/**
 * Create a potted plant at the given position.
 * @param {THREE.Scene} scene
 * @param {number} x
 * @param {number} z
 * @param {THREE.Material} potMat
 * @param {THREE.Material} leafMat
 * @param {number} potRadius
 * @param {number} plantHeight
 */
function buildPlant(scene, x, z, potMat, leafMat, potRadius, plantHeight) {
  // Pot
  const potGeo = cylinder(potRadius, potRadius * 0.8, potRadius * 1.2, 12);
  const pot = new THREE.Mesh(potGeo, potMat);
  pot.position.set(x, potRadius * 0.6, z);
  pot.castShadow = true;
  scene.add(pot);

  // Foliage (cone)
  const foliageGeo = cone(potRadius * 1.5, plantHeight, 12);
  const foliage = new THREE.Mesh(foliageGeo, leafMat);
  foliage.position.set(x, potRadius * 1.2 + plantHeight / 2, z);
  foliage.castShadow = true;
  scene.add(foliage);
}

/**
 * Build a bookshelf on the left wall.
 * @param {THREE.Scene} scene
 */
function buildBookshelf(scene) {
  const halfW = _floorWidth / 2;
  const x = -halfW + 0.5;
  const z = 0;

  const shelfMat = lambert({ color: COLOR.shelf });

  // Shelf frame
  const frame = new THREE.Mesh(box(0.8, 1.8, 0.3), shelfMat);
  frame.position.set(x, 0.9, z);
  frame.castShadow = true;
  scene.add(frame);

  // Books (3 rows, different colours)
  const bookColors = [COLOR.book1, COLOR.book2, COLOR.book3];
  for (let row = 0; row < 3; row++) {
    const book = new THREE.Mesh(
      box(0.6, 0.15, 0.2),
      lambert({ color: bookColors[row] }),
    );
    book.position.set(x, 0.4 + row * 0.5, z);
    scene.add(book);
  }
}

/**
 * Build a kitchen corner with counter, coffee machine, and water server.
 * @param {THREE.Scene} scene
 */
function buildKitchenCorner(scene) {
  const halfW = _floorWidth / 2;
  const halfD = _floorDepth / 2;
  const x = halfW - 1.5;
  const z = halfD - 1.5;

  // Counter
  const counterMat = lambert({ color: 0xcccccc });
  const counter = new THREE.Mesh(box(0.8, 0.85, 0.4), counterMat);
  counter.position.set(x, 0.425, z);
  counter.castShadow = true;
  scene.add(counter);

  // Coffee machine
  const machineMat = lambert({ color: 0x444444 });
  const machine = new THREE.Mesh(box(0.2, 0.3, 0.2), machineMat);
  machine.position.set(x - 0.15, 0.85 + 0.15, z);
  machine.castShadow = true;
  scene.add(machine);

  // Water server
  const waterMat = lambert({ color: 0x88aacc });
  const water = new THREE.Mesh(cylinder(0.1, 0.1, 0.5, 12), waterMat);
  water.position.set(x + 0.25, 0.85 + 0.25, z);
  water.castShadow = true;
  scene.add(water);
}

/**
 * Set up ambient and directional lighting.
 * @param {THREE.Scene} scene
 */
function buildLighting(scene) {
  // Ambient fill
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  // Main directional light with shadows
  const directional = new THREE.DirectionalLight(0xffffff, 0.8);
  directional.position.set(5, 10, 5);
  directional.castShadow = true;

  // Shadow camera covers the whole floor
  const shadowExtent = Math.max(_floorWidth, _floorDepth) / 2 + 2;
  directional.shadow.mapSize.width = 2048;
  directional.shadow.mapSize.height = 2048;
  directional.shadow.camera.left = -shadowExtent;
  directional.shadow.camera.right = shadowExtent;
  directional.shadow.camera.top = shadowExtent;
  directional.shadow.camera.bottom = -shadowExtent;
  directional.shadow.camera.near = 0.5;
  directional.shadow.camera.far = 30;
  directional.shadow.bias = -0.001;
  scene.add(directional);

  // Soft fill from the opposite side
  const fill = new THREE.DirectionalLight(0xffffff, 0.3);
  fill.position.set(-5, 8, -3);
  scene.add(fill);
}

// ── Highlight System ──────────────────────

/**
 * Create the shared highlight indicator mesh (a flat ring under the desk).
 * @returns {THREE.Mesh}
 */
function createHighlightMesh() {
  const geo = new THREE.RingGeometry(0.6, 0.8, 32);
  _disposables.add(geo);

  const mat = phong({
    color: COLOR.highlight,
    emissive: COLOR.highlight,
    emissiveIntensity: 0.5,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.02;
  mesh.visible = false;
  return mesh;
}

// ── Camera Setup ──────────────────────

/**
 * Create an orthographic camera with isometric-like perspective.
 * Frustum is scaled to fit the office.
 * @param {number} aspect - container width / height
 * @returns {THREE.OrthographicCamera}
 */
function createCamera(aspect) {
  const frustum = Math.max(8, Math.max(_floorWidth, _floorDepth) / 2 + 2);
  const camera = new THREE.OrthographicCamera(
    -frustum * aspect,
     frustum * aspect,
     frustum,
    -frustum,
    0.1,
    100,
  );

  // Isometric-ish viewing angle (~45 degrees down)
  camera.position.set(10, 12, 10);
  camera.lookAt(0, 0, 0);
  return camera;
}

/**
 * Update camera frustum on container resize.
 * @param {number} width
 * @param {number} height
 */
function updateCameraFrustum(width, height) {
  if (!_camera || !_renderer) return;

  const aspect = width / height;
  const frustum = Math.max(8, Math.max(_floorWidth, _floorDepth) / 2 + 2);
  _camera.left = -frustum * aspect;
  _camera.right = frustum * aspect;
  _camera.top = frustum;
  _camera.bottom = -frustum;
  _camera.updateProjectionMatrix();
  _renderer.setSize(width, height);
}

// ── Raycasting / Click ──────────────────────

/**
 * Handle pointer click for desk/character selection.
 * @param {MouseEvent} event
 */
function onPointerClick(event) {
  if (!_renderer || !_camera || !_scene) return;

  const rect = _renderer.domElement.getBoundingClientRect();
  _mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  _mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  _raycaster.setFromCamera(_mouse, _camera);

  const intersects = _raycaster.intersectObjects(_scene.children, true);

  for (const hit of intersects) {
    const name = _meshToName.get(hit.object.uuid);
    if (name) {
      if (_characterClickHandler) {
        _characterClickHandler(name);
      }
      return;
    }
  }
}

// ── Character Update Hook ──────────────────────

/** @type {((dt: number, elapsed: number) => void) | null} */
let _characterUpdateFn = null;

/**
 * Register a callback to update characters each frame.
 * Called by app.js after initCharacters().
 * @param {(deltaTime: number, elapsedTime: number) => void} fn
 */
export function setCharacterUpdateHook(fn) {
  _characterUpdateFn = fn;
}

// ── Render Loop ──────────────────────

const _clock = new THREE.Clock();

/**
 * Main render loop. Renders the scene, updates controls, and animates characters each frame.
 */
function renderLoop() {
  _animationFrameId = requestAnimationFrame(renderLoop);

  const dt = _clock.getDelta();
  const elapsed = _clock.getElapsedTime();

  // Update character animations
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

// ── Public API ──────────────────────

/**
 * Initialize the 3D office scene inside the given container element.
 * Desk layout is computed dynamically from the animas array.
 *
 * @param {HTMLElement} container - The DOM element to host the Three.js canvas
 * @param {Array<{name: string, role?: string, supervisor?: string}>} [animas] - Anima data for dynamic layout
 */
export function initOffice(container, animas = []) {
  if (_renderer) {
    // Already initialized — dispose first
    disposeOffice();
  }

  _container = container;
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  // Compute dynamic desk layout from anima data
  if (animas.length > 0) {
    const roots = buildOrgTree(animas);
    _deskLayout = computeTreeLayout(roots);
  } else {
    _deskLayout = {};
  }

  // Compute floor dimensions
  const dims = computeFloorDimensions(_deskLayout);
  _floorWidth = dims.width;
  _floorDepth = dims.depth;

  // Scene
  _scene = new THREE.Scene();
  _scene.background = new THREE.Color(0xf0ece3);

  // Camera
  _camera = createCamera(width / height);

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
  _controls.maxPolarAngle = Math.PI / 2.5;   // prevent looking from below
  _controls.minPolarAngle = Math.PI / 6;     // prevent looking from directly above
  _controls.target.set(0, 0, 0);
  _controls.update();

  // Build scene
  buildFloor(_scene);
  buildWalls(_scene);
  buildDesks(_scene);
  buildConnectors(_scene, animas);
  buildDecorations(_scene);
  buildBookshelf(_scene);
  buildKitchenCorner(_scene);
  buildLighting(_scene);

  // Highlight mesh (shared, repositioned on highlight)
  _highlightMesh = createHighlightMesh();
  _scene.add(_highlightMesh);

  // Events
  _renderer.domElement.addEventListener("click", onPointerClick);

  // Resize handling via ResizeObserver (preferred) or window resize fallback
  if (typeof ResizeObserver !== "undefined") {
    _resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0) {
          updateCameraFrustum(w, h);
        }
      }
    });
    _resizeObserver.observe(container);
  } else {
    window.addEventListener("resize", _onWindowResize);
  }

  // Start render loop
  renderLoop();
}

/**
 * Fallback resize handler when ResizeObserver is unavailable.
 */
function _onWindowResize() {
  if (!_container) return;
  updateCameraFrustum(_container.clientWidth, _container.clientHeight);
}

/**
 * Dispose of all Three.js resources and stop the render loop.
 * Must be called when the office view is unmounted or replaced.
 */
export function disposeOffice() {
  // Stop render loop
  if (_animationFrameId !== null) {
    cancelAnimationFrame(_animationFrameId);
    _animationFrameId = null;
  }

  // Remove event listeners
  if (_renderer) {
    _renderer.domElement.removeEventListener("click", onPointerClick);
  }

  // Disconnect resize observer
  if (_resizeObserver) {
    _resizeObserver.disconnect();
    _resizeObserver = null;
  } else {
    window.removeEventListener("resize", _onWindowResize);
  }

  // Dispose controls
  if (_controls) {
    _controls.dispose();
    _controls = null;
  }

  // Dispose all tracked geometries, materials, textures
  for (const resource of _disposables) {
    if (typeof resource.dispose === "function") {
      resource.dispose();
    }
  }
  _disposables.clear();

  // Dispose renderer and remove canvas
  if (_renderer) {
    _renderer.dispose();
    if (_renderer.domElement.parentElement) {
      _renderer.domElement.parentElement.removeChild(_renderer.domElement);
    }
    _renderer = null;
  }

  // Clear references
  _scene = null;
  _camera = null;
  _container = null;
  _highlightMesh = null;
  _highlightedName = null;
  _deskLayout = {};
  _deskGroups.clear();
  _meshToName.clear();
}

/**
 * Return desk positions as a plain object keyed by anima name.
 * Character.js uses this to know where to place 3D characters.
 *
 * @returns {Record<string, {x: number, y: number, z: number}>}
 */
export function getDesks() {
  /** @type {Record<string, {x: number, y: number, z: number}>} */
  const result = {};
  for (const [name, pos] of Object.entries(_deskLayout)) {
    result[name] = { x: pos.x, y: pos.y, z: pos.z };
  }
  return result;
}

/**
 * Highlight a specific desk by showing a glowing ring underneath it.
 * Replaces any existing highlight.
 *
 * @param {string} name - Anima name whose desk to highlight
 */
export function highlightDesk(name) {
  if (!_highlightMesh) return;

  const pos = _deskLayout[name];
  if (!pos) {
    console.warn(`[office3d] Unknown desk name: "${name}"`);
    return;
  }

  _highlightMesh.position.set(pos.x, 0.02, pos.z);
  _highlightMesh.visible = true;
  _highlightedName = name;
}

/**
 * Remove any active desk highlight.
 */
export function clearHighlight() {
  if (_highlightMesh) {
    _highlightMesh.visible = false;
  }
  _highlightedName = null;
}

/**
 * Register a callback invoked when a desk or character mesh is clicked.
 * The callback receives the anima name as its sole argument.
 *
 * @param {(name: string) => void} fn - Click handler
 */
export function setCharacterClickHandler(fn) {
  _characterClickHandler = fn;
}

/**
 * Get the Three.js scene reference. Useful for character.js to add meshes.
 * Returns null if the office has not been initialized.
 *
 * @returns {THREE.Scene | null}
 */
export function getScene() {
  return _scene;
}

/**
 * Register additional meshes for raycasting (e.g., character meshes added by character.js).
 *
 * @param {string} animaName - The anima this mesh represents
 * @param {THREE.Object3D} object - The mesh or group to register
 */
export function registerClickTarget(animaName, object) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      _meshToName.set(child.uuid, animaName);
    }
  });
}

/**
 * Unregister meshes from raycasting (e.g., when a character is removed).
 *
 * @param {THREE.Object3D} object - The mesh or group to unregister
 */
export function unregisterClickTarget(object) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      _meshToName.delete(child.uuid);
    }
  });
}

/**
 * Return obstacle data for navigation grid construction.
 * Each obstacle is { cx, cz, hw, hd } (center, half-width, half-depth).
 * @returns {{ cx: number, cz: number, hw: number, hd: number }[]}
 */
export function getObstacles() {
  const obstacles = [];
  const halfW = _floorWidth / 2;
  const halfD = _floorDepth / 2;

  // Desks
  for (const pos of Object.values(_deskLayout)) {
    obstacles.push({ cx: pos.x, cz: pos.z, hw: 0.8, hd: 0.5 });
  }

  // Corner plants
  for (const [px, pz] of [[-halfW+1, -halfD+1],[halfW-1,-halfD+1],[-halfW+1,halfD-1],[halfW-1,halfD-1]]) {
    obstacles.push({ cx: px, cz: pz, hw: 0.3, hd: 0.3 });
  }
  // Center plant
  if (_floorWidth >= 14 && _floorDepth >= 10) {
    obstacles.push({ cx: 0, cz: 0, hw: 0.4, hd: 0.4 });
  }

  // Bookshelf
  obstacles.push({ cx: -halfW + 0.5, cz: 0, hw: 0.5, hd: 0.25 });

  // Kitchen corner
  obstacles.push({ cx: halfW - 1.5, cz: halfD - 1.5, hw: 0.5, hd: 0.35 });

  return obstacles;
}

/**
 * Return the floor dimensions.
 * @returns {{ width: number, depth: number }}
 */
export function getFloorDimensions() {
  return { width: _floorWidth, depth: _floorDepth };
}
