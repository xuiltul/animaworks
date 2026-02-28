// ── Office Furniture ──────────────────────
// Geometry/material helpers, desk builder, and decorative furniture.

import * as THREE from "three";
import { COLOR } from "./office-structure.js";

// ── Geometry / Material Helper Factory ──────────────────────

/**
 * Create disposal-tracked geometry and material helper functions.
 * @param {Set<THREE.BufferGeometry | THREE.Material | THREE.Texture>} disposables
 * @returns {{box: Function, plane: Function, cylinder: Function, cone: Function, lambert: Function, phong: Function}}
 */
export function createHelpers(disposables) {
  const box = (w, h, d) => {
    const g = new THREE.BoxGeometry(w, h, d);
    disposables.add(g);
    return g;
  };
  const plane = (w, h) => {
    const g = new THREE.PlaneGeometry(w, h);
    disposables.add(g);
    return g;
  };
  const cylinder = (rt, rb, h, seg = 16) => {
    const g = new THREE.CylinderGeometry(rt, rb, h, seg);
    disposables.add(g);
    return g;
  };
  const cone = (r, h, seg = 16) => {
    const g = new THREE.ConeGeometry(r, h, seg);
    disposables.add(g);
    return g;
  };
  const lambert = (params) => {
    const m = new THREE.MeshLambertMaterial(params);
    disposables.add(m);
    return m;
  };
  const phong = (params) => {
    const m = new THREE.MeshPhongMaterial(params);
    disposables.add(m);
    return m;
  };

  return { box, plane, cylinder, cone, lambert, phong };
}

// ── Desk Builder ──────────────────────

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
 * Render an anima name to a CanvasTexture and return a label mesh.
 * @param {string} name
 * @param {Set} disposables
 * @param {{plane: Function, lambert: Function}} helpers
 * @returns {THREE.Mesh | null}
 */
function createNameLabel(name, disposables, helpers) {
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
  disposables.add(texture);

  const mat = helpers.lambert({
    map: texture,
    transparent: true,
    side: THREE.DoubleSide,
  });
  const geo = helpers.plane(0.8, 0.2);
  return new THREE.Mesh(geo, mat);
}

/**
 * Create a single desk group with surface, legs, and monitor.
 * @param {string} animaName
 * @param {{x: number, y: number, z: number}} pos
 * @param {Set} disposables
 * @param {Map<string, string>} meshToName
 * @param {{box: Function, plane: Function, lambert: Function, phong: Function}} helpers
 * @returns {THREE.Group}
 */
export function createDesk(animaName, pos, disposables, meshToName, helpers) {
  const group = new THREE.Group();
  group.position.set(pos.x, 0, pos.z);

  const deskMat = helpers.lambert({ color: COLOR.desk });
  const legMat = helpers.lambert({ color: COLOR.deskLeg });

  // Desk surface
  const surfaceGeo = helpers.box(1.2, 0.05, 0.6);
  const surface = new THREE.Mesh(surfaceGeo, deskMat);
  surface.position.y = 0.4;
  surface.castShadow = true;
  surface.receiveShadow = true;
  group.add(surface);

  // Desk legs (4)
  const legH = 0.4 - 0.05 / 2;
  const legGeo = helpers.box(0.03, legH, 0.03);
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
  const monitorMat = helpers.phong({
    color: COLOR.monitor,
    emissive: COLOR.monitorGlow,
    emissiveIntensity: 0.15,
  });
  const screenGeo = helpers.box(0.4, 0.3, 0.02);
  const screen = new THREE.Mesh(screenGeo, monitorMat);
  screen.position.set(0, 0.4 + 0.05 / 2 + 0.15 + 0.02, 0.15);
  screen.castShadow = true;
  group.add(screen);

  // Monitor stand
  const standGeo = helpers.box(0.03, 0.15, 0.03);
  const stand = new THREE.Mesh(standGeo, legMat);
  stand.position.set(0, 0.4 + 0.05 / 2 + 0.075, 0.15);
  group.add(stand);

  // Name label
  const label = createNameLabel(animaName, disposables, helpers);
  if (label) {
    label.position.set(0, 0.02, 0.5);
    label.rotation.x = -Math.PI / 2;
    group.add(label);
  }

  // Register all child meshes for raycasting
  group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      meshToName.set(child.uuid, animaName);
    }
  });

  return group;
}

// ── Decorations ──────────────────────

/**
 * Create a potted plant at the given position.
 * @param {THREE.Scene} scene
 * @param {number} x
 * @param {number} z
 * @param {THREE.Material} potMat
 * @param {THREE.Material} leafMat
 * @param {number} potRadius
 * @param {number} plantHeight
 * @param {{cylinder: Function, cone: Function}} helpers
 */
function buildPlant(scene, x, z, potMat, leafMat, potRadius, plantHeight, helpers) {
  const potGeo = helpers.cylinder(potRadius, potRadius * 0.8, potRadius * 1.2, 12);
  const pot = new THREE.Mesh(potGeo, potMat);
  pot.position.set(x, potRadius * 0.6, z);
  pot.castShadow = true;
  scene.add(pot);

  const foliageGeo = helpers.cone(potRadius * 1.5, plantHeight, 12);
  const foliage = new THREE.Mesh(foliageGeo, leafMat);
  foliage.position.set(x, potRadius * 1.2 + plantHeight / 2, z);
  foliage.castShadow = true;
  scene.add(foliage);
}

/**
 * Build decorative plants around the office.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {{cylinder: Function, cone: Function, lambert: Function}} helpers
 */
export function buildDecorations(scene, floorWidth, floorDepth, helpers) {
  const potMat = helpers.lambert({ color: COLOR.plantPot });
  const leafMat = helpers.lambert({ color: COLOR.plantLeaf });

  const halfW = floorWidth / 2 - 1;
  const halfD = floorDepth / 2 - 1;

  buildPlant(scene, -halfW, -halfD, potMat, leafMat, 0.2, 0.5, helpers);
  buildPlant(scene,  halfW, -halfD, potMat, leafMat, 0.2, 0.5, helpers);
  buildPlant(scene, -halfW,  halfD, potMat, leafMat, 0.15, 0.4, helpers);
  buildPlant(scene,  halfW,  halfD, potMat, leafMat, 0.15, 0.4, helpers);
}

/**
 * Build a bookshelf on the left wall.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {{box: Function, lambert: Function}} helpers
 */
export function buildBookshelf(scene, floorWidth, helpers) {
  const halfW = floorWidth / 2;
  const x = -halfW + 0.5;
  const z = 0;

  const shelfMat = helpers.lambert({ color: COLOR.shelf });

  const frame = new THREE.Mesh(helpers.box(0.8, 1.8, 0.3), shelfMat);
  frame.position.set(x, 0.9, z);
  frame.castShadow = true;
  scene.add(frame);

  const bookColors = [COLOR.book1, COLOR.book2, COLOR.book3];
  for (let row = 0; row < 3; row++) {
    const book = new THREE.Mesh(
      helpers.box(0.6, 0.15, 0.2),
      helpers.lambert({ color: bookColors[row] }),
    );
    book.position.set(x, 0.4 + row * 0.5, z);
    scene.add(book);
  }
}

/**
 * Build a kitchen corner with counter, coffee machine, and water server.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {{box: Function, cylinder: Function, lambert: Function}} helpers
 */
export function buildKitchenCorner(scene, floorWidth, floorDepth, helpers) {
  const halfW = floorWidth / 2;
  const halfD = floorDepth / 2;
  const x = halfW - 1.5;
  const z = halfD - 1.5;

  const counterMat = helpers.lambert({ color: 0xcccccc });
  const counter = new THREE.Mesh(helpers.box(0.8, 0.85, 0.4), counterMat);
  counter.position.set(x, 0.425, z);
  counter.castShadow = true;
  scene.add(counter);

  const machineMat = helpers.lambert({ color: 0x444444 });
  const machine = new THREE.Mesh(helpers.box(0.2, 0.3, 0.2), machineMat);
  machine.position.set(x - 0.15, 0.85 + 0.15, z);
  machine.castShadow = true;
  scene.add(machine);

  const waterMat = helpers.lambert({ color: 0x88aacc });
  const water = new THREE.Mesh(helpers.cylinder(0.1, 0.1, 0.5, 12), waterMat);
  water.position.set(x + 0.25, 0.85 + 0.25, z);
  water.castShadow = true;
  scene.add(water);
}
