// ── Office Layout ──────────────────────
// Organizational tree construction, desk placement, and hierarchy connectors.

import * as THREE from "three";
import { COLOR } from "./office-structure.js";

// ── Constants ──────────────────────

/** Spacing between desks in a row (X axis). */
const DESK_SPACING_X = 3.5;
/** Spacing between hierarchy levels (Z axis). */
const DESK_SPACING_Z = 4.0;
/** Minimum floor padding around desks. */
const FLOOR_PADDING = 4;

// ── Tree Construction ──────────────────────

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
 *
 * @param {Array<{name: string, role?: string, supervisor?: string}>} animas
 * @returns {TreeNode[]}
 */
export function buildOrgTree(animas) {
  /** @type {Map<string, TreeNode>} */
  const nodeMap = new Map();

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

  if (roots.length === 0) {
    return [...nodeMap.values()];
  }
  return roots;
}

/**
 * Compute desk positions from the organizational tree via BFS.
 *
 * @param {TreeNode[]} roots
 * @returns {Record<string, {x: number, y: number, z: number}>}
 */
export function computeTreeLayout(roots) {
  /** @type {Record<string, {x: number, y: number, z: number}>} */
  const layout = {};

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

  const totalDepth = levels.length;
  const startZ = -((totalDepth - 1) * DESK_SPACING_Z) / 2;

  for (let levelIdx = 0; levelIdx < levels.length; levelIdx++) {
    const level = levels[levelIdx];
    const z = startZ + levelIdx * DESK_SPACING_Z;
    const levelWidth = (level.length - 1) * DESK_SPACING_X;
    const startX = -levelWidth / 2;

    for (let i = 0; i < level.length; i++) {
      const node = level[i];
      layout[node.name] = { x: startX + i * DESK_SPACING_X, y: 0, z };
    }
  }

  return layout;
}

/**
 * Compute dynamic floor dimensions from desk layout.
 * @param {Record<string, {x: number, y: number, z: number}>} layout
 * @returns {{width: number, depth: number}}
 */
export function computeFloorDimensions(layout) {
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

// ── Desk Placement ──────────────────────

/**
 * Build all desks from the computed layout and add them to the scene.
 * @param {THREE.Scene} scene
 * @param {Record<string, {x: number, y: number, z: number}>} deskLayout
 * @param {Map<string, THREE.Group>} deskGroups
 * @param {Function} createDeskFn - (name, pos) => THREE.Group
 */
export function buildDesks(scene, deskLayout, deskGroups, createDeskFn) {
  for (const [name, pos] of Object.entries(deskLayout)) {
    const group = createDeskFn(name, pos);
    scene.add(group);
    deskGroups.set(name, group);
  }
}

/**
 * Draw organizational hierarchy connector lines between supervisor and subordinate desks.
 * @param {THREE.Scene} scene
 * @param {Array<{name: string, role?: string, supervisor?: string}>} animas
 * @param {Record<string, {x: number, y: number, z: number}>} deskLayout
 * @param {Set} disposables
 */
export function buildConnectors(scene, animas, deskLayout, disposables) {
  const LINE_W = 0.1;
  const Y = 0.03;
  const lineMat = new THREE.MeshBasicMaterial({
    color: COLOR.connector,
    transparent: true,
    opacity: 0.7,
  });
  disposables.add(lineMat);

  const addSegment = (x1, z1, x2, z2) => {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.sqrt(dx * dx + dz * dz);
    if (len < 0.001) return;
    const geo = new THREE.BoxGeometry(len, 0.01, LINE_W);
    disposables.add(geo);
    const mesh = new THREE.Mesh(geo, lineMat);
    mesh.position.set((x1 + x2) / 2, Y, (z1 + z2) / 2);
    mesh.rotation.y = -Math.atan2(dz, dx);
    scene.add(mesh);
  };

  const roots = buildOrgTree(animas);

  /** @param {TreeNode} node */
  const walkNode = (node) => {
    const parentPos = deskLayout[node.name];
    if (!parentPos) return;

    const kids = node.children.filter((c) => deskLayout[c.name]);
    if (kids.length > 0) {
      const childPositions = kids.map((c) => deskLayout[c.name]);

      const childZ = childPositions[0].z;
      const busZ = (parentPos.z + childZ) / 2;

      addSegment(parentPos.x, parentPos.z + 1.0, parentPos.x, busZ);

      const xs = [parentPos.x, ...childPositions.map((p) => p.x)];
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      if (minX < maxX) {
        addSegment(minX, busZ, maxX, busZ);
      }

      for (const cp of childPositions) {
        addSegment(cp.x, busZ, cp.x, cp.z - 1.0);
      }
    }

    for (const child of node.children) {
      walkNode(child);
    }
  };

  for (const root of roots) {
    walkNode(root);
  }
}
