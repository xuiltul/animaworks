// ── Navigation Module ──────────────────────
// Grid-based A* pathfinding for character movement in the 3D office.
// Converts between world coordinates and grid cells, builds an obstacle
// map, and finds smoothed paths using Catmull-Rom interpolation.

import * as THREE from "three";

// ── Constants ──────────────────────

/** Size of each navigation grid cell in world units. */
const GRID_CELL_SIZE = 0.5;

/** Cost multiplier for diagonal movement (sqrt(2)). */
const SQRT2 = Math.SQRT2;

// ── Coordinate Conversion ──────────────────────

/**
 * Convert world coordinates to grid cell indices.
 * The grid origin is at the top-left corner of the floor
 * (world coordinate -halfW, -halfD).
 *
 * @param {number} wx - World X
 * @param {number} wz - World Z
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {{ gx: number, gz: number }}
 */
export function worldToGrid(wx, wz, floorWidth, floorDepth) {
  const halfW = floorWidth / 2;
  const halfD = floorDepth / 2;
  const gx = Math.floor((wx + halfW) / GRID_CELL_SIZE);
  const gz = Math.floor((wz + halfD) / GRID_CELL_SIZE);
  return { gx, gz };
}

/**
 * Convert grid cell indices to world coordinates (cell centre).
 *
 * @param {number} gx
 * @param {number} gz
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {{ wx: number, wz: number }}
 */
export function gridToWorld(gx, gz, floorWidth, floorDepth) {
  const halfW = floorWidth / 2;
  const halfD = floorDepth / 2;
  const wx = gx * GRID_CELL_SIZE + GRID_CELL_SIZE / 2 - halfW;
  const wz = gz * GRID_CELL_SIZE + GRID_CELL_SIZE / 2 - halfD;
  return { wx, wz };
}

// ── Grid Construction ──────────────────────

/**
 * @typedef {Object} NavGrid
 * @property {Uint8Array} grid - 0 = walkable, 1 = obstacle
 * @property {number} cols
 * @property {number} rows
 */

/**
 * Create a navigation grid from floor dimensions and obstacle data.
 * Marks floor boundary (outer 1 cell) and all obstacles as blocked.
 *
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {{ cx: number, cz: number, hw: number, hd: number }[]} obstacles
 * @returns {NavGrid}
 */
export function createNavGrid(floorWidth, floorDepth, obstacles) {
  const cols = Math.ceil(floorWidth / GRID_CELL_SIZE);
  const rows = Math.ceil(floorDepth / GRID_CELL_SIZE);
  const grid = new Uint8Array(cols * rows);

  // Block outer boundary (1 cell border = wall)
  for (let gz = 0; gz < rows; gz++) {
    for (let gx = 0; gx < cols; gx++) {
      if (gx === 0 || gx === cols - 1 || gz === 0 || gz === rows - 1) {
        grid[gz * cols + gx] = 1;
      }
    }
  }

  // Block obstacles
  for (const obs of obstacles) {
    const minW = obs.cx - obs.hw;
    const maxW = obs.cx + obs.hw;
    const minD = obs.cz - obs.hd;
    const maxD = obs.cz + obs.hd;

    const gMin = worldToGrid(minW, minD, floorWidth, floorDepth);
    const gMax = worldToGrid(maxW, maxD, floorWidth, floorDepth);

    for (let gz = gMin.gz; gz <= gMax.gz; gz++) {
      for (let gx = gMin.gx; gx <= gMax.gx; gx++) {
        if (gx >= 0 && gx < cols && gz >= 0 && gz < rows) {
          grid[gz * cols + gx] = 1;
        }
      }
    }
  }

  return { grid, cols, rows };
}

// ── A* Pathfinder ──────────────────────

/**
 * 8-directional A* pathfinding on the navigation grid.
 *
 * @param {Uint8Array} grid - 0 = walkable, 1 = obstacle
 * @param {number} cols
 * @param {{ gx: number, gz: number }} start
 * @param {{ gx: number, gz: number }} end
 * @returns {{ gx: number, gz: number }[]} Grid coordinate path (empty if unreachable)
 */
function astar(grid, cols, start, end) {
  const rows = grid.length / cols;
  const startIdx = start.gz * cols + start.gx;
  const endIdx = end.gz * cols + end.gx;

  // Early exit: start or end is blocked
  if (grid[startIdx] === 1 || grid[endIdx] === 1) return [];

  // Octile distance heuristic
  const heuristic = (idx) => {
    const x = idx % cols;
    const z = Math.floor(idx / cols);
    const dx = Math.abs(x - end.gx);
    const dz = Math.abs(z - end.gz);
    return Math.max(dx, dz) + (SQRT2 - 1) * Math.min(dx, dz);
  };

  // Open set as a simple sorted array (adequate for office-sized grids)
  const gScore = new Float32Array(grid.length).fill(Infinity);
  const fScore = new Float32Array(grid.length).fill(Infinity);
  const cameFrom = new Int32Array(grid.length).fill(-1);
  const closed = new Uint8Array(grid.length);

  gScore[startIdx] = 0;
  fScore[startIdx] = heuristic(startIdx);

  /** @type {number[]} */
  const open = [startIdx];

  // 8-directional neighbours: dx, dz, cost
  const DIRS = [
    [-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1],
    [-1, -1, SQRT2], [1, -1, SQRT2], [-1, 1, SQRT2], [1, 1, SQRT2],
  ];

  while (open.length > 0) {
    // Find node with lowest fScore
    let bestI = 0;
    for (let i = 1; i < open.length; i++) {
      if (fScore[open[i]] < fScore[open[bestI]]) bestI = i;
    }
    const current = open[bestI];
    open.splice(bestI, 1);

    if (current === endIdx) {
      // Reconstruct path
      /** @type {{ gx: number, gz: number }[]} */
      const path = [];
      let node = current;
      while (node !== -1) {
        path.push({ gx: node % cols, gz: Math.floor(node / cols) });
        node = cameFrom[node];
      }
      path.reverse();
      return path;
    }

    closed[current] = 1;

    const cx = current % cols;
    const cz = Math.floor(current / cols);

    for (const [dx, dz, cost] of DIRS) {
      const nx = cx + dx;
      const nz = cz + dz;

      if (nx < 0 || nx >= cols || nz < 0 || nz >= rows) continue;

      const nIdx = nz * cols + nx;
      if (closed[nIdx] || grid[nIdx] === 1) continue;

      // For diagonal moves, ensure both adjacent cardinal cells are walkable
      if (dx !== 0 && dz !== 0) {
        if (grid[cz * cols + nx] === 1 || grid[nz * cols + cx] === 1) continue;
      }

      const tentG = gScore[current] + cost;
      if (tentG >= gScore[nIdx]) continue;

      cameFrom[nIdx] = current;
      gScore[nIdx] = tentG;
      fScore[nIdx] = tentG + heuristic(nIdx);

      if (!open.includes(nIdx)) {
        open.push(nIdx);
      }
    }
  }

  return []; // Unreachable
}

// ── Path Smoothing ──────────────────────

/**
 * Convert a grid path to world coordinates and smooth it with
 * Catmull-Rom interpolation. Collinear intermediate points are
 * removed before interpolation.
 *
 * @param {{ gx: number, gz: number }[]} gridPath
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {THREE.Vector3[]}
 */
function smoothPath(gridPath, floorWidth, floorDepth) {
  if (gridPath.length < 2) return [];

  // Convert to world coordinates
  /** @type {{ x: number, z: number }[]} */
  const worldPts = gridPath.map((g) => {
    const { wx, wz } = gridToWorld(g.gx, g.gz, floorWidth, floorDepth);
    return { x: wx, z: wz };
  });

  // Remove collinear intermediate points
  /** @type {{ x: number, z: number }[]} */
  const pruned = [worldPts[0]];
  for (let i = 1; i < worldPts.length - 1; i++) {
    const prev = worldPts[i - 1];
    const curr = worldPts[i];
    const next = worldPts[i + 1];
    const dx1 = curr.x - prev.x;
    const dz1 = curr.z - prev.z;
    const dx2 = next.x - curr.x;
    const dz2 = next.z - curr.z;
    // Cross product check for collinearity
    if (Math.abs(dx1 * dz2 - dz1 * dx2) > 0.001) {
      pruned.push(curr);
    }
  }
  pruned.push(worldPts[worldPts.length - 1]);

  // If only 2 points, return them directly
  if (pruned.length <= 2) {
    return pruned.map((p) => new THREE.Vector3(p.x, 0, p.z));
  }

  // Catmull-Rom interpolation
  const tension = 0.5;
  /** @type {THREE.Vector3[]} */
  const smooth = [];

  for (let i = 0; i < pruned.length - 1; i++) {
    const p0 = pruned[Math.max(0, i - 1)];
    const p1 = pruned[i];
    const p2 = pruned[Math.min(pruned.length - 1, i + 1)];
    const p3 = pruned[Math.min(pruned.length - 1, i + 2)];

    const segments = 4; // subdivisions per segment
    for (let s = 0; s < segments; s++) {
      const t = s / segments;
      const t2 = t * t;
      const t3 = t2 * t;

      const x = tension * (
        (-t3 + 2 * t2 - t) * p0.x +
        (3 * t3 - 5 * t2 + 2) * p1.x +
        (-3 * t3 + 4 * t2 + t) * p2.x +
        (t3 - t2) * p3.x
      );
      const z = tension * (
        (-t3 + 2 * t2 - t) * p0.z +
        (3 * t3 - 5 * t2 + 2) * p1.z +
        (-3 * t3 + 4 * t2 + t) * p2.z +
        (t3 - t2) * p3.z
      );

      smooth.push(new THREE.Vector3(x, 0, z));
    }
  }

  // Add final point
  const last = pruned[pruned.length - 1];
  smooth.push(new THREE.Vector3(last.x, 0, last.z));

  return smooth;
}

// ── Public API ──────────────────────

/**
 * Find a path between two world positions.
 * Returns an array of smoothed world-coordinate waypoints,
 * or an empty array if no path exists.
 *
 * @param {NavGrid} navGrid
 * @param {{ x: number, z: number }} startWorld
 * @param {{ x: number, z: number }} endWorld
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {THREE.Vector3[]}
 */
export function findPath(navGrid, startWorld, endWorld, floorWidth, floorDepth) {
  const start = worldToGrid(startWorld.x, startWorld.z, floorWidth, floorDepth);
  const end = worldToGrid(endWorld.x, endWorld.z, floorWidth, floorDepth);

  // Clamp to grid bounds
  start.gx = Math.max(0, Math.min(navGrid.cols - 1, start.gx));
  start.gz = Math.max(0, Math.min(navGrid.rows - 1, start.gz));
  end.gx = Math.max(0, Math.min(navGrid.cols - 1, end.gx));
  end.gz = Math.max(0, Math.min(navGrid.rows - 1, end.gz));

  const gridPath = astar(navGrid.grid, navGrid.cols, start, end);
  if (gridPath.length === 0) return [];

  return smoothPath(gridPath, floorWidth, floorDepth);
}

/**
 * Check whether a world position is walkable on the nav grid.
 *
 * @param {NavGrid} navGrid
 * @param {number} wx
 * @param {number} wz
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @returns {boolean}
 */
export function isWalkable(navGrid, wx, wz, floorWidth, floorDepth) {
  const { gx, gz } = worldToGrid(wx, wz, floorWidth, floorDepth);
  if (gx < 0 || gx >= navGrid.cols || gz < 0 || gz >= navGrid.rows) return false;
  return navGrid.grid[gz * navGrid.cols + gx] === 0;
}
