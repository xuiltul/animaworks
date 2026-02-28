// ── Office Structure ──────────────────────
// Floor, walls, windows, and lighting for the 3D office scene.

import * as THREE from "three";

// ── Constants ──────────────────────

export const COLOR = {
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

/**
 * Build the floor plane.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {{plane: Function, lambert: Function}} helpers
 */
export function buildFloor(scene, floorWidth, floorDepth, helpers) {
  const geo = helpers.plane(floorWidth, floorDepth);
  const mat = helpers.lambert({ color: COLOR.floor });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  scene.add(mesh);
}

/**
 * Build side and back walls with windows.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {number} floorDepth
 * @param {{box: Function, plane: Function, lambert: Function, phong: Function}} helpers
 */
export function buildWalls(scene, floorWidth, floorDepth, helpers) {
  const wallMat = helpers.lambert({ color: COLOR.wallSide });
  const wallHeight = 3;
  const halfFloorW = floorWidth / 2;
  const halfFloorD = floorDepth / 2;

  // Left wall
  const leftGeo = helpers.box(0.1, wallHeight, floorDepth);
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
  const backMat = helpers.lambert({ color: COLOR.wallBack });
  const pillarWidth = 2;

  // Left pillar
  const pillarGeo = helpers.box(pillarWidth, wallHeight, 0.1);
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
  const windowWidth = floorWidth - pillarWidth * 2;
  const frameThickness = 0.3;
  const frameGeo = helpers.box(windowWidth, frameThickness, 0.1);
  const topFrame = new THREE.Mesh(frameGeo, backMat);
  topFrame.position.set(0, wallHeight - frameThickness / 2, -halfFloorD);
  scene.add(topFrame);

  // Window frame bottom (sill)
  const sillHeight = 0.8;
  const sillGeo = helpers.box(windowWidth, sillHeight, 0.15);
  const sill = new THREE.Mesh(sillGeo, backMat);
  sill.position.set(0, sillHeight / 2, -halfFloorD);
  scene.add(sill);

  // Window glass (semi-transparent)
  const glassHeight = wallHeight - sillHeight - frameThickness;
  const glassGeo = helpers.plane(windowWidth, glassHeight);
  const glassMat = helpers.phong({
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
  const mullionGeo = helpers.box(0.05, glassHeight, 0.05);
  const mullionMat = helpers.lambert({ color: 0xcccccc });
  for (let i = 1; i < mullionCount; i++) {
    const xPos = -windowWidth / 2 + (windowWidth / mullionCount) * i;
    const mullion = new THREE.Mesh(mullionGeo, mullionMat);
    mullion.position.set(xPos, sillHeight + glassHeight / 2, -halfFloorD);
    scene.add(mullion);
  }
}

/**
 * Set up ambient and directional lighting.
 * @param {THREE.Scene} scene
 * @param {number} floorWidth
 * @param {number} floorDepth
 */
export function buildLighting(scene, floorWidth, floorDepth) {
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);

  const directional = new THREE.DirectionalLight(0xffffff, 0.8);
  directional.position.set(5, 10, 5);
  directional.castShadow = true;

  const shadowExtent = Math.max(floorWidth, floorDepth) / 2 + 2;
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

  const fill = new THREE.DirectionalLight(0xffffff, 0.3);
  fill.position.set(-5, 8, -3);
  scene.add(fill);
}
