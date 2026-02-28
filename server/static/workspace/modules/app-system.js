// ── System Status & 3D Office Initialization ──────────────────────
// Office scene bootstrap, system status polling, and status display updates.

import { getState, setState } from "./state.js";
import { fetchSystemStatus } from "./api.js";
import { initOffice, getScene, getDesks, setCharacterUpdateHook, setCharacterClickHandler, registerClickTarget, highlightDesk, getFloorDimensions, getObstacles } from "./office3d.js";
import { initCharacters, createCharacter, updateAllCharacters, getCharacterGroup, getCharacterHome, setAppearance, updateCharacterState } from "./character.js";
import { setLive2dAppearance } from "./live2d.js";
import { createNavGrid } from "./navigation.js";
import { initMovement, registerCharacter, updateMovements, moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { computePOIs, initIdleBehaviors, updateIdleBehaviors } from "./idle_behavior.js";
import { initInteractions, showMessageEffect, showConversation, updateInteractions } from "./interactions.js";
import { initTimeline, loadHistory } from "./timeline.js";
import { initMessagePopup } from "./message-popup.js";
import { selectAnima } from "./anima.js";
import { mapAnimaStatusToAnim } from "./app-websocket.js";
import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-app-system");

// ── 3D Office Initialization ──────────────────────

export async function initOfficeIfNeeded(dom) {
  if (getState().officeInitialized) return;
  setState({ officeInitialized: true });

  try {
    const { animas } = getState();
    initOffice(dom.officeCanvas, animas);

    const scene = getScene();
    initCharacters(scene);

    setCharacterUpdateHook((dt, elapsed) => {
      updateAllCharacters(dt, elapsed);
      updateMovements(dt);
      updateIdleBehaviors(dt);
      updateInteractions(dt);
    });

    const desks = getDesks();
    for (const p of animas) {
      if (p.appearance) {
        setAppearance(p.name, p.appearance);
        setLive2dAppearance(p.name, p.appearance);
      }
      const deskPos = desks[p.name];
      if (deskPos) {
        const group = await createCharacter(p.name, { x: deskPos.x, y: 0, z: deskPos.z - 0.55 });
        if (group) {
          group.traverse((child) => {
            if (child.isMesh) {
              registerClickTarget(p.name, child);
            }
          });
        }
        const animState = mapAnimaStatusToAnim(p.status);
        updateCharacterState(p.name, animState);
      }
    }

    // ── Navigation + Movement system init ──────────
    const { width: floorW, depth: floorD } = getFloorDimensions();
    const obstacles = getObstacles();
    const navGrid = createNavGrid(floorW, floorD, obstacles);
    initMovement(navGrid, floorW, floorD);

    for (const p of animas) {
      const group = getCharacterGroup(p.name);
      const home = getCharacterHome(p.name);
      if (group && home) {
        registerCharacter(p.name, group, home);
      }
    }

    // ── Interactions + Idle Behaviors + Timeline init ──────────
    const movementSystem = { moveTo, moveToHome, stopMovement, isMoving };
    const characterMap = Object.fromEntries(animas.map((p) => [p.name, true]));
    initInteractions(scene, characterMap, movementSystem);

    const pois = computePOIs(floorW, floorD);
    initIdleBehaviors(characterMap, movementSystem, pois);

    initTimeline(dom.officePanel, { showMessageEffect, showConversation });
    initMessagePopup(dom.officePanel);
    loadHistory(24);

    setCharacterClickHandler((animaName) => {
      selectAnima(animaName);
    });

    const { selectedAnima } = getState();
    if (selectedAnima) {
      highlightDesk(selectedAnima);
    }
  } catch (err) {
    logger.error("Failed to initialize 3D office", { error: err.message });
    setState({ officeInitialized: false });
  }
}

// ── System Status ──────────────────────

export async function loadSystemStatus(dom) {
  if (!dom.systemStatus) return;
  try {
    const data = await fetchSystemStatus();
    updateStatusDisplay(
      dom,
      data.scheduler_running,
      `${data.scheduler_running ? "稼働中" : "停止"} (${data.animas}名)`
    );
  } catch {
    updateStatusDisplay(dom, false, "接続失敗");
  }
}

export function updateStatusDisplay(dom, ok, text) {
  if (!dom.systemStatus) return;
  const dot = dom.systemStatus.querySelector(".status-dot");
  const label = dom.systemStatus.querySelector(".status-text");
  if (dot) dot.className = `status-dot ${ok ? "status-idle" : "status-error"}`;
  if (label) label.textContent = text;
}
