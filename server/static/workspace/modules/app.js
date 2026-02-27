// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.
// Chat/Board/Activity/Sidebar logic extracted to separate modules.

import { initI18n, applyTranslations } from "/shared/i18n.js";
import { getState, setState, subscribe } from "./state.js";
import { fetchSystemStatus } from "./api.js";
import { connect, onEvent } from "./websocket.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initAnima, loadAnimas, selectAnima, renderAnimaSelector, renderStatus } from "./anima.js";
import { initMemory, loadMemoryTab } from "./memory.js";
import { initSession, loadSessions } from "./session.js";
import { escapeHtml } from "./utils.js";
import { initOffice, disposeOffice, getDesks, highlightDesk, setCharacterClickHandler, getScene, registerClickTarget, setCharacterUpdateHook, getObstacles, getFloorDimensions } from "./office3d.js";
import { initCharacters, createCharacter, removeCharacter, updateCharacterState, updateAllCharacters, getCharacterGroup, getCharacterHome, setAppearance } from "./character.js";
import { setCharacter, setExpression, setLive2dAppearance } from "./live2d.js";
import { createNavGrid } from "./navigation.js";
import { initMovement, registerCharacter, updateMovements, moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { computePOIs, initIdleBehaviors, updateIdleBehaviors, cancelBehavior } from "./idle_behavior.js";
import { initInteractions, showMessageEffect, showConversation, updateInteractions } from "./interactions.js";
import { initTimeline, addTimelineEvent, loadHistory, localISOString } from "./timeline.js";
import { initMessagePopup, isVisible as isMessagePopupVisible, hide as hideMessagePopup } from "./message-popup.js";
import { playReveal } from "./reveal.js";
import { createLogger } from "../../shared/logger.js";
import { initOrgDashboard, disposeOrgDashboard, updateAnimaStatus, addActivityItem } from "./org-dashboard.js";

import { initActivity, addActivity } from "./activity.js";
import { initSidebar, activateRightTab } from "./sidebar.js";
import { initBoard, initBoardTab, getSelectedBoard, appendBoardMessage } from "./board.js";
import { initChatController, openConversation, closeConversation, submitConversation, isConvStreaming } from "./chat-controller.js";

const logger = createLogger("ws-app");

// ── DOM References ──────────────────────

const dom = {};

function cacheDom() {
  dom.loginContainer = document.getElementById("wsLogin");
  dom.dashboard = document.getElementById("wsDashboard");
  dom.animaSelector = document.getElementById("wsAnimaSelector");
  dom.systemStatus = document.getElementById("wsSystemStatus");
  dom.userInfo = document.getElementById("wsUserInfo");
  dom.rightTabs = document.getElementById("wsRightTabs");
  dom.tabState = document.getElementById("wsTabState");
  dom.tabActivity = document.getElementById("wsTabActivity");
  dom.tabBoard = document.getElementById("wsTabBoard");
  dom.tabHistory = document.getElementById("wsTabHistory");
  dom.paneState = document.getElementById("wsPaneState");
  dom.paneActivity = document.getElementById("wsPaneActivity");
  dom.paneBoard = document.getElementById("wsPaneBoard");
  dom.paneHistory = document.getElementById("wsPaneHistory");
  dom.memoryPanel = document.getElementById("wsMemoryPanel");
  dom.logoutBtn = document.getElementById("wsLogoutBtn");

  // 3D Office
  dom.officePanel = document.getElementById("wsOfficePanel");
  dom.officeCanvas = document.getElementById("wsOfficeCanvas");
  dom.orgPanel = document.getElementById("wsOrgPanel");
  dom.viewToggle = document.getElementById("wsViewToggle");

  // Conversation overlay (3-column)
  dom.convOverlay = document.getElementById("wsConvOverlay");
  dom.convLayout = document.getElementById("wsConvLayout");
  dom.convBack = document.getElementById("wsConvBack");
  dom.convAnimaName = document.getElementById("wsConvAnimaName");
  dom.threadTabs = document.getElementById("wsThreadTabs");
  dom.convCanvas = document.getElementById("wsConvCanvas");
  dom.convMessages = document.getElementById("wsConvMessages");
  dom.convInput = document.getElementById("wsConvInput");
  dom.convSend = document.getElementById("wsConvSend");
  dom.convPreviewBar = document.getElementById("wsConvPreviewBar");
  dom.convAttachBtn = document.getElementById("wsConvAttachBtn");
  dom.convFileInput = document.getElementById("wsConvFileInput");
  dom.convPending = document.getElementById("wsConvPending");
  dom.convPendingList = document.getElementById("wsConvPendingList");
  dom.convPendingLabel = document.getElementById("wsConvPendingLabel");
  dom.convPendingCancel = document.getElementById("wsConvPendingCancel");
  dom.convQueueBtn = document.getElementById("wsConvQueueBtn");

  // Mobile controls
  dom.mobileSidebarToggle = document.getElementById("wsMobileSidebarToggle");
  dom.mobileCharacterToggle = document.getElementById("wsMobileCharacterToggle");
  dom.sidebarBackdrop = document.getElementById("wsSidebarBackdrop");
  dom.mobileMemoryClose = document.getElementById("wsMobileMemoryClose");
  dom.convSidebar = document.querySelector(".ws-conv-sidebar");
  dom.convCharacter = document.querySelector(".ws-conv-character");
}

// (Activity, Sidebar, Board extracted to separate modules)

// ── 3D Office Initialization ──────────────────────

async function initOfficeIfNeeded() {
  if (getState().officeInitialized) return;
  setState({ officeInitialized: true });

  try {
    // Initialize 3D scene with anima data for dynamic desk layout
    const { animas } = getState();
    initOffice(dom.officeCanvas, animas);

    // Initialize characters in the scene
    const scene = getScene();
    initCharacters(scene);

    // Register character animation update in the render loop
    setCharacterUpdateHook((dt, elapsed) => {
      updateAllCharacters(dt, elapsed);
      updateMovements(dt);
      updateIdleBehaviors(dt);
      updateInteractions(dt);
    });

    // Create characters at their dynamically assigned desk positions
    const desks = getDesks();
    for (const p of animas) {
      // Register appearance from API before creating character
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

    // Register all characters with the movement system
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

    // Handle character clicks → open conversation in right panel
    // selectAnima → onAnimaSelected → openConversation の一本化フロー
    setCharacterClickHandler((animaName) => {
      selectAnima(animaName);
    });

    // Highlight selected anima's desk
    const { selectedAnima } = getState();
    if (selectedAnima) {
      highlightDesk(selectedAnima);
    }
  } catch (err) {
    logger.error("Failed to initialize 3D office", { error: err.message });
    setState({ officeInitialized: false });
  }
}

// ── View Switching ──────────────────────

let _currentView = null; // '3d' | 'org'

function getDefaultView() {
  const theme = localStorage.getItem("aw-theme") || "default";
  return theme === "business" ? "org" : "3d";
}

function getCurrentView() {
  return localStorage.getItem("aw-workspace-view") || getDefaultView();
}

async function switchView(view) {
  if (_currentView === view) return;
  _currentView = view;
  localStorage.setItem("aw-workspace-view", view);

  if (view === "org") {
    // Dispose 3D resources before switching
    disposeOffice();
    setState({ officeInitialized: false });

    dom.officePanel.classList.add("hidden");
    dom.orgPanel.classList.remove("hidden");

    const { animas } = getState();
    await initOrgDashboard(dom.orgPanel, animas, {
      onNodeClick: (name) => selectAnima(name),
    });
  } else {
    // Dispose org dashboard before switching
    dom.orgPanel.classList.add("hidden");
    dom.officePanel.classList.remove("hidden");
    disposeOrgDashboard();

    await initOfficeIfNeeded();
  }

  updateViewToggle();
}

function updateViewToggle() {
  if (!dom.viewToggle) return;
  const is3d = _currentView === "3d";
  dom.viewToggle.querySelector(".ws-view-toggle-3d").style.fontWeight = is3d ? "700" : "400";
  dom.viewToggle.querySelector(".ws-view-toggle-org").style.fontWeight = is3d ? "400" : "700";
}

// ── Status Mapping ──────────────────────

function mapAnimaStatusToAnim(status) {
  if (!status) return "idle";
  const s = typeof status === "object" ? status.state || status.status || "idle" : String(status);
  const lower = s.toLowerCase();
  if (lower === "not_found" || lower === "stopped") return "sleeping";
  if (lower.includes("bootstrap")) return "thinking";
  if (lower.includes("think") || lower.includes("process")) return "thinking";
  if (lower.includes("work") || lower.includes("busy") || lower.includes("running")) return "working";
  if (lower.includes("error") || lower.includes("fail")) return "error";
  if (lower.includes("sleep") || lower.includes("stop") || lower.includes("inactive")) return "sleeping";
  if (lower.includes("talk") || lower.includes("chat")) return "talking";
  if (lower.includes("report")) return "reporting";
  return "idle";
}

// (Conversation panel extracted to chat-controller.js)

// All conversation/thread/streaming/board/queue/mobile/voice functions
// have been moved to chat-controller.js, board.js, activity.js, sidebar.js.
// The removed block was ~1270 lines (openConversation through sendBoardMessage).
//
// Placeholder to mark the boundary — remove this comment block in final cleanup.



// ── System Status ──────────────────────

async function loadSystemStatus() {
  if (!dom.systemStatus) return;
  try {
    const data = await fetchSystemStatus();
    updateStatusDisplay(
      data.scheduler_running,
      `${data.scheduler_running ? "稼働中" : "停止"} (${data.animas}名)`
    );
  } catch {
    updateStatusDisplay(false, "接続失敗");
  }
}

function updateStatusDisplay(ok, text) {
  if (!dom.systemStatus) return;
  const dot = dom.systemStatus.querySelector(".status-dot");
  const label = dom.systemStatus.querySelector(".status-text");
  if (dot) dot.className = `status-dot ${ok ? "status-idle" : "status-error"}`;
  if (label) label.textContent = text;
}

// ── WebSocket Handlers ──────────────────────

const wsUnsubscribers = [];
const lastAnimaStatus = {};

function setupWebSocket() {
  wsUnsubscribers.forEach((fn) => fn());
  wsUnsubscribers.length = 0;

  connect();

  wsUnsubscribers.push(onEvent("anima.status", (data) => {
    const { animas, selectedAnima } = getState();
    const idx = animas.findIndex((p) => p.name === data.name);
    if (idx >= 0) {
      animas[idx] = { ...animas[idx], ...data };
      setState({ animas: [...animas] });
      renderAnimaSelector(dom.animaSelector);
    }
    if (data.name === selectedAnima) {
      renderStatus(dom.paneState);
    }
    // Update 3D character animation
    if (getState().officeInitialized) {
      const animState = mapAnimaStatusToAnim(data.status);
      updateCharacterState(data.name, animState);
      setState({ characterStates: { ...getState().characterStates, [data.name]: animState } });
    }
    // Only log to activity feed when status actually changes
    if (lastAnimaStatus[data.name] !== data.status) {
      lastAnimaStatus[data.name] = data.status;
      addActivity("system", data.name, `Status: ${data.status}`);
    }
    if (_currentView === "org") {
      updateAnimaStatus(data.name, data.status || data);
    }
  }));

  // ── anima.interaction — inter-anima messaging visualization ──
  wsUnsubscribers.push(onEvent("anima.interaction", (data) => {
    cancelBehavior(data.from_person);
    cancelBehavior(data.to_person);

    if (data.type === "message") {
      showMessageEffect(data.from_person, data.to_person, data.summary || "");
    }

    addTimelineEvent({
      id: Date.now().toString(),
      type: "message",
      anima: `${data.from_person} → ${data.to_person}`,
      ts: data.ts || localISOString(),
      summary: `${data.from_person} → ${data.to_person}: ${data.summary || ""}`,
      meta: {
        text: data.summary || "",
        message_id: data.message_id || "",
        from_person: data.from_person,
        to_person: data.to_person,
      },
    });
    if (_currentView === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.interaction",
        from: data.from_person || data.from || "",
        summary: data.summary || "",
      });
    }
  }));

  wsUnsubscribers.push(onEvent("anima.heartbeat", (data) => {
    addActivity("heartbeat", data.name, data.summary || "heartbeat completed");
    const { selectedAnima } = getState();
    if (data.name === selectedAnima) {
      renderStatus(dom.paneState);
    }
    addTimelineEvent({
      id: Date.now().toString(),
      type: "heartbeat",
      anima: data.name,
      ts: data.ts || localISOString(),
      summary: data.summary || "heartbeat completed",
    });
    if (_currentView === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.heartbeat",
        from: data.name || "",
        summary: data.summary || "heartbeat completed",
      });
    }
  }));

  wsUnsubscribers.push(onEvent("anima.cron", (data) => {
    addActivity("cron", data.name, data.summary || `cron: ${data.task || ""}`);
    addTimelineEvent({
      id: Date.now().toString(),
      type: "cron",
      anima: data.name,
      ts: data.ts || localISOString(),
      summary: data.summary || `cron: ${data.task || ""}`,
    });
    if (_currentView === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.cron",
        from: data.name || "",
        summary: data.summary || `cron: ${data.task || ""}`,
      });
    }
  }));

  // ── board.post — shared channel message ──
  wsUnsubscribers.push(onEvent("board.post", (data) => {
    const from = data.from || "?";
    const channel = data.channel || "?";
    const text = data.text || "";

    const boardSel = getSelectedBoard();
    if (boardSel.type === "channel" && boardSel.channel === channel) {
      appendBoardMessage({
        ts: data.ts || new Date().toISOString(),
        from,
        text,
        source: data.source || "",
      });
    }

    // Add to activity feed
    addActivity("chat", from, `[#${channel}] ${text}`);

    // Add to timeline
    addTimelineEvent({
      id: Date.now().toString(),
      type: "board",
      anima: from,
      ts: data.ts || localISOString(),
      summary: `#${channel}: ${from} — ${text}`,
      meta: {
        channel,
        from,
        text,
        source: data.source || "",
      },
    });
    if (_currentView === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "board.post",
        from,
        summary: `[#${channel}] ${text}`,
      });
    }
  }));

  // ── anima.proactive_message — autonomous outbound messages ──
  wsUnsubscribers.push(onEvent("anima.proactive_message", (data) => {
    const animaName = data.name || data.anima || "";
    const summary = data.summary || data.message || "";
    if (animaName) {
      addTimelineEvent({
        id: Date.now().toString(),
        type: "dm_sent",
        anima: animaName,
        ts: data.ts || localISOString(),
        summary: summary.slice(0, 100),
      });
    }
  }));

  // ── anima.notification — human notifications ──
  wsUnsubscribers.push(onEvent("anima.notification", (data) => {
    // Timeline entry handled by anima.proactive_message to avoid duplicates
  }));

  wsUnsubscribers.push(onEvent("anima.bootstrap", (data) => {
    const { name, status: bsStatus } = data;
    if (bsStatus === "started") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "bootstrapping", bootstrapping: true };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "thinking");
      }
      addActivity("system", name, "ブートストラップ開始");
    } else if (bsStatus === "completed") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "idle", bootstrapping: false };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "idle");
      }
      addActivity("system", name, "ブートストラップ完了");
    } else if (bsStatus === "failed") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "error", bootstrapping: false };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "error");
      }
      addActivity("system", name, "ブートストラップ失敗");
    }
  }));

  wsUnsubscribers.push(onEvent("anima.assets_updated", async (data) => {
    const animaName = data.name;
    addActivity("system", animaName, `アセット更新: ${(data.assets || []).join(", ")}`);

    // ── Reveal animation (Anima birth) ──
    const assets = data.assets || [];
    const hasAvatar = assets.some((a) => a.startsWith("avatar_"));
    if (hasAvatar) {
      const avatarUrl = `/api/animas/${encodeURIComponent(animaName)}/assets/avatar_bustup.png`;
      await playReveal({ name: animaName, avatarUrl });
    }

    // Refresh 3D character if office is initialised
    if (getState().officeInitialized) {
      const desks = getDesks();
      const deskPos = desks[animaName];
      if (deskPos) {
        removeCharacter(animaName);
        const group = await createCharacter(
          animaName,
          { x: deskPos.x, y: deskPos.y + 0.4, z: deskPos.z - 0.3 },
        );
        if (group) {
          group.traverse((child) => {
            if (child.isMesh) registerClickTarget(animaName, child);
          });
        }
      }
    }

    // Refresh bust-up if conversation is open for this anima
    if (getState().conversationAnima === animaName) {
      await setCharacter(animaName);
    }
  }));

  // Track connection state for status indicator
  wsUnsubscribers.push(subscribe((state) => {
    if (state.wsConnected) {
      updateStatusDisplay(true, `接続済 (${state.animas.length}名)`);
    } else {
      updateStatusDisplay(false, "再接続中...");
    }
  }));
}

// ── Viewport / UI Helpers ──────────────────────

/** Set CSS custom property for viewport height fallback (svh-unsupported browsers). */
function initViewportHeightFallback() {
  if (CSS.supports && CSS.supports('height', '100svh')) return;

  function setVh() {
    const vh = window.visualViewport?.height || window.innerHeight;
    document.documentElement.style.setProperty('--vh-fallback', `${vh}px`);
  }
  setVh();
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', setVh);
  }
  window.addEventListener('resize', setVh);
}

/** Toggle timeline collapse on iPad-width viewports. */
function initTimelineCollapseToggle() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.timeline-toggle-btn');
    if (!btn) return;
    const timeline = btn.closest('.ws-timeline');
    if (timeline) {
      timeline.classList.toggle('collapsed');
    }
  });
}

function initMobileKeyboard() {
  const vv = window.visualViewport;

  function scrollInputIntoView() {
    const active = document.activeElement;
    if (!active?.matches(".ws-conv-input, .chat-input")) return;
    requestAnimationFrame(() => {
      active.scrollIntoView({ block: "nearest" });
    });
  }

  if (vv) {
    vv.addEventListener("resize", scrollInputIntoView);
  } else {
    // Fallback for browsers without visualViewport
    let lastHeight = window.innerHeight;
    window.addEventListener("resize", () => {
      const current = window.innerHeight;
      if (Math.abs(current - lastHeight) > 100) {
        scrollInputIntoView();
      }
      lastHeight = current;
    });
  }
}

// ── Dashboard Bootstrap ──────────────────────

let dashboardInitialized = false;

async function startDashboard() {
  if (!dom.dashboard) return;

  dom.dashboard.classList.remove("hidden");
  if (dom.userInfo) {
    dom.userInfo.textContent = getCurrentUser() || "";
  }

  if (dashboardInitialized) {
    await loadAnimas();
    await loadSystemStatus();
    const initialView = getCurrentView();
    await switchView(initialView);
    return;
  }
  dashboardInitialized = true;

  initAnima(dom.animaSelector, dom.paneState, onAnimaSelected);
  initMemory(dom.memoryPanel);
  initSession(dom.paneHistory);

  initActivity(dom.paneActivity);
  initBoard(dom.paneBoard);
  initSidebar({
    tabState: dom.tabState, tabActivity: dom.tabActivity, tabBoard: dom.tabBoard, tabHistory: dom.tabHistory,
    paneState: dom.paneState, paneActivity: dom.paneActivity, paneBoard: dom.paneBoard, paneHistory: dom.paneHistory,
  }, { onBoardInit: initBoardTab });

  initChatController(dom);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (isMessagePopupVisible()) {
        hideMessagePopup();
      } else if (getState().conversationOpen) {
        closeConversation();
      }
    }
  });

  dom.logoutBtn?.addEventListener("click", () => {
    dom.dashboard.classList.add("hidden");
    logout();
  });

  await loadAnimas();
  await loadSystemStatus();
  setupWebSocket();
  activateRightTab("state");

  const initialView = getCurrentView();
  await switchView(initialView);

  if (dom.viewToggle) {
    dom.viewToggle.addEventListener("click", () => {
      const next = _currentView === "3d" ? "org" : "3d";
      switchView(next);
    });
  }

  initViewportHeightFallback();
  initTimelineCollapseToggle();
  initMobileKeyboard();
}

// ── Anima Selection Callback ──────────────────────

async function onAnimaSelected(name) {
  // Highlight desk in 3D
  if (getState().officeInitialized) {
    highlightDesk(name);
  }

  // Open conversation panel + load memory/sessions in parallel
  await Promise.all([
    openConversation(name),
    loadMemoryTab(getState().activeMemoryTab),
    loadSessions(),
  ]);
}

// ── Main Init ──────────────────────

function applyTheme() {
  const theme = localStorage.getItem("aw-theme") || "default";
  document.body.classList.toggle("theme-business", theme === "business");
}

export async function init() {
  await initI18n();
  applyTranslations();

  applyTheme();
  cacheDom();

  const savedUser = getCurrentUser();
  if (savedUser) {
    initLogin(dom.loginContainer, onLoginSuccess);
    startDashboard();
  } else {
    dom.dashboard?.classList.add("hidden");
    initLogin(dom.loginContainer, onLoginSuccess);
  }
}

function onLoginSuccess(_username) {
  startDashboard();
}

// Auto-init on DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
