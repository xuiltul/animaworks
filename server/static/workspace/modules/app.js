// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.
// Chat/Board/Activity/Sidebar logic extracted to separate modules.

import { initI18n, applyTranslations } from "/shared/i18n.js";
import { getState, setState } from "./state.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initAnima, loadAnimas, selectAnima } from "./anima.js";
import { initMemory, loadMemoryTab } from "./memory.js";
import { initSession, loadSessions } from "./session.js";
import { disposeOffice, highlightDesk } from "./office3d.js";
import { initOrgDashboard, disposeOrgDashboard } from "./org-dashboard.js";
import { isVisible as isMessagePopupVisible, hide as hideMessagePopup } from "./message-popup.js";
import { initActivity } from "./activity.js";
import { initSidebar, activateRightTab } from "./sidebar.js";
import { initBoard, initBoardTab } from "./board.js";
import { initChatController, openConversation, closeConversation } from "./chat-controller.js";

import { setupWebSocket } from "./app-websocket.js";
import { initOfficeIfNeeded, loadSystemStatus, updateStatusDisplay } from "./app-system.js";
import { initViewportHeightFallback, initTimelineCollapseToggle, initMobileKeyboard } from "./app-mobile.js";

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
    disposeOffice();
    setState({ officeInitialized: false });

    dom.officePanel.classList.add("hidden");
    dom.orgPanel.classList.remove("hidden");

    const { animas } = getState();
    await initOrgDashboard(dom.orgPanel, animas, {
      onNodeClick: (name) => selectAnima(name),
    });
  } else {
    dom.orgPanel.classList.add("hidden");
    dom.officePanel.classList.remove("hidden");
    disposeOrgDashboard();

    await initOfficeIfNeeded(dom);
  }

  updateViewToggle();
}

function updateViewToggle() {
  if (!dom.viewToggle) return;
  const is3d = _currentView === "3d";
  dom.viewToggle.querySelector(".ws-view-toggle-3d").style.fontWeight = is3d ? "700" : "400";
  dom.viewToggle.querySelector(".ws-view-toggle-org").style.fontWeight = is3d ? "400" : "700";
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
    await loadSystemStatus(dom);
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
  await loadSystemStatus(dom);
  setupWebSocket({
    dom,
    updateStatusDisplay: (ok, text) => updateStatusDisplay(dom, ok, text),
    getCurrentView: () => _currentView,
  });
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
  if (getState().officeInitialized) {
    highlightDesk(name);
  }

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
