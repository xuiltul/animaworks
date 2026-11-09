// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.
// Chat/Board/Activity/Sidebar logic extracted to separate modules.

import { initI18n, applyTranslations, t } from "/shared/i18n.js";
import { api } from "../../modules/api.js";
import { getState, setState } from "./state.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initAnima, loadAnimas, selectAnima } from "./anima.js";
import { mountOffice, disposeOfficeRenderer, highlightAnima } from "./office-renderer.js";
import { normalizeWorkspaceView, selectWorkspaceView } from "./office-renderer-selection.js";
import { createBattleView } from "../../battle/modules/view.js";
import { initOrgDashboard, disposeOrgDashboard, startReplay, stopReplay, isReplayMode } from "./org-dashboard.js";
import { isVisible as isMessagePopupVisible, hide as hideMessagePopup } from "./message-popup.js";
import {
  initChatDock,
  openConversation,
  closeConversation,
  isConversationOpen,
  isChatPopoverOpen,
} from "./chat-dock.js";

import { setupWebSocket } from "./app-websocket.js";
import { initOfficeShell, loadSystemStatus, updateStatusDisplay } from "./app-system.js";
import { initViewportHeightFallback, initTimelineCollapseToggle, initMobileKeyboard } from "./app-mobile.js";

// ── DOM References ──────────────────────

const dom = {};

function cacheDom() {
  dom.loginContainer = document.getElementById("wsLogin");
  dom.dashboard = document.getElementById("wsDashboard");
  dom.animaSelector = document.getElementById("wsAnimaSelector");
  dom.systemStatus = document.getElementById("wsSystemStatus");
  dom.userInfo = document.getElementById("wsUserInfo");
  dom.currentUserLabel = document.getElementById("currentUserLabel");
  dom.logoutBtn = document.getElementById("wsLogoutBtn");

  // Replaceable office renderer, stage and embedded Chat dock
  dom.officePanel = document.getElementById("wsOfficePanel");
  dom.officeCanvas = document.getElementById("wsOfficeCanvas");
  dom.orgPanel = document.getElementById("wsOrgPanel");
  dom.battlePanel = document.getElementById("wsBattlePanel");
  dom.stage = document.getElementById("wsStage");
  dom.chatDock = document.getElementById("wsChatDock");
  dom.viewToggle = document.getElementById("wsViewToggle");
}

// ── View Switching ──────────────────────

let _currentView = null; // 'office' | 'org' | 'battle'
let _battleView = null;
let _viewSwitchGeneration = 0;

function getDefaultView() {
  const mode = localStorage.getItem("aw-display-mode") || "realistic";
  return mode === "realistic" ? "org" : "office";
}

function getCurrentView() {
  let storedView = null;
  try {
    storedView = localStorage.getItem("aw-workspace-view");
  } catch { /* use the URL/default when storage is unavailable */ }

  const preference = selectWorkspaceView(window.location.search, storedView, getDefaultView());
  if (preference.source === "url" || (preference.source === "localStorage" && storedView !== preference.view)) {
    try {
      localStorage.setItem("aw-workspace-view", preference.view);
    } catch { /* the selected view still works without storage */ }
  }
  return preference.view;
}

async function switchView(view) {
  const nextView = normalizeWorkspaceView(view, getDefaultView());
  if (_currentView === nextView) {
    updateViewToggle();
    return;
  }

  const generation = ++_viewSwitchGeneration;
  _currentView = nextView;
  try {
    localStorage.setItem("aw-workspace-view", nextView);
  } catch { /* the current session can still switch views */ }

  if (nextView !== "office") {
    disposeOfficeRenderer();
    setState({ officeInitialized: false });
  }
  if (nextView !== "org") disposeOrgDashboard();
  if (nextView !== "battle") {
    _battleView?.dispose();
    _battleView = null;
  }

  dom.officePanel.classList.toggle("hidden", nextView !== "office");
  dom.orgPanel.classList.toggle("hidden", nextView !== "org");
  dom.battlePanel.classList.toggle("hidden", nextView !== "battle");
  updateViewToggle();

  if (nextView === "battle") {
    _battleView = createBattleView({
      root: dom.battlePanel,
      onAnimaClick: (name) => selectAnima(name),
    });
  } else if (nextView === "org") {
    const { animas } = getState();
    await initOrgDashboard(dom.orgPanel, animas, {
      onNodeClick: (name) => selectAnima(name),
    });
    if (generation !== _viewSwitchGeneration) return;
  } else {
    initOfficeShell(dom);
    await mountOffice(dom);
    if (generation !== _viewSwitchGeneration) return;
  }
}

function updateViewToggle() {
  if (!dom.viewToggle) return;
  dom.viewToggle.setAttribute("aria-label", t("ws.view_toggle_title"));
  for (const button of dom.viewToggle.querySelectorAll("[data-view]")) {
    const selected = button.dataset.view === _currentView;
    button.classList.toggle("ws-view-toggle-option--active", selected);
    button.style.fontWeight = selected ? "700" : "400";
    button.setAttribute("aria-pressed", String(selected));
  }

  let replayBtn = dom.viewToggle.querySelector(".ws-replay-btn");
  if (_currentView === "org" && !replayBtn) {
    replayBtn = document.createElement("button");
    replayBtn.className = "ws-replay-btn";
    replayBtn.textContent = "▶ Replay";
    replayBtn.title = "Replay activity";
    replayBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      replayBtn.disabled = true;
      try {
        if (isReplayMode()) {
          stopReplay();
          replayBtn.textContent = "▶ Replay";
          replayBtn.classList.remove("ws-replay-btn--active");
        } else {
          const started = await startReplay(24);
          if (started) {
            replayBtn.textContent = "■ Live";
            replayBtn.classList.add("ws-replay-btn--active");
          } else {
            replayBtn.textContent = "▶ Replay";
            replayBtn.classList.remove("ws-replay-btn--active");
          }
        }
      } finally {
        replayBtn.disabled = false;
      }
    });
    dom.viewToggle.appendChild(replayBtn);
  } else if (_currentView !== "org" && replayBtn) {
    if (isReplayMode()) stopReplay();
    replayBtn.remove();
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
    await loadSystemStatus(dom);
    const initialView = getCurrentView();
    await switchView(initialView);
    return;
  }
  dashboardInitialized = true;

  initAnima(dom.animaSelector, onAnimaSelected);
  initChatDock(dom.chatDock, { onClosed: () => highlightAnima(null) });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (isMessagePopupVisible()) {
      hideMessagePopup();
    } else if (isChatPopoverOpen()) {
      // The chat pane's own handlers close the popover first.
      return;
    } else if (isConversationOpen()) {
      closeConversation();
    }
  });

  dom.logoutBtn?.addEventListener("click", () => {
    closeConversation();
    dom.dashboard.classList.add("hidden");
    logout();
  });

  await loadAnimas();
  await loadSystemStatus(dom);
  if (dom.currentUserLabel) dom.currentUserLabel.textContent = getCurrentUser() || "";
  setupWebSocket({
    dom,
    updateStatusDisplay: (ok, text) => updateStatusDisplay(dom, ok, text),
    getCurrentView: () => _currentView,
  });
  const initialView = getCurrentView();
  await switchView(initialView);

  if (dom.viewToggle) {
    dom.viewToggle.addEventListener("click", (event) => {
      const button = event.target.closest("[data-view]");
      if (!button || !dom.viewToggle.contains(button)) return;
      void switchView(button.dataset.view);
    });
  }

  initViewportHeightFallback();
  initTimelineCollapseToggle();
  initMobileKeyboard();
}

// ── Anima Selection Callback ──────────────────────

function onAnimaSelected(name) {
  highlightAnima(name);
  void openConversation(name);
}

// ── Main Init ──────────────────────

const ALL_THEMES = [
  "business", "graphite", "ocean", "forest", "sunset",
  "rose", "lavender", "nord", "monokai", "midnight", "solarized"
];

function applyTheme() {
  let theme = localStorage.getItem("aw-theme");
  if (!theme) theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "midnight" : "default";

  ALL_THEMES.forEach(t => document.body.classList.remove(`theme-${t}`));
  if (theme !== "default") {
    document.body.classList.add(`theme-${theme}`);
  }
  if (theme === "monokai" || theme === "midnight") {
    document.body.setAttribute("data-theme", "dark");
  } else {
    document.body.removeAttribute("data-theme");
  }
}

function applyDisplayMode() {
  const mode = localStorage.getItem("aw-display-mode") || "realistic";
  document.body.classList.remove("mode-anime", "mode-realistic");
  document.body.classList.add(`mode-${mode}`);
}

export async function init() {
  await initI18n();
  applyTranslations();

  applyDisplayMode();
  applyTheme();
  cacheDom();

  let savedUser = getCurrentUser();
  if (!savedUser) {
    // Arriving from the dashboard with a session: reuse that identity
    // instead of asking the same person to pick a user again.
    try {
      const me = await api("/api/auth/me", { redirectOnUnauthorized: false, logErrors: false });
      if (me?.username) {
        savedUser = me.username;
        setState({ currentUser: savedUser });
        localStorage.setItem("animaworks_user", savedUser);
      }
    } catch { /* not authenticated: fall through to the picker */ }
  }
  if (savedUser) {
    initLogin(dom.loginContainer, onLoginSuccess);
    startDashboard();
  } else {
    dom.dashboard?.classList.add("hidden");
    initLogin(dom.loginContainer, onLoginSuccess);
  }
}

function onLoginSuccess(username) {
  if (dom.currentUserLabel) dom.currentUserLabel.textContent = username || "";
  startDashboard();
}

// Auto-init on DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
