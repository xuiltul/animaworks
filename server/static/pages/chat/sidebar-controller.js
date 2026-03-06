// ── Sidebar / Tab Switching Controller ─────────
import { t } from "../../shared/i18n.js";

const RIGHT_PANE_VISIBLE_KEY = "aw-chat-right-pane-visible";

export function createSidebarController(ctx) {
  const $root = ctx.$root;
  const { state } = ctx;

  function applyRightPaneToggleButton(visible) {
    const btn = $root("chatRightPaneToggleBtn");
    if (!btn) return;
    btn.classList.toggle("is-collapsed", !visible);
    btn.setAttribute("aria-pressed", visible ? "true" : "false");
    btn.setAttribute("aria-label", visible ? t("pane.sidebar_hide") : t("pane.sidebar_show"));
    btn.setAttribute("title", visible ? t("pane.sidebar_hide") : t("pane.sidebar_show"));
  }

  function setRightPaneVisible(visible, { persist = true } = {}) {
    const nextVisible = Boolean(visible);
    state.rightPaneVisible = nextVisible;
    const layout = $root("chatPageLayout");
    if (layout) layout.classList.toggle("sidebar-hidden", !nextVisible);
    // Keep mobile-hidden in sync so both mechanisms agree
    const sidebar = state.container?.querySelector(".chat-page-sidebar");
    if (sidebar) sidebar.classList.toggle("mobile-hidden", !nextVisible);
    applyRightPaneToggleButton(nextVisible);
    if (persist) localStorage.setItem(RIGHT_PANE_VISIBLE_KEY, nextVisible ? "1" : "0");
  }

  function toggleRightPane() {
    setRightPaneVisible(!state.rightPaneVisible);
  }

  function initRightPaneVisibility() {
    const stored = localStorage.getItem(RIGHT_PANE_VISIBLE_KEY);
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    // On mobile, default to sidebar hidden (user can toggle to open)
    const isVisible = isMobile ? false : stored !== "0";
    setRightPaneVisible(isVisible, { persist: false });
  }

  function switchMobileTab(panel) {
    const chatTab = $root("chatMobileTabChat");
    const infoTab = $root("chatMobileTabInfo");
    const mainPanel = state.container?.querySelector(".chat-page-main");
    const sidePanel = state.container?.querySelector(".chat-page-sidebar");
    if (!mainPanel || !sidePanel) return;

    if (panel === "chat") {
      chatTab?.classList.add("active");
      infoTab?.classList.remove("active");
      mainPanel.classList.remove("mobile-hidden");
      sidePanel.classList.add("mobile-hidden");
    } else {
      chatTab?.classList.remove("active");
      infoTab?.classList.add("active");
      mainPanel.classList.add("mobile-hidden");
      sidePanel.classList.remove("mobile-hidden");
    }
  }

  function switchRightTab(tab) {
    state.activeRightTab = tab;
    const tabs = { state: "chatPaneState", activity: "chatPaneActivity", history: "chatPaneHistory" };

    for (const btn of (state.container?.querySelectorAll(".right-tab") || [])) {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    }
    for (const [key, id] of Object.entries(tabs)) {
      const el = $root(id);
      if (el) el.style.display = key === tab ? "" : "none";
    }

    if (tab === "history" && state.selectedAnima) {
      const detail = $root("chatHistoryDetail");
      const list = $root("chatHistorySessionList");
      if (detail) detail.style.display = "none";
      if (list) list.style.display = "";
      ctx.controllers.history.loadSessionList();
    }
    if (tab === "activity") ctx.controllers.activity.loadActivity();
  }

  return { switchMobileTab, switchRightTab, toggleRightPane, initRightPaneVisibility };
}
