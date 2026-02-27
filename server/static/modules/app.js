/* ============================================
   AnimaWorks — Dashboard App (Entry Point)
   ============================================ */

import { initI18n, applyTranslations, t } from "/shared/i18n.js";
import { state } from "./state.js";
import { connectWebSocket } from "./websocket.js";
// state.authMode is set by login.js checkAuth()
import { loadSystemStatus } from "./status.js";

// ── Theme ────────────────────────────────────

function applyTheme(theme) {
  state.uiTheme = theme;
  document.body.classList.toggle("theme-business", theme === "business");
  const toggle = document.getElementById("themeSwitch");
  if (toggle) toggle.checked = theme === "business";
  localStorage.setItem("aw-theme", theme);
}

async function initTheme() {
  const stored = localStorage.getItem("aw-theme");
  if (stored) {
    applyTheme(stored);
    return;
  }
  try {
    const resp = await fetch("/api/system/config");
    if (resp.ok) {
      const cfg = await resp.json();
      const theme = cfg?.ui?.theme || "default";
      applyTheme(theme);
      return;
    }
  } catch { /* ignore */ }
  applyTheme("default");
}
import { checkAuth, logout, showLoginScreen, hideLoginScreen, setStartDashboard } from "./login.js";
import { initRouter } from "./router.js";

// ── Dashboard Startup ───────────────────────

async function startDashboard() {
  initRouter("pageContent");
  connectWebSocket();
  loadSystemStatus();
  showAuthBannerIfNeeded();
}

function showAuthBannerIfNeeded() {
  // Remove existing banner if any
  const existing = document.getElementById("authBanner");
  if (existing) existing.remove();

  if (state.authMode !== "local_trust") return;

  const banner = document.createElement("div");
  banner.id = "authBanner";
  banner.className = "auth-banner";
  banner.innerHTML = `
    <span>${t("app.auth_banner")}</span>
    <button class="auth-banner-close" aria-label="${t("common.aria_close")}">&times;</button>
  `;
  banner.querySelector(".auth-banner-close").addEventListener("click", () => banner.remove());

  const shell = document.querySelector(".app-shell");
  if (shell) shell.parentElement.insertBefore(banner, shell);
}

setStartDashboard(startDashboard);

// ── Mobile Navigation ────────────────────────

const _SIDEBAR_COLLAPSED_KEY = "aw-sidebar-collapsed";

function _isMobileViewport() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function _applySidebarCollapsed(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const btn = document.getElementById("sidebarCollapseBtn");
  if (!btn) return;
  btn.setAttribute("aria-pressed", collapsed ? "true" : "false");
  btn.setAttribute("title", collapsed ? "サイドバーを開く" : "サイドバーを閉じる");
}

function initDesktopSidebarCollapse() {
  const btn = document.getElementById("sidebarCollapseBtn");
  if (!btn) return;

  const initialCollapsed = localStorage.getItem(_SIDEBAR_COLLAPSED_KEY) === "1";
  _applySidebarCollapsed(initialCollapsed);

  btn.addEventListener("click", () => {
    if (_isMobileViewport()) return;
    const nextCollapsed = !document.body.classList.contains("sidebar-collapsed");
    _applySidebarCollapsed(nextCollapsed);
    localStorage.setItem(_SIDEBAR_COLLAPSED_KEY, nextCollapsed ? "1" : "0");
  });
}

function initMobileNav() {
  const hamburgerBtn = document.getElementById("hamburgerBtn");
  const backdrop = document.getElementById("mobileNavBackdrop");
  const sidebarNav = document.getElementById("sidebarNav");

  function openNav() {
    document.body.classList.add("mobile-nav-open");
  }

  function closeNav() {
    document.body.classList.remove("mobile-nav-open");
  }

  if (hamburgerBtn) {
    hamburgerBtn.addEventListener("click", () => {
      if (document.body.classList.contains("mobile-nav-open")) {
        closeNav();
      } else {
        openNav();
      }
    });
  }

  if (backdrop) {
    backdrop.addEventListener("click", closeNav);
  }

  if (sidebarNav) {
    sidebarNav.addEventListener("click", (e) => {
      if (e.target.closest(".nav-item")) {
        closeNav();
      }
    });
  }
}

// ── Init ────────────────────────────────────

async function init() {
  // i18n (before auth so UI looks correct)
  await initI18n();
  applyTranslations();

  // Theme (before auth so UI looks correct immediately)
  await initTheme();
  const themeSwitch = document.getElementById("themeSwitch");
  if (themeSwitch) {
    themeSwitch.addEventListener("change", (e) => {
      applyTheme(e.target.checked ? "business" : "default");
    });
  }

  // Logout button binding
  document.getElementById("logoutBtn").addEventListener("click", logout);

  // Mobile navigation
  initMobileNav();
  initDesktopSidebarCollapse();

  // Try to authenticate via existing session cookie
  const authenticated = await checkAuth();
  if (authenticated) {
    hideLoginScreen();
    await startDashboard();
  } else {
    showLoginScreen();
  }

  // Periodic refresh: system status every 60s
  setInterval(loadSystemStatus, 60000);
}

// Start when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
