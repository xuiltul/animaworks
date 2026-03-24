/* ============================================
   AnimaWorks — Dashboard App (Entry Point)
   ============================================ */

import { initI18n, applyTranslations, t } from "/shared/i18n.js";
import { state } from "./state.js";
import { connectWebSocket } from "./websocket.js";
// state.authMode is set by login.js checkAuth()
import { loadSystemStatus } from "./status.js";

// ── Display Mode ─────────────────────────────

export function applyDisplayMode(mode) {
  document.body.classList.remove("mode-anime", "mode-realistic");
  document.body.classList.add(`mode-${mode}`);
  localStorage.setItem("aw-display-mode", mode);
}

export function getDisplayMode() {
  return localStorage.getItem("aw-display-mode") || "realistic";
}

function initDisplayMode() {
  applyDisplayMode(getDisplayMode());
}

// ── Font Size ────────────────────────────────
const _LS_FONT_SIZE = "aw-font-size";
const _FONT_SIZE_DEFAULT = 14;

export function applyFontSize(px) {
  const clamped = Math.min(22, Math.max(10, Number(px) || _FONT_SIZE_DEFAULT));
  document.documentElement.style.setProperty("--aw-font-size-base", `${clamped}px`);
  localStorage.setItem(_LS_FONT_SIZE, String(clamped));
}

export function getFontSize() {
  return parseInt(localStorage.getItem(_LS_FONT_SIZE) || String(_FONT_SIZE_DEFAULT), 10);
}

function initFontSize() {
  applyFontSize(getFontSize());
}

// ── Theme ────────────────────────────────────

const ALL_THEMES = [
  "business", "graphite", "ocean", "forest", "sunset",
  "rose", "lavender", "nord", "monokai", "midnight", "solarized"
];

const DARK_THEMES = ["monokai", "midnight"];

export function applyTheme(theme) {
  state.uiTheme = theme;
  ALL_THEMES.forEach(t => document.body.classList.remove(`theme-${t}`));
  if (theme !== "default") {
    document.body.classList.add(`theme-${theme}`);
  }
  if (DARK_THEMES.includes(theme)) {
    document.body.setAttribute("data-theme", "dark");
  } else {
    document.body.removeAttribute("data-theme");
  }
  localStorage.setItem("aw-theme", theme);
}

async function initTheme() {
  const mql = window.matchMedia("(prefers-color-scheme: dark)");

  mql.addEventListener("change", (e) => {
    if (!localStorage.getItem("aw-theme")) {
      applyTheme(e.matches ? "midnight" : "default");
    }
  });

  const stored = localStorage.getItem("aw-theme");
  if (stored) {
    applyTheme(stored);
    return;
  }

  let serverTheme = "default";
  try {
    const resp = await fetch("/api/system/config");
    if (resp.ok) {
      const cfg = await resp.json();
      if (cfg?.ui?.theme) {
        serverTheme = cfg.ui.theme;
      }
    }
  } catch { /* ignore */ }

  if (serverTheme === "default" && mql.matches) {
    applyTheme("midnight");
  } else {
    applyTheme(serverTheme);
  }
}
import { checkAuth, logout, showLoginScreen, hideLoginScreen, setStartDashboard } from "./login.js";
import { initRouter } from "./router.js";

// ── Demo Mode ────────────────────────────────

async function initDemoMode() {
  try {
    const resp = await fetch("/api/system/config");
    if (resp.ok) {
      const cfg = await resp.json();
      state.demoMode = cfg?.ui?.demo_mode === true;
    } else {
      state.demoMode = false;
    }
  } catch {
    state.demoMode = false;
  }
}

// ── Dashboard Startup ───────────────────────

async function startDashboard() {
  await initDemoMode();
  initRouter("pageContent");
  connectWebSocket();
  loadSystemStatus();
  showAuthBannerIfNeeded();
  if (state.demoMode) showDemoSplashIfNeeded();
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

function showDemoSplashIfNeeded() {
  if (localStorage.getItem("aw-demo-splash-seen")) return;

  const splash = document.createElement("div");
  splash.id = "demoSplash";
  splash.className = "demo-splash";

  const locale = document.documentElement.lang || "ja";
  const isJa = locale.startsWith("ja");

  splash.innerHTML = `
    <div class="demo-splash-inner">
      <h1 class="demo-splash-title">AnimaWorks Demo</h1>
      <p class="demo-splash-subtitle">${t("demo.splash_subtitle")}</p>
      <div class="demo-splash-achievements">
        <div class="demo-splash-card">
          <div class="demo-splash-card-icon">📊</div>
          <div class="demo-splash-card-name">${isJa ? "Sora" : "Kai"} <span class="demo-splash-card-role">${t("demo.splash_role_engineer")}</span></div>
          <p class="demo-splash-card-text">${t("demo.splash_engineer_desc")}</p>
        </div>
        <div class="demo-splash-card">
          <div class="demo-splash-card-icon">🤝</div>
          <div class="demo-splash-card-name">${isJa ? "Hina" : "Nova"} <span class="demo-splash-card-role">${t("demo.splash_role_assistant")}</span></div>
          <p class="demo-splash-card-text">${t("demo.splash_assistant_desc")}</p>
        </div>
        <div class="demo-splash-card">
          <div class="demo-splash-card-icon">🎯</div>
          <div class="demo-splash-card-name">${isJa ? "Kaito" : "Alex"} <span class="demo-splash-card-role">${t("demo.splash_role_pm")}</span></div>
          <p class="demo-splash-card-text">${t("demo.splash_pm_desc")}</p>
        </div>
      </div>
      <div class="demo-splash-actions">
        <button class="demo-splash-btn demo-splash-btn-primary" data-action="chat">${t("demo.splash_cta_chat")}</button>
        <button class="demo-splash-btn demo-splash-btn-secondary" data-action="activity">${t("demo.splash_cta_activity")}</button>
      </div>
    </div>
  `;

  splash.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    localStorage.setItem("aw-demo-splash-seen", "1");
    splash.remove();
    const existing = document.getElementById("authBanner");
    if (existing) existing.remove();
    if (btn.dataset.action === "activity") {
      window.location.hash = "#/activity";
    }
  });

  document.body.prepend(splash);
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
  btn.setAttribute("title", collapsed ? t("nav.sidebar_open") : t("nav.sidebar_close"));
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

  // Display mode, theme & font size (before auth so UI looks correct immediately)
  initDisplayMode();
  initFontSize();
  await initTheme();

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

// Start when DOM is ready (guarded against double-init from duplicate module URLs)
if (!window.__awAppInitialized) {
  window.__awAppInitialized = true;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
