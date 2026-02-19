/* ============================================
   AnimaWorks — Dashboard App (Entry Point)
   ============================================ */

import { state } from "./state.js";
import { connectWebSocket } from "./websocket.js";
// state.authMode is set by login.js checkAuth()
import { loadSystemStatus } from "./status.js";
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
    <span>パスワードが未設定です。セキュリティのため、<a href="#/setup">セットアップページ</a>でパスワードを設定してください。</span>
    <button class="auth-banner-close" aria-label="閉じる">&times;</button>
  `;
  banner.querySelector(".auth-banner-close").addEventListener("click", () => banner.remove());

  const shell = document.querySelector(".app-shell");
  if (shell) shell.parentElement.insertBefore(banner, shell);
}

setStartDashboard(startDashboard);

// ── Mobile Navigation ────────────────────────

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
  // Logout button binding
  document.getElementById("logoutBtn").addEventListener("click", logout);

  // Mobile navigation
  initMobileNav();

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
