/* ============================================
   AnimaWorks — Dashboard App (Entry Point)
   ============================================ */

import { state } from "./state.js";
import { connectWebSocket } from "./websocket.js";
import { loadSystemStatus } from "./status.js";
import { loginAs, logout, showLoginScreen, hideLoginScreen, setStartDashboard } from "./login.js";
import { initRouter } from "./router.js";

// ── Dashboard Startup ───────────────────────

async function startDashboard() {
  initRouter("pageContent");
  connectWebSocket();
  loadSystemStatus();
}

setStartDashboard(startDashboard);

// ── Init ────────────────────────────────────

async function init() {
  // Login/logout bindings
  document.getElementById("guestLoginBtn").addEventListener("click", () => loginAs("human"));
  document.getElementById("logoutBtn").addEventListener("click", logout);

  if (state.currentUser) {
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
