/* ── Login / Logout ────────────────────────── */

import { state, dom, escapeHtml } from "./state.js";
import { api } from "./api.js";
import { initI18n, t } from "/shared/i18n.js";

let _startDashboard = null;

export function setStartDashboard(fn) {
  _startDashboard = fn;
}

export async function initLoginScreen() {
  await initI18n();
  dom.loginScreen.innerHTML = `
    <div class="login-card">
      <h2 class="login-title">AnimaWorks</h2>
      <p class="login-subtitle">${t("login.subtitle")}</p>
      <form id="loginForm" class="login-form">
        <input type="text" id="loginUsername" placeholder="${t("login.placeholder_username")}" autocomplete="username" required>
        <input type="password" id="loginPassword" placeholder="${t("login.placeholder_password")}" autocomplete="current-password" required>
        <div id="loginError" class="login-error hidden"></div>
        <button type="submit" class="btn-login">${t("login.button")}</button>
      </form>
    </div>
  `;

  document.getElementById("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("loginUsername").value.trim();
    const password = document.getElementById("loginPassword").value;
    const errorEl = document.getElementById("loginError");
    errorEl.classList.add("hidden");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        errorEl.textContent = data.error || t("login.error");
        errorEl.classList.remove("hidden");
        return;
      }

      // Get user info
      const user = await res.json();
      state.currentUser = user.username;
      state.currentUserRole = user.role;
      hideLoginScreen();
      if (_startDashboard) _startDashboard();
    } catch (err) {
      errorEl.textContent = t("login.error_network");
      errorEl.classList.remove("hidden");
    }
  });
}

export async function checkAuth() {
  try {
    const res = await fetch("/api/auth/me", { credentials: "same-origin" });
    if (res.ok) {
      const user = await res.json();
      state.currentUser = user.username;
      state.currentUserRole = user.role;
      state.authMode = user.auth_mode || null;
      return true;
    }
  } catch { /* not authenticated */ }
  return false;
}

export async function logout() {
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
    });
  } catch { /* ignore */ }
  state.currentUser = null;
  state.currentUserRole = null;
  localStorage.removeItem("animaworks_user");
  showLoginScreen();
}

export function showLoginScreen() {
  dom.loginScreen.classList.remove("hidden");
  initLoginScreen();
}

export function hideLoginScreen() {
  dom.loginScreen.classList.add("hidden");
  const label = state.currentUser || "";
  if (dom.currentUserLabel) dom.currentUserLabel.textContent = label;
}
