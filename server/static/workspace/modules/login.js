// ── Login Screen ──────────────────────
// User selection login UI. Renders user buttons + guest login.

import { getState, setState } from "./state.js";
import { fetchSharedUsers } from "./api.js";
import { escapeHtml } from "./utils.js";

// ── Private State ──────────────────────

let _container = null;
let _onLoginSuccess = null;

// ── Render ──────────────────────

function renderLoginScreen() {
  if (!_container) return;
  _container.innerHTML = `
    <div class="login-screen" id="wsLoginScreen">
      <div class="login-card">
        <div class="login-title">AnimaWorks</div>
        <div class="login-subtitle">Digital Anima Workspace</div>
        <div class="user-list" id="wsUserList">
          <div class="loading-placeholder">Loading users...</div>
        </div>
        <div class="login-guest">
          <button class="btn-guest" id="wsGuestLoginBtn">Login as Guest</button>
        </div>
      </div>
    </div>
  `;

  // Bind guest button
  const guestBtn = _container.querySelector("#wsGuestLoginBtn");
  if (guestBtn) {
    guestBtn.addEventListener("click", () => loginAs("guest"));
  }

  // Load user list
  loadUsers();
}

async function loadUsers() {
  const userListEl = _container?.querySelector("#wsUserList");
  if (!userListEl) return;

  try {
    const users = await fetchSharedUsers();
    if (!users || users.length === 0) {
      userListEl.innerHTML = '<p style="color:#999;font-size:0.85rem;">No registered users</p>';
      return;
    }

    userListEl.innerHTML = users
      .map(
        (name) =>
          `<button class="user-btn" data-user="${escapeHtml(name)}">${escapeHtml(name)}</button>`
      )
      .join("");

    // Bind click events
    userListEl.querySelectorAll(".user-btn").forEach((btn) => {
      btn.addEventListener("click", () => loginAs(btn.dataset.user));
    });
  } catch (err) {
    console.error("Failed to load shared users:", err);
    userListEl.innerHTML =
      '<p style="color:#ef4444;font-size:0.85rem;">Failed to load users</p>';
  }
}

function loginAs(username) {
  setState({ currentUser: username });
  localStorage.setItem("animaworks_user", username);
  hideLogin();
  if (_onLoginSuccess) {
    _onLoginSuccess(username);
  }
}

// ── Public API ──────────────────────

/**
 * Render the login UI into the given container element.
 */
export function showLogin(container) {
  _container = container || _container;
  if (!_container) return;
  renderLoginScreen();
  const screen = _container.querySelector("#wsLoginScreen");
  if (screen) {
    screen.classList.remove("hidden");
  }
}

/**
 * Hide the login screen overlay.
 */
export function hideLogin() {
  if (!_container) return;
  const screen = _container.querySelector("#wsLoginScreen");
  if (screen) {
    screen.classList.add("hidden");
  }
}

/**
 * Return the current logged-in username, or null.
 */
export function getCurrentUser() {
  return getState().currentUser;
}

/**
 * Initialize the login module.
 * @param {HTMLElement} container - DOM element to render login UI into
 * @param {function} onLoginSuccess - Callback invoked with username on successful login
 */
export function initLogin(container, onLoginSuccess) {
  _container = container;
  _onLoginSuccess = onLoginSuccess;

  const saved = getState().currentUser;
  if (saved) {
    // Already logged in — skip rendering
    return;
  }
  showLogin(container);
}

/**
 * Log out the current user. Clears state and localStorage.
 */
export function logout() {
  setState({ currentUser: null });
  localStorage.removeItem("animaworks_user");
  showLogin(_container);
}
