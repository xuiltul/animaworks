// ── User Management ─────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("users.page_title")}</h2>
    </div>
    <div id="usersContent">
      <div class="loading-placeholder">${t("common.loading")}</div>
    </div>
  `;

  _loadUsers();
}

export function destroy() {
  // No intervals to clean up
}

// ── Data Loading ───────────────────────────

async function _loadUsers() {
  const content = document.getElementById("usersContent");
  if (!content) return;

  try {
    const users = await api("/api/shared/users");

    if (!users || users.length === 0) {
      content.innerHTML = `
        <div class="card">
          <div class="card-body">
            <div class="loading-placeholder">${t("users.no_users")}</div>
            <p style="text-align:center; color:var(--text-secondary, #666); margin-top:0.5rem;">
              ${t("users.auto_create_hint")}
            </p>
          </div>
        </div>
      `;
      return;
    }

    content.innerHTML = `
      <div class="card-grid" style="grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));">
        ${users.map(name => `
          <div class="card">
            <div class="card-body" style="text-align:center; padding:1.25rem;">
              <div class="anima-avatar-placeholder" style="width:56px;height:56px;font-size:1.5rem;margin:0 auto 0.75rem;">
                ${escapeHtml(name.charAt(0).toUpperCase())}
              </div>
              <div style="font-weight:600; font-size:1.05rem; margin-bottom:0.25rem;">${escapeHtml(name)}</div>
              <div style="color:var(--text-secondary, #666); font-size:0.85rem;">${t("users.shared_user")}</div>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  } catch (err) {
    content.innerHTML = `
      <div class="card">
        <div class="card-body">
          <div class="loading-placeholder">${t("users.fetch_failed")}: ${escapeHtml(err.message)}</div>
        </div>
      </div>
    `;
  }
}
