// ── AuthUser Management ─────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";
import { t } from "/shared/i18n.js";
import { basePath } from "/shared/base-path.js";

let _container = null;
let _currentUser = null;

export function render(container) {
  _container = container;
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
  _container = null;
  _currentUser = null;
}

// ── Data Loading ───────────────────────────

async function _loadUsers() {
  const content = document.getElementById("usersContent");
  if (!content) return;

  try {
    const [me, users] = await Promise.all([
      api("/api/auth/me").catch(() => null),
      api("/api/users"),
    ]);
    _currentUser = me;

    if (!users || users.length === 0) {
      content.innerHTML = `
        <div class="card">
          <div class="card-body">
            <div class="loading-placeholder">${t("users.no_users")}</div>
          </div>
        </div>
        ${_renderPasswordSection(me)}
      `;
      _bindPasswordForm(me);
      return;
    }

    const isOwner = me && me.role === "owner";

    content.innerHTML = `
      <div class="card" style="margin-bottom:1.5rem;">
        <div class="card-body">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t("users.col_username")}</th>
                <th>${t("users.col_role")}</th>
                <th>${t("users.col_created_at")}</th>
                <th>${t("users.col_actions")}</th>
              </tr>
            </thead>
            <tbody>
              ${users.map((u) => _renderUserRow(u, me, isOwner)).join("")}
            </tbody>
          </table>
        </div>
      </div>

      ${isOwner ? _renderAddUserForm() : ""}
      ${_renderPasswordSection(me)}
    `;

    if (isOwner) {
      _bindAddUserForm();
      content.querySelectorAll(".btn-delete-user").forEach((btn) => {
        btn.addEventListener("click", () => _onDeleteUser(btn.dataset.user));
      });
    }
    _bindPasswordForm(me);
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

function _renderUserRow(user, me, isOwner) {
  const isOwnerAccount = user.role === "owner";
  const isSelf = me && me.username === user.username;
  const canDelete = isOwner && !isOwnerAccount && !isSelf;
  const created = user.created_at ? _formatDate(user.created_at) : "-";

  let actions = "-";
  if (isOwner) {
    if (canDelete) {
      actions = `<button class="btn-delete-user" data-user="${escapeHtml(user.username)}"
        style="color:#ef4444;cursor:pointer;border:none;background:none;">
        ${t("users.delete")}
      </button>`;
    } else if (isOwnerAccount || isSelf) {
      actions = `<span style="color:var(--text-secondary,#666);font-size:0.85rem;">${t("users.delete_disabled")}</span>`;
    }
  }

  return `
    <tr>
      <td>${escapeHtml(user.username)}</td>
      <td>${escapeHtml(user.role || "")}</td>
      <td>${escapeHtml(created)}</td>
      <td>${actions}</td>
    </tr>
  `;
}

function _formatDate(iso) {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

function _renderAddUserForm() {
  return `
    <div class="card" style="margin-bottom:1.5rem;">
      <div class="card-body">
        <h3 style="margin:0 0 0.75rem;font-size:1.05rem;">${t("users.add_title")}</h3>
        <form id="addUserForm" style="display:flex;flex-direction:column;gap:0.5rem;max-width:320px;">
          <input type="text" id="newUsername" placeholder="${t("users.placeholder_username")}" required autocomplete="off">
          <input type="password" id="newUserPassword" placeholder="${t("users.placeholder_password")}" required autocomplete="new-password">
          <select id="newUserRole">
            <option value="user">${t("users.role_user")}</option>
            <option value="admin">${t("users.role_admin")}</option>
          </select>
          <div id="addUserResult" class="login-error hidden"></div>
          <button type="submit" class="btn-login" style="width:auto;">${t("users.add_btn")}</button>
        </form>
      </div>
    </div>
  `;
}

function _renderPasswordSection(me) {
  if (!me) return "";
  const skipCurrent = me.auth_mode === "local_trust" || !me.has_password;
  return `
    <div class="card">
      <div class="card-body">
        <h3 style="margin:0 0 0.75rem;font-size:1.05rem;">${t("users.password_title")}</h3>
        <p style="color:var(--text-secondary,#666);font-size:0.85rem;margin:0 0 0.75rem;">
          ${t("users.password_hint", { username: me.username || "" })}
        </p>
        <form id="changePasswordForm" style="display:flex;flex-direction:column;gap:0.5rem;max-width:320px;">
          ${skipCurrent ? "" : `<input type="password" id="currentPassword" placeholder="${t("users.placeholder_current_password")}" required autocomplete="current-password">`}
          <input type="password" id="newPassword" placeholder="${t("users.placeholder_new_password")}" required autocomplete="new-password">
          <input type="password" id="confirmPassword" placeholder="${t("users.placeholder_confirm_password")}" required autocomplete="new-password">
          <div id="pwChangeResult" class="login-error hidden"></div>
          <button type="submit" class="btn-login" style="width:auto;">${t("users.password_btn")}</button>
        </form>
      </div>
    </div>
  `;
}

function _bindAddUserForm() {
  const form = document.getElementById("addUserForm");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const result = document.getElementById("addUserResult");
    try {
      const res = await fetch(`${basePath}/api/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          username: document.getElementById("newUsername").value.trim(),
          password: document.getElementById("newUserPassword").value,
          role: document.getElementById("newUserRole").value || "user",
        }),
      });
      const data = await res.json();
      if (res.ok) {
        result.textContent = t("users.add_success", { username: data.username });
        result.style.color = "#22c55e";
        result.classList.remove("hidden");
        form.reset();
        _loadUsers();
      } else {
        result.style.color = "#ef4444";
        result.textContent = data.error || t("users.add_failed");
        result.classList.remove("hidden");
      }
    } catch {
      result.style.color = "#ef4444";
      result.textContent = t("users.network_error");
      result.classList.remove("hidden");
    }
  });
}

async function _onDeleteUser(username) {
  if (!username) return;
  if (!confirm(t("users.delete_confirm", { username }))) return;
  try {
    const res = await fetch(`${basePath}/api/users/${encodeURIComponent(username)}`, {
      method: "DELETE",
      credentials: "same-origin",
    });
    if (res.ok) {
      _loadUsers();
    } else {
      const data = await res.json().catch(() => ({}));
      alert(data.error || t("users.delete_failed"));
    }
  } catch {
    alert(t("users.network_error"));
  }
}

function _bindPasswordForm(me) {
  const form = document.getElementById("changePasswordForm");
  if (!form || !me) return;
  const skipCurrent = me.auth_mode === "local_trust" || !me.has_password;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const result = document.getElementById("pwChangeResult");
    const newPw = document.getElementById("newPassword").value;
    const confirmPw = document.getElementById("confirmPassword").value;

    if (newPw !== confirmPw) {
      result.textContent = t("users.password_mismatch");
      result.style.color = "#ef4444";
      result.classList.remove("hidden");
      return;
    }

    try {
      const curPwEl = document.getElementById("currentPassword");
      const res = await fetch(`${basePath}/api/users/me/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          current_password: curPwEl ? curPwEl.value : "",
          new_password: newPw,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        result.textContent = t("users.password_success");
        result.style.color = "#22c55e";
        result.classList.remove("hidden");
        form.reset();
        if (skipCurrent) setTimeout(() => location.reload(), 1000);
      } else {
        result.style.color = "#ef4444";
        result.textContent = data.error || t("users.password_failed");
        result.classList.remove("hidden");
      }
    } catch {
      result.style.color = "#ef4444";
      result.textContent = t("users.network_error");
      result.classList.remove("hidden");
    }
  });
}
