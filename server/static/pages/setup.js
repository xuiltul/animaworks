// ── Setup Status ────────────────────────────
import { api } from "../modules/api.js";
import { escapeHtml } from "../modules/state.js";

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>セットアップ状況</h2>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">初期化チェックリスト</div>
      <div class="card-body" id="setupChecklist">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">現在の設定</div>
      <div class="card-body" id="setupConfig">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">APIキー設定状況</div>
      <div class="card-body" id="setupApiKeys">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">認証設定</div>
      <div class="card-body" id="authSettings">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>
  `;

  _loadChecklist();
  _loadConfig();
  _loadAuthSettings();
}

export function destroy() {
  // No intervals to clean up
}

// ── Checklist ──────────────────────────────

async function _loadChecklist() {
  const el = document.getElementById("setupChecklist");
  if (!el) return;

  let initData = null;
  let systemData = null;
  let animas = [];

  // Try init-status endpoint first
  try {
    initData = await api("/api/system/init-status");
  } catch { /* may not exist */ }

  // Fallback: infer from other endpoints
  try {
    systemData = await api("/api/system/status");
  } catch { /* ignore */ }

  try {
    animas = await api("/api/animas");
  } catch { /* ignore */ }

  const checks = [];

  if (initData) {
    // Use init-status data
    if (Array.isArray(initData.checks)) {
      // Structured checks array format
      for (const item of initData.checks) {
        const label = item.detail ? `${item.label} (${item.detail})` : item.label;
        checks.push({ label, ok: !!item.ok });
      }
    } else {
      // Legacy flat object format — use truthy evaluation
      const items = initData.checks || initData;
      if (typeof items === "object") {
        for (const [key, val] of Object.entries(items)) {
          const ok = !!val && val !== "error" && val !== "missing";
          checks.push({ label: key, ok });
        }
      }
    }
  } else {
    // Infer status
    checks.push({ label: "サーバー稼働中", ok: !!systemData });
    checks.push({ label: "パーソンディレクトリ", ok: animas.length > 0 });
    checks.push({ label: "スケジューラ", ok: systemData?.scheduler_running ?? false });
  }

  if (checks.length === 0) {
    el.innerHTML = '<div class="loading-placeholder">チェック項目を取得できませんでした</div>';
    return;
  }

  el.innerHTML = checks.map(c => `
    <div style="display:flex; align-items:center; gap:0.75rem; padding:0.5rem 0; border-bottom:1px solid var(--border-color, #eee);">
      <span style="font-size:1.2rem;">${c.ok ? "\u2705" : "\u274C"}</span>
      <span>${escapeHtml(c.label)}</span>
    </div>
  `).join("");
}

// ── Config ────────────────────────────────

async function _loadConfig() {
  const configEl = document.getElementById("setupConfig");
  const keysEl = document.getElementById("setupApiKeys");
  if (!configEl) return;

  try {
    const config = await api("/api/system/config");

    // Render config table
    const rows = [];
    function flattenConfig(obj, prefix = "") {
      for (const [key, val] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (val && typeof val === "object" && !Array.isArray(val)) {
          flattenConfig(val, fullKey);
        } else {
          // Mask potential secrets
          let displayVal = String(val);
          if (/key|secret|token|password/i.test(key) && displayVal.length > 4) {
            displayVal = displayVal.slice(0, 4) + "****";
          }
          rows.push({ key: fullKey, val: displayVal });
        }
      }
    }
    flattenConfig(config);

    configEl.innerHTML = `
      <div class="data-table-wrapper">
        <table class="data-table">
          <thead><tr><th>設定項目</th><th>値</th></tr></thead>
          <tbody>
            ${rows.map(r => `<tr><td>${escapeHtml(r.key)}</td><td><code>${escapeHtml(r.val)}</code></td></tr>`).join("")}
          </tbody>
        </table>
      </div>
    `;

    // Extract API key info
    if (keysEl) {
      const keyEntries = rows.filter(r => /key|api_key|token/i.test(r.key));
      if (keyEntries.length > 0) {
        keysEl.innerHTML = keyEntries.map(k => {
          const configured = k.val && k.val !== "null" && k.val !== "None" && k.val !== "";
          return `
            <div style="display:flex; align-items:center; gap:0.75rem; padding:0.5rem 0; border-bottom:1px solid var(--border-color, #eee);">
              <span style="font-size:1.2rem;">${configured ? "\u2705" : "\u274C"}</span>
              <span style="font-weight:500;">${escapeHtml(k.key)}</span>
              <code style="margin-left:auto; color:var(--text-secondary, #666);">${escapeHtml(k.val)}</code>
            </div>
          `;
        }).join("");
      } else {
        keysEl.innerHTML = '<div class="loading-placeholder">APIキー情報が設定に含まれていません</div>';
      }
    }
  } catch {
    configEl.innerHTML = '<div class="loading-placeholder">APIが未実装です</div>';
    if (keysEl) keysEl.innerHTML = '<div class="loading-placeholder">設定APIが利用できません</div>';
  }
}

// ── Auth Settings ──────────────────────────

async function _loadAuthSettings() {
  const el = document.getElementById("authSettings");
  if (!el) return;

  try {
    const me = await api("/api/auth/me");
    const users = await api("/api/users");

    let html = "";

    // Auth mode info
    const modeLabel = { local_trust: "ローカルトラスト（未認証）", password: "パスワード認証", multi_user: "マルチユーザー認証" };
    html += `<div style="margin-bottom: 1rem;">
      <strong>認証モード:</strong> <code>${escapeHtml(modeLabel[me.auth_mode] || me.auth_mode || "不明")}</code>
    </div>`;

    // Password change / initial setup form
    const isInitial = me.auth_mode === "local_trust";
    html += `
      <div style="margin-bottom: 1.5rem;">
        <h4 style="margin-bottom: 0.5rem;">${isInitial ? "パスワード設定" : "パスワード変更"}</h4>
        <form id="changePasswordForm" style="display:flex; flex-direction:column; gap:0.5rem; max-width:300px;">
          ${isInitial ? "" : '<input type="password" id="currentPassword" placeholder="現在のパスワード" required>'}
          <input type="password" id="newPassword" placeholder="新しいパスワード" required>
          <input type="password" id="confirmPassword" placeholder="新しいパスワード（確認）" required>
          <div id="pwChangeResult" class="login-error hidden"></div>
          <button type="submit" class="btn-login" style="width:auto;">${isInitial ? "設定" : "変更"}</button>
        </form>
      </div>
    `;

    // User management (owner only)
    if (me.role === "owner") {
      html += `
        <div style="margin-bottom: 1.5rem;">
          <h4 style="margin-bottom: 0.5rem;">ユーザー管理</h4>
          <table class="data-table">
            <thead><tr><th>ユーザー名</th><th>表示名</th><th>ロール</th><th>操作</th></tr></thead>
            <tbody>
              ${users.map(u => `
                <tr>
                  <td>${escapeHtml(u.username)}</td>
                  <td>${escapeHtml(u.display_name)}</td>
                  <td>${escapeHtml(u.role)}</td>
                  <td>${u.role !== "owner" ? `<button class="btn-delete-user" data-user="${escapeHtml(u.username)}" style="color:#ef4444;cursor:pointer;border:none;background:none;">削除</button>` : "-"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>

        <div>
          <h4 style="margin-bottom: 0.5rem;">ユーザー追加</h4>
          <form id="addUserForm" style="display:flex; flex-direction:column; gap:0.5rem; max-width:300px;">
            <input type="text" id="newUsername" placeholder="ユーザー名" required>
            <input type="text" id="newDisplayName" placeholder="表示名">
            <input type="password" id="newUserPassword" placeholder="パスワード" required>
            <div id="addUserResult" class="login-error hidden"></div>
            <button type="submit" class="btn-login" style="width:auto;">追加</button>
          </form>
        </div>
      `;
    }

    el.innerHTML = html;

    // Password change handler
    const pwForm = document.getElementById("changePasswordForm");
    if (pwForm) {
      pwForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const result = document.getElementById("pwChangeResult");
        const newPw = document.getElementById("newPassword").value;
        const confirmPw = document.getElementById("confirmPassword").value;

        if (newPw !== confirmPw) {
          result.textContent = "新しいパスワードが一致しません";
          result.classList.remove("hidden");
          return;
        }

        try {
          const curPwEl = document.getElementById("currentPassword");
          const res = await fetch("/api/users/me/password", {
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
            result.textContent = isInitial ? "パスワードを設定しました。ページをリロードします…" : "パスワードを変更しました";
            result.style.color = "#22c55e";
            result.classList.remove("hidden");
            pwForm.reset();
            if (isInitial) setTimeout(() => location.reload(), 1000);
          } else {
            result.style.color = "#ef4444";
            result.textContent = data.error || "変更に失敗しました";
            result.classList.remove("hidden");
          }
        } catch {
          result.style.color = "#ef4444";
          result.textContent = "通信エラー";
          result.classList.remove("hidden");
        }
      });
    }

    // Delete user handlers
    el.querySelectorAll(".btn-delete-user").forEach(btn => {
      btn.addEventListener("click", async () => {
        const username = btn.dataset.user;
        if (!confirm(`ユーザー "${username}" を削除しますか？`)) return;

        try {
          const res = await fetch(`/api/users/${username}`, {
            method: "DELETE",
            credentials: "same-origin",
          });
          if (res.ok) {
            _loadAuthSettings(); // Reload
          } else {
            const data = await res.json();
            alert(data.error || "削除に失敗しました");
          }
        } catch {
          alert("通信エラー");
        }
      });
    });

    // Add user handler
    const addForm = document.getElementById("addUserForm");
    if (addForm) {
      addForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const result = document.getElementById("addUserResult");

        try {
          const res = await fetch("/api/users", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "same-origin",
            body: JSON.stringify({
              username: document.getElementById("newUsername").value.trim(),
              display_name: document.getElementById("newDisplayName").value.trim(),
              password: document.getElementById("newUserPassword").value,
            }),
          });
          const data = await res.json();
          if (res.ok) {
            result.textContent = `ユーザー "${data.username}" を追加しました`;
            result.style.color = "#22c55e";
            result.classList.remove("hidden");
            addForm.reset();
            _loadAuthSettings(); // Reload user list
          } else {
            result.style.color = "#ef4444";
            result.textContent = data.error || "追加に失敗しました";
            result.classList.remove("hidden");
          }
        } catch {
          result.style.color = "#ef4444";
          result.textContent = "通信エラー";
          result.classList.remove("hidden");
        }
      });
    }
  } catch {
    el.innerHTML = '<div class="loading-placeholder">認証設定を取得できませんでした（ローカルトラストモード）</div>';
  }
}
