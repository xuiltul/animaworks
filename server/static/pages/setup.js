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

    <div class="card">
      <div class="card-header">APIキー設定状況</div>
      <div class="card-body" id="setupApiKeys">
        <div class="loading-placeholder">読み込み中...</div>
      </div>
    </div>
  `;

  _loadChecklist();
  _loadConfig();
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
  let persons = [];

  // Try init-status endpoint first
  try {
    initData = await api("/api/system/init-status");
  } catch { /* may not exist */ }

  // Fallback: infer from other endpoints
  try {
    systemData = await api("/api/system/status");
  } catch { /* ignore */ }

  try {
    persons = await api("/api/persons");
  } catch { /* ignore */ }

  const checks = [];

  if (initData) {
    // Use init-status data
    const items = initData.checks || initData;
    if (typeof items === "object") {
      for (const [key, val] of Object.entries(items)) {
        const ok = val === true || val === "ok" || val === "configured";
        checks.push({ label: key, ok });
      }
    }
  } else {
    // Infer status
    checks.push({ label: "サーバー稼働中", ok: !!systemData });
    checks.push({ label: "パーソンディレクトリ", ok: persons.length > 0 });
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
      <table class="data-table">
        <thead><tr><th>設定項目</th><th>値</th></tr></thead>
        <tbody>
          ${rows.map(r => `<tr><td>${escapeHtml(r.key)}</td><td><code>${escapeHtml(r.val)}</code></td></tr>`).join("")}
        </tbody>
      </table>
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
