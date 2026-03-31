// ── Home Dashboard ──────────────────────────
import { t } from "/shared/i18n.js";
import { api } from "../modules/api.js";
import { escapeHtml, timeStr, statusClass } from "../modules/state.js";
import { animaHashColor } from "../modules/animas.js";
import { getIcon, getDisplaySummary } from "../shared/activity-types.js";
import { bustupCandidates, resolveAvatar } from "../modules/avatar-resolver.js";

let _refreshInterval = null;
let _usageInterval = null;

// ── Render ─────────────────────────────────

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("home.dashboard")}</h2>
    </div>

    <div class="usage-panel-header">
      <span></span>
      <button class="btn-secondary usage-refresh-btn" id="usageRefreshBtn">&#x21BB; Refresh</button>
    </div>
    <div class="usage-panel" id="homeUsagePanel">
      <div class="usage-card" id="usageCardClaude">
        <div class="usage-card-header">
          <span class="usage-provider-name">Claude</span>
          <span class="usage-sub-type" id="usageClaudeSub"></span>
        </div>
        <div class="usage-card-body" id="usageClaudeBody">
          <div class="usage-loading">${t("common.loading")}</div>
        </div>
      </div>
      <div class="usage-card" id="usageCardOpenai">
        <div class="usage-card-header">
          <span class="usage-provider-name">OpenAI</span>
          <span class="usage-sub-type" id="usageOpenaiSub"></span>
        </div>
        <div class="usage-card-body" id="usageOpenaiBody">
          <div class="usage-loading">${t("common.loading")}</div>
        </div>
      </div>
    </div>
    <div id="usageGovernorBar" style="display:none;"></div>

    <div class="card-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 1.5rem;">
      <div class="stat-card" id="homeStatAnimas">
        <div class="stat-label">${t("home.anima_count")}</div>
        <div class="stat-value" id="homeAnimaCount">--</div>
      </div>
      <div class="stat-card" id="homeStatScheduler">
        <div class="stat-label">${t("home.scheduler")}</div>
        <div class="stat-value" id="homeSchedulerStatus">--</div>
      </div>
      <div class="stat-card" id="homeStatProcesses">
        <div class="stat-label">${t("home.process_count")}</div>
        <div class="stat-value" id="homeProcessCount">--</div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">${t("home.anima_list")}</div>
      <div class="card-body">
        <div class="org-tree" id="homeOrgTree">
          <div class="loading-placeholder">${t("common.loading")}</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1.5rem;">
      <div class="card-header">${t("home.recent_activity")}</div>
      <div class="card-body">
        <div id="homeActivityTimeline">
          <div class="loading-placeholder">${t("common.loading")}</div>
        </div>
      </div>
    </div>

    <div class="card" id="homeExternalTasksCard" style="margin-bottom: 1.5rem;">
      <div class="card-header" id="extTasksHeader" style="cursor:pointer;display:flex;align-items:center;gap:0.5rem;">
        <span id="extTasksToggle" style="font-size:0.7rem;">&#x25BC;</span>
        ${t("home.external_tasks")}
        <span id="extTasksBadge" style="display:none;font-size:0.7rem;background:var(--accent-color,#2563eb);color:#fff;border-radius:10px;padding:0.1rem 0.5rem;margin-left:0.25rem;"></span>
        <span style="flex:1;"></span>
        <span id="extTasksLastUpdated" style="font-size:0.75rem;color:var(--text-secondary,#666);margin-right:0.5rem;"></span>
        <button id="extTasksRefresh" class="btn-icon" title="${t("home.ext_refresh")}" style="font-size:0.85rem;">&#x21BB;</button>
      </div>
      <div class="card-body" id="extTasksBody">
        <div id="extTasksList">
          <div class="loading-placeholder">${t("common.loading")}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">${t("home.quick_links")}</div>
      <div class="card-body" style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
        <a href="/workspace/" class="btn-primary" style="text-decoration:none;">${t("home.link_workspace")}</a>
        <a href="#/chat" class="btn-secondary" style="text-decoration:none;">${t("home.link_chat")}</a>
        <a href="#/animas" class="btn-secondary" style="text-decoration:none;">${t("home.link_animas")}</a>
        <a href="#/memory" class="btn-secondary" style="text-decoration:none;">${t("home.link_memory")}</a>
      </div>
    </div>
  `;

  _loadAll();
  _loadUsage();
  _initExternalTasksWidget();
  _refreshInterval = setInterval(_loadAll, 30000);
  _usageInterval = setInterval(_loadUsage, 60000);

  const refreshBtn = document.getElementById("usageRefreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => _loadUsage(true));
  }
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
  if (_usageInterval) {
    clearInterval(_usageInterval);
    _usageInterval = null;
  }
}

// ── Data Loading ───────────────────────────

async function _loadAll() {
  _loadSystemStatus();
  _loadOrgChart();
  _loadActivity();
  _loadExternalTasks();
}

// ── Usage Panel ────────────────────────────

function _resetToJst(value) {
  if (!value) return "";
  try {
    // Accept ISO string or unix seconds (number < 1e12 → seconds, else ms)
    let ms;
    if (typeof value === "number") {
      ms = value < 1e12 ? value * 1000 : value;
    } else {
      ms = new Date(value).getTime();
    }
    const d = new Date(ms);
    if (isNaN(d.getTime())) return String(value);
    const jst = new Date(d.getTime() + 9 * 3600000);
    const days = ["日","月","火","水","木","金","土"];
    const day = days[jst.getUTCDay()];
    const yyyy = jst.getUTCFullYear();
    const mm = String(jst.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(jst.getUTCDate()).padStart(2, "0");
    const hh = String(jst.getUTCHours()).padStart(2, "0");
    const mi = String(jst.getUTCMinutes()).padStart(2, "0");
    const ss = String(jst.getUTCSeconds()).padStart(2, "0");
    return `${yyyy}/${mm}/${dd}(${day}) ${hh}:${mi}:${ss}`;
  } catch { return String(value); }
}

function _remainingColor(remaining) {
  if (remaining <= 10) return "var(--aw-color-error, #dc2626)";
  if (remaining <= 30) return "var(--aw-color-warning, #d97706)";
  return "var(--aw-color-success, #16a34a)";
}

function _calcTimePct(resetAt, windowSeconds) {
  // Calculate time_remaining_pct for the time-proportional marker
  if (!resetAt || !windowSeconds) return null;
  const resetMs = (typeof resetAt === "number")
    ? (resetAt < 1e12 ? resetAt * 1000 : resetAt)
    : new Date(resetAt).getTime();
  if (isNaN(resetMs)) return null;
  const remainingSec = (resetMs - Date.now()) / 1000;
  if (remainingSec <= 0) return 100;
  const pct = (remainingSec / windowSeconds) * 100;
  return Math.min(pct, 100);
}

function _usageForecast(utilization, resetAt, windowSeconds) {
  if (!resetAt || !windowSeconds) return null;
  const resetMs = (typeof resetAt === "number")
    ? (resetAt < 1e12 ? resetAt * 1000 : resetAt)
    : new Date(resetAt).getTime();
  if (isNaN(resetMs)) return null;

  const now = Date.now();
  const windowMs = windowSeconds * 1000;
  const windowStartMs = resetMs - windowMs;
  const elapsedMs = now - windowStartMs;
  if (elapsedMs <= 0) return null;

  const msToReset = resetMs - now;
  if (msToReset <= 0) return { runway: "-", landing: "-", label: "" };

  const used = utilization;
  const remain = Math.max(0, 100 - used);
  const elapsedDays = elapsedMs / 86400000;
  const daysToReset = msToReset / 86400000;
  const burnPerDay = used / elapsedDays;

  if (burnPerDay <= 0.01) {
    return { runway: "\u221E", daysToReset, landing: `${remain.toFixed(1)}%`, label: "" };
  }

  const runwayDays = remain / burnPerDay;
  const projectedRemain = remain - (burnPerDay * daysToReset);

  // Format runway
  let runwayStr, daysToResetStr;
  if (windowSeconds <= 86400) {
    // sub-day windows: show in hours
    runwayStr = `${(runwayDays * 24).toFixed(1)}h`;
    daysToResetStr = `${(daysToReset * 24).toFixed(1)}h`;
  } else {
    runwayStr = `${runwayDays.toFixed(1)}d`;
    daysToResetStr = `${daysToReset.toFixed(1)}d`;
  }

  const delta = runwayDays - daysToReset;
  let deltaLabel = "";
  if (delta >= 0) {
    const deltaStr = windowSeconds <= 86400 ? `${(delta * 24).toFixed(1)}h` : `${delta.toFixed(1)}d`;
    deltaLabel = `<span style="color:var(--aw-color-success,#16a34a);">+${deltaStr}</span>`;
  } else {
    const deltaStr = windowSeconds <= 86400 ? `${(Math.abs(delta) * 24).toFixed(1)}h` : `${Math.abs(delta).toFixed(1)}d`;
    deltaLabel = `<span style="color:var(--aw-color-error,#dc2626);">\u25B2${deltaStr}</span>`;
  }

  const landingStr = `${projectedRemain.toFixed(1)}%`;
  const landingColor = projectedRemain < 0
    ? "var(--aw-color-error,#dc2626)"
    : projectedRemain < 10
      ? "var(--aw-color-warning,#d97706)"
      : "var(--aw-color-text-secondary,#666)";

  return { runway: runwayStr, daysToReset: daysToResetStr, deltaLabel, landing: landingStr, landingColor };
}

function _renderUsageBar(label, utilization, resetAt, windowSeconds) {
  const resetMs = resetAt
    ? (typeof resetAt === "number" ? (resetAt < 1e12 ? resetAt * 1000 : resetAt) : new Date(resetAt).getTime())
    : 0;
  const resetInPast = resetMs > 0 && resetMs <= Date.now();
  // Window already reset — treat as fully available
  const effectiveUtil = resetInPast ? 0 : utilization;
  const remaining = Math.max(0, 100 - effectiveUtil);
  const resetStr = resetInPast ? "" : (resetAt ? _resetToJst(resetAt) : "");
  const color = _remainingColor(remaining);

  // Time-proportional marker — shown on ALL windows (5h and 7d)
  const timePct = _calcTimePct(resetAt, windowSeconds);
  let markerHtml = "";
  if (timePct !== null && windowSeconds) {
    const markerLabel = `${timePct.toFixed(0)}%`;
    markerHtml = `<div class="usage-bar-time-marker" style="left:${timePct}%" data-label="${markerLabel}"></div>`;
  }

  // Deficit warning text
  let deficitHtml = "";
  if (timePct !== null && windowSeconds && timePct > remaining) {
    const gap = (timePct - remaining).toFixed(0);
    deficitHtml = `<span class="usage-deficit" style="color:var(--aw-color-error,#dc2626);font-size:0.7rem;margin-left:0.5rem;">-${gap}pt</span>`;
  }

  // Forecast: Runway + 着地予測
  let forecastHtml = "";
  const fc = resetInPast ? null : _usageForecast(utilization, resetAt, windowSeconds);
  if (fc) {
    forecastHtml = `<div class="usage-forecast">`;
    forecastHtml += `<span class="usage-forecast-item"><span class="usage-forecast-label">Runway</span> ${fc.runway}`;
    if (fc.daysToReset) forecastHtml += ` / ${fc.daysToReset}`;
    if (fc.deltaLabel) forecastHtml += ` ${fc.deltaLabel}`;
    forecastHtml += `</span>`;
    forecastHtml += `<span class="usage-forecast-item"><span class="usage-forecast-label">着地</span> <span style="color:${fc.landingColor || "inherit"}">${fc.landing}</span></span>`;
    forecastHtml += `</div>`;
  }

  return `
    <div class="usage-row">
      <div class="usage-row-header">
        <span class="usage-label">${escapeHtml(label)}</span>
        <span class="usage-pct" style="color:${color}">${remaining.toFixed(0)}%${deficitHtml}</span>
      </div>
      <div class="usage-bar-track">
        <div class="usage-bar-fill" style="width:${remaining}%;background:${color}"></div>
        ${markerHtml}
      </div>
      ${forecastHtml || `<div class="usage-forecast">&nbsp;</div>`}
      ${resetStr ? `<div class="usage-reset">${t("home.usage_reset")}: ${escapeHtml(resetStr)}</div>` : `<div class="usage-reset">&nbsp;</div>`}
    </div>
  `;
}

function _usageCanRelogin(errorCode) {
  return new Set(["rate_limited", "unauthorized", "no_credentials"]).has(errorCode);
}

function _renderUsageError(provider, data, msg) {
  const showButton = _usageCanRelogin(data.error);
  const buttonLabel = provider === "claude" ? "Claude 再認証" : "Codex ログイン";
  return `
    <div class="usage-error">${escapeHtml(msg)}</div>
    ${showButton ? `
      <div style="margin-top:0.75rem;">
        <button class="btn-secondary usage-auth-btn" data-provider="${provider}">${buttonLabel}</button>
      </div>
    ` : ""}
  `;
}

async function _runUsageRelogin(provider) {
  const path = provider === "claude" ? "/api/usage/claude/relogin" : "/api/usage/openai/relogin";
  try {
    const res = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json().catch(() => ({}));

    if (provider === "openai" && data.login_url) {
      window.open(data.login_url, "_blank", "noopener,noreferrer");
    }

    const lines = [];
    if (data.message) lines.push(data.message);
    if (data.device_code) lines.push(`Code: ${data.device_code}`);
    if (data.login_url) lines.push(`URL: ${data.login_url}`);
    if (!data.success && data.manual_command) {
      lines.push("");
      lines.push(`Terminal: ${data.manual_command}`);
    }
    alert(lines.join("\n") || (res.ok ? "Done" : "Failed"));
  } catch (err) {
    alert(err.message || "Failed");
  }
  await _loadUsage();
}

function _renderClaudeUsage(data) {
  const el = document.getElementById("usageClaudeBody");
  if (!el) return;

  if (data.error) {
    if (data.error === "rate_limited" && data._show_relogin) {
      // Auto token refresh succeeded but still rate-limited → show relogin button
      const msg = data._relogin_message || "Token refreshed but still rate-limited — try re-login";
      el.innerHTML = _renderUsageError("claude", data, msg);
      const btn = el.querySelector("[data-provider='claude']");
      btn?.addEventListener("click", () => _runUsageRelogin("claude"));
      return;
    }
    const msg = data.error === "no_credentials"
      ? t("home.usage_no_credentials")
      : data.message || data.error;
    el.innerHTML = _renderUsageError("claude", data, msg);
    const btn = el.querySelector("[data-provider='claude']");
    btn?.addEventListener("click", () => _runUsageRelogin("claude"));
    return;
  }

  let html = "";
  if (data.five_hour) {
    html += _renderUsageBar("5h", data.five_hour.utilization, data.five_hour.resets_at, data.five_hour.window_seconds);
  }
  if (data.seven_day) {
    html += _renderUsageBar("7d", data.seven_day.utilization, data.seven_day.resets_at, data.seven_day.window_seconds);
  }
  if (data.additional_capacity) {
    const ac = data.additional_capacity;
    const usedM = (ac.used_tokens / 1_000_000).toFixed(2);
    const limitM = (ac.limit_tokens / 1_000_000).toFixed(2);
    html += _renderUsageBar(`Add (${usedM}M/${limitM}M)`, ac.utilization, null, null);
  }
  el.innerHTML = html || `<div class="usage-ok">${t("home.usage_within_limit")}</div>`;
}

function _renderOpenaiUsage(data) {
  const el = document.getElementById("usageOpenaiBody");
  if (!el) return;

  if (data.error) {
    const msg = data.error === "no_credentials"
      ? t("home.usage_no_credentials")
      : data.error === "not_available"
        ? t("home.usage_not_available")
        : data.message || data.error;
    el.innerHTML = _renderUsageError("openai", data, msg);
    const btn = el.querySelector("[data-provider='openai']");
    btn?.addEventListener("click", () => _runUsageRelogin("openai"));
    return;
  }

  // Render usage windows (keys like "5h", "Week", etc.)
  const skip = new Set(["provider"]);
  let html = "";
  for (const [key, win] of Object.entries(data)) {
    if (skip.has(key) || !win || typeof win !== "object" || win.utilization === undefined) continue;
    html += _renderUsageBar(key, win.utilization, win.resets_at, win.window_seconds);
  }
  el.innerHTML = html || `<div class="usage-ok">${t("home.usage_within_limit")}</div>`;
}

function _renderGovernor(gov) {
  const el = document.getElementById("usageGovernorBar");
  if (!el) return;
  if (!gov || !gov.active) {
    el.style.display = "none";
    return;
  }
  const suspended = (gov.suspended_animas || []).join(", ") || "none";
  const reason = gov.reason || "throttling";
  const needsRelogin = /rate_limited|unauthorized|no_credentials/.test(reason);
  el.style.display = "block";
  el.innerHTML = `
    <div class="governor-bar governor-bar--active">
      <span class="governor-icon">&#x26A0;</span>
      <span class="governor-text">
        <strong>Usage Governor</strong>: ${escapeHtml(reason)}
      </span>
      <span class="governor-suspended">${escapeHtml(suspended)}</span>
      ${needsRelogin ? `<button class="btn-secondary governor-relogin-btn" id="govReloginBtn" style="margin-left:0.75rem;font-size:0.78rem;padding:3px 10px;">Claude 再認証</button>` : ""}
    </div>
  `;
  if (needsRelogin) {
    const btn = document.getElementById("govReloginBtn");
    btn?.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "...";
      await _runUsageRelogin("claude");
      btn.disabled = false;
      btn.textContent = "Claude 再認証";
    });
  }
}

async function _loadUsage(forceRefresh = false) {
  try {
    const url = forceRefresh ? "/api/usage?skip_cache=true" : "/api/usage";
    const data = await api(url);

    // Auto-refresh: if Claude returns rate_limited, try relogin (token refresh)
    // then retry once before showing the error
    if (data.claude?.error === "rate_limited") {
      try {
        const reloginRes = await fetch("/api/usage/claude/relogin", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
        });
        const reloginData = await reloginRes.json().catch(() => ({}));
        if (reloginData.success) {
          // Token refreshed — retry usage fetch (skip_cache)
          const retry = await api("/api/usage?skip_cache=true");
          if (retry.claude && !retry.claude.error) {
            data.claude = retry.claude;
          } else if (retry.claude?.error === "rate_limited") {
            // Still rate-limited after refresh — show relogin button
            data.claude = {
              ...retry.claude,
              _show_relogin: true,
              _relogin_message: reloginData.message,
            };
          }
        } else {
          // Refresh failed — show relogin button
          data.claude._show_relogin = true;
        }
      } catch { /* ignore — will render original error */ }
    }

    if (data.claude) _renderClaudeUsage(data.claude);
    if (data.openai) _renderOpenaiUsage(data.openai);
    _renderGovernor(data.governor);
  } catch (err) {
    const claudeEl = document.getElementById("usageClaudeBody");
    const openaiEl = document.getElementById("usageOpenaiBody");
    const msg = `<div class="usage-error">${escapeHtml(err.message)}</div>`;
    if (claudeEl) claudeEl.innerHTML = msg;
    if (openaiEl) openaiEl.innerHTML = msg;
  }
}

// ── System Status ──────────────────────────

async function _loadSystemStatus() {
  try {
    const data = await api("/api/system/status");
    const animaCountEl = document.getElementById("homeAnimaCount");
    const schedulerEl = document.getElementById("homeSchedulerStatus");
    const processCountEl = document.getElementById("homeProcessCount");
    if (animaCountEl) animaCountEl.textContent = data.animas ?? 0;
    if (schedulerEl) schedulerEl.textContent = data.scheduler_running ? t("home.scheduler_running") : t("home.scheduler_stopped");
    const processes = data.processes || {};
    const runningCount = Object.values(processes).filter(p => p.status === "running").length;
    if (processCountEl) processCountEl.textContent = runningCount;
  } catch {
    const el = document.getElementById("homeAnimaCount");
    if (el) el.textContent = t("home.status_fetch_failed");
  }
}

async function _loadOrgChart() {
  const container = document.getElementById("homeOrgTree");
  if (!container) return;

  try {
    const data = await api("/api/org/chart?include_disabled=true");
    const tree = data.tree || [];
    if (tree.length === 0) {
      container.innerHTML = `<div class="loading-placeholder">${t("animas.not_registered")}</div>`;
      return;
    }

    const topRow = document.createElement("div");
    topRow.className = "org-tree-top-row";
    for (const node of tree) {
      topRow.appendChild(_renderColumn(node));
    }
    container.innerHTML = "";
    container.appendChild(topRow);

    _loadOrgAvatars(container);
  } catch (err) {
    container.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _shortModel(model) {
  if (!model) return "";
  return model
    .replace(/^(openai|google|vertex_ai|azure|ollama|bedrock)\//, "")
    .replace(/^jp\.anthropic\./, "")
    .replace(/^anthropic\./, "");
}

function _renderColumn(node) {
  const col = document.createElement("div");
  col.className = "org-tree-column";

  col.appendChild(_buildCard(node, true));

  const children = node.children || [];
  if (children.length > 0) {
    const stem = document.createElement("div");
    stem.className = "org-tree-stem";
    col.appendChild(stem);

    const ul = document.createElement("ul");
    ul.className = "org-tree-list";
    for (const child of children) {
      ul.appendChild(_renderSubNode(child));
    }
    col.appendChild(ul);
  }

  return col;
}

function _renderSubNode(node) {
  const li = document.createElement("li");
  li.className = "org-tree-node";

  li.appendChild(_buildCard(node, false));

  const children = node.children || [];
  if (children.length > 0) {
    const childUl = document.createElement("ul");
    childUl.className = "org-tree-list";
    for (const child of children) {
      childUl.appendChild(_renderSubNode(child));
    }
    li.appendChild(childUl);
  }

  return li;
}

function _buildCard(node, isRoot = false) {
  const initial = escapeHtml(node.name.charAt(0).toUpperCase());
  const color = animaHashColor(node.name);
  const dotClass = statusClass(node.status);
  const role = node.speciality || "";
  const model = _shortModel(node.model);
  const disabled = node.status === "disabled";

  const card = document.createElement("div");
  let cls = "org-tree-card";
  if (isRoot) cls += " org-tree-card--root";
  if (disabled) cls += " org-tree-card--disabled";
  card.className = cls;
  const department = node.department || "";
  const title = node.title || "";
  const metaParts = [];
  if (department) metaParts.push(`<span class="org-tree-dept">${escapeHtml(department)}</span>`);
  if (title) metaParts.push(`<span class="org-tree-title">${escapeHtml(title)}</span>`);
  if (role) metaParts.push(`<span class="org-tree-role">${escapeHtml(role)}</span>`);
  if (model) metaParts.push(`<span class="org-tree-model">${escapeHtml(model)}</span>`);

  card.innerHTML = `
    <div class="org-tree-avatar" id="orgAvatar_${escapeHtml(node.name)}" style="background:${color};">
      ${initial}
    </div>
    <div class="org-tree-info">
      <div class="org-tree-name">
        ${escapeHtml(node.name)}
        <span class="org-tree-status ${dotClass}"></span>
      </div>
      ${metaParts.length ? `<div class="org-tree-meta">${metaParts.join("")}</div>` : ""}
    </div>
  `;
  card.addEventListener("click", () => { location.hash = `#/animas/${encodeURIComponent(node.name)}`; });
  return card;
}

async function _loadOrgAvatars(root) {
  const avatarEls = root.querySelectorAll("[id^='orgAvatar_']");
  for (const el of avatarEls) {
    const name = el.id.replace("orgAvatar_", "");
    const url = await resolveAvatar(name, bustupCandidates());
    if (url) {
      el.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
    }
  }
}

async function _loadActivity() {
  const timeline = document.getElementById("homeActivityTimeline");
  if (!timeline) return;

  try {
    const data = await api("/api/activity/recent?hours=12&limit=10");
    const events = data.events || [];
    if (events.length === 0) {
      timeline.innerHTML = `<div class="loading-placeholder">${t("activity.recent_none")}</div>`;
      return;
    }

    const eventsHtml = events.map(evt => {
      const icon = getIcon(evt.type);
      const ts = timeStr(evt.ts);
      const anima = evt.anima || "";
      const summary = getDisplaySummary(evt);
      return `
        <div style="display:flex; align-items:flex-start; gap:0.5rem; padding:0.4rem 0; border-bottom:1px solid var(--border-color, #eee);">
          <span style="flex-shrink:0;">${icon}</span>
          <span style="color:var(--text-secondary, #666); flex-shrink:0; min-width:3rem;">${escapeHtml(ts)}</span>
          <span style="font-weight:500; flex-shrink:0; max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(anima)}</span>
          <span style="color:var(--text-secondary, #666); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(summary)}</span>
        </div>
      `;
    }).join("");

    timeline.innerHTML = `
      <div id="homeActivityEvents">${eventsHtml}</div>
      <div style="text-align:right;margin-top:0.5rem;"><a href="#/activity" style="color:var(--accent-color,#2563eb);text-decoration:none;font-size:0.85rem;">${t("activity.view_more")}</a></div>
    `;
  } catch (err) {
    timeline.innerHTML = `<div class="loading-placeholder">${t("activity.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

// ── External Tasks Widget ──────────────────

const _SOURCE_ICONS = {
  github: "\u{1F4BB}",
  slack: "\u{1F4AC}",
  gmail: "\u{2709}\uFE0F",
  jira: "\u{1F4CB}",
  notion: "\u{1F4D3}",
  other: "\u{1F517}",
};

const _STATUS_COLORS = {
  open: "#2563eb",
  in_progress: "#d97706",
  done: "#16a34a",
  cancelled: "#6b7280",
};

let _extTasksExpanded = true;
let _extTasksLimit = 2;
let _extTasksRetryCount = 0;

function _initExternalTasksWidget() {
  const header = document.getElementById("extTasksHeader");
  const refreshBtn = document.getElementById("extTasksRefresh");
  if (header) {
    header.addEventListener("click", (e) => {
      if (e.target === refreshBtn || refreshBtn?.contains(e.target)) return;
      _extTasksExpanded = !_extTasksExpanded;
      _updateExtTasksToggle();
      if (_extTasksExpanded) _loadExternalTasks();
    });
  }
  if (refreshBtn) {
    refreshBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      _extTasksRetryCount = 0;
      _loadExternalTasks(true);
    });
  }
}

function _updateExtTasksToggle() {
  const toggle = document.getElementById("extTasksToggle");
  const body = document.getElementById("extTasksBody");
  if (toggle) toggle.innerHTML = _extTasksExpanded ? "&#x25BC;" : "&#x25B6;";
  if (body) body.style.display = _extTasksExpanded ? "" : "none";
}

function _relativeTime(isoStr) {
  if (!isoStr) return "";
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return t("home.ext_just_now");
  if (mins < 60) return `${mins}${t("home.ext_min_ago")}`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}${t("home.ext_hour_ago")}`;
  const days = Math.floor(hrs / 24);
  return `${days}${t("home.ext_day_ago")}`;
}

function _renderTaskItem(task) {
  const icon = _SOURCE_ICONS[task.source_type] || _SOURCE_ICONS.other;
  const title = escapeHtml((task.title || "").slice(0, 40));
  const statusColor = _STATUS_COLORS[task.status] || _STATUS_COLORS.open;
  const statusLabel = task.status === "in_progress" ? "in progress" : task.status;
  const relTime = _relativeTime(task.last_updated_at);
  const clickAttr = task.source_url
    ? `onclick="window.open('${escapeHtml(task.source_url)}','_blank')"`
    : "";
  const cursorStyle = task.source_url ? "cursor:pointer;" : "";

  return `
    <div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0;border-bottom:1px solid var(--border-color,#eee);${cursorStyle}" ${clickAttr} role="link" tabindex="0" aria-label="${escapeHtml(task.title)}">
      <span style="flex-shrink:0;font-size:1.1rem;" aria-label="${escapeHtml(task.source_type)}">${icon}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${title}</span>
      <span style="flex-shrink:0;font-size:0.7rem;padding:0.15rem 0.4rem;border-radius:4px;background:${statusColor};color:#fff;">${escapeHtml(statusLabel)}</span>
      <span style="flex-shrink:0;font-size:0.75rem;color:var(--text-secondary,#666);min-width:3.5rem;text-align:right;">${escapeHtml(relTime)}</span>
    </div>
  `;
}

async function _loadExternalTasks(forceRefresh = false) {
  const listEl = document.getElementById("extTasksList");
  const badgeEl = document.getElementById("extTasksBadge");
  const lastUpdEl = document.getElementById("extTasksLastUpdated");
  if (!listEl) return;

  const limit = _extTasksExpanded ? _extTasksLimit : 2;
  let url = `/api/external-tasks?limit=${limit}&status=open,in_progress&sort=priority&order=desc`;
  if (forceRefresh) url += "&_t=" + Date.now();

  try {
    const data = await api(url);
    const tasks = data.data || [];
    const totalCount = data.meta?.total_count ?? 0;

    if (badgeEl) {
      if (totalCount > 0) {
        badgeEl.textContent = totalCount;
        badgeEl.style.display = "inline";
      } else {
        badgeEl.style.display = "none";
      }
    }

    if (lastUpdEl) {
      lastUpdEl.textContent = `${t("home.ext_last_updated")}: ${timeStr(new Date().toISOString())}`;
    }

    if (tasks.length === 0) {
      listEl.innerHTML = `
        <div style="text-align:center;padding:1.5rem 0;color:var(--text-secondary,#666);">
          <div style="font-size:1.5rem;margin-bottom:0.5rem;">&#x2714;</div>
          <div>${t("home.ext_empty")}</div>
          <div style="font-size:0.8rem;margin-top:0.25rem;">${t("home.ext_empty_hint")}</div>
        </div>
      `;
      _extTasksRetryCount = 0;
      return;
    }

    let html = tasks.map(_renderTaskItem).join("");
    if (data.meta?.has_more) {
      html += `<div style="text-align:right;margin-top:0.5rem;">
        <a href="#" id="extTasksShowAll" style="color:var(--accent-color,#2563eb);text-decoration:none;font-size:0.85rem;">
          ${t("home.ext_show_all")}(${totalCount}${t("home.ext_count_suffix")})
        </a>
      </div>`;
    }

    listEl.innerHTML = html;

    const showAllLink = document.getElementById("extTasksShowAll");
    if (showAllLink) {
      showAllLink.addEventListener("click", (e) => {
        e.preventDefault();
        _extTasksLimit = 10;
        _loadExternalTasks();
      });
    }
    _extTasksRetryCount = 0;
  } catch (err) {
    _extTasksRetryCount++;
    const errMsg = _extTasksRetryCount >= 3
      ? t("home.ext_error_persistent")
      : t("home.ext_error");
    listEl.innerHTML = `
      <div style="text-align:center;padding:1.5rem 0;color:var(--text-secondary,#666);">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">&#x26A0;</div>
        <div>${errMsg}</div>
        <button id="extTasksRetryBtn" class="btn-secondary" style="margin-top:0.5rem;font-size:0.85rem;">${t("home.ext_retry")}</button>
      </div>
    `;
    const retryBtn = document.getElementById("extTasksRetryBtn");
    if (retryBtn) {
      retryBtn.addEventListener("click", () => _loadExternalTasks(true));
    }
  }
}
