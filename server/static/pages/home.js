// ── Home Dashboard ──────────────────────────
import { t } from "/shared/i18n.js";
import { api } from "../modules/api.js";
import { escapeHtml, timeStr, statusClass } from "../modules/state.js";
import { animaHashColor } from "../modules/animas.js";
import { getIcon, getDisplaySummary } from "../shared/activity-types.js";
import { bustupCandidates, resolveAvatar } from "../modules/avatar-resolver.js";

let _refreshInterval = null;

// ── Render ─────────────────────────────────

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2>${t("home.dashboard")}</h2>
    </div>

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
          <div class="loading-placeholder">読み込み中...</div>
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
  _refreshInterval = setInterval(_loadAll, 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Data Loading ───────────────────────────

async function _loadAll() {
  _loadSystemStatus();
  _loadOrgChart();
  _loadActivity();
}

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
  const metaParts = [];
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
  card.addEventListener("click", () => { location.hash = "#/animas"; });
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
