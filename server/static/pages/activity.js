// ── Activity Timeline Page ──────────────────
import { api } from "../modules/api.js";
import { escapeHtml, smartTimestamp, renderMarkdown } from "../modules/state.js";
import { getIcon, getDisplaySummary, GROUP_TYPE_CATEGORIES } from "../shared/activity-types.js";
import {
  createContextBadge,
  decorateContextElement,
  getParallelTaskCounts,
  renderRunningTasksStrip,
} from "../shared/activity-context.js";
import { t } from "/shared/i18n.js";

let _refreshInterval = null;
let _groups = [];
let _totalGroups = 0;
let _totalEvents = 0;
let _groupOffset = 0;
let _hasMore = false;
/** @type {Set<string>} */
let _expandedGroups = new Set();
let _selectedAnima = "";
/** @type {string[]} selected group types (trigger-based) */
let _selectedGroupTypes = [];

const GROUP_LIMIT = 50;

const GROUP_ICONS = {
  heartbeat: "💓",
  chat: "💬",
  dm: "✉️",
  cron: "⏰",
  task: "📋",
  inbox: "📬",
  task_exec: "🔨",
  single: "⚙️",
};

// ── Render ─────────────────────────────────

export function render(container) {
  _groups = [];
  _totalGroups = 0;
  _totalEvents = 0;
  _groupOffset = 0;
  _hasMore = false;
  _expandedGroups = new Set();
  _selectedAnima = "";
  _selectedGroupTypes = [];

  container.innerHTML = `
    <div class="activity-page">
      <div class="activity-header">
        <h2>${t("activity.page_title")}</h2>
        <span class="activity-count" id="activityCount"></span>
      </div>

      <div class="running-tasks-strip" id="activityRunningTasks" hidden></div>

      <div class="activity-filters">
        <select class="activity-anima-select" id="activityAnimaSelect">
          <option value="">${t("activity.all_animas")}</option>
        </select>
        <div id="activityTypeChips" style="display:flex; gap:0.35rem; flex-wrap:wrap;"></div>
      </div>

      <div class="activity-list" id="activityList">
        <div class="loading-placeholder">${t("common.loading")}</div>
      </div>

      <div id="activityLoadMoreWrap"></div>
    </div>
  `;

  _buildAnimaSelect();
  _buildTypeChips();
  _loadEvents(true);
  _loadRunningTasks();
  _refreshInterval = setInterval(() => {
    _loadEvents(true);
    _loadRunningTasks();
  }, 30000);
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// ── Filter UI ──────────────────────────────

async function _buildAnimaSelect() {
  const sel = document.getElementById("activityAnimaSelect");
  if (!sel) return;

  try {
    const animas = await api("/api/animas");
    for (const a of animas) {
      const opt = document.createElement("option");
      opt.value = a.name;
      opt.textContent = a.name;
      sel.appendChild(opt);
    }
  } catch (err) {
    console.error("Failed to load animas for activity filter:", err);
  }

  sel.addEventListener("change", () => {
    _selectedAnima = sel.value;
    _groupOffset = 0;
    _groups = [];
    _expandedGroups.clear();
    _loadEvents(true);
    _loadRunningTasks();
  });
}

function _buildTypeChips() {
  const wrap = document.getElementById("activityTypeChips");
  if (!wrap) return;

  wrap.innerHTML = "";
  for (let i = 0; i < GROUP_TYPE_CATEGORIES.length; i++) {
    const chip = GROUP_TYPE_CATEGORIES[i];
    const btn = document.createElement("button");
    btn.className = "activity-type-chip" + (i === 0 ? " active" : "");
    btn.textContent = chip.i18nKey ? t(chip.i18nKey) : chip.label;
    btn.dataset.index = String(i);

    btn.addEventListener("click", () => {
      const allChip = wrap.querySelector('[data-index="0"]');

      if (i === 0) {
        for (const b of wrap.querySelectorAll(".activity-type-chip")) {
          b.classList.remove("active");
        }
        btn.classList.add("active");
        _selectedGroupTypes = [];
      } else {
        allChip?.classList.remove("active");
        btn.classList.toggle("active");

        _selectedGroupTypes = [];
        for (const b of wrap.querySelectorAll(".activity-type-chip.active")) {
          const idx = parseInt(b.dataset.index, 10);
          if (idx > 0 && GROUP_TYPE_CATEGORIES[idx]?.groupTypes) {
            _selectedGroupTypes.push(...GROUP_TYPE_CATEGORIES[idx].groupTypes);
          }
        }

        if (_selectedGroupTypes.length === 0) {
          allChip?.classList.add("active");
        }
      }

      _groupOffset = 0;
      _groups = [];
      _expandedGroups.clear();
      _loadEvents(true);
    });

    wrap.appendChild(btn);
  }
}

// ── Data Loading ───────────────────────────

async function _loadEvents(reset) {
  if (reset) {
    _groupOffset = 0;
    _groups = [];
    _expandedGroups.clear();
  }

  let url = `/api/activity/recent?hours=48&grouped=true&group_limit=${GROUP_LIMIT}&group_offset=${_groupOffset}`;
  if (_selectedAnima) {
    url += `&anima=${encodeURIComponent(_selectedAnima)}`;
  }
  if (_selectedGroupTypes.length > 0) {
    url += `&group_type=${encodeURIComponent(_selectedGroupTypes.join(","))}`;
  }

  const list = document.getElementById("activityList");

  try {
    const data = await api(url);
    const newGroups = data.groups || [];
    _totalGroups = data.total_groups ?? 0;
    _totalEvents = data.total_events ?? 0;
    _hasMore = data.has_more ?? false;

    if (reset) {
      _groups = newGroups;
    } else {
      _groups = _groups.concat(newGroups);
    }

    _groupOffset = _groups.length;
    _renderList();
    _updateCount();
    _renderLoadMore();
  } catch (err) {
    if (list) {
      list.innerHTML = `<div class="activity-empty">${t("activity.load_failed")}: ${escapeHtml(err.message)}</div>`;
    }
  }
}

async function _loadRunningTasks() {
  const strip = document.getElementById("activityRunningTasks");
  if (!strip) return;
  let url = "/api/activity/running-tasks";
  if (_selectedAnima) url += `?anima=${encodeURIComponent(_selectedAnima)}`;
  try {
    const data = await api(url);
    renderRunningTasksStrip(strip, data, t);
  } catch (err) {
    console.error("Failed to load running activity tasks:", err);
  }
}

// ── Rendering ──────────────────────────────

function _updateCount() {
  const el = document.getElementById("activityCount");
  if (el) {
    el.textContent = t("activity.count_display", { groups: _groups.length, events: _totalEvents });
  }
}

function _renderList() {
  const list = document.getElementById("activityList");
  if (!list) return;

  if (_groups.length === 0) {
    list.innerHTML = `<div class="activity-empty">${t("activity.empty")}</div>`;
    return;
  }

  list.innerHTML = "";
  const allEvents = _groups.flatMap((grp) => grp.events || []);
  const parallelCounts = getParallelTaskCounts(allEvents);
  for (const grp of _groups) {
    const header = _createGroupHeader(grp, parallelCounts);
    list.appendChild(header);

    if (_expandedGroups.has(grp.id)) {
      const eventsContainer = _createGroupEvents(grp, parallelCounts);
      list.appendChild(eventsContainer);
    }
  }
}

function _formatTimeRange(startTs, endTs) {
  const start = startTs ? startTs.slice(11, 16) : "";
  const end = endTs ? endTs.slice(11, 16) : "";
  if (!start) return "";
  return start === end ? `[${start}]` : `[${start}-${end}]`;
}

function _parallelBadge(count) {
  if (count < 2) return null;
  const badge = document.createElement("span");
  badge.className = "activity-parallel-badge";
  badge.textContent = t("activity.parallel_count", { count });
  return badge;
}

function _groupParallelCount(grp, parallelCounts) {
  let count = 0;
  for (const evt of grp.events || []) count = Math.max(count, parallelCounts.get(evt) || 0);
  return count;
}

function _createGroupHeader(grp, parallelCounts) {
  const header = document.createElement("div");
  const isExpanded = _expandedGroups.has(grp.id);
  header.className = "activity-group-header" + (isExpanded ? " expanded" : "");
  const parallelCount = _groupParallelCount(grp, parallelCounts);

  if (grp.type === "single") {
    const evt = grp.events[0];
    const icon = getIcon(evt.type);
    const time = smartTimestamp(evt.ts);
    const anima = evt.anima || grp.anima || "";
    const summary = getDisplaySummary(evt);

    header.innerHTML =
      `<span class="activity-row-time">${escapeHtml(time)}</span>` +
      `<span class="activity-row-icon">${icon}</span>` +
      `<span class="activity-row-anima">${escapeHtml(anima)}</span>` +
      `<span class="activity-row-summary">${escapeHtml(summary)}</span>`;
    decorateContextElement(header, evt.ctx);
    const contextBadge = createContextBadge(evt.ctx);
    if (contextBadge) header.querySelector(".activity-row-summary")?.before(contextBadge);
    const parallelBadge = _parallelBadge(parallelCount);
    if (parallelBadge) header.querySelector(".activity-row-summary")?.before(parallelBadge);

    header.addEventListener("click", () => {
      if (_expandedGroups.has(grp.id)) {
        _expandedGroups.delete(grp.id);
      } else {
        _expandedGroups.add(grp.id);
      }
      _renderList();
    });

    return header;
  }

  const icon = GROUP_ICONS[grp.type] || "⚙️";
  const timeRange = _formatTimeRange(grp.start_ts, grp.end_ts);
  const anima = grp.anima || "";
  const count = grp.event_count || grp.events.length;
  const chevron = isExpanded ? "▼" : "▶";

  let label = "";
  if (grp.type === "heartbeat") label = `Heartbeat ${timeRange}`;
  else if (grp.type === "chat") label = `${t("activity.label_user_chat")} ${timeRange}`;
  else if (grp.type === "dm") label = `DM ${grp.summary || ""} ${timeRange}`;
  else if (grp.type === "cron") label = `Cron ${grp.summary || ""} ${timeRange}`;
  else if (grp.type === "task") label = `Task ${grp.summary || ""} ${timeRange}`;
  else if (grp.type === "inbox") label = `${t("activity.label_inbox")} ${timeRange}`;
  else if (grp.type === "task_exec") label = `${t("activity.label_task_exec")} ${grp.summary || ""} ${timeRange}`;
  else label = `${grp.type} ${timeRange}`;

  const openBadge = grp.is_open ? `<span class="activity-group-badge-open">${t("activity.badge_ongoing")}</span>` : "";

  header.innerHTML =
    `<span class="activity-group-chevron">${chevron}</span>` +
    `<span class="activity-row-icon">${icon}</span>` +
    `<span class="activity-row-anima">${escapeHtml(anima)}</span>` +
    `<span class="activity-group-label">${escapeHtml(label)}</span>` +
    openBadge +
    `<span class="activity-group-count">${t("activity.count_items", { count })}</span>`;

  const contexts = new Set((grp.events || []).map((evt) => evt.ctx).filter(Boolean));
  if (contexts.size === 1) {
    const ctx = contexts.values().next().value;
    decorateContextElement(header, ctx);
    const contextBadge = createContextBadge(ctx);
    if (contextBadge) header.querySelector(".activity-group-label")?.after(contextBadge);
  }
  const parallelBadge = _parallelBadge(parallelCount);
  if (parallelBadge) header.querySelector(".activity-group-label")?.after(parallelBadge);

  header.addEventListener("click", () => {
    if (_expandedGroups.has(grp.id)) {
      _expandedGroups.delete(grp.id);
    } else {
      _expandedGroups.add(grp.id);
    }
    _renderList();
  });

  return header;
}

function _createGroupEvents(grp, parallelCounts) {
  const container = document.createElement("div");
  container.className = "activity-group-events";

  const events = grp.events || [];
  const maxInitial = 30;
  const toShow = events.slice(0, maxInitial);

  for (let i = 0; i < toShow.length; i++) {
    const evt = toShow[i];
    const isLast = i === toShow.length - 1 && events.length <= maxInitial;
    const row = _createEventRow(evt, isLast, parallelCounts.get(evt) || 0);
    container.appendChild(row);
  }

  if (events.length > maxInitial) {
    const moreBtn = document.createElement("div");
    moreBtn.className = "activity-group-show-more";
    moreBtn.textContent = t("activity.show_more", { count: events.length - maxInitial });
    moreBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      moreBtn.remove();
      const remaining = events.slice(maxInitial);
      for (let i = 0; i < remaining.length; i++) {
        const evt = remaining[i];
        const row = _createEventRow(evt, i === remaining.length - 1, parallelCounts.get(evt) || 0);
        container.appendChild(row);
      }
    });
    container.appendChild(moreBtn);
  }

  return container;
}

function _createEventRow(evt, isLast, parallelCount) {
  const row = document.createElement("div");
  row.className = "activity-group-event";

  const connector = isLast ? "└" : "├";
  const icon = getIcon(evt.type);
  const time = evt.ts ? evt.ts.slice(11, 16) : "";

  let summary = "";
  if (evt.type === "tool_use") {
    const toolName = evt.tool || "";
    const result = evt.tool_result;
    if (result) {
      const resultText = result.content || "";
      const errClass = result.is_error ? " tool-error" : "";
      summary = `${toolName} → <span class="activity-tool-result${errClass}">${escapeHtml(resultText)}</span>`;
    } else {
      summary = toolName;
    }
  } else {
    summary = escapeHtml(getDisplaySummary(evt));
  }

  row.innerHTML =
    `<span class="activity-tree-connector">${connector}</span>` +
    `<span class="activity-row-time">${escapeHtml(time)}</span>` +
    `<span class="activity-row-icon">${icon}</span>` +
    `<span class="activity-row-summary">${summary}</span>`;

  decorateContextElement(row, evt.ctx);
  const contextBadge = createContextBadge(evt.ctx);
  if (contextBadge) row.querySelector(".activity-row-summary")?.before(contextBadge);
  const parallelBadge = _parallelBadge(parallelCount);
  if (parallelBadge) row.querySelector(".activity-row-summary")?.before(parallelBadge);

  return row;
}

function _renderLoadMore() {
  const wrap = document.getElementById("activityLoadMoreWrap");
  if (!wrap) return;

  if (_hasMore) {
    wrap.innerHTML = `<button class="activity-load-more" id="activityLoadMoreBtn">${t("activity.load_more_btn", { current: _groups.length, total: _totalGroups })}</button>`;
    const btn = document.getElementById("activityLoadMoreBtn");
    if (btn) {
      btn.addEventListener("click", () => {
        btn.disabled = true;
        btn.textContent = t("common.loading");
        _loadEvents(false);
      });
    }
  } else {
    wrap.innerHTML = "";
  }
}
