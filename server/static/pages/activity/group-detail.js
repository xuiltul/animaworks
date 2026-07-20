// ── Group detail panel for activity swimlane ────────────────
// Extracted from the former vertical list so session-replay can reuse it.

import { escapeHtml, smartTimestamp } from "../../modules/state.js";
import { getIcon, getDisplaySummary } from "../../shared/activity-types.js";
import {
  createContextBadge,
  decorateContextElement,
  getParallelTaskCounts,
} from "../../shared/activity-context.js";
import { t } from "/shared/i18n.js";
import { isGroupInProgress } from "./swimlane-layout.js";

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

function _headerLabel(grp) {
  const timeRange = _formatTimeRange(grp.start_ts, grp.end_ts);
  if (grp.type === "single" && grp.events?.[0]) {
    const evt = grp.events[0];
    return getDisplaySummary(evt) || timeRange;
  }
  if (grp.type === "heartbeat") return `Heartbeat ${timeRange}`;
  if (grp.type === "chat") return `${t("activity.label_user_chat")} ${timeRange}`;
  if (grp.type === "dm") return `DM ${grp.summary || ""} ${timeRange}`;
  if (grp.type === "cron") return `Cron ${grp.summary || ""} ${timeRange}`;
  if (grp.type === "task") return `Task ${grp.summary || ""} ${timeRange}`;
  if (grp.type === "inbox") return `${t("activity.label_inbox")} ${timeRange}`;
  if (grp.type === "task_exec") {
    return `${t("activity.label_task_exec")} ${grp.summary || ""} ${timeRange}`;
  }
  return `${grp.type || ""} ${timeRange}`;
}

/**
 * Render a group detail panel into container.
 * @param {HTMLElement} container
 * @param {object|null} group  null clears the panel
 * @param {{ nowMs?: number }} [opts]
 */
export function renderGroupDetail(container, group, opts = {}) {
  if (!container) return;
  container.innerHTML = "";
  if (!group) {
    container.hidden = true;
    container.classList.remove("is-open");
    return;
  }

  container.hidden = false;
  container.classList.add("is-open");

  const parallelCounts = getParallelTaskCounts(group.events || []);
  const parallelCount = _groupParallelCount(group, parallelCounts);
  const nowMs = opts.nowMs ?? Date.now();
  const ongoing = isGroupInProgress(group, nowMs);

  const header = document.createElement("div");
  header.className = "activity-group-header expanded swimlane-detail-header";

  if (group.type === "single" && group.events?.[0]) {
    const evt = group.events[0];
    const icon = getIcon(evt.type);
    const time = smartTimestamp(evt.ts);
    const anima = evt.anima || group.anima || "";
    const summary = getDisplaySummary(evt);
    header.innerHTML =
      `<span class="activity-row-time">${escapeHtml(time)}</span>` +
      `<span class="activity-row-icon">${icon}</span>` +
      `<span class="activity-row-anima">${escapeHtml(anima)}</span>` +
      `<span class="activity-row-summary">${escapeHtml(summary)}</span>`;
    decorateContextElement(header, evt.ctx);
    const contextBadge = createContextBadge(evt.ctx);
    if (contextBadge) header.querySelector(".activity-row-summary")?.before(contextBadge);
  } else {
    const icon = GROUP_ICONS[group.type] || "⚙️";
    const anima = group.anima || "";
    const count = group.event_count || group.events?.length || 0;
    const openBadge = ongoing
      ? `<span class="activity-group-badge-open">${t("activity.badge_ongoing")}</span>`
      : "";
    header.innerHTML =
      `<span class="activity-row-icon">${icon}</span>` +
      `<span class="activity-row-anima">${escapeHtml(anima)}</span>` +
      `<span class="activity-group-label">${escapeHtml(_headerLabel(group))}</span>` +
      openBadge +
      `<span class="activity-group-count">${t("activity.count_items", { count })}</span>`;

    const contexts = new Set((group.events || []).map((evt) => evt.ctx).filter(Boolean));
    if (contexts.size === 1) {
      const ctx = contexts.values().next().value;
      decorateContextElement(header, ctx);
      const contextBadge = createContextBadge(ctx);
      if (contextBadge) header.querySelector(".activity-group-label")?.after(contextBadge);
    }
  }

  const parallelBadge = _parallelBadge(parallelCount);
  if (parallelBadge) {
    const anchor = header.querySelector(".activity-group-label, .activity-row-summary");
    anchor?.after(parallelBadge);
  }

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "swimlane-detail-close";
  closeBtn.setAttribute("aria-label", t("activity.swimlane_detail_close"));
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", () => {
    renderGroupDetail(container, null);
    container.dispatchEvent(new CustomEvent("swimlane-detail-close"));
  });
  header.appendChild(closeBtn);

  container.appendChild(header);
  container.appendChild(_createGroupEvents(group, parallelCounts));

  if (typeof window !== "undefined" && window.lucide) {
    window.lucide.createIcons({ nodes: [container] });
  }
}
