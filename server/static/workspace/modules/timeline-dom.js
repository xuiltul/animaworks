/**
 * timeline-dom.js — DOM construction + rendering for the activity timeline.
 *
 * DOM helpers receive timeline state through arguments or callbacks from the
 * orchestrator (timeline.js). Only resolved avatar promises are cached here.
 */

import { getIcon, getDisplaySummary } from "../../shared/activity-types.js";
import {
  buildParallelGroups,
  createContextBadge,
  decorateContextElement,
} from "../../shared/activity-context.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";
import { renderSimpleMarkdown } from "./utils.js";
import { t } from "/shared/i18n.js";
import { resolveEventPersons } from "./activity-normalize.js";

const _avatarUrlPromises = new Map();

// ── Time formatting ────────────────────────────────

/**
 * Format an ISO timestamp to HH:MM for display.
 * @param {string} isoString
 * @returns {string}
 */
export function formatTime(isoString) {
  if (!isoString) return "--:--";
  try {
    const d = new Date(isoString);
    const h = d.getHours().toString().padStart(2, "0");
    const m = d.getMinutes().toString().padStart(2, "0");
    return `${h}:${m}`;
  } catch {
    return "--:--";
  }
}

/**
 * Extract from/to person names and text from an event,
 * handling both WS format (meta.from_person) and API format (top-level from_person).
 * @param {object} event
 * @returns {{ from: string, to: string, text: string }}
 */
export function resolvePersons(event) {
  // Compatibility wrapper. The shared resolver uses this precedence:
  // meta.from_person → event.from_person → event.from → event.anima,
  // meta.to_person → event.to_person → event.to,
  // meta.text → event.content → event.summary.
  const resolved = resolveEventPersons(event);
  return {
    from: resolved.from,
    to: resolved.to,
    text: resolved.text,
  };
}

// ── DOM construction ───────────────────────────────

/**
 * Build the full timeline DOM and append it to officePanel.
 *
 * @param {HTMLElement} officePanel
 * @param {object} opts
 * @param {Array<{label:string, types:string[]}>} opts.filterDefs
 * @param {(index: number) => void} opts.onFilterClick
 * @param {() => void} opts.onLoadMore
 * @returns {{ container: HTMLElement, listEl: HTMLElement, countEl: HTMLElement, bodyEl: HTMLElement, runningTasksEl: HTMLElement }}
 */
export function buildTimelineDOM(officePanel, opts) {
  const { filterDefs, onFilterClick, onLoadMore } = opts;

  let expanded = false;

  const timeline = document.createElement("div");
  timeline.className = "ws-timeline";
  timeline.id = "wsTimeline";

  // Toggle bar
  const bar = document.createElement("div");
  bar.className = "ws-timeline-bar";
  bar.id = "wsTimelineToggle";

  const title = document.createElement("span");
  title.className = "timeline-title";
  title.textContent = t("ws.timeline_title");

  const count = document.createElement("span");
  count.className = "timeline-count";
  count.id = "wsTimelineCount";
  count.textContent = "0";

  const toggleBtn = document.createElement("button");
  toggleBtn.className = "timeline-toggle-btn";
  toggleBtn.textContent = "\u25B2"; // ▲

  bar.appendChild(title);
  bar.appendChild(count);
  bar.appendChild(toggleBtn);

  // Body (hidden by default)
  const body = document.createElement("div");
  body.className = "ws-timeline-body hidden";
  body.id = "wsTimelineBody";

  // Filters
  const filters = document.createElement("div");
  filters.className = "ws-timeline-filters";

  for (let i = 0; i < filterDefs.length; i++) {
    const fd = filterDefs[i];
    const btn = document.createElement("button");
    btn.className = "tl-filter" + (i === 0 ? " active" : "");
    btn.dataset.index = i;
    btn.textContent = fd.label;
    btn.addEventListener("click", () => {
      for (const b of filters.querySelectorAll(".tl-filter")) {
        b.classList.toggle("active", parseInt(b.dataset.index) === i);
      }
      onFilterClick(i);
    });
    filters.appendChild(btn);
  }

  // Event list
  const list = document.createElement("div");
  list.className = "ws-timeline-list";
  list.id = "wsTimelineList";

  // Load-more button
  const loadMoreBtn = document.createElement("button");
  loadMoreBtn.className = "tl-load-more";
  loadMoreBtn.id = "wsTimelineLoadMore";
  loadMoreBtn.textContent = t("ws.load_more");
  loadMoreBtn.style.cssText = "display:none; width:100%; padding:0.5rem; margin-top:0.5rem; background:var(--bg-secondary, #f3f4f6); border:1px solid var(--border-color, #e5e7eb); border-radius:6px; cursor:pointer; color:var(--text-secondary, #666); font-size:0.8rem;";
  loadMoreBtn.addEventListener("click", () => onLoadMore());

  body.appendChild(filters);
  body.appendChild(list);
  body.appendChild(loadMoreBtn);

  const runningTasks = document.createElement("div");
  runningTasks.className = "running-tasks-strip ws-running-tasks-strip";
  runningTasks.id = "wsRunningTasks";
  runningTasks.hidden = true;

  timeline.appendChild(bar);
  timeline.appendChild(runningTasks);
  timeline.appendChild(body);

  officePanel.appendChild(timeline);

  // Toggle behavior
  bar.addEventListener("click", () => {
    expanded = !expanded;
    body.classList.toggle("hidden", !expanded);
    toggleBtn.textContent = expanded ? "\u25BC" : "\u25B2"; // ▼ or ▲
  });

  return { container: timeline, listEl: list, countEl: count, bodyEl: body, runningTasksEl: runningTasks };
}

// ── Render helpers ─────────────────────────────────

function _matchesFilter(evt, currentFilter) {
  return currentFilter.length === 0 || currentFilter.includes(evt.type);
}

function _resolveTimelineAvatar(anima) {
  const candidates = bustupCandidates();
  const cacheKey = `${anima}\u0000${candidates.join("\u0000")}`;
  if (!_avatarUrlPromises.has(cacheKey)) {
    _avatarUrlPromises.set(
      cacheKey,
      resolveCachedAvatar(anima, candidates, "S").catch(() => null),
    );
  }
  return _avatarUrlPromises.get(cacheKey);
}

function _createAvatarElement(anima, className = "tl-event-avatar") {
  if (!anima) return null;
  const avatar = document.createElement("img");
  avatar.className = className;
  avatar.alt = "";
  avatar.loading = "lazy";
  avatar.decoding = "async";
  avatar.hidden = true;
  avatar.addEventListener("load", () => {
    avatar.hidden = false;
  });
  avatar.addEventListener("error", () => {
    avatar.hidden = true;
    avatar.removeAttribute("src");
  });
  _resolveTimelineAvatar(anima).then((url) => {
    if (url) avatar.src = url;
  });
  return avatar;
}

function _visibleParallelGroup(group, currentFilter) {
  const lanes = group.lanes.map((lane) => ({
    ...lane,
    events: lane.events.filter((evt) => _matchesFilter(evt, currentFilter)),
  }));
  const taskEventCount = lanes.reduce((count, lane) => count + lane.events.length, 0);
  if (taskEventCount === 0) return null;
  return {
    anima: group.anima,
    lanes,
    flatEvents: group.flatEvents.filter((evt) => _matchesFilter(evt, currentFilter)),
  };
}

function _createParallelGroupElement(group, createElementFn) {
  const container = document.createElement("section");
  container.className = "tl-parallel-group";
  container.dataset.anima = group.anima;

  const animaHeader = document.createElement("div");
  animaHeader.className = "tl-parallel-anima-header";
  const avatar = _createAvatarElement(
    group.anima,
    "tl-event-avatar tl-parallel-anima-avatar",
  );
  if (avatar) animaHeader.appendChild(avatar);
  const animaName = document.createElement("span");
  animaName.className = "tl-parallel-anima-name";
  animaName.textContent = group.anima;
  animaHeader.appendChild(animaName);
  container.appendChild(animaHeader);

  const lanes = document.createElement("div");
  lanes.className = "tl-task-lanes";
  for (const lane of group.lanes) {
    const laneEl = document.createElement("div");
    laneEl.className = "tl-task-lane";
    laneEl.dataset.taskId = lane.taskId;

    const laneHeader = document.createElement("div");
    laneHeader.className = "tl-task-lane-header";
    const marker = document.createElement("span");
    marker.className = "tl-task-lane-marker";
    marker.textContent = "⚙";
    const title = document.createElement("span");
    title.className = "tl-task-lane-title";
    title.textContent = lane.title;
    title.title = lane.title;
    laneHeader.appendChild(marker);
    laneHeader.appendChild(title);
    laneEl.appendChild(laneHeader);

    const laneEvents = document.createElement("div");
    laneEvents.className = "tl-task-lane-events";
    for (const evt of lane.events) laneEvents.appendChild(createElementFn(evt));
    laneEl.appendChild(laneEvents);
    lanes.appendChild(laneEl);
  }
  container.appendChild(lanes);

  if (group.flatEvents.length > 0) {
    const flatEvents = document.createElement("div");
    flatEvents.className = "tl-parallel-flat-events";
    for (const evt of group.flatEvents) flatEvents.appendChild(createElementFn(evt));
    container.appendChild(flatEvents);
  }
  return container;
}

/**
 * Re-render the event list using the given events and filter.
 *
 * @param {HTMLElement} listEl
 * @param {Array} events
 * @param {string[]} currentFilter — empty means "all"
 * @param {(evt: object) => HTMLElement} createElementFn
 */
export function renderList(listEl, events, currentFilter, createElementFn) {
  if (!listEl) return;
  listEl.innerHTML = "";

  const filtered = currentFilter.length === 0
    ? events
    : events.filter((e) => currentFilter.includes(e.type));
  const parallelGroups = buildParallelGroups(events);
  const groupByEvent = new Map();
  for (const group of parallelGroups) {
    const visibleGroup = _visibleParallelGroup(group, currentFilter);
    if (!visibleGroup) continue;
    for (const lane of group.lanes) {
      for (const evt of lane.events) groupByEvent.set(evt, visibleGroup);
    }
    for (const evt of group.flatEvents) groupByEvent.set(evt, visibleGroup);
  }

  const renderedGroups = new Set();
  for (const evt of filtered) {
    const group = groupByEvent.get(evt);
    if (!group) {
      listEl.appendChild(createElementFn(evt));
    } else if (!renderedGroups.has(group)) {
      listEl.appendChild(_createParallelGroupElement(group, createElementFn));
      renderedGroups.add(group);
    }
  }
}

/**
 * Create a single event DOM element.
 *
 * @param {object} evt
 * @param {object} opts
 * @param {(evt: object, el: HTMLElement) => void} opts.onReplay
 * @returns {HTMLElement}
 */
export function createEventElement(evt, opts) {
  const { onReplay } = opts;

  const detailText = (evt.content && evt.content.trim()) || (evt.summary && evt.summary.length > 80 ? evt.summary : "");
  const hasContent = !!detailText;

  const wrapper = document.createElement("div");
  wrapper.className = "tl-event-wrapper";
  wrapper.dataset.eventId = evt.id;

  const el = document.createElement("div");
  el.className = "tl-event";
  decorateContextElement(el, evt.ctx);

  // Time
  const timeEl = document.createElement("span");
  timeEl.className = "tl-event-time";
  timeEl.textContent = formatTime(evt.ts);
  timeEl.style.cssText = "flex-shrink:0; color:var(--aw-color-text-muted, #aaa); font-size:0.75rem; min-width:45px;";

  // Icon
  const iconEl = document.createElement("span");
  iconEl.className = "tl-event-icon";
  iconEl.innerHTML = getIcon(evt.type);
  iconEl.style.cssText = "flex-shrink:0;";

  const avatarEl = _createAvatarElement(evt.anima || "");

  // Anima
  const animasEl = document.createElement("span");
  animasEl.className = "tl-event-animas";
  animasEl.textContent = evt.anima || "";
  animasEl.style.cssText = "font-weight:600; color:var(--aw-color-accent, #2563eb); flex-shrink:0; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";

  // Summary
  const summaryEl = document.createElement("span");
  summaryEl.className = "tl-event-summary";
  summaryEl.textContent = getDisplaySummary(evt);
  summaryEl.style.cssText = "color:var(--aw-color-text-secondary, #555); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;";

  if (hasContent) {
    const chevron = document.createElement("span");
    chevron.className = "tl-event-chevron";
    chevron.textContent = "\u25B6"; // ▶
    el.appendChild(chevron);
  }

  el.appendChild(timeEl);
  el.appendChild(iconEl);
  if (avatarEl) el.appendChild(avatarEl);
  el.appendChild(animasEl);
  const contextBadge = createContextBadge(evt.ctx);
  if (contextBadge) el.appendChild(contextBadge);
  el.appendChild(summaryEl);

  wrapper.appendChild(el);
  if (window.lucide) lucide.createIcons({ nodes: [iconEl] });

  if (hasContent) {
    const detail = document.createElement("div");
    detail.className = "tl-event-detail";
    detail.innerHTML = renderSimpleMarkdown(detailText);
    wrapper.appendChild(detail);
  }

  el.addEventListener("click", (e) => {
    e.stopPropagation();
    if (hasContent) {
      const isExpanded = wrapper.classList.toggle("expanded");
      const chevron = el.querySelector(".tl-event-chevron");
      if (chevron) chevron.textContent = isExpanded ? "\u25BC" : "\u25B6";
    }
    onReplay(evt, el);
  });

  return wrapper;
}

/**
 * Show or hide the "Load more" button.
 * @param {boolean} hasMore
 */
export function updateLoadMoreButton(hasMore) {
  const btn = document.getElementById("wsTimelineLoadMore");
  if (btn) {
    btn.style.display = hasMore ? "block" : "none";
  }
}
