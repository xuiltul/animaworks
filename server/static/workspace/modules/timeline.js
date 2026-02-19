/**
 * timeline.js — Activity timeline UI + event replay.
 *
 * Renders a collapsible timeline panel at the bottom of the office view,
 * shows real-time and historical activity events, and replays selected
 * events in the 3D scene (e.g. message particles, desk highlights).
 */

// ── Types ──────────────────────────────────────────

/**
 * @typedef {Object} TimelineEvent
 * @property {string}  id         — unique ID (timestamp-based)
 * @property {string}  type       — event type (WS simplified or API detailed)
 * @property {string}  anima      — related Anima name
 * @property {string}  ts         — ISO 8601
 * @property {string}  summary    — display text
 * @property {Object}  [meta]     — extra data for replay
 */

import { showMessage as showMessagePopup } from "./message-popup.js";
import { getIcon, getDisplaySummary, normalizeEvent } from "../../shared/activity-types.js";
import { renderSimpleMarkdown } from "./utils.js";

// ── Timestamp helper ──────────────────────────────

/**
 * Generate a naive local ISO string (no "Z" suffix) matching the server's
 * ``datetime.now().isoformat()`` format.  Used for real-time events so they
 * sort correctly alongside historical entries from the API.
 * @returns {string} e.g. "2026-02-18T21:41:02.123"
 */
export function localISOString() {
  const d = new Date();
  const pad = (n, len = 2) => String(n).padStart(len, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `.${pad(d.getMilliseconds(), 3)}`
  );
}

/**
 * Compare two ISO timestamp strings by their Date value (newest first).
 * Handles both naive-local ("2026-02-18T21:41:02") and UTC ("...Z") formats.
 */
function _tsDescending(a, b) {
  const ta = new Date(a.ts || 0).getTime();
  const tb = new Date(b.ts || 0).getTime();
  return tb - ta;
}

// ── Module state ───────────────────────────────────

/** @type {TimelineEvent[]} */
const _events = [];

const MAX_EVENTS = 200;

const filterDefs = [
  { label: "All", types: [] },
  { label: "💬", types: ["message_received", "response_sent", "dm_received", "dm_sent", "message", "chat"] },
  { label: "📋", types: ["channel_read", "channel_post", "board"] },
  { label: "💗", types: ["heartbeat_start", "heartbeat_end", "heartbeat"] },
  { label: "⏰", types: ["cron_executed", "cron"] },
  { label: "🔧", types: ["tool_use", "memory_write"] },
  { label: "📣", types: ["human_notify", "notification"] },
  { label: "⚠️", types: ["error", "issue_resolved"] },
];

/** @type {string[]} current filter — empty array means "all" */
let _currentFilter = [];

/** @type {HTMLElement|null} */
let _container = null;

/** @type {HTMLElement|null} */
let _listEl = null;

/** @type {HTMLElement|null} */
let _countEl = null;

/** @type {HTMLElement|null} */
let _bodyEl = null;

/** @type {object|null} interaction manager reference { showMessageEffect, showConversation } */
let _interactionManager = null;

/** @type {boolean} */
let _expanded = false;

/** @type {number} current offset for pagination */
let _currentOffset = 0;

/** @type {boolean} whether there are more events to load */
let _hasMore = false;

/** @type {number} total count from server */
let _totalCount = 0;

/** @type {number} hours parameter used for history queries */
let _currentHours = 48;

// ── Highlight helper (imported lazily from office3d) ──

let _highlightDesk = null;
let _clearHighlight = null;

function _ensureHighlightFns() {
  if (_highlightDesk) return;
  try {
    // These are available as globals from the office3d module loaded by app.js
    // We access them via dynamic import or assume they're on window/globalThis
    // For safety, wrap in try/catch
    import("./office3d.js").then((mod) => {
      _highlightDesk = mod.highlightDesk;
      _clearHighlight = mod.clearHighlight;
    });
  } catch {
    // Will operate without highlight capability
  }
}

// ── DOM construction ───────────────────────────────

function _buildDOM(officePanel) {
  const timeline = document.createElement("div");
  timeline.className = "ws-timeline";
  timeline.id = "wsTimeline";

  // Toggle bar
  const bar = document.createElement("div");
  bar.className = "ws-timeline-bar";
  bar.id = "wsTimelineToggle";

  const title = document.createElement("span");
  title.className = "timeline-title";
  title.textContent = "Activity Timeline";

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
    btn.addEventListener("click", () => _onFilterClick(i, filters));
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
  loadMoreBtn.textContent = "もっと読み込む";
  loadMoreBtn.style.cssText = "display:none; width:100%; padding:0.5rem; margin-top:0.5rem; background:var(--bg-secondary, #f3f4f6); border:1px solid var(--border-color, #e5e7eb); border-radius:6px; cursor:pointer; color:var(--text-secondary, #666); font-size:0.8rem;";
  loadMoreBtn.addEventListener("click", () => _loadMore());

  body.appendChild(filters);
  body.appendChild(list);
  body.appendChild(loadMoreBtn);

  timeline.appendChild(bar);
  timeline.appendChild(body);

  officePanel.appendChild(timeline);

  // Toggle behavior
  bar.addEventListener("click", () => {
    _expanded = !_expanded;
    body.classList.toggle("hidden", !_expanded);
    toggleBtn.textContent = _expanded ? "\u25BC" : "\u25B2"; // ▼ or ▲
  });

  _container = timeline;
  _listEl = list;
  _countEl = count;
  _bodyEl = body;
}

// ── Filter logic ───────────────────────────────────

function _onFilterClick(index, filtersEl) {
  const fd = filterDefs[index];
  if (index === 0 || !fd) {
    _currentFilter = [];
  } else {
    _currentFilter = fd.types || [];
  }

  // Update active class
  for (const btn of filtersEl.querySelectorAll(".tl-filter")) {
    btn.classList.toggle("active", parseInt(btn.dataset.index) === index);
  }

  _renderList();
}

// ── Render event list ──────────────────────────────

function _renderList() {
  if (!_listEl) return;

  _listEl.innerHTML = "";

  const filtered = _currentFilter.length === 0
    ? _events
    : _events.filter((e) => _currentFilter.includes(e.type));

  for (const evt of filtered) {
    const el = _createEventElement(evt);
    _listEl.appendChild(el);
  }
}

/**
 * Create a single event DOM element.
 * @param {TimelineEvent} evt
 * @returns {HTMLElement}
 */
function _createEventElement(evt) {
  // Use content, or fall back to summary for heartbeat etc. where content is empty
  const detailText = (evt.content && evt.content.trim()) || (evt.summary && evt.summary.length > 80 ? evt.summary : "");
  const hasContent = !!detailText;

  const wrapper = document.createElement("div");
  wrapper.className = "tl-event-wrapper";
  wrapper.dataset.eventId = evt.id;

  const el = document.createElement("div");
  el.className = "tl-event";

  // Time
  const timeEl = document.createElement("span");
  timeEl.className = "tl-event-time";
  timeEl.textContent = _formatTime(evt.ts);
  timeEl.style.cssText = "flex-shrink:0; color:#aaa; font-size:0.75rem; min-width:45px;";

  // Icon
  const iconEl = document.createElement("span");
  iconEl.className = "tl-event-icon";
  iconEl.textContent = getIcon(evt.type);
  iconEl.style.cssText = "flex-shrink:0;";

  // Anima
  const animasEl = document.createElement("span");
  animasEl.className = "tl-event-animas";
  animasEl.textContent = evt.anima || "";
  animasEl.style.cssText = "font-weight:600; color:#2563eb; flex-shrink:0; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";

  // Summary
  const summaryEl = document.createElement("span");
  summaryEl.className = "tl-event-summary";
  summaryEl.textContent = getDisplaySummary(evt);
  summaryEl.style.cssText = "color:#555; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;";

  // Expand chevron (only for events with content)
  if (hasContent) {
    const chevron = document.createElement("span");
    chevron.className = "tl-event-chevron";
    chevron.textContent = "\u25B6"; // ▶
    el.appendChild(chevron);
  }

  el.appendChild(timeEl);
  el.appendChild(iconEl);
  el.appendChild(animasEl);
  el.appendChild(summaryEl);

  wrapper.appendChild(el);

  // Expandable content panel
  if (hasContent) {
    const detail = document.createElement("div");
    detail.className = "tl-event-detail";
    detail.innerHTML = renderSimpleMarkdown(detailText);
    wrapper.appendChild(detail);
  }

  // Click → toggle expand + replay
  el.addEventListener("click", (e) => {
    e.stopPropagation();
    if (hasContent) {
      const isExpanded = wrapper.classList.toggle("expanded");
      const chevron = el.querySelector(".tl-event-chevron");
      if (chevron) chevron.textContent = isExpanded ? "\u25BC" : "\u25B6"; // ▼ or ▶
    }
    _replayEvent(evt, el);
  });

  return wrapper;
}

// ── Time formatting ────────────────────────────────

function _formatTime(isoString) {
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
 */
function _resolvePersons(event) {
  const meta = event.meta || {};
  return {
    from: meta.from_person || event.from_person || event.anima || "",
    to:   meta.to_person   || event.to_person   || "",
    text: meta.text         || event.content     || event.summary || "",
  };
}

// ── Replay ─────────────────────────────────────────

function _replayEvent(event, el) {
  const { type, anima, meta } = event;

  // Visual feedback
  el.classList.add("replaying");
  setTimeout(() => el.classList.remove("replaying"), 2000);

  _ensureHighlightFns();

  switch (type) {
    case "message":
    case "dm_received":
    case "dm_sent": {
      const p = _resolvePersons(event);
      if (_interactionManager && p.from && p.to) {
        _interactionManager.showMessageEffect(p.from, p.to, p.text);
      } else if (_highlightDesk && (p.from || p.to)) {
        _highlightDesk(p.from || p.to);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      if (meta && meta.message_id) {
        showMessagePopup(meta.message_id);
      }
      break;
    }

    case "chat":
    case "message_received":
    case "response_sent":
    case "board":
    case "channel_read":
    case "channel_post":
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;

    case "heartbeat":
    case "heartbeat_start":
    case "heartbeat_end":
    case "cron":
    case "cron_executed":
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;

    default:
      if (_highlightDesk && anima) {
        _highlightDesk(anima);
        setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, 3000);
      }
      break;
  }
}

// ── Public API ─────────────────────────────────────

/**
 * Initialize the timeline UI.
 * @param {HTMLElement} officePanel — the office panel container
 * @param {object} interactionManager — { showMessageEffect, showConversation }
 */
export function initTimeline(officePanel, interactionManager) {
  _interactionManager = interactionManager;
  _buildDOM(officePanel);
  _ensureHighlightFns();
}

/**
 * Add a real-time event to the timeline.
 * @param {TimelineEvent} event
 */
export function addTimelineEvent(event) {
  // Insert at the front (newest first)
  _events.unshift(event);

  // Cap
  if (_events.length > MAX_EVENTS) {
    _events.length = MAX_EVENTS;
  }

  // Update count
  if (_countEl) {
    _countEl.textContent = _events.length.toString();
  }

  // If currently displayed, prepend element
  if (_listEl && (_currentFilter.length === 0 || _currentFilter.includes(event.type))) {
    const el = _createEventElement(event);
    _listEl.insertBefore(el, _listEl.firstChild);
  }
}

/**
 * Load historical events from the backend.
 * @param {number} hours — how many hours of history (default 24)
 */
export async function loadHistory(hours = 48) {
  _currentHours = hours;
  try {
    const res = await fetch(`/api/activity/recent?hours=${hours}&limit=200&offset=0`);
    if (!res.ok) return;
    const data = await res.json();
    const events = data.events || [];

    _currentOffset = events.length;
    _hasMore = data.has_more || false;
    _totalCount = data.total || 0;

    for (const raw of events) {
      const evt = normalizeEvent(raw);
      if (!evt.id) {
        evt.id = evt.ts || Date.now().toString();
      }
      _events.push(evt);
    }

    // De-duplicate by id
    const seen = new Set();
    for (let i = _events.length - 1; i >= 0; i--) {
      if (seen.has(_events[i].id)) {
        _events.splice(i, 1);
      } else {
        seen.add(_events[i].id);
      }
    }

    // Sort newest first (Date comparison to handle mixed tz formats)
    _events.sort(_tsDescending);

    // Cap
    if (_events.length > MAX_EVENTS) {
      _events.length = MAX_EVENTS;
    }

    if (_countEl) {
      _countEl.textContent = _totalCount > 0 ? `${_events.length}/${_totalCount}` : _events.length.toString();
    }

    _updateLoadMoreButton();
    _renderList();
  } catch (err) {
    console.warn("[timeline] Failed to load history:", err);
  }
}

async function _loadMore() {
  const btn = document.getElementById("wsTimelineLoadMore");
  if (btn) {
    btn.textContent = "読み込み中...";
    btn.disabled = true;
  }
  try {
    const res = await fetch(`/api/activity/recent?hours=${_currentHours}&limit=200&offset=${_currentOffset}`);
    if (!res.ok) return;
    const data = await res.json();
    const newEvents = data.events || [];

    _hasMore = data.has_more || false;
    _totalCount = data.total || 0;
    _currentOffset += newEvents.length;

    for (const raw of newEvents) {
      const evt = normalizeEvent(raw);
      if (!evt.id) {
        evt.id = evt.ts || Date.now().toString();
      }
      _events.push(evt);
    }

    // De-duplicate
    const seen = new Set();
    for (let i = _events.length - 1; i >= 0; i--) {
      if (seen.has(_events[i].id)) {
        _events.splice(i, 1);
      } else {
        seen.add(_events[i].id);
      }
    }

    _events.sort(_tsDescending);

    if (_countEl) {
      _countEl.textContent = _totalCount > 0 ? `${_events.length}/${_totalCount}` : _events.length.toString();
    }

    _updateLoadMoreButton();
    _renderList();
  } catch (err) {
    console.warn("[timeline] Failed to load more:", err);
  } finally {
    if (btn) {
      btn.textContent = "もっと読み込む";
      btn.disabled = false;
    }
  }
}

function _updateLoadMoreButton() {
  const btn = document.getElementById("wsTimelineLoadMore");
  if (btn) {
    btn.style.display = _hasMore ? "block" : "none";
  }
}

/**
 * Clean up the timeline DOM and state.
 */
export function dispose() {
  if (_container && _container.parentNode) {
    _container.parentNode.removeChild(_container);
  }
  _events.length = 0;
  _currentOffset = 0;
  _hasMore = false;
  _totalCount = 0;
  _container = null;
  _listEl = null;
  _countEl = null;
  _bodyEl = null;
  _interactionManager = null;
}
